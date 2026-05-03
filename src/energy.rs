use bevy::prelude::*;
use crate::colony::*;
use crate::environment::WATER_LEVEL;
use crate::krishi::Krishi;

const ENERGY_TICK_INTERVAL: f32 = 0.5;
pub const MAX_ENERGY_PER_CELL: f32 = 10.0;

const PHOTO_PRODUCTION_PER_CELL:      f32 = 0.8;
const NON_PHOTO_CONSUMPTION_PER_CELL: f32 = 0.3;

// Movement-cost coefficients tuned so a max-speed (20) sprint is heavily
// punitive on heavy organisms but doesn't immediately kill them.
const K_GROUND_FRICTION: f32 = 0.08;
const K_FLUID_DRAG:      f32 = 0.005;

/// Energy cost per metre of elevation gained — the gravitational-PE
/// analogue. Charged on every climb step accumulated since the last energy
/// tick and reset afterwards. Krishi is filtered out of the energy system
/// entirely, so its accumulated debt is never drained (never spent).
pub const ELEVATION_ENERGY_PER_UNIT: f32 = 0.5;

/// Maximum energy storage = grown cell count × per-cell capacity.
pub fn get_max_energy(organism: &Organism) -> f32 {
    organism.grown_cell_count() as f32 * MAX_ENERGY_PER_CELL
}

#[derive(Resource)]
pub struct EnergyTickTimer {
    pub timer: Timer,
}

impl Default for EnergyTickTimer {
    fn default() -> Self {
        Self {
            timer: Timer::from_seconds(ENERGY_TICK_INTERVAL, TimerMode::Repeating),
        }
    }
}

pub struct EnergyPlugin;

impl Plugin for EnergyPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EnergyTickTimer>();
        app.add_systems(Update, manage_energy);
    }
}

fn manage_energy(
    mut commands:  Commands,
    time:          Res<Time>,
    mut timer:     ResMut<EnergyTickTimer>,
    // `Without<Krishi>` excludes the fixed-mesh predator class from energy
    // bookkeeping entirely — no consumption, no production, no starvation
    // despawn. Krishi live indefinitely. This keeps the energy system itself
    // unmodified for the procedural organisms; the only cost is one extra
    // archetype filter at query construction time.
    mut organisms: Query<
        (Entity, &mut Organism, &Transform, Has<Photoautotroph>),
        (With<OrganismRoot>, Without<Krishi>),
    >,
) {
    timer.timer.tick(time.delta());
    if !timer.timer.just_finished() { return; }

    for (entity, mut organism, transform, is_photoautotroph) in organisms.iter_mut() {
        let max_energy = get_max_energy(&organism);

        // Tally cell types from grown cells across all alive body parts.
        let (photo_count, non_photo_count) = organism.cell_counts();

        // Submersion ∈ [0, 1]: 0 = entirely above water, 1 = fully submerged.
        let bounding = organism.bounding_radius().max(1.0);
        let depth           = WATER_LEVEL - transform.translation.y;
        let submersion      = (depth / bounding).clamp(0.0, 1.0);
        let ground_fraction = 1.0 - submersion;

        let speed  = organism.movement_speed;
        let weight = organism.weight();

        // Ground friction power ∝ weight × speed.
        let friction_cost = ground_fraction
            * (K_GROUND_FRICTION * weight * speed)
            * ENERGY_TICK_INTERVAL;

        // Fluid drag power ∝ weight^(2/3) × speed³ (square–cube area, drag ∝ v²).
        let fluid_cost = submersion
            * (K_FLUID_DRAG * weight.powf(2.0 / 3.0) * speed.powi(3))
            * ENERGY_TICK_INTERVAL;

        // Only photoautotroph-tagged organisms photosynthesise. A heterotroph
        // that mutated to carry Photo cells does not get to free-ride on the
        // sun — without prey it must starve. The cell-level tally still
        // matters as a proxy for "biological investment in photosynthesis":
        // a photoautotroph that mutates *away* from Photo cells loses
        // production proportionally and can starve too.
        let production = if is_photoautotroph {
            photo_count as f32 * PHOTO_PRODUCTION_PER_CELL
        } else {
            0.0
        };
        let elevation_cost = organism.climb_energy_debt * ELEVATION_ENERGY_PER_UNIT;
        organism.climb_energy_debt = 0.0;

        let consumption = non_photo_count as f32 * NON_PHOTO_CONSUMPTION_PER_CELL
                          + friction_cost + fluid_cost + elevation_cost;

        organism.energy = (organism.energy + production - consumption).clamp(0.0, max_energy);

        if organism.energy <= 0.0 {
            commands.entity(entity).despawn();
        }
    }
}
