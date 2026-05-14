use bevy::prelude::*;
use crate::colony::*;
use crate::environment::WATER_LEVEL;
use crate::krishi::Krishi;

const ENERGY_TICK_INTERVAL: f32 = 0.5;
pub const MAX_ENERGY_PER_CELL: f32 = 10.0;

/// Per-tick energy a fully-surrounded (18 RD neighbours) photo cell
/// produces. Read by `physiology.rs::PhotosyntheticCell::new` to derive
/// the per-cell `energy_production` cache; the photosynthesis tick itself
/// runs in `physiology.rs`, not here.
pub const PHOTO_PRODUCTION_PER_CELL:  f32 = 4.0;
const NON_PHOTO_CONSUMPTION_PER_CELL: f32 = 0.01;

// Movement-cost coefficients tuned so a max-speed (20) sprint is heavily
// punitive on heavy organisms but doesn't immediately kill them.
const K_GROUND_FRICTION: f32 = 0.003;
const K_FLUID_DRAG:      f32 = 0.03;

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
        (Entity, &mut Organism, &Transform, Has<Heterotroph>),
        (With<OrganismRoot>, Without<Krishi>),
    >,
) {
    timer.timer.tick(time.delta());
    if !timer.timer.just_finished() { return; }

    for (entity, mut organism, transform, is_hetero) in organisms.iter_mut() {
        let max_energy = get_max_energy(&organism);

        // Cached cell counts — kept in sync by physiology / predation, no
        // per-tick iteration needed here.
        let non_photo_count = organism.non_photo_cell_count.max(0) as f32;

        // Submersion ∈ [0, 1]: 0 = entirely above water, 1 = fully submerged.
        let bounding = organism.bounding_radius().max(1.0);
        let depth           = WATER_LEVEL - transform.translation.y;
        let submersion      = (depth / bounding).clamp(0.0, 1.0);
        let ground_fraction = 1.0 - submersion;

        let speed  = organism.movement_speed;
        let weight = organism.weight();

        // Ground friction power ∝ weight × speed, and fluid drag
        // power ∝ weight^(2/3) × speed³ (square–cube area, drag ∝ v²).
        // Both terms are gated by `MOVEMENT_ENERGY_COSTS_ENABLED`
        // so the RL training environment can switch them off and
        // test whether movement-cost punishment is what's keeping
        // the policy from learning to pursue prey. Per-cell upkeep
        // and climb-cost are NOT gated — they're not movement
        // expressions.
        let (friction_cost, fluid_cost) =
            if crate::simulation_settings::MOVEMENT_ENERGY_COSTS_ENABLED {
                let friction = ground_fraction
                    * (K_GROUND_FRICTION * weight * speed)
                    * ENERGY_TICK_INTERVAL;
                let fluid = submersion
                    * (K_FLUID_DRAG * weight.powf(2.0 / 3.0) * speed.powi(3))
                    * ENERGY_TICK_INTERVAL;
                (friction, fluid)
            } else {
                (0.0, 0.0)
            };

        // Photosynthesis is owned by `physiology.rs` now (per-cell,
        // neighbour-count-weighted). Only consumption + clamp + starvation
        // despawn happen here.
        let elevation_cost = organism.climb_energy_debt * ELEVATION_ENERGY_PER_UNIT;
        organism.climb_energy_debt = 0.0;

        let consumption = non_photo_count * NON_PHOTO_CONSUMPTION_PER_CELL
                          + friction_cost + fluid_cost + elevation_cost;

        organism.energy = (organism.energy - consumption).clamp(0.0, max_energy);

        if organism.energy <= 0.0 {
            // AI training mode: keep starved heterotrophs alive (energy
            // stays clamped at 0) so the RL training cohort isn't lost.
            // Hunger still accrues; only the despawn step is suppressed.
            let suppress_despawn = is_hetero
                && crate::simulation_settings::AI_TRAINING_MODE;
            if !suppress_despawn {
                commands.entity(entity).despawn();
            }
        }
    }
}
