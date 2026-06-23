use bevy::prelude::*;
use crate::colony::*;
use crate::environment::WaterLevel;

use crate::simulation_settings::ENERGY_TICK_INTERVAL;
pub use crate::simulation_settings::MAX_ENERGY_PER_CELL;

pub use crate::simulation_settings::PHOTO_PRODUCTION_PER_CELL;
use crate::simulation_settings::NON_PHOTO_CONSUMPTION_PER_CELL;

use crate::simulation_settings::{K_GROUND_FRICTION, K_FLUID_DRAG};

pub use crate::simulation_settings::ELEVATION_ENERGY_PER_UNIT;

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
        app.init_resource::<DopamineDepletionTimer>();
        app.add_systems(Update, manage_energy);
        app.add_systems(Update, update_hunger_levels);
        app.add_systems(Update, deplete_dopamine);
    }
}


// ── Dopamine depletion ─────────────────────────────────────────────────────
//
// Deplete `dopamine` by `hunger/3` each virtual second (clamped at 0). The
// REINFORCE update reads per-tick Δdopamine, so depletion is the negative
// branch that punishes idling while hungry.

use crate::simulation_settings::DOPAMINE_DEPLETION_INTERVAL;

#[derive(Resource)]
pub struct DopamineDepletionTimer { pub timer: Timer }

impl Default for DopamineDepletionTimer {
    fn default() -> Self {
        Self { timer: Timer::from_seconds(DOPAMINE_DEPLETION_INTERVAL, TimerMode::Repeating) }
    }
}

fn deplete_dopamine(
    time:      Res<Time>,
    mut timer: ResMut<DopamineDepletionTimer>,
    mut q:     Query<&mut crate::colony::Organism>,
) {
    timer.timer.tick(time.delta());
    if !timer.timer.just_finished() { return; }

    for mut org in &mut q {
        let delta = org.hunger / 3.0;
        org.dopamine = (org.dopamine - delta).max(0.0);
    }
}


// ── Hunger ──────────────────────────────────────────────────────────────────
//
// `Organism::hunger` is a normalised pursuit-aggression signal in `[0, 1]`,
// recomputed each frame from normalised energy by a classification-specific
// curve (see `compute_hunger`). The herbivore_1 brain reads it directly.
// Refreshed per-frame (not per energy tick) so the brain reacts immediately
// to energy changes between energy ticks.

/// Hunger signal for one organism from normalised energy + classification.
pub fn compute_hunger(
    energy_norm:  f32,
    is_photo:     bool,
    is_carnivore: bool,
) -> f32 {
    if is_photo { return 0.0; }
    if is_carnivore {
        // Reserved: carnivores share the herbivore curve until a
        // carnivore-specific formula exists.
        return herbivore_curve(energy_norm);
    }
    herbivore_curve(energy_norm)
}

#[inline]
fn herbivore_curve(e: f32) -> f32 {
    // H = 5^(-E + 0.1): 1.0 at E=0.1 (>1.0 below, clamped), ≈0.232 at E=1.0.
    let h = 5.0_f32.powf(-e + 0.1);
    h.clamp(0.0, 1.0)
}

/// Walk every organism, refresh its `hunger` field. Runs every frame.
fn update_hunger_levels(
    mut q: Query<(
        &mut crate::colony::Organism,
        Has<crate::colony::Photoautotroph>,
        Has<crate::colony::Carnivore>,
    )>,
) {
    for (mut organism, is_photo, is_carn) in &mut q {
        let max_e = get_max_energy(&organism).max(1.0);
        let e_norm = (organism.energy / max_e).clamp(0.0, 1.0);
        let new_hunger = compute_hunger(e_norm, is_photo, is_carn);
        // Skip the write if unchanged so `Changed<Organism>` consumers
        // aren't flagged dirty every frame. Bit-compare is exact here.
        if organism.hunger.to_bits() != new_hunger.to_bits() {
            organism.hunger = new_hunger;
        }
    }
}

fn manage_energy(
    mut commands:    Commands,
    time:            Res<Time>,
    mut timer:       ResMut<EnergyTickTimer>,
    ai_training:     Res<crate::simulation_settings::AiTrainingMode>,
    water:           Res<WaterLevel>,
    mut organisms: Query<
        (Entity, &mut Organism, &Transform, Has<Heterotroph>),
        With<OrganismRoot>,
    >,
) {
    timer.timer.tick(time.delta());
    if !timer.timer.just_finished() { return; }

    for (entity, mut organism, transform, is_hetero) in organisms.iter_mut() {
        let max_energy = get_max_energy(&organism);

        // Per-cell upkeep is weighted by cell type (jelly = half, inert = 10%);
        // a creature of only standard tissue weighs exactly its non-photo count.
        let upkeep_weight = organism.upkeep_cell_weight();

        // Submersion ∈ [0, 1]: 0 = entirely above water, 1 = fully submerged.
        let bounding = organism.bounding_radius().max(1.0);
        let depth           = water.0 - transform.translation.y;
        let submersion      = (depth / bounding).clamp(0.0, 1.0);
        let ground_fraction = 1.0 - submersion;

        let speed  = organism.movement_speed;
        let weight = organism.weight();

        // Ground friction ∝ weight × speed; fluid drag ∝ weight^(2/3) × speed³
        // (square–cube area, drag ∝ v²). Both gated by
        // `MOVEMENT_ENERGY_COSTS_ENABLED` (off for RL training); per-cell
        // upkeep and climb cost are not gated.
        //
        // HETEROTROPHS ONLY: this propulsion cost penalises active locomotion
        // (RL realism). Phototrophs are passive drifters — charging a wandering
        // ground-based phototroph the cubic `speed³` drag while it's submerged
        // produces a cost that dwarfs photosynthesis and starves it in a tick or
        // two → mass despawn + reproduction-refill churn + lag when many are
        // spawned. So a phototroph never pays a movement cost (its upkeep is its
        // only outgo, and photo cells have ~zero upkeep). (`PHOTO_WANDER_MAX_SPEED`
        // also keeps the wander slow for realism.)
        let (friction_cost, fluid_cost) =
            if crate::simulation_settings::MOVEMENT_ENERGY_COSTS_ENABLED && is_hetero {
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

        // Photosynthesis lives in `physiology.rs`; only consumption + clamp +
        // starvation despawn happen here.
        let elevation_cost = organism.climb_energy_debt * ELEVATION_ENERGY_PER_UNIT;
        organism.climb_energy_debt = 0.0;

        let consumption = upkeep_weight * NON_PHOTO_CONSUMPTION_PER_CELL
                          + friction_cost + fluid_cost + elevation_cost;

        organism.energy = (organism.energy - consumption).clamp(0.0, max_energy);

        if organism.energy <= 0.0 {
            // AI training mode keeps starved organisms alive (energy stays clamped at
            // 0) so the RL cohort isn't lost — and, critically, so the PREY don't go
            // extinct: loaded phototrophs were starving to 0 and despawning within ~1
            // min (no respawn), leaving herbivores nothing to seek/eat (prey crashed
            // 118→0; K_EAT never fired). Prey persist as a stable food source; real
            // predation still removes them (eaten body parts despawn the prey).
            let suppress_despawn = ai_training.0;
            if !suppress_despawn {
                commands.entity(entity).despawn();
            }
        }
    }
}
