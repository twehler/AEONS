use bevy::prelude::*;
use crate::colony::*;
use crate::environment::WaterLevel;

use crate::simulation_settings::ENERGY_TICK_INTERVAL;
pub use crate::simulation_settings::MAX_ENERGY_PER_CELL;

pub use crate::simulation_settings::PHOTO_PRODUCTION_PER_CELL;
use crate::simulation_settings::NON_PHOTO_CONSUMPTION_PER_CELL;

use crate::simulation_settings::{K_GROUND_FRICTION, K_FLUID_DRAG};
use crate::simulation_settings::MOVEMENT_COST_MAX_FRACTION_PER_TICK;

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
///
/// Parallel: each organism's hunger depends only on its own state (entity-disjoint
/// writes, no shared mutation, no Commands), so it fans out over `ComputeTaskPool`.
fn update_hunger_levels(
    mut q: Query<(
        &mut crate::colony::Organism,
        Has<crate::colony::Photoautotroph>,
        Has<crate::colony::Carnivore>,
    )>,
) {
    q.par_iter_mut().for_each(|(mut organism, is_photo, is_carn)| {
        let max_e = get_max_energy(&organism).max(1.0);
        let e_norm = (organism.energy / max_e).clamp(0.0, 1.0);
        let new_hunger = compute_hunger(e_norm, is_photo, is_carn);
        // Skip the write if unchanged so `Changed<Organism>` consumers
        // aren't flagged dirty every frame (and so disjoint threads don't all
        // mark their chunks dirty). Bit-compare is exact here.
        if organism.hunger.to_bits() != new_hunger.to_bits() {
            organism.hunger = new_hunger;
        }
    });
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

    // Per-organism upkeep is entity-disjoint, so the compute fans out over
    // `ComputeTaskPool`. Despawning a starved organism needs deferred `Commands`
    // (not usable inside a parallel closure), so starved entities are collected
    // into a `Mutex<Vec>` — contention is near-zero since the push only fires for
    // an organism that actually hit 0 energy this tick — then despawned serially.
    let water_y = water.0;
    let suppress_despawn = ai_training.0;
    let starved: std::sync::Mutex<Vec<Entity>> = std::sync::Mutex::new(Vec::new());

    organisms.par_iter_mut().for_each(|(entity, mut organism, transform, is_hetero)| {
        let max_energy = get_max_energy(&organism);

        // Per-cell upkeep is weighted by cell type (jelly = half, inert = 10%);
        // a creature of only standard tissue weighs exactly its non-photo count.
        let upkeep_weight = organism.upkeep_cell_weight();

        // Submersion ∈ [0, 1]: 0 = entirely above water, 1 = fully submerged.
        // Used only by the water-based (swimmer/floater) fluid-drag branch below;
        // ground-based crawlers pay full friction regardless of water depth.
        let bounding = organism.bounding_radius().max(1.0);
        let depth      = water_y - transform.translation.y;
        let submersion = (depth / bounding).clamp(0.0, 1.0);

        let speed  = organism.movement_speed;
        let weight = organism.weight();

        // Ground friction ∝ weight × speed (linear); fluid drag ∝ weight^(2/3) ×
        // speed³ (cubic — square–cube area, drag ∝ v²). Gated by
        // `MOVEMENT_ENERGY_COSTS_ENABLED` (off for RL training); per-cell upkeep
        // and climb cost are not gated.
        //
        // HETEROTROPHS ONLY (`is_hetero`): this propulsion cost penalises active
        // locomotion (RL realism). Phototrophs are passive drifters — charging a
        // wandering ground-based phototroph the cubic `speed³` drag while submerged
        // dwarfs photosynthesis and starves it in a tick or two, so they never pay.
        //
        // SLIDING ONLY (`is_sliding`): `movement_speed` is a meaningful self-
        // propulsion measure only for sliders (their brain writes it; `apply_movement`
        // / `apply_surface_adhesion` translate them by it). Limb/swimming/flying
        // creatures locomote via joint motors and NOTHING resets `movement_speed`
        // for them — it keeps its spawn value (heteros spawn at 15..25) forever, and
        // charging cubic drag on that PHANTOM speed annihilated a submerged limb/swim
        // hetero in one tick. So the field (and this cost) is only used for sliders.
        //
        // CRAWLER vs SWIMMER split keys on `ground_based`, NOT water depth: a
        // ground-based slider is glued to and slides ALONG the terrain surface
        // (`apply_surface_adhesion`) whether that surface is above or below water —
        // it never swims, so it pays full LINEAR ground friction and no fluid drag.
        // (The old split scaled friction by `1 - submersion` and fluid by
        // `submersion`, so a SUBMERGED benthic crawler wrongly paid full cubic drag
        // and zero friction → starved in ~1 tick.) Water-based movers (swimmers/
        // floaters — none are sliding heteros today, but kept future-safe) pay the
        // cubic fluid drag for moving through the water column instead.
        let (friction_cost, fluid_cost) =
            if crate::simulation_settings::MOVEMENT_ENERGY_COSTS_ENABLED
                && is_hetero
                && organism.movement_mode.is_sliding()
            {
                if organism.ground_based {
                    let friction = K_GROUND_FRICTION * weight * speed * ENERGY_TICK_INTERVAL;
                    (friction, 0.0)
                } else {
                    let fluid = submersion
                        * (K_FLUID_DRAG * weight.powf(2.0 / 3.0) * speed.powi(3))
                        * ENERGY_TICK_INTERVAL;
                    (0.0, fluid)
                }
            } else {
                (0.0, 0.0)
            };

        // Photosynthesis lives in `physiology.rs`; only consumption + clamp +
        // starvation despawn happen here.
        let elevation_cost = organism.climb_energy_debt * ELEVATION_ENERGY_PER_UNIT;
        organism.climb_energy_debt = 0.0;

        // Defense-in-depth: bound the (unbounded, cubic) movement cost so NO mover
        // can lose more than a fixed fraction of its reserve per tick. Catches the
        // submerged-slider latent twin (a benthic sliding hetero on a deep-sea floor
        // still pays cubic fluid drag on a real, floored `movement_speed`) and any
        // future high-speed/heavy case. Inert in the normal regime (see the const).
        let movement_cost =
            (friction_cost + fluid_cost).min(max_energy * MOVEMENT_COST_MAX_FRACTION_PER_TICK);

        let consumption = upkeep_weight * NON_PHOTO_CONSUMPTION_PER_CELL
                          + movement_cost + elevation_cost;

        organism.energy = (organism.energy - consumption).clamp(0.0, max_energy);

        if organism.energy <= 0.0 && !suppress_despawn {
            // AI training mode keeps starved organisms alive (energy stays clamped at
            // 0) so the RL cohort isn't lost — and, critically, so the PREY don't go
            // extinct: loaded phototrophs were starving to 0 and despawning within ~1
            // min (no respawn), leaving herbivores nothing to seek/eat (prey crashed
            // 118→0; K_EAT never fired). Prey persist as a stable food source; real
            // predation still removes them (eaten body parts despawn the prey).
            starved.lock().unwrap().push(entity);
        }
    });

    // Apply the collected despawns serially (Commands are deferred anyway).
    for entity in starved.into_inner().unwrap() {
        commands.entity(entity).despawn();
    }
}
