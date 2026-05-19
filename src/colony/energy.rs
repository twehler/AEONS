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
        app.init_resource::<DopamineDepletionTimer>();
        app.add_systems(Update, manage_energy);
        app.add_systems(Update, update_hunger_levels);
        app.add_systems(Update, deplete_dopamine);
    }
}


// ── Dopamine depletion ─────────────────────────────────────────────────────
//
// Every virtual second, deplete every organism's `dopamine` by
// `hunger / 3`, clamped at 0. The brain's REINFORCE update reads
// per-tick Δdopamine as reward; depletion creates the *negative*
// branch of that signal so the policy is punished for sitting idle
// when hungry. A sated organism (hunger ≈ 0.23 at full energy)
// depletes ~0.077/s; a starving one (hunger = 1) depletes 0.33/s.

const DOPAMINE_DEPLETION_INTERVAL: f32 = 1.0;

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
// `Organism::hunger` is a normalised aggression signal in `[0, 1]`. It's
// recomputed every frame from the organism's normalised energy by a
// classification-specific curve. The brain pools (currently the
// herbivore_1 supervised brain) read `organism.hunger` directly to
// scale how aggressively the agent pursues its target.
//
// Curves:
//   * Photoautotroph  →  0.0
//       They photosynthesise, never need to chase food.
//   * Herbivore       →  `clamp(5^(-E + 0.1), 0, 1)`
//       At E = 0.1 hunger maxes out at exactly 1.0; at E = 1.0 it
//       sits at 5^(-0.9) ≈ 0.23 (the "relaxed cruise" baseline).
//   * Carnivore       →  same as herbivore for now — reserved
//                        slot for a future carnivore-specific curve.
//                        Plug a new function in here when ready;
//                        every caller already routes through
//                        `compute_hunger` so no downstream code
//                        changes.
//
// The system is a per-frame walk over every `Organism` with the
// markers in question, doing one scalar formula + one float write. On
// a 4 k-organism world this is well under 100 µs; cheaper than the
// existing energy tick (which iterates the same set every 0.5 s but
// also touches cell counts). Per-frame ticking is fine because hunger
// is a derived quantity — the underlying energy can change between
// energy-tick events (predation, movement cost), and a per-frame
// hunger refresh lets the brain react immediately on the next brain
// tick.

/// Compute the hunger signal for one organism given its normalised
/// energy and classification. Pure function — easy to unit-test if
/// we ever want to.
pub fn compute_hunger(
    energy_norm:  f32,
    is_photo:     bool,
    is_carnivore: bool,
) -> f32 {
    if is_photo { return 0.0; }
    if is_carnivore {
        // Reserved: carnivores currently share the herbivore curve.
        // Replace this block with the carnivore-specific formula when
        // the carnivore brain is ready to consume it.
        return herbivore_curve(energy_norm);
    }
    herbivore_curve(energy_norm)
}

#[inline]
fn herbivore_curve(e: f32) -> f32 {
    // H = 5^(-E + 0.1). At E = 0.1 returns exactly 1.0; below E = 0.1
    // returns > 1.0 → clamped. At E = 1.0 returns ≈ 0.232.
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
        // Skip the write if unchanged so Bevy's change-detection
        // doesn't flag this organism dirty every frame for systems
        // that key off `Changed<Organism>`. The integer-bit
        // comparison is exact for the f32 produced by our curve.
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
            let suppress_despawn = is_hetero && ai_training.0;
            if !suppress_despawn {
                commands.entity(entity).despawn();
            }
        }
    }
}
