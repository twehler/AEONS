// Simulation settings — runtime control state.
//
// Holds the resources describing whether the simulation is running and whether
// the player has captured the viewport. Other modules read these flags to decide
// whether to advance time, accept input, or route mouse motion to the camera.
// This file is the single source of truth for "live" simulation controls.

use bevy::prelude::*;
use std::time::Duration;

/// Default seeded into `MaxPhotoautotrophs` when nothing else (launcher / CLI)
/// sets it: the running-population cap on photoautotrophs only (heterotrophs use
/// the independent `MaxHerbivores`; the GPU brain pool sizes off that instead).
/// 150 keeps prey plentiful for herbivores while holding ~60 FPS — higher counts
/// bloat meshes + colliders + photosynthesis until frame rate collapses.
pub const DEFAULT_MAX_PHOTOAUTOTROPHS: usize = 150;

/// Launcher-side default for the herbivore reproduction cap; reproduction stops
/// scheduling herbivore births once reached. Each limb herbivore is many dynamic
/// bodies + joints (CPU physics), so 60 holds frame rate while leaving a healthy
/// population to study; the launcher field lifts it for AI-training runs.
pub const DEFAULT_MAX_HERBIVORES: usize = 60;

/// Launcher-side default for the initial herbivore cohort at `spawn_colony` (when
/// no colony save is loaded). Independent of `DEFAULT_MAX_HERBIVORES` so the user
/// can seed a small starter population and let reproduction grow it to the cap.
pub const DEFAULT_START_HETEROTROPHS: usize = 100;

/// Launcher-side default for the initial photoautotroph cohort at `spawn_colony`
/// (when no colony save is loaded). Independent of `DEFAULT_MAX_PHOTOAUTOTROPHS`
/// so the user can seed a small starter population and let reproduction grow it.
pub const DEFAULT_START_PHOTOAUTOTROPHS: usize = 800;


/// Heterotrophs to spawn at `spawn_colony` startup (launcher field /
/// `--start-heteros N`). Drives only the initial cohort, not the running cap
/// (`MaxHerbivores`).
#[derive(Resource)]
pub struct StartHeterotrophs(pub usize);

impl Default for StartHeterotrophs {
    fn default() -> Self { Self(DEFAULT_START_HETEROTROPHS) }
}

/// Photoautotrophs to spawn at `spawn_colony` startup (launcher field /
/// `--start-photos N`). Drives only the initial cohort, not the running cap
/// (`MaxPhotoautotrophs`).
#[derive(Resource)]
pub struct StartPhotoautotrophs(pub usize);

impl Default for StartPhotoautotrophs {
    fn default() -> Self { Self(DEFAULT_START_PHOTOAUTOTROPHS) }
}

pub const DEFAULT_MAP_X:           f32        = 1000.0;
pub const DEFAULT_MAP_Z:           f32        = 1000.0;


/// AI-training mode toggle (statistics-panel checkbox). When `true`, heterotrophs
/// never despawn (energy still drains/clamps at 0; only the despawn step is
/// suppressed); when `false`, starved heterotrophs despawn. Read only by
/// `energy.rs::manage_energy`; reproduction is not gated by this flag.
#[derive(Resource, Default)]
pub struct AiTrainingMode(pub bool);

/// When true, a loaded `.colony` does NOT override the launcher-chosen
/// environment dimensions (currently: the water level) — the launcher value
/// is kept and the colony is "adjusted" into it. Default false: a loaded
/// colony restores its own saved water level. Inserted from argv in phase 5;
/// read by `spawn_colony` at load time.
#[derive(Resource, Default)]
pub struct AdjustColonyDimensions(pub bool);

/// When true, saved limb-brain weights are honoured even under the STANDING task
/// (`--reload-limb-brains`). Default false: standing fresh-inits because legacy
/// colonies hold locomotion-trained, tanh-saturated brains that freeze a stand.
/// Set true to RELOAD a colony saved after a successful standing run — the
/// durable-success artifact, so a proven standing policy can be re-demonstrated
/// without retraining (sidesteps the unseedable GPU RNG). Read by the
/// `assign_brains_*_limb` lifecycle systems.
#[derive(Resource, Default)]
pub struct ReloadLimbBrains(pub bool);


/// Upper bound on the live herbivore (heterotroph − carnivore) count;
/// `reproduction_system` skips herbivore births while at/above it (`0` disables
/// herbivore reproduction). Runtime-editable via the statistics-panel field.
#[derive(Resource)]
pub struct MaxHerbivores(pub usize);

impl Default for MaxHerbivores {
    fn default() -> Self { Self(DEFAULT_MAX_HERBIVORES) }
}


/// Top-level window-mode toggle (top bar of the frontend layout). `Simulation` is
/// the default: stats panel + navigator + click-to-capture player camera. The
/// editor/view modes all auto-pause the simulation on entry (and do NOT auto-resume
/// when leaving); `EditColony` keeps the world rendering for placement.
#[derive(Resource, PartialEq, Eq, Clone, Copy, Debug)]
pub enum WindowMode {
    Simulation,
    /// Colony editor: editor panels (creation / tool / inventory) + a
    /// hold-LMB-rotate flycam, world still rendered so organisms can be placed.
    EditColony,
    /// Tree-of-life view — renders the `SpeciesRegistry` ancestry tree in place
    /// of the viewport + side panels.
    Lineages,
    /// Species editor — hides the world and presents a top+bottom panel UI for
    /// building an organism cell-by-cell, output to a `.species` binary.
    SpeciesEditor,
}

impl Default for WindowMode {
    fn default() -> Self { Self::Simulation }
}


/// True when the simulation is advancing (`Time<Virtual>` unpaused) and virtual-time
/// systems are doing work. Toggled by the Start/Stop button. Defaults to `true`
/// (auto-start) so observers see life immediately; player controls still default off.
#[derive(Resource)]
pub struct SimulationRunning(pub bool);

impl Default for SimulationRunning {
    fn default() -> Self { Self(true) }
}


/// True when the player has captured the viewport and WASD / mouse-look consume
/// input. Activated by left-click in the 3D viewport (only while running),
/// deactivated by Esc. Independent of `SimulationRunning`. Defaults to `false`.
#[derive(Resource, Default)]
pub struct PlayerControlsActive(pub bool);


/// Cinematic mode: when `true`, all UI chrome (top mode bar, statistics
/// panel, both navigators, the dividers) is hidden and the 3D viewport
/// fills the whole window. Toggled with F1 while in
/// `WindowMode::Simulation`. Auto-cleared if the window mode ever
/// leaves Simulation. Default `false`.
#[derive(Resource, Default)]
pub struct CinematicMode(pub bool);


/// Compile-time switch for speed-dependent energy costs (ground friction + fluid
/// drag) charged each energy tick: `true` = movement costs energy (linear in speed
/// on ground, cubic in fluid per `energy.rs::manage_energy`); `false` = both zeroed
/// (free movement). Per-cell upkeep and climb-elevation cost are unaffected.
/// Provided as an off-switch to isolate movement cost as a reward-shaping
/// confound in RL training (it can punish movement right after a predation spike).
pub const MOVEMENT_ENERGY_COSTS_ENABLED: bool = true;


/// LIMB (foot) friction coefficient, combined with terrain via
/// `CoefficientCombine::Max` so a planted foot GRIPS (μ → 1.0) and gives the leg
/// stroke an anchor to drive the body against. Lifting a foot zeroes its normal
/// force (and thus friction), so the swing phase is frictionless regardless of μ.
/// Paired with the slippery `BASE_FRICTION_COEFFICIENT` belly, this asymmetry is
/// the propulsion mechanism for emergent crawling. Consumed by
/// `colony::spawn_organism` / `spawn_loaded_organism`.
pub const LIMB_FRICTION_COEFFICIENT: f32 = 1.0;

/// BASE-body (belly) friction: LOW, combined via `CoefficientCombine::Min`
/// (μ → 0.0) so the grippy feet's push slides the body forward instead of pinning
/// it. `enforce_limb_floor` keeps every part at/above the terrain so this stays
/// non-penetrating.
pub const BASE_FRICTION_COEFFICIENT: f32 = 0.0;



/// Global simulation-time multiplier via `Time<Virtual>::set_relative_speed`, so
/// every `Res<Time>` reader (energy/brain ticks, photosynthesis, predation,
/// movement, reproduction, panel timer) inherits the scaled delta. The player
/// camera reads `Time<Real>` directly and is unaffected. 1.0 = baseline, 0.0
/// freezes virtual time; past ~5–10× the GPU brain pools get stressed (more ticks
/// per real second).
#[derive(Resource)]
pub struct TimeSpeed(pub f32);

impl Default for TimeSpeed {
    fn default() -> Self { Self(1.0) }
}


/// When `true`, adult organisms get their meshes smoothed (Jacobi smoother in
/// `volumetric_growth::smooth_vertices`), at most once per organism (at spawn for
/// non-variable-form; on the growth tick crossing `MAX_CELLS` for variable-form).
/// When `false`, the faceted rhombic-dodecahedron mesh is kept. Toggling at runtime
/// is non-retroactive — only future spawn / adult-transition events read it.
/// Defaults to `true`.
#[derive(Resource)]
pub struct Smoothing(pub bool);

impl Default for Smoothing {
    fn default() -> Self { Self(true) }
}





/// Runtime-adjustable upper bound on the live photoautotroph count. Reproduction
/// suppresses photo births at/above it; lowering it below the live count makes
/// `apply_max_phototrophs_cull` (statistics_panel.rs) despawn a random subset to
/// meet the cap in one step. Heterotrophs use `MaxHerbivores` instead.
#[derive(Resource)]
pub struct MaxPhotoautotrophs(pub usize);

impl Default for MaxPhotoautotrophs {
    fn default() -> Self { Self(DEFAULT_MAX_PHOTOAUTOTROPHS) }
}


/// Real-time interval between autosaves, in minutes. Uses `Time<Real>` so the
/// cadence is wall-clock regardless of `TimeSpeed`. Lower → more frequent backups
/// but more disk churn (each save is the whole colony state); higher → more
/// progress at risk between saves.
pub const AUTOSAVE_INTERVAL_MINUTES: f32 = 10.0;


/// Default minimum heterotroph count enforced by `AutoSpawnHeteros`. Low so a
/// fresh enable doesn't dump a huge cohort before the user dials the value in.
pub const DEFAULT_MIN_HETERO_COUNT: usize = 50;


/// When `true`, the auto-spawn system tops the heterotroph population up to
/// `MinHeteroCount` on each heterotroph death event. Off by default (navigator
/// checkbox). Only does work on death events and when enabled, so off costs nothing.
#[derive(Resource, Default)]
pub struct AutoSpawnHeteros(pub bool);


/// Target lower bound on the live heterotroph count. Read by the
/// auto-spawn system only while `AutoSpawnHeteros(true)`.
#[derive(Resource)]
pub struct MinHeteroCount(pub usize);

impl Default for MinHeteroCount {
    fn default() -> Self { Self(DEFAULT_MIN_HETERO_COUNT) }
}


/// Edit-state for the navigator's "Min heterotroph count" field: `focused` while
/// typing, `buffer` holds in-progress digits (mirrors `MaxOrganismsEditState`).
#[derive(Resource, Default)]
pub struct MinHeteroCountEditState {
    pub buffer:  String,
    pub focused: bool,
}


/// Brain-pool batch dimension `N`, chosen at startup and fixed for the process
/// lifetime: the `BrainPool*` resources size their GPU tensors against it in
/// `FromWorld`, and the CubeCL kernel cache + burn-cuda allocations pin this shape
/// (cannot change at runtime). Heterotrophs born past this limit exist as entities
/// but get no brain slot. Set by `main.rs::run_simulation` from `MaxHerbivores`
/// plus headroom; independent of `MaxPhotoautotrophs` (photos have no brain slots).
#[derive(Resource)]
pub struct OrganismPoolSize(pub usize);

impl Default for OrganismPoolSize {
    fn default() -> Self { Self(DEFAULT_MAX_PHOTOAUTOTROPHS) }
}


// ── Level 1 (heterotroph RL pool) tuning RANGES ─────────────────────────────
//
// Per-organism hyperparameters: every heterotroph is born with its
// own sample drawn uniformly from each `(min, max)` range below.
// Offspring inherit the parent's values plus small Gaussian noise
// (`L1_GENE_MUTATION_REL_STDDEV` × range width), clamped to the
// range. Natural selection acts on the resulting trait diversity —
// organisms whose hyperparameters produce more successful policies
// reproduce more, so the population gradually concentrates around
// the values that work.
//
// Architecture-side constants (`IN`, `HIDDEN`, `OUT`, `ROLLOUT_LEN`,
// `GAMMA`, `BASELINE_ALPHA`, `LR`, `MAX_SPEED`) stay inside the
// brain module — they describe the algorithm rather than the
// trained-behaviour preferences and are NOT per-organism.

/// Range for σ — Gaussian exploration noise on the policy mean.
/// Wider σ ⇒ more diverse action samples; narrower σ ⇒ tighter
/// commitment to the current policy mean.
pub const L1_SIGMA_RANGE:         (f32, f32) = (0.2, 0.8);

/// Range for `K_EAT` — one-shot reward on a predation event (via the
/// `Organism::predations` delta). Floor kept high (4.0) so the eat jackpot
/// dominates the accumulated dense per-tick rewards, pinning the policy gradient on
/// "catch prey" rather than the "fast wanderer" attractor.
pub const L1_K_EAT_RANGE:         (f32, f32) = (4.0, 12.0);

/// Range for `K_REPRO` — one-shot reward on reproduction
/// (detected via `Organism::reproductions` increment).
pub const L1_K_REPRO_RANGE:       (f32, f32) = (5.0, 30.0);

/// Range for `LAMBDA_ENERGY` — coefficient on the negative part
/// of `ΔE` per brain tick (the "energy-loss punishment").
pub const L1_LAMBDA_ENERGY_RANGE: (f32, f32) = (0.3, 2.0);

/// Range for `K_CURIOSITY` — per-tick reward proportional to
/// `applied_speed_norm − 0.5` (stillness penalised, full speed rewarded equally).
/// Wide enough (0.4, 1.5) that the min-vs-max-speed gradient survives the per-slot
/// EMA baseline absorption that would otherwise track it to zero and let the policy
/// settle into the standstill local optimum.
pub const L1_K_CURIOSITY_RANGE:   (f32, f32) = (0.4, 1.5);

/// Range for `K_PROGRESS` — per-tick reward proportional to the
/// distance CLOSED to the currently-locked target. Only positive
/// closing distance is rewarded; receding doesn't fire (so the
/// reward never punishes a target switch).
pub const L1_K_PROGRESS_RANGE:    (f32, f32) = (0.5, 3.0);

/// Target-lock window, in virtual seconds. While locked, the policy's target-choice
/// logits are ignored and direction stays geometric toward the locked entity —
/// without it, σ-noise flickers the choice each tick and the hetero stutters between
/// prey it never reaches. In virtual seconds so it scales with `TimeSpeed`.
pub const L1_TARGET_LOCK_SECS: f32 = 10.0;

/// After the lock window expires, the network may switch targets,
/// but only if the new winner's logit beats the current target's
/// logit by at least this margin. Prevents oscillation when two
/// prey have near-equal scores. Tanh outputs are in `[-1, +1]`, so
/// a 0.15 margin is ~7.5% of the full output range.
pub const L1_TARGET_SWITCH_MARGIN: f32 = 0.15;

/// EMA factor for output-side speed momentum:
/// `applied_speed = α · prev_applied + (1 − α) · new_sample`. At 0.3 the executed
/// speed moves 70% toward the fresh sample each tick (commanded speed reached in
/// ~2 ticks). Kept low because `L1_APPROACH_RADIUS` already prevents overshoot, so
/// extra momentum only makes motion sluggish.
pub const L1_SPEED_MOMENTUM_ALPHA: f32 = 0.3;

/// XZ distance (world units) at which L1 arrival-braking kicks in: inside it the
/// applied speed is scaled by `clamp(distance / L1_APPROACH_RADIUS, 0, 1)` (zero on
/// the target). Must be ≥ `MAX_SPEED · TICK_SECS = 3` to avoid per-tick overshoot;
/// smaller radii cause ping-pong oscillation that reads as "stuck". Contact range
/// (whether the body touches prey) is a separate dial — `CELL_COLLISION_RADIUS`.
pub const L1_APPROACH_RADIUS: f32 = 3.0;

/// Consecutive no-progress brain ticks before the target lock is force-dropped and
/// re-picked — escapes the "stuck behind a blocker" pattern. At ~6.67 ticks/virtual
/// sec, 6 ≈ 0.9 s: long enough to ride out contact-bounce jitter, short enough not to
/// waste the full lock window on an unreachable target.
pub const L1_STUCK_TICKS: u16 = 6;

/// Minimum distance-closed (world units) per brain tick that counts as "progress";
/// anything less is noise and feeds the `L1_STUCK_TICKS` counter. 0.3 is just below
/// the per-tick travel at 1/4 throttle — below it the agent isn't really converging.
pub const L1_STUCK_PROGRESS_EPS: f32 = 0.3;

/// XZ distance (world units) below which the L1 brain freezes its commanded
/// direction instead of recomputing `(target − self).normalize()`. At close range
/// the unit-vector is hypersensitive to prey wobble (a tiny lateral nudge swings the
/// heading several degrees), producing micro-oscillation. Kept small (0.1) so it only
/// fires once cells essentially overlap — larger values let the agent park just shy
/// of contact and never eat.
pub const L1_DIRECTION_FREEZE_DIST: f32 = 0.1;

/// Speed scale applied when the hetero has no locked target (no Photo in the
/// K-nearest window). Without it the policy cruises at full `MAX_SPEED` with nothing
/// to chase, turning prey-empty hetero clusters into a collision-bounce mess. 0.3
/// lets it patrol for prey without enough force to deadlock against neighbours.
pub const L1_NO_TARGET_SPEED_SCALE: f32 = 0.3;

/// Floor on the EMA-smoothed `applied_speed_a` before it's multiplied by
/// `MAX_SPEED · brake_scale`, so every hetero always moves at least this fraction of
/// brake-adjusted max speed — stillness is structurally impossible whatever the
/// policy outputs. Backstops the curiosity gradient: the per-slot EMA baseline can
/// absorb constant reward offsets and strand the policy in a learned standstill, and
/// this hard clamp prevents that. Brake-zone behaviour is unaffected (`brake_scale →
/// 0` at the target). 0.5 = half-throttle default cruise.
pub const L1_MIN_APPLIED_SPEED: f32 = 0.5;

/// Max per-tick random rotation (radians, ≈ 8.6° at 0.15) added to
/// `movement_direction` when the hetero has no locked target — a slow Brownian
/// wander that breaks targetless heteros out of stable mutual deadlock. Sampled
/// uniformly from `[-this, +this]` each brain tick.
pub const L1_NO_TARGET_WANDER_ANGLE: f32 = 0.15;

/// Brain ticks a force-dropped target stays blacklisted (excluded from argmax) after
/// stuck-detection fires, breaking the "drop → re-pick same unreachable target → drop"
/// loop. At ~6.67 ticks/virtual sec, 20 ≈ 3 s: long enough to commit elsewhere and
/// move off the blocker, short enough that a newly-reachable target isn't ignored long.
pub const L1_TARGET_BLACKLIST_TICKS: u16 = 20;

/// Mutation strength for offspring inheritance, expressed as a
/// fraction of the range width. Each gene gets `N(0, σ²)` noise added
/// with `σ = L1_GENE_MUTATION_REL_STDDEV × (max − min)`, then clamped
/// back into the range. Small enough that offspring stay near the
/// parent (selection acts on a coherent gradient), large enough that
/// the population explores the range over generations.
pub const L1_GENE_MUTATION_REL_STDDEV: f32 = 0.05;


// ── Lineages / speciation ───────────────────────────────────────────────────

/// Normalised DNA distance (mean-abs-diff, components in `[0,1]`, per
/// `lineages::dna::distance`) above which an organism is split into a NEW species
/// from a candidate centroid. Also gates the trimmed-mean centroid in
/// `speciation::update_species_averages` — members beyond this from the mean are
/// excluded, so one drifter can't drag the centroid.
pub const SPECIES_SEPARATION_THRESHOLD: f32 = 0.10;


// ════════════════════════════════════════════════════════════════════════════
// CENTRALISED TUNING CONSTANTS
//
// The constants below were relocated here from their originating modules so
// every "knob" lives in one file. Each original module re-exports its
// constant via `pub use` / `use crate::simulation_settings::X;` so all
// existing reference sites — including full-path `crate::module::X` accesses
// from other modules — keep resolving unchanged.
// ════════════════════════════════════════════════════════════════════════════


// ── Physics / Limb (rapier_setup.rs, colony.rs) ──────────────────────────────

// (rapier_setup.rs)
/// Avian XPBD substep count for the limb physics world. Higher substeps keep the
/// rigid joints convergent under high torque but multiply the whole solver cost
/// linearly (dominant once dozens of limb organisms are alive). 8 balances cost vs
/// stability now that joints hold via compliance + mass + damping rather than brute
/// substepping; nudge to 10–12 if joints visibly drift under fast commands.
pub const LIMB_SOLVER_SUBSTEPS: u32 = 8;

/// Rapier solver iteration counts (`RapierContextSimulation::integration_parameters`),
/// set once at startup by `rapier_setup::configure_solver`. Rapier's default is 4 —
/// far too low for CHAINED impulse-jointed legs on near-massless bodies, which
/// separate when the solver doesn't converge (measured: joint_sep median 0.28, max
/// 14 at the default). Rapier's own guidance: "large assemblies easily break without
/// a large number of solver iterations" (8–12 for stacks/machinery). 16 with extra
/// internal stabilization holds the leg chains together (verified via `joint_sep_max`
/// telemetry — guiderail 2). Raise further if separation persists; it costs solver time.
pub const LIMB_SOLVER_ITERATIONS: usize = 8;
pub const LIMB_STABILIZATION_ITERATIONS: usize = 2;

/// Compliance (inverse stiffness) on every limb joint. `0.0` = a perfectly rigid
/// point-constraint: the solver fully projects the anchors together each substep so
/// the limb can't drift apart under high commanded ω; a non-zero give would let it
/// separate when the reaction exceeds the solver's holding capacity at
/// `LIMB_SOLVER_SUBSTEPS`.
pub const LIMB_JOINT_COMPLIANCE: f32 = 0.0;

/// `max_torque` clamp (N·m) on each limb hinge's spring-damper `AngularMotor` — the
/// motor's "muscle strength" (per-substep impulse capped at `max_torque · dt²`).
/// High enough that a planted limb can press the (light) body upward
/// (`max_torque ≳ body weight × lever`); raise if limbs feel weak, lower if strides
/// look violent. Scales with limb inertia, so it tracks `BODY_PART_DENSITY`. Launch
/// is contained by the velocity governor / `SelfCollisionFilter` /
/// `LIMB_ANGULAR_DAMPING`, not by starving this.
pub const MAX_LIMB_TORQUE: f32 = 10.0;

/// Hinge swing-angle limit on every limb `RevoluteJoint` (`±LIMB_SWING_LIMIT`). The
/// limb is a 1-DOF hinge (the revolute joint rigidly locks the two non-hinge axes, so
/// it can't orbit the body). ±80° caps the in-plane swing so the limb can't fold
/// absurdly through the body while leaving a generous stride.
pub const LIMB_SWING_LIMIT: f32 = 80.0 * std::f32::consts::PI / 180.0;

/// Twist knob: compliance (inverse stiffness) of the `RevoluteJoint`
/// axis-ALIGNMENT constraint. `0.0` = perfectly rigid hinge, zero twist; larger
/// softens the two off-hinge DOF so the limb can rotate OFF its single swing
/// plane — the substrate the brain-driven twist (`ConstantLocalTorque` ∝
/// `MAX_LIMB_TWIST_TORQUE`, see `rapier_setup::drive_limb_twist`) acts on, so limb
/// movement is no longer confined to one axis. Raised from the old near-rigid
/// 3e-2 to allow a usable brain-controlled twist range; the hinge swing MOTOR
/// still dominates the primary stroke (joints never separate — the point
/// anchor is unaffected by this).
pub const LIMB_HINGE_ALIGN_COMPLIANCE: f32 = 3e-2;

/// Max torque (about each limb's LOCAL long axis) of the brain-driven TWIST.
/// The policy outputs a twist command in [-1,1] per limb (the second half of its
/// action vector); `drive_limb_twist` writes `axis · cmd · MAX_LIMB_TWIST_TORQUE`
/// into the limb's `ConstantLocalTorque`. This is the off-swing-axis "roll" DOF
/// that lets limbs move in 3D, not just swing fore/aft. Kept modest: open-loop
/// twist torque reacts on the near-massless base, so a large value destabilises
/// the gait. Paired with the opt-in `K_TWIST` reward cost so the brain only
/// twists when it pays.
pub const MAX_LIMB_TWIST_TORQUE: f32 = 0.2;

/// Opt-in cost on mean |twist command| in the limb reward. Small: makes the
/// brain twist only when it buys enough locomotion reward to offset the cost,
/// instead of twisting gratuitously (which destabilises the light body and
/// regressed walking). Keeps the twist DOF available without making it free.
pub const K_TWIST: f32 = 0.4;

/// Spring-damper parameters for the limb `RevoluteJoint` motors. Avian's built-in
/// `MotorModel::SpringDamper` (solved inside XPBD) tracks the brain's per-tick target
/// hinge angle — no external PD controller. The in-solver motor drives the hinge axis
/// only, is implicit-Euler (unconditionally stable at any frequency), and reads the
/// angle via `atan2` (no gimbal singularity). `frequency` (Hz) = convergence speed /
/// stiffness (∝ frequency²): high enough that planted feet HOLD the light body up
/// rather than folding; `damping_ratio` 1.0 = critically damped (no overshoot).
/// `MAX_LIMB_TORQUE` is the motor's `max_torque` clamp.
pub const LIMB_MOTOR_FREQUENCY: f32 = 4.0;
pub const LIMB_MOTOR_DAMPING_RATIO: f32 = 1.0;

/// Rapier revolute position-motor gains for limb hinges (`rapier_setup::revolute_data`).
/// Rapier's motor is a PD controller: `stiffness` is the proportional gain pulling the
/// hinge toward the brain's target angle (must be high enough that planted feet HOLD the
/// light body up rather than folding under its weight) and `damping` the derivative gain
/// (critically-damped tracking, no overshoot/oscillation). Tuned against the very low
/// `BODY_PART_DENSITY`; the per-step motor impulse is internally capped so even large
/// stiffness can't explode the light bodies. Raise `LIMB_MOTOR_STIFFNESS` if legs sag
/// under load; raise `LIMB_MOTOR_DAMPING` if hinges jitter/overshoot.
pub const LIMB_MOTOR_STIFFNESS: f32 = 6.0;
pub const LIMB_MOTOR_DAMPING:   f32 = 1.2;

// (colony.rs)
/// Angular damping for the BASE body of a limb-based organism. High,
/// because the base has no PD actuator of its own — joint-constraint
/// reaction torques from the limbs would otherwise integrate unbounded
/// and spin the body up. Raised to 12: a COMPACT body (near-zero yaw inertia,
/// e.g. the Lancer) spins on every asymmetric stroke and wanders instead of
/// walking straight; strong yaw drag gives it the rotational stability to hold
/// a heading. Spread-out forms have high yaw inertia and are barely affected.
pub const BASE_ANGULAR_DAMPING: f32 = 12.0;

/// Angular damping for LIMB bodies. Lower than the base so the motor can still swing
/// the legs, but high enough to bleed the runaway sustained-high-ω build-up behind
/// limb "explosions"/joint-separation without blocking the brief propulsive kicks
/// that drive walking.
pub const LIMB_ANGULAR_DAMPING: f32 = 7.0;

/// Linear damping for every limb-based body part (base + limbs). Light — bleeds
/// drift between strokes without freezing the slow crawl. (Damping cannot fix
/// the hop: ≥16 froze the crawl too, ≤1 drifted — the hop-vs-crawl distinction
/// is a gait/control matter handled by the ground-gated movement reward.)
pub const LIMB_LINEAR_DAMPING:  f32 = 0.6;

/// Material density for deriving the mass of EVERY body part of a limb organism
/// (base + limbs identically; sliding organisms are kinematic so it doesn't apply).
/// Density × volume → mass. Kept very low (amoeba/paramecium/ant scale) so the legs
/// easily hold the light body up, feet rest on the surface without penetrating, and
/// the motor torques can lift/move it; coupled with `MAX_LIMB_TORQUE` (needed torque
/// ∝ inertia). Heavier values collapse the body belly-down and drive feet through
/// the terrain.
// Raised from the Avian-era 0.0012: that near-zero mass/inertia is why Rapier
// impulse joints couldn't hold the limbs (flung 4–9u) and the multibody solver
// NaN'd. With ACCELERATION-BASED limb motors the base-weight ÷ leg-torque ratio
// is density-independent, so a sane density restores joint stability WITHOUT
// changing whether the legs can hold the body up. Tune via the joint-separation
// telemetry; lower toward 0.01 if standing needs lighter bodies.
pub const BODY_PART_DENSITY: f32 = 0.012;


// ── Brain RL hyperparameters (limb_ppo.rs) ──────────────────────────────────

/// Per-organism rollout length. "Short rollout" per the design choice;
/// each agent fills its own buffer independently — when a buffer is
/// full, that agent runs its own PPO update against its own data.
pub const ROLLOUT_LEN: usize = 64;

/// PPO update epochs per filled rollout.
pub const PPO_EPOCHS: usize = 4;

/// PPO clip range (ε).
pub const CLIP_EPS: f32 = 0.2;

/// Discount factor for value bootstrapping.
pub const GAMMA: f32 = 0.99;

/// GAE smoothing factor.
pub const LAMBDA: f32 = 0.95;

/// Adam learning rate.
pub const LR: f64 = 3e-4;

/// Coefficient on the value (critic) loss term.
pub const VALUE_LOSS_COEF: f32 = 0.5;

/// Coefficient on the entropy bonus term. Zeroed: with the sparse movement
/// advantage, any positive bonus diffused the policy (entropy rose while return fell)
/// faster than a propulsive stroke could be concentrated, preventing gait
/// convergence. Exploration is instead supplied by the fixed sampling σ
/// (`LOG_STD_INIT`), so this is redundant here.
pub const ENTROPY_COEF: f32 = 0.0;

/// Exploration std as `log σ` for the diagonal-Gaussian limb policy; sampler uses
/// `σ = exp(LOG_STD_INIT)` (`exp(-1.2) ≈ 0.30`). Moderate noise that perturbs the
/// warm-started rhythmic gait (see `BrainPoolLimb::new`) so PPO can shape it without
/// drowning the ±0.46 oscillation. Larger σ (≈1.0) makes each joint target essentially
/// random and breaks locomotion; raise if learning stalls, lower if the gait jitters.
pub const LOG_STD_INIT: f32 = -1.6;



// ── Limb locomotion: EMERGENT, brain-driven (rapier_setup::drive_limb_motors) ──
//
// No CPG, no pursuit force. Each limb hinge's target angle is set DIRECTLY from the
// brain's per-joint output (`Organism::limb_targets[joint] · LIMB_SWING_LIMIT`) and
// the in-solver spring-damper motor tracks it; walking must EMERGE from RL (reward
// shapes forward velocity / progress / anti-spin). The only rhythm aid is a phase
// signal given as an OBSERVATION (sin/cos of a slow virtual-time clock) so a
// feedforward policy can phase-lock a sustained oscillation — a phase-conditioned
// policy, the standard way to get learned (not scripted) legged gaits.

/// Number of limb hinge joints the brain independently controls / observes — the
/// SWING action dimension and per-joint observation bound; brain output `k` drives
/// body-part `k+1`'s hinge. Sized (8) to cover the multi-segment Bilateral
/// morphologies in `species/` (e.g. Crawler: base + 2 hips + 2 knees → 8 limb parts
/// after mirror expansion) so each hip/knee gets its own command. Parts beyond this
/// wrap modulo. `limb_ppo::OUT = MAX_LIMB_JOINTS + N_LIMB_TWIST_GROUPS`.
pub const MAX_LIMB_JOINTS: usize = 8;

/// Number of GROUPED brain-driven twist commands (the off-swing-axis DOF). Kept
/// SMALL (2) on purpose: per-limb twist (8 outputs) doubled the policy action
/// space and measurably degraded locomotion learning (the cost was the action
/// dimension, not the twist torque). 2 interleaved groups (limb i → group
/// `(i-1) % N`) still let the brain twist limbs off their single swing plane,
/// at +2 outputs instead of +8. `drive_limb_twist` maps limbs to groups.
pub const N_LIMB_TWIST_GROUPS: usize = 2;

/// Frequency (Hz of virtual time) of the phase-clock OBSERVATION fed to the limb
/// brain (NOT a motor command — it reads `sin/cos(2π·f·t)`). Sets the cadence the
/// learned gait tends to lock onto; ~1 Hz is a plausible stride rate for these small
/// bodies. Invariant to `TimeSpeed`/frame rate (from virtual elapsed time).
pub const GAIT_FREQUENCY_HZ: f32 = 1.0;

/// Hard per-part velocity governor (`MaxLinearSpeed` / `MaxAngularSpeed`). Safety net
/// against runaway "explosions"/joint-separation: a destabilising limb otherwise
/// reaches huge velocity whose momentum the rigid constraint can't arrest in the
/// substep budget on these ultra-light bodies. Capping keeps the joint impulse bounded
/// (joints hold) and forbids the airborne fling; a real walking stride is well below.
pub const MAX_LIMB_LINEAR_SPEED:  f32 = 4.0;  // u/s — tightened to clip the unstable minority that flings limbs (a static stand needs almost no body speed)
pub const MAX_LIMB_ANGULAR_SPEED: f32 = 8.0;  // rad/s — clips spin-out while leaving room for a leg to swing into a stance


// ── Brain reward weights (limb_ppo.rs) ──────────────────────────────────────

pub const K_EAT:      f32 = 4.0;
pub const K_REPRO:    f32 = 5.0;
/// Reward per unit of forward (heading-projected) velocity. Signed: directed travel
/// pays, spin nets ~0, backward drift is penalised — biases discovered motion toward
/// directed travel rather than undirected drift.
pub const K_FWD:      f32 = 0.3;
pub const K_UP:       f32 = 0.02;
/// Flat idle penalty `−K_IDLE·(1−motion_gate)`. Zeroed: a flat penalty is a
/// gradient-free negative plateau (near-identical for every still organism) that
/// taught nothing. Replaced by `K_MOVE`, a positive climbable movement gradient.
pub const K_IDLE:     f32 = 0.0;
/// Reward per world-unit of XZ distance closed to the nearest photoautotroph since
/// the last tick — a dense directional signal paid even before locomotion is
/// discovered. Mirrors the sliding pools' `K_PROGRESS`.
pub const K_PROGRESS: f32 = 3.0;
/// Reward per tick-over-tick improvement in heading-alignment toward the nearest
/// photoautotroph. `alignment = dot(body_heading_xz, prey_dir_xz) ∈ [-1,1]`; reward
/// is the RECTIFIED delta `max(0, now − prev)`, so only turning TOWARD the target is
/// paid (turning away is free). Rewards the ACT of turning, not static facing, to
/// avoid the alive-bonus trap — the limb analogue of the sliding pool's "rotate to
/// face travel".
pub const K_HEADING: f32 = 2.0;
/// Reward per LIMB-contact TRANSITION (foot lifting off / planting) since the last
/// tick — intended as a dense gradient toward stepping rather than dragging or
/// air-waving feet (base contact excluded). Currently ZEROED: the policy farmed it by
/// flailing feet on/off the ground in place without locomoting, so only the unfarmable
/// movement terms (`K_MOVE`/`K_FWD`, zero at rest) remain climbable. A feet-air-time
/// shaping (cf. legged_gym) could reintroduce stepping credit without the exploit.
pub const K_STEP: f32 = 0.0;

/// Speed (world-units/sec) at which the uprightness reward fully activates and the
/// idle penalty bottoms out. Below this the brain counts as "not really moving" and
/// the alive-bonus is dimmed proportionally.
pub const IDLE_THRESH: f32 = 0.5;

/// Reward per (capped) world-unit/s of RAW body speed — the primary dense, climbable
/// "move at all" gradient that bootstraps locomotion out of the still basin. Unlike
/// `K_FWD` (forward-projected, ~0 until the body already travels facing-wise), this
/// pays ANY motion; `K_FWD`/`K_PROGRESS`/`K_HEADING` then bend it toward prey and
/// `K_SPIN` discourages spin. GATED on uprightness (`limb_ppo.rs`) and capped
/// (`SPEED_REWARD_CAP`), so it can't be won by tumbling or ballistic flight.
pub const K_MOVE: f32 = 0.9;

/// Cap (world-units/s) on the speed `K_MOVE` rewards. Set near crawl speed so the
/// cap actually bites: above it, extra speed earns nothing, so a fast ballistic
/// hop earns no more `K_MOVE` than a steady grounded crawl — removing the speed
/// incentive that drove hopping (a high cap rewards the hop's raw velocity).
pub const SPEED_REWARD_CAP: f32 = 1.2;

/// Penalty per rad/s of base ANGULAR speed — steers the movement gradient toward
/// translation rather than the high-speed spin/flight that an unstable body falls
/// into, without touching the legitimate limb-joint angular velocities.
pub const K_SPIN: f32 = 0.08;

/// Number of planted feet at which the movement reward (`K_MOVE`/`K_FWD`) earns
/// full credit; fewer scales it down linearly, zero (airborne) earns nothing.
/// This is the ground gate that stops a ballistic hop — which has high XZ speed
/// while airborne — from out-scoring a grounded crawl. Morphology-general: a few
/// planted feet suffice, so it works for any limb count/form.
pub const GROUND_GATE_MIN_FEET: f32 = 2.0;

/// Penalty per (world-units/s) of UPWARD base velocity. Taxes the hop's defining
/// feature (a launch has large +vy at take-off) while leaving the horizontal
/// crawl (vy≈0) untouched — complements the ground gate to suppress hopping.
pub const K_VERT: f32 = 0.6;

/// Per-tick penalty while FULLY airborne (no feet planted), graded between 0 and
/// 1 planted foot. Taxes the whole airborne GLIDE, not just take-off (`K_VERT`),
/// so a "launch once, glide far" strategy is strictly dominated by landing —
/// pulls the stragglers out of the ballistic local optimum into grounded walking.
/// Only fires when ALL feet are off the ground, so a normal alternating gait
/// (always ≥1 foot down) is never penalised. Kept moderate: stronger values just
/// shuffled which (stochastic) stragglers stayed stuck without raising the count.
pub const K_AIR: f32 = 0.5;

/// Per-tick penalty while the BASE (belly) touches the ground. Makes the policy
/// hold the body UP on its legs (stand/walk) instead of collapsing to a
/// belly-drag, which is otherwise effortless and "free". Composes with `K_AIR`
/// (don't go airborne) + the planted-feet movement gate so the optimum is
/// "feet planted, belly off the floor" = legged walking, for any morphology.
pub const K_BELLY: f32 = 1.2;


// ── STANDING task reward (limb_ppo.rs) ──────────────────────────────────────
// Goal: four-legged runners learn to STAND UPRIGHT on their whole legs (incl.
// sub-limbs) — no directed locomotion yet. Recipe adapted from the proven
// standing/balancing RL literature (DeepMind Control Suite walker "stand":
// reward = mix of uprightness + minimal-torso-height; ETH legged_gym
// projected-gravity orientation + base-height-target terms; quadruped
// fall-recovery height-constraint as the dominant term). The PRIMARY term is
// the MULTIPLICATIVE product tall×level×feet-planted, so the policy scores only
// when the base is held HIGH (forcing knee/sub-limb extension — the whole-leg
// requirement), LEVEL, and supported on planted legs. Penalties suppress the
// known failure modes (belly-resting, tipping, hovering/bouncing, spinning,
// trembling). When `STANDING_TASK` is false the reward reverts to locomotion.

/// Master switch: when true, the limb PPO reward optimises STANDING (terms
/// below) instead of locomotion (K_MOVE/K_FWD/K_PROGRESS/K_HEADING). Lets a run
/// target standing first (curriculum: posture before locomotion).
pub const STANDING_TASK: bool = false;

/// Weight on the primary standing reward `stand = standing·upright·foot_support`
/// (each factor ∈ [0,1]). Dominant term (cf. the weight-10 height constraint in
/// quadruped fall-recovery work, and dm_control's height×upright "stand").
pub const K_STAND: f32 = 2.5;

/// Target base-clearance (world units above terrain) at which the height factor
/// `standing` saturates to 1. Set ≈ the creature's standing leg reach so that
/// reaching it is geometrically impossible without EXTENDING the lower
/// sub-limb segments — this is what forces "stand on the whole leg" rather than
/// crouching. `standing = clamp(base_clearance / target, 0, 1)` (a robust linear
/// ramp; over/under-estimating the target only changes where it saturates, never
/// inverts the gradient). Provisional — refine from the first run's measured
/// `base_clearance` distribution (telemetry column).
// DATA (2026-06-06): a standing Runner's measured base_clearance is ~5.4 (part-
// CENTRE height; the geometric foot-to-base reach is ~4.6). At the old target 6.0
// the `tall` factor capped at ~0.9 for a perfect stand — it could never saturate,
// so the policy was always told its best posture was "incomplete". Set just below
// the achieved stance so a real stand saturates the primary term. (Reward-only;
// the saved standing policy in Runners_standing.colony is unaffected.)
pub const STAND_HEIGHT_TARGET: f32 = 5.0;

/// Planted limb-segment count at which `foot_support` saturates to 1 (fewer
/// scales it down linearly). Counts ALL planted limb parts incl. sub-limbs, so
/// weight-bearing on the lower segments (guiderail: whole legs) is what pays.
pub const STAND_MIN_FEET: f32 = 3.0;

/// Penalty per unit of NON-uprightness `(1 - (1+up_z)/2)` — an explicit, always-on
/// levelness gradient on top of the multiplicative `upright` factor (guiderail:
/// any non-horizontal base orientation is penalised).
pub const K_TILT: f32 = 0.6;

/// Penalty per (world-units/s) of horizontal base speed: a STAND should be quiet,
/// not wander. (Lifted into locomotion phases by flipping `STANDING_TASK`.)
pub const K_DRIFT: f32 = 0.15;

/// Penalty on mean squared SWING action magnitude — a light energy/torque
/// regulariser discouraging the policy from jamming joints at the limits or
/// burning torque to hold a clenched pose (legged_gym/Heess control penalty).
pub const K_TORQUE_REG: f32 = 0.08;

/// Penalty on mean |action − prev_action| over the swing joints — action-rate
/// smoothness, the standard cure for trembling/buzzing standing policies.
pub const K_ACTRATE: f32 = 0.1;

/// Small per-tick ALIVE bonus paid only while the body is upright AND its belly
/// is off the floor — the survival signal that, with the stillness penalties,
/// makes "hold a stable upright stance" the optimum (cf. Heess humanoid +0.02
/// alive bonus + fall termination; we have no per-creature episode reset, so the
/// gated bonus is the continuous analogue).
pub const K_ALIVE: f32 = 2.5;

/// Uprightness `(1+up_z)/2` threshold above which (and belly-off-floor) the alive
/// bonus is paid and the body counts as "standing" for diagnostics.
pub const STAND_UPRIGHT_MIN: f32 = 0.6;

/// During the STANDING task, cull the loaded limb-organism cohort down to this
/// many (one-shot, after load). The Runners are heavy multi-part creatures
/// (~50× a Crawler), so 40 of them crater the frame rate to ~1 FPS; a handful is
/// plenty for PPO (each organism is an independent learner) and keeps the sim
/// near the 60-FPS target so the physics stepping stays well-conditioned.
pub const STANDING_MAX_LIMB_ORGS: usize = 6;

/// Locomotion-task limb-organism cap (cull, after load). Each limb herbivore is many
/// dynamic bodies + joints, and rapier physics is CPU-bound, so the cohort size sets
/// the achievable time-speed before FixedUpdate death-spirals to ~0 FPS. 8 (down from
/// 16) halves the CPU physics load so a modest workstation (e.g. Ryzen 1700x) sustains
/// a few × real-time without freezing; with a SHARED swim policy 8 agents pool training
/// data as well as 16, so learning isn't hurt. A multi-part sub-limbed Swimmer (9
/// dynamic parts) is ~2× a simple limb organism, so keep this small for swimmers.
pub const LOCOMOTION_MAX_LIMB_ORGS: usize = 8;

/// Training scaffold: suppress LIMB-herbivore reproduction under `AiTrainingMode`
/// (read by `reproduction.rs`). Keeps the learning cohort fixed at the culled count
/// and stops mid-run offspring from hitting the steering rotation half-initialised
/// (which momentarily separated their joints — G2). Decoupled from `--max-herbivores`
/// so the GPU brain-pool size (and thus FPS) is unaffected.
pub const DISABLE_LIMB_HERBIVORE_REPRODUCTION: bool = true;

/// PROXIMITY-predation radius (world units) for limb herbivores. Eating is triggered
/// when a limb herbivore's base gets within this of a phototroph, NOT by physical
/// collider contact — and the limb↔phototroph physical collision is filtered out. A
/// successful seeker converges on prey; the dense limb↔prey CONTACT pile that
/// contact-predation needs craters the frame rate (~1 FPS). Proximity eating keeps
/// "move to prey → eat it" while letting herbivores pass through prey (no pile) → FPS.
pub const EAT_RADIUS: f32 = 6.0;

/// STANDING curriculum ("training wheels"): early in a standing run the limb
/// bodies' gravity is scaled DOWN so the legs can easily hold the body up and the
/// policy learns the bracing/balance pose under light load; the scale then ramps
/// linearly to full gravity (1.0) over `STANDING_GRAVITY_FADE_SECS`, so by the end
/// the creature maintains the learned stance under its true weight. Reduced
/// gravity is mass-independent, so it doesn't bias which legs/poses work — it just
/// makes the unstable tall stance learnable instead of collapsing to a belly-squat.
pub const STANDING_GRAVITY_START:     f32 = 0.15;  // ×gravity at t=0
pub const STANDING_GRAVITY_FADE_SECS: f32 = 450.0; // slow ramp → more learning per weight level

/// STANDING fall-reset (episode reset). Standard practice in legged-robot RL: when
/// the agent falls it is reset to the standing pose so the policy keeps
/// experiencing the standing region and learns to RECOVER/HOLD it. AEONS has no
/// episode boundaries, so without this a Runner that collapses to its belly stays
/// there forever and ALL subsequent experience is belly-state — the policy can
/// never relearn standing (the documented "stuck in the belly-squat attractor"
/// failure). `reset_fallen_standers` (rapier_setup) teleports a fallen limb
/// organism's parts back to their `LimbRestPose` (spawn standing pose) and zeroes
/// velocities. A "fall" = base belly-contact AND base uprightness below
/// `STAND_RESET_UPRIGHT`, sustained `STAND_RESET_GRACE_SECS`; after a reset, a
/// `STAND_RESET_COOLDOWN_SECS` window lets the policy attempt the stance before it
/// can be reset again.
pub const STAND_RESET_UPRIGHT:        f32 = 0.6;  // base up·Y below this (≈ >53° tilt) counts as fallen
pub const STAND_RESET_GRACE_SECS:     f32 = 1.0;  // sustained-fall time before a reset fires
pub const STAND_RESET_COOLDOWN_SECS:  f32 = 2.5;  // post-reset standing-attempt window (snappy → more standing-region experience)


// ── Swimming (rapier_setup.rs, colony.rs) — tune with data-analysis ──────────
//
// Real swimming locomotion: Swimming organisms are DYNAMIC per-part bodies (the
// limb physics path) with neutral buoyancy. Each part generates ANISOTROPIC
// blade-element drag as it moves through the water; the reaction force propels
// and rotates the whole organism (emergent, not scripted). All knobs below are
// gated on the `SwimmerBody` marker so walkers/standers stay byte-identical.

/// Lumped fluid density ρ for the per-part quadratic drag model.
pub const WATER_DENSITY: f32 = 1.0;
/// Base drag coefficient. Anisotropy comes from the per-axis presented AREA
/// (`LimbDragShape::area_local`), NOT from Cd — Cd is the same on all 3 axes.
pub const SWIM_DRAG_CD: f32 = 2.0;
/// Quadratic angular drag coefficient (`τ = −coef·|ω|·ω`).
pub const SWIM_ANGULAR_DRAG_COEF: f32 = 0.5;
/// Isotropic `Damping` override for swimmer parts — small, so the explicit
/// blade-element drag dominates the linear motion.
pub const SWIM_LINEAR_DAMPING: f32 = 0.05;
/// Ditto, angular.
pub const SWIM_ANGULAR_DAMPING: f32 = 0.2;
/// Inward restoring force per unit of out-of-bounds overshoot (soft world
/// borders on XZ). The water SURFACE needs no spring: water-based bodies get
/// real gravity while above it (`rapier_setup::apply_water_gravity` for
/// dynamic swimmers, `movement::apply_gravity` for kinematic floaters).
pub const SWIM_BORDER_STIFFNESS: f32 = 30.0;
/// Hard cap on each border restoring force. Bounds the `stiffness × overshoot`
/// spring so a swimmer pushed far out of bounds can't receive an explosive
/// impulse — it gets firmly but finitely pushed back instead of launched.
pub const SWIM_CONFINE_MAX_FORCE: f32 = 50.0;
/// Mass density for SWIMMER body parts. The terrestrial `BODY_PART_DENSITY`
/// (0.012) is deliberately near-massless so weak motors beat ground friction —
/// but underwater (no gravity, no friction) that tiny mass makes the quadratic
/// drag deceleration (`F/m`) explosive: the integrator overshoots, velocity
/// flips and grows to NaN, and Rapier's solver panics. Swimmers use a realistic
/// ~neutral-buoyancy density (≈ WATER_DENSITY) so the dynamics are stable. (May
/// need a matching motor-torque bump for lively strokes — that's tuning.)
pub const SWIM_BODY_DENSITY: f32 = 1.0;
/// Hard caps on the blade-element drag force (per axis) and angular-drag torque.
/// Bound the v² / |ω|² terms so a velocity spike can't produce an unbounded force
/// on a light body — belt-and-suspenders with `SWIM_BODY_DENSITY`.
pub const SWIM_DRAG_MAX_FORCE:  f32 = 25.0;
pub const SWIM_DRAG_MAX_TORQUE: f32 = 10.0;
/// Swimmer-specific HINGE MOTOR gains. Making swimmer bodies ~neutral-buoyancy
/// (SWIM_BODY_DENSITY) raised their mass ~83× over the near-massless terrestrial
/// `BODY_PART_DENSITY`, so the terrestrial motor (stiffness 6 / damping 1.2 /
/// max-torque 10) can no longer budge the limbs. Scaling all three gains by the
/// same mass ratio reproduces the (working) terrestrial per-limb angular dynamics
/// — same accelerations, same damping ratio (stable) — so the limbs actually
/// stroke. Tune from here: raise the ratio multiplier for livelier strokes, lower
/// it if they thrash. (`ForceBased` motor → these are real torques that propel.)
pub const SWIM_MASS_RATIO:      f32 = SWIM_BODY_DENSITY / BODY_PART_DENSITY;
pub const SWIM_MAX_LIMB_TORQUE: f32 = MAX_LIMB_TORQUE     * SWIM_MASS_RATIO;
pub const SWIM_MOTOR_STIFFNESS: f32 = LIMB_MOTOR_STIFFNESS * SWIM_MASS_RATIO;
pub const SWIM_MOTOR_DAMPING:   f32 = LIMB_MOTOR_DAMPING   * SWIM_MASS_RATIO;
/// Vertical clearance used when placing a swimmer in the water column at spawn:
/// kept this far below the surface and above the terrain floor so no body part
/// breaches the water plane on the first frame (which would trigger the ceiling
/// restoring force). If the column is too shallow, the swimmer is placed just
/// above the floor and the (now-bounded) ceiling force settles it.
pub const SWIM_SPAWN_CLEARANCE: f32 = 3.0;
// ── Swimming brain reward weights (swim_ppo.rs) ──────────────────────────────
//
// Swimming organisms train in their OWN PPO pool (swimming_movement/), separate
// from the limb-walking pools. Two oracles feed the policy (3D target bearing +
// the body-rotation needed to face it); the reward mirrors them: a BIG sparse
// eat event, a SMALL dense closing-progress term, and a rotation-toward-target
// objective. All 3D — a swimmer has no "up", no ground gate, no uprightness.

/// BIG sparse reward per prey eaten (Δ`Organism::predations`). The terminal
/// objective ("reach prey and eat it"); raised 20→40 so the eat clearly tops a
/// whole rollout of dense closing reward and the full approach→eat chain is
/// reinforced.
pub const K_SWIM_EAT: f32 = 40.0;
/// DOMINANT dense reward per unit of CLOSING SPEED toward the nearest
/// phototroph: `base_lin_vel · dir_to_prey` (signed — swimming away is
/// penalised). This is the instantaneous closing rate (≈ −d(dist)/dt), used
/// instead of a tick-to-tick distance delta because it is (a) available EVERY
/// tick a target exists (no prev-distance bookkeeping, no first-tick gap),
/// (b) far lower variance (reads the body's own velocity, not a noisy distance
/// difference across a moving/relocating target), and (c) a clean per-tick
/// gradient the critic can actually fit — the distance-delta version averaged
/// ~0 for a wandering policy and the value fn couldn't learn it (return
/// oscillated, 0 eats). It rewards "swim toward the target NOW", tightening the
/// oracle→act→reward loop. Clamped to the velocity governor range in swim_ppo.
pub const K_SWIM_PROGRESS: f32 = 3.0;
/// Tick-over-tick facing-alignment GAIN reward. DISABLED (was 1.5): a rectified
/// turn-toward delta is farmable (turn away free, turn back paid) and produced
/// the "orient but never approach" policy. Facing now emerges implicitly —
/// you can't close `K_SWIM_PROGRESS` distance efficiently without facing the
/// target. Kept as a knob (0.0) rather than deleted.
pub const K_SWIM_ALIGN_GAIN: f32 = 0.0;
/// Small ABSOLUTE facing bonus (`max(0, alignment)` per tick) — non-farmable
/// (rewards the STATE of facing, not the act of turning), a mild bootstrap so
/// early policies orient before they can propel. Small vs `K_SWIM_PROGRESS` so
/// it can't substitute for actually closing distance.
pub const K_SWIM_ALIGN: f32 = 0.1;
/// Mild penalty per rad/s of base angular speed — anti-corkscrew, keeps the
/// rotation objective pointed at controlled turns rather than permanent spin.
pub const K_SWIM_SPIN: f32 = 0.02;

/// Prey-sensing radius (worldspace units) for SWIMMING herbivores — the range
/// of the swim brain's target + rotation oracles, and the normaliser for the
/// observed target distance. Decoupled from the limb/sliding `PREY_SCAN_RADIUS`
/// (250) so swimmers can be tuned independently.
pub const SWIM_SENSORY_RADIUS: f32 = 200.0;

// ── Swimming exploration (swim_ppo.rs) ───────────────────────────────────────
//
// Swimmers explore limb rotation HARDER than walkers, on purpose: underwater
// there is no falling over, no ballistic-hop basin, and no joint-fling-on-
// ground-impact failure mode — the punishments that forced the terrestrial
// pools' timid exploration settings. And propulsion through quadratic fluid
// drag is only discovered by COHERENT, vigorous strokes: thrust scales with
// stroke speed², and per-tick white jitter averages to ~zero net force.

/// Sampling `log σ` for the SWIMMING policy (`σ = exp(-0.9) ≈ 0.41`, ~2× the
/// terrestrial `LOG_STD_INIT = -1.6` ⇒ σ ≈ 0.20). Bigger exploratory joint
/// excursions actually displace water; the per-axis joint limits + drag-force
/// caps bound the worst case. Lower toward -1.6 if learned strokes jitter.
pub const SWIM_LOG_STD_INIT: f32 = -0.9;

/// Per-tick autocorrelation ρ of the exploration noise — variance-preserving
/// AR(1) (discrete Ornstein–Uhlenbeck): `n_t = ρ·n_{t−1} + √(1−ρ²)·ε_t`,
/// `ε ~ N(0,1)`, so the MARGINAL of each action stays `N(μ, σ²)` (the recorded
/// log-prob stays the correct density) while excursions persist for
/// ~`tick/(1−ρ)` ≈ 1 s of virtual time at ρ=0.85 — the stroke timescale
/// (`GAIT_FREQUENCY_HZ` = 1). Exploration then looks like sustained trial
/// strokes (which produce net thrust the reward can see) instead of 150 ms
/// jitter (which fluid drag cancels). ρ=0 recovers plain white noise.
pub const SWIM_NOISE_CORR: f32 = 0.85;

/// Oscillatory warm-start amplitude for the swim actor (per-output
/// `A·sin(phase+φ)`). The limb pools keep this small (0.15) because a big
/// swing lifts all feet at once and seeds the ballistic hop; underwater there
/// is no hop basin and thrust needs stroke speed, so swimmers start vigorous.
/// Phases `φ` are RANDOMISED per organism and per output at pool init, so the
/// initial population explores DIVERSE stroke patterns (linear paddles, 3D
/// orbits, different joint orderings) instead of one shared gait.
pub const SWIM_WARMSTART_AMP: f32 = 0.35;


// ── Energy (energy.rs) ──────────────────────────────────────────────────────

pub const ENERGY_TICK_INTERVAL: f32 = 0.5;
pub const MAX_ENERGY_PER_CELL: f32 = 10.0;

/// Per-tick energy a fully-surrounded (18 RD neighbours) photo cell produces. Read by
/// `physiology.rs::PhotosyntheticCell::new` for the per-cell `energy_production`
/// cache; the photosynthesis tick runs in `physiology.rs`, not here.
pub const PHOTO_PRODUCTION_PER_CELL:  f32 = 4.0;
pub const NON_PHOTO_CONSUMPTION_PER_CELL: f32 = 0.01;

// Movement-cost coefficients (ground friction linear, fluid drag cubic in speed),
// tuned so a max-speed sprint is heavily punitive but not instantly fatal.
pub const K_GROUND_FRICTION: f32 = 0.003;
pub const K_FLUID_DRAG:      f32 = 0.03;

/// Energy cost per unit of elevation gained (gravitational-PE analogue), charged on
/// the climb accumulated since the last energy tick. Krishi is excluded from the
/// energy system, so its debt is never spent.
pub const ELEVATION_ENERGY_PER_UNIT: f32 = 0.5;

pub const DOPAMINE_DEPLETION_INTERVAL: f32 = 1.0;


// ── Physiology (physiology.rs) ──────────────────────────────────────────────

/// Physiology tick interval. 0.5 s matches the energy tick so per-cell and
/// per-organism updates stay roughly in phase.
pub const PHYSIOLOGY_TICK_INTERVAL: f32 = 0.5;

/// Hard ceiling on per-cell energy — cells never exceed it however much they'd gain
/// in a tick, keeping the value comparable across cell types and bounded.
pub const MAX_CELL_ENERGY: f32 = 1.0;


// ── Photosynthesis (photosynthesis.rs) ──────────────────────────────────────

/// Direction TOWARD the sun, as a unit vector. Must mirror the directional-light
/// orientation in `main.rs` (the light points roughly `(-0.5, -√2/2, -0.5)`; this is
/// its opposite).
pub const SUN_DIRECTION: Vec3 = Vec3::new(0.5, std::f32::consts::FRAC_1_SQRT_2, 0.5);

pub const SHADOW_CHECK_INTERVAL: Duration = Duration::from_secs(10);

/// Step length of the shadow raymarch (world units). Finer than `HEIGHTMAP_CELL_SIZE`
/// (4.0) so the ray oversamples terrain and can't skip an occluder between cell centres.
pub const RAY_STEP_SIZE: f32 = 1.0;

/// Max raymarch steps before the ray is declared escaped to the sky. 300 steps lift
/// it ~210 units (sun y ≈ 0.707) above its origin — above any plausible terrain peak.
pub const MAX_RAY_STEPS: usize = 300;


// ── Reproduction (reproduction.rs) ──────────────────────────────────────────

pub const REPRODUCTION_CHECK_INTERVAL: f32 = 2.0;

/// Energy split between parent and offspring at reproduction (50/50).
pub const OFFSPRING_ENERGY_FRACTION: f32 = 0.5;

/// Threshold (fraction of `max_energy`) above which an organism becomes a
/// reproduction candidate.
pub const REPRODUCTION_ENERGY_THRESHOLD: f32 = 0.8;

pub const HETEROTROPH_REPRODUCTION_CAP:    u8 = 2;
// (Phototrophs have NO per-individual reproduction cap — see reproduction.rs.
// Their population is bounded by `MaxPhotoautotrophs` + energy/sunlight and a
// minimum is guaranteed by the plankton auto-spawn below.)

/// Plankton auto-spawn floor: `colony::auto_spawn_plankton` keeps at least this
/// many `ball_plankton.species` organisms alive at all times (spawning the
/// deficit from the species file), guaranteeing a reliable prey field — e.g.
/// food for swimming heterotrophs — independent of reproduction dynamics.
pub const MIN_PLANKTON_COUNT: usize = 20;
/// Species file the plankton auto-spawn loads. A sessile, WATER-BASED (floating)
/// phototroph, so it stays in the water column as reachable prey.
pub const PLANKTON_SPECIES_PATH: &str = "species/ball_plankton.species";


// ── Growth (volumetric_growth/mod.rs, continuous_growth.rs) ─────────────────

/// Growth cap for variable-form (plant-like photoautotroph) organisms:
/// `continuous_growth` stops adding cells at this count. Kept at 30 to keep
/// photoautotrophs compact. Heterotrophs are bounded by body-part count instead.
pub const MAX_CELLS: usize = 30;

/// Per-organism growth cadence: each variable-form organism gets one growth tick this
/// often. 1.0 s gives a visible "growing" silhouette over ~30 s to reach the cap.
pub const CONTINUOUS_GROWTH_INTERVAL: f32 = 1.0;

/// Phase slices the per-second growth workload is spread across — each tick processes
/// ~1/30th of the population (entity-index modulo the rotating phase). Total work is
/// unchanged but the per-second allocator/command-buffer spike is smoothed out.
/// Aligned with the 30 Hz brain tick.
pub const GROWTH_PHASE_PERIOD: u32 = 30;

/// Wall-clock interval between phase steps (`CONTINUOUS_GROWTH_INTERVAL /
/// GROWTH_PHASE_PERIOD`) so the per-organism cadence is preserved.
pub const GROWTH_PHASE_STEP_SECS: f32 =
    CONTINUOUS_GROWTH_INTERVAL / GROWTH_PHASE_PERIOD as f32;


// ── Movement (movement.rs) ──────────────────────────────────────────────────

pub const MIN_DIRECTION_INTERVAL: f32 = 1.0;
pub const MAX_DIRECTION_INTERVAL: f32 = 10.0;

pub const GRAVITY:          f32 = 9.8;
pub const MAX_CLIMB_HEIGHT: f32 = 4.0;

/// Global kill-floor: any organism below this Y is despawned, reclaiming
/// brain/physics cycles from ones that slipped off the map edge or through a mesh gap
/// and fall forever. Set well below the lowest plausible terrain so legitimate
/// low-lying organisms survive.
pub const ORGANISM_DESPAWN_Y: f32 = -500.0;


// ── Collision (organism_collision.rs) ───────────────────────────────────────

/// Broad phase: organism root positions must be closer than this for any
/// further checks to run. Set generously — it's only a cheap distance test.
pub const ORGANISM_BROAD_RADIUS: f32 = 10.0;

/// Collision-pipeline tick interval. Running every frame is wasteful at large
/// organism counts; contacts emerge cleanly at 10 Hz.
pub const COLLISION_TICK: f32 = 0.1;

/// Max positional separation applied to one organism per collision tick (XZ, world
/// units). Caps the integrated push so deeply-overlapping pairs (which can generate
/// hundreds of narrow-phase contacts) don't snap apart absurdly. At 10 Hz this is
/// ~5 u/s — fast enough to resolve penetration in ~1 s, slow enough to read as a
/// firm push, not a teleport.
pub const MAX_SEPARATION_PER_TICK: f32 = 0.5;


// ── Brain tick intervals (behaviour.rs) ─────────────────────────────────────

// `PHOTO_BRAIN_TICK_INTERVAL` was retired with the L1-photo brain;
// kept here only as a documentation breadcrumb until the photo
// pool is removed entirely:
#[allow(dead_code)]
pub const PHOTO_BRAIN_TICK_INTERVAL:  Duration = Duration::from_millis(33);

/// Heterotroph brain tick rate (≈ 6.7 Hz). Slow because the reward signal is sparse
/// (small per-energy-tick loss + a rare predation jump): at faster rates the σ-noise
/// dominates displacement and per-tick rewards average to ~0. 150 ms gives larger
/// per-tick energy/progress deltas for the reward shaper and less visible direction
/// jitter.
pub const HETERO_BRAIN_TICK_INTERVAL: Duration = Duration::from_millis(150);


// ── World model (world_model.rs) ────────────────────────────────────────────

/// Radius (world units) within which neighbour organisms are considered
/// part of the heterotroph's world model.
pub const WORLD_MODEL_RADIUS: f32 = 60.0;

/// Radius (world units) the limb brain scans for the NEAREST PREY (prey-direction
/// observation + K_PROGRESS/K_HEADING reward), DECOUPLED from `WORLD_MODEL_RADIUS`.
/// The neighbour-encoding (gait observation) stays at the small radius so the
/// learned gait's input scale is preserved (widening it slowed gait convergence,
/// iter8), while prey perception reaches far enough to STEER toward food that is
/// typically 50-150u away on this map. `nearest_prey` probes the bucket range that
/// covers this; the prey-distance obs normalises by it.

/// Velocity normalisation factor — roughly expected top speed (world-units/s),
/// matching the hetero pools' `MAX_SPEED`, so a neighbour at MAX_SPEED registers as
/// `±1` on its velocity dim.
pub const VELOCITY_NORM_SCALE: f32 = 20.0;

/// See above — prey-perception radius for the limb brain, decoupled from the
/// neighbour-encoding radius.
pub const PREY_SCAN_RADIUS: f32 = 250.0;

/// HEADING-STEERING ASSIST (rapier_setup::steer_base_toward_prey). The emergent
/// limb gait produces forward crawl but the long-slab Crawler's high yaw inertia
/// + the hard RL credit-assignment mean the brain doesn't learn to AIM that crawl
/// at prey (data: 0/40 steer toward prey even with perception + steering reward +
/// long training). This assist gently sets the BASE body's yaw rate toward the
/// nearest perceived prey so the (still fully brain-driven) forward stroke carries
/// the body to food → contact → eat. Only the heading is assisted; the leg motion
/// stays emergent. Implemented as a sustained YAW TORQUE on the base (a one-shot
/// velocity poke is washed out by the base angular damping + gait joint reactions),
/// steering so the base's actual TRAVEL (velocity) direction points at prey — which
/// is exactly `approach_align`, robust to whichever body axis the gait propels along.
/// Coherent whole-creature yaw step per tick toward prey (`steer_base_toward_prey`):
/// rotate base + all limbs (position, orientation, velocity) rigidly about the base
/// so the emergent crawl's TRAVEL aims at prey while joints stay satisfied. `GAIN` =
/// fraction of the travel-vs-prey error closed per tick (clamped to the remaining
/// error → no overshoot); `MAX_YAW` caps the per-tick step. The gait stays emergent;
/// only heading is assisted. (Reproduction is disabled during training so no mid-run
/// offspring hits the rotation in a half-initialised state — that caused rare G2 spikes.)
pub const STEER_ASSIST_GAIN:    f32 = 0.20; // fraction of travel-vs-prey error closed per tick
pub const STEER_ASSIST_MAX_YAW: f32 = 0.08; // rad cap on the per-tick yaw step
/// Run the steering teleport every Nth physics tick. Throttling was tested for FPS
/// and did NOT help (the FPS sink is the limb↔prey collision pile when seekers
/// converge on prey, not the teleport's Transform writes) and it degraded aiming, so
/// it's left at 1 (every tick).
pub const STEER_INTERVAL: u32 = 1;


// ── Sensory (sensory.rs) ────────────────────────────────────────────────────

/// Radius (world units) the sensory algorithm scans for a target photo; beyond it is
/// "no target" and `Organism::target_distance` saturates here so the observation
/// stays bounded.
pub const SENSORY_RADIUS: f32 = 50.0;


// ── Predation (predation.rs) ────────────────────────────────────────────────

/// Fraction of the prey body part's energy share that becomes predator
/// energy. The "lost" 20% models metabolic inefficiency in digestion.
pub const ENERGY_TRANSFER_RATE: f32 = 0.8;


// ── World (world_geometry.rs) ───────────────────────────────────────────────

/// Edge safety zone (world units): organisms stay inside
/// `[MARGIN, MapSize − MARGIN]²` on XZ. Enforced as a movement clamp
/// (`movement::apply_world_bounds`) and as a spawn rule (every XZ-coordinate
/// `rng.random_range(..)` — initial cohort, reproduction, auto-spawn).
pub const WORLD_SAFETY_MARGIN: f32 = 15.0;


// ── Water (water.rs) ────────────────────────────────────────────────────────

pub const BUOYANCY_STRENGTH:    f32 = 12.0;
pub const TRUE_WATER_DRAG_COEF: f32 = 0.05;


// ── Krishi (krishi.rs) ──────────────────────────────────────────────────────

/// Spawn altitude above the heightmap floor, matching the clearance the colony uses
/// for procedural organisms.
pub const KRISHI_SPAWN_ALTITUDE: f32 = 1.0;

/// Uniform size multiplier applied to BOTH the visual (glb SceneRoot child) AND the
/// collision footprint (cell layout in `make_krishi_body`), kept locked so a Krishi
/// touches prey at the size it looks. Consumption scales with cell count, so a scaled
/// Krishi must eat proportionally more often (starvation time is unchanged); tune
/// alongside the `energy.rs` constants when changing ecological pressure.
pub const KRISHI_SCALE: f32 = 6.0;


// ── Spawn cohort (colony.rs) ────────────────────────────────────────────────

/// Initial Krishi cohort size. `pub` so `krishi.rs` reads it directly —
/// keeps every "how many of X spawn at startup" knob in one place.
pub const INITIAL_KRISHI: u32 = 1;

/// Fresh-start cohort (spawned when no `.colony` file is loaded):
/// authored `.species` files and how many of each to seed. Paths are
/// relative to the working directory (repo root), matching the
/// `species/` autoload convention used by the colony editor.
pub const SPAWN_SWIMMER_PATH:  &str  = "species/swimmer.species";
// = LOCOMOTION_MAX_LIMB_ORGS: every limb slot goes to a swimmer (Striders off),
// so the cull keeps a full swimmer cohort rather than an arbitrary swimmer/walker mix.
pub const SPAWN_SWIMMER_COUNT: usize = 16;
pub const SPAWN_STRIDER_PATH:  &str  = "species/Strider.species";
// Walkers disabled in the default waterworld cohort (they'd spawn stranded on the
// seabed under deep water and compete for the limb-organism cull cap).
pub const SPAWN_STRIDER_COUNT: usize = 0;
pub const SPAWN_ALGAE_PATH:    &str  = "species/sessile_algae.species";
pub const SPAWN_ALGAE_COUNT:   usize = 800;
