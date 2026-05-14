// Simulation settings — runtime control state.
//
// Holds the resources that describe whether the simulation is currently
// running and whether the player has captured the viewport. Other modules
// read these flags to decide whether to advance time, accept input, or
// route mouse motion to the camera.
//
// The original idea of a left-side panel was scrapped; this file is now
// the single source of truth for "live" simulation controls. Future
// additions (time-scale slider, debug overlays, gene-pool browser bindings)
// land here and get surfaced in the statistics panel UI.

use bevy::prelude::*;

/// Default value seeded into both `MaxOrganisms` and `OrganismPoolSize`
/// when nothing else (launcher / CLI flag) sets them. Picked low so a
/// fresh launch doesn't allocate huge GPU tensors before the user has
/// had a chance to choose. The launcher is the canonical way to raise
/// the value; that value flows through `run_simulation` and becomes
/// both the brain-pool size *and* the initial reproduction cap.
pub const DEFAULT_MAX_ORGANISMS: usize = 4096;

pub const DEFAULT_MAP_X:           f32        = 2048.0;
pub const DEFAULT_MAP_Z:           f32        = 2048.0;


/// Secret AI-training mode for the heterotroph movement-RL experiment.
///
///   * `true`  → training mode active. Heterotrophs never despawn (energy
///     drain still happens; the value just clamps at 0) and never
///     reproduce, so the training cohort stays at a fixed identity for
///     the duration of the run. Every other system — predation,
///     photoautotroph reproduction, Krishi, initial-cohort sizing — runs
///     exactly as in a normal simulation.
///   * `false` → normal simulation. Heterotrophs reproduce and starve as
///     usual.
///
/// Read at compile time by `energy.rs` (heterotroph despawn skip) and
/// `reproduction.rs` (heterotroph reproduction skip). No other site
/// gates on this flag.
pub const AI_TRAINING_MODE: bool = false;


/// True when the simulation is advancing (`Time<Virtual>` is unpaused) and
/// every gameplay system that depends on virtual time is doing useful work.
/// Toggled by the Start/Stop button in the statistics panel.
///
/// Initial value: `true` — the simulation auto-starts so observers see
/// life immediately. Player controls still default to off; the user
/// must click into the viewport to capture the camera.
#[derive(Resource)]
pub struct SimulationRunning(pub bool);

impl Default for SimulationRunning {
    fn default() -> Self { Self(true) }
}


/// True when the player has captured the viewport and WASD / mouse-look
/// systems should consume input. Activated by a left-click inside the
/// 3D viewport (only when the simulation is running). Deactivated by Esc.
///
/// Independent of `SimulationRunning`: pausing the simulation leaves
/// player controls untouched, and releasing player controls leaves the
/// simulation running.
///
/// Initial value: `false` — startup leaves the cursor visible and the
/// player camera idle.
#[derive(Resource, Default)]
pub struct PlayerControlsActive(pub bool);


/// Compile-time switch for the speed-dependent energy costs (ground
/// friction + fluid drag) charged to organisms every energy tick.
///
/// When `true`, movement costs energy linearly in speed (ground) and
/// cubically in speed (fluid) per `energy.rs::manage_energy`.
///
/// When `false`, both terms are zeroed — movement is effectively
/// "free" energetically. Per-cell upkeep and the climb-elevation
/// cost are NOT affected (those don't depend on movement speed).
///
/// Why this exists: in the heterotroph RL training environment the
/// agent kept collapsing to a "don't move" policy. After a predation
/// event spikes energy up from 0, the next several `manage_energy`
/// ticks drain it back to 0 via friction. During that window
/// `energy_now − prev_energy < 0` shows up as a strong negative
/// `W_ENERGY · ΔE` term in the shaped reward — punishing movement
/// at exactly the moment the policy is being reinforced for the
/// action that produced the predation. Disabling movement-cost
/// energy lets us isolate whether that's the dominant catastrophic-
/// forgetting source.
pub const MOVEMENT_ENERGY_COSTS_ENABLED: bool = true;


/// Global simulation-time multiplier. Drives `Time<Virtual>::set_relative_speed`,
/// so every system reading `Res<Time>` inherits the scaled delta —
/// energy ticks, brain ticks, photosynthesis, predation, movement,
/// reproduction, the panel timer, all of it.
///
/// The player camera is not affected: it reads `Res<Time<Real>>`
/// directly so the user can still navigate at normal speed while
/// the simulation runs at e.g. 10x.
///
/// 1.0 = baseline (real-time-ish). 0.0 freezes virtual time without
/// invoking the explicit pause path. Values past ~5–10 will start
/// stressing the GPU brain pools (more ticks per real second) — the
/// user is expected to dial this up empirically.
#[derive(Resource)]
pub struct TimeSpeed(pub f32);

impl Default for TimeSpeed {
    fn default() -> Self { Self(1.0) }
}


/// When `true`, adult organisms get their body-part meshes smoothed via
/// the Jacobi vertex smoother in `volumetric_growth::smooth_vertices`.
/// Smoothing happens at most once per organism — at spawn for
/// non-variable-form organisms, on the continuous-growth tick that
/// crosses `MAX_CELLS` for variable-form organisms.
///
/// When `false`, the faceted rhombic-dodecahedron mesh is used
/// throughout the organism's life. Toggling at runtime is non-retroactive:
/// already-smoothed meshes stay smoothed, already-faceted ones stay
/// faceted; only future spawn / adult-transition events read the
/// current value.
///
/// Initial value: `true` — preserves the most recently-implemented
/// visual default.
#[derive(Resource)]
pub struct Smoothing(pub bool);

impl Default for Smoothing {
    fn default() -> Self { Self(true) }
}





/// Runtime-adjustable upper bound on the live OrganismRoot count.
///
/// Reproduction reads this resource each tick — when set lower than the
/// current population, `apply_max_organisms_cull` (in
/// `statistics_panel.rs`) despawns a random subset to meet the new cap
/// in one step.
///
/// The hard ceiling for this value is `OrganismPoolSize` (the GPU
/// brain-pool tensor size, fixed at startup). The statistics-panel
/// commit clamps user input to `[0, OrganismPoolSize]`.
#[derive(Resource)]
pub struct MaxOrganisms(pub usize);

impl Default for MaxOrganisms {
    fn default() -> Self { Self(DEFAULT_MAX_ORGANISMS) }
}


/// Real-time interval between consecutive autosaves, in **minutes**.
/// The autosave system uses `Time<Real>` so it ticks at wall-clock
/// pace regardless of the simulation's virtual-time multiplier — saves
/// remain on a predictable real-time cadence even at 10× sim speed.
///
/// Lower values produce more frequent backups at the cost of more
/// disk churn (each save writes the entire colony state, ~tens of
/// KB to a few MB depending on population). Higher values risk
/// losing more progress between manual saves.
pub const AUTOSAVE_INTERVAL_MINUTES: f32 = 5.0;


/// Default minimum heterotroph count enforced by `AutoSpawnHeteros`.
/// Picked low so a fresh enable doesn't dump a huge cohort onto the
/// map before the user has had a chance to dial the value in.
pub const DEFAULT_MIN_HETERO_COUNT: usize = 50;


/// When `true`, the auto-spawn system tops the heterotroph population
/// up to `MinHeteroCount` whenever a death event fires. Off by default
/// — the user enables it via the checkbox at the bottom of the
/// individuum-navigator panel.
///
/// The system itself only does work on heterotroph death events
/// (`RemovedComponents<Heterotroph>` is empty in steady state) AND when
/// this flag is `true`, so the off-state has zero overhead.
#[derive(Resource, Default)]
pub struct AutoSpawnHeteros(pub bool);


/// Target lower bound on the live heterotroph count. Read by the
/// auto-spawn system only while `AutoSpawnHeteros(true)`.
#[derive(Resource)]
pub struct MinHeteroCount(pub usize);

impl Default for MinHeteroCount {
    fn default() -> Self { Self(DEFAULT_MIN_HETERO_COUNT) }
}


/// Edit-state for the navigator panel's "Min heterotroph count" input
/// field. Mirrors the `MaxOrganismsEditState` model: `focused` is true
/// while the user is typing; `buffer` holds the in-progress digits.
#[derive(Resource, Default)]
pub struct MinHeteroCountEditState {
    pub buffer:  String,
    pub focused: bool,
}


/// Brain-pool batch dimension `N` chosen at startup, fixed for the
/// lifetime of the process.
///
/// The four `BrainPool*` resources size their GPU tensors against this
/// value during `FromWorld::from_world`. Because the CubeCL kernel
/// cache + the burn-cuda tensor allocations are pinned to this shape,
/// it CANNOT change at runtime — the statistics panel's editable
/// "Max Organisms" field is clamped to it.
///
/// Set by `main.rs::run_simulation` from the launcher's input
/// (`--max-organisms N`). When no flag is provided this falls back to
/// `DEFAULT_MAX_ORGANISMS`.
#[derive(Resource)]
pub struct OrganismPoolSize(pub usize);

impl Default for OrganismPoolSize {
    fn default() -> Self { Self(DEFAULT_MAX_ORGANISMS) }
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

/// Range for `K_EAT` — one-shot reward on a predation event
/// (detected via energy spike).
pub const L1_K_EAT_RANGE:         (f32, f32) = (1.0, 6.0);

/// Range for `K_REPRO` — one-shot reward on reproduction
/// (detected via `Organism::reproductions` increment).
pub const L1_K_REPRO_RANGE:       (f32, f32) = (5.0, 30.0);

/// Range for `LAMBDA_ENERGY` — coefficient on the negative part
/// of `ΔE` per brain tick (the "energy-loss punishment").
pub const L1_LAMBDA_ENERGY_RANGE: (f32, f32) = (0.3, 2.0);

/// Range for `K_CURIOSITY` — per-tick reward proportional to the
/// previous action's normalised speed. Lower bound is 0.0 so the
/// population can include "no curiosity" phenotypes that only learn
/// from real eat / repro events.
pub const L1_K_CURIOSITY_RANGE:   (f32, f32) = (0.0, 0.5);

/// Range for `K_PROGRESS` — per-tick reward proportional to the
/// distance CLOSED to the currently-locked target. Only positive
/// closing distance is rewarded; receding doesn't fire (so the
/// reward never punishes a target switch).
pub const L1_K_PROGRESS_RANGE:    (f32, f32) = (0.5, 3.0);

/// Duration of the target-lock window, in **virtual seconds**. While
/// the lock is active the policy's target-choice logits are ignored —
/// direction stays geometric toward the locked entity. Without this,
/// σ-noise on the choice logits flickers target choice every brain
/// tick and the heterotroph stutters between prey it never reaches.
///
/// Expressed in virtual seconds (not ticks / real seconds) so the
/// duration scales naturally with `TimeSpeed`: at 10× sim speed, a
/// 10-second virtual lock plays out in 1 real second. The brain
/// itself ticks on `Time<Virtual>` so the tick-counter representation
/// (`lock_ticks_remaining` on the brain pool) is just a
/// straightforward `secs / brain_tick_interval` conversion done at
/// the use site.
pub const L1_TARGET_LOCK_SECS: f32 = 10.0;

/// After the lock window expires, the network may switch targets,
/// but only if the new winner's logit beats the current target's
/// logit by at least this margin. Prevents oscillation when two
/// prey have near-equal scores. Tanh outputs are in `[-1, +1]`, so
/// a 0.15 margin is ~7.5% of the full output range.
pub const L1_TARGET_SWITCH_MARGIN: f32 = 0.15;

/// EMA factor for output-side speed momentum:
/// `applied_speed = α · prev_applied + (1 − α) · new_sample`.
/// 0.6 means each tick the actually-executed speed moves 40% toward
/// the freshly sampled action — smooth enough to avoid jerk, fast
/// enough that the agent can decelerate when needed.
pub const L1_SPEED_MOMENTUM_ALPHA: f32 = 0.6;

/// Mutation strength for offspring inheritance, expressed as a
/// fraction of the range width. Each gene gets `N(0, σ²)` noise added
/// with `σ = L1_GENE_MUTATION_REL_STDDEV × (max − min)`, then clamped
/// back into the range. Small enough that offspring stay near the
/// parent (selection acts on a coherent gradient), large enough that
/// the population explores the range over generations.
pub const L1_GENE_MUTATION_REL_STDDEV: f32 = 0.05;
