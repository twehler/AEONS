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


/// AI-training mode toggle for the heterotroph movement-RL
/// experiment. Runtime-editable via the "AI-training mode"
/// checkbox in the statistics panel.
///
///   * `true`  → training mode active. Heterotrophs never despawn
///               (energy still drains and clamps at 0); only the
///               despawn step is suppressed. Reproduction is NOT
///               gated by this flag anymore (so the +1.0 dopamine
///               reproduction reward still fires in training mode).
///   * `false` → normal simulation. Starved heterotrophs despawn.
///
/// Read by `energy.rs::manage_energy` only. Other systems no
/// longer gate on this flag.
#[derive(Resource, Default)]
pub struct AiTrainingMode(pub bool);


/// Upper bound on the live herbivore (heterotroph - carnivore) count.
/// When the population reaches or exceeds this number,
/// `reproduction_system` skips all reproduction events for
/// herbivores until enough have died for the count to fall below
/// the cap again. Runtime-editable via the "Max Herbivores" text
/// field in the statistics panel.
///
/// Default `100` is a starting value; the field is intended to
/// be tuned at runtime as the player explores the parameter space.
/// `0` effectively disables herbivore reproduction.
#[derive(Resource)]
pub struct MaxHerbivores(pub usize);

impl Default for MaxHerbivores {
    fn default() -> Self { Self(100) }
}


/// Top-level window-mode toggle exposed to the user via the thin top bar
/// at the top of the frontend layout. `Simulation` is the original
/// behaviour: stats panel + navigator + click-to-capture player camera.
/// `EditColony` swaps in the colony-editor panels (creation / tool /
/// inventory) and a hold-LMB-rotate flycam input mode, so the user can
/// place organisms while the existing world keeps rendering — the
/// simulation auto-pauses on entry so the population is frozen for
/// edits. Switching back to Simulation does NOT auto-resume.
#[derive(Resource, PartialEq, Eq, Clone, Copy, Debug)]
pub enum WindowMode {
    Simulation,
    EditColony,
    /// Tree-of-life view — pauses the simulation (like EditColony)
    /// and renders the `SpeciesRegistry`'s ancestry tree in place
    /// of the viewport + side panels.
    Lineages,
    /// Species editor — pauses the simulation, hides the world, and
    /// presents a top + bottom panel UI for manually constructing
    /// an organism's body cell-by-cell. The output is a `.species`
    /// binary file the user can later import into a colony.
    SpeciesEditor,
}

impl Default for WindowMode {
    fn default() -> Self { Self::Simulation }
}


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
/// (detected via the `Organism::predations` delta). Floor raised
/// from 1.0 to 4.0 so the eat jackpot is large enough to dominate
/// the per-rollout K_CURIOSITY accumulation and pin the policy
/// gradient on "actually catch prey" rather than "wander fast."
/// At the prior 1.0 floor the eat signal was within the noise band
/// of the dense per-tick rewards, leaving the population stuck in
/// the "fast wanderer" attractor rather than evolving toward the
/// "fast hunter" attractor.
pub const L1_K_EAT_RANGE:         (f32, f32) = (4.0, 12.0);

/// Range for `K_REPRO` — one-shot reward on reproduction
/// (detected via `Organism::reproductions` increment).
pub const L1_K_REPRO_RANGE:       (f32, f32) = (5.0, 30.0);

/// Range for `LAMBDA_ENERGY` — coefficient on the negative part
/// of `ΔE` per brain tick (the "energy-loss punishment").
pub const L1_LAMBDA_ENERGY_RANGE: (f32, f32) = (0.3, 2.0);

/// Range for `K_CURIOSITY` — per-tick reward proportional to
/// `applied_speed_norm − 0.5`, so stillness is negatively rewarded
/// and full speed is positively rewarded by the same magnitude.
/// Range widened to (0.4, 1.5) so the per-tick gradient between
/// min-speed and max-speed trajectories is large enough to survive
/// the per-slot EMA baseline absorption — the previous (0.15, 0.5)
/// range produced gradients that the baseline tracked to zero within
/// ~20 rollouts, killing the gradient pressure that was supposed to
/// push the policy away from the standstill local optimum.
pub const L1_K_CURIOSITY_RANGE:   (f32, f32) = (0.4, 1.5);

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
/// 0.3 means each tick the actually-executed speed moves 70% toward
/// the freshly sampled action — the agent reaches its commanded
/// speed within ~2 ticks (≈ 300 ms) instead of 5. Was 0.6 originally
/// to dampen post-eat overshoot, but with `L1_APPROACH_RADIUS = 3`
/// the brake already guarantees no overshoot for any applied speed
/// ≤ 1, so the high momentum was unnecessary friction that just
/// made motion feel sluggish.
pub const L1_SPEED_MOMENTUM_ALPHA: f32 = 0.3;

/// XZ distance (world units) at which "arrival braking" kicks in for
/// the L1 hetero brain. When the locked target is closer than this,
/// the applied movement speed is scaled by
/// `clamp(distance / L1_APPROACH_RADIUS, 0, 1)` — full speed beyond
/// the radius, linear decel inside it, zero exactly on top of the
/// target.
///
/// Sizing rationale: per-tick travel is
/// `applied_speed_a · MAX_SPEED · TICK_SECS · brake_scale`. With
/// `brake_scale = d / R` this becomes `applied · 3 · d / R`. For
/// no overshoot at maximum applied (= 1), we need `3 / R ≤ 1`, so
/// `R ≥ MAX_SPEED · TICK_SECS = 3`. Smaller radii produce
/// rapid-fire ping-pong oscillation around the target that
/// visually reads as "stuck in place, no motion".
///
/// The "agent parks outside cell-contact range" failure mode that
/// motivated trying R = 1 is handled separately by `CELL_COLLISION_RADIUS`
/// (cell.rs) — that controls the contact disc width and is what
/// determines whether the bilateral V-shape actually touches prey
/// at parking distance. Brake radius and contact radius are
/// independent dials; tune each on its own merit.
pub const L1_APPROACH_RADIUS: f32 = 3.0;

/// Number of consecutive brain ticks the locked target can stay at
/// the same distance (no measurable progress) before the lock is
/// force-dropped and the brain re-picks. Solves the "stuck against
/// a blocker" pattern where the locked photo sits past another
/// hetero or a wall: the agent stops making distance progress, this
/// counter fires, the lock drops, fresh argmax fires next tick.
///
/// At ~6.67 brain ticks per virtual second, 6 ticks ≈ 0.9 virtual
/// seconds — long enough to ride through normal contact-bounce
/// jitter, short enough that the agent doesn't burn its 10-second
/// lock window on a target it can't reach.
pub const L1_STUCK_TICKS: u16 = 6;

/// Minimum distance-closed (world units) between two brain ticks
/// that counts as "progress." Distance decreases below this
/// threshold are treated as noise and count toward the stuck
/// counter. 0.3 is just below the per-tick travel of a hetero
/// approaching at 1/4 throttle — anything less than that and the
/// agent isn't really converging on the target.
pub const L1_STUCK_PROGRESS_EPS: f32 = 0.3;

/// XZ distance (world units) below which the L1 hetero brain
/// freezes its commanded direction at the previous tick's value
/// instead of recomputing `(target − self).normalize()`. At very
/// close range the unit-vector becomes hypersensitive to tiny prey
/// wobble — a 0.05-unit lateral perturbation at d = 0.2 produces a
/// ~14° swing — which together with the per-tick reorient produces
/// visible micro-oscillation.
///
/// Earlier the threshold was 0.5, but combined with the (then 3.0)
/// arrival brake it created a terminal absorbing state OUTSIDE the
/// 1.2-unit cell-contact zone — the agent parked just shy of
/// contact and never ate. 0.1 keeps the anti-jitter behaviour while
/// only firing once the cells are essentially overlapping
/// (root-to-root ≪ 0.1 is deep inside the contact disc), so the
/// agent can still apply late lateral correction during the
/// approach.
pub const L1_DIRECTION_FREEZE_DIST: f32 = 0.1;

/// Speed scale applied to the hetero when it has no locked target
/// — i.e. no Photo entity in the K-nearest window. Without this, the
/// brain's `speed_a` sample drives the agent at full `MAX_SPEED`
/// even when there's nothing to chase, which produces high-energy
/// blind cruising and turns any cluster of heteros in a prey-empty
/// region into a thrashing collision-bounce mess.
///
/// 0.3 lets the agent still patrol the area (handy for finding the
/// nearest photo when one wanders into range) without exerting
/// enough force to deadlock against neighbouring heteros.
pub const L1_NO_TARGET_SPEED_SCALE: f32 = 0.3;

/// Minimum value the EMA-smoothed `applied_speed_a` is clamped to
/// before being multiplied by `MAX_SPEED · brake_scale` for the
/// world-facing movement command. Forces every hetero to ALWAYS
/// move at least `L1_MIN_APPLIED_SPEED · MAX_SPEED · brake_scale`
/// units/sec — stillness becomes structurally impossible no matter
/// what the policy outputs.
///
/// This sits on top of (not instead of) the curiosity-based gradient
/// pressure away from low speeds. The reward shaping still rewards
/// high speed and penalises low speed; this floor exists because
/// the per-slot EMA baseline can absorb constant offsets in the
/// reward, eventually leaving the policy with no gradient pressure
/// to escape a learned standstill. Hard-clamping the world-facing
/// speed means the policy CANNOT learn its way back to true
/// stillness, regardless of how the baseline drifts.
///
/// 0.5 = half of brake-adjusted max speed at minimum. Half-throttle
/// cruise is the default for any hetero whose policy hasn't
/// converged to a higher-speed output, so the population's baseline
/// movement intensity is visibly aggressive rather than tentative.
/// Brake-zone behaviour is unaffected because `brake_scale → 0` as
/// d → 0 still scales the effective speed to zero at the target.
pub const L1_MIN_APPLIED_SPEED: f32 = 0.5;

/// Maximum per-brain-tick random rotation applied to `movement_direction`
/// when the hetero has no locked target, in radians (≈ 8.6° at 0.15).
/// Acts as a slow Brownian wander, breaking the case where multiple
/// heteros without prey settle into a stable deadlock pointing at
/// each other. Sampled uniformly from `[-L1_NO_TARGET_WANDER_ANGLE,
/// +L1_NO_TARGET_WANDER_ANGLE]` each brain tick (≈ 6.7 Hz).
pub const L1_NO_TARGET_WANDER_ANGLE: f32 = 0.15;

/// Number of brain ticks a force-dropped target stays on the
/// per-slot blacklist after stuck-detection fires. While
/// blacklisted, the target-selection scan excludes that entity from
/// argmax — the brain MUST pick a different photo (or none). After
/// the cooldown the entity is re-eligible. Prevents the
/// "force-drop → re-pick same unreachable target → force-drop"
/// loop that otherwise produces visible oscillation against blockers.
///
/// At ~6.67 brain ticks per virtual second, 20 ≈ 3 virtual seconds —
/// long enough for the agent to commit to a different bearing and
/// physically move away from the blocker, short enough that a target
/// that has just become reachable doesn't get ignored for long.
pub const L1_TARGET_BLACKLIST_TICKS: u16 = 20;

/// Mutation strength for offspring inheritance, expressed as a
/// fraction of the range width. Each gene gets `N(0, σ²)` noise added
/// with `σ = L1_GENE_MUTATION_REL_STDDEV × (max − min)`, then clamped
/// back into the range. Small enough that offspring stay near the
/// parent (selection acts on a coherent gradient), large enough that
/// the population explores the range over generations.
pub const L1_GENE_MUTATION_REL_STDDEV: f32 = 0.05;


// ── Lineages / speciation ───────────────────────────────────────────────────

/// Per-component normalised "factor of difference" at which an
/// organism is considered to belong to a NEW species relative to a
/// candidate species' centroid. Every DNA component lives in
/// `[0, 1]` (see `lineages::dna`), so this threshold compares
/// directly against the mean-absolute-difference returned by
/// `lineages::dna::distance`. 0.05 ⇒ "5% factor difference".
///
/// Also gates the trimmed-mean computation in
/// `lineages::speciation::update_species_averages`: members whose
/// DNA is further than this from the simple mean are excluded from
/// the trimmed mean, so a single drifting individual can't drag
/// the whole species' centroid along with it.
pub const SPECIES_SEPARATION_THRESHOLD: f32 = 0.10;
