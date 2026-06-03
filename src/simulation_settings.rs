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
use std::time::Duration;

/// Default value seeded into `MaxPhotoautotrophs` when nothing else
/// (launcher / CLI flag) sets it. This is the *running-population
/// cap* on photoautotrophs only — heterotrophs have their own
/// independent cap (`MaxHerbivores`). The GPU brain-pool batch dim
/// (`OrganismPoolSize`) is derived separately from `MaxHerbivores`
/// at startup, since only heterotrophs use brain slots.
// Lowered 800 → 150 (2026-06-02) to protect frame rate: photo organisms
// reproduce up to this cap, and at 800 the scene bloated (meshes + colliders
// + photosynthesis) until FPS collapsed progressively from ~60 to <1 over a
// few minutes. 150 keeps prey plentiful for herbivores while holding ~60 FPS.
pub const DEFAULT_MAX_PHOTOAUTOTROPHS: usize = 150;

/// Launcher-side default for the herbivore reproduction cap. The
/// reproduction system stops scheduling new herbivore births once
/// this number is reached. Kept small by default so a fresh launch
/// stays manageable; the launcher text field lifts it for AI-training
/// runs.
// Lowered 100 → 60 (2026-06-02): each limb herbivore is many dynamic bodies
// + joints (CPU physics), so an unbounded herbivore population also crushes
// FPS. 60 holds frame rate while leaving a healthy population to study.
pub const DEFAULT_MAX_HERBIVORES: usize = 60;

/// Launcher-side default for the initial herbivore cohort size at
/// `spawn_colony` (when no colony save is loaded). Independent from
/// `DEFAULT_MAX_HERBIVORES` so the user can seed a small starter
/// population and let reproduction grow it up to the cap.
pub const DEFAULT_START_HETEROTROPHS: usize = 100;

/// Launcher-side default for the initial photoautotroph cohort size
/// at `spawn_colony` (when no colony save is loaded). Independent
/// from `DEFAULT_MAX_PHOTOAUTOTROPHS` so the user can seed a small
/// starter population and let reproduction grow it up to the cap.
pub const DEFAULT_START_PHOTOAUTOTROPHS: usize = 800;


/// Number of heterotrophs to spawn at `spawn_colony` startup. Set
/// from the launcher's "Start Heterotroph Number" field (or the
/// `--start-heteros N` argv flag). Distinct from `MaxHerbivores`,
/// which caps the running herbivore population — this resource only
/// drives the *initial* cohort.
#[derive(Resource)]
pub struct StartHeterotrophs(pub usize);

impl Default for StartHeterotrophs {
    fn default() -> Self { Self(DEFAULT_START_HETEROTROPHS) }
}

/// Number of photoautotrophs to spawn at `spawn_colony` startup. Set
/// from the launcher's "Spawn Phototrophic Organisms" field (or the
/// `--start-photos N` argv flag). Distinct from `MaxPhotoautotrophs`,
/// which caps the running photo population — this resource only
/// drives the *initial* cohort.
#[derive(Resource)]
pub struct StartPhotoautotrophs(pub usize);

impl Default for StartPhotoautotrophs {
    fn default() -> Self { Self(DEFAULT_START_PHOTOAUTOTROPHS) }
}

pub const DEFAULT_MAP_X:           f32        = 1000.0;
pub const DEFAULT_MAP_Z:           f32        = 1000.0;


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
/// Default `250` is a starting value; the field is intended to
/// be tuned at runtime as the player explores the parameter space.
/// `0` effectively disables herbivore reproduction.
#[derive(Resource)]
pub struct MaxHerbivores(pub usize);

impl Default for MaxHerbivores {
    fn default() -> Self { Self(DEFAULT_MAX_HERBIVORES) }
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


/// Cinematic mode: when `true`, all UI chrome (top mode bar, statistics
/// panel, both navigators, the dividers) is hidden and the 3D viewport
/// fills the whole window. Toggled with F1 while in
/// `WindowMode::Simulation`. Auto-cleared if the window mode ever
/// leaves Simulation. Default `false`.
#[derive(Resource, Default)]
pub struct CinematicMode(pub bool);


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


/// Friction coefficient applied to every limb-based body part's Avian
/// collider (combined with the terrain's default μ = 0.5 via
/// `CoefficientCombine::Average`, so a limb↔terrain contact resolves
/// to μ = 0.75 — firm grip when planted).
///
/// Locomotion needs GRIP, not slip: to "press a foot down and back to
/// drive the body forward" the planted foot must not slide. An earlier
/// pass set this to 0.05 (with a `Min` combine rule) to stop limbs
/// sticking — but that starved the organism of traction (feet slid,
/// body never propelled). Raised to 1.0 / `Average` for grip. This does
/// NOT prevent lifting: a foot lifted off the ground has zero normal
/// force and therefore zero friction regardless of μ, so the
/// lift→reposition→plant→press gait cycle works (swing is frictionless
/// for free). Consumed by `colony::spawn_organism` /
/// `spawn_loaded_organism`.
// LIMB (foot) friction: HIGH, combined with `CoefficientCombine::Max` against
// the terrain so a planted foot GRIPS (μ → 1.0) and gives the leg stroke an
// anchor to pull the body against. Paired with a LOW-friction belly
// (`BASE_FRICTION_COEFFICIENT`) this is the propulsion mechanism for emergent
// crawling: data (2026-06-03) showed the creatures lie belly-down
// (base_contact ≈ 0.91) and were friction-PINNED under the old uniform μ=0.75,
// so no learned limb motion could translate them. Gripping feet + a slippery
// belly let a leg stroke drag the body — the missing ground-reaction asymmetry.
pub const LIMB_FRICTION_COEFFICIENT: f32 = 1.0;

/// BASE-body (belly) friction: LOW, combined with `CoefficientCombine::Min`
/// (μ → 0.0). The grippy LIMB feet do the propulsion (a stroke pushes off the
/// ground); a low-friction belly lets that push slide the body forward instead
/// of pinning it. The per-part floor (`enforce_limb_floor`) keeps every part
/// at/above the terrain so this stays non-penetrating and natural-looking.
pub const BASE_FRICTION_COEFFICIENT: f32 = 0.0;



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





/// Runtime-adjustable upper bound on the live photoautotroph count.
///
/// Reproduction reads this resource each tick — when the live photo
/// count is at or above this cap, the reproduction system suppresses
/// further photo births. When the cap is lowered below the current
/// photo count, `apply_max_phototrophs_cull` (in
/// `statistics_panel.rs`) despawns a random subset of photos to
/// meet the new cap in one step.
///
/// Heterotrophs are NOT bounded by this resource — they have their
/// own `MaxHerbivores` cap and their own brain-pool sizing via
/// `OrganismPoolSize`.
#[derive(Resource)]
pub struct MaxPhotoautotrophs(pub usize);

impl Default for MaxPhotoautotrophs {
    fn default() -> Self { Self(DEFAULT_MAX_PHOTOAUTOTROPHS) }
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
pub const AUTOSAVE_INTERVAL_MINUTES: f32 = 10.0;


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
/// it CANNOT change at runtime. Heterotroph reproduction past this
/// limit will skip brain-slot assignment for the extras (they exist
/// as entities but the brain pool can't enrol them).
///
/// Set by `main.rs::run_simulation` from a conservative bound on the
/// heterotroph population — typically `MaxHerbivores` plus headroom.
/// Independent from the photo cap (`MaxPhotoautotrophs`) since
/// photos don't have brain slots.
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


// ════════════════════════════════════════════════════════════════════════════
// CENTRALISED TUNING CONSTANTS
//
// The constants below were relocated here from their originating modules so
// every "knob" lives in one file. Each original module re-exports its
// constant via `pub use` / `use crate::simulation_settings::X;` so all
// existing reference sites — including full-path `crate::module::X` accesses
// from other modules — keep resolving unchanged.
// ════════════════════════════════════════════════════════════════════════════


// ── Physics / Limb (avian_setup.rs, colony.rs) ──────────────────────────────

// (avian_setup.rs)
/// Avian XPBD substep count for the limb-based physics world. Higher
/// substeps keep the rigid `SphericalJoint`s convergent under high PD torque,
/// but substeps multiply the ENTIRE solver cost linearly and dominated the
/// frame time once dozens of limb organisms (each = several dynamic bodies +
/// joints + contacts) were alive. Lowered to 8 now that joint stability comes
/// from the small joint compliance (`LIMB_JOINT_COMPLIANCE`), proper mass
/// properties, and the damping splits rather than from brute-force
/// substepping — 8 is half the cost of 16 while still well above the default.
/// If joints visibly drift again under fast commands, nudge back up to 10–12
/// before reaching for 16.
pub const LIMB_SOLVER_SUBSTEPS: u32 = 8;

/// "Compliance" (inverse stiffness) on every limb `SphericalJoint`.
/// `0.0` means a perfectly rigid point-constraint: the XPBD solver fully
/// projects the anchors back together each substep, so the joint can't
/// drift apart under the high angular velocities the brain commands (and
/// that future "hit"/melee behaviour will rely on) — unlike a non-zero
/// give, which lets the limb separate when the reaction exceeds the
/// solver's holding capacity at `LIMB_SOLVER_SUBSTEPS = 8`. The old note
/// that `0.0` was "numerically fragile" predates the current base-damping
/// setup; trying rigid first since it keeps full torque AND full ω.
pub const LIMB_JOINT_COMPLIANCE: f32 = 0.0;

/// `max_torque` clamp (N·m) on each limb hinge's spring-damper
/// `AngularMotor`. Avian clamps the motor's per-substep impulse to
/// `max_torque · dt²`, so this bounds how hard the hinge can drive toward
/// its target angle — the motor's "muscle strength". High enough that a
/// planted limb commanded "into the ground" can press the body upward
/// (`max_torque ≳ body weight × lever`); raise if limbs feel weak, lower
/// if a stride looks too violent. (Superseded the external-PD gains
/// `KP_TORQUE` / `KD_TORQUE` / `MAX_JOINT_ANGLE`, which were removed when
/// the controller moved to the in-solver motor — see
/// `LIMB_MOTOR_FREQUENCY`.)
// EMERGENT-WALK regime (2026-06-03): the brain now drives each limb hinge's
// target angle DIRECTLY (no CPG generating the rhythm), so the motor must have
// enough authority to actually lift/push the body for a stride to net thrust —
// 8 N·m was deliberately gentle/cosmetic for the old CPG-plus-pursuit design
// and far too weak for active, learned locomotion. Raised to 25. The old
// "explosions are self-launches at high torque" failure is now contained
// structurally (not by starving the motor): the per-part velocity governor
// (`MAX_LIMB_LINEAR_SPEED`/`MAX_LIMB_ANGULAR_SPEED`) bounds any launch impulse,
// `SelfCollisionFilter` removes the constraint-conflict energy injection, and
// `LIMB_ANGULAR_DAMPING` bleeds runaway build-up — so torque can be spent on
// walking rather than capped to prevent flinging. Lower if data shows strides
// look violent / bodies still launch despite the governor.
// Scaled 25 → 6 with the 5× mass cut (2026-06-03): the spring-damper motor's
// torque to track a target / hold the body up scales with the limb's rotational
// inertia, so the lighter microscopic body needs proportionally less. Keeping
// 25 on the light body would over-actuate and fling it. 6 is enough to stand
// the body up and stroke the legs without launching it.
pub const MAX_LIMB_TORQUE: f32 = 10.0;

/// Hinge swing-angle limit on every limb `RevoluteJoint`
/// (`with_angle_limits(-LIMB_SWING_LIMIT, +LIMB_SWING_LIMIT)`). The limb is
/// a 1-DOF hinge (it CANNOT orbit/spin around the body — the prior 3-DOF
/// ball joint's fragile angle-limits could be blown through; a revolute
/// joint rigidly locks the two non-hinge axes via a bilateral constraint).
/// This ±80° caps the in-plane swing range so the limb can't fold absurdly
/// far through the body, while leaving a generous stride for locomotion.
pub const LIMB_SWING_LIMIT: f32 = 80.0 * std::f32::consts::PI / 180.0;

/// "Little twist" knob for limb hinges: the compliance (inverse stiffness,
/// N·m/rad) of the `RevoluteJoint` axis-ALIGNMENT constraint
/// (`with_align_compliance`). `0.0` = a perfectly rigid hinge with zero
/// twist; a non-zero value softens the two off-hinge DOF so the limb can
/// deviate from the pure hinge plane — a real shoulder/hip twist.
///
/// IMPORTANT — this knob only became safe to raise once the limb motor
/// moved into the solver: the old external PD applied torque on the
/// off-hinge axes too, so softening this let that torque deflect the limb
/// into the gimbal singularity (the "spastic explosion" bug at `5e-3`).
/// The in-solver `AngularMotor` drives ONLY the hinge axis, so there is no
/// longer any controller torque exciting the off-hinge DOF — the twist is
/// governed solely by gravity/contact against this spring (damped by
/// `LIMB_ANGULAR_DAMPING`) and stays bounded at any value. Raise for more
/// twist; lower toward `0.0` for a stiffer hinge.
///
/// Raised 5e-3 → 3e-2 (2026-06-03) per the goal "limbs should be able to twist
/// a little to make emergent walking easier": the primary swing stays the
/// motorised 1-DOF hinge, but this softer alignment lets the planted foot
/// rotate/conform a little against the ground rather than being locked to a
/// single plane, giving the learned gait an extra passive DOF to find purchase.
/// Still small — the hinge is the dominant DOF — so the limb cannot orbit.
pub const LIMB_HINGE_ALIGN_COMPLIANCE: f32 = 3e-2;

/// Spring-damper motor parameters for the limb `RevoluteJoint`s. The limb
/// is driven by Avian's **built-in angular motor** (`MotorModel::SpringDamper`,
/// solved inside the XPBD step) toward a target hinge angle the brain sets
/// each tick — NOT by an external `Forces::apply_torque` PD controller.
/// This is what finally killed the limb instability: the in-solver motor
/// acts on the hinge axis ONLY (1-DOF, structurally cannot push off-hinge),
/// uses implicit-Euler integration (unconditionally stable — no
/// explicit-`−KD·ω` blow-up), and reads the true hinge angle via `atan2`
/// (no Euler-XYZ decomposition → no gimbal singularity). All three of the
/// old failure modes (gimbal flip, explicit-damping divergence, off-hinge
/// drive against a soft constraint) are removed at once.
///
/// `frequency` (Hz) = how fast the hinge converges on its target; higher =
/// stiffer/snappier. `damping_ratio` 1.0 = critically damped (fastest
/// approach with no overshoot). `MAX_LIMB_TORQUE` is reused as the motor's
/// `max_torque` clamp.
///
/// Raised 1.0 → 4.0 (2026-06-03): the brain re-commands each joint's target
/// angle every brain tick (~6.7 Hz of virtual time) and the gait it must learn
/// cycles at ~`GAIT_FREQUENCY_HZ`. A 1 Hz motor settles too slowly to track a
/// moving setpoint — the leg would lag a full cycle behind the command and
/// never realise the intended pose. 4 Hz lets the hinge reach the commanded
/// angle well within a tick so the learned joint trajectory actually happens.
///
/// Raised 4 → 10 (2026-06-03): the motor's stiffness ∝ frequency², and the legs
/// must be STIFF enough that the planted feet HOLD the (light) body up off the
/// ground rather than folding under it — at freq 4 the body sank onto its belly
/// / drove the feet through the terrain. At freq 10 the leg holds a
/// weight-bearing stance, so the body stands on its feet (natural posture, feet
/// resting at the surface). The implicit-Euler motor is unconditionally stable
/// at any frequency, so this adds no instability.
pub const LIMB_MOTOR_FREQUENCY: f32 = 4.0;
pub const LIMB_MOTOR_DAMPING_RATIO: f32 = 1.0;

// (colony.rs)
/// Angular damping for the BASE body of a limb-based organism. High,
/// because the base has no PD actuator of its own — joint-constraint
/// reaction torques from the limbs would otherwise integrate unbounded
/// and spin the body up.
pub const BASE_ANGULAR_DAMPING: f32 = 3.0;

/// Angular damping for LIMB bodies. Much lower than the base so the
/// PD controller can produce dynamic swings — the policy can't lift
/// the legs off the ground if every torque it commands gets drained
/// to friction within a frame.
// Raised 1 → 5 (2026-06-02): the limb "separations"/"explosions" are an
// ESCALATING energy build-up on a minority of organisms (sep grows over
// time) while stable walkers run bounded cycles. Strong angular damping
// bleeds the runaway build-up (which is sustained high ω) without blocking
// the brief propulsive kicks that drive walking — decoupling the otherwise
// linked propulsion/separation. (Substeps couldn't decouple them.)
pub const LIMB_ANGULAR_DAMPING: f32 = 7.0;

/// Linear damping for every limb-based body part (base + limbs).
/// Light — enough to bleed drift between actuator pulses, not enough
/// to lock the organism in place.
pub const LIMB_LINEAR_DAMPING:  f32 = 0.6;

/// Material density used when deriving the mass of **every** body part of a
/// limb-based organism from its collider — the base body (index 0) AND every
/// appendage limb, identically (only their angular damping differs). (Sliding
/// organisms are kinematic, so density doesn't apply to them.) Density × volume
/// → mass; lower density → lower mass → lower normal force at ground contacts →
/// less friction resisting rotation, AND less body weight for the PD torques to
/// lift/support. Lowered to 0.04 (an 80 % cut from the previous 0.2, itself
/// down from the natural 1.0): with many cells per body part (large collider
/// volume) the heavier mass left organisms too weak to stand or move — the
/// fixed-magnitude PD torques couldn't overcome the body weight. Lighter mass
/// restores the torque-to-weight ratio that lets them push up and walk.
// MICROSCOPIC scale (2026-06-03): cut 0.04 → 0.008 (5× lighter). AEONS
// organisms are amoeba/paramecium/ant-scale — VERY lightweight relative to
// their volume. A light body lets gravity + Avian's per-contact resolution
// settle the creature NATURALLY onto its feet: the legs easily hold the light
// body up (standing posture, belly off the ground), the feet rest ON the
// surface (no penetration), and no part is left rigidly hoisted/dangling. The
// heavy 0.04 body collapsed belly-down and drove the feet through the terrain.
// `MAX_LIMB_TORQUE` is scaled down with this (the spring-damper motor's needed
// torque ∝ inertia), so the lighter body isn't flung.
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

/// Coefficient on the entropy bonus term.
///
/// **Cut 0.03 → 0.005 after the 5-min data dive**: the limb pool's entropy
/// was RISING over training (1.32 → 1.47) while `mean_return` FELL
/// (−0.24 → −3.07). That is the signature of an entropy bonus larger than
/// the (near-zero, noisy) advantage signal — PPO was maximising entropy
/// (diffusing the policy) instead of reward. With a denser reward (below)
/// the advantage now carries real signal; a small entropy term keeps
/// exploration without dominating it.
///
/// Cut 0.005 → 0.0 (2026-06-03): across the emergent-walk runs entropy was
/// still RISING (1.76 → 2.08) while the policy failed to lock onto any
/// locomotion — the bonus was re-diffusing the policy faster than the (sparse,
/// hard-won) movement advantage could concentrate it. Exploration is already
/// supplied by the fixed sampling σ (`LOG_STD_INIT`), so the entropy bonus is
/// redundant here and was actively preventing convergence onto a gait. Zeroing
/// it lets any discovered propulsive stroke actually be reinforced.
pub const ENTROPY_COEF: f32 = 0.0;

/// Exploration std (as `log σ`) for the diagonal-Gaussian limb policy.
/// The sampler uses `σ = exp(LOG_STD_INIT)` directly. `exp(-1.2) ≈ 0.30`
/// — moderate exploration on the gait-parameter outputs.
///
/// **Was `0.0` (σ = 1.0), which broke locomotion**: a unit-variance noise
/// on a `[-1, 1]` action spans the whole range, so every 150 ms tick each
/// joint target was essentially random and the policy mean barely mattered.
/// The brain now sets each hinge target angle directly, so the noise perturbs
/// the joint command itself; σ ≈ 0.3 keeps exploration meaningful while still
/// letting a coherent phase-locked gait take shape rather than thrashing.
/// Tuning lever: raise for more exploration if learning stalls, lower if the
/// gait is too jittery.
///
/// Back to −1.2 (σ ≈ 0.30, 2026-06-03): the actor now WARM-STARTS into a
/// rhythmic leg oscillation (see `BrainPoolLimb::new`), so exploration no
/// longer has to DISCOVER a coherent stroke from white noise — it only needs to
/// perturb the warm-started gait so PPO can shape it. A large σ (0.5) would
/// drown the warm-start's ±0.46 oscillation in noise; σ ≈ 0.30 explores around
/// the rhythm without erasing it.
pub const LOG_STD_INIT: f32 = -1.2;


// ── Limb locomotion: EMERGENT, brain-driven (avian_setup::drive_limb_motors) ──
//
// Locomotion is NO LONGER generated by a built-in CPG and is NOT assisted by a
// pursuit force. Each limb hinge's target angle is set DIRECTLY from the
// brain's per-joint output (`Organism::limb_targets[joint] · LIMB_SWING_LIMIT`),
// and the in-solver spring-damper motor tracks it. Walking must EMERGE from RL:
// the brain learns, per joint, a phase-locked angle trajectory that produces
// ground reaction → forward thrust, shaped by the reward (forward velocity,
// progress toward prey, stepping, anti-spin). The only "rhythm aid" is a phase
// signal handed to the brain as an OBSERVATION (sin/cos of a slow virtual-time
// clock) — the brain still decides every joint angle; the clock merely lets a
// feedforward policy phase-lock a sustained oscillation instead of having to
// invent a limit cycle from a reactive map. This is a phase-conditioned policy,
// the standard way to get genuinely learned (not scripted) legged gaits.

/// Number of limb hinge joints the brain can independently control / observe.
/// This is the action dimension (`limb_ppo::OUT`) and the per-joint observation
/// bound: brain output `k` drives the hinge of body-part index `k+1`. Chosen to
/// cover the multi-segment Bilateral morphologies in `species/` (e.g. Crawler:
/// base + 2 hips + 2 knees → 8 runtime limb parts after the mirror expansion),
/// so every joint — left/right hip AND knee — gets its own learned command and
/// an alternating gait can emerge. Parts beyond this wrap modulo (rare).
/// MUST equal `limb_ppo::OUT`.
pub const MAX_LIMB_JOINTS: usize = 8;

/// Frequency (Hz of virtual time) of the phase-clock OBSERVATION fed to the
/// limb brain. NOT a motor command — the brain reads `sin/cos(2π·f·t)` and
/// learns how to map that phase onto each joint's target angle. Sets the
/// natural cadence the learned gait tends to lock onto: ~1 Hz is a plausible
/// stride rate for these small bodies. Invariant to `TimeSpeed`/frame rate
/// (derived from virtual elapsed time at brain-tick time).
pub const GAIT_FREQUENCY_HZ: f32 = 1.0;

/// Hard velocity governor on every limb body part (`MaxLinearSpeed` /
/// `MaxAngularSpeed` in Avian). The robust safety net against runaway
/// "explosions"/joint-separation: a destabilising limb otherwise reaches huge
/// velocity (data showed 70+ u/s linear, 90 rad/s angular while AIRBORNE) and
/// the momentum is what the rigid point-constraint can't arrest in 8 substeps
/// on the ultra-light bodies — so capping per-part speed keeps the joint
/// impulse bounded (joints hold), forbids the airborne fling, and keeps motion
/// moderate, while leaving a real walking stride (well under these) untouched.
pub const MAX_LIMB_LINEAR_SPEED:  f32 = 9.0;  // u/s (caps fly-off; walking is far below)
pub const MAX_LIMB_ANGULAR_SPEED: f32 = 14.0; // rad/s (caps spin-out; leg swings are below)


// ── Brain reward weights (limb_ppo.rs) ──────────────────────────────────────

pub const K_EAT:      f32 = 4.0;
pub const K_REPRO:    f32 = 5.0;
/// Reward per unit of forward (heading-projected) velocity. Signed:
/// directed travel pays, spin nets ~0, backward drift is penalised.
// Raised 0.1 → 0.3 (2026-06-03): once an organism translates at all, pay it
// more for translating in its FACING direction, so the movement the brain
// discovers is biased toward directed travel rather than undirected drift.
pub const K_FWD:      f32 = 0.3;
pub const K_UP:       f32 = 0.02;
/// Flat idle penalty `−K_IDLE·(1−motion_gate)`. **Set to 0 after the data
/// dive.** With the gait barely propelling most organisms, this term was
/// the DOMINANT reward (≈ −0.04/tick, near-identical for every still
/// organism regardless of its limb action) — a gradient-free negative
/// plateau that made `mean_return` uniformly negative and taught nothing.
/// It is replaced by `K_MOVE` (a positive, climbable movement gradient).
pub const K_IDLE:     f32 = 0.0;
/// Reward per world-unit of XZ distance closed to the nearest
/// photoautotroph since the last brain tick. Gives a dense
/// directional signal: any motion that nets closer-to-food is paid
/// even before the brain has discovered locomotion. Mirrors the
/// `K_PROGRESS` term in the sliding pools.
pub const K_PROGRESS: f32 = 0.5;
/// Reward per unit of improvement in heading-alignment toward the
/// nearest photoautotroph, tick-over-tick. `alignment ∈ [-1, 1]` is
/// `dot(body_heading_xz, prey_dir_xz)`; the reward is the RECTIFIED
/// delta `max(0, alignment_now − alignment_prev)`, so only turning
/// TOWARD the target is paid (turning away costs nothing — same
/// rectified philosophy as `K_PROGRESS`). This is the limb pool's
/// learned analogue of the sliding pool's hard-coded "rotate to face
/// travel direction": the brain must discover which limb motions
/// rotate the base toward prey, and gets dense credit for doing so
/// even before net translation begins. Rewarding the ACT of turning
/// (not static facing) avoids the alive-bonus trap.
pub const K_HEADING: f32 = 0.3;
/// Reward per limb-contact TRANSITION (a foot lifting off or planting
/// down) since the last tick, summed over all limb contact flags. A
/// walking gait cycles feet on and off the ground; rewarding the
/// transition gives a dense gradient toward *stepping* rather than
/// either static contact (foot dragging) or static lift (foot waving
/// in the air). Only the LIMB contact flags are counted — the
/// BASE contact flag is excluded (the base touching the ground is
/// the body dragging, which we don't want to reward). Kept modest so
/// it can't be farmed by high-frequency contact jitter — the speed /
/// progress terms still dominate genuine locomotion. (A `feet
/// air-time` shaping, as in legged_gym, is the natural refinement if
/// jitter-farming shows up.)
// DISABLED 0.1 → 0.0 (2026-06-03, data-driven): the first emergent-walk run
// showed PPO mean_return climbing 0.65 → 4.9 over 18 updates with ZERO body
// translation and every limb pinned at the ±swing-limit — the policy was
// FARMING this term by flailing its feet on/off the ground (each contact
// transition pays K_STEP) without ever locomoting. That is exactly the
// "jitter-farming" failure this term's own doc warned about. Zeroed so the
// only climbable rewards are the unfarmable movement terms (K_MOVE/K_FWD,
// which are 0 at rest and require the body to actually translate). A proper
// feet-air-time shaping could reintroduce stepping credit later without the
// in-place-jitter exploit.
pub const K_STEP: f32 = 0.0;

/// Speed (world-units/sec) at which the uprightness reward fully
/// activates and the idle penalty bottoms out. Below this the brain
/// is treated as "not really moving" and the alive-bonus is dimmed
/// proportionally.
pub const IDLE_THRESH: f32 = 0.5;

/// Reward per (capped) world-unit/s of RAW body speed — the dense,
/// climbable "move at all" gradient that bootstraps exploration of
/// locomotion. The data showed 76/80 organisms stuck at ~0.1 u/s with a
/// flat (gradient-free) reward, so PPO never found that moving the limbs
/// can move the body. Unlike `K_FWD` (forward-projected, ≈0 until the
/// body already travels in its facing direction), this pays ANY motion,
/// giving a slope out of the still basin. `K_FWD` + `K_PROGRESS` +
/// `K_HEADING` then bend that motion toward prey; `K_SPIN` keeps it from
/// degenerating into spin/flail. The term is GATED on uprightness in
/// `limb_ppo.rs` so a tumbling/ballistic body earns ~nothing — controlled
/// upright translation is the only way to collect it.
///
/// Raised 0.4 → 0.9 (2026-06-03): with the farmable `K_STEP` removed, this is
/// now the primary bootstrap gradient out of "flail in place" — it must be the
/// dominant climbable reward so the policy is pulled toward genuinely moving
/// the body (which only happens when learned joint motion nets ground thrust)
/// rather than sitting still. Still capped (`SPEED_REWARD_CAP`) and
/// uprightness-gated, so it cannot be won by spinning or ballistic flight.
pub const K_MOVE: f32 = 0.9;

/// Cap (world-units/s) on the speed that `K_MOVE` rewards. Beyond this,
/// extra speed earns nothing — so the ballistic "fling the body at 40 u/s"
/// regime is not preferred over controlled locomotion at a sane pace.
pub const SPEED_REWARD_CAP: f32 = 4.0;

/// Penalty per rad/s of base ANGULAR speed. The only organisms that moved
/// in the data were a few that destabilised into high-speed spin/flight
/// (angular speed up to ~16 rad/s). Penalising base spin steers the
/// movement gradient toward translation rather than tumbling, without
/// touching the (legitimate) limb-joint angular velocities.
pub const K_SPIN: f32 = 0.03;


// ── Energy (energy.rs) ──────────────────────────────────────────────────────

pub const ENERGY_TICK_INTERVAL: f32 = 0.5;
pub const MAX_ENERGY_PER_CELL: f32 = 10.0;

/// Per-tick energy a fully-surrounded (18 RD neighbours) photo cell
/// produces. Read by `physiology.rs::PhotosyntheticCell::new` to derive
/// the per-cell `energy_production` cache; the photosynthesis tick itself
/// runs in `physiology.rs`, not here.
pub const PHOTO_PRODUCTION_PER_CELL:  f32 = 4.0;
pub const NON_PHOTO_CONSUMPTION_PER_CELL: f32 = 0.01;

// Movement-cost coefficients tuned so a max-speed (20) sprint is heavily
// punitive on heavy organisms but doesn't immediately kill them.
pub const K_GROUND_FRICTION: f32 = 0.003;
pub const K_FLUID_DRAG:      f32 = 0.03;

/// Energy cost per metre of elevation gained — the gravitational-PE
/// analogue. Charged on every climb step accumulated since the last energy
/// tick and reset afterwards. Krishi is filtered out of the energy system
/// entirely, so its accumulated debt is never drained (never spent).
pub const ELEVATION_ENERGY_PER_UNIT: f32 = 0.5;

pub const DOPAMINE_DEPLETION_INTERVAL: f32 = 1.0;


// ── Physiology (physiology.rs) ──────────────────────────────────────────────

/// How often the physiology tick runs. 0.5 s matches the energy tick so
/// per-cell and per-organism updates stay roughly in phase, simplifying
/// reasoning about which lags which.
pub const PHYSIOLOGY_TICK_INTERVAL: f32 = 0.5;

/// Hard ceiling on per-cell energy. Cells never exceed this regardless of
/// how much they would otherwise gain in one tick — keeps the value
/// comparable across cell types and prevents overflow into pathological
/// regimes that future rules would have to special-case.
pub const MAX_CELL_ENERGY: f32 = 1.0;


// ── Photosynthesis (photosynthesis.rs) ──────────────────────────────────────

/// Direction *toward* the sun, as a unit vector.
///
/// Mirrors the directional light orientation in `main.rs`:
/// `Quat::from_euler(EulerRot::XYZ, -π/4, π/4, 0)` applied to Bevy's default
/// directional light forward (`-Z`) yields a light pointing roughly
/// `(-0.5, -√2/2, -0.5)`. The opposite of that — the direction *toward* the
/// light source — is the unit vector below.
pub const SUN_DIRECTION: Vec3 = Vec3::new(0.5, std::f32::consts::FRAC_1_SQRT_2, 0.5);

pub const SHADOW_CHECK_INTERVAL: Duration = Duration::from_secs(10);

/// Step length of the shadow raymarch in world units. Chosen to match the
/// heightmap cell size (1.0) so each step samples a fresh terrain cell.
pub const RAY_STEP_SIZE: f32 = 1.0;

/// Maximum number of steps before declaring the ray escaped to the sky.
/// With sun y-component √2/2 ≈ 0.707 and step size 1.0, 300 steps lift
/// the ray ~210 units above its origin — comfortably above any plausible
/// terrain peak in normalised worlds.
pub const MAX_RAY_STEPS: usize = 300;


// ── Reproduction (reproduction.rs) ──────────────────────────────────────────

pub const REPRODUCTION_CHECK_INTERVAL: f32 = 2.0;

/// Energy split between parent and offspring at reproduction (50/50).
pub const OFFSPRING_ENERGY_FRACTION: f32 = 0.5;

/// Threshold (fraction of `max_energy`) above which an organism becomes a
/// reproduction candidate.
pub const REPRODUCTION_ENERGY_THRESHOLD: f32 = 0.8;

pub const HETEROTROPH_REPRODUCTION_CAP:    u8 = 2;
pub const PHOTOAUTOTROPH_REPRODUCTION_CAP: u8 = 2;


// ── Growth (volumetric_growth/mod.rs, continuous_growth.rs) ─────────────────

/// Growth cap for variable-form (plant-like photoautotroph) organisms:
/// `continuous_growth` stops adding cells once a body reaches this many.
/// Heterotrophs don't use this — they're bounded by their body-part
/// count instead. Lowered 60 → 30 to keep photoautotrophs more compact.
pub const MAX_CELLS: usize = 30;

/// Effective growth cadence per organism. Each variable-form organism
/// receives one growth tick every `CONTINUOUS_GROWTH_INTERVAL` seconds.
/// 1.0 s gives a noticeable "growing" silhouette over ~30 seconds for
/// a fresh seed reaching the 30-cell cap.
pub const CONTINUOUS_GROWTH_INTERVAL: f32 = 1.0;

/// Number of phase slices the per-second growth workload is sliced into.
/// At 30, the system fires every `1/30` s ≈ 33 ms and each tick
/// processes only the organisms whose entity-index modulo 30 matches
/// the rotating phase counter — roughly 1/30th of the variable-form
/// population per tick. Total work per second is unchanged; the
/// per-tick allocator + Bevy command-buffer spike that was visible
/// every second goes away. Aligned with the 30 Hz brain tick so both
/// throttled subsystems share the same timing rhythm.
pub const GROWTH_PHASE_PERIOD: u32 = 30;

/// Wall-clock interval between phase steps. `CONTINUOUS_GROWTH_INTERVAL
/// / GROWTH_PHASE_PERIOD` so the per-organism cadence is preserved.
pub const GROWTH_PHASE_STEP_SECS: f32 =
    CONTINUOUS_GROWTH_INTERVAL / GROWTH_PHASE_PERIOD as f32;


// ── Movement (movement.rs) ──────────────────────────────────────────────────

pub const MIN_DIRECTION_INTERVAL: f32 = 1.0;
pub const MAX_DIRECTION_INTERVAL: f32 = 10.0;

pub const GRAVITY:          f32 = 9.8;
pub const MAX_CLIMB_HEIGHT: f32 = 4.0;

/// Global kill-floor. Any organism whose true world position falls below
/// this Y is despawned. Organisms that slip off the map edge (or through
/// a mesh gap) otherwise fall forever under gravity, burning brain/physics
/// cycles on an entity that can never recover. Set well below the lowest
/// plausible terrain so a legitimately low-lying organism is never culled.
pub const ORGANISM_DESPAWN_Y: f32 = -500.0;


// ── Collision (organism_collision.rs) ───────────────────────────────────────

/// Broad phase: organism root positions must be closer than this for any
/// further checks to run. Set generously — it's only a cheap distance test.
pub const ORGANISM_BROAD_RADIUS: f32 = 10.0;

/// Tick interval — running the full pipeline every frame is wasteful at
/// 1100-organism scale, and contacts emerge cleanly at 10 Hz.
pub const COLLISION_TICK: f32 = 0.1;

/// Maximum positional separation applied to any single organism per
/// collision tick (XZ plane, world units). Caps the integrated push
/// so deeply-overlapping organisms — e.g. a 30-cell vs 30-cell pair
/// can generate up to 900 narrow-phase contacts in one tick — don't
/// snap apart by an absurd amount. At 10 Hz this allows a maximum
/// separation speed of 5 world units / second, fast enough to
/// resolve any plausible penetration within ~1 s yet slow enough
/// that the eye reads it as a firm push rather than a teleport.
pub const MAX_SEPARATION_PER_TICK: f32 = 0.5;


// ── Brain tick intervals (behaviour.rs) ─────────────────────────────────────

// `PHOTO_BRAIN_TICK_INTERVAL` was retired with the L1-photo brain;
// kept here only as a documentation breadcrumb until the photo
// pool is removed entirely:
#[allow(dead_code)]
pub const PHOTO_BRAIN_TICK_INTERVAL:  Duration = Duration::from_millis(33);

/// Heterotroph brain tick rate (≈ 6.7 Hz).
///
/// Slower than the photo brain because the heterotroph's reward
/// signal is sparse: photos gain energy continuously from sunlight
/// (per-tick signal), but heterotrophs lose a tiny amount per
/// energy-plugin tick (0.5 s) and only receive a positive jump on
/// the rare predation event. At 30 Hz the σ-noise on actions
/// dominated displacement and rewards averaged to ~0 per tick — the
/// brain saw mostly noise. Slowing to ~150 ms gives:
///   * less visible direction jitter (one fresh action sample per
///     ~150 ms instead of ~33 ms),
///   * larger per-tick energy deltas (about 1/3 of an energy-plugin
///     tick fits inside one brain tick), and
///   * the reward shaper in `intelligence_level_1_hetero.rs` —
///     progress + facing — gets a meaningful per-tick state delta
///     to base its signal on.
pub const HETERO_BRAIN_TICK_INTERVAL: Duration = Duration::from_millis(150);


// ── World model (world_model.rs) ────────────────────────────────────────────

/// Radius (world units) within which neighbour organisms are considered
/// part of the heterotroph's world model.
pub const WORLD_MODEL_RADIUS: f32 = 60.0;

/// Velocity normalisation factor. Roughly the expected top speed in
/// world-units / second — matches the active hetero pools'
/// `MAX_SPEED`. A neighbour cruising at MAX_SPEED registers as `±1`
/// on the corresponding velocity dim.
pub const VELOCITY_NORM_SCALE: f32 = 20.0;


// ── Sensory (sensory.rs) ────────────────────────────────────────────────────

/// Radius (world units) within which the sensory algorithm looks
/// for a target photo. Anything beyond this is treated as "no
/// target", and `Organism::target_distance` saturates at this
/// value so the input observation stays bounded.
pub const SENSORY_RADIUS: f32 = 50.0;


// ── Predation (predation.rs) ────────────────────────────────────────────────

/// Fraction of the prey body part's energy share that becomes predator
/// energy. The "lost" 20% models metabolic inefficiency in digestion.
pub const ENERGY_TRANSFER_RATE: f32 = 0.8;


// ── World (world_geometry.rs) ───────────────────────────────────────────────

/// Edge safety zone (world units). No organism may move closer to
/// any of the four XZ map borders than this distance, and no spawn
/// position is ever generated inside the band — together those two
/// rules keep organisms strictly inside `[MARGIN, MapSize - MARGIN]²`
/// on the XZ plane. The clamp is enforced by
/// `movement::apply_world_bounds`; the spawn rule by every
/// `rng.random_range(..)` call that produces an XZ coordinate
/// (initial cohort, reproduction, auto-spawn).
pub const WORLD_SAFETY_MARGIN: f32 = 15.0;


// ── Water (water.rs) ────────────────────────────────────────────────────────

pub const BUOYANCY_STRENGTH:    f32 = 12.0;
pub const TRUE_WATER_DRAG_COEF: f32 = 0.05;


// ── Krishi (krishi.rs) ──────────────────────────────────────────────────────

/// Pixels-of-spawn-altitude above the heightmap floor, mirroring the
/// initial heightmap-clearance the colony uses for the procedural
/// organisms.
pub const KRISHI_SPAWN_ALTITUDE: f32 = 1.0;

/// Uniform size multiplier applied to BOTH the visual (the glb SceneRoot
/// child Transform) AND the collision footprint (the body part's cell
/// layout in `make_krishi_body`). Visual and collision stay locked
/// together so a Krishi that *looks* a certain size also *touches* prey
/// at that size.
///
/// Note on energy: Krishi consumption scales linearly with cell count,
/// so a scaled Krishi spends ~7x more energy per tick than a 1-cell
/// heterotroph. Starvation TIME is unchanged (the 0.5 starting energy
/// fraction also scales with cell count), but the predator must eat
/// ~7x more frequently to stay alive. Tune `KRISHI_SCALE` together with
/// the energy constants in `energy.rs` if you change ecological pressure.
pub const KRISHI_SCALE: f32 = 6.0;


// ── Spawn cohort (colony.rs) ────────────────────────────────────────────────

/// Initial Krishi cohort size. `pub` so `krishi.rs` reads it directly —
/// keeps every "how many of X spawn at startup" knob in one place.
pub const INITIAL_KRISHI: u32 = 1;

/// Fresh-start cohort (spawned when no `.colony` file is loaded):
/// authored `.species` files and how many of each to seed. Paths are
/// relative to the working directory (repo root), matching the
/// `species/` autoload convention used by the colony editor.
pub const SPAWN_CRAWLER_PATH:  &str  = "species/Crawler.species";
pub const SPAWN_CRAWLER_COUNT: usize = 40;
pub const SPAWN_STRIDER_PATH:  &str  = "species/Strider.species";
pub const SPAWN_STRIDER_COUNT: usize = 40;
pub const SPAWN_ALGAE_PATH:    &str  = "species/sessile_algae.species";
pub const SPAWN_ALGAE_COUNT:   usize = 800;
