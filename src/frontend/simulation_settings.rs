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


/// Global switch for the heterotroph-movement RL debugging environment.
///
/// Following the user-specified polarity:
///   * `false` → **debug mode active**. Heterotrophs never despawn even when
///     their energy hits zero, and they never reproduce. Hunger (energy
///     drain) still happens. Companion call sites also disable the Krishi
///     plugin and reduce the initial heterotroph cohort to a single
///     organism so the RL training environment is isolated.
///   * `true`  → **normal simulation**. Starvation despawn, reproduction,
///     Krishi plugin, and the full initial heterotroph cohort are all
///     active.
///
/// Read at compile time by `energy.rs` (despawn skip), `reproduction.rs`
/// (heterotroph reproduction skip), `main.rs` (Krishi plugin gate), and
/// `colony.rs` (initial heterotroph count).
pub const HETEROTROPH_MOVEMENT_AI_DEBUGGING: bool = false;


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
