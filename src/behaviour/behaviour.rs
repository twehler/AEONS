// Behaviour plugin — wires every brain pool, the shared world model,
// and the photosynthesis sunlight-check into the schedule.
//
// One plugin instead of four because every brain runs on the same
// 30 Hz clock and shares the same shared inputs (heterotrophs read
// the world-model resource; photoautotrophs read the sunlight flag
// `Organism::in_sunlight` written by `PhotosynthesisPlugin`).
// Bundling them here keeps the wiring near the systems' execution
// order and lets us guarantee the world-model rebuild completes
// before any heterotroph apply system reads it.
//
// Schedule layout (per tick):
//
//   PreUpdate  ── photosynthesis sunlight ray-march (10 Hz, gated
//                 inside PhotosynthesisPlugin).
//                 ↓ assign / free systems for all four pools, chained
//                 so all reservations land before any apply tick reads
//                 a slot. Why PreUpdate: both `predation_system` and
//                 `manage_energy` despawn organisms in Update; running
//                 assign/free here keeps their inserts off dead
//                 entities (cf. issue #10166 — try_insert is the
//                 belt-and-braces).
//
//   Update     ── (gated by BRAIN_TICK_INTERVAL timer)
//                 1. rebuild_world_model_grid
//                    — clears + repopulates `WorldModelGrid` with
//                    every photo + hetero position. Must run before
//                    any hetero apply system this tick.
//                 2. apply_intelligence_level_1_photo
//                 3. apply_intelligence_level_1_hetero
//                 4. apply_intelligence_level_2
//                 5. apply_intelligence_level_3
//                    — all four chained. They each `NonSendMut`
//                    their own pool resource so they'd serialise on
//                    the main thread anyway (CUDA state isn't
//                    Send); the `.chain()` just makes the order
//                    explicit AND ensures the world-model rebuild
//                    completes before any reader. The photo pool
//                    doesn't need the world model but joining the
//                    chain costs nothing — the overall schedule is
//                    main-thread-bound on these systems either way.

use bevy::prelude::*;
use bevy::time::common_conditions::on_timer;
use std::time::Duration;

use crate::intelligence_level_1_photo::{
    BrainPoolL1Photo, assign_brains_l1_photo, free_brains_l1_photo, apply_intelligence_level_1_photo,
};
use crate::intelligence_level_1_hetero::{
    BrainPoolL1Hetero, assign_brains_l1_hetero, free_brains_l1_hetero, apply_intelligence_level_1_hetero,
};
// Levels 2 and 3 are placeholder markers only — no pool, no systems
// (see `intelligence_level_2.rs` / `intelligence_level_3.rs`). Initial
// spawn rolls that would have produced L2/L3 are routed back to L1
// inside `IntelligenceLevel::for_initial_spawn`.
use crate::photosynthesis::PhotosynthesisPlugin;
use crate::world_model::{WorldModelGrid, rebuild_world_model_grid};


/// Photoautotroph brain tick rate (30 Hz).
///
/// Each tick fires one batched forward + REINFORCE step per pool,
/// evaluating every organism's private MLP in parallel on the GPU.
/// Between ticks, organisms continue to consume their last commanded
/// `movement_*` per frame in `apply_movement`, so the simulation
/// stays smooth.
///
/// 33 ms keeps the GPU pipeline full between ticks (vs the 200 Hz
/// earlier rate which forced 200 host→device→GPU→device→host
/// round-trips per second per pool and produced the "low CPU + low
/// GPU + low FPS" symptom).
///
/// `on_timer` ticks with `Time<Virtual>::delta()`, so brains
/// naturally pause when the simulation is paused.
const PHOTO_BRAIN_TICK_INTERVAL:  Duration = Duration::from_millis(33);

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
const HETERO_BRAIN_TICK_INTERVAL: Duration = Duration::from_millis(150);


pub struct BehaviourPlugin;

impl Plugin for BehaviourPlugin {
    fn build(&self, app: &mut App) {
        // Photoautotroph sunlight check (10 Hz, PreUpdate). Updates
        // `Organism::in_sunlight` which the L1 photo brain reads as
        // input.
        app.add_plugins(PhotosynthesisPlugin);

        // World model resource — populated each brain tick by
        // `rebuild_world_model_grid`, queried by every hetero apply
        // system. Initialised empty here so the resource exists from
        // tick 1.
        app.init_resource::<WorldModelGrid>();

        // ── Pool resources (non-send: CUDA state isn't Send). ───────
        app.init_non_send_resource::<BrainPoolL1Photo>();
        app.init_non_send_resource::<BrainPoolL1Hetero>();

        // ── PreUpdate: assign / free for every pool. ────────────────
        // Two pools, two pairs. Each pool's `assign` runs before its
        // `free` within a tick to avoid a recycled-slot ordering
        // race; the two pairs are unconstrained relative to each
        // other.
        app.add_systems(PreUpdate, (assign_brains_l1_photo,  free_brains_l1_photo) .chain());
        app.add_systems(PreUpdate, (assign_brains_l1_hetero, free_brains_l1_hetero).chain());

        // ── Update: split into a fast photo chain and a slow hetero
        // chain. They share no read/write set with each other (each
        // pool has its own `NonSendMut` resource and the photo brain
        // doesn't read the world-model grid), so the two timers fire
        // independently and may both run on the same frame.
        //
        // Fast chain: photo apply only, 30 Hz.
        app.add_systems(
            Update,
            apply_intelligence_level_1_photo
                .run_if(on_timer(PHOTO_BRAIN_TICK_INTERVAL)),
        );

        // Slow chain: world-model rebuild + the L1 hetero apply,
        // ~6.7 Hz. The chain ordering is load-bearing — rebuild
        // must run before the hetero apply so it reads the same
        // up-to-date grid produced this tick.
        app.add_systems(
            Update,
            (
                rebuild_world_model_grid,
                apply_intelligence_level_1_hetero,
            )
                .chain()
                .run_if(on_timer(HETERO_BRAIN_TICK_INTERVAL)),
        );
    }
}
