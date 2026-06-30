// Behaviour plugin — wires every brain pool, the shared world model,
// and the photosynthesis sunlight-check into the schedule.
//
// One plugin because all brains share a clock and inputs (heteros read
// the world-model resource; photos read `Organism::in_sunlight`).
//
// Schedule:
//   PreUpdate  — photosynthesis sunlight ray-march, then assign/free
//                for all pools (chained). PreUpdate because Update
//                despawns organisms (predation/energy); assign/free
//                here keeps inserts off dead entities (Bevy #10166).
//   FixedUpdate (gated by HETERO_BRAIN_TICK_INTERVAL) — world-model
//                rebuild → sensory → each pool's apply, chained so the
//                grid is fresh before any reader. Pools serialise on
//                the main thread anyway (NonSendMut CUDA state); chain
//                just makes the order explicit.

use bevy::prelude::*;
use bevy::time::common_conditions::on_timer;
use std::time::Instant;

use crate::intelligence_level_herbivore_1_sliding::{
    BrainPoolHerbivore1, assign_brains_herbivore_1, free_brains_herbivore_1,
    apply_intelligence_level_herbivore_1,
};
use crate::intelligence_level_2_sliding::{
    BrainPoolL2, assign_brains_l2, free_brains_l2, apply_intelligence_level_2,
};
use crate::intelligence_level_3_sliding::{
    BrainPoolL3, assign_brains_l3, free_brains_l3, apply_intelligence_level_3,
};
// Limb-based PPO pools. Mirror the three sliding pools but gated on
// `!Organism::movement_mode.is_sliding()`. Each is a separate non-send
// resource with its own GPU-batched actor + critic.
use crate::intelligence_level_herbivore_1_limb::{
    BrainPoolHerbivore1Limb, assign_brains_herbivore_1_limb,
    free_brains_herbivore_1_limb, apply_intelligence_level_herbivore_1_limb,
    share_limb_policies_herbivore_1,
};
use crate::intelligence_level_2_limb::{
    BrainPoolL2Limb, assign_brains_l2_limb, free_brains_l2_limb,
    apply_intelligence_level_2_limb,
};
use crate::intelligence_level_3_limb::{
    BrainPoolL3Limb, assign_brains_l3_limb, free_brains_l3_limb,
    apply_intelligence_level_3_limb,
};
// Swimming PPO pool (Level1 only for now) — gated on
// `Organism::movement_mode.is_swimming()`; the limb pools exclude swimmers,
// so the populations stay disjoint. Ball-joint 3-axis control + the 3D
// target/rotation oracles (swimming_movement/swim_ppo.rs).
use crate::intelligence_level_1_swimming::{
    BrainPoolSwim1, assign_brains_swim_1, free_brains_swim_1,
    apply_intelligence_level_swim_1,
};
use crate::intelligence_level_simple_aquatic::{
    BrainPoolSimpleAquatic, assign_brains_simple_aquatic, free_brains_simple_aquatic,
    apply_intelligence_level_simple_aquatic,
};
use crate::photosynthesis::PhotosynthesisPlugin;
use crate::world_model::{WorldModelGrid, rebuild_world_model_grid};


/// Photoautotroph brain tick rate (30 Hz). One batched forward +
/// REINFORCE step per pool per tick; between ticks organisms keep
/// consuming their last `movement_*` in `apply_movement`. 33 ms keeps
/// the GPU pipeline full (vs frequent host↔device round-trips). Ticks
/// on virtual time, so brains pause with the sim.
#[allow(unused_imports)]
use crate::simulation_settings::PHOTO_BRAIN_TICK_INTERVAL;

use crate::simulation_settings::HETERO_BRAIN_TICK_INTERVAL;


pub struct BehaviourPlugin;

impl Plugin for BehaviourPlugin {
    fn build(&self, app: &mut App) {
        // Photoautotroph sunlight check (10 Hz, PreUpdate); writes
        // `Organism::in_sunlight`.
        app.add_plugins(PhotosynthesisPlugin);

        // World model resource — rebuilt each brain tick. Init empty so
        // it exists from tick 1.
        app.init_resource::<WorldModelGrid>();

        // Brain-tick wall-time instrumentation (Amdahl gate). Behaviour-neutral.
        app.init_resource::<BrainTickTimer>();

        // Pool resources (non-send: CUDA state isn't Send).
        // Sliding pools (REINFORCE): herbivore_1, L2, L3.
        app.init_non_send_resource::<BrainPoolHerbivore1>();
        app.init_non_send_resource::<BrainPoolL2>();
        app.init_non_send_resource::<BrainPoolL3>();
        // Limb pools (PPO), each at full `OrganismPoolSize` (no shared
        // weights, no batch splitting).
        app.init_non_send_resource::<BrainPoolHerbivore1Limb>();
        app.init_non_send_resource::<BrainPoolL2Limb>();
        app.init_non_send_resource::<BrainPoolL3Limb>();
        // Swimming pool (PPO, ball-joint 3-axis control), Level1 only.
        app.init_non_send_resource::<BrainPoolSwim1>();
        // SimpleAquatic pool (REINFORCE, kinematic 3D mover), Level1 only.
        app.init_non_send_resource::<BrainPoolSimpleAquatic>();

        // PreUpdate: assign / free for every active pool.
        app.add_systems(PreUpdate, (assign_brains_herbivore_1, free_brains_herbivore_1).chain());
        app.add_systems(PreUpdate, (assign_brains_l2,          free_brains_l2)         .chain());
        app.add_systems(PreUpdate, (assign_brains_l3,          free_brains_l3)         .chain());
        // Limb pools — `!movement_mode.is_sliding()` filter keeps the two
        // populations disjoint.
        app.add_systems(PreUpdate, (assign_brains_herbivore_1_limb, free_brains_herbivore_1_limb).chain());
        // Social learning: periodically propagate the best walker's policy to
        // stragglers so the WHOLE population walks (self-gated by virtual time).
        app.add_systems(PreUpdate, share_limb_policies_herbivore_1);
        app.add_systems(PreUpdate, (assign_brains_l2_limb,          free_brains_l2_limb)         .chain());
        app.add_systems(PreUpdate, (assign_brains_l3_limb,          free_brains_l3_limb)         .chain());
        // Swimming pool — `is_swimming()` filter keeps it disjoint from both
        // the sliding and the limb populations.
        app.add_systems(PreUpdate, (assign_brains_swim_1,           free_brains_swim_1)          .chain());
        // SimpleAquatic pool — `is_simple_aquatic()` filter keeps it disjoint from
        // the sliding, limb, and swim populations.
        app.add_systems(PreUpdate, (assign_brains_simple_aquatic,   free_brains_simple_aquatic)  .chain());

        // FixedUpdate (NOT Update) so the brain cadence is driven by
        // virtual time, not frame rate: FixedUpdate runs as many times
        // per frame as needed to consume accumulated virtual time, so
        // the brain fires every HETERO_BRAIN_TICK_INTERVAL regardless of
        // TimeSpeed (fast-forward learning == real-time learning).
        // `on_timer` ticks on Time<Fixed>.
        app.add_systems(
            FixedUpdate,
            (
                // Amdahl gate: stamp the brain-chain start (first) / end (last).
                brain_tick_begin,
                rebuild_world_model_grid,
                // Sensory — after the grid rebuild (consumes it), before
                // the herbivore brain (reads target_distance as input + reward).
                crate::sensory::update_target_distance,
                apply_intelligence_level_herbivore_1,
                apply_intelligence_level_2,
                apply_intelligence_level_3,
                apply_intelligence_level_herbivore_1_limb,
                apply_intelligence_level_2_limb,
                apply_intelligence_level_3_limb,
                apply_intelligence_level_swim_1,
                apply_intelligence_level_simple_aquatic,
                brain_tick_end,
            )
                .chain()
                .run_if(on_timer(HETERO_BRAIN_TICK_INTERVAL)),
        );
    }
}


// ── Instrumentation: brain-tick wall-time (Amdahl gate) ─────────────────────────
//
// Measures the wall-clock cost of the WHOLE brain chain (world-model rebuild →
// sensory → every pool's apply) per brain tick. Because the per-tick forward
// readback (`into_data`) blocks on the GPU, this CPU wall-time correctly
// includes GPU compute + readback stalls — exactly the figure the optimisation
// plan's "Amdahl gate" needs. Behaviour-neutral (no logic touched).
//
// To get the subsystem's share of a frame: compare `avg`/`ema` against the
// frame time (≈ 1000 / FPS). Watch the number drop as the levers land — it is
// the single metric the whole optimisation is moving.
#[derive(Resource, Default)]
pub struct BrainTickTimer {
    start:        Option<Instant>,
    /// Most recent single-tick duration (ms).
    pub last_ms:  f32,
    /// Exponential moving average of tick duration (ms).
    pub ema_ms:   f32,
    accum_ms:     f32,
    accum_n:      u32,
}

/// First system in the gated brain chain: stamp the tick start.
fn brain_tick_begin(mut t: ResMut<BrainTickTimer>) {
    t.start = Some(Instant::now());
}

/// Last system in the gated brain chain: record elapsed, update the EMA, and
/// log a concise greppable line (`BRAINTICK …`) every 64 ticks so unattended
/// `--exit-after-secs` runs capture it on stdout.
fn brain_tick_end(mut t: ResMut<BrainTickTimer>) {
    let Some(start) = t.start.take() else { return };
    let ms = start.elapsed().as_secs_f32() * 1000.0;
    t.last_ms = ms;
    t.ema_ms  = if t.ema_ms == 0.0 { ms } else { 0.9 * t.ema_ms + 0.1 * ms };
    t.accum_ms += ms;
    t.accum_n  += 1;
    if t.accum_n >= 64 {
        let avg = t.accum_ms / t.accum_n as f32;
        info!(
            "BRAINTICK avg={:.3}ms ema={:.3}ms last={:.3}ms (over {} ticks)",
            avg, t.ema_ms, t.last_ms, t.accum_n,
        );
        t.accum_ms = 0.0;
        t.accum_n  = 0;
    }
}
