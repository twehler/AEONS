// Swimming Level-1 brain — PPO pool for Level1 heterotrophs with
// `Organism::movement_mode.is_swimming()`.
//
// Mirrors the limb herbivore_1 pool structurally but delegates everything to
// the shared `swim_ppo` engine: a small MLP (`IN → 32 → 24`) commanding three
// target angles per BALL joint, fed by the 3D target oracle (body-local
// bearing to the nearest phototroph) and the rotation oracle (body-local
// rotation vector that faces the front at the target). Swimming organisms are
// EXCLUDED from the limb pools (their assign systems skip `is_swimming()`),
// so the two populations are disjoint.

use bevy::prelude::*;

use crate::colony::{IntelligenceLevel, Organism, Heterotroph};
use crate::rapier_setup::SwimJointTargets;
use crate::swim_ppo::{BrainPoolSwim, SwimSlot, gather_swim_obs_inputs};
use crate::simulation_settings::OrganismPoolSize;


// ── Component marker ──────────────────────────────────────────────────────────

/// Slot index into `BrainPoolSwim1`. One per swimming Level1 heterotroph;
/// freed when the organism despawns.
#[derive(Component, Clone, Copy)]
pub struct BrainSlotSwim1(pub u32);

impl SwimSlot for BrainSlotSwim1 {
    fn slot(&self) -> u32 { self.0 }
}


// ── Resource ──────────────────────────────────────────────────────────────────

/// Per-organism PPO pool for swimming Level1 heterotrophs. `NonSend`
/// (burn-cuda tensors aren't `Send`). Wraps the shared `swim_ppo` engine.
pub struct BrainPoolSwim1(pub BrainPoolSwim);

impl FromWorld for BrainPoolSwim1 {
    fn from_world(world: &mut World) -> Self {
        let n = world.resource::<OrganismPoolSize>().0;
        let device = burn_cuda::CudaDevice::default();
        Self(BrainPoolSwim::new(n, device))
    }
}


// ── Lifecycle systems ─────────────────────────────────────────────────────────

/// Enrol newly-spawned swimming Level1 heterotrophs.
///
/// Notes:
///   * Any stale `BrainRestoreLimb` payload (a `.colony`/`.species` saved
///     LIMB-pool brain — swimmers used to enrol there) is DROPPED: its tensor
///     shapes belong to the limb architecture and can't seed this pool.
///     Swim-brain persistence is not implemented yet; loaded swimmers re-init.
///   * `BrainInheritance(parent)` is a NO-OP under the shared policy: every
///     swimmer already shares the one set of weights, so a newborn is born
///     competent automatically — no per-slot weight copy is needed (the
///     individual-learning variant in `src_individual_learning` does copy).
pub fn assign_brains_swim_1(
    mut pool:     NonSendMut<BrainPoolSwim1>,
    new:          Query<(Entity, &Organism, Option<&crate::limb_ppo::BrainRestoreLimb>, Option<&crate::rl_helpers::BrainInheritance>), (
        With<Heterotroph>,
        Without<BrainSlotSwim1>,
    )>,
    mut commands: Commands,
) {
    for (e, organism, restore, inheritance) in new.iter() {
        if !matches!(organism.intelligence_level, IntelligenceLevel::Level1) { continue; }
        if !organism.movement_mode.is_swimming() { continue; }
        let Some(s) = pool.0.enrol(e) else { continue };
        // A `BrainRestoreLimb` payload on a SWIMMER is a saved/exported SWIM net
        // (limb & swim PPO brains share the payload struct; movement mode routes
        // it here). Restore it into this organism's species net. Shape-guarded:
        // a genuine terrestrial-limb payload would fail the swim-dim check and
        // degrade to a fresh warm-start. Keyed by the organism's species_id,
        // which `.species` imports pin eagerly at spawn so the restore lands in
        // the right shared net.
        if let Some(r) = restore {
            let key = organism.species_id.unwrap_or(crate::swim_ppo::UNCLASSIFIED);
            pool.0.restore_species(key, r);
            commands.entity(e).try_remove::<crate::limb_ppo::BrainRestoreLimb>();
        }
        // SHARED policy → no per-slot weight inheritance; the slot is pure
        // bookkeeping. Just clear any inheritance marker so it isn't reprocessed.
        if inheritance.is_some() { commands.entity(e).try_remove::<crate::rl_helpers::BrainInheritance>(); }
        commands.entity(e).try_insert(BrainSlotSwim1(s));
    }
}

/// Release pool slots for organisms that lost `BrainSlotSwim1`.
pub fn free_brains_swim_1(
    mut pool:    NonSendMut<BrainPoolSwim1>,
    mut removed: RemovedComponents<BrainSlotSwim1>,
) {
    let slots: Vec<(Entity, u32)> = removed.read()
        .filter_map(|e| pool.0.map.get(&e).map(|&s| (e, s)))
        .collect();
    for (e, s) in slots {
        pool.0.release(e, s);
    }
}


// ── Apply ───────────────────────────────────────────────────────────────────

/// Per-tick brain step. Delegates to the shared `swim_ppo::apply_step`:
/// gather 3D physics inputs + the two oracles, batched actor forward, sample
/// actions, write them to `SwimJointTargets` (→ `drive_swim_motors`), record
/// rollout entries for the PPO update.
pub fn apply_intelligence_level_swim_1(
    mut pool: NonSendMut<BrainPoolSwim1>,
    organisms: Query<(Entity, &Organism, &mut SwimJointTargets, &BrainSlotSwim1)>,
    body_parts: Query<(
        &bevy::prelude::ChildOf,
        &crate::cell::BodyPartIndex,
        &bevy::prelude::GlobalTransform,
        &bevy_rapier3d::prelude::Velocity,
    )>,
    world_grid: Res<crate::world_model::WorldModelGrid>,
    virtual_time: Res<bevy::prelude::Time<bevy::prelude::Virtual>>,
) {
    let obs_inputs = gather_swim_obs_inputs(&body_parts, &world_grid);
    pool.0.apply_step(organisms, &obs_inputs, virtual_time.elapsed_secs());
}
