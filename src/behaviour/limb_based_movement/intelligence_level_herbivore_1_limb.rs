// Limb-based herbivore_1 brain — PPO pool for Level1 + Heterotroph +
// non-Carnivore organisms WHOSE `Organism::sliding_movement == false`.
//
// Mirrors `sliding_movement::intelligence_level_herbivore_1_sliding`
// structurally (component / resource / system trio), but delegates the
// network, rollout buffer, GAE and PPO update to the shared engine in
// `limb_ppo.rs`. This keeps the three per-level limb brains tiny — they
// only differ in their enrolment filter.
//
// Phase 3 deliverable: pool + assign / free systems + stub apply. The
// observation/action wiring to Avian physics and the actual PPO
// gradient code arrive in Phase 4.

use bevy::prelude::*;

use crate::colony::{IntelligenceLevel, Organism, Heterotroph, Carnivore};
use crate::limb_ppo::{BrainPoolLimb, BrainRestoreLimb, LimbSlot, gather_limb_obs_inputs};
use crate::simulation_settings::OrganismPoolSize;


// ── Component marker ──────────────────────────────────────────────────────────

/// Slot index into `BrainPoolHerbivore1Limb`. One per limb-based
/// Level1 non-carnivore heterotroph; freed when the organism despawns.
#[derive(Component, Clone, Copy)]
pub struct BrainSlotHerbivore1Limb(pub u32);

impl LimbSlot for BrainSlotHerbivore1Limb {
    fn slot(&self) -> u32 { self.0 }
}


// ── Resource ──────────────────────────────────────────────────────────────────

/// Per-organism PPO pool for limb-based Level1 herbivores. Held as a
/// `NonSend` resource because the burn-cuda backend's tensors aren't
/// `Send`. Wraps the shared engine in `limb_ppo`.
pub struct BrainPoolHerbivore1Limb(pub BrainPoolLimb);

impl FromWorld for BrainPoolHerbivore1Limb {
    fn from_world(world: &mut World) -> Self {
        let n = world.resource::<OrganismPoolSize>().0;
        let device = burn_cuda::CudaDevice::default();
        Self(BrainPoolLimb::new(n, device))
    }
}


// ── Lifecycle systems ─────────────────────────────────────────────────────────

/// Enrol newly-spawned limb-based Level1 non-carnivore heterotrophs.
/// Mirrors the gating logic of the sliding pool, with the
/// `!sliding_movement` extra filter so the two populations are disjoint.
pub fn assign_brains_herbivore_1_limb(
    mut pool:     NonSendMut<BrainPoolHerbivore1Limb>,
    new:          Query<(Entity, &Organism, Option<&BrainRestoreLimb>, Option<&crate::rl_helpers::BrainInheritance>), (
        With<Heterotroph>,
        Without<Carnivore>,
        Without<BrainSlotHerbivore1Limb>,
    )>,
    mut commands: Commands,
) {
    for (e, organism, restore, inheritance) in new.iter() {
        if !matches!(organism.intelligence_level, IntelligenceLevel::Level1) { continue; }
        if organism.sliding_movement { continue; }
        let Some(s) = pool.0.enrol(e) else { continue };
        // If the entity arrived with a saved-weights component (loaded
        // from a `.colony` file), write its weights into the freshly-
        // allocated row before we remove the marker.
        if let Some(r) = restore {
            pool.0.restore_slot(s, r);
            commands.entity(e).try_remove::<BrainRestoreLimb>();
        } else {
            // No saved weights → INHERIT a trained brain so the new organism
            // isn't born helpless (a fresh warm-start collapses until it learns
            // from scratch). Prefer the explicit parent (reproduction); else any
            // other occupied slot — join the already-trained population. `src`
            // is computed (and the immutable borrow released) before the
            // mutable `inherit_slot` call.
            let src = inheritance.and_then(|inh| pool.0.map.get(&inh.0).copied())
                .or_else(|| pool.0.map.iter().filter_map(|(ent, sl)| (*ent != e && *sl != s).then_some(*sl)).next());
            if let Some(src) = src { pool.0.inherit_slot(s, src); }
        }
        if inheritance.is_some() { commands.entity(e).try_remove::<crate::rl_helpers::BrainInheritance>(); }
        commands.entity(e).try_insert(BrainSlotHerbivore1Limb(s));
    }
}

/// Release pool slots for organisms that have lost their
/// `BrainSlotHerbivore1Limb` component (despawned or had the marker
/// stripped). Uses the entity-keyed slot map on the pool to find the
/// row index even after the component is gone.
pub fn free_brains_herbivore_1_limb(
    mut pool:    NonSendMut<BrainPoolHerbivore1Limb>,
    mut removed: RemovedComponents<BrainSlotHerbivore1Limb>,
) {
    let slots: Vec<(Entity, u32)> = removed.read()
        .filter_map(|e| pool.0.map.get(&e).map(|&s| (e, s)))
        .collect();
    for (e, s) in slots {
        pool.0.release(e, s);
    }
}


// ── Apply (stub) ──────────────────────────────────────────────────────────────

/// Per-tick brain step. Delegates to the shared `apply_step` engine,
/// which: builds observations, runs the batched actor forward,
/// samples actions, writes them to `Organism::limb_targets`, and
/// records `RolloutEntry`s for the per-slot PPO update. Reward and
/// PPO gradient are currently placeholders (see `apply_step` doc).
pub fn apply_intelligence_level_herbivore_1_limb(
    mut pool: NonSendMut<BrainPoolHerbivore1Limb>,
    organisms: Query<(Entity, &mut Organism, &BrainSlotHerbivore1Limb)>,
    body_parts: Query<(
        &bevy::prelude::ChildOf,
        &crate::cell::BodyPartIndex,
        &avian3d::prelude::Position,
        &avian3d::prelude::Rotation,
        &avian3d::prelude::AngularVelocity,
        &avian3d::prelude::LinearVelocity,
        Option<&crate::avian_setup::LimbContact>,
    )>,
    world_grid: Res<crate::world_model::WorldModelGrid>,
    virtual_time: Res<bevy::prelude::Time<bevy::prelude::Virtual>>,
) {
    let obs_inputs = gather_limb_obs_inputs(&body_parts, &world_grid);
    pool.0.apply_step(organisms, &obs_inputs, virtual_time.elapsed_secs());
}
