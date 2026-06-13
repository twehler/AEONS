// Limb-based herbivore_1 brain — PPO pool for Level1 + Heterotroph +
// non-Carnivore organisms with `!Organism::movement_mode.is_sliding()`.
//
// Mirrors the sliding herbivore_1 pool structurally but delegates the
// network, rollout buffer, GAE and PPO update to the shared `limb_ppo`
// engine, so the three per-level limb brains differ only in their
// enrolment filter.

use bevy::prelude::*;

use std::collections::HashMap;

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

/// Per-organism PPO pool for limb-based Level1 herbivores. `NonSend`
/// (burn-cuda tensors aren't `Send`). Wraps the shared `limb_ppo` engine.
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
/// `!movement_mode.is_sliding()` extra filter so the two populations are disjoint.
pub fn assign_brains_herbivore_1_limb(
    mut pool:     NonSendMut<BrainPoolHerbivore1Limb>,
    new:          Query<(Entity, &Organism, Option<&BrainRestoreLimb>, Option<&crate::rl_helpers::BrainInheritance>), (
        With<Heterotroph>,
        Without<Carnivore>,
        Without<BrainSlotHerbivore1Limb>,
    )>,
    reload_brains: Res<crate::simulation_settings::ReloadLimbBrains>,
    mut commands: Commands,
) {
    for (e, organism, restore, inheritance) in new.iter() {
        if !matches!(organism.intelligence_level, IntelligenceLevel::Level1) { continue; }
        if organism.movement_mode.is_sliding() { continue; }
        // Swimmers train in their own pool (intelligence_level_1_swimming).
        if organism.movement_mode.is_swimming() { continue; }
        let Some(s) = pool.0.enrol(e) else { continue };
        // STANDING task: by default ignore the saved Runner brains — they are
        // locomotion-trained and tanh-SATURATED (μ pinned at ±1 → zero actor
        // gradient → the policy can't move, the body freezes in one pose). Keep
        // the pool's fresh fan-in + neutral-init weights so PPO actually explores
        // and learns to stand. EXCEPTION: `--reload-limb-brains` (set when loading
        // a colony saved AFTER a standing run) restores the trained STANDING policy
        // through the normal restore path below — the durable-success artifact.
        if crate::simulation_settings::STANDING_TASK && !reload_brains.0 {
            if restore.is_some() { commands.entity(e).try_remove::<BrainRestoreLimb>(); }
            if inheritance.is_some() { commands.entity(e).try_remove::<crate::rl_helpers::BrainInheritance>(); }
            commands.entity(e).try_insert(BrainSlotHerbivore1Limb(s));
            continue;
        }
        // Saved weights (loaded `.colony`) → overwrite this organism's SPECIES
        // net (keyed by species_id, UNCLASSIFIED until first classified).
        if let Some(r) = restore {
            pool.0.restore_species(organism.species_id.unwrap_or(0), r);
            commands.entity(e).try_remove::<BrainRestoreLimb>();
        }
        // SHARED policy → no per-slot weight inheritance; a newborn of an
        // existing species already shares that species' trained net. Just clear
        // any inheritance marker so it isn't reprocessed.
        if inheritance.is_some() { commands.entity(e).try_remove::<crate::rl_helpers::BrainInheritance>(); }
        commands.entity(e).try_insert(BrainSlotHerbivore1Limb(s));
    }
}

/// Release pool slots for organisms that lost `BrainSlotHerbivore1Limb`.
/// Uses the pool's entity-keyed map to find the row after the component
/// is gone.
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


// ── Social learning (population policy sharing) ───────────────────────────────

/// Every `SHARE_INTERVAL` of virtual time, copy the BEST walker's brain (max net
/// base displacement over the window) into the worst non-walkers, so the WHOLE
/// population converges to a walking policy that only a fraction of organisms
/// discover independently (PPO stragglers stuck in still/hop local optima). Net
/// displacement of the base body part is the "is it actually travelling" signal —
/// a hopper/spinner racks up little. Recipients keep training, so they refine the
/// inherited policy rather than freezing. Only shares when the best is genuinely
/// walking, so a bad policy is never propagated.
pub fn share_limb_policies_herbivore_1(
    _pool:          NonSendMut<BrainPoolHerbivore1Limb>,
    _slots:         Query<(Entity, &BrainSlotHerbivore1Limb)>,
    _base_parts:    Query<(&bevy::prelude::ChildOf, &crate::cell::BodyPartIndex, &bevy::prelude::GlobalTransform)>,
    _heightmap:     Option<Res<crate::world_geometry::HeightmapSampler>>,
    _time:          Res<bevy::prelude::Time<bevy::prelude::Virtual>>,
    _anchors:       Local<HashMap<Entity, Vec3>>,
    _next_share:    Local<f32>,
) {
    // NO-OP under the per-SPECIES shared-policy model. Social policy-copy across
    // individuals is now intrinsic: every organism of a species trains and uses
    // the SAME net, so a "best walker"'s improvement is shared with its whole
    // species automatically every PPO update — there is no per-slot weight to
    // copy. (Cross-species copying would be wrong: species hold deliberately
    // divergent policies.) The system is left registered (in `behaviour.rs`) but
    // does nothing; the per-species PPO update is the convergence mechanism.
}


// ── Apply ───────────────────────────────────────────────────────────────────

/// Per-tick brain step. Delegates to the shared `apply_step` engine:
/// build observations, batched actor forward, sample actions, write to
/// `Organism::limb_targets`, record rollout entries for the PPO update.
pub fn apply_intelligence_level_herbivore_1_limb(
    mut pool: NonSendMut<BrainPoolHerbivore1Limb>,
    organisms: Query<(Entity, &mut Organism, &BrainSlotHerbivore1Limb)>,
    body_parts: Query<(
        &bevy::prelude::ChildOf,
        &crate::cell::BodyPartIndex,
        &bevy::prelude::GlobalTransform,
        &bevy_rapier3d::prelude::Velocity,
        Option<&crate::rapier_setup::LimbContact>,
    )>,
    world_grid: Res<crate::world_model::WorldModelGrid>,
    heightmap: Option<Res<crate::world_geometry::HeightmapSampler>>,
    virtual_time: Res<bevy::prelude::Time<bevy::prelude::Virtual>>,
) {
    let obs_inputs = gather_limb_obs_inputs(&body_parts, &world_grid, heightmap.as_deref());
    pool.0.apply_step(organisms, &obs_inputs, virtual_time.elapsed_secs());
}
