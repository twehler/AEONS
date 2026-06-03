// Limb-based L3 brain — PPO pool for Level3 heterotrophs WHOSE
// `Organism::sliding_movement == false`.
//
// Structurally identical to the herbivore_1 / L2 limb pools — only the
// enrolment filter (`IntelligenceLevel::Level3`) differs. Network sizes
// and PPO hyperparameters come from the shared `limb_ppo` engine.

use bevy::prelude::*;

use crate::colony::{IntelligenceLevel, Organism, Heterotroph};
use crate::limb_ppo::{BrainPoolLimb, BrainRestoreLimb, LimbSlot, gather_limb_obs_inputs};
use crate::simulation_settings::OrganismPoolSize;


#[derive(Component, Clone, Copy)]
pub struct BrainSlotL3Limb(pub u32);

impl LimbSlot for BrainSlotL3Limb {
    fn slot(&self) -> u32 { self.0 }
}


pub struct BrainPoolL3Limb(pub BrainPoolLimb);

impl FromWorld for BrainPoolL3Limb {
    fn from_world(world: &mut World) -> Self {
        let n = world.resource::<OrganismPoolSize>().0;
        let device = burn_cuda::CudaDevice::default();
        Self(BrainPoolLimb::new(n, device))
    }
}


pub fn assign_brains_l3_limb(
    mut pool:     NonSendMut<BrainPoolL3Limb>,
    new:          Query<(Entity, &Organism, Option<&BrainRestoreLimb>, Option<&crate::rl_helpers::BrainInheritance>), (
        With<Heterotroph>,
        Without<BrainSlotL3Limb>,
    )>,
    mut commands: Commands,
) {
    for (e, organism, restore, inheritance) in new.iter() {
        if !matches!(organism.intelligence_level, IntelligenceLevel::Level3) { continue; }
        if organism.sliding_movement { continue; }
        let Some(s) = pool.0.enrol(e) else { continue };
        if let Some(r) = restore {
            pool.0.restore_slot(s, r);
            commands.entity(e).try_remove::<BrainRestoreLimb>();
        } else {
            // Inherit a trained brain so the new organism isn't born helpless;
            // prefer the explicit parent, else any other occupied slot.
            let src = inheritance.and_then(|inh| pool.0.map.get(&inh.0).copied())
                .or_else(|| pool.0.map.iter().filter_map(|(ent, sl)| (*ent != e && *sl != s).then_some(*sl)).next());
            if let Some(src) = src { pool.0.inherit_slot(s, src); }
        }
        if inheritance.is_some() { commands.entity(e).try_remove::<crate::rl_helpers::BrainInheritance>(); }
        commands.entity(e).try_insert(BrainSlotL3Limb(s));
    }
}


pub fn free_brains_l3_limb(
    mut pool:    NonSendMut<BrainPoolL3Limb>,
    mut removed: RemovedComponents<BrainSlotL3Limb>,
) {
    let slots: Vec<(Entity, u32)> = removed.read()
        .filter_map(|e| pool.0.map.get(&e).map(|&s| (e, s)))
        .collect();
    for (e, s) in slots {
        pool.0.release(e, s);
    }
}


pub fn apply_intelligence_level_3_limb(
    mut pool: NonSendMut<BrainPoolL3Limb>,
    organisms: Query<(Entity, &mut Organism, &BrainSlotL3Limb)>,
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
