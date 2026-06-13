// Limb-based L3 brain — PPO pool for Level3 heterotrophs with
// `!Organism::movement_mode.is_sliding()`.
//
// Structurally identical to the herbivore_1 / L2 limb pools; only the
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
        if organism.movement_mode.is_sliding() { continue; }
        // Swimmers train in their own pool (intelligence_level_1_swimming).
        if organism.movement_mode.is_swimming() { continue; }
        let Some(s) = pool.0.enrol(e) else { continue };
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
