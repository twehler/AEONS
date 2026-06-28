// Intelligence Level 2 — Predator sliding REINFORCE pool.
//
// Thin per-level wrapper over the shared `sliding_reinforce` engine (mirrors how
// the limb pools wrap `limb_ppo::BrainPoolLimb`). The engine owns the network,
// REINFORCE math, target-lock/blacklist/brake machinery, save/load, snapshot,
// and telemetry; this file supplies only the per-level axes:
//   * `IntelligenceLevel::Level2` enrolment filter.
//   * `HIDDEN = 16`.
//   * Prey type: `Hetero` if carnivore else `Photo`.
//
// The public symbols other modules import (`BrainPoolL2`, `BrainSlotL2`,
// `assign_brains_l2`, `free_brains_l2`, `apply_intelligence_level_2`) keep their
// names + signatures. The pool resource derefs to the shared engine so
// `pool.snapshot()`, `pool.sigma[..]` etc. work unchanged at the call sites.

use bevy::prelude::*;
use std::ops::{Deref, DerefMut};

use crate::colony::{IntelligenceLevel, Organism, Heterotroph, Carnivore};
use crate::rl_helpers::{BrainInheritance, BrainRestore};
use crate::simulation_settings::OrganismPoolSize;
use crate::sliding_reinforce::{
    BrainPoolSliding, SlidingConfig, SlidingSlot, warmup,
};
use crate::world_model::OrganismType;


/// HIDDEN width for this pool. (Engine `IN`/`OUT` are shared constants.)
const HIDDEN: usize = 16;

/// Per-level config: carnivores hunt heterotrophs, others hunt photoautotrophs.
const CONFIG: SlidingConfig = SlidingConfig {
    hidden:    HIDDEN,
    prey_type: |is_carnivore| if is_carnivore { OrganismType::Hetero } else { OrganismType::Photo },
};


// ── Slot marker ─────────────────────────────────────────────────────────────

#[derive(Component, Clone, Copy)]
pub struct BrainSlotL2(pub u32);

impl SlidingSlot for BrainSlotL2 {
    fn slot(&self) -> u32 { self.0 }
}


// ── Pool resource (newtype over the shared engine) ──────────────────────────

pub struct BrainPoolL2(pub BrainPoolSliding);

impl Deref for BrainPoolL2 {
    type Target = BrainPoolSliding;
    fn deref(&self) -> &Self::Target { &self.0 }
}
impl DerefMut for BrainPoolL2 {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl FromWorld for BrainPoolL2 {
    fn from_world(world: &mut World) -> Self {
        let n = world
            .get_resource::<OrganismPoolSize>()
            .map(|r| r.0.max(1))
            .unwrap_or(1);
        let device = burn_cuda::CudaDevice::default();
        warmup(&device, HIDDEN);
        Self(BrainPoolSliding::new(device, n, CONFIG))
    }
}


// ── Slot allocation systems ─────────────────────────────────────────────────

pub fn assign_brains_l2(
    mut pool:     NonSendMut<BrainPoolL2>,
    new:          Query<(Entity, &Organism, Option<&BrainInheritance>, Option<&BrainRestore>), (
        With<Heterotroph>,
        Without<BrainSlotL2>,
    )>,
    mut commands: Commands,
) {
    let mut rng = rand::rng();
    for (e, organism, _inheritance, restore) in new.iter() {
        // Strict per-level dispatch: Level2 heterotrophs only.
        if !matches!(organism.intelligence_level, IntelligenceLevel::Level2) { continue; }
        // Sliding pool — limb-based organisms enrol in the parallel limb pool.
        if !organism.movement_mode.is_sliding() { continue; }

        let Some(slot) = pool.0.enrol(e, organism, restore, &mut rng) else { continue };

        commands.entity(e).try_insert(BrainSlotL2(slot));
        commands.entity(e).try_remove::<BrainInheritance>();
        commands.entity(e).try_remove::<BrainRestore>();
    }
}

pub fn free_brains_l2(
    mut pool:    NonSendMut<BrainPoolL2>,
    mut removed: RemovedComponents<Heterotroph>,
) {
    for e in removed.read() {
        pool.0.release(e);
    }
}


// ── Apply / train tick ──────────────────────────────────────────────────────

pub fn apply_intelligence_level_2(
    time:        Res<Time<Virtual>>,
    world_grid:  Res<crate::world_model::WorldModelGrid>,
    mut pool:    NonSendMut<BrainPoolL2>,
    heteros:     Query<(Entity, &mut Organism, &Transform, &BrainSlotL2, Option<&Carnivore>), With<Heterotroph>>,
    mut input_buf: Local<Vec<f32>>,
) {
    pool.0.apply_step(&time, &world_grid, heteros, &mut input_buf);
}
