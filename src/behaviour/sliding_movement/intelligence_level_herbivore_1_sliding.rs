// Intelligence Level 1 (herbivore, non-carnivore) — sliding REINFORCE pool.
//
// Thin per-level wrapper over the shared `sliding_reinforce` engine (mirrors
// how the limb pools wrap `limb_ppo::BrainPoolLimb`). The engine owns the
// network, REINFORCE math, target-lock/blacklist/brake machinery, save/load,
// snapshot, and telemetry; this file supplies only the per-level axes:
//   * `IntelligenceLevel::Level1` + `Without<Carnivore>` enrolment filter.
//   * `HIDDEN = 32`.
//   * Prey type: always `Photo` (herbivores hunt photoautotrophs).
//
// The public symbols other modules import (`BrainPoolHerbivore1`,
// `BrainSlotHerbivore1`, `assign_brains_herbivore_1`, `free_brains_herbivore_1`,
// `apply_intelligence_level_herbivore_1`, `TrainingStep`, `BrainTelemetry`,
// `BrainRestoreHerbivore1`, `encode_brain_restore`, `decode_brain_restore`)
// keep their names + signatures. The pool resource derefs to the shared engine
// so `pool.snapshot()`, `pool.snapshot_telemetry()`, `pool.extract_slot(..)`,
// `pool.training_history()`, `pool.map`, `pool.sigma[..]`, `pool.n()` all work
// unchanged at the call sites.

use bevy::prelude::*;
use std::ops::{Deref, DerefMut};

use crate::colony::{IntelligenceLevel, Organism, Heterotroph, Carnivore};
use crate::rl_helpers::{BrainInheritance, BrainRestore};
use crate::simulation_settings::OrganismPoolSize;
use crate::sliding_reinforce::{
    BrainPoolSliding, SlidingConfig, SlidingSlot, warmup,
};
use crate::world_model::OrganismType;

// Re-export the engine's public types under the names other modules import.
pub use crate::sliding_reinforce::{
    BrainRestoreHerbivore1, BrainTelemetry, TrainingStep,
    encode_brain_restore, decode_brain_restore,
};


/// HIDDEN width for this pool. (Engine `IN`/`OUT` are shared constants.)
const HIDDEN: usize = 32;

/// Per-level config: always hunt photoautotrophs.
const CONFIG: SlidingConfig = SlidingConfig {
    hidden:    HIDDEN,
    prey_type: |_is_carnivore| OrganismType::Photo,
    direction_3d: false,
};


// ── Slot marker ─────────────────────────────────────────────────────────────

#[derive(Component, Clone, Copy)]
pub struct BrainSlotHerbivore1(pub u32);

impl SlidingSlot for BrainSlotHerbivore1 {
    fn slot(&self) -> u32 { self.0 }
}


// ── Pool resource (newtype over the shared engine) ──────────────────────────

pub struct BrainPoolHerbivore1(pub BrainPoolSliding);

impl Deref for BrainPoolHerbivore1 {
    type Target = BrainPoolSliding;
    fn deref(&self) -> &Self::Target { &self.0 }
}
impl DerefMut for BrainPoolHerbivore1 {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl FromWorld for BrainPoolHerbivore1 {
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

pub fn assign_brains_herbivore_1(
    mut pool:     NonSendMut<BrainPoolHerbivore1>,
    new:          Query<(Entity, &Organism, Option<&BrainInheritance>, Option<&BrainRestore>), (
        With<Heterotroph>,
        Without<Carnivore>,
        Without<BrainSlotHerbivore1>,
    )>,
    mut commands: Commands,
) {
    let mut rng = rand::rng();
    for (e, organism, _inheritance, restore) in new.iter() {
        // Level1 non-carnivore heterotrophs only (carnivores→L2, Level3→L3).
        if !matches!(organism.intelligence_level, IntelligenceLevel::Level1) { continue; }
        // Sliding pool only; limb organisms enrol in the parallel limb pool.
        if !organism.movement_mode.is_sliding() { continue; }

        // `BrainInheritance` is a NO-OP under the per-species shared policy
        // (same-species newborns already share the trained net); the marker is
        // dropped below.
        let Some(slot) = pool.0.enrol(e, organism, restore, &mut rng) else { continue };

        commands.entity(e).try_insert(BrainSlotHerbivore1(slot));
        commands.entity(e).try_remove::<BrainInheritance>();
        commands.entity(e).try_remove::<BrainRestore>();
    }
}

pub fn free_brains_herbivore_1(
    mut pool:    NonSendMut<BrainPoolHerbivore1>,
    mut removed: RemovedComponents<Heterotroph>,
) {
    for e in removed.read() {
        pool.0.release(e);
    }
}


// ── Apply / train tick ──────────────────────────────────────────────────────

pub fn apply_intelligence_level_herbivore_1(
    time:        Res<Time<Virtual>>,
    world_grid:  Res<crate::world_model::WorldModelGrid>,
    mut pool:    NonSendMut<BrainPoolHerbivore1>,
    heteros:     Query<
        (Entity, &mut Organism, &Transform, &BrainSlotHerbivore1, Option<&Carnivore>),
        (With<Heterotroph>, Without<Carnivore>),
    >,
    mut input_buf: Local<Vec<f32>>,
) {
    pool.0.apply_step(&time, &world_grid, heteros, &mut input_buf);
}
