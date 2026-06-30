// Intelligence Level 1 — SimpleAquatic (basic 3D movement) REINFORCE pool.
//
// The simplest mover in the simulation: a limbless, KINEMATIC creature that swims
// freely through the water volume, hunting prey in full 3D. It reuses the shared
// `sliding_reinforce` engine VERBATIM (same network, REINFORCE math, target-lock /
// blacklist / brake machinery, save/load, telemetry) — the ONLY differences from
// the benthic sliding herbivore pool are:
//   * `is_simple_aquatic()` enrolment gate (disjoint from the sliding + limb + swim
//     pools, all of which explicitly skip SimpleAquatic).
//   * `HIDDEN = 16` (cheap).
//   * `direction_3d: true` — the geometric pursuit direction keeps its Y component,
//     so the mover climbs/descends toward prey instead of the XZ-only slide.
//
// The brain only PICKS a target + speed; `apply_3d_kinematic_movement` (movement.rs)
// integrates the 3D direction and auto-rotates the body to face travel, and
// `contain_simple_aquatic` keeps it between the seafloor and the water surface.
//
// Brain blocks serialise with the same `encode_brain_restore`/`decode_brain_restore`
// codec as the sliding herbivore pool; `colony_save_load` routes SimpleAquatic L1
// heteros here by `(movement_mode, intelligence_level)`.

use bevy::prelude::*;
use std::ops::{Deref, DerefMut};

use crate::colony::{IntelligenceLevel, Organism, Heterotroph, Carnivore};
use crate::rl_helpers::{BrainInheritance, BrainRestore};
use crate::simulation_settings::OrganismPoolSize;
use crate::sliding_reinforce::{BrainPoolSliding, SlidingConfig, SlidingSlot, warmup};
use crate::world_model::OrganismType;


/// HIDDEN width for this pool — small/cheap (the mode is meant to be the lightest).
const HIDDEN: usize = 16;

/// Per-level config: hunt photoautotrophs, pursue in full 3D.
const CONFIG: SlidingConfig = SlidingConfig {
    hidden:       HIDDEN,
    prey_type:    |_is_carnivore| OrganismType::Photo,
    direction_3d: true,
};


// ── Slot marker ─────────────────────────────────────────────────────────────

#[derive(Component, Clone, Copy)]
pub struct BrainSlotSimpleAquatic(pub u32);

impl SlidingSlot for BrainSlotSimpleAquatic {
    fn slot(&self) -> u32 { self.0 }
}


// ── Pool resource (newtype over the shared engine) ──────────────────────────

pub struct BrainPoolSimpleAquatic(pub BrainPoolSliding);

impl Deref for BrainPoolSimpleAquatic {
    type Target = BrainPoolSliding;
    fn deref(&self) -> &Self::Target { &self.0 }
}
impl DerefMut for BrainPoolSimpleAquatic {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl FromWorld for BrainPoolSimpleAquatic {
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

pub fn assign_brains_simple_aquatic(
    mut pool:     NonSendMut<BrainPoolSimpleAquatic>,
    new:          Query<(Entity, &Organism, Option<&BrainInheritance>, Option<&BrainRestore>), (
        With<Heterotroph>,
        Without<Carnivore>,
        Without<BrainSlotSimpleAquatic>,
    )>,
    mut commands: Commands,
) {
    let mut rng = rand::rng();
    for (e, organism, _inheritance, restore) in new.iter() {
        // Level1 non-carnivore heterotrophs that are SimpleAquatic only — disjoint
        // from the sliding / limb / swim pools (each of which skips this mode).
        if !matches!(organism.intelligence_level, IntelligenceLevel::Level1) { continue; }
        if !organism.movement_mode.is_simple_aquatic() { continue; }

        let Some(slot) = pool.0.enrol(e, organism, restore, &mut rng) else { continue };

        commands.entity(e).try_insert(BrainSlotSimpleAquatic(slot));
        commands.entity(e).try_remove::<BrainInheritance>();
        commands.entity(e).try_remove::<BrainRestore>();
    }
}

pub fn free_brains_simple_aquatic(
    mut pool:    NonSendMut<BrainPoolSimpleAquatic>,
    mut removed: RemovedComponents<Heterotroph>,
) {
    for e in removed.read() {
        pool.0.release(e);
    }
}


// ── Apply / train tick ──────────────────────────────────────────────────────

pub fn apply_intelligence_level_simple_aquatic(
    time:        Res<Time<Virtual>>,
    world_grid:  Res<crate::world_model::WorldModelGrid>,
    mut pool:    NonSendMut<BrainPoolSimpleAquatic>,
    heteros:     Query<
        (Entity, &mut Organism, &Transform, &BrainSlotSimpleAquatic, Option<&Carnivore>),
        (With<Heterotroph>, Without<Carnivore>),
    >,
    mut input_buf: Local<Vec<f32>>,
) {
    pool.0.apply_step(&time, &world_grid, heteros, &mut input_buf);
}
