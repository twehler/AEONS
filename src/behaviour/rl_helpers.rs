// RL helpers shared by every sliding brain pool. The level-independent
// scaffolding (backend alias, inheritance marker, save/load snapshot,
// noise sampler) lives here so each pool file only carries its distinct
// constants and apply system.

use bevy::prelude::*;
use burn::backend::Autodiff;
use burn_cuda::Cuda;
use rand::{Rng, RngExt};
use std::collections::HashMap;


/// Autodiff backend over CUDA — every pool's `Module` is parameterised on
/// this so gradients flow on the GPU.
pub type MyBackend = Autodiff<Cuda>;

/// No-autodiff INFERENCE backend — the inner backend of `MyBackend`.
///
/// Per-tick forward passes that only drive motors do not need gradients, so
/// they run on this plain `Cuda` backend: no autograd tape is built and then
/// discarded every tick (the dominant per-tick waste when the nets are tiny).
/// Training still runs on `MyBackend`. Obtain a module's inference view with
/// `burn::module::AutodiffModule::valid()` (e.g. `actor.valid()` →
/// `LimbActor<InferBackend>`) and build its input tensor on `InferBackend`;
/// `<MyBackend as AutodiffBackend>::InnerBackend == InferBackend`, and both
/// share the same `CudaDevice`.
pub type InferBackend = Cuda;


/// Marker on offspring: "when assigned a slot, copy this parent's row
/// first." `reproduction.rs` attaches it (parent must be non-Level0);
/// the next `assign_brains_*` copies the parent's weight rows then
/// removes the marker. If the parent has left the pool, inheritance
/// degrades silently to the recycled slot's existing weights.
#[derive(Component, Clone, Copy)]
pub struct BrainInheritance(pub Entity);


/// One brain's state as flat CPU vectors. Carried on loaded entities so
/// `assign_brains_*` rehydrates the entity's SPECIES net exactly (weights,
/// REINFORCE prev_*, baseline, has_prev).
///
/// SHARED-POLICY layout: the sliding pools are now per-SPECIES, so `w1/b1/
/// w2/b2` are ONE 2-D net's flat weights (not a per-slot row out of a
/// batched `[N, …]` arena). `restore_species` builds/overwrites the
/// entity's species net from this. The `prev_*`/`baseline`/`has_prev`
/// fields remain PER INDIVIDUAL (one organism's REINFORCE bookkeeping;
/// `baseline` carries the entity's species baseline at save time).
///
/// Layouts (constants `IN`/`HIDDEN`/`OUT`):
///   w1 — `IN*HIDDEN` row-major `[IN,HIDDEN]`; b1 — `HIDDEN`;
///   w2 — `HIDDEN*OUT` row-major `[HIDDEN,OUT]`; b2 — `OUT`;
///   prev_state — `IN`; prev_action — `OUT`.
///
/// `restore_species` validates counts and degrades to a fresh species net
/// on mismatch (so a save under a different `HIDDEN` degrades, not panics).
/// Adam moments are NOT serialised — they readapt quickly, and including
/// them would couple the save to Burn's optimiser internals.
#[derive(Component, Clone, Debug)]
pub struct BrainRestore {
    pub w1:           Vec<f32>,
    pub b1:           Vec<f32>,
    pub w2:           Vec<f32>,
    pub b2:           Vec<f32>,
    pub prev_state:   Vec<f32>,
    pub prev_action:  Vec<f32>,
    pub prev_energy:  f32,
    pub baseline:     f32,
    pub has_prev:     bool,
}


/// One species' shared net as flat CPU weight vectors plus its species key.
/// Produced per live species by a pool's `snapshot()`.
#[derive(Clone, Debug)]
pub struct SpeciesWeights {
    pub w1: Vec<f32>,    // [IN * HIDDEN]
    pub b1: Vec<f32>,    // [HIDDEN]
    pub w2: Vec<f32>,    // [HIDDEN * OUT]
    pub b2: Vec<f32>,    // [OUT]
    pub baseline: f32,   // this species' REINFORCE EMA baseline
}


/// Full read-only snapshot of one PER-SPECIES sliding pool's GPU + CPU
/// state, produced once per pool by the save system (a few GPU→CPU syncs
/// per live species). `extract(entity)` then derives a `BrainRestore` per
/// organism — the entity's SPECIES net weights + that entity's
/// per-individual prev_* bookkeeping — with no further GPU traffic.
pub struct PoolSnapshot {
    /// Per-species shared net weights, keyed by species id (UNCLASSIFIED = 0).
    pub species:      HashMap<u32, SpeciesWeights>,
    /// Entity → its species key (read fresh each apply tick into the pool's
    /// `slot_species`; snapshotted via `map`+`slot_species`).
    pub entity_species: HashMap<Entity, u32>,
    /// Entity → slot, so per-individual prev_* can be sliced.
    pub map:          HashMap<Entity, u32>,
    pub prev_state:   Vec<f32>,
    pub prev_action:  Vec<f32>,
    pub prev_energy:  Vec<f32>,
    pub has_prev:     Vec<bool>,
    pub in_dim:       usize,
    pub hidden_dim:   usize,
    pub out_dim:      usize,
}

impl PoolSnapshot {
    /// Derive a `BrainRestore` for one entity — its SPECIES net weights plus
    /// that entity's per-individual prev_* bookkeeping — or `None` if the
    /// entity has no slot. Pure CPU memcpy — no GPU access. `baseline` is the
    /// entity's species baseline (per-species EMA, mirrored into each member's
    /// restore payload).
    pub fn extract(&self, entity: Entity) -> Option<BrainRestore> {
        let slot = *self.map.get(&entity)? as usize;
        let key  = self.entity_species.get(&entity).copied().unwrap_or(0);
        let sp   = self.species.get(&key)?;
        Some(BrainRestore {
            w1: sp.w1.clone(),
            b1: sp.b1.clone(),
            w2: sp.w2.clone(),
            b2: sp.b2.clone(),
            prev_state:  self.prev_state [slot * self.in_dim  .. (slot + 1) * self.in_dim ].to_vec(),
            prev_action: self.prev_action[slot * self.out_dim .. (slot + 1) * self.out_dim].to_vec(),
            prev_energy: self.prev_energy[slot],
            baseline:    sp.baseline,
            has_prev:    self.has_prev[slot],
        })
    }
}


/// Single standard-normal sample via polar Box–Muller (rejection
/// sampling on the unit disc; no sin/cos). `rand` 0.10 has no
/// `StandardNormal`, hence the hand-rolled sampler. The companion `v`
/// is discarded to keep a simple `f32` return.
pub fn gaussian_noise<G: Rng + ?Sized>(rng: &mut G) -> f32 {
    loop {
        let u: f32 = rng.random_range(-1.0_f32..1.0_f32);
        let v: f32 = rng.random_range(-1.0_f32..1.0_f32);
        let s = u * u + v * v;
        if s > 0.0_f32 && s < 1.0_f32 {
            return u * (-2.0_f32 * s.ln() / s).sqrt();
        }
    }
}
