// RL helpers shared by every brain pool (`intelligence_level_1_photo`,
// `intelligence_level_1_hetero`, `intelligence_level_2`,
// `intelligence_level_3`).
//
// Each pool implements REINFORCE on a private per-organism MLP — the
// pieces below are the bits that don't depend on which level (input
// width, hidden width) the pool is. Keeping them here means the pool
// files can stay focused on their distinct constants and apply
// systems instead of re-declaring scaffolding four times.
//
// What lives here:
//   * `MyBackend` — the burn autodiff-on-CUDA backend type alias.
//   * `BrainInheritance` — component placed on offspring entities by
//     `reproduction.rs` so the pool's `assign_brains_*` can copy the
//     parent's slot weights to the new slot before training begins.
//     This is how "weights inherited from parent → offspring" is
//     implemented; without it, recycled slots would inherit the
//     PREVIOUS occupant's weights (which is the existing "instinct
//     survives slot recycling" behaviour for newly-spawned organisms
//     that don't have a parent in the pool).
//   * `gaussian_noise` — Box–Muller scalar sampler. Each REINFORCE
//     tick samples `N × OUT` standard-normal values; rand 0.10 has no
//     `StandardNormal` so we generate them ourselves. Per-tick cost
//     is trivial (≤ 8000 calls × a few flops).

use bevy::prelude::*;
use burn::backend::Autodiff;
use burn_cuda::Cuda;
use rand::{Rng, RngExt};
use std::collections::HashMap;


/// Autodiff backend over CUDA. Every pool's `Module` is parameterised on
/// this so gradients flow on the GPU. Aliased here so each pool file
/// doesn't have to repeat the `Autodiff<Cuda>` spelling.
pub type MyBackend = Autodiff<Cuda>;


/// Marker placed on a freshly-spawned offspring entity: "when the
/// brain pool assigns me a slot, copy this parent's row first".
///
/// Lifecycle:
///   1. `reproduction.rs` spawns the offspring root entity, then
///      attaches `BrainInheritance(parent_entity)` (only if the
///      parent's `intelligence_level` is non-Level0 — Level0 has no
///      pool).
///   2. The next `assign_brains_*` system in PreUpdate sees the new
///      entity (via the `Without<BrainSlot…>` query), pops a free slot,
///      reads `BrainInheritance.0` to find the parent's slot index in
///      `pool.map`, and runs `pool.inherit_row(parent_slot, new_slot)`
///      to deep-copy the four `[N, …]` weight tensors row-wise.
///   3. The same system removes `BrainInheritance` from the entity
///      to keep the component set clean.
///
/// If the parent is no longer in the pool (e.g. it died between
/// spawn and the next PreUpdate), inheritance silently degrades to
/// "use the recycled slot's existing weights". That is structurally
/// equivalent to the pre-RL behaviour and is acceptable — orphaned
/// offspring just get whichever instinct the slot's previous tenant
/// had trained.
#[derive(Component, Clone, Copy)]
pub struct BrainInheritance(pub Entity);


/// One slot's worth of pool state, as flat CPU vectors. Carried on
/// loaded entities by the save/load pipeline (`colony.rs`) so that
/// `assign_brains_*` can rehydrate the slot to *exactly* the state
/// it was in at save time — weights, REINFORCE prev_*, baseline,
/// and the has-prev flag.
///
/// Layouts (for a pool with constants `IN`, `HIDDEN`, `OUT`):
///   * `w1`           — `IN * HIDDEN` floats, row-major as `[IN, HIDDEN]`
///   * `b1`           — `HIDDEN` floats
///   * `w2`           — `HIDDEN * OUT` floats, row-major as `[HIDDEN, OUT]`
///   * `b2`           — `OUT` floats
///   * `prev_state`   — `IN` floats
///   * `prev_action`  — `OUT` floats
///
/// Counts are validated by the receiving pool's `restore_slot` and
/// the slot is reset to defaults on mismatch (logged) so a save
/// produced under different `HIDDEN` (e.g. after a code change)
/// degrades cleanly to "fresh weights" rather than panicking.
///
/// Note: Adam optimiser moments are NOT serialised. Adam adapts
/// quickly from zeroed moments after load, and including them
/// would make the save couple to Burn's internal optimiser
/// representation which we'd rather not pin.
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


/// Full read-only snapshot of one brain pool's GPU + CPU state.
/// Produced once per pool by the save system (one GPU→CPU sync per
/// weight tensor — four total per pool). `extract(entity)` then
/// derives a `BrainRestore` for any organism whose slot is in
/// `map`, with no further GPU traffic.
///
/// Returning the snapshot from the pool keeps the per-organism
/// extraction loop free of locking on the `NonSend<BrainPool*>`
/// resource and lets the save system iterate organisms in
/// whatever order the Bevy query yields them.
pub struct PoolSnapshot {
    pub w1:           Vec<f32>,    // [N * IN * HIDDEN]
    pub b1:           Vec<f32>,    // [N * HIDDEN]
    pub w2:           Vec<f32>,    // [N * HIDDEN * OUT]
    pub b2:           Vec<f32>,    // [N * OUT]
    pub map:          HashMap<Entity, u32>,
    pub prev_state:   Vec<f32>,
    pub prev_action:  Vec<f32>,
    pub prev_energy:  Vec<f32>,
    pub has_prev:     Vec<bool>,
    pub baseline:     Vec<f32>,
    pub in_dim:       usize,
    pub hidden_dim:   usize,
    pub out_dim:      usize,
}

impl PoolSnapshot {
    /// Slice out one slot's worth of state into a `BrainRestore`,
    /// or return `None` if the entity has no slot in this pool.
    /// Pure CPU memcpy — no GPU access — so iteration over all
    /// organisms in the save loop is cheap.
    pub fn extract(&self, entity: Entity) -> Option<BrainRestore> {
        let slot = *self.map.get(&entity)? as usize;
        let w1_per = self.in_dim * self.hidden_dim;
        let w2_per = self.hidden_dim * self.out_dim;
        Some(BrainRestore {
            w1: self.w1[slot * w1_per .. (slot + 1) * w1_per].to_vec(),
            b1: self.b1[slot * self.hidden_dim .. (slot + 1) * self.hidden_dim].to_vec(),
            w2: self.w2[slot * w2_per .. (slot + 1) * w2_per].to_vec(),
            b2: self.b2[slot * self.out_dim .. (slot + 1) * self.out_dim].to_vec(),
            prev_state:  self.prev_state [slot * self.in_dim  .. (slot + 1) * self.in_dim ].to_vec(),
            prev_action: self.prev_action[slot * self.out_dim .. (slot + 1) * self.out_dim].to_vec(),
            prev_energy: self.prev_energy[slot],
            baseline:    self.baseline[slot],
            has_prev:    self.has_prev[slot],
        })
    }
}


/// Single standard-normal sample via the polar Box–Muller method.
/// Cheaper than the trigonometric form (no sin/cos), and gives one
/// f32 per call which is what the apply-system inner loop wants.
///
/// Implemented as rejection sampling on the unit disc — each call
/// samples `(u, v) ∈ [-1, 1]²` until `s = u² + v² ∈ (0, 1]`, then
/// returns `u * sqrt(-2 ln s / s)`. Average ~1.27 disc draws per
/// sample (acceptance ≈ π/4); the discarded `v` companion is
/// thrown away to keep the API a simple `f32` return.
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
