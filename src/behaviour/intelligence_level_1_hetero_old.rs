// Intelligence Level 1 — Heterotroph REINFORCE pool (smallest hetero pool).
//
// Mechanically identical to `intelligence_level_1_photo.rs` — same
// REINFORCE loop, same per-organism MLP, same Gaussian policy with
// fixed σ. The only differences are:
//
//   * the input vector (heterotrophs read a per-organism world model
//     instead of position + sunlight),
//   * the marker / slot component types (so heterotrophs land in
//     this pool's row catalogue instead of the photoautotroph one),
//   * a slight wider input width because of the 16-dim world-model
//     contribution (`IN = 17` vs photoautotroph's 5).
//
// Inputs (17):
//   * normalised energy        (∈ [0, 1])             — 1 dim
//   * world model               (`WORLD_MODEL_DIMS`)  — 16 dims
//
// The world model packs the K=4 nearest neighbouring organisms within
// `WORLD_MODEL_RADIUS = 60`, each encoded as `(rel_x_norm, rel_z_norm,
// is_photo, is_hetero)` — see `world_model.rs`. That gives the brain
// just enough spatial awareness to seek prey (cells flagged
// `is_photo`) and avoid same-species collisions, without paying for a
// full per-tick neighbour scan inside this system (the scan happens
// once per tick in `rebuild_world_model_grid`).
//
// Outputs (3): tanh-squashed `(speed, dir.x, dir.z)`. Heterotroph
// motion is XZ-planar; we don't waste a policy dimension on dir.y
// (a deleted output here is one fewer noisy gradient column going
// into the shared trunk via the policy loss). Speed maps to
// `[0, MAX_SPEED]` via `max(0, speed_a)·MAX_SPEED` so the lower
// half of the action space is "stand still" — the random-init
// policy reaches zero-speed about 50% of the time, matching the
// natural "conserve energy" baseline behaviour.
//
// Hidden width (16) deliberately matches the photo L1 pool: this is
// the *smallest* heterotroph brain. Levels 2 and 3 use 32 / 64
// respectively for the same input/output shape — that's the only
// thing the level number selects in the new RL world.
//
// See `intelligence_level_1_photo.rs` for the full REINFORCE
// derivation and per-tick flow narration; this file mirrors it
// without re-narrating.

use bevy::prelude::*;
use burn::module::{Initializer, Module, Param};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use burn_cuda::CudaDevice;
use std::collections::HashMap;

use crate::colony::{IntelligenceLevel, Organism, Heterotroph};
use crate::energy::get_max_energy;
use crate::rl_helpers::{BrainInheritance, BrainRestore, MyBackend, PoolSnapshot, gaussian_noise};
use crate::simulation_settings::OrganismPoolSize;
use crate::world_model::{
    WorldModelGrid, WORLD_MODEL_DIMS, WORLD_MODEL_RADIUS,
    fill_world_model, nearest_prey,
};


// ── Architecture constants ──────────────────────────────────────────────────

/// 1 (energy) + WORLD_MODEL_DIMS (= 16, K=4 neighbours × 4 dims) = 17.
const IN:        usize = 1 + WORLD_MODEL_DIMS;
const HIDDEN:    usize = 16;
const OUT:       usize = 3;
const MAX_SPEED: f32   = 20.0;
const LR:        f64   = 1e-3;

/// Standard deviation of the Gaussian exploration noise added to the
/// policy mean before sampling actions. Larger σ ⇒ broader
/// exploration each tick, which the actor-critic baseline can
/// absorb (the value head learns to subtract the average return
/// from each state, leaving only the per-action signal in the
/// advantage). Bumped from 0.2 → 0.5 so the heterotroph genuinely
/// "tries things out" instead of barely deviating from a still-
/// undertrained policy mean.
const SIGMA:          f32 = 0.5;
/// Discount factor for the TD bootstrap target `r + γ·V(s')`. With
/// the brain ticking at ~6.7 Hz, γ = 0.95 means a reward stays
/// "valuable" for roughly 1/(1-γ) = 20 ticks ≈ 3 seconds — long
/// enough for the brain to associate "approached prey" with the
/// eventual eat event.
const GAMMA:          f32 = 0.95;
/// Weight on the value-loss term relative to the policy-loss term
/// in the combined A2C objective. Standard A2C value: 0.5.
const VALUE_COEF:     f32 = 0.5;
/// Hard clip on the energy-delta component of the reward. Predation
/// can dump several units of energy in one tick; this clamp keeps
/// that one event from saturating the gradient.
const REWARD_CLAMP:   f32 = 1.0;

// ── Reward-shaping weights ─────────────────────────────────────────────────
//
// The energy delta alone is a near-zero signal at brain-tick scale —
// per-cell upkeep is tiny, predation is rare, photosynthesis doesn't
// apply to heterotrophs. The brain saw effectively no signal across
// most ticks, so REINFORCE was learning from pure noise. We add two
// dense components:
//
//   * `W_PROGRESS · (Δdistance to nearest prey) / WORLD_MODEL_RADIUS`
//     — positive when this tick's action moved the heterotroph closer
//     to its nearest in-range prey. State-delta signal, ~0.05/tick at
//     full speed.
//   * `W_FACING · cos(angle between forward and unit-vec-to-prey)`
//     — positive when the heterotroph is currently pointing at prey,
//     even at zero speed. Always present (in [-1, +1]) when prey is
//     in range; gives the brain something to learn early when speed
//     is still ≈ 0 from the random-init policy.
//
// Weight choices, post-audit (see `rl_il1_hetero_findings.txt`):
//
//   * Predation events still dominate the reward signal — the
//     energy delta clamps at ±1, far above anything the shaped
//     terms can produce on a single tick. Keep `W_ENERGY = 1.0`
//     even though energy gets pinned at 0 in debug mode (a rare
//     predation contact still spikes the signal cleanly).
//
//   * Crucial: PROGRESS MUST DOMINATE FACING in the per-tick
//     dense-reward budget. With the previous values (0.5 / 0.05),
//     a "stand still and rotate to face the nearest prey" policy
//     scored `0.05/tick`, while moving toward prey at MAX_SPEED
//     scored only `0.5 · (20·0.15/60) = 0.025/tick`. The agent
//     correctly identified standing-still-facing as the optimal
//     reward-hacking strategy — which exactly matches the
//     observed "slow + erratic" pathology. Rebalanced so:
//       - Max-speed chase: 2.0 · 0.05 = +0.10/tick
//       - Face-while-still: 0.01 · 1.0 = +0.01/tick
//     i.e. chasing is ~10× more valuable than facing, so the
//     value head can't sit at the facing-only fixed point.
//
//   * Facing kept non-zero (not 0.0) because it remains useful as
//     an early-training direction signal when the agent isn't yet
//     moving fast enough to register meaningful progress; just
//     small enough that it can't dominate.
const W_ENERGY:    f32 = 1.0;
const W_PROGRESS:  f32 = 2.0;
const W_FACING:    f32 = 0.01;


// ── Slot component ──────────────────────────────────────────────────────────

#[derive(Component, Clone, Copy)]
pub struct BrainSlotL1Hetero(pub u32);


// ── Per-organism MLP ────────────────────────────────────────────────────────
//
// Two-headed actor-critic on a shared trunk:
//
//   trunk:  ReLU(input · w1 + b1)             [HIDDEN]
//   policy: tanh(trunk · w2  + b2)            [OUT = 3]   — Gaussian mean
//   value:       (trunk · v_w + v_b)          [1]         — V(s) baseline
//
// `forward_full` returns both in one pass; the apply tick uses the
// value head as a state-dependent baseline for the policy gradient
// (replacing the per-slot scalar EMA we had before).

#[derive(Module, Debug)]
pub struct PoolMlpL1Hetero<B: Backend> {
    w1:  Param<Tensor<B, 3>>,
    b1:  Param<Tensor<B, 2>>,
    w2:  Param<Tensor<B, 3>>,
    b2:  Param<Tensor<B, 2>>,
    /// Value head: per-slot `[HIDDEN, 1]` row that maps the trunk
    /// activations to a scalar V(s). Linear (no tanh) so values can
    /// span the full reward range.
    v_w: Param<Tensor<B, 3>>,
    v_b: Param<Tensor<B, 2>>,
}

impl<B: Backend> PoolMlpL1Hetero<B> {
    fn new(device: &B::Device, n: usize) -> Self {
        let w = Initializer::Uniform { min: -0.5, max: 0.5 };
        let z = Initializer::Zeros;
        Self {
            w1:  w.init([n, IN, HIDDEN], device),
            b1:  z.init([n, HIDDEN], device),
            w2:  w.init([n, HIDDEN, OUT], device),
            b2:  z.init([n, OUT], device),
            // Zero-init the VALUE head so V(s₀) = 0 instead of a
            // random projection of the (small but non-zero) trunk
            // activations. With Uniform v_w, V(s) at init was
            // ~±2 in magnitude — about 20× the typical reward, so
            // the bootstrap target r + γV(s') was dominated by
            // random V for the first few hundred ticks and the
            // policy got random-sign advantages. Starting from
            // zero, the bootstrap target collapses to just `r`
            // until the value head learns useful predictions; the
            // policy gradient direction is correct from tick 1.
            v_w: z.init([n, HIDDEN, 1], device),
            v_b: z.init([n, 1], device),
        }
    }

    /// Single-output forward — kept around so the warmup function can
    /// trigger only the policy-trunk kernels when value-head gradients
    /// aren't needed. Unused on the hot path.
    #[allow(dead_code)]
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward_full(x).0
    }

    /// Two-headed forward. Returns `(mu, v)` from a SINGLE trunk
    /// activation — the trunk is computed once and `clone()`d to feed
    /// both heads, so backprop accumulates gradients into the trunk
    /// from both losses (standard A2C topology).
    fn forward_full(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // Trunk: ReLU(input · w1 + b1).
        let h = x.unsqueeze_dim::<3>(1).matmul(self.w1.val()).squeeze::<2>() + self.b1.val();
        let h = burn::tensor::activation::relu(h);

        // Policy head: tanh-squashed.
        let mu_pre = h.clone()
            .unsqueeze_dim::<3>(1)
            .matmul(self.w2.val())
            .squeeze::<2>() + self.b2.val();
        let mu = burn::tensor::activation::tanh(mu_pre);

        // Value head: linear scalar. The matmul output is `[N, 1, 1]`
        // (two size-1 dims), so we cannot use the same `.squeeze::<2>()`
        // trick the policy head uses — `squeeze::<2>()` collapses
        // every size-1 dim and would land us at rank 1 (`[N]`),
        // mismatching the rank-2 type. `reshape([N, 1])` is explicit
        // about the intended target shape and is free at runtime.
        let n = self.v_b.val().dims()[0];
        let v = h.unsqueeze_dim::<3>(1)
            .matmul(self.v_w.val())
            .reshape([n, 1]) + self.v_b.val();

        (mu, v)
    }
}


trait BrainOptL1Hetero {
    fn step(
        &mut self,
        lr: f64,
        m:  PoolMlpL1Hetero<MyBackend>,
        g:  GradientsParams,
    ) -> PoolMlpL1Hetero<MyBackend>;
}

impl<O: Optimizer<PoolMlpL1Hetero<MyBackend>, MyBackend>> BrainOptL1Hetero for O {
    fn step(
        &mut self,
        lr: f64,
        m:  PoolMlpL1Hetero<MyBackend>,
        g:  GradientsParams,
    ) -> PoolMlpL1Hetero<MyBackend> {
        Optimizer::step(self, lr, m, g)
    }
}


// ── Pool resource ───────────────────────────────────────────────────────────

pub struct BrainPoolL1Hetero {
    model:        PoolMlpL1Hetero<MyBackend>,
    opt:          Box<dyn BrainOptL1Hetero>,
    free:         Vec<u32>,
    map:          HashMap<Entity, u32>,
    prev_state:   Vec<f32>,
    prev_action:  Vec<f32>,
    prev_energy:  Vec<f32>,
    has_prev:     Vec<bool>,
    baseline:     Vec<f32>,
    /// Distance to the nearest in-range prey at the previous brain
    /// tick, per slot. Read by the reward shaper to compute the
    /// per-tick "approach progress" signal (`prev - cur`). Reset
    /// whenever no prey was in range last tick (`has_prev_prey[s] =
    /// false`), so the next tick's progress term is zero rather than
    /// a stale arbitrary number.
    prev_prey_dist: Vec<f32>,
    has_prev_prey:  Vec<bool>,
    /// Identity of the nearest prey at the previous brain tick
    /// (`None` when no prey was in range). Compared against this
    /// tick's nearest-prey entity to gate the progress reward —
    /// when the identity flips between ticks (a different photo
    /// became closest, or one was eaten), `(prev_dist - cur_dist)`
    /// is comparing distances to two different organisms, so we
    /// zero `progress` for that tick to avoid spuriously
    /// reinforcing whatever action coincidentally took the agent
    /// across the perpendicular bisector between two photos.
    prev_prey_entity: Vec<Option<Entity>>,
    pub device:   CudaDevice,
    pub n:        usize,
}

impl BrainPoolL1Hetero {
    fn new(device: CudaDevice, n: usize) -> Self {
        Self {
            model:            PoolMlpL1Hetero::<MyBackend>::new(&device, n),
            opt:              Box::new(AdamConfig::new().init()),
            free:             (0..n as u32).rev().collect(),
            map:              HashMap::with_capacity(n),
            prev_state:       vec![0.0; n * IN],
            prev_action:      vec![0.0; n * OUT],
            prev_energy:      vec![0.0; n],
            has_prev:         vec![false; n],
            baseline:         vec![0.0; n],
            prev_prey_dist:   vec![0.0; n],
            has_prev_prey:    vec![false; n],
            prev_prey_entity: vec![None; n],
            device,
            n,
        }
    }

    fn inherit_row(&mut self, parent: usize, child: usize) {
        let p = self.model.w1.val().slice([parent..parent+1, 0..IN, 0..HIDDEN]);
        self.model.w1 = self.model.w1.clone().map(|t| {
            t.slice_assign([child..child+1, 0..IN, 0..HIDDEN], p)
        });
        let p = self.model.b1.val().slice([parent..parent+1, 0..HIDDEN]);
        self.model.b1 = self.model.b1.clone().map(|t| {
            t.slice_assign([child..child+1, 0..HIDDEN], p)
        });
        let p = self.model.w2.val().slice([parent..parent+1, 0..HIDDEN, 0..OUT]);
        self.model.w2 = self.model.w2.clone().map(|t| {
            t.slice_assign([child..child+1, 0..HIDDEN, 0..OUT], p)
        });
        let p = self.model.b2.val().slice([parent..parent+1, 0..OUT]);
        self.model.b2 = self.model.b2.clone().map(|t| {
            t.slice_assign([child..child+1, 0..OUT], p)
        });
        // Value head — same row-slice copy. Without this the offspring
        // would inherit the parent's policy but a stale value
        // estimator, which would briefly mis-predict V(s) and pump
        // bad advantages into the policy gradient.
        let p = self.model.v_w.val().slice([parent..parent+1, 0..HIDDEN, 0..1]);
        self.model.v_w = self.model.v_w.clone().map(|t| {
            t.slice_assign([child..child+1, 0..HIDDEN, 0..1], p)
        });
        let p = self.model.v_b.val().slice([parent..parent+1, 0..1]);
        self.model.v_b = self.model.v_b.clone().map(|t| {
            t.slice_assign([child..child+1, 0..1], p)
        });
    }

    /// See `intelligence_level_1_photo::BrainPoolL1Photo::snapshot`.
    pub fn snapshot(&self) -> PoolSnapshot {
        PoolSnapshot {
            w1:          self.model.w1.val().into_data().into_vec::<f32>().expect("w1 to vec"),
            b1:          self.model.b1.val().into_data().into_vec::<f32>().expect("b1 to vec"),
            w2:          self.model.w2.val().into_data().into_vec::<f32>().expect("w2 to vec"),
            b2:          self.model.b2.val().into_data().into_vec::<f32>().expect("b2 to vec"),
            map:         self.map.clone(),
            prev_state:  self.prev_state.clone(),
            prev_action: self.prev_action.clone(),
            prev_energy: self.prev_energy.clone(),
            has_prev:    self.has_prev.clone(),
            baseline:    self.baseline.clone(),
            in_dim:      IN,
            hidden_dim:  HIDDEN,
            out_dim:     OUT,
        }
    }

    /// See `intelligence_level_1_photo::BrainPoolL1Photo::restore_slot`.
    pub fn restore_slot(&mut self, slot: usize, r: &BrainRestore) -> Result<(), String> {
        if r.w1.len() != IN * HIDDEN { return Err(format!("w1 size {} != {}", r.w1.len(), IN * HIDDEN)); }
        if r.b1.len() != HIDDEN      { return Err(format!("b1 size {} != {}", r.b1.len(), HIDDEN)); }
        if r.w2.len() != HIDDEN * OUT { return Err(format!("w2 size {} != {}", r.w2.len(), HIDDEN * OUT)); }
        if r.b2.len() != OUT         { return Err(format!("b2 size {} != {}", r.b2.len(), OUT)); }
        if r.prev_state.len()  != IN  { return Err(format!("prev_state size {} != {}", r.prev_state.len(), IN)); }
        if r.prev_action.len() != OUT { return Err(format!("prev_action size {} != {}", r.prev_action.len(), OUT)); }

        let device = &self.device;
        let w1_t = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(r.w1.clone(), [1, IN, HIDDEN]), device);
        self.model.w1 = self.model.w1.clone().map(|t| {
            t.slice_assign([slot..slot + 1, 0..IN, 0..HIDDEN], w1_t)
        });
        let b1_t = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(r.b1.clone(), [1, HIDDEN]), device);
        self.model.b1 = self.model.b1.clone().map(|t| {
            t.slice_assign([slot..slot + 1, 0..HIDDEN], b1_t)
        });
        let w2_t = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(r.w2.clone(), [1, HIDDEN, OUT]), device);
        self.model.w2 = self.model.w2.clone().map(|t| {
            t.slice_assign([slot..slot + 1, 0..HIDDEN, 0..OUT], w2_t)
        });
        let b2_t = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(r.b2.clone(), [1, OUT]), device);
        self.model.b2 = self.model.b2.clone().map(|t| {
            t.slice_assign([slot..slot + 1, 0..OUT], b2_t)
        });

        let in_off = slot * IN;
        self.prev_state[in_off .. in_off + IN].copy_from_slice(&r.prev_state);
        let out_off = slot * OUT;
        self.prev_action[out_off .. out_off + OUT].copy_from_slice(&r.prev_action);
        self.prev_energy[slot] = r.prev_energy;
        self.baseline[slot]    = r.baseline;
        self.has_prev[slot]    = r.has_prev;
        Ok(())
    }
}

impl FromWorld for BrainPoolL1Hetero {
    fn from_world(world: &mut World) -> Self {
        let n = world
            .get_resource::<OrganismPoolSize>()
            .map(|r| r.0.max(1))
            .unwrap_or(1);
        let device = CudaDevice::default();
        warmup(&device, n);
        Self::new(device, n)
    }
}


/// Force-compile every kernel the steady-state apply tick will use,
/// so the user's first real frame doesn't pay the JIT cost. Mirrors
/// the apply tick's exact tensor shapes and op chain — both heads of
/// the forward, both loss terms, single combined backward.
fn warmup(device: &CudaDevice, n: usize) {
    let m = PoolMlpL1Hetero::<MyBackend>::new(device, n);
    let mut o: Box<dyn BrainOptL1Hetero> = Box::new(AdamConfig::new().init());

    let prev_state  = Tensor::<MyBackend, 2>::zeros([n, IN],  device);
    let prev_action = Tensor::<MyBackend, 2>::zeros([n, OUT], device);
    let mask        = Tensor::<MyBackend, 2>::zeros([n, 1],   device);
    let target      = Tensor::<MyBackend, 2>::zeros([n, 1],   device);
    let adv_const   = Tensor::<MyBackend, 2>::zeros([n, 1],   device);

    let (mu_prev, v_prev) = m.forward_full(prev_state);

    let value_diff = target - v_prev;
    let value_loss = (value_diff.powf_scalar(2.0) * mask.clone())
        .sum().mul_scalar(0.5).div_scalar(1.0_f32);

    let diff   = prev_action - mu_prev;
    let sum_sq = diff.powf_scalar(2.0).sum_dim(1);
    let scale  = 0.5_f32 / (SIGMA * SIGMA);
    let policy_loss = (sum_sq * adv_const * mask)
        .sum().mul_scalar(scale).div_scalar(1.0_f32);

    let total_loss = policy_loss + value_loss.mul_scalar(VALUE_COEF);
    let g = GradientsParams::from_grads(total_loss.backward(), &m);
    let _ = o.step(LR, m, g);
}


// ── Slot allocation systems ─────────────────────────────────────────────────

pub fn assign_brains_l1_hetero(
    mut pool:     NonSendMut<BrainPoolL1Hetero>,
    new:          Query<(Entity, &Organism, Option<&BrainInheritance>, Option<&BrainRestore>), (
        With<Heterotroph>,
        Without<BrainSlotL1Hetero>,
    )>,
    mut commands: Commands,
) {
    for (e, organism, inheritance, restore) in new.iter() {
        // Filter to Level1 heterotrophs only — Level2 / Level3 go to
        // their own pools.
        if !matches!(organism.intelligence_level, IntelligenceLevel::Level1) { continue; }

        let Some(slot) = pool.free.pop() else { continue };
        let s = slot as usize;

        // See L1 photo's `assign_brains_l1_photo` for the priority
        // order: BrainRestore > BrainInheritance > recycled-slot
        // default. BrainRestore overwrites both weights AND
        // REINFORCE prev_*; the other two reset prev_*.
        let mut restored = false;
        if let Some(r) = restore {
            match pool.restore_slot(s, r) {
                Ok(())   => { restored = true; }
                Err(err) => error!("L1 hetero brain restore failed for {e:?}: {err} — using fresh slot"),
            }
        } else if let Some(BrainInheritance(parent)) = inheritance {
            if let Some(&parent_slot) = pool.map.get(parent) {
                pool.inherit_row(parent_slot as usize, s);
            }
        }

        if !restored {
            pool.has_prev[s]         = false;
            pool.baseline[s]         = 0.0;
            pool.prev_energy[s]      = organism.energy;
            pool.has_prev_prey[s]    = false;
            pool.prev_prey_dist[s]   = 0.0;
            pool.prev_prey_entity[s] = None;
        }

        pool.map.insert(e, slot);
        commands.entity(e).try_insert(BrainSlotL1Hetero(slot));
        commands.entity(e).try_remove::<BrainInheritance>();
        commands.entity(e).try_remove::<BrainRestore>();
    }
}

pub fn free_brains_l1_hetero(
    mut pool:    NonSendMut<BrainPoolL1Hetero>,
    mut removed: RemovedComponents<Heterotroph>,
) {
    for e in removed.read() {
        if let Some(slot) = pool.map.remove(&e) {
            let s = slot as usize;
            pool.has_prev[s]         = false;
            pool.has_prev_prey[s]    = false;
            pool.prev_prey_entity[s] = None;
            pool.free.push(slot);
        }
    }
}


// ── Apply / train tick ──────────────────────────────────────────────────────

/// Per-active-organism scratch entry. Captures everything the
/// reward / apply loops need so we don't re-query the ECS twice.
/// `pub(crate)` because Bevy's system registration sees this type
/// inside the `apply_intelligence_level_1_hetero` signature (via the
/// `Local<Vec<ActiveEntry>>` system parameter), and its visibility
/// must reach the registration site in `behaviour.rs`.
pub(crate) struct ActiveEntry {
    entity:          Entity,
    slot:            u32,
    energy_now:      f32,
    /// Heterotroph forward-axis projected onto the XZ plane and
    /// normalised. Used by the facing-alignment reward.
    forward_xz:      Vec2,
    /// Distance to the nearest in-range prey this tick, or `None`
    /// when no prey is in range.
    cur_prey_dist:   Option<f32>,
    /// Identity of that nearest prey. Compared against the slot's
    /// `prev_prey_entity` to gate the progress reward across
    /// identity flips.
    cur_prey_entity: Option<Entity>,
    /// Unit vector toward that prey on the XZ plane (for facing).
    prey_dir_xz:     Vec2,
}

pub fn apply_intelligence_level_1_hetero(
    time:           Res<Time<Virtual>>,
    world_grid:     Res<WorldModelGrid>,
    mut pool:       NonSendMut<BrainPoolL1Hetero>,
    mut heteros:    Query<(Entity, &mut Organism, &Transform, &BrainSlotL1Hetero), With<Heterotroph>>,
    mut input_buf:  Local<Vec<f32>>,
    mut adv_buf:    Local<Vec<f32>>,
    mut mask_buf:   Local<Vec<f32>>,
    mut active_buf: Local<Vec<ActiveEntry>>,
) {
    if time.is_paused() { return; }

    let n = pool.n;
    input_buf.clear();   input_buf.resize(n * IN, 0.0);
    adv_buf.clear();     adv_buf.resize(n, 0.0);
    mask_buf.clear();    mask_buf.resize(n, 0.0);
    active_buf.clear();

    for (e, organism, transform, slot) in heteros.iter() {
        let s = slot.0 as usize;
        if s >= n { continue; }
        let pos = transform.translation;

        let max_e    = get_max_energy(&organism).max(1.0);
        let energy_n = (organism.energy / max_e).clamp(0.0, 1.0);

        let off = s * IN;
        input_buf[off] = energy_n;
        // World-model rows fill `[off + 1 .. off + 1 + WORLD_MODEL_DIMS]`.
        let wm_slice = &mut input_buf[off + 1 .. off + 1 + WORLD_MODEL_DIMS];
        fill_world_model(&world_grid, pos, wm_slice);

        // Reward-shaping inputs — nearest-prey lookup and the
        // organism's current forward direction. Both are evaluated
        // here (during the input-fill pass) to avoid a second query
        // walk further down.
        let nearest = nearest_prey(&world_grid, pos);
        let (cur_prey_dist, cur_prey_entity, prey_dir_xz) = match nearest {
            Some((rel, dist, ent)) => {
                let dir = Vec2::new(rel.x, rel.z).normalize_or_zero();
                (Some(dist), Some(ent), dir)
            }
            None => (None, None, Vec2::ZERO),
        };
        // `Transform::forward()` returns `-local_z`. But
        // `movement_physics::apply_movement` yaws the organism so
        // local **+Z** points along `movement_direction` — i.e. the
        // organism's "front" is local +Z, so the actual forward
        // axis is the *negation* of `transform.forward()`. Without
        // this negation the facing reward would punish heading
        // toward prey (the organism would learn to flee). The
        // player camera applies the same sign flip in
        // `player_plugin.rs::player_move`.
        let forward = -transform.forward();
        let forward_xz = Vec2::new(forward.x, forward.z).normalize_or_zero();

        active_buf.push(ActiveEntry {
            entity:          e,
            slot:            slot.0,
            energy_now:      organism.energy,
            forward_xz,
            cur_prey_dist,
            cur_prey_entity,
            prey_dir_xz,
        });
    }
    if active_buf.is_empty() { return; }

    // ── Forward CURRENT state ── produces `mu` (the action-sampling
    // mean for THIS tick) and `v` (the value-head estimate of
    // V(s_t), used as the bootstrap target for the previous-tick
    // transition). Both are immediately pulled to CPU because the
    // backward pass uses fresh forward(prev_state) calls instead of
    // these graph nodes.
    let cur_t = Tensor::<MyBackend, 2>::from_data(
        TensorData::new(input_buf.clone(), [n, IN]),
        &pool.device,
    );
    let (mu_cur, v_cur) = pool.model.forward_full(cur_t);
    let mu_data    = mu_cur.into_data().into_vec::<f32>().expect("mu_cur to vec");
    let v_cur_data = v_cur.into_data().into_vec::<f32>().expect("v_cur to vec");

    // ── Shaped reward = energy + progress + facing ───────────────
    // Build the TD target `target[s] = r + γ·V(s_t)` per active
    // slot, plus the mask. Advantage is computed later from a fresh
    // forward(prev_state) so it stays on the autograd graph.
    let mut count = 0.0_f32;
    let mut target_buf = vec![0.0_f32; n];
    for entry in active_buf.iter() {
        let s = entry.slot as usize;
        if !pool.has_prev[s] { continue; }

        let energy_delta = (entry.energy_now - pool.prev_energy[s])
            .clamp(-REWARD_CLAMP, REWARD_CLAMP);

        // Progress is meaningful ONLY when the nearest-prey
        // identity is the SAME this tick as last tick. When it
        // flips (a different photo became closest, or the previous
        // one was eaten / left the radius), `prev_dist` and
        // `cur_dist` are measuring distances to two different
        // organisms — their delta would credit-assign whatever
        // action the agent just took to a frame the agent didn't
        // cause. Zeroing progress on identity flips drops the
        // signal for one tick but is far better than spurious
        // reinforcement.
        let progress = match (
            pool.has_prev_prey[s],
            pool.prev_prey_entity[s],
            entry.cur_prey_dist,
            entry.cur_prey_entity,
        ) {
            (true, Some(prev_e), Some(cur_dist), Some(cur_e)) if prev_e == cur_e => {
                let raw = (pool.prev_prey_dist[s] - cur_dist) / WORLD_MODEL_RADIUS;
                raw.clamp(-1.0, 1.0)
            }
            _ => 0.0,
        };

        // Facing — same identity gate as progress. If the nearest
        // prey changed between ticks (different entity), the
        // organism's `forward_xz` was set by the previous action
        // *aiming at a different prey*, so attributing facing
        // credit against the new prey would be a spurious signal.
        // Also requires a prey to be in range this tick.
        let facing = match (
            pool.has_prev_prey[s],
            pool.prev_prey_entity[s],
            entry.cur_prey_entity,
        ) {
            (true, Some(prev_e), Some(cur_e)) if prev_e == cur_e => {
                entry.forward_xz.dot(entry.prey_dir_xz)
            }
            _ => 0.0,
        };

        let r = W_ENERGY   * energy_delta
              + W_PROGRESS * progress
              + W_FACING   * facing;

        target_buf[s] = r + GAMMA * v_cur_data[s];
        mask_buf[s]   = 1.0;
        count        += 1.0;
    }

    if count > 0.0 {
        let prev_state_t = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(pool.prev_state.clone(), [n, IN]),
            &pool.device,
        );
        let prev_action_t = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(pool.prev_action.clone(), [n, OUT]),
            &pool.device,
        );
        let target_t = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(target_buf.clone(), [n, 1]),
            &pool.device,
        );
        let mask_t = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(mask_buf.clone(), [n, 1]),
            &pool.device,
        );

        // Fresh forward over the previous state — both heads' outputs
        // are needed (mu_prev for policy loss, v_prev for value
        // loss). Sharing the trunk means a single backward
        // accumulates gradients on `w1`/`b1` from BOTH terms.
        let (mu_prev, v_prev) = pool.model.forward_full(prev_state_t);

        // ── Value loss = mean (target - V(s_{t-1}))² over active ─
        // Gradient flows through `v_prev` (and the trunk via the
        // shared `h.clone()`); `target_t` is a leaf so no gradient
        // flows back into the bootstrap V(s_t) we read from CPU.
        let value_diff = target_t.clone() - v_prev.clone();
        let value_loss = (value_diff.powf_scalar(2.0) * mask_t.clone())
            .sum().mul_scalar(0.5).div_scalar(count);

        // ── Policy loss = mean (a - μ_prev)² · adv / (2σ²) ─────
        // For policy gradient, `adv` MUST be a constant (no gradient
        // flow through it back into the value head — that would
        // mean the policy loss tries to drag V down whenever it
        // also pushes μ toward the action). We materialise `adv` as
        // numeric values on the CPU and re-tensor it so the
        // resulting `adv_const_t` is a leaf in the autograd graph.
        // The single ~8 KB GPU→CPU sync per tick is negligible.
        let v_prev_data = v_prev.clone().into_data().into_vec::<f32>().expect("v_prev to vec");
        let mut adv_buf_local = vec![0.0_f32; n];
        for s in 0..n {
            if mask_buf[s] > 0.5 {
                adv_buf_local[s] = target_buf[s] - v_prev_data[s];
            }
        }
        // Stash a copy into the system-local advantage scratch buf
        // so its size stays correct (used elsewhere as a sanity
        // mirror; the autograd loss only reads `adv_const_t`).
        adv_buf.clone_from(&adv_buf_local);

        let adv_const_t = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(adv_buf_local, [n, 1]),
            &pool.device,
        );

        let diff   = prev_action_t - mu_prev;
        let sum_sq = diff.powf_scalar(2.0).sum_dim(1);
        let scale  = 0.5_f32 / (SIGMA * SIGMA);
        let policy_loss = (sum_sq * adv_const_t * mask_t)
            .sum().mul_scalar(scale).div_scalar(count);

        let total_loss = policy_loss + value_loss.mul_scalar(VALUE_COEF);

        let cm = pool.model.clone();
        let gp = GradientsParams::from_grads(total_loss.backward(), &pool.model);
        pool.model = pool.opt.step(LR, cm, gp);
    }

    let mut rng = rand::rng();
    for entry in active_buf.iter() {
        let slot = entry.slot;
        let s    = slot as usize;
        let off  = s * OUT;

        let mut action = [0.0_f32; OUT];
        for i in 0..OUT {
            action[i] = mu_data[off + i] + SIGMA * gaussian_noise(&mut rng);
        }

        let speed_a = action[0].clamp(-1.0, 1.0);
        // OUT == 3: action[0] = speed, action[1] = dir.x, action[2] = dir.z.
        // Heterotroph motion is XZ-planar so dir.y is fixed at 0 (no
        // policy dimension wasted on it).
        let dir     = Vec3::new(action[1], 0.0, action[2]);
        let Ok((_, mut org, _, _)) = heteros.get_mut(entry.entity) else { continue };
        if dir.length_squared() > 0.01 { org.movement_direction = dir.normalize(); }
        // Speed mapping: `max(0, speed_a) · MAX_SPEED`. The lower
        // half of the [-1, 1] action range collapses to zero speed,
        // so the random-init policy (μ ≈ 0, σ = 0.5) sits at zero
        // speed about 50% of the time. This makes "conserve energy
        // by staying still" trivially reachable, replacing the old
        // mapping `((speed_a + 1)/2)·MAX` which forced the
        // un-trained policy to permanently sprint at MAX/2.
        org.movement_speed = speed_a.max(0.0) * MAX_SPEED;

        let in_off = s * IN;
        for i in 0..IN  { pool.prev_state [in_off + i] = input_buf[in_off + i]; }
        // Index 0 stores the CLAMPED speed action, not the raw
        // (post-noise) sample. With σ=0.5, ~30% of raw samples land
        // outside [-1, 1] but only the clamped value is what the
        // simulator actually executed. Without the clamp the policy
        // gradient would push μ toward unreachable values and the
        // mean speed would saturate — visible as the agent
        // permanently running near max-speed regardless of state.
        pool.prev_action[off + 0] = speed_a;
        for i in 1..OUT { pool.prev_action[off + i] = action[i]; }
        pool.prev_energy[s] = entry.energy_now;
        pool.has_prev[s]    = true;

        // Persist this tick's nearest-prey distance AND identity
        // for next tick's progress-reward calc. Reset has_prev_prey
        // (and the entity slot) when prey left the radius so the
        // next tick's progress is 0 instead of a stale delta.
        match (entry.cur_prey_dist, entry.cur_prey_entity) {
            (Some(d), Some(e)) => {
                pool.prev_prey_dist[s]   = d;
                pool.prev_prey_entity[s] = Some(e);
                pool.has_prev_prey[s]    = true;
            }
            _ => {
                pool.has_prev_prey[s]    = false;
                pool.prev_prey_entity[s] = None;
            }
        }
    }
}
