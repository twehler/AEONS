// Intelligence Level 1 — Herbivore RL substrate v2.1 (per-organism A2C).
//
// Restores per-organism weight isolation from the shared-trunk MVP.
// Every herbivore now has its own copy of the entire PolicyNet
// (backbone + actor + critic). Inheritance carries the parent's
// trained weights to its offspring's slot at reproduction; weights
// are also persisted to `.colony` save files (AEONS004 format).
//
// Architecture (unchanged from the shared MVP):
//   * Body-frame state observation — rotation-invariant geometry.
//   * Angle-based delta action — no absolute-bearing population drift.
//   * n-step A2C with learned per-output σ.
//   * Decoupled dopamine (observation only) + Δtarget_distance shaping
//     for the reward signal.
//
// What changed from the MVP:
//   * `PolicyNet` weights are now 3D `[N, IN, HIDDEN]` (and `[N, DIM]`
//     for biases). Each row is one organism's private brain.
//   * Forward / training passes use the per-N batched matmul pattern
//     from the L2/L3 pools — `[N, T, IN] @ [N, IN, HIDDEN] = [N, T, HIDDEN]`.
//   * `inherit_row` propagates a parent's row to its offspring's
//     slot at reproduction.
//   * `extract_slot` / `restore_slot` round-trip a single slot's
//     weights + REINFORCE state through `BrainRestoreHerbivore1`
//     for the colony save/load pipeline.
//
// Deferred (Phase 2+):
//   * Experience replay buffer + V-trace.
//   * Random Network Distillation (intrinsic curiosity).
//   * Per-organism gene vector concatenated to the observation.

use bevy::prelude::*;
use burn::module::{Initializer, Module, Param};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use burn_cuda::CudaDevice;
use std::collections::{HashMap, VecDeque};
use std::f32::consts::PI;

use crate::colony::{Carnivore, IntelligenceLevel, Organism, Heterotroph};
use crate::energy::get_max_energy;
use crate::rl_helpers::{BrainInheritance, MyBackend, gaussian_noise};
use crate::sensory::SENSORY_RADIUS;
use crate::simulation_settings::OrganismPoolSize;
use crate::world_model::{WorldModelGrid, WORLD_MODEL_RADIUS, nearest_prey};


// ── Architecture constants ──────────────────────────────────────────────────

pub const STATE_DIM:    usize = 16;
pub const HIDDEN:       usize = 64;
pub const HEAD_HIDDEN:  usize = 16;
pub const ACTOR_OUT:    usize = 4;
pub const ACTION_DIM:   usize = 2;
pub const ROLLOUT_LEN:  usize = 16;

const GAMMA:            f32 = 0.98;
const VALUE_COEF:       f32 = 0.5;
/// Entropy bonus is disabled — the May-2026 brain-probe analysis
/// (datasets/extended_herbivore_behaviour_analysis.R) showed σ
/// pinned at exp(0) = 1.0 with `ENTROPY_COEF = 0.01`, and the
/// supervised bootstrap loss (`BOOTSTRAP_STEPS`) now drives the
/// policy out of the small-init regime. Letting σ collapse
/// organically as advantages settle is the cleaner regime.
const ENTROPY_COEF:     f32 = 0.0;
const LR:               f64 = 3e-4;
const LOG_SIGMA_MIN:    f32 = -5.0;
const LOG_SIGMA_MAX:    f32 =  2.0;
/// Constant offset added to the network's raw ls_speed/ls_angle
/// output BEFORE clamping. The actor's last-layer bias starts at
/// zero so the network's contribution is zero at init, but the
/// effective log σ starts at -1.2 → σ ≈ 0.30. Tighter sampling
/// noise means μ matters more for behaviour, which means the
/// reward-gradient feedback loop has a chance of escaping the
/// μ ≈ 0 degenerate equilibrium identified in the May-2026
/// brain-probe deep-dive.
const INITIAL_LOG_SIGMA: f32 = -1.2;
const MAX_SPEED:        f32 = 40.0;
/// Reward weights — scaled 50× from the previous (W_EAT=2,
/// W_PROGRESS=1) so advantages emerge from the Adam noise floor.
/// The May-2026 datasets confirmed actor output weights only
/// drifted +0.4% across 30 min of training because advantage
/// magnitudes were ≈0.04 — too small to overcome Adam's adaptive
/// scaling. Multiplying both channels by 50 brings typical
/// advantages into the 1-2 range where Adam's directional signal
/// dominates the gradient RMS denominator.
///
///   reward = W_EAT · (Δpredations)
///          + W_PROGRESS · clamp((prev_td − td) / SENSORY_RADIUS, -1, 1)
///          + W_ORACLE · hunger · cos(movement_direction, dir_to_nearest_prey)
///
/// The third (oracle-alignment) channel turns the same body-frame
/// geometry the supervised bootstrap uses into a permanent reward
/// signal. Scaled by `hunger` ∈ [0, 1] so the pressure to chase prey
/// fades as the agent fills up. Cosine similarity in [-1, 1] gives
/// signed feedback — moving away from prey actively *costs* reward
/// when hungry, not just zero — which gives the policy gradient a
/// clear direction to climb. Survives the BOOTSTRAP_STEPS expiry, so
/// the agent keeps learning the chase even after the MSE supervision
/// is gone.
///
/// Dopamine + hunger are still STATE OBSERVATIONS fed to the input
/// layer — the brain can condition behaviour on them. Hunger now ALSO
/// gates the oracle reward channel.
const W_EAT:      f32 = 100.0;
const W_PROGRESS: f32 = 50.0;
const W_ORACLE:   f32 = 50.0;

/// Number of training steps over which the supervised "chase
/// nearest prey" bootstrap loss decays from 1.0 to 0.0. The loss
/// pushes μ_angle toward atanh(atan2(body_right, body_fwd) / π) and
/// μ_speed toward +2 (post-tanh ≈ +0.96 → speed ≈ MAX_SPEED) on
/// every rollout step where `has_photo == 1`. Once `step_counter`
/// reaches this value the bootstrap loss contributes nothing and
/// the policy is driven entirely by RL.
const BOOTSTRAP_STEPS:   u64 = 200;
/// Pre-tanh target for μ_speed during the supervised phase.
/// tanh(2.0) ≈ 0.964, which post-affine maps to ≈ 0.982 × MAX_SPEED.
const BOOTSTRAP_MU_SPEED_TARGET: f32 = 2.0;


// ── Slot marker ─────────────────────────────────────────────────────────────

/// Component placed on every herbivore enroled in this pool. Holds the
/// row index into the per-slot tracking vectors AND the row of weight
/// tensors that this organism's brain occupies.
#[derive(Component, Clone, Copy)]
pub struct BrainSlotHerbivore1(pub u32);


// ── PolicyNet — per-organism rows (backbone + actor + critic) ──────────────

#[derive(Module, Debug)]
pub struct PolicyNet<B: Backend> {
    // Backbone — each row `s` is slot s's private MLP.
    bk_w1: Param<Tensor<B, 3>>,    // [N, STATE_DIM, HIDDEN]
    bk_b1: Param<Tensor<B, 2>>,    // [N, HIDDEN]
    bk_w2: Param<Tensor<B, 3>>,    // [N, HIDDEN, HIDDEN]
    bk_b2: Param<Tensor<B, 2>>,    // [N, HIDDEN]
    // Actor head
    a_w1:  Param<Tensor<B, 3>>,    // [N, HIDDEN, HEAD_HIDDEN]
    a_b1:  Param<Tensor<B, 2>>,    // [N, HEAD_HIDDEN]
    a_w2:  Param<Tensor<B, 3>>,    // [N, HEAD_HIDDEN, ACTOR_OUT]
    a_b2:  Param<Tensor<B, 2>>,    // [N, ACTOR_OUT]
    // Critic head
    c_w1:  Param<Tensor<B, 3>>,    // [N, HIDDEN, HEAD_HIDDEN]
    c_b1:  Param<Tensor<B, 2>>,    // [N, HEAD_HIDDEN]
    c_w2:  Param<Tensor<B, 3>>,    // [N, HEAD_HIDDEN, 1]
    c_b2:  Param<Tensor<B, 2>>,    // [N, 1]
}

impl<B: Backend> PolicyNet<B> {
    fn new(device: &B::Device, n: usize) -> Self {
        // Kaiming-Uniform bound for ReLU layers: √(6/fan_in).
        let bound = |fan_in: usize| -> f64 { (6.0 / fan_in as f64).sqrt() };
        let hidden_from_state  = Initializer::Uniform {
            min: -bound(STATE_DIM), max: bound(STATE_DIM),
        };
        let hidden_from_hidden = Initializer::Uniform {
            min: -bound(HIDDEN), max: bound(HIDDEN),
        };
        let head_from_hidden   = Initializer::Uniform {
            min: -bound(HIDDEN), max: bound(HIDDEN),
        };
        // Output heads start near zero — the initial policy is
        // "no preference" and gets shaped by gradient.
        let actor_out  = Initializer::Uniform { min: -0.01, max: 0.01 };
        let critic_out = Initializer::Uniform { min: -0.01, max: 0.01 };
        let z = Initializer::Zeros;
        Self {
            bk_w1: hidden_from_state.init ([n, STATE_DIM, HIDDEN],     device),
            bk_b1: z.init                ([n, HIDDEN],                device),
            bk_w2: hidden_from_hidden.init([n, HIDDEN, HIDDEN],        device),
            bk_b2: z.init                ([n, HIDDEN],                device),
            a_w1:  head_from_hidden.init  ([n, HIDDEN, HEAD_HIDDEN],   device),
            a_b1:  z.init                ([n, HEAD_HIDDEN],           device),
            a_w2:  actor_out.init         ([n, HEAD_HIDDEN, ACTOR_OUT], device),
            a_b2:  z.init                ([n, ACTOR_OUT],             device),
            c_w1:  head_from_hidden.init  ([n, HIDDEN, HEAD_HIDDEN],   device),
            c_b1:  z.init                ([n, HEAD_HIDDEN],           device),
            c_w2:  critic_out.init        ([n, HEAD_HIDDEN, 1],        device),
            c_b2:  z.init                ([n, 1],                    device),
        }
    }

    /// Per-N batched forward. `x` shape `[N, T, STATE_DIM]` where T is
    /// the time dim (1 for action sampling, ROLLOUT_LEN for training).
    /// Each row `s ∈ [0,N)` consults slot `s`'s private weight rows;
    /// outputs preserve the leading `(N, T)` shape.
    fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        // bk_b1 is `[N, HIDDEN]` — unsqueeze to `[N, 1, HIDDEN]` so it
        // broadcasts cleanly across the T axis of `[N, T, HIDDEN]`.
        let h = x.matmul(self.bk_w1.val()) + self.bk_b1.val().unsqueeze_dim::<3>(1);
        let h = burn::tensor::activation::relu(h);
        let h = h.matmul(self.bk_w2.val()) + self.bk_b2.val().unsqueeze_dim::<3>(1);
        let h = burn::tensor::activation::relu(h);

        let a = h.clone().matmul(self.a_w1.val()) + self.a_b1.val().unsqueeze_dim::<3>(1);
        let a = burn::tensor::activation::relu(a);
        let a = a.matmul(self.a_w2.val()) + self.a_b2.val().unsqueeze_dim::<3>(1);

        let v = h.matmul(self.c_w1.val()) + self.c_b1.val().unsqueeze_dim::<3>(1);
        let v = burn::tensor::activation::relu(v);
        let v = v.matmul(self.c_w2.val()) + self.c_b2.val().unsqueeze_dim::<3>(1);

        (a, v)
    }
}


// ── Brain inheritance + restore helpers on the model ───────────────────────
//
// `inherit_row` and `restore_slot` both mutate one row of every weight
// tensor while leaving the rest of the population untouched. Burn's
// `Param` wraps tensors that are routed through Adam; the
// `.clone().map(|t| t.slice_assign(...))` pattern threads the gradient
// graph correctly. Adam moment buffers attached to the param are
// shape-matched and propagate automatically.

impl PolicyNet<MyBackend> {
    /// Copy parent's row to child's row across every weight tensor.
    /// Called when a freshly-assigned slot has a `BrainInheritance`
    /// marker pointing at a still-living parent organism.
    pub fn inherit_row(&mut self, parent: usize, child: usize) {
        macro_rules! copy_3d {
            ($field:ident, $d1:expr, $d2:expr) => {{
                let p = self.$field.val().slice([parent..parent+1, 0..$d1, 0..$d2]);
                self.$field = self.$field.clone().map(|t| {
                    t.slice_assign([child..child+1, 0..$d1, 0..$d2], p)
                });
            }};
        }
        macro_rules! copy_2d {
            ($field:ident, $d:expr) => {{
                let p = self.$field.val().slice([parent..parent+1, 0..$d]);
                self.$field = self.$field.clone().map(|t| {
                    t.slice_assign([child..child+1, 0..$d], p)
                });
            }};
        }
        copy_3d!(bk_w1, STATE_DIM,   HIDDEN);
        copy_2d!(bk_b1, HIDDEN);
        copy_3d!(bk_w2, HIDDEN,      HIDDEN);
        copy_2d!(bk_b2, HIDDEN);
        copy_3d!(a_w1,  HIDDEN,      HEAD_HIDDEN);
        copy_2d!(a_b1,  HEAD_HIDDEN);
        copy_3d!(a_w2,  HEAD_HIDDEN, ACTOR_OUT);
        copy_2d!(a_b2,  ACTOR_OUT);
        copy_3d!(c_w1,  HIDDEN,      HEAD_HIDDEN);
        copy_2d!(c_b1,  HEAD_HIDDEN);
        copy_3d!(c_w2,  HEAD_HIDDEN, 1);
        copy_2d!(c_b2,  1);
    }
}


trait BrainOptHerbivore1 {
    fn step(
        &mut self,
        lr: f64,
        m:  PolicyNet<MyBackend>,
        g:  GradientsParams,
    ) -> PolicyNet<MyBackend>;
}

impl<O: Optimizer<PolicyNet<MyBackend>, MyBackend>> BrainOptHerbivore1 for O {
    fn step(
        &mut self,
        lr: f64,
        m:  PolicyNet<MyBackend>,
        g:  GradientsParams,
    ) -> PolicyNet<MyBackend> {
        Optimizer::step(self, lr, m, g)
    }
}


// ── Save/load payload ──────────────────────────────────────────────────────
//
// One value per organism — flat per-slot weights + REINFORCE state.
// Lives as a Bevy `Component` so the `.colony` load path can attach it
// to a freshly-spawned entity and let `assign_brains_herbivore_1` pick
// it up on the next PreUpdate, restoring the slot to the saved state
// before training resumes.

#[derive(Component, Clone, Debug)]
pub struct BrainRestoreHerbivore1 {
    pub bk_w1: Vec<f32>, pub bk_b1: Vec<f32>,
    pub bk_w2: Vec<f32>, pub bk_b2: Vec<f32>,
    pub a_w1:  Vec<f32>, pub a_b1:  Vec<f32>,
    pub a_w2:  Vec<f32>, pub a_b2:  Vec<f32>,
    pub c_w1:  Vec<f32>, pub c_b1:  Vec<f32>,
    pub c_w2:  Vec<f32>, pub c_b2:  Vec<f32>,
    pub prev_state:           Vec<f32>,   // [STATE_DIM]
    pub prev_action:          Vec<f32>,   // [ACTION_DIM]
    pub prev_dopamine:        f32,
    pub prev_target_distance: f32,
}


// ── Brain-restore binary serialisation ─────────────────────────────────────
//
// Defined HERE (not duplicated at every call site) so the .colony and
// .species file formats share a single binary layout for the brain
// payload. The caller is responsible for writing/reading the
// `brain_present` byte that surrounds the block — `encode` /
// `decode` deal only with the 12 weight tensors + REINFORCE state.

/// Serialise a `BrainRestoreHerbivore1` into `buf`. Each tensor is
/// prefixed with its `u32` length so the decoder can validate shape
/// against the current architecture.
pub fn encode_brain_restore(buf: &mut Vec<u8>, b: &BrainRestoreHerbivore1) {
    let write_vec = |buf: &mut Vec<u8>, v: &Vec<f32>| {
        buf.extend_from_slice(&(v.len() as u32).to_le_bytes());
        for &x in v { buf.extend_from_slice(&x.to_le_bytes()); }
    };
    write_vec(buf, &b.bk_w1); write_vec(buf, &b.bk_b1);
    write_vec(buf, &b.bk_w2); write_vec(buf, &b.bk_b2);
    write_vec(buf, &b.a_w1);  write_vec(buf, &b.a_b1);
    write_vec(buf, &b.a_w2);  write_vec(buf, &b.a_b2);
    write_vec(buf, &b.c_w1);  write_vec(buf, &b.c_b1);
    write_vec(buf, &b.c_w2);  write_vec(buf, &b.c_b2);
    write_vec(buf, &b.prev_state);
    write_vec(buf, &b.prev_action);
    buf.extend_from_slice(&b.prev_dopamine.to_le_bytes());
    buf.extend_from_slice(&b.prev_target_distance.to_le_bytes());
}

/// Deserialise a `BrainRestoreHerbivore1` from `bytes[*c..]`,
/// advancing `*c`. Hard-errors on shape mismatch — every length
/// prefix must equal the dimension demanded by the current
/// architecture (`STATE_DIM`, `HIDDEN`, etc.).
pub fn decode_brain_restore(
    bytes: &[u8],
    c:     &mut usize,
) -> std::io::Result<BrainRestoreHerbivore1> {
    fn read_u32(bytes: &[u8], c: &mut usize) -> std::io::Result<u32> {
        if *c + 4 > bytes.len() { return Err(std::io::Error::other("brain truncated (u32)")); }
        let v = u32::from_le_bytes(bytes[*c..*c+4].try_into().unwrap());
        *c += 4;
        Ok(v)
    }
    fn read_f32(bytes: &[u8], c: &mut usize) -> std::io::Result<f32> {
        if *c + 4 > bytes.len() { return Err(std::io::Error::other("brain truncated (f32)")); }
        let v = f32::from_le_bytes(bytes[*c..*c+4].try_into().unwrap());
        *c += 4;
        Ok(v)
    }
    let read_vec_checked = |c: &mut usize, expected: usize, label: &str|
        -> std::io::Result<Vec<f32>>
    {
        let n = read_u32(bytes, c)? as usize;
        if n != expected {
            return Err(std::io::Error::other(format!(
                "brain shape mismatch on {label}: saved {n}, expected {expected}"
            )));
        }
        let mut v = Vec::with_capacity(n);
        for _ in 0..n { v.push(read_f32(bytes, c)?); }
        Ok(v)
    };
    let bk_w1 = read_vec_checked(c, STATE_DIM   * HIDDEN,      "bk_w1")?;
    let bk_b1 = read_vec_checked(c, HIDDEN,                   "bk_b1")?;
    let bk_w2 = read_vec_checked(c, HIDDEN      * HIDDEN,     "bk_w2")?;
    let bk_b2 = read_vec_checked(c, HIDDEN,                   "bk_b2")?;
    let a_w1  = read_vec_checked(c, HIDDEN      * HEAD_HIDDEN, "a_w1" )?;
    let a_b1  = read_vec_checked(c, HEAD_HIDDEN,              "a_b1" )?;
    let a_w2  = read_vec_checked(c, HEAD_HIDDEN * ACTOR_OUT,   "a_w2" )?;
    let a_b2  = read_vec_checked(c, ACTOR_OUT,                "a_b2" )?;
    let c_w1  = read_vec_checked(c, HIDDEN      * HEAD_HIDDEN, "c_w1" )?;
    let c_b1  = read_vec_checked(c, HEAD_HIDDEN,              "c_b1" )?;
    let c_w2  = read_vec_checked(c, HEAD_HIDDEN * 1,           "c_w2" )?;
    let c_b2  = read_vec_checked(c, 1,                        "c_b2" )?;
    let prev_state  = read_vec_checked(c, STATE_DIM,           "prev_state")?;
    let prev_action = read_vec_checked(c, ACTION_DIM,          "prev_action")?;
    let prev_dopamine        = read_f32(bytes, c)?;
    let prev_target_distance = read_f32(bytes, c)?;
    Ok(BrainRestoreHerbivore1 {
        bk_w1, bk_b1, bk_w2, bk_b2,
        a_w1, a_b1, a_w2, a_b2,
        c_w1, c_b1, c_w2, c_b2,
        prev_state, prev_action,
        prev_dopamine, prev_target_distance,
    })
}


// ── Pool resource ───────────────────────────────────────────────────────────

pub struct BrainPoolHerbivore1 {
    pub device: CudaDevice,
    model:      PolicyNet<MyBackend>,
    opt:        Box<dyn BrainOptHerbivore1>,

    n:          usize,
    free:       Vec<u32>,
    pub map:    HashMap<Entity, u32>,

    // Per-slot rollout buffers.
    rollout_states:  Vec<f32>,   // [n * ROLLOUT_LEN * STATE_DIM]
    rollout_actions: Vec<f32>,   // [n * ROLLOUT_LEN * ACTION_DIM]
    rollout_rewards: Vec<f32>,   // [n * ROLLOUT_LEN]

    rollout_active: Vec<bool>,

    prev_dopamine:        Vec<f32>,
    prev_target_distance: Vec<f32>,
    /// Last brain tick's `Organism::predations` counter — sampled
    /// at the end of each apply tick. The reward channel uses the
    /// per-tick delta `(organism.predations - prev_predations[s])`
    /// as a sharp eat-event spike that doesn't decay between ticks.
    /// Initialised to 0 on slot reset; saturating-u8 wrapping is
    /// harmless because the delta is computed with signed
    /// subtraction in i32 space.
    prev_predations:      Vec<u8>,
    prev_action:          Vec<f32>,        // [n * ACTION_DIM]
    /// Captured at apply time so save/load can round-trip the slot.
    prev_state:           Vec<f32>,        // [n * STATE_DIM]

    // ── Diagnostic telemetry (not part of brain state; recorded for
    // dataset-export inspection only). All reset on slot recycle.
    /// Per-slot ring buffer of the last 64 rewards. `mean_reward_64`
    /// is computed as the unconditional mean of all 64 entries —
    /// freshly-reset slots show zero until they've been around long
    /// enough for the ring to fill.
    recent_rewards:       Vec<f32>,        // [n * 64]
    recent_reward_head:   Vec<u8>,         // [n]
    /// The two reward components from the most recent tick, kept
    /// separate so the export can tell whether the eat-event channel
    /// is firing OR whether the progress channel is firing for each
    /// organism.
    last_eat:             Vec<f32>,        // [n] — W_EAT · Δpredations
    last_progress:        Vec<f32>,        // [n] — W_PROGRESS · Δtd
    last_oracle:          Vec<f32>,        // [n] — W_ORACLE · hunger · cos(move, prey_dir)

    /// Ring buffer of the last `TRAINING_HISTORY_CAP` training
    /// steps. Each entry is the scalar losses + return statistics
    /// taken at the end of one `train_step` call. Exported as a
    /// separate `training_stats_*.csv` file alongside the organism
    /// dataset.
    training_history:     VecDeque<TrainingStep>,
    /// Monotonically increasing counter of completed training
    /// steps. Survives slot recycling — counts substrate-wide
    /// gradient updates, not per-organism.
    step_counter:         u64,

    global_step: u32,
}

impl BrainPoolHerbivore1 {
    fn new(device: CudaDevice, n: usize) -> Self {
        Self {
            model:      PolicyNet::<MyBackend>::new(&device, n),
            opt:        Box::new(AdamConfig::new().init()),
            device,
            n,
            free:       (0..n as u32).rev().collect(),
            map:        HashMap::with_capacity(n),
            rollout_states:  vec![0.0; n * ROLLOUT_LEN * STATE_DIM],
            rollout_actions: vec![0.0; n * ROLLOUT_LEN * ACTION_DIM],
            rollout_rewards: vec![0.0; n * ROLLOUT_LEN],
            rollout_active:  vec![false; n],
            prev_dopamine:        vec![0.0; n],
            prev_target_distance: vec![SENSORY_RADIUS; n],
            prev_predations:      vec![0u8; n],
            prev_action:          vec![0.0; n * ACTION_DIM],
            prev_state:           vec![0.0; n * STATE_DIM],
            recent_rewards:       vec![0.0; n * 64],
            recent_reward_head:   vec![0u8; n],
            last_eat:             vec![0.0; n],
            last_progress:        vec![0.0; n],
            last_oracle:          vec![0.0; n],
            training_history:     VecDeque::with_capacity(TRAINING_HISTORY_CAP),
            step_counter:         0,
            global_step: 0,
        }
    }
    pub fn n(&self) -> usize { self.n }

    fn reset_slot_state(&mut self, s: usize) {
        for i in 0..(ROLLOUT_LEN * STATE_DIM)  { self.rollout_states  [s * ROLLOUT_LEN * STATE_DIM  + i] = 0.0; }
        for i in 0..(ROLLOUT_LEN * ACTION_DIM) { self.rollout_actions [s * ROLLOUT_LEN * ACTION_DIM + i] = 0.0; }
        for i in 0..ROLLOUT_LEN                { self.rollout_rewards [s * ROLLOUT_LEN              + i] = 0.0; }
        for i in 0..ACTION_DIM                 { self.prev_action     [s * ACTION_DIM              + i] = 0.0; }
        for i in 0..STATE_DIM                  { self.prev_state      [s * STATE_DIM               + i] = 0.0; }
        self.prev_dopamine[s]        = 0.0;
        self.prev_target_distance[s] = SENSORY_RADIUS;
        self.prev_predations[s]      = 0;
        self.rollout_active[s]       = false;
        for i in 0..64 { self.recent_rewards[s * 64 + i] = 0.0; }
        self.recent_reward_head[s]   = 0;
        self.last_eat[s]             = 0.0;
        self.last_progress[s]        = 0.0;
        self.last_oracle[s]          = 0.0;
    }

    /// Snapshot one slot's weights + REINFORCE state into a
    /// `BrainRestoreHerbivore1`. The weight tensors are sliced from
    /// the GPU and pulled to CPU one row at a time — cheap per
    /// organism, dominated by the GPU→CPU sync cost. Called from
    /// `colony.rs::save_colony_system` for each living herbivore.
    pub fn extract_slot(&self, slot: u32) -> BrainRestoreHerbivore1 {
        let s = slot as usize;
        let pull_3d = |t: &Tensor<MyBackend, 3>, d1: usize, d2: usize| {
            t.clone().slice([s..s+1, 0..d1, 0..d2])
             .into_data().into_vec::<f32>().expect("3d slot pull")
        };
        let pull_2d = |t: &Tensor<MyBackend, 2>, d: usize| {
            t.clone().slice([s..s+1, 0..d])
             .into_data().into_vec::<f32>().expect("2d slot pull")
        };
        BrainRestoreHerbivore1 {
            bk_w1: pull_3d(&self.model.bk_w1.val(), STATE_DIM,   HIDDEN),
            bk_b1: pull_2d(&self.model.bk_b1.val(), HIDDEN),
            bk_w2: pull_3d(&self.model.bk_w2.val(), HIDDEN,      HIDDEN),
            bk_b2: pull_2d(&self.model.bk_b2.val(), HIDDEN),
            a_w1:  pull_3d(&self.model.a_w1.val(),  HIDDEN,      HEAD_HIDDEN),
            a_b1:  pull_2d(&self.model.a_b1.val(),  HEAD_HIDDEN),
            a_w2:  pull_3d(&self.model.a_w2.val(),  HEAD_HIDDEN, ACTOR_OUT),
            a_b2:  pull_2d(&self.model.a_b2.val(),  ACTOR_OUT),
            c_w1:  pull_3d(&self.model.c_w1.val(),  HIDDEN,      HEAD_HIDDEN),
            c_b1:  pull_2d(&self.model.c_b1.val(),  HEAD_HIDDEN),
            c_w2:  pull_3d(&self.model.c_w2.val(),  HEAD_HIDDEN, 1),
            c_b2:  pull_2d(&self.model.c_b2.val(),  1),
            prev_state:  self.prev_state [s * STATE_DIM  .. (s+1) * STATE_DIM ].to_vec(),
            prev_action: self.prev_action[s * ACTION_DIM .. (s+1) * ACTION_DIM].to_vec(),
            prev_dopamine:        self.prev_dopamine[s],
            prev_target_distance: self.prev_target_distance[s],
        }
    }

    /// Restore one slot to a saved `BrainRestoreHerbivore1`. Hard-
    /// errors on shape mismatch (the load path in `colony.rs` checks
    /// shapes BEFORE attaching the component, so this is the second
    /// line of defence — a runtime invariant violation, not an
    /// expected error). Adam moments aren't restored — they reset to
    /// zero and re-adapt over the next ~few brain ticks.
    pub fn restore_slot(&mut self, slot: u32, r: &BrainRestoreHerbivore1) {
        let s = slot as usize;
        let device = &self.device;

        macro_rules! restore_3d {
            ($field:ident, $vec:expr, $d1:expr, $d2:expr) => {{
                let row = Tensor::<MyBackend, 3>::from_data(
                    TensorData::new($vec.clone(), [1, $d1, $d2]),
                    device,
                );
                self.model.$field = self.model.$field.clone().map(|t| {
                    t.slice_assign([s..s+1, 0..$d1, 0..$d2], row)
                });
            }};
        }
        macro_rules! restore_2d {
            ($field:ident, $vec:expr, $d:expr) => {{
                let row = Tensor::<MyBackend, 2>::from_data(
                    TensorData::new($vec.clone(), [1, $d]),
                    device,
                );
                self.model.$field = self.model.$field.clone().map(|t| {
                    t.slice_assign([s..s+1, 0..$d], row)
                });
            }};
        }
        restore_3d!(bk_w1, r.bk_w1, STATE_DIM,   HIDDEN);
        restore_2d!(bk_b1, r.bk_b1, HIDDEN);
        restore_3d!(bk_w2, r.bk_w2, HIDDEN,      HIDDEN);
        restore_2d!(bk_b2, r.bk_b2, HIDDEN);
        restore_3d!(a_w1,  r.a_w1,  HIDDEN,      HEAD_HIDDEN);
        restore_2d!(a_b1,  r.a_b1,  HEAD_HIDDEN);
        restore_3d!(a_w2,  r.a_w2,  HEAD_HIDDEN, ACTOR_OUT);
        restore_2d!(a_b2,  r.a_b2,  ACTOR_OUT);
        restore_3d!(c_w1,  r.c_w1,  HIDDEN,      HEAD_HIDDEN);
        restore_2d!(c_b1,  r.c_b1,  HEAD_HIDDEN);
        restore_3d!(c_w2,  r.c_w2,  HEAD_HIDDEN, 1);
        restore_2d!(c_b2,  r.c_b2,  1);

        for (i, v) in r.prev_state .iter().enumerate() { self.prev_state [s * STATE_DIM  + i] = *v; }
        for (i, v) in r.prev_action.iter().enumerate() { self.prev_action[s * ACTION_DIM + i] = *v; }
        self.prev_dopamine[s]        = r.prev_dopamine;
        self.prev_target_distance[s] = r.prev_target_distance;
    }
}

// ── Public telemetry snapshot ──────────────────────────────────────────────
//
// Surfaced to `dataset_export` so each organism's CSV row can
// carry a snapshot of its brain's current outputs + recent reward
// statistics. Computed via one batched forward pass on the cached
// `prev_state` of every slot — cost is one apply tick's worth of
// inference, run only when an export fires.

/// One row of the training-statistics CSV. Logged once per
/// completed train_step (every `ROLLOUT_LEN` brain ticks ≈ 2.4 s).
/// Capped at 1024 rows of history so a long-running simulation
/// doesn't grow the pool's memory footprint unboundedly — when
/// the user exports, the most recent 1024 steps land in the CSV.
#[derive(Clone, Debug)]
pub struct TrainingStep {
    pub step:               u64,
    pub virtual_time_secs:  f32,
    pub n_active:           u32,
    pub actor_loss:         f32,
    pub critic_loss:        f32,
    pub entropy:            f32,
    pub total_loss:         f32,
    pub mean_return:        f32,
    pub return_var:         f32,
    /// Weighted MSE pull from the supervised "chase nearest prey"
    /// bootstrap. Zero once `step` exceeds `BOOTSTRAP_STEPS`.
    pub supervised_loss:    f32,
}

const TRAINING_HISTORY_CAP: usize = 1024;


#[derive(Clone, Debug)]
pub struct BrainTelemetry {
    pub mu_speed:        f32,
    pub mu_angle:        f32,
    pub log_sigma_speed: f32,
    pub log_sigma_angle: f32,
    pub value_v:         f32,
    /// Most recent reward stored in the ring buffer (the reward
    /// from the tick that just ended).
    pub last_reward:     f32,
    /// Unconditional mean of the last 64 rewards. Freshly-reset
    /// slots show 0 until the ring fills.
    pub mean_reward_64:  f32,
    /// W_EAT · Δpredations component of the most recent reward.
    /// Non-zero exactly on the tick a predation happened.
    pub last_eat_component:      f32,
    /// W_PROGRESS · (Δtarget_distance / SENSORY_RADIUS) component
    /// of the most recent reward.
    pub last_progress_component: f32,
    /// W_ORACLE · hunger · cos(movement_direction, dir_to_prey)
    /// component of the most recent reward. Zero when no prey is
    /// in sensor range.
    pub last_oracle_component:   f32,
}

impl BrainPoolHerbivore1 {
    /// Run a single batched forward pass over every slot's cached
    /// `prev_state` and return a per-slot `BrainTelemetry`. Caller
    /// indexes by `pool.map[entity] as usize`. Slots that aren't
    /// currently assigned still produce a record (the policy's
    /// output for whatever state was last observed by the slot's
    /// previous tenant — harmless because callers filter on
    /// `pool.map.contains_key(entity)`).
    /// Read-only view of the recent training history. The export
    /// system writes these rows directly to the training-stats CSV.
    pub fn training_history(&self) -> &VecDeque<TrainingStep> {
        &self.training_history
    }

    pub fn snapshot_telemetry(&self) -> Vec<BrainTelemetry> {
        let n = self.n;
        let x = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(self.prev_state.clone(), [n, 1, STATE_DIM]),
            &self.device,
        );
        let (actor_out, value) = self.model.forward(x);
        let actor_data = actor_out.into_data().into_vec::<f32>().expect("telemetry actor");
        let value_data = value.into_data().into_vec::<f32>().expect("telemetry value");

        let mut out = Vec::with_capacity(n);
        for s in 0..n {
            let row = s * ACTOR_OUT;
            // Mean of recent_rewards across all 64 ring slots.
            let mut sum = 0.0_f32;
            for i in 0..64 { sum += self.recent_rewards[s * 64 + i]; }
            let mean_r = sum / 64.0;
            // last_reward = entry just behind the head pointer.
            let head = self.recent_reward_head[s] as usize;
            let last_idx = if head == 0 { 63 } else { head - 1 };
            let last_r = self.recent_rewards[s * 64 + last_idx];

            out.push(BrainTelemetry {
                mu_speed:        actor_data[row + 0],
                mu_angle:        actor_data[row + 1],
                // Report EFFECTIVE log σ (with INITIAL_LOG_SIGMA
                // offset applied) so the export reflects the σ that
                // actually drives sampling, not the raw network
                // output which starts at 0 but means σ ≈ 0.30.
                log_sigma_speed: actor_data[row + 2] + INITIAL_LOG_SIGMA,
                log_sigma_angle: actor_data[row + 3] + INITIAL_LOG_SIGMA,
                value_v:         value_data[s],   // [n, 1, 1] flattens to length n
                last_reward:     last_r,
                mean_reward_64:  mean_r,
                last_eat_component:      self.last_eat[s],
                last_progress_component: self.last_progress[s],
                last_oracle_component:   self.last_oracle[s],
            });
        }
        out
    }
}


impl FromWorld for BrainPoolHerbivore1 {
    fn from_world(world: &mut World) -> Self {
        let n = world.get_resource::<OrganismPoolSize>().map(|r| r.0.max(1)).unwrap_or(1);
        let device = CudaDevice::default();
        warmup(&device, n);
        Self::new(device, n)
    }
}

fn warmup(device: &CudaDevice, n: usize) {
    let m = PolicyNet::<MyBackend>::new(device, n);
    let mut o: Box<dyn BrainOptHerbivore1> = Box::new(AdamConfig::new().init());
    let x = Tensor::<MyBackend, 3>::zeros([n, 1, STATE_DIM], device);
    let (a, v) = m.forward(x);
    let loss = (a.powf_scalar(2.0).sum() + v.powf_scalar(2.0).sum()).div_scalar(1.0_f32);
    let g = GradientsParams::from_grads(loss.backward(), &m);
    let _ = o.step(LR, m, g);
}


// ── Slot lifecycle ─────────────────────────────────────────────────────────

pub fn assign_brains_herbivore_1(
    mut pool:     NonSendMut<BrainPoolHerbivore1>,
    new:          Query<
        (Entity, &Organism, Option<&BrainInheritance>, Option<&BrainRestoreHerbivore1>),
        (
            With<Heterotroph>,
            Without<BrainSlotHerbivore1>,
            Without<Carnivore>,
        ),
    >,
    mut commands: Commands,
) {
    for (e, organism, inheritance, restore) in new.iter() {
        if !matches!(organism.intelligence_level, IntelligenceLevel::Level1) { continue; }
        let Some(slot) = pool.free.pop() else { continue };
        let s = slot as usize;

        pool.reset_slot_state(s);

        // Priority: explicit `BrainRestoreHerbivore1` (load-from-save)
        // wins over `BrainInheritance` (offspring → parent copy).
        // Loaded organisms shouldn't also have an inheritance link;
        // if both are present, the save data is the authoritative
        // source.
        if let Some(r) = restore {
            pool.restore_slot(slot, r);
        } else if let Some(BrainInheritance(parent)) = inheritance {
            if let Some(&parent_slot) = pool.map.get(parent) {
                pool.model.inherit_row(parent_slot as usize, s);
            }
        }

        pool.rollout_active[s] = false;
        pool.map.insert(e, slot);
        commands.entity(e).try_insert(BrainSlotHerbivore1(slot));
        commands.entity(e).try_remove::<BrainInheritance>();
        commands.entity(e).try_remove::<BrainRestoreHerbivore1>();
    }
}

pub fn free_brains_herbivore_1(
    mut pool:    NonSendMut<BrainPoolHerbivore1>,
    mut removed: RemovedComponents<Heterotroph>,
) {
    for e in removed.read() {
        if let Some(slot) = pool.map.remove(&e) {
            pool.free.push(slot);
            pool.rollout_active[slot as usize] = false;
        }
    }
}


// ── Apply tick ─────────────────────────────────────────────────────────────

pub fn apply_intelligence_level_herbivore_1(
    time:        Res<Time<Virtual>>,
    world_grid:  Res<WorldModelGrid>,
    mut pool:    NonSendMut<BrainPoolHerbivore1>,
    mut heteros: Query<
        (Entity, &mut Organism, &Transform, &BrainSlotHerbivore1),
        (With<Heterotroph>, Without<Carnivore>),
    >,
    mut input_buf: Local<Vec<f32>>,
) {
    if time.is_paused() { return; }
    let n = pool.n();

    // ── Step 1: reward credit from PREVIOUS tick's action. ───────
    //
    // Two channels, both clean:
    //   * `eat_event` = Δpredations between brain ticks. Fires
    //     sharply (+1 per body part consumed) on the tick a
    //     predation happens; zero otherwise. No decay tail to fight
    //     and no dopamine intermediary that washes out 96%+ of
    //     ticks. Weighted by `W_EAT` so a single eat outranks the
    //     densest possible progress signal in one tick.
    //   * `progress` = Δtarget_distance / SENSORY_RADIUS. Dense —
    //     a non-zero value on essentially every brain tick the
    //     agent is moving relative to its nearest prey. Weighted
    //     by `W_PROGRESS` to give the policy gradient a continuous
    //     signal between rare eat events.
    if pool.global_step > 0 {
        let prev_idx = pool.global_step as usize - 1;
        for (_, organism, transform, slot) in heteros.iter() {
            let s = slot.0 as usize;
            if s >= n { continue; }
            if !pool.rollout_active[s] { continue; }

            // Signed subtraction in i32 space so u8 wraparound on
            // saturated `Organism::predations` (it caps at 255 via
            // `saturating_add`) never produces a phantom negative
            // delta. In practice the counter rarely exceeds a few
            // hundred during a single brain tick — this is just
            // defensive against the u8 ceiling.
            let eat_event = (organism.predations as i32
                             - pool.prev_predations[s] as i32) as f32;
            let progress  = ((pool.prev_target_distance[s] - organism.target_distance)
                             / SENSORY_RADIUS).clamp(-1.0, 1.0);

            // Oracle-alignment reward: cosine similarity of the
            // movement direction the policy actually produced last
            // tick against the world-frame direction to the nearest
            // visible prey. Scaled by hunger so the signal vanishes
            // when the agent doesn't need to eat. Zero when there
            // is no prey in range, or when either vector is
            // degenerate (NaN-guard).
            let oracle_alignment = match nearest_prey(&world_grid, transform.translation) {
                Some((rel, _, _)) => {
                    let rel_mag2 = rel.x * rel.x + rel.z * rel.z;
                    let dir = organism.movement_direction;
                    let dir_mag2 = dir.x * dir.x + dir.z * dir.z;
                    if rel_mag2 < 1e-12 || dir_mag2 < 1e-12 { 0.0 }
                    else {
                        (rel.x * dir.x + rel.z * dir.z)
                            / (rel_mag2.sqrt() * dir_mag2.sqrt())
                    }
                }
                None => 0.0,
            };
            let eat_term      = W_EAT      * eat_event;
            let progress_term = W_PROGRESS * progress;
            let oracle_term   = W_ORACLE   * organism.hunger * oracle_alignment;
            let reward        = eat_term + progress_term + oracle_term;

            pool.rollout_rewards[s * ROLLOUT_LEN + prev_idx] = reward;

            // Diagnostic telemetry — ring buffer + last components.
            // Used by `snapshot_telemetry` at export time; never
            // read by the policy.
            let head = pool.recent_reward_head[s] as usize;
            pool.recent_rewards[s * 64 + head] = reward;
            pool.recent_reward_head[s] = ((head + 1) % 64) as u8;
            pool.last_eat[s]      = eat_term;
            pool.last_progress[s] = progress_term;
            pool.last_oracle[s]   = oracle_term;
        }
    }

    // ── Step 2: build the FULL-N input batch (zeros for inactive
    //           slots). Each slot's row of weights consumes only its
    //           own row of inputs, so inactive slots produce
    //           bias-only outputs we then ignore.
    input_buf.clear();
    input_buf.resize(n * STATE_DIM, 0.0);

    #[derive(Clone, Copy)]
    struct Active {
        entity:          Entity,
        slot:            u32,
        dopamine:        f32,
        target_distance: f32,
        predations:      u8,
        h_x:             f32,
        h_z:             f32,
    }
    let mut active: Vec<Active> = Vec::new();

    for (e, organism, transform, slot) in heteros.iter() {
        let s = slot.0 as usize;
        if s >= n { continue; }
        let pos = transform.translation;

        // Body-frame reference for this tick.
        let heading = organism.movement_direction;
        let h_mag = (heading.x * heading.x + heading.z * heading.z).sqrt();
        let (h_x, h_z) = if h_mag > 1e-6 {
            (heading.x / h_mag, heading.z / h_mag)
        } else {
            (0.0, 1.0)
        };

        let (body_fwd, body_right, has_photo) = match nearest_prey(&world_grid, pos) {
            Some((rel, _, _)) => {
                let fwd   = rel.x * h_x + rel.z * h_z;
                let right = rel.x * h_z - rel.z * h_x;
                (
                    (fwd   / WORLD_MODEL_RADIUS).clamp(-1.0, 1.0),
                    (right / WORLD_MODEL_RADIUS).clamp(-1.0, 1.0),
                    1.0,
                )
            }
            None => (0.0, 0.0, 0.0),
        };
        let max_e   = get_max_energy(&organism).max(1.0);
        let energy_n = (organism.energy / max_e).clamp(0.0, 1.0);
        let speed_n  = (organism.movement_speed / MAX_SPEED).clamp(0.0, 1.0);
        let pa_off   = s * ACTION_DIM;
        let prev_speed_a = pool.prev_action[pa_off + 0];
        let prev_angle   = pool.prev_action[pa_off + 1];
        let td_norm = (organism.target_distance / SENSORY_RADIUS).clamp(0.0, 1.0);

        let obs: [f32; STATE_DIM] = [
            organism.hunger, organism.dopamine, td_norm, has_photo,
            body_fwd, body_right, speed_n, energy_n,
            prev_speed_a, prev_angle,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let in_off = s * STATE_DIM;
        for d in 0..STATE_DIM { input_buf[in_off + d] = obs[d]; }

        active.push(Active {
            entity: e, slot: slot.0,
            dopamine: organism.dopamine,
            target_distance: organism.target_distance,
            predations: organism.predations,
            h_x, h_z,
        });
    }
    if active.is_empty() {
        pool.global_step = pool.global_step.wrapping_add(1);
        if (pool.global_step as usize) >= ROLLOUT_LEN { pool.global_step = 0; }
        return;
    }

    // ── Step 3: forward pass — [N, 1, STATE_DIM] → [N, 1, ACTOR_OUT]
    let x_t = Tensor::<MyBackend, 3>::from_data(
        TensorData::new(input_buf.clone(), [n, 1, STATE_DIM]),
        &pool.device,
    );
    let (actor_out_t, value_t) = pool.model.forward(x_t);
    // `into_data` flattens the [N, 1, K] tensor row-major; index by
    // `s * K + k` to read slot s's k-th output.
    let actor_out_data = actor_out_t.into_data().into_vec::<f32>().expect("actor fwd");
    let _ = value_t.into_data();   // discard; bootstrap value is recomputed at training time

    // ── Step 4: sample, apply, store. ────────────────────────────
    let mut rng = rand::rng();
    let rollout_idx = pool.global_step as usize;
    for a in active.iter() {
        let s   = a.slot as usize;
        let row = s * ACTOR_OUT;

        let mu_speed  = actor_out_data[row + 0];
        let mu_angle  = actor_out_data[row + 1];
        let ls_speed  = (actor_out_data[row + 2] + INITIAL_LOG_SIGMA)
                            .clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX);
        let ls_angle  = (actor_out_data[row + 3] + INITIAL_LOG_SIGMA)
                            .clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX);
        let sig_speed = ls_speed.exp();
        let sig_angle = ls_angle.exp();

        let u_speed = mu_speed + sig_speed * gaussian_noise(&mut rng);
        let u_angle = mu_angle + sig_angle * gaussian_noise(&mut rng);
        let speed_a = u_speed.tanh();
        let theta   = u_angle.tanh();

        let alpha = PI * theta;
        let (sa, ca) = alpha.sin_cos();
        let new_dx = a.h_x * ca - a.h_z * sa;
        let new_dz = a.h_x * sa + a.h_z * ca;

        let speed = ((speed_a + 1.0) * 0.5) * MAX_SPEED;

        if let Ok((_, mut org, _, _)) = heteros.get_mut(a.entity) {
            org.movement_speed     = speed;
            org.movement_direction = Vec3::new(new_dx, 0.0, new_dz);
        }

        pool.prev_dopamine[s]        = a.dopamine;
        pool.prev_target_distance[s] = a.target_distance;
        pool.prev_predations[s]      = a.predations;
        pool.prev_action[s * ACTION_DIM + 0] = speed_a;
        pool.prev_action[s * ACTION_DIM + 1] = theta;
        let in_off = s * STATE_DIM;
        for d in 0..STATE_DIM { pool.prev_state[in_off + d] = input_buf[in_off + d]; }

        // Store raw (unsquashed) u in the rollout — training takes
        // the Gaussian log-prob of u given (μ, σ).
        let st_off = s * ROLLOUT_LEN * STATE_DIM + rollout_idx * STATE_DIM;
        for d in 0..STATE_DIM { pool.rollout_states[st_off + d] = input_buf[in_off + d]; }
        let ac_off = s * ROLLOUT_LEN * ACTION_DIM + rollout_idx * ACTION_DIM;
        pool.rollout_actions[ac_off + 0] = u_speed;
        pool.rollout_actions[ac_off + 1] = u_angle;

        pool.rollout_active[s] = true;
    }

    // ── Step 5: maybe train. ─────────────────────────────────────
    pool.global_step = pool.global_step.wrapping_add(1);
    if (pool.global_step as usize) >= ROLLOUT_LEN {
        // Bootstrap value: V(s_T) from the latest observation, one
        // per slot. Use the same `input_buf` (current state ≈ s_T).
        let bootstrap_in = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(input_buf.clone(), [n, 1, STATE_DIM]),
            &pool.device,
        );
        let (_, v_boot_t) = pool.model.forward(bootstrap_in);
        let v_boot_data   = v_boot_t.into_data().into_vec::<f32>().expect("bootstrap V");

        train_step(&mut pool, &v_boot_data, time.elapsed_secs());

        pool.global_step = 0;
        let active_slots: Vec<u32> = pool.map.values().copied().collect();
        for s in active_slots {
            pool.rollout_active[s as usize] = true;
        }
    }
}


// ── A2C training step (per-organism) ───────────────────────────────────────

fn train_step(
    pool:              &mut BrainPoolHerbivore1,
    bootstrap_v:       &[f32],   // [n]
    virtual_time_secs: f32,
) {
    let n = pool.n();

    // Build per-slot returns walking backward through the rollout.
    let mut returns = vec![0.0_f32; n * ROLLOUT_LEN];
    let mut mask    = vec![0.0_f32; n * ROLLOUT_LEN];

    let mut active_count = 0usize;
    for s in 0..n {
        if !pool.rollout_active[s] { continue; }
        let v_boot = bootstrap_v[s];   // [N, 1, 1] flattened → index s
        let mut r_next = v_boot;
        for t in (0..ROLLOUT_LEN).rev() {
            let r = pool.rollout_rewards[s * ROLLOUT_LEN + t];
            r_next = r + GAMMA * r_next;
            returns[s * ROLLOUT_LEN + t] = r_next;
            mask   [s * ROLLOUT_LEN + t] = 1.0;
        }
        active_count += 1;
    }
    if active_count == 0 { return; }

    // Pre-compute return statistics from the CPU vec (before
    // ownership moves into TensorData::new below). Used by the
    // training-stats CSV — gives a sense of the magnitude /
    // dispersion of the n-step returns that drove this step.
    let (mean_return, return_var) = {
        let mut sum = 0.0_f32; let mut count = 0usize;
        for s in 0..n {
            if !pool.rollout_active[s] { continue; }
            for t in 0..ROLLOUT_LEN {
                sum += returns[s * ROLLOUT_LEN + t];
                count += 1;
            }
        }
        let mean = if count > 0 { sum / count as f32 } else { 0.0 };
        let mut ss = 0.0_f32;
        for s in 0..n {
            if !pool.rollout_active[s] { continue; }
            for t in 0..ROLLOUT_LEN {
                let d = returns[s * ROLLOUT_LEN + t] - mean;
                ss += d * d;
            }
        }
        let var = if count > 1 { ss / (count - 1) as f32 } else { 0.0 };
        (mean, var)
    };

    let device = &pool.device;
    let model  = pool.model.clone();

    // Batched [N, ROLLOUT_LEN, STATE_DIM / ACTION_DIM] tensors.
    let states_t = Tensor::<MyBackend, 3>::from_data(
        TensorData::new(pool.rollout_states.clone(),  [n, ROLLOUT_LEN, STATE_DIM]),
        device,
    );
    let actions_t = Tensor::<MyBackend, 3>::from_data(
        TensorData::new(pool.rollout_actions.clone(), [n, ROLLOUT_LEN, ACTION_DIM]),
        device,
    );
    let returns_t = Tensor::<MyBackend, 3>::from_data(
        TensorData::new(returns, [n, ROLLOUT_LEN, 1]),
        device,
    );
    let mask_t = Tensor::<MyBackend, 3>::from_data(
        TensorData::new(mask, [n, ROLLOUT_LEN, 1]),
        device,
    );

    let (actor_out, value) = model.forward(states_t);   // actor: [N, T, 4]; value: [N, T, 1]

    let mu_speed = actor_out.clone().slice([0..n, 0..ROLLOUT_LEN, 0..1]);
    let mu_angle = actor_out.clone().slice([0..n, 0..ROLLOUT_LEN, 1..2]);
    // INITIAL_LOG_SIGMA offset keeps the train-time and apply-time
    // log σ definitions in lockstep (effective log σ = network
    // output + INITIAL_LOG_SIGMA, clamped).
    let ls_speed = actor_out.clone().slice([0..n, 0..ROLLOUT_LEN, 2..3])
                            .add_scalar(INITIAL_LOG_SIGMA)
                            .clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX);
    let ls_angle = actor_out.slice([0..n, 0..ROLLOUT_LEN, 3..4])
                            .add_scalar(INITIAL_LOG_SIGMA)
                            .clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX);
    let sigma_speed = ls_speed.clone().exp();
    let sigma_angle = ls_angle.clone().exp();

    let a_speed = actions_t.clone().slice([0..n, 0..ROLLOUT_LEN, 0..1]);
    let a_angle = actions_t.slice([0..n, 0..ROLLOUT_LEN, 1..2]);

    // Clone μ tensors — the original handles flow into the diff
    // computation below, but the supervised bootstrap loss further
    // down also needs to compute (μ − target).
    let diff_speed = (a_speed - mu_speed.clone()) / sigma_speed.clone();
    let diff_angle = (a_angle - mu_angle.clone()) / sigma_angle.clone();
    let lp_speed = diff_speed.powf_scalar(2.0).mul_scalar(-0.5) - ls_speed.clone();
    let lp_angle = diff_angle.powf_scalar(2.0).mul_scalar(-0.5) - ls_angle.clone();
    let log_prob = lp_speed + lp_angle;     // [N, T, 1]

    let value_detached = value.clone().detach();
    let advantage = returns_t.clone() - value_detached;

    // Apply per-(N, T) mask so inactive slots / inactive rollouts
    // contribute zero gradient.
    let actor_loss  = ((-log_prob * advantage) * mask_t.clone()).mean();
    let critic_loss = ((returns_t - value).powf_scalar(2.0).mul_scalar(0.5)
                       * mask_t.clone()).mean();
    let entropy     = ((ls_speed + ls_angle) * mask_t).mean();

    // Extract scalar loss components for the training-stats CSV.
    // Cloning a burn `Tensor` is shallow (Arc-style); the original
    // is still in the gradient graph and `loss.backward()` below
    // operates on the same data.
    let actor_loss_val  = actor_loss .clone().into_data().into_vec::<f32>().expect("scalar")[0];
    let critic_loss_val = critic_loss.clone().into_data().into_vec::<f32>().expect("scalar")[0];
    let entropy_val     = entropy    .clone().into_data().into_vec::<f32>().expect("scalar")[0];

    // ── Supervised bootstrap (decays linearly over BOOTSTRAP_STEPS) ──
    //
    // For every (slot, t) where has_photo == 1, push μ toward an
    // oracle target derived purely from the body-frame geometry that
    // the agent observed at that tick:
    //
    //   target_μ_speed = BOOTSTRAP_MU_SPEED_TARGET     (full speed)
    //   target_μ_angle = atanh(clamp(atan2(body_right, body_fwd) / π,
    //                          -0.99, 0.99))           (face the prey)
    //
    // The clamp inside atanh keeps the target bounded — the edge case
    // is prey almost exactly behind the agent, where atan2/π → ±1 and
    // atanh diverges. Targets live in pre-tanh space so MSE compares
    // apples-to-apples against the network's raw μ output.
    //
    // Weight decays from 1.0 (step 0) to 0.0 (step BOOTSTRAP_STEPS).
    // After that the RL signal flies solo; the supervised tensors
    // aren't even allocated.
    let bootstrap_weight = if pool.step_counter < BOOTSTRAP_STEPS {
        1.0 - pool.step_counter as f32 / BOOTSTRAP_STEPS as f32
    } else {
        0.0
    };
    let (loss, supervised_loss_val) = if bootstrap_weight > 0.0 {
        let mut target_speed = vec![0.0_f32; n * ROLLOUT_LEN];
        let mut target_angle = vec![0.0_f32; n * ROLLOUT_LEN];
        let mut sup_mask     = vec![0.0_f32; n * ROLLOUT_LEN];
        for s in 0..n {
            if !pool.rollout_active[s] { continue; }
            for t in 0..ROLLOUT_LEN {
                let st_off = s * ROLLOUT_LEN * STATE_DIM + t * STATE_DIM;
                if pool.rollout_states[st_off + 3] <= 0.5 { continue; }
                let body_fwd   = pool.rollout_states[st_off + 4];
                let body_right = pool.rollout_states[st_off + 5];
                let theta = (body_right.atan2(body_fwd) / PI).clamp(-0.99, 0.99);
                let idx = s * ROLLOUT_LEN + t;
                target_speed[idx] = BOOTSTRAP_MU_SPEED_TARGET;
                target_angle[idx] = theta.atanh();
                sup_mask[idx]     = 1.0;
            }
        }
        let target_speed_t = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(target_speed, [n, ROLLOUT_LEN, 1]), device,
        );
        let target_angle_t = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(target_angle, [n, ROLLOUT_LEN, 1]), device,
        );
        let sup_mask_t = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(sup_mask, [n, ROLLOUT_LEN, 1]), device,
        );
        let sup_speed = (mu_speed - target_speed_t).powf_scalar(2.0);
        let sup_angle = (mu_angle - target_angle_t).powf_scalar(2.0);
        let sup_loss = ((sup_speed + sup_angle) * sup_mask_t)
                         .mean()
                         .mul_scalar(bootstrap_weight);
        let v = sup_loss.clone().into_data().into_vec::<f32>().expect("scalar")[0];
        let loss = actor_loss
            + critic_loss.mul_scalar(VALUE_COEF)
            - entropy.mul_scalar(ENTROPY_COEF)
            + sup_loss;
        (loss, v)
    } else {
        // Drop the now-unused μ slices so the borrow checker is happy.
        let _ = (mu_speed, mu_angle);
        let loss = actor_loss
            + critic_loss.mul_scalar(VALUE_COEF)
            - entropy.mul_scalar(ENTROPY_COEF);
        (loss, 0.0)
    };
    let total_loss_val  = loss.clone().into_data().into_vec::<f32>().expect("scalar")[0];

    let grads = GradientsParams::from_grads(loss.backward(), &model);
    pool.model = pool.opt.step(LR, model, grads);

    // Push one TrainingStep onto the ring buffer. Drop the oldest
    // entry if at capacity so memory stays bounded over long runs.
    pool.step_counter = pool.step_counter.saturating_add(1);
    if pool.training_history.len() >= TRAINING_HISTORY_CAP {
        pool.training_history.pop_front();
    }
    pool.training_history.push_back(TrainingStep {
        step:              pool.step_counter,
        virtual_time_secs,
        n_active:          active_count as u32,
        actor_loss:        actor_loss_val,
        critic_loss:       critic_loss_val,
        entropy:           entropy_val,
        total_loss:        total_loss_val,
        mean_return,
        return_var,
        supervised_loss:   supervised_loss_val,
    });
}
