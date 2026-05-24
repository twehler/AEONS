// Intelligence Level 1 (herbivore non-carnivore) — REINFORCE pool with
// target-selection logits.
//
// Architectural port of `intelligence_level_3.rs` (the Krishi brain),
// applied to non-Carnivore Level-1 herbivores. The previous A2C
// implementation here trained an angular policy from raw body-frame
// floats, used a critic value head, and relied on a supervised
// bootstrap to clear the cold-start hump. The May-2026 dataset
// analysis showed that this configuration was structurally unable to
// produce a chase policy:
//
//   * `ρ(μ_angle, oracle_direction) ≈ 0` after 30 min of training
//   * `ρ(critic_loss, return_var) = 0.96`  (critic tracks magnitude,
//                                            doesn't predict)
//   * Top-mover agents on `a_w2` ATE LESS than bottom-mover agents
//                                            (policy drift, no learning)
//
// Meanwhile Krishi (same Heterotroph marker, Level3 intelligence)
// hunted cleanly out of the box. The architectural reason was the
// target-selection-via-logits-with-geometric-direction layout: the
// network picks WHICH neighbour to chase, then the math chases it.
// The network never has to learn trigonometry. This file ports that
// design to Level-1 herbivores.
//
// Architecture (identical to L3):
//   * Network: 31 inputs → 32 hidden ReLU → 5 outputs (tanh).
//   * Inputs (31): normalised energy + 16-dim world model (K = 4
//     nearest neighbours) + 5-dim previous action + 1 locked-target
//     flag.
//   * Outputs (5): `speed_a` + 4 target logits. Speed maps via
//     `max(0, speed_a) * MAX_SPEED`. Direction is GEOMETRIC, derived
//     from the chosen target's position, not learned.
//
// Reward (every brain tick, attributed to the previous action):
//
//   r =  K_EAT        on a predation contact
//      + K_REPRO      on a reproduction event
//      + LAMBDA · min(0, ΔE)       energy penalty
//      + K_CURIOSITY · (applied_speed_norm − 0.5)   stillness penalty
//      + K_PROGRESS · max(0, distance_closed_to_locked_target)
//
// Algorithm: pure Monte Carlo REINFORCE with per-slot EMA baseline.
// Every brain tick fills the previous action's reward, samples a new
// action, and stores both in a ring buffer of length ROLLOUT_LEN.
// Every ROLLOUT_LEN ticks the whole pool trains in one batched
// forward + backward.
//
// Save/load: weight format is the shared `BrainRestore` from
// `rl_helpers.rs`, identical to L2 and L3. Old herbivore_1 save
// payloads (12-tensor backbone+actor+critic layout) are no longer
// compatible — the colony format magic is bumped to AEONS005.

use bevy::prelude::*;
use burn::module::{Initializer, Module, Param};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use burn_cuda::CudaDevice;
use std::collections::{HashMap, VecDeque};

use crate::colony::{IntelligenceLevel, Organism, Heterotroph, Carnivore};
use crate::rl_helpers::{BrainInheritance, BrainRestore, MyBackend, PoolSnapshot, gaussian_noise};
use crate::simulation_settings::{
    OrganismPoolSize,
    L1_SIGMA_RANGE,
    L1_K_EAT_RANGE,
    L1_K_REPRO_RANGE,
    L1_LAMBDA_ENERGY_RANGE,
    L1_K_CURIOSITY_RANGE,
    L1_K_PROGRESS_RANGE,
    L1_GENE_MUTATION_REL_STDDEV,
    L1_TARGET_LOCK_SECS,
    L1_TARGET_SWITCH_MARGIN,
    L1_SPEED_MOMENTUM_ALPHA,
    L1_APPROACH_RADIUS,
    L1_STUCK_TICKS,
    L1_STUCK_PROGRESS_EPS,
    L1_DIRECTION_FREEZE_DIST,
    L1_NO_TARGET_SPEED_SCALE,
    L1_MIN_APPLIED_SPEED,
    L1_NO_TARGET_WANDER_ANGLE,
    L1_TARGET_BLACKLIST_TICKS,
};
use crate::world_model::{
    WorldModelGrid, WORLD_MODEL_DIMS, WORLD_MODEL_K, OrganismType,
    collect_neighbours, encode_neighbours,
};


// ── Architecture constants (must match L3 for the shared BrainRestore
//   format to work identically across both pools) ──────────────────

const PREV_ACTION_DIMS:    usize = 5;
const LOCKED_FLAG_DIMS:    usize = 1;

const HETERO_BRAIN_TICK_SECS: f32 = 0.150;

const L1_TARGET_LOCK_TICKS: u16 = {
    let f = L1_TARGET_LOCK_SECS / HETERO_BRAIN_TICK_SECS;
    let truncated = f as u16;
    if (truncated as f32) < f { truncated + 1 } else { truncated }
};

pub const IN:      usize = 1 + WORLD_MODEL_DIMS + PREV_ACTION_DIMS + LOCKED_FLAG_DIMS;
pub const HIDDEN:  usize = 32;
pub const OUT:     usize = 1 + WORLD_MODEL_K;
const ROLLOUT_LEN: usize = 32;
const MAX_SPEED:   f32   = 40.0;
const LR:          f64   = 1e-3;

const GAMMA:           f32 = 0.95;
const BASELINE_ALPHA:  f32 = 0.05;
const WARMUP_SIGMA:    f32 = 0.5;

const TRAINING_HISTORY_CAP: usize = 1024;


fn sample_range(range: (f32, f32), rng: &mut impl rand::Rng) -> f32 {
    use rand::RngExt;
    let (lo, hi) = range;
    lo + (hi - lo) * rng.random::<f32>()
}

fn mutate_in_range(value: f32, range: (f32, f32), rng: &mut impl rand::Rng) -> f32 {
    let (lo, hi) = range;
    let stddev = L1_GENE_MUTATION_REL_STDDEV * (hi - lo);
    (value + stddev * crate::rl_helpers::gaussian_noise(rng)).clamp(lo, hi)
}


// ── Slot marker ─────────────────────────────────────────────────────────────

#[derive(Component, Clone, Copy)]
pub struct BrainSlotHerbivore1(pub u32);


// ── Per-organism MLP ────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct PolicyNet<B: Backend> {
    pub w1: Param<Tensor<B, 3>>,  // [N, IN, HIDDEN]
    pub b1: Param<Tensor<B, 2>>,  // [N, HIDDEN]
    pub w2: Param<Tensor<B, 3>>,  // [N, HIDDEN, OUT]
    pub b2: Param<Tensor<B, 2>>,  // [N, OUT]
}

impl<B: Backend> PolicyNet<B> {
    fn new(device: &B::Device, n: usize) -> Self {
        let w = Initializer::Uniform { min: -0.5, max: 0.5 };
        let z = Initializer::Zeros;
        Self {
            w1: w.init([n, IN, HIDDEN], device),
            b1: z.init([n, HIDDEN], device),
            w2: w.init([n, HIDDEN, OUT], device),
            b2: z.init([n, OUT], device),
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = x.unsqueeze_dim::<3>(1).matmul(self.w1.val()).squeeze::<2>() + self.b1.val();
        let h = burn::tensor::activation::relu(h);
        let mu_pre = h.unsqueeze_dim::<3>(1).matmul(self.w2.val()).squeeze::<2>() + self.b2.val();
        burn::tensor::activation::tanh(mu_pre)
    }

    fn forward_rollout(&self, states: Tensor<B, 3>) -> Tensor<B, 3> {
        let h_pre = states.matmul(self.w1.val());
        let b1_b = self.b1.val().unsqueeze_dim::<3>(1);
        let h = burn::tensor::activation::relu(h_pre + b1_b);
        let mu_pre = h.matmul(self.w2.val());
        let b2_b = self.b2.val().unsqueeze_dim::<3>(1);
        burn::tensor::activation::tanh(mu_pre + b2_b)
    }
}


trait BrainOpt {
    fn step(
        &mut self,
        lr: f64,
        m:  PolicyNet<MyBackend>,
        g:  GradientsParams,
    ) -> PolicyNet<MyBackend>;
}

impl<O: Optimizer<PolicyNet<MyBackend>, MyBackend>> BrainOpt for O {
    fn step(
        &mut self,
        lr: f64,
        m:  PolicyNet<MyBackend>,
        g:  GradientsParams,
    ) -> PolicyNet<MyBackend> {
        Optimizer::step(self, lr, m, g)
    }
}


// ── Save / load payload ─────────────────────────────────────────────────────

/// Re-export of the shared `BrainRestore` (4-tensor MLP + REINFORCE
/// prev_*). The previous 12-tensor (backbone+actor+critic) format has
/// been retired with the L3-architecture port.
pub type BrainRestoreHerbivore1 = BrainRestore;

/// Magic prefixed to the brain block so a future format change (or
/// stale data from the old A2C format) is caught at decode time.
const HERBIVORE1_BRAIN_MAGIC: &[u8; 8] = b"HV1B0001";

/// Serialise a `BrainRestoreHerbivore1` into `buf`. Format:
///
///   magic (8 B)                  HV1B0001
///   w1 length (u32)              IN * HIDDEN
///   w1 floats                    f32×N
///   b1 length (u32)              HIDDEN
///   b1 floats                    f32×N
///   w2 length (u32)              HIDDEN * OUT
///   w2 floats                    f32×N
///   b2 length (u32)              OUT
///   b2 floats                    f32×N
///   prev_state length (u32)      IN
///   prev_state floats            f32×N
///   prev_action length (u32)     OUT
///   prev_action floats           f32×N
///   prev_energy (f32)
///   baseline    (f32)
///   has_prev    (u8, 0 or 1)
pub fn encode_brain_restore(buf: &mut Vec<u8>, b: &BrainRestoreHerbivore1) {
    buf.extend_from_slice(HERBIVORE1_BRAIN_MAGIC);
    let write_vec = |buf: &mut Vec<u8>, v: &Vec<f32>| {
        buf.extend_from_slice(&(v.len() as u32).to_le_bytes());
        for &x in v { buf.extend_from_slice(&x.to_le_bytes()); }
    };
    write_vec(buf, &b.w1);
    write_vec(buf, &b.b1);
    write_vec(buf, &b.w2);
    write_vec(buf, &b.b2);
    write_vec(buf, &b.prev_state);
    write_vec(buf, &b.prev_action);
    buf.extend_from_slice(&b.prev_energy.to_le_bytes());
    buf.extend_from_slice(&b.baseline.to_le_bytes());
    buf.push(if b.has_prev { 1 } else { 0 });
}

/// Deserialise a `BrainRestoreHerbivore1` from `bytes[*c..]`,
/// advancing `*c`. Hard-errors on magic / shape mismatch — the
/// `colony.rs` caller logs the failure and continues with a fresh
/// brain slot for that organism.
pub fn decode_brain_restore(
    bytes: &[u8],
    c:     &mut usize,
) -> std::io::Result<BrainRestoreHerbivore1> {
    fn read_u32(bytes: &[u8], c: &mut usize) -> std::io::Result<u32> {
        if *c + 4 > bytes.len() {
            return Err(std::io::Error::other("brain truncated (u32)"));
        }
        let v = u32::from_le_bytes(bytes[*c..*c+4].try_into().unwrap());
        *c += 4;
        Ok(v)
    }
    fn read_f32(bytes: &[u8], c: &mut usize) -> std::io::Result<f32> {
        if *c + 4 > bytes.len() {
            return Err(std::io::Error::other("brain truncated (f32)"));
        }
        let v = f32::from_le_bytes(bytes[*c..*c+4].try_into().unwrap());
        *c += 4;
        Ok(v)
    }
    fn read_u8(bytes: &[u8], c: &mut usize) -> std::io::Result<u8> {
        if *c >= bytes.len() {
            return Err(std::io::Error::other("brain truncated (u8)"));
        }
        let v = bytes[*c]; *c += 1; Ok(v)
    }
    if *c + 8 > bytes.len() {
        return Err(std::io::Error::other("brain truncated (magic)"));
    }
    let magic: [u8; 8] = bytes[*c..*c+8].try_into().unwrap();
    *c += 8;
    if &magic != HERBIVORE1_BRAIN_MAGIC {
        return Err(std::io::Error::other(format!(
            "brain magic mismatch: saved {:?}, expected {:?}",
            std::str::from_utf8(&magic).unwrap_or("<binary>"),
            std::str::from_utf8(HERBIVORE1_BRAIN_MAGIC).unwrap(),
        )));
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
    let w1          = read_vec_checked(c, IN * HIDDEN,  "w1")?;
    let b1          = read_vec_checked(c, HIDDEN,       "b1")?;
    let w2          = read_vec_checked(c, HIDDEN * OUT, "w2")?;
    let b2          = read_vec_checked(c, OUT,          "b2")?;
    let prev_state  = read_vec_checked(c, IN,           "prev_state")?;
    let prev_action = read_vec_checked(c, OUT,          "prev_action")?;
    let prev_energy = read_f32(bytes, c)?;
    let baseline    = read_f32(bytes, c)?;
    let has_prev    = read_u8(bytes, c)? != 0;
    Ok(BrainRestoreHerbivore1 {
        w1, b1, w2, b2,
        prev_state, prev_action,
        prev_energy, baseline, has_prev,
    })
}


// ── Training-statistics ring buffer ─────────────────────────────────────────

/// One row of the training-statistics CSV. Logged once per completed
/// REINFORCE update (every ROLLOUT_LEN brain ticks ≈ 4.8 s). Fields
/// match the columns the existing R analysis scripts expect; channels
/// that no longer apply to the L3-style architecture (critic_loss,
/// entropy, supervised_loss) are zeroed so the CSV format stays
/// stable.
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
    pub supervised_loss:    f32,
}


// ── Per-organism telemetry snapshot ─────────────────────────────────────────

/// Per-slot snapshot for `dataset_export` and `time_series_log`.
/// Fields are kept stable across the L1→L3-style architecture port:
/// channels that no longer apply (`mu_angle`, `log_sigma_*`,
/// `value_v`, `last_oracle_component`) are populated with the
/// closest sensible analogue or zero, so the existing R analysis
/// scripts keep parsing.
#[derive(Clone, Debug)]
pub struct BrainTelemetry {
    /// Speed channel raw output (post-tanh, before EMA / brake).
    pub mu_speed:                 f32,
    /// Largest target logit (post-tanh) — the strength of commitment
    /// to the currently-favoured prey slot. Replaces the old
    /// `mu_angle` field; same column name in the CSV so analysis
    /// pipelines pick it up.
    pub mu_angle:                 f32,
    /// log of the per-slot σ used for action sampling. Kept under
    /// the old field name for CSV stability.
    pub log_sigma_speed:          f32,
    pub log_sigma_angle:          f32,
    /// No critic in the new architecture — emitted as 0 to keep the
    /// column intact for downstream code.
    pub value_v:                  f32,
    /// Most recent reward stored in the per-slot ring buffer.
    pub last_reward:              f32,
    /// Unconditional mean of the last 64 rewards.
    pub mean_reward_64:           f32,
    /// K_EAT · Δpredations contribution from the most recent tick.
    pub last_eat_component:       f32,
    /// K_PROGRESS · distance_closed contribution from the most
    /// recent tick.
    pub last_progress_component:  f32,
    /// Retired oracle channel — kept for CSV stability, always 0.
    pub last_oracle_component:    f32,
}


// ── Pool resource ───────────────────────────────────────────────────────────

pub struct BrainPoolHerbivore1 {
    model:      PolicyNet<MyBackend>,
    opt:        Box<dyn BrainOpt>,
    free:       Vec<u32>,
    pub map:    HashMap<Entity, u32>,

    buf_states:  Vec<f32>,    // [N * ROLLOUT_LEN * IN]
    buf_actions: Vec<f32>,    // [N * ROLLOUT_LEN * OUT]
    buf_rewards: Vec<f32>,    // [N * ROLLOUT_LEN]
    buf_count:   Vec<usize>,  // [N]

    prev_state:         Vec<f32>,
    prev_action:        Vec<f32>,
    prev_energy:        Vec<f32>,
    prev_reproductions: Vec<u8>,
    prev_predations:    Vec<u8>,
    has_prev:           Vec<bool>,

    target_entity:           Vec<Option<Entity>>,
    lock_ticks_remaining:    Vec<u16>,
    prev_distance_to_target: Vec<f32>,
    prev_target_entity:      Vec<Option<Entity>>,
    min_dist_to_target:      Vec<f32>,
    ticks_since_progress:    Vec<u16>,
    blacklisted_target:      Vec<Option<Entity>>,
    blacklist_ticks_remaining: Vec<u16>,
    applied_speed_a:         Vec<f32>,

    baseline:    Vec<f32>,

    pub sigma:         Vec<f32>,
    pub k_eat:         Vec<f32>,
    pub k_repro:       Vec<f32>,
    pub lambda_energy: Vec<f32>,
    pub k_curiosity:   Vec<f32>,
    pub k_progress:    Vec<f32>,

    /// Most recent eat / progress reward contribution per slot —
    /// surfaced through `BrainTelemetry` for the dataset export.
    last_eat:                 Vec<f32>,
    last_progress:            Vec<f32>,
    /// Per-slot reward ring buffer (length 64). Mean over this is
    /// surfaced as `mean_reward_64`.
    recent_rewards:           Vec<f32>,
    recent_reward_head:       Vec<u8>,

    tick:        u64,

    /// Ring buffer of the last `TRAINING_HISTORY_CAP` training
    /// updates — populated at the bottom of `train()`. Read-only via
    /// `training_history()`.
    training_history:   VecDeque<TrainingStep>,
    step_counter:       u64,

    pub device:  CudaDevice,
}

impl BrainPoolHerbivore1 {
    fn new(device: CudaDevice, n: usize) -> Self {
        Self {
            model:              PolicyNet::<MyBackend>::new(&device, n),
            opt:                Box::new(AdamConfig::new().init()),
            free:               (0..n as u32).rev().collect(),
            map:                HashMap::with_capacity(n),
            buf_states:         vec![0.0; n * ROLLOUT_LEN * IN],
            buf_actions:        vec![0.0; n * ROLLOUT_LEN * OUT],
            buf_rewards:        vec![0.0; n * ROLLOUT_LEN],
            buf_count:          vec![0; n],
            prev_state:         vec![0.0; n * IN],
            prev_action:        vec![0.0; n * OUT],
            prev_energy:        vec![0.0; n],
            prev_reproductions: vec![0; n],
            prev_predations:    vec![0; n],
            has_prev:           vec![false; n],
            target_entity:      vec![None; n],
            lock_ticks_remaining:    vec![0; n],
            prev_distance_to_target: vec![0.0; n],
            prev_target_entity:      vec![None; n],
            min_dist_to_target:      vec![f32::INFINITY; n],
            ticks_since_progress:    vec![0; n],
            blacklisted_target:      vec![None; n],
            blacklist_ticks_remaining: vec![0; n],
            applied_speed_a:    vec![0.0; n],
            baseline:           vec![0.0; n],
            sigma:              vec![WARMUP_SIGMA; n],
            k_eat:              vec![0.0; n],
            k_repro:            vec![0.0; n],
            lambda_energy:      vec![0.0; n],
            k_curiosity:        vec![0.0; n],
            k_progress:         vec![0.0; n],
            last_eat:           vec![0.0; n],
            last_progress:      vec![0.0; n],
            recent_rewards:     vec![0.0; n * 64],
            recent_reward_head: vec![0u8; n],
            tick:               0,
            training_history:   VecDeque::with_capacity(TRAINING_HISTORY_CAP),
            step_counter:       0,
            device,
        }
    }

    pub fn n(&self) -> usize { self.buf_count.len() }

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
    }

    /// One GPU→CPU pull of the full weight + state arena. Iterated by
    /// `extract_slot` for per-organism slicing.
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

    /// Slice one slot's weights + prev_* state into a
    /// `BrainRestoreHerbivore1`. Convenience wrapper that pulls all
    /// four weight tensors from the GPU each call — for the colony
    /// save path use `snapshot()` once and call
    /// `PoolSnapshot::extract` per organism instead.
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
            w1: pull_3d(&self.model.w1.val(), IN, HIDDEN),
            b1: pull_2d(&self.model.b1.val(), HIDDEN),
            w2: pull_3d(&self.model.w2.val(), HIDDEN, OUT),
            b2: pull_2d(&self.model.b2.val(), OUT),
            prev_state:  self.prev_state [s * IN  .. (s+1) * IN ].to_vec(),
            prev_action: self.prev_action[s * OUT .. (s+1) * OUT].to_vec(),
            prev_energy: self.prev_energy[s],
            baseline:    self.baseline[s],
            has_prev:    self.has_prev[s],
        }
    }

    pub fn restore_slot(&mut self, slot: u32, r: &BrainRestoreHerbivore1) -> Result<(), String> {
        let slot_us = slot as usize;
        if r.w1.len() != IN * HIDDEN  { return Err(format!("w1 size {} != {}", r.w1.len(), IN * HIDDEN)); }
        if r.b1.len() != HIDDEN       { return Err(format!("b1 size {} != {}", r.b1.len(), HIDDEN)); }
        if r.w2.len() != HIDDEN * OUT { return Err(format!("w2 size {} != {}", r.w2.len(), HIDDEN * OUT)); }
        if r.b2.len() != OUT          { return Err(format!("b2 size {} != {}", r.b2.len(), OUT)); }
        if r.prev_state.len()  != IN  { return Err(format!("prev_state {} != {}", r.prev_state.len(), IN)); }
        if r.prev_action.len() != OUT { return Err(format!("prev_action {} != {}", r.prev_action.len(), OUT)); }

        let device = &self.device;
        let w1_t = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(r.w1.clone(), [1, IN, HIDDEN]), device);
        self.model.w1 = self.model.w1.clone().map(|t| {
            t.slice_assign([slot_us..slot_us + 1, 0..IN, 0..HIDDEN], w1_t)
        });
        let b1_t = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(r.b1.clone(), [1, HIDDEN]), device);
        self.model.b1 = self.model.b1.clone().map(|t| {
            t.slice_assign([slot_us..slot_us + 1, 0..HIDDEN], b1_t)
        });
        let w2_t = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(r.w2.clone(), [1, HIDDEN, OUT]), device);
        self.model.w2 = self.model.w2.clone().map(|t| {
            t.slice_assign([slot_us..slot_us + 1, 0..HIDDEN, 0..OUT], w2_t)
        });
        let b2_t = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(r.b2.clone(), [1, OUT]), device);
        self.model.b2 = self.model.b2.clone().map(|t| {
            t.slice_assign([slot_us..slot_us + 1, 0..OUT], b2_t)
        });

        let in_off  = slot_us * IN;
        let out_off = slot_us * OUT;
        self.prev_state [in_off  .. in_off + IN ].copy_from_slice(&r.prev_state);
        self.prev_action[out_off .. out_off + OUT].copy_from_slice(&r.prev_action);
        self.prev_energy[slot_us]   = r.prev_energy;
        self.baseline[slot_us]      = r.baseline;
        self.has_prev[slot_us]      = r.has_prev;
        self.buf_count[slot_us]            = 0;
        self.prev_reproductions[slot_us]   = 0;
        self.prev_predations[slot_us]      = 0;
        self.target_entity[slot_us]        = None;
        self.lock_ticks_remaining[slot_us] = 0;
        self.prev_distance_to_target[slot_us] = 0.0;
        self.prev_target_entity[slot_us]   = None;
        self.min_dist_to_target[slot_us]   = f32::INFINITY;
        self.ticks_since_progress[slot_us] = 0;
        self.blacklisted_target[slot_us]      = None;
        self.blacklist_ticks_remaining[slot_us] = 0;
        self.applied_speed_a[slot_us]      = 0.0;
        Ok(())
    }

    /// Read-only view of the recent training history. The export
    /// system writes these rows directly to the training-stats CSV.
    pub fn training_history(&self) -> &VecDeque<TrainingStep> {
        &self.training_history
    }

    /// Per-slot snapshot of the policy's most recent output + reward
    /// contributions. Computed via a single batched forward pass on
    /// every slot's cached `prev_state` — cost is one apply tick's
    /// worth of inference, run only when an export fires.
    pub fn snapshot_telemetry(&self) -> Vec<BrainTelemetry> {
        let n = self.n();
        let x = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(self.prev_state.clone(), [n, IN]),
            &self.device,
        );
        let mu = self.model.forward(x);
        let mu_data = mu.into_data().into_vec::<f32>().expect("telemetry mu");

        let mut out = Vec::with_capacity(n);
        for s in 0..n {
            let row = s * OUT;
            let mu_speed = mu_data[row + 0];
            // "Mu_angle" slot is repurposed to report the maximum
            // target logit (i.e. the agent's commitment strength to
            // its top-pick prey). Column name preserved for
            // downstream R-analysis compatibility.
            let max_logit = mu_data[row + 1 .. row + OUT]
                              .iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            let mut sum = 0.0_f32;
            for i in 0..64 { sum += self.recent_rewards[s * 64 + i]; }
            let mean_r = sum / 64.0;
            let head = self.recent_reward_head[s] as usize;
            let last_idx = if head == 0 { 63 } else { head - 1 };
            let last_r = self.recent_rewards[s * 64 + last_idx];

            let log_sigma = self.sigma[s].max(1e-6).ln();

            out.push(BrainTelemetry {
                mu_speed,
                mu_angle:                max_logit,
                log_sigma_speed:         log_sigma,
                log_sigma_angle:         log_sigma,
                value_v:                 0.0,
                last_reward:             last_r,
                mean_reward_64:          mean_r,
                last_eat_component:      self.last_eat[s],
                last_progress_component: self.last_progress[s],
                last_oracle_component:   0.0,
            });
        }
        out
    }
}

impl FromWorld for BrainPoolHerbivore1 {
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


fn warmup(device: &CudaDevice, n: usize) {
    let m = PolicyNet::<MyBackend>::new(device, n);
    let mut o: Box<dyn BrainOpt> = Box::new(AdamConfig::new().init());

    let i = Tensor::<MyBackend, 2>::zeros([n, IN], device);
    let _ = m.forward(i);

    let states  = Tensor::<MyBackend, 3>::zeros([n, ROLLOUT_LEN, IN],  device);
    let actions = Tensor::<MyBackend, 3>::zeros([n, ROLLOUT_LEN, OUT], device);
    let mask    = Tensor::<MyBackend, 3>::zeros([n, ROLLOUT_LEN, 1],   device);
    let adv     = Tensor::<MyBackend, 3>::zeros([n, ROLLOUT_LEN, 1],   device);
    let mu      = m.forward_rollout(states);
    let diff    = actions - mu;
    let sum_sq  = diff.powf_scalar(2.0).sum_dim(2);
    let scale   = 0.5_f32 / (WARMUP_SIGMA * WARMUP_SIGMA);
    let loss    = (sum_sq * adv * mask).sum().mul_scalar(scale).div_scalar(1.0_f32);
    let g = GradientsParams::from_grads(loss.backward(), &m);
    let _ = o.step(LR, m, g);
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
    for (e, organism, inheritance, restore) in new.iter() {
        // Enrol only Level1 non-carnivore heterotrophs. Carnivores
        // would go through L2; Level3 heteros go through L3.
        if !matches!(organism.intelligence_level, IntelligenceLevel::Level1) { continue; }

        let Some(slot) = pool.free.pop() else { continue };
        let s = slot as usize;

        let mut restored = false;
        let mut inherited_genes_from: Option<usize> = None;
        if let Some(r) = restore {
            match pool.restore_slot(slot, r) {
                Ok(())   => { restored = true; }
                Err(err) => error!("herbivore_1 brain restore failed for {e:?}: {err} — using fresh slot"),
            }
        } else if let Some(BrainInheritance(parent)) = inheritance {
            if let Some(&parent_slot) = pool.map.get(parent) {
                pool.inherit_row(parent_slot as usize, s);
                inherited_genes_from = Some(parent_slot as usize);
            }
        }

        if let Some(parent_slot) = inherited_genes_from {
            pool.sigma[s]         = mutate_in_range(pool.sigma[parent_slot],         L1_SIGMA_RANGE,         &mut rng);
            pool.k_eat[s]         = mutate_in_range(pool.k_eat[parent_slot],         L1_K_EAT_RANGE,         &mut rng);
            pool.k_repro[s]       = mutate_in_range(pool.k_repro[parent_slot],       L1_K_REPRO_RANGE,       &mut rng);
            pool.lambda_energy[s] = mutate_in_range(pool.lambda_energy[parent_slot], L1_LAMBDA_ENERGY_RANGE, &mut rng);
            pool.k_curiosity[s]   = mutate_in_range(pool.k_curiosity[parent_slot],   L1_K_CURIOSITY_RANGE,   &mut rng);
            pool.k_progress[s]    = mutate_in_range(pool.k_progress[parent_slot],    L1_K_PROGRESS_RANGE,    &mut rng);
        } else {
            pool.sigma[s]         = sample_range(L1_SIGMA_RANGE,         &mut rng);
            pool.k_eat[s]         = sample_range(L1_K_EAT_RANGE,         &mut rng);
            pool.k_repro[s]       = sample_range(L1_K_REPRO_RANGE,       &mut rng);
            pool.lambda_energy[s] = sample_range(L1_LAMBDA_ENERGY_RANGE, &mut rng);
            pool.k_curiosity[s]   = sample_range(L1_K_CURIOSITY_RANGE,   &mut rng);
            pool.k_progress[s]    = sample_range(L1_K_PROGRESS_RANGE,    &mut rng);
        }

        pool.target_entity[s]           = None;
        pool.lock_ticks_remaining[s]    = 0;
        pool.prev_distance_to_target[s] = 0.0;
        pool.prev_target_entity[s]      = None;
        pool.min_dist_to_target[s]      = f32::INFINITY;
        pool.ticks_since_progress[s]    = 0;
        pool.blacklisted_target[s]      = None;
        pool.blacklist_ticks_remaining[s] = 0;
        pool.applied_speed_a[s]         = 0.0;
        pool.prev_predations[s]         = organism.predations;
        pool.last_eat[s]                = 0.0;
        pool.last_progress[s]           = 0.0;
        // Per-slot ring buffer reset — fresh tenant starts with zero
        // recent rewards rather than inheriting the previous occupant's.
        for i in 0..64 { pool.recent_rewards[s * 64 + i] = 0.0; }
        pool.recent_reward_head[s] = 0;

        if !restored {
            pool.has_prev[s]           = false;
            pool.baseline[s]           = 0.0;
            pool.prev_energy[s]        = organism.energy;
            pool.prev_reproductions[s] = organism.reproductions;
            pool.buf_count[s]          = 0;
        } else {
            pool.prev_reproductions[s] = organism.reproductions;
            pool.buf_count[s]          = 0;
        }

        pool.map.insert(e, slot);
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
        if let Some(slot) = pool.map.remove(&e) {
            let s = slot as usize;
            pool.has_prev[s]             = false;
            pool.buf_count[s]            = 0;
            pool.target_entity[s]        = None;
            pool.lock_ticks_remaining[s] = 0;
            pool.prev_target_entity[s]   = None;
            pool.min_dist_to_target[s]   = f32::INFINITY;
            pool.ticks_since_progress[s] = 0;
            pool.blacklisted_target[s]      = None;
            pool.blacklist_ticks_remaining[s] = 0;
            pool.applied_speed_a[s]      = 0.0;
            pool.free.push(slot);
        }
    }
}


// ── Apply / train tick ──────────────────────────────────────────────────────

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

    input_buf.clear();
    input_buf.resize(n * IN, 0.0);

    struct Active {
        entity:        Entity,
        slot:          u32,
        pos:           Vec3,
        energy_now:    f32,
        reproductions: u8,
        predations:    u8,
        neighbours:    [Option<crate::world_model::Neighbour>; WORLD_MODEL_K],
    }
    let mut active: Vec<Active> = Vec::new();

    for (e, organism, transform, slot) in heteros.iter() {
        let s = slot.0 as usize;
        if s >= n { continue; }
        let pos = transform.translation;

        let max_e    = crate::energy::get_max_energy(&organism).max(1.0);
        let energy_n = (organism.energy / max_e).clamp(0.0, 1.0);
        let neighbours = collect_neighbours(&world_grid, pos);

        let off = s * IN;
        input_buf[off] = energy_n;
        encode_neighbours(&neighbours, &mut input_buf[off + 1 .. off + 1 + WORLD_MODEL_DIMS]);

        let pa_off = off + 1 + WORLD_MODEL_DIMS;
        if pool.has_prev[s] {
            let src = s * OUT;
            for i in 0..OUT {
                input_buf[pa_off + i] = pool.prev_action[src + i];
            }
        }

        let flag_off = pa_off + PREV_ACTION_DIMS;
        input_buf[flag_off] = if pool.target_entity[s].is_some() { 1.0 } else { 0.0 };

        active.push(Active {
            entity: e, slot: slot.0, pos,
            energy_now: organism.energy,
            reproductions: organism.reproductions,
            predations: organism.predations,
            neighbours,
        });
    }
    if active.is_empty() {
        pool.tick = pool.tick.wrapping_add(1);
        return;
    }

    let cur_t = Tensor::<MyBackend, 2>::from_data(
        TensorData::new(input_buf.clone(), [n, IN]),
        &pool.device,
    );
    let mu_cur  = pool.model.forward(cur_t);
    let mu_data = mu_cur.into_data().into_vec::<f32>().expect("forward output");

    let mut rng = rand::rng();

    for a in &active {
        let s = a.slot as usize;

        // (3a) Reward for the previous action (if any).
        if pool.has_prev[s] {
            let new_eats   = a.predations.wrapping_sub(pool.prev_predations[s]);
            let new_repros = a.reproductions.saturating_sub(pool.prev_reproductions[s]);
            let energy_delta = a.energy_now - pool.prev_energy[s];

            let r_eat   = pool.k_eat[s]   * new_eats   as f32;
            let r_repro = pool.k_repro[s] * new_repros as f32;
            let r_energy = if energy_delta < 0.0 {
                pool.lambda_energy[s] * energy_delta
            } else { 0.0 };

            let prev_speed_norm = pool.applied_speed_a[s].max(0.0);
            let r_curiosity = pool.k_curiosity[s] * (prev_speed_norm - 0.5);

            let r_progress = if let Some(prev_target) = pool.prev_target_entity[s] {
                let cur_d = a.neighbours.iter()
                    .find_map(|n| n.and_then(|nn|
                        if nn.entity == prev_target { Some(nn.rel.length()) } else { None }
                    ));
                match cur_d {
                    Some(d) => {
                        let closed = pool.prev_distance_to_target[s] - d;
                        pool.k_progress[s] * closed.max(0.0)
                    }
                    None => 0.0,
                }
            } else { 0.0 };

            let r = r_eat + r_repro + r_energy + r_curiosity + r_progress;

            // Surface reward components for telemetry.
            pool.last_eat[s]      = r_eat;
            pool.last_progress[s] = r_progress;
            let head = pool.recent_reward_head[s] as usize;
            pool.recent_rewards[s * 64 + head] = r;
            pool.recent_reward_head[s] = ((head + 1) % 64) as u8;

            let count = pool.buf_count[s];
            if count < ROLLOUT_LEN {
                let prev_state  = pool.prev_state[s * IN .. (s + 1) * IN].to_vec();
                let prev_action = pool.prev_action[s * OUT .. (s + 1) * OUT].to_vec();
                let buf_in_off  = s * ROLLOUT_LEN * IN + count * IN;
                let buf_out_off = s * ROLLOUT_LEN * OUT + count * OUT;
                pool.buf_states [buf_in_off  .. buf_in_off  + IN ].copy_from_slice(&prev_state);
                pool.buf_actions[buf_out_off .. buf_out_off + OUT].copy_from_slice(&prev_action);
                pool.buf_rewards[s * ROLLOUT_LEN + count] = r;
                pool.buf_count[s] = count + 1;
            }
        }

        // (3b) Sample a new action.
        let off = s * OUT;
        let sigma_s = pool.sigma[s];
        let mut action = [0.0_f32; OUT];
        for i in 0..OUT {
            action[i] = mu_data[off + i] + sigma_s * gaussian_noise(&mut rng);
        }
        let speed_a = action[0].clamp(-1.0, 1.0);
        let target_logits = [action[1], action[2], action[3], action[4]];

        // (3c-pre) Stuck detection.
        let cur_locked_dist = pool.target_entity[s].and_then(|tgt| {
            a.neighbours.iter().find_map(|n|
                n.and_then(|nn| if nn.entity == tgt {
                    Some((nn.rel.x * nn.rel.x + nn.rel.z * nn.rel.z).sqrt())
                } else { None })
            )
        });
        if let Some(d) = cur_locked_dist {
            if d + L1_STUCK_PROGRESS_EPS < pool.min_dist_to_target[s] {
                pool.min_dist_to_target[s] = d;
                pool.ticks_since_progress[s] = 0;
            } else if d > L1_APPROACH_RADIUS {
                pool.ticks_since_progress[s] = pool.ticks_since_progress[s].saturating_add(1);
            }
        }
        let force_drop = pool.ticks_since_progress[s] >= L1_STUCK_TICKS;

        if pool.blacklist_ticks_remaining[s] > 0 {
            pool.blacklist_ticks_remaining[s] -= 1;
            if pool.blacklist_ticks_remaining[s] == 0 {
                pool.blacklisted_target[s] = None;
            }
        }
        if force_drop {
            pool.blacklisted_target[s] = pool.target_entity[s];
            pool.blacklist_ticks_remaining[s] = L1_TARGET_BLACKLIST_TICKS;
        }
        let blacklisted_entity = pool.blacklisted_target[s];

        // (3c) Target selection — herbivores hunt PHOTOAUTOTROPHS.
        let locked_entity = if force_drop { None } else { pool.target_entity[s] };
        let mut locked_slot: Option<usize> = None;
        let mut best_slot:   Option<usize> = None;
        let mut best_logit:        f32      = f32::NEG_INFINITY;

        for (i, slot) in a.neighbours.iter().enumerate() {
            let Some(nn) = slot else { continue };
            // Herbivore (non-carnivore) — prey is Photo.
            if nn.ty != OrganismType::Photo { continue; }
            if Some(nn.entity) == blacklisted_entity { continue; }
            let l = target_logits[i];
            if l > best_logit { best_logit = l; best_slot = Some(i); }
            if Some(nn.entity) == locked_entity { locked_slot = Some(i); }
        }

        let chosen_slot: Option<usize> = match (locked_slot, best_slot) {
            (Some(li), _) if pool.lock_ticks_remaining[s] > 0 => {
                pool.lock_ticks_remaining[s] -= 1;
                Some(li)
            }
            (Some(li), Some(bi)) => {
                let chosen = if bi != li
                    && best_logit > target_logits[li] + L1_TARGET_SWITCH_MARGIN
                { bi } else { li };
                pool.lock_ticks_remaining[s] = L1_TARGET_LOCK_TICKS;
                Some(chosen)
            }
            (None, Some(bi)) => {
                pool.lock_ticks_remaining[s] = L1_TARGET_LOCK_TICKS;
                Some(bi)
            }
            _ => {
                pool.lock_ticks_remaining[s] = 0;
                None
            }
        };

        let new_target: Option<(Entity, Vec3)> = chosen_slot
            .and_then(|i| a.neighbours[i].map(|n| (n.entity, n.rel)));

        // (3d) Apply to the world.
        let Ok((_, mut org, _, _)) = heteros.get_mut(a.entity) else { continue };

        let smoothed = L1_SPEED_MOMENTUM_ALPHA * pool.applied_speed_a[s]
                     + (1.0 - L1_SPEED_MOMENTUM_ALPHA) * speed_a;
        pool.applied_speed_a[s] = smoothed;

        let brake_scale = match new_target {
            Some((_, rel)) => {
                let d_xz = (rel.x * rel.x + rel.z * rel.z).sqrt();
                (d_xz / L1_APPROACH_RADIUS).clamp(0.0, 1.0)
            }
            None => L1_NO_TARGET_SPEED_SCALE,
        };
        let applied_floored = smoothed.max(L1_MIN_APPLIED_SPEED);
        org.movement_speed = applied_floored * MAX_SPEED * brake_scale;

        match new_target {
            Some((_, rel)) => {
                let dx = rel.x;
                let dz = rel.z;
                let mag2 = dx * dx + dz * dz;
                if mag2 > L1_DIRECTION_FREEZE_DIST * L1_DIRECTION_FREEZE_DIST {
                    let inv = mag2.sqrt().recip();
                    org.movement_direction = Vec3::new(dx * inv, 0.0, dz * inv);
                }
            }
            None => {
                use rand::RngExt;
                let angle = (rng.random::<f32>() - 0.5) * 2.0 * L1_NO_TARGET_WANDER_ANGLE;
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let dx = org.movement_direction.x;
                let dz = org.movement_direction.z;
                let new_dx = dx * cos_a - dz * sin_a;
                let new_dz = dx * sin_a + dz * cos_a;
                let len_sq = new_dx * new_dx + new_dz * new_dz;
                if len_sq > 1e-6 {
                    let inv = len_sq.sqrt().recip();
                    org.movement_direction = Vec3::new(new_dx * inv, 0.0, new_dz * inv);
                }
            }
        }

        // (3e) Update prev_* state for next tick.
        let in_off = s * IN;
        for i in 0..IN { pool.prev_state[in_off + i] = input_buf[in_off + i]; }
        pool.prev_action[off + 0] = speed_a;
        pool.prev_action[off + 1] = target_logits[0];
        pool.prev_action[off + 2] = target_logits[1];
        pool.prev_action[off + 3] = target_logits[2];
        pool.prev_action[off + 4] = target_logits[3];
        pool.prev_energy[s]        = a.energy_now;
        pool.prev_reproductions[s] = a.reproductions;
        pool.prev_predations[s]    = a.predations;
        pool.has_prev[s]           = true;

        let prev_target_ent = pool.target_entity[s];
        pool.prev_target_entity[s] = prev_target_ent;
        let new_target_ent = new_target.map(|(e, _)| e);
        if let Some((ent, rel)) = new_target {
            pool.target_entity[s]           = Some(ent);
            pool.prev_distance_to_target[s] = rel.length();
        } else {
            pool.target_entity[s]           = None;
            pool.prev_distance_to_target[s] = 0.0;
        }

        if new_target_ent != prev_target_ent {
            pool.min_dist_to_target[s] = match new_target {
                Some((_, rel)) => (rel.x * rel.x + rel.z * rel.z).sqrt(),
                None           => f32::INFINITY,
            };
            pool.ticks_since_progress[s] = 0;
        }
    }

    pool.tick = pool.tick.wrapping_add(1);
    if pool.tick % ROLLOUT_LEN as u64 != 0 { return; }
    train(&mut pool, time.elapsed_secs());
}


fn train(pool: &mut BrainPoolHerbivore1, virtual_time_secs: f32) {
    let n = pool.n();

    let mut adv_buf  = vec![0.0_f32; n * ROLLOUT_LEN];
    let mut mask_buf = vec![0.0_f32; n * ROLLOUT_LEN];
    let mut total_count = 0.0_f32;
    let mut active_slots: u32 = 0;
    let mut return_sum:  f64 = 0.0;
    let mut return_sumsq: f64 = 0.0;
    let mut return_n:    u64 = 0;

    for s in 0..n {
        let count = pool.buf_count[s];
        if count == 0 { continue; }
        active_slots += 1;

        let mut returns = vec![0.0_f32; count];
        let mut g = 0.0_f32;
        for t in (0..count).rev() {
            let r = pool.buf_rewards[s * ROLLOUT_LEN + t];
            g = r + GAMMA * g;
            returns[t] = g;
            return_sum += g as f64;
            return_sumsq += (g as f64) * (g as f64);
            return_n += 1;
        }

        let mean_g: f32 = returns.iter().sum::<f32>() / count as f32;
        pool.baseline[s] = (1.0 - BASELINE_ALPHA) * pool.baseline[s]
                         + BASELINE_ALPHA * mean_g;

        for t in 0..count {
            adv_buf [s * ROLLOUT_LEN + t] = returns[t] - pool.baseline[s];
            mask_buf[s * ROLLOUT_LEN + t] = 1.0;
            total_count += 1.0;
        }
    }

    if total_count < 1.0 {
        for s in 0..n { pool.buf_count[s] = 0; }
        return;
    }

    let states_t = Tensor::<MyBackend, 3>::from_data(
        TensorData::new(pool.buf_states.clone(),  [n, ROLLOUT_LEN, IN]),
        &pool.device,
    );
    let actions_t = Tensor::<MyBackend, 3>::from_data(
        TensorData::new(pool.buf_actions.clone(), [n, ROLLOUT_LEN, OUT]),
        &pool.device,
    );
    let adv_t = Tensor::<MyBackend, 3>::from_data(
        TensorData::new(adv_buf, [n, ROLLOUT_LEN, 1]),
        &pool.device,
    );
    let mask_t = Tensor::<MyBackend, 3>::from_data(
        TensorData::new(mask_buf, [n, ROLLOUT_LEN, 1]),
        &pool.device,
    );

    let mut scale_buf = vec![0.0_f32; n];
    for s in 0..n {
        let sigma = pool.sigma[s].max(1e-3);
        scale_buf[s] = 0.5 / (sigma * sigma);
    }
    let scale_t = Tensor::<MyBackend, 3>::from_data(
        TensorData::new(scale_buf, [n, 1, 1]),
        &pool.device,
    );

    let mu = pool.model.forward_rollout(states_t);
    let diff = actions_t - mu;
    let sum_sq = diff.powf_scalar(2.0).sum_dim(2);
    let loss = (sum_sq * adv_t * mask_t * scale_t).sum()
                  .div_scalar(total_count);

    // Extract the scalar loss for the training-history record before
    // backprop (the tensor's data API consumes the graph on read).
    let loss_val = loss.clone().into_data().into_vec::<f32>().expect("loss")[0];

    let cm = pool.model.clone();
    let gp = GradientsParams::from_grads(loss.backward(), &pool.model);
    pool.model = pool.opt.step(LR, cm, gp);

    let mean_return = if return_n > 0 { (return_sum / return_n as f64) as f32 } else { 0.0 };
    let return_var  = if return_n > 1 {
        let mean = return_sum / return_n as f64;
        let var  = (return_sumsq / return_n as f64) - mean * mean;
        var.max(0.0) as f32
    } else { 0.0 };

    pool.step_counter = pool.step_counter.saturating_add(1);
    if pool.training_history.len() >= TRAINING_HISTORY_CAP {
        pool.training_history.pop_front();
    }
    pool.training_history.push_back(TrainingStep {
        step:              pool.step_counter,
        virtual_time_secs,
        n_active:          active_slots,
        actor_loss:        loss_val,
        critic_loss:       0.0,
        entropy:           0.0,
        total_loss:        loss_val,
        mean_return,
        return_var,
        supervised_loss:   0.0,
    });

    for s in 0..n { pool.buf_count[s] = 0; }
}
