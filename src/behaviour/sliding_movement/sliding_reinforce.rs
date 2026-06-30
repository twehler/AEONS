// Shared REINFORCE engine for the sliding brain pools (herbivore_1, L2, L3).
//
// The three sliding pools were ~99% identical; this module owns the ONE
// implementation and the per-level files (`intelligence_level_*_sliding.rs`)
// are thin newtype wrappers that delegate to it — exactly mirroring how the
// limb-based pools wrap the shared `limb_ppo::BrainPoolLimb` engine.
//
// SHARED-POLICY (per-SPECIES) model: every organism of a species shares ONE
// network — all members' experience trains the SAME weights, so the population
// learns as a single agent with ~N× the data. Different species hold different
// weights (behaviour diverges by species); same-species individuality is
// preserved only as momentary exploration noise. Keeps REINFORCE (Monte-Carlo
// policy gradient + per-SPECIES EMA baseline).
//
// Architecture (the `BrainRestore` save format keys off IN/HIDDEN/OUT and is
// validated at restore, so HIDDEN may differ per level — herbivore_1 = 32,
// L2 = 16, L3 = 32 — without breaking the format; a mismatch degrades to a
// fresh net rather than panicking):
//   * Network: IN → HIDDEN ReLU → OUT (tanh).
//   * Inputs (IN=31): energy + 24-dim world model (K=4 neighbours) +
//     5-dim prev action + 1 locked-target flag.
//   * Outputs (OUT=5): speed_a + 4 target logits. Speed = max(0,speed_a)*MAX_SPEED;
//     direction is geometric from the chosen target.
//
// Reward (attributed to the previous action each tick):
//   K_EAT·eats + K_REPRO·repros + LAMBDA·min(0,ΔE)
//     + K_CURIOSITY·(speed_norm−0.5) + K_PROGRESS·max(0,distance_closed).
//
// Algorithm: Monte Carlo REINFORCE with per-SPECIES EMA baseline; rollout
// ring buffer of length ROLLOUT_LEN, batched train every ROLLOUT_LEN ticks,
// grouped per species.
//
// Per-LEVEL axes (everything else is shared) are carried in `SlidingConfig`:
//   * `hidden`    — hidden width (32 / 16 / 32).
//   * `prey_type` — herbivore_1 always Photo; L2/L3 = Hetero if carnivore else
//                   Photo. (The IntelligenceLevel filter + the `Without<Carnivore>`
//                   query bound stay in each wrapper's `assign_*` system.)

use bevy::prelude::*;
use burn::module::{AutodiffModule, Initializer, Module, Param};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use burn_cuda::CudaDevice;
use std::collections::{HashMap, VecDeque};

use crate::colony::{Organism, Carnivore};
use crate::rl_helpers::{BrainRestore, MyBackend, PoolSnapshot, SpeciesWeights, gaussian_noise};
use crate::simulation_settings::{
    L1_SIGMA_RANGE,
    L1_K_EAT_RANGE,
    L1_K_REPRO_RANGE,
    L1_LAMBDA_ENERGY_RANGE,
    L1_K_CURIOSITY_RANGE,
    L1_K_PROGRESS_RANGE,
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


// ── Architecture constants (IN/OUT identical for all three levels; HIDDEN is
//   per-level via `SlidingConfig`). ──────────────────────────────────────────

const PREV_ACTION_DIMS:    usize = 5;
const LOCKED_FLAG_DIMS:    usize = 1;

/// Brain-tick interval in *virtual* seconds. Mirrors
/// `behaviour.rs::HETERO_BRAIN_TICK_INTERVAL` (150 ms) — kept here as an `f32`
/// so the target-lock seconds-to-ticks conversion can run at const time. If you
/// change one, change the other.
const HETERO_BRAIN_TICK_SECS: f32 = 0.150;

/// Number of brain ticks the target-lock window spans, derived from
/// `L1_TARGET_LOCK_SECS / HETERO_BRAIN_TICK_SECS` (ceiled). Since the brain runs
/// on `Time<Virtual>`, a constant tick count gives a constant *virtual*-time
/// lock duration, which scales with `TimeSpeed` for real-time observation.
const L1_TARGET_LOCK_TICKS: u16 = {
    let f = L1_TARGET_LOCK_SECS / HETERO_BRAIN_TICK_SECS;
    let truncated = f as u16;
    if (truncated as f32) < f { truncated + 1 } else { truncated }
};

pub const IN:      usize = 1 + WORLD_MODEL_DIMS + PREV_ACTION_DIMS + LOCKED_FLAG_DIMS;
/// `(speed_a, target_logit_0, target_logit_1, target_logit_2, target_logit_3)`.
pub const OUT:     usize = 1 + WORLD_MODEL_K;
const ROLLOUT_LEN: usize = 32;
const MAX_SPEED:   f32   = 20.0;  // u/s — sized for the 250-unit map (was 40, which read as a fixed sprint on the smaller, submerged world)
const LR:          f64   = 1e-3;

/// Discount for the Monte Carlo return.
const GAMMA:           f32 = 0.95;
/// EMA decay rate for the per-species baseline (mean of recent returns).
const BASELINE_ALPHA:  f32 = 0.05;

const TRAINING_HISTORY_CAP: usize = 1024;

/// Sentinel species key for organisms not yet classified by the speciation
/// system (real `species_id`s start at 1). All unclassified heterotrophs share
/// this one transitional net until the ~1 Hz speciation tick assigns them; the
/// next apply tick then re-points them at their real species' net automatically.
pub const UNCLASSIFIED: u32 = 0;


/// Per-LEVEL configuration. The ONLY axes that differ between the three sliding
/// pools beyond their public symbol names.
#[derive(Clone, Copy)]
pub struct SlidingConfig {
    /// Hidden layer width (herbivore_1 = 32, L2 = 16, L3 = 32).
    pub hidden: usize,
    /// Prey filter given an organism's carnivore flag. herbivore_1 ignores the
    /// flag (always `Photo`); L2/L3 hunt `Hetero` when carnivore else `Photo`.
    pub prey_type: fn(is_carnivore: bool) -> OrganismType,
    /// When `false` (all terrestrial/benthic sliding pools), the geometric pursuit
    /// direction is XZ-only (Y forced to 0) — sliders crawl on a surface. When
    /// `true` (the SimpleAquatic 3D pool), the toward-target direction preserves its
    /// Y component so the mover pursues prey in full 3D through the water volume.
    pub direction_3d: bool,
}


/// Uniform sample within `(min, max)`.
fn sample_range(range: (f32, f32), rng: &mut impl rand::Rng) -> f32 {
    use rand::RngExt;
    let (lo, hi) = range;
    lo + (hi - lo) * rng.random::<f32>()
}


// ── Slot trait ────────────────────────────────────────────────────────────────

/// Marker-component trait so the shared apply step can extract a slot index
/// regardless of which per-level component (`BrainSlotHerbivore1`, `BrainSlotL2`,
/// `BrainSlotL3`) the query carries. Mirrors `limb_ppo::LimbSlot`.
pub trait SlidingSlot {
    fn slot(&self) -> u32;
}


// ── Shared (per-species) MLP ──────────────────────────────────────────────────
//
// 2-D weights (`[IN, HIDDEN]`, `[HIDDEN, OUT]`); biases `[1, ·]` so a forward
// over a batch of `M` observations is a plain `[M, IN] · [IN, HIDDEN]` matmul
// with broadcast bias. Serves both inference and the pooled REINFORCE update.

#[derive(Module, Debug)]
pub struct PolicyNet<B: Backend> {
    pub w1: Param<Tensor<B, 2>>,  // [IN, HIDDEN]
    pub b1: Param<Tensor<B, 2>>,  // [1, HIDDEN]
    pub w2: Param<Tensor<B, 2>>,  // [HIDDEN, OUT]
    pub b2: Param<Tensor<B, 2>>,  // [1, OUT]
}

impl<B: Backend> PolicyNet<B> {
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = burn::tensor::activation::relu(x.matmul(self.w1.val()) + self.b1.val());
        burn::tensor::activation::tanh(h.matmul(self.w2.val()) + self.b2.val())
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


/// One species' shared net + its optimiser + its REINFORCE EMA baseline.
/// Created lazily by `new_species_brain` (fresh-init weights — sliding has no
/// oscillatory warm-start). `tick` counts apply ticks for this species.
pub struct SpeciesBrain {
    pub model:    PolicyNet<MyBackend>,
    pub opt:      Box<dyn BrainOpt>,
    /// Per-species REINFORCE baseline (EMA over mean rollout return).
    pub baseline: f32,
    pub tick:     u64,
}

/// Build one freshly-initialised species brain. Weights uniform in [-0.5,0.5]
/// (the original sliding init), biases zero. No oscillatory warm-start.
pub fn new_species_brain(device: &CudaDevice, hidden: usize) -> SpeciesBrain {
    let w = Initializer::Uniform { min: -0.5, max: 0.5 };
    let z = Initializer::Zeros;
    let model = PolicyNet::<MyBackend> {
        w1: w.init([IN, hidden], device),
        b1: z.init([1, hidden],  device),
        w2: w.init([hidden, OUT], device),
        b2: z.init([1, OUT],     device),
    };
    let opt: Box<dyn BrainOpt> = Box::new(AdamConfig::new().init());
    SpeciesBrain { model, opt, baseline: 0.0, tick: 0 }
}


// ── Save / load payload ─────────────────────────────────────────────────────

/// Re-export of the shared `BrainRestore` (one 2-D species net's flat weights
/// + REINFORCE prev_*).
pub type BrainRestoreHerbivore1 = BrainRestore;

/// Magic prefixed to the brain block so a format change / stale data is caught
/// at decode time.
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

/// Deserialise a `BrainRestoreHerbivore1` from `bytes[*c..]`, advancing `*c`.
/// Hard-errors on magic / shape mismatch — the caller logs the failure and
/// continues with a fresh brain slot for that organism.
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
    // Length-prefixed reads, but WITHOUT a fixed-size check here (HIDDEN varies
    // per level, and the consuming `restore_species` validates against the
    // pool's actual HIDDEN). Read the declared count verbatim.
    let read_vec = |c: &mut usize| -> std::io::Result<Vec<f32>> {
        let n = read_u32(bytes, c)? as usize;
        let mut v = Vec::with_capacity(n.min(1 << 24));
        for _ in 0..n { v.push(read_f32(bytes, c)?); }
        Ok(v)
    };
    let w1          = read_vec(c)?;
    let b1          = read_vec(c)?;
    let w2          = read_vec(c)?;
    let b2          = read_vec(c)?;
    let prev_state  = read_vec(c)?;
    let prev_action = read_vec(c)?;
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

/// One row of the training-statistics CSV, logged once per REINFORCE update.
/// Fields match the columns the R analysis scripts expect; channels that don't
/// apply here (critic_loss, entropy, supervised_loss) are zeroed for CSV-format
/// stability.
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

/// Per-slot snapshot for `dataset_export` and `time_series_log`. Fields kept
/// stable for the R analysis scripts: channels that don't apply (`value_v`,
/// `last_oracle_component`) carry the closest analogue or zero.
#[derive(Clone, Debug)]
pub struct BrainTelemetry {
    /// Speed channel raw output (post-tanh, before EMA / brake).
    pub mu_speed:                 f32,
    /// Largest target logit (post-tanh) — commitment to the favoured prey slot.
    /// Reuses the `mu_angle` CSV column name.
    pub mu_angle:                 f32,
    /// ln of the per-slot σ used for sampling (old field name, CSV stable).
    pub log_sigma_speed:          f32,
    pub log_sigma_angle:          f32,
    /// No critic — emitted as 0 to keep the column intact.
    pub value_v:                  f32,
    /// Most recent reward in the per-slot ring buffer.
    pub last_reward:              f32,
    /// Mean of the last 64 rewards.
    pub mean_reward_64:           f32,
    /// K_EAT · Δpredations from the most recent tick.
    pub last_eat_component:       f32,
    /// K_PROGRESS · distance_closed from the most recent tick.
    pub last_progress_component:  f32,
    /// Retired oracle channel — always 0, kept for CSV stability.
    pub last_oracle_component:    f32,
}


// ── Pool (shared engine) ───────────────────────────────────────────────────

/// Shared engine wrapped by each per-level pool resource. Holds the per-species
/// nets, the per-slot REINFORCE bookkeeping, the free-slot list, and the
/// entity↔slot map. The per-level pool resources (`BrainPoolHerbivore1`,
/// `BrainPoolL2`, `BrainPoolL3`) own one of these (newtype + `Deref`) and a
/// `SlidingConfig`.
pub struct BrainPoolSliding {
    pub cfg: SlidingConfig,

    /// Per-species shared nets (model + optimiser + baseline). Keyed by
    /// `Organism::species_id` (`UNCLASSIFIED` until first classification);
    /// created lazily by `ensure_species`.
    species:    HashMap<u32, SpeciesBrain>,
    /// Per-slot species key as of the last apply tick.
    slot_species: Vec<u32>,
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

    /// Per-slot reward-shaping + exploration hyperparameters ("genes"). `pub`
    /// so the lineages plugin's `sync_dna_from_brain_pool` can read them.
    pub sigma:         Vec<f32>,
    pub k_eat:         Vec<f32>,
    pub k_repro:       Vec<f32>,
    pub lambda_energy: Vec<f32>,
    pub k_curiosity:   Vec<f32>,
    pub k_progress:    Vec<f32>,

    /// Most recent eat / progress reward contribution per slot — surfaced
    /// through `BrainTelemetry` for the dataset export.
    last_eat:                 Vec<f32>,
    last_progress:            Vec<f32>,
    /// Per-slot reward ring buffer (length 64). Mean is `mean_reward_64`.
    recent_rewards:           Vec<f32>,
    recent_reward_head:       Vec<u8>,

    tick:        u64,

    /// Ring buffer of the last `TRAINING_HISTORY_CAP` training updates.
    training_history:   VecDeque<TrainingStep>,
    step_counter:       u64,

    pub device:  CudaDevice,
}

impl BrainPoolSliding {
    pub fn new(device: CudaDevice, n: usize, cfg: SlidingConfig) -> Self {
        Self {
            cfg,
            species:            HashMap::new(),
            slot_species:       vec![UNCLASSIFIED; n],
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
            sigma:              vec![0.0; n],
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

    fn hidden(&self) -> usize { self.cfg.hidden }

    /// Lazily create a species' shared brain (fresh-init) on first sighting.
    pub fn ensure_species(&mut self, key: u32) {
        if !self.species.contains_key(&key) {
            let brain = new_species_brain(&self.device, self.cfg.hidden);
            self.species.insert(key, brain);
        }
    }

    /// Drop the shared nets of species with no enrolled member left. Without
    /// this the per-species map — and its GPU tensors + Adam state — grows
    /// unbounded over a long run. `UNCLASSIFIED` is always kept.
    fn prune_species(&mut self) {
        let mut live: std::collections::HashSet<u32> =
            self.map.values().map(|&s| self.slot_species[s as usize]).collect();
        live.insert(UNCLASSIFIED);
        self.species.retain(|k, _| live.contains(k));
    }

    /// One GPU→CPU pull of each live species' shared weights, plus the per-slot
    /// REINFORCE prev_* state. Iterated by `PoolSnapshot::extract`.
    pub fn snapshot(&self) -> PoolSnapshot {
        let mut species: HashMap<u32, SpeciesWeights> = HashMap::new();
        for (&key, brain) in &self.species {
            species.insert(key, SpeciesWeights {
                w1: brain.model.w1.val().into_data().into_vec::<f32>().expect("w1 to vec"),
                b1: brain.model.b1.val().into_data().into_vec::<f32>().expect("b1 to vec"),
                w2: brain.model.w2.val().into_data().into_vec::<f32>().expect("w2 to vec"),
                b2: brain.model.b2.val().into_data().into_vec::<f32>().expect("b2 to vec"),
                baseline: brain.baseline,
            });
        }
        let mut entity_species: HashMap<Entity, u32> = HashMap::with_capacity(self.map.len());
        for (&e, &slot) in &self.map {
            entity_species.insert(e, self.slot_species[slot as usize]);
        }
        PoolSnapshot {
            species,
            entity_species,
            map:         self.map.clone(),
            prev_state:  self.prev_state.clone(),
            prev_action: self.prev_action.clone(),
            prev_energy: self.prev_energy.clone(),
            has_prev:    self.has_prev.clone(),
            in_dim:      IN,
            hidden_dim:  self.cfg.hidden,
            out_dim:     OUT,
        }
    }

    /// Slice one slot's SPECIES net weights + that slot's prev_* state into a
    /// `BrainRestoreHerbivore1`. Pulls the species net's four weight tensors
    /// from the GPU each call. The slot's species is read from `slot_species`;
    /// an unknown species degrades to a fresh net's weights.
    pub fn extract_slot(&self, slot: u32) -> BrainRestoreHerbivore1 {
        let s = slot as usize;
        let key = self.slot_species[s];
        let pull_2d = |t: &Tensor<MyBackend, 2>| {
            t.clone().into_data().into_vec::<f32>().expect("2d net pull")
        };
        let hidden = self.cfg.hidden;
        let (w1, b1, w2, b2, baseline) = match self.species.get(&key) {
            Some(brain) => (
                pull_2d(&brain.model.w1.val()),
                pull_2d(&brain.model.b1.val()),
                pull_2d(&brain.model.w2.val()),
                pull_2d(&brain.model.b2.val()),
                brain.baseline,
            ),
            None => (
                vec![0.0; IN * hidden],
                vec![0.0; hidden],
                vec![0.0; hidden * OUT],
                vec![0.0; OUT],
                0.0,
            ),
        };
        BrainRestoreHerbivore1 {
            w1, b1, w2, b2,
            prev_state:  self.prev_state [s * IN  .. (s+1) * IN ].to_vec(),
            prev_action: self.prev_action[s * OUT .. (s+1) * OUT].to_vec(),
            prev_energy: self.prev_energy[s],
            baseline,
            has_prev:    self.has_prev[s],
        }
    }

    /// Build / overwrite a species' shared net from a saved `BrainRestore`.
    /// Mismatched dims → graceful failure. Seeds the species baseline from the
    /// restore payload. Does NOT touch per-slot prev_* (the `assign_*` writes
    /// those).
    pub fn restore_species(&mut self, key: u32, r: &BrainRestoreHerbivore1) -> Result<(), String> {
        let hidden = self.cfg.hidden;
        if r.w1.len() != IN * hidden  { return Err(format!("w1 size {} != {}", r.w1.len(), IN * hidden)); }
        if r.b1.len() != hidden       { return Err(format!("b1 size {} != {}", r.b1.len(), hidden)); }
        if r.w2.len() != hidden * OUT { return Err(format!("w2 size {} != {}", r.w2.len(), hidden * OUT)); }
        if r.b2.len() != OUT          { return Err(format!("b2 size {} != {}", r.b2.len(), OUT)); }
        if r.prev_state.len()  != IN  { return Err(format!("prev_state {} != {}", r.prev_state.len(), IN)); }
        if r.prev_action.len() != OUT { return Err(format!("prev_action {} != {}", r.prev_action.len(), OUT)); }

        self.ensure_species(key);
        let device = self.device.clone();
        let model = PolicyNet::<MyBackend> {
            w1: Param::from_tensor(Tensor::<MyBackend, 2>::from_data(
                TensorData::new(r.w1.clone(), [IN, hidden]), &device)),
            b1: Param::from_tensor(Tensor::<MyBackend, 2>::from_data(
                TensorData::new(r.b1.clone(), [1, hidden]), &device)),
            w2: Param::from_tensor(Tensor::<MyBackend, 2>::from_data(
                TensorData::new(r.w2.clone(), [hidden, OUT]), &device)),
            b2: Param::from_tensor(Tensor::<MyBackend, 2>::from_data(
                TensorData::new(r.b2.clone(), [1, OUT]), &device)),
        };
        let brain = self.species.get_mut(&key).expect("species ensured above");
        brain.model    = model;
        brain.baseline = r.baseline;
        Ok(())
    }

    /// Read-only view of the recent training history.
    pub fn training_history(&self) -> &VecDeque<TrainingStep> {
        &self.training_history
    }

    /// Per-slot snapshot of the policy's most recent output + reward
    /// contributions. Runs one batched forward PER SPECIES over that species'
    /// members' cached `prev_state`; scatters μ back per slot.
    pub fn snapshot_telemetry(&self) -> Vec<BrainTelemetry> {
        let n = self.n();

        let mut groups: HashMap<u32, Vec<usize>> = HashMap::new();
        for (_e, &slot) in &self.map {
            let s = slot as usize;
            groups.entry(self.slot_species[s]).or_default().push(s);
        }
        let mut mu_data = vec![0.0_f32; n * OUT];
        for (key, slots) in &groups {
            let Some(brain) = self.species.get(key) else { continue };
            let cnt = slots.len();
            // Pad the batch row count to a power of two so CubeCL sees a bounded
            // set of shapes (one autotune each) instead of recompiling on every
            // count change. Padded rows are zero and never read back (the scatter
            // loop below only touches the first `cnt` rows).
            let pad = crate::limb_ppo::pad_batch(cnt);
            let mut rows = vec![0.0_f32; pad * IN];
            for (i, &s) in slots.iter().enumerate() {
                rows[i * IN .. i * IN + IN].copy_from_slice(&self.prev_state[s * IN .. s * IN + IN]);
            }
            let x = Tensor::<MyBackend, 2>::from_data(TensorData::new(rows, [pad, IN]), &self.device);
            let mu = brain.model.forward(x).into_data().into_vec::<f32>().expect("telemetry mu");
            for (i, &s) in slots.iter().enumerate() {
                mu_data[s * OUT .. s * OUT + OUT].copy_from_slice(&mu[i * OUT .. i * OUT + OUT]);
            }
        }

        let mut out = Vec::with_capacity(n);
        for s in 0..n {
            let row = s * OUT;
            let mu_speed = mu_data[row + 0];
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

    /// Pop a fresh slot for a new organism, optionally restoring saved weights +
    /// per-individual prev_* (loaded `.colony`). Resets all per-slot transient
    /// state and (re)samples the hyperparameter genes. `None` once the pool is
    /// full. Mirrors the old `assign_brains_*` slot-setup body verbatim.
    ///
    /// Returns the allocated slot; the caller inserts the `BrainSlot*` component.
    pub fn enrol(
        &mut self,
        e:        Entity,
        organism: &Organism,
        restore:  Option<&BrainRestoreHerbivore1>,
        rng:      &mut impl rand::Rng,
    ) -> Option<u32> {
        let slot = self.free.pop()?;
        let s = slot as usize;

        let mut restored = false;
        if let Some(r) = restore {
            let key = organism.species_id.unwrap_or(UNCLASSIFIED);
            match self.restore_species(key, r) {
                Ok(()) => {
                    restored = true;
                    let in_off  = s * IN;
                    let out_off = s * OUT;
                    self.prev_state [in_off  .. in_off + IN ].copy_from_slice(&r.prev_state);
                    self.prev_action[out_off .. out_off + OUT].copy_from_slice(&r.prev_action);
                    self.prev_energy[s] = r.prev_energy;
                    self.has_prev[s]    = r.has_prev;
                }
                Err(err) => error!("sliding brain restore failed for {e:?}: {err} — using fresh slot"),
            }
        }

        // Fresh genes (uniform sample within each range) for the new tenant.
        self.sigma[s]         = sample_range(L1_SIGMA_RANGE,         rng);
        self.k_eat[s]         = sample_range(L1_K_EAT_RANGE,         rng);
        self.k_repro[s]       = sample_range(L1_K_REPRO_RANGE,       rng);
        self.lambda_energy[s] = sample_range(L1_LAMBDA_ENERGY_RANGE, rng);
        self.k_curiosity[s]   = sample_range(L1_K_CURIOSITY_RANGE,   rng);
        self.k_progress[s]    = sample_range(L1_K_PROGRESS_RANGE,    rng);

        self.slot_species[s]            = organism.species_id.unwrap_or(UNCLASSIFIED);
        self.target_entity[s]           = None;
        self.lock_ticks_remaining[s]    = 0;
        self.prev_distance_to_target[s] = 0.0;
        self.prev_target_entity[s]      = None;
        self.min_dist_to_target[s]      = f32::INFINITY;
        self.ticks_since_progress[s]    = 0;
        self.blacklisted_target[s]      = None;
        self.blacklist_ticks_remaining[s] = 0;
        self.applied_speed_a[s]         = 0.0;
        self.prev_predations[s]         = organism.predations;
        self.last_eat[s]                = 0.0;
        self.last_progress[s]           = 0.0;
        for i in 0..64 { self.recent_rewards[s * 64 + i] = 0.0; }
        self.recent_reward_head[s] = 0;
        self.buf_count[s]          = 0;

        if !restored {
            self.has_prev[s]    = false;
            self.prev_energy[s] = organism.energy;
        }
        self.prev_reproductions[s] = organism.reproductions;

        self.map.insert(e, slot);
        Some(slot)
    }

    /// Release a slot when its organism is removed. Mirrors the old
    /// `free_brains_*` body verbatim.
    pub fn release(&mut self, e: Entity) {
        if let Some(slot) = self.map.remove(&e) {
            let s = slot as usize;
            self.has_prev[s]             = false;
            self.buf_count[s]            = 0;
            self.slot_species[s]         = UNCLASSIFIED;
            self.target_entity[s]        = None;
            self.lock_ticks_remaining[s] = 0;
            self.prev_target_entity[s]   = None;
            self.min_dist_to_target[s]   = f32::INFINITY;
            self.ticks_since_progress[s] = 0;
            self.blacklisted_target[s]      = None;
            self.blacklist_ticks_remaining[s] = 0;
            self.applied_speed_a[s]      = 0.0;
            self.free.push(slot);
        }
    }

    /// Per-tick apply step shared by all three sliding pools. Generic over the
    /// per-level slot component `S` and the caller's query FILTER `F` (so
    /// herbivore_1 can keep its `(With<Heterotroph>, Without<Carnivore>)` bound
    /// and L2/L3 their `With<Heterotroph>`). Prey type comes from
    /// `self.cfg.prey_type`.
    pub fn apply_step<S, F>(
        &mut self,
        time:        &Time<Virtual>,
        world_grid:  &WorldModelGrid,
        mut heteros: Query<(Entity, &mut Organism, &Transform, &S, Option<&Carnivore>), F>,
        input_buf:   &mut Vec<f32>,
    )
    where
        S: Component + Copy + SlidingSlot,
        F: bevy::ecs::query::QueryFilter,
    {
        if time.is_paused() { return; }
        let n = self.n();
        // Captured once (Copy) to avoid borrowing `self` inside the per-organism loop.
        let direction_3d = self.cfg.direction_3d;

        input_buf.clear();
        input_buf.resize(n * IN, 0.0);

        struct Active {
            entity:        Entity,
            slot:          u32,
            energy_now:    f32,
            reproductions: u8,
            predations:    u8,
            is_carnivore:  bool,
            is_ground_based: bool,
            neighbours:    [Option<crate::world_model::Neighbour>; WORLD_MODEL_K],
        }
        let mut active: Vec<Active> = Vec::new();
        let mut groups: HashMap<u32, Vec<usize>> = HashMap::new();
        let mut nb_scratch: Vec<(f32, crate::world_model::Neighbour)> = Vec::new();

        for (e, organism, transform, slot, carn) in heteros.iter() {
            let s = slot.slot() as usize;
            if s >= n { continue; }
            let pos = transform.translation;

            let max_e    = crate::energy::get_max_energy(&organism).max(1.0);
            let energy_n = (organism.energy / max_e).clamp(0.0, 1.0);
            let neighbours = collect_neighbours(world_grid, pos, &mut nb_scratch);

            let off = s * IN;
            input_buf[off] = energy_n;
            encode_neighbours(&neighbours, &mut input_buf[off + 1 .. off + 1 + WORLD_MODEL_DIMS]);

            let pa_off = off + 1 + WORLD_MODEL_DIMS;
            if self.has_prev[s] {
                let src = s * OUT;
                for i in 0..OUT {
                    input_buf[pa_off + i] = self.prev_action[src + i];
                }
            }

            let flag_off = pa_off + PREV_ACTION_DIMS;
            input_buf[flag_off] = if self.target_entity[s].is_some() { 1.0 } else { 0.0 };

            let key = organism.species_id.unwrap_or(UNCLASSIFIED);
            self.slot_species[s] = key;
            groups.entry(key).or_default().push(s);

            active.push(Active {
                entity: e, slot: slot.slot(),
                energy_now: organism.energy,
                reproductions: organism.reproductions,
                predations: organism.predations,
                is_carnivore: carn.is_some(),
                is_ground_based: organism.ground_based,
                neighbours,
            });
        }
        if active.is_empty() {
            self.tick = self.tick.wrapping_add(1);
            return;
        }

        // ── Forward inference — ONE readback for the whole pool. Issue ALL
        //    per-species forwards first (they queue back-to-back on the GPU
        //    stream), then read every μ row back in a SINGLE sync via one
        //    concatenated tensor — instead of a blocking `into_data()` between
        //    each species (which ping-ponged CPU↔GPU and stalled both). Mirrors
        //    the limb/swim pools' batched-readback. A deterministic species order
        //    keeps the cat/scatter offsets aligned. ──
        let device = self.device.clone();
        let mut mu_data = vec![0.0_f32; n * OUT];

        // ensure_species needs &mut self — do it BEFORE borrowing species nets
        // immutably for the forwards.
        for key in groups.keys().copied().collect::<Vec<u32>>() {
            self.ensure_species(key);
        }
        let mut order: Vec<(u32, &Vec<usize>)> = groups.iter().map(|(k, v)| (*k, v)).collect();
        order.sort_by_key(|(k, _)| *k);

        let mut actor_outs: Vec<Tensor<crate::rl_helpers::InferBackend, 2>> = Vec::with_capacity(order.len());
        let mut pads: Vec<usize> = Vec::with_capacity(order.len());
        for (key, slots) in &order {
            let brain = self.species.get(key).expect("species ensured above");
            let cnt = slots.len();
            // Pad the batch row count to a power of two so CubeCL sees a bounded
            // set of shapes (one autotune each) instead of recompiling the forward
            // kernel on every birth/death that changes a species' live count.
            // Padded rows are zero and sliced off in the scatter below.
            let pad = crate::limb_ppo::pad_batch(cnt);
            pads.push(pad);
            let mut rows = vec![0.0_f32; pad * IN];
            for (i, &s) in slots.iter().enumerate() {
                rows[i * IN .. i * IN + IN].copy_from_slice(&input_buf[s * IN .. s * IN + IN]);
            }
            // No-grad inference: forward on the inner non-autodiff backend.
            let obs_t = Tensor::<crate::rl_helpers::InferBackend, 2>::from_data(TensorData::new(rows, [pad, IN]), &device);
            actor_outs.push(brain.model.valid().forward(obs_t));
        }
        // ONE readback for all actor rows.
        let mu_all = Tensor::cat(actor_outs, 0).into_data().into_vec::<f32>().expect("forward output");
        // Scatter back: concatenated rows follow `order`, each species contributing
        // `pad` rows (only the first `cnt` = slots.len() are real).
        let mut row_off = 0usize;
        for ((_key, slots), &pad) in order.iter().zip(pads.iter()) {
            for (i, &s) in slots.iter().enumerate() {
                let r = row_off + i;
                mu_data[s * OUT .. s * OUT + OUT].copy_from_slice(&mu_all[r * OUT .. r * OUT + OUT]);
            }
            row_off += pad;
        }

        let mut rng = rand::rng();
        let prey_type = self.cfg.prey_type;

        for a in &active {
            let s = a.slot as usize;

            // (3a) Reward for the previous action (if any).
            if self.has_prev[s] {
                let new_eats   = a.predations.wrapping_sub(self.prev_predations[s]);
                let new_repros = a.reproductions.saturating_sub(self.prev_reproductions[s]);
                let energy_delta = a.energy_now - self.prev_energy[s];

                let r_eat   = self.k_eat[s]   * new_eats   as f32;
                let r_repro = self.k_repro[s] * new_repros as f32;
                let r_energy = if energy_delta < 0.0 {
                    self.lambda_energy[s] * energy_delta
                } else { 0.0 };

                let prev_speed_norm = self.applied_speed_a[s].max(0.0);
                let r_curiosity = self.k_curiosity[s] * (prev_speed_norm - 0.5);

                let r_progress = if let Some(prev_target) = self.prev_target_entity[s] {
                    let cur_d = a.neighbours.iter()
                        .find_map(|nbr| nbr.and_then(|nn|
                            if nn.entity == prev_target { Some(nn.rel.length()) } else { None }
                        ));
                    match cur_d {
                        Some(d) => {
                            let closed = self.prev_distance_to_target[s] - d;
                            self.k_progress[s] * closed.max(0.0)
                        }
                        None => 0.0,
                    }
                } else { 0.0 };

                let r = r_eat + r_repro + r_energy + r_curiosity + r_progress;

                // Surface reward components for telemetry.
                self.last_eat[s]      = r_eat;
                self.last_progress[s] = r_progress;
                let head = self.recent_reward_head[s] as usize;
                self.recent_rewards[s * 64 + head] = r;
                self.recent_reward_head[s] = ((head + 1) % 64) as u8;

                let count = self.buf_count[s];
                if count < ROLLOUT_LEN {
                    let prev_state  = self.prev_state[s * IN .. (s + 1) * IN].to_vec();
                    let prev_action = self.prev_action[s * OUT .. (s + 1) * OUT].to_vec();
                    let buf_in_off  = s * ROLLOUT_LEN * IN + count * IN;
                    let buf_out_off = s * ROLLOUT_LEN * OUT + count * OUT;
                    self.buf_states [buf_in_off  .. buf_in_off  + IN ].copy_from_slice(&prev_state);
                    self.buf_actions[buf_out_off .. buf_out_off + OUT].copy_from_slice(&prev_action);
                    self.buf_rewards[s * ROLLOUT_LEN + count] = r;
                    self.buf_count[s] = count + 1;
                }
            }

            // (3b) Sample a new action ~ N(μ, σ²) — 5 dims.
            let off = s * OUT;
            let sigma_s = self.sigma[s];
            let mut action = [0.0_f32; OUT];
            for i in 0..OUT {
                action[i] = mu_data[off + i] + sigma_s * gaussian_noise(&mut rng);
            }
            let speed_a = action[0].clamp(-1.0, 1.0);
            let target_logits = [action[1], action[2], action[3], action[4]];

            // (3c-pre) Stuck detection — EAT-DRIVEN safety net, so a slider can
            // NEVER permanently freeze on a target it can't consume. The
            // grounding filter above already steers benthic sliders to reachable
            // prey when any exist; but the fallback (chasing floating prey it
            // can't reach in Y) would otherwise re-create the freeze trap: the
            // arrival-brake drives speed → 0 as XZ distance → 0, and the old
            // `d > L1_APPROACH_RADIUS` gate meant the no-progress counter never
            // ran in that "arrived" zone → the slider sat on an uneatable target
            // forever. The only legitimate reason to dwell on a locked target is
            // to be EATING it, so the counter now runs at ALL distances and is
            // reset by genuine closing OR by an eat; if neither happens for
            // `L1_STUCK_TICKS`, the target is blacklisted and the slider moves on.
            let ate_this_tick = a.predations != self.prev_predations[s];
            let cur_locked_dist = self.target_entity[s].and_then(|tgt| {
                a.neighbours.iter().find_map(|nbr|
                    nbr.and_then(|nn| if nn.entity == tgt {
                        Some((nn.rel.x * nn.rel.x + nn.rel.z * nn.rel.z).sqrt())
                    } else { None })
                )
            });
            if let Some(d) = cur_locked_dist {
                let closing = d + L1_STUCK_PROGRESS_EPS < self.min_dist_to_target[s];
                if closing {
                    self.min_dist_to_target[s] = d;
                }
                if ate_this_tick || closing {
                    // Eating it, or genuinely closing in — making progress.
                    self.ticks_since_progress[s] = 0;
                } else {
                    // Neither closing nor eating — count toward the timeout,
                    // whether too far to approach OR parked on an unreachable
                    // in-range target (the freeze trap the old gate never caught).
                    self.ticks_since_progress[s] = self.ticks_since_progress[s].saturating_add(1);
                }
            }
            let force_drop = self.ticks_since_progress[s] >= L1_STUCK_TICKS;

            if self.blacklist_ticks_remaining[s] > 0 {
                self.blacklist_ticks_remaining[s] -= 1;
                if self.blacklist_ticks_remaining[s] == 0 {
                    self.blacklisted_target[s] = None;
                }
            }
            if force_drop {
                self.blacklisted_target[s] = self.target_entity[s];
                self.blacklist_ticks_remaining[s] = L1_TARGET_BLACKLIST_TICKS;
            }
            let blacklisted_entity = self.blacklisted_target[s];

            // (3c) Target selection — prey type per-level (Photo for herbivores,
            //      Hetero for carnivores in L2/L3).
            let valid_ty = prey_type(a.is_carnivore);
            let locked_entity = if force_drop { None } else { self.target_entity[s] };

            // Reachability preference for GROUND-BASED (ocean-floor / benthic)
            // sliders: they crawl the terrain and can't reach prey floating in
            // the water column, so they prefer prey that ALSO lives on the
            // ground. Only when NO ground-based prey is in view do they fall
            // back to chasing floating/swimming prey (unreachable, but trying
            // beats idling — same behaviour as before this gate). Water-based
            // predators are unaffected (`restrict_to_ground` stays false).
            let restrict_to_ground = a.is_ground_based
                && a.neighbours.iter().any(|slot| slot.is_some_and(|nn|
                    nn.ty == valid_ty
                    && nn.ground_based
                    && Some(nn.entity) != blacklisted_entity));

            let mut locked_slot: Option<usize> = None;
            let mut best_slot:   Option<usize> = None;
            let mut best_logit:        f32      = f32::NEG_INFINITY;

            for (i, slot) in a.neighbours.iter().enumerate() {
                let Some(nn) = slot else { continue };
                if nn.ty != valid_ty { continue; }
                if Some(nn.entity) == blacklisted_entity { continue; }
                // Tier 1 (a ground prey is in view): consider only reachable,
                // ground-based prey — even drops a currently-locked floating
                // target in favour of a reachable one. Tier 2 (no ground prey):
                // gate is off, so all valid prey are considered as before.
                if restrict_to_ground && !nn.ground_based { continue; }
                let l = target_logits[i];
                if l > best_logit { best_logit = l; best_slot = Some(i); }
                if Some(nn.entity) == locked_entity { locked_slot = Some(i); }
            }

            let chosen_slot: Option<usize> = match (locked_slot, best_slot) {
                (Some(li), _) if self.lock_ticks_remaining[s] > 0 => {
                    self.lock_ticks_remaining[s] -= 1;
                    Some(li)
                }
                (Some(li), Some(bi)) => {
                    let chosen = if bi != li
                        && best_logit > target_logits[li] + L1_TARGET_SWITCH_MARGIN
                    { bi } else { li };
                    self.lock_ticks_remaining[s] = L1_TARGET_LOCK_TICKS;
                    Some(chosen)
                }
                (None, Some(bi)) => {
                    self.lock_ticks_remaining[s] = L1_TARGET_LOCK_TICKS;
                    Some(bi)
                }
                _ => {
                    self.lock_ticks_remaining[s] = 0;
                    None
                }
            };

            let new_target: Option<(Entity, Vec3)> = chosen_slot
                .and_then(|i| a.neighbours[i].map(|nbr| (nbr.entity, nbr.rel)));

            // (3d) Apply to the world.
            let Ok((_, mut org, _, _, _)) = heteros.get_mut(a.entity) else { continue };

            let smoothed = L1_SPEED_MOMENTUM_ALPHA * self.applied_speed_a[s]
                         + (1.0 - L1_SPEED_MOMENTUM_ALPHA) * speed_a;
            self.applied_speed_a[s] = smoothed;

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
                        if direction_3d {
                            // 3D pursuit (SimpleAquatic): keep the Y component so the
                            // mover climbs/descends toward prey through the water column.
                            let d3 = Vec3::new(dx, rel.y, dz);
                            if let Some(unit) = d3.try_normalize() {
                                org.movement_direction = unit;
                            }
                        } else {
                            let inv = mag2.sqrt().recip();
                            org.movement_direction = Vec3::new(dx * inv, 0.0, dz * inv);
                        }
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
            for i in 0..IN { self.prev_state[in_off + i] = input_buf[in_off + i]; }
            self.prev_action[off + 0] = speed_a;
            self.prev_action[off + 1] = target_logits[0];
            self.prev_action[off + 2] = target_logits[1];
            self.prev_action[off + 3] = target_logits[2];
            self.prev_action[off + 4] = target_logits[3];
            self.prev_energy[s]        = a.energy_now;
            self.prev_reproductions[s] = a.reproductions;
            self.prev_predations[s]    = a.predations;
            self.has_prev[s]           = true;

            let prev_target_ent = self.target_entity[s];
            self.prev_target_entity[s] = prev_target_ent;
            let new_target_ent = new_target.map(|(e, _)| e);
            if let Some((ent, rel)) = new_target {
                self.target_entity[s]           = Some(ent);
                self.prev_distance_to_target[s] = rel.length();
            } else {
                self.target_entity[s]           = None;
                self.prev_distance_to_target[s] = 0.0;
            }

            if new_target_ent != prev_target_ent {
                self.min_dist_to_target[s] = match new_target {
                    Some((_, rel)) => (rel.x * rel.x + rel.z * rel.z).sqrt(),
                    None           => f32::INFINITY,
                };
                self.ticks_since_progress[s] = 0;
            }
        }

        self.tick = self.tick.wrapping_add(1);
        if self.tick % ROLLOUT_LEN as u64 != 0 { return; }
        self.train(time.elapsed_secs());
    }

    /// Run one Monte Carlo REINFORCE update PER SPECIES. Active slots (with
    /// buffered steps) are grouped by their CURRENT species; for each species we
    /// compute discounted returns, update that species' EMA baseline, pool the
    /// transitions into one flat `[M, ·]` batch, and run one Gaussian-log-prob
    /// REINFORCE step on that species' net. One `TrainingStep` is logged per
    /// species update. After all updates, every slot's buffer is reset.
    fn train(&mut self, virtual_time_secs: f32) {
        self.prune_species();
        let n = self.n();

        let mut groups: HashMap<u32, Vec<usize>> = HashMap::new();
        for s in 0..n {
            if self.buf_count[s] > 0 {
                groups.entry(self.slot_species[s]).or_default().push(s);
            }
        }
        let device = self.device.clone();

        for (key, slots) in groups {
            self.ensure_species(key);

            struct SlotRet { slot: usize, count: usize, returns: Vec<f32> }
            let mut slot_rets: Vec<SlotRet> = Vec::with_capacity(slots.len());
            let mut mean_g_sum = 0.0_f32;
            let mut active_slots: u32 = 0;
            let mut return_sum:  f64 = 0.0;
            let mut return_sumsq: f64 = 0.0;
            let mut return_n:    u64 = 0;
            for &s in &slots {
                let count = self.buf_count[s];
                if count == 0 { continue; }
                active_slots += 1;

                let mut returns = vec![0.0_f32; count];
                let mut g = 0.0_f32;
                for t in (0..count).rev() {
                    let r = self.buf_rewards[s * ROLLOUT_LEN + t];
                    g = r + GAMMA * g;
                    returns[t] = g;
                    return_sum += g as f64;
                    return_sumsq += (g as f64) * (g as f64);
                    return_n += 1;
                }
                let mean_g: f32 = returns.iter().sum::<f32>() / count as f32;
                mean_g_sum += mean_g;
                slot_rets.push(SlotRet { slot: s, count, returns });
            }
            if slot_rets.is_empty() { continue; }

            let species_mean_g = mean_g_sum / slot_rets.len() as f32;
            {
                let brain = self.species.get_mut(&key).expect("species ensured above");
                brain.baseline = (1.0 - BASELINE_ALPHA) * brain.baseline
                               + BASELINE_ALPHA * species_mean_g;
            }
            let baseline = self.species.get(&key).expect("species ensured above").baseline;

            let mut states:  Vec<f32> = Vec::new();
            let mut actions: Vec<f32> = Vec::new();
            let mut adv:     Vec<f32> = Vec::new();
            let mut total_count = 0.0_f32;
            for sr in &slot_rets {
                let s = sr.slot;
                let sigma = self.sigma[s].max(1e-3);
                let scale = 0.5_f32 / (sigma * sigma);
                for t in 0..sr.count {
                    let buf_in_off  = s * ROLLOUT_LEN * IN + t * IN;
                    let buf_out_off = s * ROLLOUT_LEN * OUT + t * OUT;
                    states.extend_from_slice(&self.buf_states [buf_in_off  .. buf_in_off  + IN ]);
                    actions.extend_from_slice(&self.buf_actions[buf_out_off .. buf_out_off + OUT]);
                    adv.push((sr.returns[t] - baseline) * scale);
                    total_count += 1.0;
                }
            }
            let m = adv.len();
            if m == 0 || total_count < 1.0 { continue; }

            // Pad the pooled batch to a power of two so the backend sees a BOUNDED
            // set of training shapes (one autotune each) rather than one per
            // transition count. No mask tensor is needed here (unlike the limb PPO
            // pool): the loss is `(Δ²·adv).sum() / total_count`, and the padded
            // rows carry adv = 0, so their loss term AND its gradient are exactly
            // zero. `total_count` stays `= m`, so the loss scalar and the
            // gradients are identical to the unpadded batch.
            let m_pad = crate::limb_ppo::pad_batch(m);
            states.resize(m_pad * IN, 0.0);
            actions.resize(m_pad * OUT, 0.0);
            adv.resize(m_pad, 0.0);

            let states_t  = Tensor::<MyBackend, 2>::from_data(TensorData::new(states,  [m_pad, IN]),  &device);
            let actions_t = Tensor::<MyBackend, 2>::from_data(TensorData::new(actions, [m_pad, OUT]), &device);
            let adv_t     = Tensor::<MyBackend, 2>::from_data(TensorData::new(adv,     [m_pad, 1]),   &device);

            let loss_val = {
                let brain = self.species.get_mut(&key).expect("species ensured above");
                let mu = brain.model.forward(states_t);
                let diff = actions_t - mu;
                let sum_sq = diff.powf_scalar(2.0).sum_dim(1);
                let loss = (sum_sq * adv_t).sum().div_scalar(total_count);

                let loss_val = loss.clone().into_data().into_vec::<f32>().expect("loss")[0];

                let cm = brain.model.clone();
                let gp = GradientsParams::from_grads(loss.backward(), &brain.model);
                brain.model = brain.opt.step(LR, cm, gp);
                brain.tick = brain.tick.wrapping_add(1);
                loss_val
            };

            let mean_return = if return_n > 0 { (return_sum / return_n as f64) as f32 } else { 0.0 };
            let return_var  = if return_n > 1 {
                let mean = return_sum / return_n as f64;
                let var  = (return_sumsq / return_n as f64) - mean * mean;
                var.max(0.0) as f32
            } else { 0.0 };

            self.step_counter = self.step_counter.saturating_add(1);
            if self.training_history.len() >= TRAINING_HISTORY_CAP {
                self.training_history.pop_front();
            }
            self.training_history.push_back(TrainingStep {
                step:              self.step_counter,
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
        }

        for s in 0..n { self.buf_count[s] = 0; }
    }
}


/// Force-compile every kernel the steady-state apply tick + training step will
/// use, so the user's first real frame doesn't pay the JIT cost.
pub fn warmup(device: &CudaDevice, hidden: usize) {
    const WARMUP_SIGMA: f32 = 0.5;
    const M: usize = ROLLOUT_LEN;   // representative pooled batch size
    let m = new_species_brain(device, hidden);
    let mut o: Box<dyn BrainOpt> = Box::new(AdamConfig::new().init());

    let i = Tensor::<MyBackend, 2>::zeros([4, IN], device);
    let _ = m.model.forward(i);

    let states  = Tensor::<MyBackend, 2>::zeros([M, IN],  device);
    let actions = Tensor::<MyBackend, 2>::zeros([M, OUT], device);
    let adv     = Tensor::<MyBackend, 2>::zeros([M, 1],   device);
    let mu      = m.model.forward(states);
    let diff    = actions - mu;
    let sum_sq  = diff.powf_scalar(2.0).sum_dim(1);
    let scale   = 0.5_f32 / (WARMUP_SIGMA * WARMUP_SIGMA);
    let loss    = (sum_sq * adv).sum().mul_scalar(scale).div_scalar(1.0_f32);
    let g = GradientsParams::from_grads(loss.backward(), &m.model);
    let _ = o.step(LR, m.model, g);
}
