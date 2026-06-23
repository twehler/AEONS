// Intelligence Level 3 — Predator REINFORCE pool (Monte-Carlo
// rollouts, no value head). Sliding-movement variant.
//
// SHARED-POLICY (per-SPECIES) variant: every organism of a species shares
// ONE network — all members' experience trains the SAME weights, so the
// population learns as a single agent with ~N× the data. Different species
// hold different weights (so behaviour diverges by species); same-species
// individuality is preserved only as momentary exploration noise, not as
// distinct learned weights. Mirrors the swimming pool's structure but keeps
// REINFORCE (Monte-Carlo policy gradient + per-SPECIES EMA baseline).
//
// Network/IO layout is documented on the architecture constants below.
//
// Reward (every brain tick, attributed to the previous action):
//   r =  K_EAT       per predation event (predation-count delta)
//      + K_REPRO     per reproduction event
//      + LAMBDA · min(0, ΔE)   penalty for energy spent
//      + K_CURIOSITY · (speed_norm − 0.5)   gradient away from standing still
//      + K_PROGRESS · distance closed to the locked target
// `K_EAT < K_REPRO` so reproduction is the strongest pull.
//
// Algorithm: Monte Carlo REINFORCE with a per-SPECIES EMA baseline. Each tick
// fills the previous action's reward, samples a new action, and stores both
// in a per-slot ring buffer of length `ROLLOUT_LEN`. Every `ROLLOUT_LEN`
// ticks the pool trains: active slots are GROUPED by species and each
// species' transitions are pooled into one batched forward+backward;
// discounted returns `G_t = Σ γ^(i-t)·r_i` credit sparse eat/reproduce
// jackpots back over the trajectory that produced them.

use bevy::prelude::*;
use burn::module::{AutodiffModule, Initializer, Module, Param};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use burn_cuda::CudaDevice;
use std::collections::HashMap;

use crate::colony::{IntelligenceLevel, Organism, Heterotroph, Carnivore};
use crate::rl_helpers::{
    BrainInheritance, BrainRestore, MyBackend, PoolSnapshot, SpeciesWeights, gaussian_noise,
};
use crate::simulation_settings::{
    OrganismPoolSize,
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


// ── Architecture constants ──────────────────────────────────────────────────
//
// Input layout (31 dims):
//   [0]          : self energy / max_energy
//   [1..25]      : world model — 4 neighbours × (rel_x, rel_z, vel_x,
//                  vel_z, is_photo, is_hetero), normalised
//   [25..30]     : previous action sample (speed_a + 4 target logits)
//                  — provides policy memory / recurrence at the input
//                  side so the network can condition on its own past
//                  decision without an explicit recurrent layer.
//   [30]         : has_locked_target (0 / 1) — single bit signalling
//                  whether the agent is currently committed.
//
// Output layout (5 dims):
//   [0]    : speed_a  — tanh, mapped to `max(0, speed_a) · MAX_SPEED`
//                       and EMA-smoothed before being written to
//                       `Organism::movement_speed`.
//   [1..5] : target_logits — one tanh score per neighbour slot.
//                       Direction is NOT output by the network; it
//                       is derived geometrically from the chosen
//                       target's position. The target is the argmax
//                       over slots whose neighbour is a Photo (prey).

const PREV_ACTION_DIMS:    usize = 5;
const LOCKED_FLAG_DIMS:    usize = 1;

/// Brain-tick interval in *virtual* seconds. Mirrors
/// `behaviour.rs::HETERO_BRAIN_TICK_INTERVAL` (150 ms) — kept here as
/// an `f32` so the target-lock seconds-to-ticks conversion can run
/// at const time. If you change one, change the other.
const HETERO_BRAIN_TICK_SECS: f32 = 0.150;

/// Number of brain ticks the target-lock window spans, derived from
/// `L1_TARGET_LOCK_SECS / HETERO_BRAIN_TICK_SECS` (ceiled). Since the
/// brain runs on `Time<Virtual>`, the constant tick count
/// automatically gives a constant *virtual*-time lock duration, which
/// in turn scales with `TimeSpeed` for real-time observation —
/// exactly what we want.
///
/// `u16` because 10 virtual seconds × 6.67 ticks/s ≈ 67 fits but
/// longer lock windows would overflow `u8`. The cost vs `u8` is
/// negligible (one extra byte per slot).
const L1_TARGET_LOCK_TICKS: u16 = {
    // `ceil` via cast: add (denom − 1) / denom to round up positive
    // f32→u16 conversions in const context.
    let f = L1_TARGET_LOCK_SECS / HETERO_BRAIN_TICK_SECS;
    let truncated = f as u16;
    if (truncated as f32) < f { truncated + 1 } else { truncated }
};
const IN:      usize = 1 + WORLD_MODEL_DIMS + PREV_ACTION_DIMS + LOCKED_FLAG_DIMS;
const HIDDEN:  usize = 32;
/// `(speed_a, target_logit_0, target_logit_1, target_logit_2, target_logit_3)`.
const OUT:     usize = 1 + WORLD_MODEL_K;
/// Rollout length in brain ticks. At ~6.7 Hz (150 ms tick) this is
/// ~4.8 s of trajectory per training update — long enough for the
/// reproduction jackpot to credit-assign back to the actions that
/// led to it (γ^32 ≈ 0.19 with γ=0.95, so ~80% of the signal
/// decays inside one rollout).
const ROLLOUT_LEN: usize = 32;
const MAX_SPEED:   f32   = 40.0;
const LR:          f64   = 1e-3;

/// Discount for the Monte Carlo return.
const GAMMA:           f32 = 0.95;
/// EMA decay rate for the per-species baseline (mean of recent returns).
const BASELINE_ALPHA:  f32 = 0.05;


// Reward + exploration tuning RANGES (`L1_*_RANGE`) live in
// `simulation_settings.rs`. Per-organism samples are stored on the
// pool as `Vec<f32>` indexed by slot (see `BrainPoolL3` below):
// at slot assignment, an organism either samples uniformly from the
// range (initial spawn / fresh slot) or inherits its parent's values
// with small Gaussian mutation. Selection on those traits emerges
// from differential reproduction.

/// Sentinel species key for organisms not yet classified by the speciation
/// system (real `species_id`s start at 1). All unclassified L3 heterotrophs
/// share this one transitional net until the ~1 Hz speciation tick assigns
/// them; the next apply tick then re-points them at their real species' net
/// automatically (the per-tick species lookup makes reclassification a no-op).
pub const UNCLASSIFIED: u32 = 0;


/// Uniform sample within `(min, max)`.
fn sample_range(range: (f32, f32), rng: &mut impl rand::Rng) -> f32 {
    use rand::RngExt;
    let (lo, hi) = range;
    lo + (hi - lo) * rng.random::<f32>()
}


// ── Slot marker ─────────────────────────────────────────────────────────────

#[derive(Component, Clone, Copy)]
pub struct BrainSlotL3(pub u32);


// ── Shared (per-species) MLP ──────────────────────────────────────────────────
//
// 2-D weights (`[IN, HIDDEN]`, `[HIDDEN, OUT]`); biases `[1, ·]` so a forward
// over a batch of `M` observations is a plain `[M, IN] · [IN, HIDDEN]` matmul
// with broadcast bias. The same `forward` serves both inference (one row per
// active organism of a species) and the REINFORCE update (all of a species'
// pooled transitions as one batch).

#[derive(Module, Debug)]
pub struct PoolMlpL3<B: Backend> {
    w1: Param<Tensor<B, 2>>,  // [IN, HIDDEN]
    b1: Param<Tensor<B, 2>>,  // [1, HIDDEN]
    w2: Param<Tensor<B, 2>>,  // [HIDDEN, OUT]
    b2: Param<Tensor<B, 2>>,  // [1, OUT]
}

impl<B: Backend> PoolMlpL3<B> {
    /// Forward `[M, IN] → [M, OUT]` (shared weights, broadcast bias). Serves
    /// both inference and the REINFORCE replay (the pooled `[M, IN]` batch).
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = burn::tensor::activation::relu(x.matmul(self.w1.val()) + self.b1.val());
        burn::tensor::activation::tanh(h.matmul(self.w2.val()) + self.b2.val())
    }
}


trait BrainOptL3 {
    fn step(
        &mut self,
        lr: f64,
        m:  PoolMlpL3<MyBackend>,
        g:  GradientsParams,
    ) -> PoolMlpL3<MyBackend>;
}

impl<O: Optimizer<PoolMlpL3<MyBackend>, MyBackend>> BrainOptL3 for O {
    fn step(
        &mut self,
        lr: f64,
        m:  PoolMlpL3<MyBackend>,
        g:  GradientsParams,
    ) -> PoolMlpL3<MyBackend> {
        Optimizer::step(self, lr, m, g)
    }
}


/// One species' shared net + its optimiser + its REINFORCE EMA baseline.
/// Created lazily by `new_species_brain` (fresh-init weights — sliding has no
/// oscillatory warm-start). `tick` counts apply ticks for this species (not
/// used in learning math, kept for parity with the swim pool's `training_step`).
pub struct SpeciesBrain {
    pub model:    PoolMlpL3<MyBackend>,
    pub opt:      Box<dyn BrainOptL3>,
    /// Per-species REINFORCE baseline (EMA over mean rollout return).
    /// Subtracted from the return to produce the advantage that scales the
    /// policy loss for every member of this species.
    pub baseline: f32,
    pub tick:     u64,
}

/// Build one freshly-initialised species brain. Weights uniform in [-0.5,0.5]
/// (the original sliding init), biases zero. No oscillatory warm-start (that
/// is swimmer-specific). Each new species gets its OWN fresh net.
pub fn new_species_brain(device: &CudaDevice) -> SpeciesBrain {
    let w = Initializer::Uniform { min: -0.5, max: 0.5 };
    let z = Initializer::Zeros;
    let model = PoolMlpL3::<MyBackend> {
        w1: w.init([IN, HIDDEN], device),
        b1: z.init([1, HIDDEN],  device),
        w2: w.init([HIDDEN, OUT], device),
        b2: z.init([1, OUT],     device),
    };
    let opt: Box<dyn BrainOptL3> = Box::new(AdamConfig::new().init());
    SpeciesBrain { model, opt, baseline: 0.0, tick: 0 }
}


// ── Pool resource ───────────────────────────────────────────────────────────

pub struct BrainPoolL3 {
    /// Per-species shared nets (model + optimiser + baseline). Keyed by
    /// `Organism::species_id` (`UNCLASSIFIED` until first classification);
    /// created lazily by `ensure_species`.
    species:    HashMap<u32, SpeciesBrain>,
    /// Per-slot species key as of the last apply tick. Lets `train` group
    /// slots by species without an organism query.
    slot_species: Vec<u32>,
    free:       Vec<u32>,
    map:        HashMap<Entity, u32>,

    /// Per-slot rollout buffers (flat row-major, indexed by
    /// `s * ROLLOUT_LEN * IN + t * IN + d` etc.). Filled in
    /// time order; `buf_count[s]` tracks how many timesteps slot
    /// `s` has accumulated since the last training event.
    buf_states:  Vec<f32>,    // [N * ROLLOUT_LEN * IN]
    buf_actions: Vec<f32>,    // [N * ROLLOUT_LEN * OUT]
    buf_rewards: Vec<f32>,    // [N * ROLLOUT_LEN]
    buf_count:   Vec<usize>,  // [N]

    /// State / action stored at the previous brain tick. Used to
    /// pair the reward computed at THIS tick with the action that
    /// produced it. `has_prev[s] = false` for the first tick after
    /// a slot is (re-)assigned.
    prev_state:         Vec<f32>,
    prev_action:        Vec<f32>,
    prev_energy:        Vec<f32>,
    prev_reproductions: Vec<u8>,
    /// Eat events fired by `predation_system` (per-slot). The brain
    /// reads `delta = predations_now - prev_predations` per tick as
    /// the eat-event reward signal — a true predation event count,
    /// not an energy-spike heuristic.
    prev_predations:    Vec<u8>,
    has_prev:           Vec<bool>,

    // ── Target-lock state (per slot) ───────────────────────────────────
    /// Entity currently locked as the prey target. `None` when no
    /// target. The lock is reset on slot assignment / free.
    target_entity:           Vec<Option<Entity>>,
    /// Ticks remaining before the policy is allowed to re-evaluate
    /// target choice. Decrements every brain tick; while > 0 the
    /// argmax is overridden by the locked target.
    lock_ticks_remaining:    Vec<u16>,
    /// Distance to the locked target at the previous brain tick.
    /// Used to compute the per-tick "distance closed" progress
    /// reward. Only valid when `prev_target_was_same == true`.
    prev_distance_to_target: Vec<f32>,
    /// Was the locked target at the *previous* tick the same as at
    /// the current tick? Reward shaping only credits progress when
    /// the target identity is stable across the comparison.
    prev_target_entity:      Vec<Option<Entity>>,

    /// Smallest XZ distance to the currently-locked target observed
    /// since the lock was created (or last refreshed by a target
    /// switch). Together with `ticks_since_progress` this implements
    /// the "drop the lock if not making progress" failsafe — handles
    /// the blocker case where the lock would otherwise persist for
    /// the full `L1_TARGET_LOCK_SECS` window on an unreachable prey.
    min_dist_to_target:      Vec<f32>,
    /// Brain ticks elapsed since `min_dist_to_target` last decreased
    /// by at least `L1_STUCK_PROGRESS_EPS`. Resets to 0 on target
    /// change or on real progress. Triggers a force-drop when it
    /// reaches `L1_STUCK_TICKS`.
    ticks_since_progress:    Vec<u16>,

    /// Most recently force-dropped target. While
    /// `blacklist_ticks_remaining > 0` the target-selection scan
    /// excludes this entity, so the brain MUST pick a different
    /// prey (or none). Prevents the "drop-stuck-target → fresh-pick
    /// same target → drop-stuck-target" cycle observed when a
    /// blocker stands persistently between agent and locked prey.
    blacklisted_target:      Vec<Option<Entity>>,
    /// Ticks remaining on the per-slot blacklist. Counts down each
    /// brain tick; clears `blacklisted_target` when it reaches 0.
    blacklist_ticks_remaining: Vec<u16>,

    /// Output-side EMA-smoothed speed actually applied to the
    /// organism. Decoupled from `prev_action[OUT.0]` so the policy
    /// gradient still uses the raw sample (correct log-prob) while
    /// the world sees a low-jerk trajectory.
    applied_speed_a:    Vec<f32>,

    /// Per-slot reward-shaping + exploration hyperparameters (one
    /// entry per slot). Sampled uniformly from `L1_*_RANGE` on initial
    /// assignment; inherited-with-mutation on reproduction. Each
    /// value functions as a "gene" the population can select on.
    /// `pub` so the lineages plugin's `sync_dna_from_brain_pool` can
    /// read each slot's current gene values without going through an
    /// accessor on every Organism every tick.
    pub sigma:         Vec<f32>,
    pub k_eat:         Vec<f32>,
    pub k_repro:       Vec<f32>,
    pub lambda_energy: Vec<f32>,
    pub k_curiosity:   Vec<f32>,
    pub k_progress:    Vec<f32>,

    /// Brain-tick counter, incremented every call. Training fires
    /// when `(tick % ROLLOUT_LEN) == 0` (after the per-tick rollout
    /// fill has happened).
    tick:        u64,

    pub device:  CudaDevice,
}

impl BrainPoolL3 {
    fn new(device: CudaDevice, n: usize) -> Self {
        Self {
            species:            HashMap::new(),
            slot_species:       vec![UNCLASSIFIED; n],
            free:               (0..n as u32).rev().collect(),
            map:                HashMap::with_capacity(n),
            buf_states:         vec![0.0; n * ROLLOUT_LEN * IN],
            buf_actions:        vec![0.0; n * ROLLOUT_LEN * OUT],
            buf_rewards:        vec![0.0; n * ROLLOUT_LEN],
            buf_count:          vec![0; n],
            prev_state:              vec![0.0; n * IN],
            prev_action:             vec![0.0; n * OUT],
            prev_energy:             vec![0.0; n],
            prev_reproductions:      vec![0; n],
            prev_predations:         vec![0; n],
            has_prev:                vec![false; n],

            target_entity:           vec![None; n],
            lock_ticks_remaining:    vec![0; n],
            prev_distance_to_target: vec![0.0; n],
            prev_target_entity:      vec![None; n],
            min_dist_to_target:      vec![f32::INFINITY; n],
            ticks_since_progress:    vec![0; n],
            blacklisted_target:      vec![None; n],
            blacklist_ticks_remaining: vec![0; n],
            applied_speed_a:         vec![0.0; n],

            sigma:              vec![0.0; n],
            k_eat:              vec![0.0; n],
            k_repro:            vec![0.0; n],
            lambda_energy:      vec![0.0; n],
            k_curiosity:        vec![0.0; n],
            k_progress:         vec![0.0; n],
            tick:               0,
            device,
        }
    }

    fn n(&self) -> usize { self.buf_count.len() }

    /// Lazily create a species' shared brain (fresh-init) on first sighting.
    /// Idempotent. New species fork off existing ones via the speciation
    /// system; their net simply starts fresh.
    pub fn ensure_species(&mut self, key: u32) {
        if !self.species.contains_key(&key) {
            let brain = new_species_brain(&self.device);
            self.species.insert(key, brain);
        }
    }

    /// Drop the shared nets of species with no enrolled member left (extinct or
    /// fully reclassified away). Without this the per-species map — and its GPU
    /// tensors + Adam state — grows unbounded over a long run, since `species_id`s
    /// are monotonic and never reused. `UNCLASSIFIED` is always kept.
    fn prune_species(&mut self) {
        let mut live: std::collections::HashSet<u32> =
            self.map.values().map(|&s| self.slot_species[s as usize]).collect();
        live.insert(UNCLASSIFIED);
        self.species.retain(|k, _| live.contains(k));
    }

    /// Snapshot every live species' shared weights + per-slot REINFORCE
    /// prev_* state for the `.colony` save format (`PoolSnapshot` from
    /// `rl_helpers.rs`). One GPU→CPU pull per live species.
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
        // Entity → species key (via slot → slot_species).
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
            hidden_dim:  HIDDEN,
            out_dim:     OUT,
        }
    }

    /// Build / overwrite a species' shared net from a saved `BrainRestore`.
    /// Mismatched dims → graceful failure (caller logs and the species keeps
    /// / falls back to a fresh net). Also seeds the species baseline from the
    /// restore payload. Does NOT touch per-slot prev_* (the per-level
    /// `assign_*` writes those directly from the same `BrainRestore`).
    pub fn restore_species(&mut self, key: u32, r: &BrainRestore) -> Result<(), String> {
        if r.w1.len() != IN * HIDDEN  { return Err(format!("w1 size {} != {}", r.w1.len(), IN * HIDDEN)); }
        if r.b1.len() != HIDDEN       { return Err(format!("b1 size {} != {}", r.b1.len(), HIDDEN)); }
        if r.w2.len() != HIDDEN * OUT { return Err(format!("w2 size {} != {}", r.w2.len(), HIDDEN * OUT)); }
        if r.b2.len() != OUT          { return Err(format!("b2 size {} != {}", r.b2.len(), OUT)); }
        if r.prev_state.len()  != IN  { return Err(format!("prev_state {} != {}", r.prev_state.len(), IN)); }
        if r.prev_action.len() != OUT { return Err(format!("prev_action {} != {}", r.prev_action.len(), OUT)); }

        self.ensure_species(key);
        let device = self.device.clone();
        let model = PoolMlpL3::<MyBackend> {
            w1: Param::from_tensor(Tensor::<MyBackend, 2>::from_data(
                TensorData::new(r.w1.clone(), [IN, HIDDEN]), &device)),
            b1: Param::from_tensor(Tensor::<MyBackend, 2>::from_data(
                TensorData::new(r.b1.clone(), [1, HIDDEN]), &device)),
            w2: Param::from_tensor(Tensor::<MyBackend, 2>::from_data(
                TensorData::new(r.w2.clone(), [HIDDEN, OUT]), &device)),
            b2: Param::from_tensor(Tensor::<MyBackend, 2>::from_data(
                TensorData::new(r.b2.clone(), [1, OUT]), &device)),
        };
        let brain = self.species.get_mut(&key).expect("species ensured above");
        brain.model    = model;
        brain.baseline = r.baseline;
        Ok(())
    }
}

impl FromWorld for BrainPoolL3 {
    fn from_world(world: &mut World) -> Self {
        let n = world
            .get_resource::<OrganismPoolSize>()
            .map(|r| r.0.max(1))
            .unwrap_or(1);
        let device = CudaDevice::default();
        warmup(&device);
        Self::new(device, n)
    }
}


/// Force-compile every kernel the steady-state apply tick + training
/// step will use, so the user's first real frame doesn't pay the JIT
/// cost. Mirrors the runtime tensor shapes (shared 2-D net, pooled batch).
fn warmup(device: &CudaDevice) {
    const WARMUP_SIGMA: f32 = 0.5;
    const M: usize = ROLLOUT_LEN;   // representative pooled batch size
    let m = new_species_brain(device);
    let mut o: Box<dyn BrainOptL3> = Box::new(AdamConfig::new().init());

    // ── Inference forward (every tick): a small batch of active rows.
    let i = Tensor::<MyBackend, 2>::zeros([4, IN], device);
    let _ = m.model.forward(i);

    // ── Training forward + backward (every ROLLOUT_LEN ticks): pooled [M, ·].
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


// ── Slot allocation systems ─────────────────────────────────────────────────

pub fn assign_brains_l3(
    mut pool:     NonSendMut<BrainPoolL3>,
    new:          Query<(Entity, &Organism, Option<&BrainInheritance>, Option<&BrainRestore>), (
        With<Heterotroph>,
        Without<BrainSlotL3>,
    )>,
    mut commands: Commands,
) {
    let mut rng = rand::rng();
    for (e, organism, _inheritance, restore) in new.iter() {
        // Enrol only Level3 heterotrophs — strict per-level dispatch. A wrong
        // level check here would double-enrol with the other pools, chaining
        // multiple apply systems that overwrite each other's movement command.
        if !matches!(organism.intelligence_level, IntelligenceLevel::Level3) { continue; }
        // Sliding pool — limb-based organisms enrol in the parallel
        // limb-based L3 pool instead (see `behaviour/limb_based_movement/`).
        if !organism.movement_mode.is_sliding() { continue; }

        let Some(slot) = pool.free.pop() else { continue };
        let s = slot as usize;

        // Per-species shared policy. A new occupant inherits its SPECIES net
        // automatically (every member uses the same weights), so:
        //   * BrainRestore (loaded colony save) → rebuild this entity's
        //     species net + restore its per-individual REINFORCE prev_*.
        //   * BrainInheritance → NO-OP (same-species newborns already share
        //     the trained species net); the marker is dropped below.
        let mut restored = false;
        if let Some(r) = restore {
            let key = organism.species_id.unwrap_or(UNCLASSIFIED);
            match pool.restore_species(key, r) {
                Ok(()) => {
                    restored = true;
                    let in_off  = s * IN;
                    let out_off = s * OUT;
                    pool.prev_state [in_off  .. in_off + IN ].copy_from_slice(&r.prev_state);
                    pool.prev_action[out_off .. out_off + OUT].copy_from_slice(&r.prev_action);
                    pool.prev_energy[s] = r.prev_energy;
                    pool.has_prev[s]    = r.has_prev;
                }
                Err(err) => error!("L3 hetero brain restore failed for {e:?}: {err} — using fresh slot"),
            }
        }

        // ── Hyperparameter genes ─────────────────────────────────
        // Fresh slots: uniform sample within each range. (Inheritance of
        // genes via the recycled slot's predecessor would require a parent
        // slot; under the shared policy the parent's WEIGHTS are already
        // shared via the species net, and genes resample like the swim pool's
        // per-organism exploration — keeping selection on differential
        // reproduction through the species net's learning, not gene copying.)
        pool.sigma[s]         = sample_range(L1_SIGMA_RANGE,         &mut rng);
        pool.k_eat[s]         = sample_range(L1_K_EAT_RANGE,         &mut rng);
        pool.k_repro[s]       = sample_range(L1_K_REPRO_RANGE,       &mut rng);
        pool.lambda_energy[s] = sample_range(L1_LAMBDA_ENERGY_RANGE, &mut rng);
        pool.k_curiosity[s]   = sample_range(L1_K_CURIOSITY_RANGE,   &mut rng);
        pool.k_progress[s]    = sample_range(L1_K_PROGRESS_RANGE,    &mut rng);

        // Reset transient target-lock + momentum state for the new
        // tenant. A stale target_entity could point at a despawned organism.
        pool.slot_species[s]            = organism.species_id.unwrap_or(UNCLASSIFIED);
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
        pool.buf_count[s]               = 0;

        if !restored {
            pool.has_prev[s]           = false;
            pool.prev_energy[s]        = organism.energy;
        }
        // `prev_reproductions` matches the organism either way so the first
        // tick doesn't fire a spurious `+K_REPRO` against a stale baseline.
        pool.prev_reproductions[s] = organism.reproductions;

        pool.map.insert(e, slot);
        commands.entity(e).try_insert(BrainSlotL3(slot));
        commands.entity(e).try_remove::<BrainInheritance>();
        commands.entity(e).try_remove::<BrainRestore>();
    }
}

pub fn free_brains_l3(
    mut pool:    NonSendMut<BrainPoolL3>,
    mut removed: RemovedComponents<Heterotroph>,
) {
    for e in removed.read() {
        if let Some(slot) = pool.map.remove(&e) {
            let s = slot as usize;
            pool.has_prev[s]             = false;
            pool.buf_count[s]            = 0;
            pool.slot_species[s]         = UNCLASSIFIED;
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

pub fn apply_intelligence_level_3(
    time:        Res<Time<Virtual>>,
    world_grid:  Res<WorldModelGrid>,
    mut pool:    NonSendMut<BrainPoolL3>,
    mut heteros: Query<(Entity, &mut Organism, &Transform, &BrainSlotL3, Option<&Carnivore>), With<Heterotroph>>,
    mut input_buf: Local<Vec<f32>>,
) {
    if time.is_paused() { return; }
    let n = pool.n();

    // ── Step 1: resolve neighbours and fill the input vector. ────
    input_buf.clear();
    input_buf.resize(n * IN, 0.0);

    // Cache per-active-slot scratch (entity, slot, position, energy_now,
    // reproductions_now, predations_now, neighbour list). Pre-resolving
    // neighbours here avoids re-walking the grid in the action-apply
    // loop where we need it for target-position lookup.
    struct Active {
        entity:        Entity,
        slot:          u32,
        energy_now:    f32,
        reproductions: u8,
        predations:    u8,
        is_carnivore:  bool,
        neighbours:    [Option<crate::world_model::Neighbour>; WORLD_MODEL_K],
    }
    let mut active: Vec<Active> = Vec::new();
    // Active slots grouped by species, for the batched forward below.
    let mut groups: HashMap<u32, Vec<usize>> = HashMap::new();

    for (e, organism, transform, slot, carn) in heteros.iter() {
        let s = slot.0 as usize;
        if s >= n { continue; }
        let pos = transform.translation;

        let max_e    = crate::energy::get_max_energy(&organism).max(1.0);
        let energy_n = (organism.energy / max_e).clamp(0.0, 1.0);
        let neighbours = collect_neighbours(&world_grid, pos);

        let off = s * IN;
        input_buf[off] = energy_n;

        // World-model block.
        encode_neighbours(&neighbours, &mut input_buf[off + 1 .. off + 1 + WORLD_MODEL_DIMS]);

        // Previous-action recurrence block — last sampled raw action
        // (speed_a + 4 target logits). Zero-filled before `has_prev`.
        let pa_off = off + 1 + WORLD_MODEL_DIMS;
        if pool.has_prev[s] {
            let src = s * OUT;
            for i in 0..OUT {
                input_buf[pa_off + i] = pool.prev_action[src + i];
            }
        }

        // Locked-target flag.
        let flag_off = pa_off + PREV_ACTION_DIMS;
        input_buf[flag_off] = if pool.target_entity[s].is_some() { 1.0 } else { 0.0 };

        // Record the organism's CURRENT species (read fresh every tick, so a
        // reclassification by the speciation system needs no special handling)
        // into `slot_species` for `train`, and group for the batched forward.
        let key = organism.species_id.unwrap_or(UNCLASSIFIED);
        pool.slot_species[s] = key;
        groups.entry(key).or_default().push(s);

        active.push(Active {
            entity: e, slot: slot.0,
            energy_now: organism.energy,
            reproductions: organism.reproductions,
            predations: organism.predations,
            is_carnivore: carn.is_some(),
            neighbours,
        });
    }
    if active.is_empty() {
        pool.tick = pool.tick.wrapping_add(1);
        return;
    }

    // ── Step 2: forward inference — ONE batched forward PER SPECIES through
    //            that species' shared net; results scatter back into a
    //            per-slot `mu_data` arena so the sampling/reward loop below
    //            is identical to the single-net case. ──
    let device = pool.device.clone();
    let mut mu_data = vec![0.0_f32; n * OUT];
    for (key, slots) in &groups {
        pool.ensure_species(*key);
        let brain = pool.species.get(key).expect("species ensured above");
        let cnt = slots.len();
        let mut rows = vec![0.0_f32; cnt * IN];
        for (i, &s) in slots.iter().enumerate() {
            rows[i * IN .. i * IN + IN].copy_from_slice(&input_buf[s * IN .. s * IN + IN]);
        }
        // No-grad inference (lever #1): forward on the inner non-autodiff backend.
        let model = brain.model.valid();
        let obs_t = Tensor::<crate::rl_helpers::InferBackend, 2>::from_data(TensorData::new(rows, [cnt, IN]), &device);
        let mu = model.forward(obs_t).into_data().into_vec::<f32>().expect("forward output");
        for (i, &s) in slots.iter().enumerate() {
            mu_data[s * OUT .. s * OUT + OUT].copy_from_slice(&mu[i * OUT .. i * OUT + OUT]);
        }
    }

    let mut rng = rand::rng();

    // ── Step 3: per-slot reward computation, action sampling,
    //             target selection (with lock + hysteresis), apply. ─
    for a in &active {
        let s = a.slot as usize;

        // (3a) Reward for the previous action (if any).
        if pool.has_prev[s] {
            // Eat detection: per-tick predation-count delta. True
            // event-based signal, replacing the old "energy spike"
            // heuristic that fired false positives on every energy
            // gain.
            let new_eats   = a.predations.wrapping_sub(pool.prev_predations[s]);
            let new_repros = a.reproductions.saturating_sub(pool.prev_reproductions[s]);
            let energy_delta = a.energy_now - pool.prev_energy[s];

            let r_eat   = pool.k_eat[s]   * new_eats   as f32;
            let r_repro = pool.k_repro[s] * new_repros as f32;
            let r_energy = if energy_delta < 0.0 {
                pool.lambda_energy[s] * energy_delta
            } else { 0.0 };

            // Curiosity / aggression: `K_CURIOSITY · (speed_norm − 0.5)`, so
            // stillness is NEGATIVE and full speed positive — an explicit
            // gradient away from the "do-nothing" optimum regardless of energy
            // / progress signals. References `applied_speed_a` (the EMA-smoothed
            // value actually written to `movement_speed` last tick — what the
            // world saw).
            let prev_speed_norm = pool.applied_speed_a[s].max(0.0);
            let r_curiosity = pool.k_curiosity[s] * (prev_speed_norm - 0.5);

            // Progress reward: how much closer to the locked target
            // did we get? Only credited when the target identity is
            // stable across the comparison window (set in step 3c
            // last tick); otherwise progress is meaningless because
            // we'd be comparing distances to two different organisms.
            let r_progress = if let Some(prev_target) = pool.prev_target_entity[s] {
                // Find prev target's current position via the
                // freshly-resolved neighbour list.
                let cur_d = a.neighbours.iter()
                    .find_map(|n| n.and_then(|nn|
                        if nn.entity == prev_target { Some(nn.rel.length()) } else { None }
                    ));
                match cur_d {
                    Some(d) => {
                        let closed = pool.prev_distance_to_target[s] - d;
                        pool.k_progress[s] * closed.max(0.0)
                    }
                    None => 0.0, // target out of sight ⇒ no progress signal
                }
            } else { 0.0 };

            let r = r_eat + r_repro + r_energy + r_curiosity + r_progress;

            // Push (prev_state, prev_action, reward) into rollout.
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

        // (3b) Sample a new action ~ N(μ, σ²) — 5 dims.
        let off = s * OUT;
        let sigma_s = pool.sigma[s];
        let mut action = [0.0_f32; OUT];
        for i in 0..OUT {
            action[i] = mu_data[off + i] + sigma_s * gaussian_noise(&mut rng);
        }
        let speed_a = action[0].clamp(-1.0, 1.0);
        let target_logits = [action[1], action[2], action[3], action[4]];

        // (3c-pre) Stuck detection.
        //
        // Find the locked target's current XZ distance (if still in
        // the neighbour list). If it has decreased by at least
        // `L1_STUCK_PROGRESS_EPS` vs. the running minimum, reset the
        // stuck counter; otherwise increment it. When the counter
        // reaches `L1_STUCK_TICKS` the lock is force-dropped this
        // tick — the agent is committed to a target it can't reach
        // (blocked by another organism, on the wrong side of a wall,
        // etc.) and should re-pick.
        //
        // Force-dropping works by zeroing the `locked_entity` we
        // hand to the target-selection scan: that scan will simply
        // not find a locked slot, falling through to a fresh argmax
        // pick. The book-keeping reset happens after the decision
        // (see (3c-post) below) regardless of which arm fired.
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
                // Only count as "stuck" OUTSIDE the brake zone. Inside it the
                // brake scales travel ∝ d, so legitimate controlled approach
                // produces sub-`L1_STUCK_PROGRESS_EPS` steps that would be
                // mis-flagged as failure-to-progress (dropping/blacklisting prey
                // the agent was about to eat). The brake guarantees d → 0 so
                // contact fires anyway; the detector still catches genuinely
                // blocked approach in the cruising zone.
                pool.ticks_since_progress[s] = pool.ticks_since_progress[s].saturating_add(1);
            }
        }
        let force_drop = pool.ticks_since_progress[s] >= L1_STUCK_TICKS;

        // Decrement the blacklist cooldown each tick; clear the
        // entry when it expires so the previously-blacklisted
        // entity becomes eligible again. Done BEFORE the scan so a
        // just-expired entity can be re-picked this tick.
        if pool.blacklist_ticks_remaining[s] > 0 {
            pool.blacklist_ticks_remaining[s] -= 1;
            if pool.blacklist_ticks_remaining[s] == 0 {
                pool.blacklisted_target[s] = None;
            }
        }
        // If we're about to force-drop, write the to-be-dropped
        // target into the blacklist NOW so the very same scan that
        // follows treats it as ineligible. The cooldown clock
        // starts from this tick.
        if force_drop {
            pool.blacklisted_target[s] = pool.target_entity[s];
            pool.blacklist_ticks_remaining[s] = L1_TARGET_BLACKLIST_TICKS;
        }
        let blacklisted_entity = pool.blacklisted_target[s];

        // (3c) Target selection — Level 1 hetero targets are PHOTOS
        //      ONLY. Type-check is applied at every gate (locked-slot
        //      match, argmax candidate) so no non-photo entity can
        //      ever survive to drive direction, even via a stale lock
        //      entry / reused Entity ID.
        //
        // Single pass over the K neighbour slots collects everything
        // the decision needs:
        //   * `locked_slot`  — index of the currently-locked entity if
        //                      it's STILL a photo neighbour this tick;
        //                      None otherwise (drops stale / non-prey
        //                      locks for free).
        //   * `best_slot`    — index of the photo neighbour with the
        //                      highest target logit (argmax over the
        //                      Photo-typed subset).
        // After the scan, a 3-arm match picks the winner.
        let locked_entity = if force_drop { None } else { pool.target_entity[s] };
        let mut locked_slot: Option<usize> = None;
        let mut best_slot:   Option<usize> = None;
        let mut best_logit:        f32      = f32::NEG_INFINITY;

        for (i, slot) in a.neighbours.iter().enumerate() {
            let Some(nn) = slot else { continue };
            // Classification-aware prey filter: carnivore agents
            // hunt heterotrophs (other animals); the default
            // herbivore agents hunt photoautotrophs.
            let valid_ty = if a.is_carnivore { OrganismType::Hetero } else { OrganismType::Photo };
            if nn.ty != valid_ty { continue; }
            // Skip blacklisted entities entirely — they're invisible
            // to both argmax AND the locked-slot match, so a hetero
            // that just dropped target T won't even consider T as
            // "still locked" while the cooldown runs.
            if Some(nn.entity) == blacklisted_entity { continue; }
            // Argmax over photo logits.
            let l = target_logits[i];
            if l > best_logit { best_logit = l; best_slot = Some(i); }
            // Locked-target match (entity ID + generation).
            if Some(nn.entity) == locked_entity { locked_slot = Some(i); }
        }

        let chosen_slot: Option<usize> = match (locked_slot, best_slot) {
            // Active lock window — keep, decrement counter, ignore
            // logits entirely.
            (Some(li), _) if pool.lock_ticks_remaining[s] > 0 => {
                pool.lock_ticks_remaining[s] -= 1;
                Some(li)
            }
            // Lock expired but locked target still in range — switch
            // only if a different photo beats it by SWITCH_MARGIN.
            (Some(li), Some(bi)) => {
                let chosen = if bi != li
                    && best_logit > target_logits[li] + L1_TARGET_SWITCH_MARGIN
                { bi } else { li };
                pool.lock_ticks_remaining[s] = L1_TARGET_LOCK_TICKS;
                Some(chosen)
            }
            // No usable lock — fresh argmax pick if any photo in sight.
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

        // (3d) Apply to the world: speed (EMA-smoothed) + direction
        //      (geometric, toward target).
        let Ok((_, mut org, _, _, _)) = heteros.get_mut(a.entity) else { continue };

        // EMA-smooth the speed sample for low-jerk movement, then
        // store the smoothed value so the curiosity reward next tick
        // references what the world actually saw.
        let smoothed = L1_SPEED_MOMENTUM_ALPHA * pool.applied_speed_a[s]
                     + (1.0 - L1_SPEED_MOMENTUM_ALPHA) * speed_a;
        pool.applied_speed_a[s] = smoothed;

        // Arrival braking + no-target speed cap.
        //
        // With a locked target: scale speed by
        // `clamp(dist_xz / L1_APPROACH_RADIUS, 0, 1)` — full speed
        // past R, linear decel inside, zero at the target. Kills
        // post-eat overshoot.
        //
        // Without a target: scale speed by `L1_NO_TARGET_SPEED_SCALE`
        // (0.3). The agent still patrols slowly so a wandering photo
        // can be discovered, but doesn't generate enough collision
        // force to deadlock against neighbouring heteros in
        // prey-empty regions.
        let brake_scale = match new_target {
            Some((_, rel)) => {
                let d_xz = (rel.x * rel.x + rel.z * rel.z).sqrt();
                (d_xz / L1_APPROACH_RADIUS).clamp(0.0, 1.0)
            }
            None => L1_NO_TARGET_SPEED_SCALE,
        };
        // Hard floor on world-facing speed: the world sees at least
        // `L1_MIN_APPLIED_SPEED · MAX_SPEED · brake_scale` even on negative
        // speed_a — an environment-side safeguard making the standstill
        // phenotype structurally impossible (the curiosity reward alone is
        // insufficient since the EMA baseline absorbs constant offsets).
        // `applied_speed_a` keeps the unclamped EMA so the policy gradient's
        // Gaussian log-prob math stays consistent.
        let applied_floored = smoothed.max(L1_MIN_APPLIED_SPEED);
        org.movement_speed = applied_floored * MAX_SPEED * brake_scale;

        // Direction is geometric from the chosen target, BUT we
        // freeze it at the previous tick's value when the target is
        // inside `L1_DIRECTION_FREEZE_DIST`. At very-close range the
        // unit-vector is hypersensitive to tiny prey wobble (a
        // 0.05-unit lateral wiggle at d = 0.2 produces a ~14° angle
        // swing), and the recompute-every-tick rule would visibly
        // micro-oscillate the agent. Combined with the arrival
        // brake the agent simply parks against the target until the
        // lock resolves naturally (eat / despawn / 10 s expiry).
        //
        // If no target picked, apply a small random rotation each
        // tick — a slow Brownian wander that breaks the case where
        // two heteros without prey settle into a stable mutual
        // deadlock. Combined with the no-target speed cap above,
        // the wander is gentle but enough to migrate out of an
        // empty region over a few seconds of virtual time.
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
                // Uniform sample in [-L1_NO_TARGET_WANDER_ANGLE,
                //                    +L1_NO_TARGET_WANDER_ANGLE].
                let angle = (rng.random::<f32>() - 0.5) * 2.0 * L1_NO_TARGET_WANDER_ANGLE;
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let dx = org.movement_direction.x;
                let dz = org.movement_direction.z;
                let new_dx = dx * cos_a - dz * sin_a;
                let new_dz = dx * sin_a + dz * cos_a;
                // Defensive renormalisation against drift (composing
                // many small rotations slowly inflates the norm).
                let len_sq = new_dx * new_dx + new_dz * new_dz;
                if len_sq > 1e-6 {
                    let inv = len_sq.sqrt().recip();
                    org.movement_direction = Vec3::new(new_dx * inv, 0.0, new_dz * inv);
                }
            }
        }

        // (3e) Update pool's prev_* state for next tick.
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

        // Target bookkeeping for next-tick reward shaping.
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

        // (3c-post) Stuck-detection book-keeping.
        //
        // If the target changed this tick (force-drop fired, switch
        // hysteresis chose a different photo, or the lock collapsed
        // because the previous target left the neighbour window),
        // reset the running-minimum distance to the new target's
        // current distance and zero the counter. The new lock window
        // gets `L1_STUCK_TICKS` fresh ticks to prove progress.
        //
        // If the target is the SAME as last tick (most common —
        // pure lock-hold), the increment-or-reset already happened
        // in (3c-pre) before the selection scan. Nothing more to do.
        if new_target_ent != prev_target_ent {
            pool.min_dist_to_target[s] = match new_target {
                Some((_, rel)) => (rel.x * rel.x + rel.z * rel.z).sqrt(),
                None           => f32::INFINITY,
            };
            pool.ticks_since_progress[s] = 0;
        }
    }

    // ── Step 4: maybe train. ────────────────────────────────────
    pool.tick = pool.tick.wrapping_add(1);
    if pool.tick % ROLLOUT_LEN as u64 != 0 { return; }
    train(&mut pool);
}


/// Run one Monte Carlo REINFORCE update PER SPECIES over the current contents
/// of every slot's rollout buffer. Active slots (with buffered steps) are
/// grouped by their CURRENT species; for each species we compute discounted
/// returns, update that species' EMA baseline, pool the transitions into one
/// flat `[M, ·]` batch, and run one Gaussian-log-prob REINFORCE step on that
/// species' net. Slots with `buf_count == 0` contribute nothing. After the
/// update, every slot's buffer is reset.
fn train(pool: &mut BrainPoolL3) {
    pool.prune_species();
    let n = pool.n();

    // Group active slots (non-empty rollout) by their current species.
    let mut groups: HashMap<u32, Vec<usize>> = HashMap::new();
    for s in 0..n {
        if pool.buf_count[s] > 0 {
            groups.entry(pool.slot_species[s]).or_default().push(s);
        }
    }
    let device = pool.device.clone();

    for (key, slots) in groups {
        pool.ensure_species(key);

        // ── Per-slot discounted returns + per-SPECIES EMA baseline update. ──
        // First compute returns for every contributing slot and accumulate
        // the mean-of-mean-returns across this species, then update the
        // species baseline once and form advantages.
        struct SlotRet { slot: usize, count: usize, returns: Vec<f32> }
        let mut slot_rets: Vec<SlotRet> = Vec::with_capacity(slots.len());
        let mut mean_g_sum = 0.0_f32;
        for &s in &slots {
            let count = pool.buf_count[s];
            if count == 0 { continue; }
            // G_t = r_t + γ · G_{t+1}.
            let mut returns = vec![0.0_f32; count];
            let mut g = 0.0_f32;
            for t in (0..count).rev() {
                let r = pool.buf_rewards[s * ROLLOUT_LEN + t];
                g = r + GAMMA * g;
                returns[t] = g;
            }
            let mean_g: f32 = returns.iter().sum::<f32>() / count as f32;
            mean_g_sum += mean_g;
            slot_rets.push(SlotRet { slot: s, count, returns });
        }
        if slot_rets.is_empty() { continue; }

        // Per-species baseline: EMA of the species' mean rollout return.
        let species_mean_g = mean_g_sum / slot_rets.len() as f32;
        {
            let brain = pool.species.get_mut(&key).expect("species ensured above");
            brain.baseline = (1.0 - BASELINE_ALPHA) * brain.baseline
                           + BASELINE_ALPHA * species_mean_g;
        }
        let baseline = pool.species.get(&key).expect("species ensured above").baseline;

        // ── Pool this species' transitions into flat [M, ·] buffers. ──
        // Per-slot σ scaling is preserved: the loss for each transition is
        // `0.5/σ_s² · Σ(a−μ)² · A`, so we fold σ into the per-transition
        // advantage (adv_scaled = adv · 0.5/σ²) before the GPU reduction.
        let mut states:  Vec<f32> = Vec::new();
        let mut actions: Vec<f32> = Vec::new();
        let mut adv:     Vec<f32> = Vec::new();
        let mut total_count = 0.0_f32;
        for sr in &slot_rets {
            let s = sr.slot;
            let sigma = pool.sigma[s].max(1e-3);
            let scale = 0.5_f32 / (sigma * sigma);
            for t in 0..sr.count {
                let buf_in_off  = s * ROLLOUT_LEN * IN + t * IN;
                let buf_out_off = s * ROLLOUT_LEN * OUT + t * OUT;
                states.extend_from_slice(&pool.buf_states [buf_in_off  .. buf_in_off  + IN ]);
                actions.extend_from_slice(&pool.buf_actions[buf_out_off .. buf_out_off + OUT]);
                adv.push((sr.returns[t] - baseline) * scale);
                total_count += 1.0;
            }
        }
        let m = adv.len();
        if m == 0 || total_count < 1.0 { continue; }

        // ── GPU tensors ([M, ·]). ──
        let states_t  = Tensor::<MyBackend, 2>::from_data(TensorData::new(states,  [m, IN]),  &device);
        let actions_t = Tensor::<MyBackend, 2>::from_data(TensorData::new(actions, [m, OUT]), &device);
        let adv_t     = Tensor::<MyBackend, 2>::from_data(TensorData::new(adv,     [m, 1]),   &device);

        // ── REINFORCE step on THIS species' net. ──
        let brain = pool.species.get_mut(&key).expect("species ensured above");
        let mu = brain.model.forward(states_t);                 // [M, OUT]
        let diff = actions_t - mu;
        let sum_sq = diff.powf_scalar(2.0).sum_dim(1);          // [M, 1]
        // `adv_t` already carries the 0.5/σ² per-transition scale.
        let loss = (sum_sq * adv_t).sum().div_scalar(total_count);

        let cm = brain.model.clone();
        let gp = GradientsParams::from_grads(loss.backward(), &brain.model);
        brain.model = brain.opt.step(LR, cm, gp);
        brain.tick = brain.tick.wrapping_add(1);
    }

    // Reset per-slot buffers for the next rollout window.
    for s in 0..n { pool.buf_count[s] = 0; }
}
