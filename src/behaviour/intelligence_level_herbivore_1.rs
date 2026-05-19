// Intelligence Level 1 — Herbivore RL pool.
//
// Pure REINFORCE policy gradient over a per-organism MLP. There is
// NO hand-coded movement logic in this file: every speed and every
// direction commanded onto an organism comes from the network
// output. The agent learns to seek out and eat photoautotrophs by
// maximising the per-tick `Δdopamine` reward.
//
// Reward sources (see also `predation.rs`, `reproduction.rs`,
// `energy.rs::deplete_dopamine`):
//   * +0.6  dopamine on every successful photo consumption (capped 1.0)
//   * +1.0  dopamine on every successful reproduction (sets, not adds)
//   *  −H/3 dopamine per virtual second (H = hunger), so an idle
//           hungry organism sees its reward signal decay and the
//           policy is punished for inaction.
//
// Architecture: 9 → 16 → 3 MLP with `tanh` head.
//
//   Inputs (9):
//     [0] hunger              ∈ [0, 1]
//     [1] dopamine            ∈ [0, 1]
//     [2] nearest_photo.rel_x / WORLD_MODEL_RADIUS   (0 if no photo)
//     [3] nearest_photo.rel_z / WORLD_MODEL_RADIUS
//     [4] has_photo                                   (0 / 1)
//     [5] prev_action.speed_a                         (∈ [-1, 1])
//     [6] prev_action.dir_x
//     [7] prev_action.dir_z
//     [8] target_distance / SENSORY_RADIUS            ∈ [0, 1]
//                                                     (1 = out of
//                                                      sensory range)
//
//   Outputs (3, tanh):
//     [0] speed_a    → speed     = ((speed_a + 1) / 2) · MAX_SPEED
//     [1] dir_x      → direction.x          (XZ-unit-normalised
//     [2] dir_z      → direction.z           before commit)
//
// Training: 1-step REINFORCE with EMA baseline.
//   For every active slot whose previous tick was also active:
//     reward    = dopamine_now − prev_dopamine
//     advantage = reward − baseline
//     loss      = 0.5 · Σ_o ((a_{t-1,o} − μ_{t-1,o}) / σ)² · advantage
//     baseline  ← α · baseline + (1 − α) · reward
//   Loss summed over slots and one Adam step taken per brain tick.
//
// Slot recycling preserves trained weights (instinct survives the
// previous tenant's death) but resets prev_*/baseline/has_prev so
// the first post-recycle tick never computes a phantom reward
// across the slot boundary.

use bevy::prelude::*;
use burn::module::{Initializer, Module, Param};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use burn_cuda::CudaDevice;
use std::collections::HashMap;

use crate::colony::{Carnivore, IntelligenceLevel, Organism, Heterotroph};
use crate::rl_helpers::{BrainInheritance, MyBackend, gaussian_noise};
use crate::sensory::SENSORY_RADIUS;
use crate::simulation_settings::OrganismPoolSize;
use crate::world_model::{WorldModelGrid, WORLD_MODEL_RADIUS, nearest_prey};


// ── Hyperparameters ─────────────────────────────────────────────────────────

const IN:        usize = 9;
const HIDDEN:    usize = 16;
const OUT:       usize = 3;
const MAX_SPEED: f32   = 40.0;
/// Exploration noise std-dev added to each output dim before clamp.
/// A larger σ gives noisier action selection (more exploration), a
/// smaller σ tightens around the policy mean. 0.25 is the L2/L3
/// brain's default.
const SIGMA:          f32 = 0.25;
const LR:             f64 = 1e-3;
const BASELINE_ALPHA: f32 = 0.95;

/// Weight on the per-tick Δ`target_distance` progress reward.
/// Composed with the Δ`dopamine` reward as
///
///     reward = Δdopamine + PROGRESS_REWARD_WEIGHT · ((prev_td − td) / SENSORY_RADIUS)
///
/// The normalised progress term lies in `[-1, 1]` per tick, the
/// same scale as Δdopamine. Picking 0.3 keeps dopamine as the
/// dominant teaching signal while still pulling the policy toward
/// "close the distance" when no eating event happens. A higher
/// weight would let progress drown out the rarer eating signal;
/// lower would leave the agent under-trained between rare meals.
const PROGRESS_REWARD_WEIGHT: f32 = 0.3;


// ── Slot marker ─────────────────────────────────────────────────────────────

#[derive(Component, Clone, Copy)]
pub struct BrainSlotHerbivore1(pub u32);


// ── Per-organism MLP ────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct PoolMlpHerbivore1<B: Backend> {
    w1: Param<Tensor<B, 3>>,
    b1: Param<Tensor<B, 2>>,
    w2: Param<Tensor<B, 3>>,
    b2: Param<Tensor<B, 2>>,
}

impl<B: Backend> PoolMlpHerbivore1<B> {
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
}

trait BrainOptHerbivore1 {
    fn step(
        &mut self,
        lr: f64,
        m:  PoolMlpHerbivore1<MyBackend>,
        g:  GradientsParams,
    ) -> PoolMlpHerbivore1<MyBackend>;
}

impl<O: Optimizer<PoolMlpHerbivore1<MyBackend>, MyBackend>> BrainOptHerbivore1 for O {
    fn step(
        &mut self,
        lr: f64,
        m:  PoolMlpHerbivore1<MyBackend>,
        g:  GradientsParams,
    ) -> PoolMlpHerbivore1<MyBackend> {
        Optimizer::step(self, lr, m, g)
    }
}


// ── Pool resource ───────────────────────────────────────────────────────────

pub struct BrainPoolHerbivore1 {
    model:  PoolMlpHerbivore1<MyBackend>,
    opt:    Box<dyn BrainOptHerbivore1>,
    free:   Vec<u32>,
    map:    HashMap<Entity, u32>,
    n:      usize,
    pub device: CudaDevice,

    /// Flat per-slot REINFORCE state. Indexed by slot (NOT by Vec
    /// capacity); see the matching field on the old supervised pool
    /// for the rationale.
    prev_state:           Vec<f32>,   // [N · IN]
    prev_action:          Vec<f32>,   // [N · OUT]
    prev_dopamine:        Vec<f32>,   // [N]
    prev_target_distance: Vec<f32>,   // [N]
    baseline:             Vec<f32>,   // [N]
    has_prev:             Vec<bool>,  // [N]
}

impl BrainPoolHerbivore1 {
    fn new(device: CudaDevice, n: usize) -> Self {
        Self {
            model: PoolMlpHerbivore1::<MyBackend>::new(&device, n),
            opt:   Box::new(AdamConfig::new().init()),
            free:  (0..n as u32).rev().collect(),
            map:   HashMap::with_capacity(n),
            n,
            device,
            prev_state:           vec![0.0;            n * IN],
            prev_action:          vec![0.0;            n * OUT],
            prev_dopamine:        vec![0.0;            n],
            // Initialise to the "out of range" sentinel so the
            // first Δ on a freshly-recycled slot reads zero
            // when no photo is in range (target_distance also
            // starts at SENSORY_RADIUS).
            prev_target_distance: vec![SENSORY_RADIUS; n],
            baseline:             vec![0.0;            n],
            has_prev:             vec![false;          n],
        }
    }
    fn n(&self) -> usize { self.n }

    fn inherit_row(&mut self, parent: usize, child: usize) {
        let p = self.model.w1.val().slice([parent..parent+1, 0..IN, 0..HIDDEN]);
        self.model.w1 = self.model.w1.clone().map(|t| t.slice_assign([child..child+1, 0..IN, 0..HIDDEN], p));
        let p = self.model.b1.val().slice([parent..parent+1, 0..HIDDEN]);
        self.model.b1 = self.model.b1.clone().map(|t| t.slice_assign([child..child+1, 0..HIDDEN], p));
        let p = self.model.w2.val().slice([parent..parent+1, 0..HIDDEN, 0..OUT]);
        self.model.w2 = self.model.w2.clone().map(|t| t.slice_assign([child..child+1, 0..HIDDEN, 0..OUT], p));
        let p = self.model.b2.val().slice([parent..parent+1, 0..OUT]);
        self.model.b2 = self.model.b2.clone().map(|t| t.slice_assign([child..child+1, 0..OUT], p));
    }

    /// Wipe the REINFORCE bookkeeping for one slot. Called every time
    /// a slot is freshly assigned so the first tick after assignment
    /// does NOT try to compute a reward against the previous tenant's
    /// dopamine value. Trained weights survive this reset (instinct).
    fn reset_slot_state(&mut self, s: usize) {
        for i in 0..IN  { self.prev_state [s * IN  + i] = 0.0; }
        for i in 0..OUT { self.prev_action[s * OUT + i] = 0.0; }
        self.prev_dopamine[s]        = 0.0;
        self.prev_target_distance[s] = SENSORY_RADIUS;
        self.baseline[s]             = 0.0;
        self.has_prev[s]             = false;
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

/// Force CubeCL to compile the forward + backward kernels at the
/// max batch shape so the first real brain tick doesn't stall.
fn warmup(device: &CudaDevice, n: usize) {
    let m = PoolMlpHerbivore1::<MyBackend>::new(device, n);
    let mut o: Box<dyn BrainOptHerbivore1> = Box::new(AdamConfig::new().init());
    let i = Tensor::<MyBackend, 2>::zeros([n, IN], device);
    let mu = m.forward(i.clone());
    let target = Tensor::<MyBackend, 2>::zeros([n, OUT], device);
    let diff = mu - target;
    let loss = diff.powf_scalar(2.0).sum().div_scalar(1.0_f32);
    let g = GradientsParams::from_grads(loss.backward(), &m);
    let _ = o.step(LR, m, g);
}


// ── Slot assignment / release ──────────────────────────────────────────────

pub fn assign_brains_herbivore_1(
    mut pool:     NonSendMut<BrainPoolHerbivore1>,
    new:          Query<
        (Entity, &Organism, Option<&BrainInheritance>),
        (
            With<Heterotroph>,
            Without<BrainSlotHerbivore1>,
            // Carnivores route through IL2 / IL3 instead — never
            // enrol them in the herbivore pool.
            Without<Carnivore>,
        ),
    >,
    mut commands: Commands,
) {
    for (e, organism, inheritance) in new.iter() {
        if !matches!(organism.intelligence_level, IntelligenceLevel::Level1) { continue; }
        let Some(slot) = pool.free.pop() else { continue };
        let s = slot as usize;
        if let Some(BrainInheritance(parent)) = inheritance {
            if let Some(&parent_slot) = pool.map.get(parent) {
                pool.inherit_row(parent_slot as usize, s);
            }
        }
        pool.reset_slot_state(s);
        pool.map.insert(e, slot);
        commands.entity(e).try_insert(BrainSlotHerbivore1(slot));
        commands.entity(e).try_remove::<BrainInheritance>();
    }
}

pub fn free_brains_herbivore_1(
    mut pool:    NonSendMut<BrainPoolHerbivore1>,
    mut removed: RemovedComponents<Heterotroph>,
) {
    for e in removed.read() {
        if let Some(slot) = pool.map.remove(&e) {
            pool.free.push(slot);
        }
    }
}


// ── Apply tick — observe, act, learn ───────────────────────────────────────

pub fn apply_intelligence_level_herbivore_1(
    time:        Res<Time<Virtual>>,
    world_grid:  Res<WorldModelGrid>,
    mut pool:    NonSendMut<BrainPoolHerbivore1>,
    mut heteros: Query<
        (Entity, &mut Organism, &Transform, &BrainSlotHerbivore1),
        (With<Heterotroph>, Without<Carnivore>),
    >,
    mut input_buf:       Local<Vec<f32>>,
    mut prev_state_buf:  Local<Vec<f32>>,
    mut prev_action_buf: Local<Vec<f32>>,
    mut adv_buf:         Local<Vec<f32>>,
    mut mask_buf:        Local<Vec<f32>>,
) {
    if time.is_paused() { return; }
    let n = pool.n();

    input_buf      .clear(); input_buf      .resize(n * IN,  0.0);
    prev_state_buf .clear(); prev_state_buf .resize(n * IN,  0.0);
    prev_action_buf.clear(); prev_action_buf.resize(n * OUT, 0.0);
    adv_buf        .clear(); adv_buf        .resize(n * OUT, 0.0);
    mask_buf       .clear(); mask_buf       .resize(n * OUT, 0.0);

    struct Active {
        entity:          Entity,
        slot:            u32,
        dopamine:        f32,
        target_distance: f32,
    }
    let mut active: Vec<Active> = Vec::new();

    // ── Pass 1: build observations, compute reward/advantage,
    //           copy prev_state + prev_action into the batch slots.
    for (e, organism, transform, slot) in heteros.iter() {
        let s = slot.0 as usize;
        if s >= n { continue; }
        let pos = transform.translation;

        // Current state observation.
        let in_off = s * IN;
        let pa_off = s * OUT;
        let (rel_x, rel_z, has_photo) = match nearest_prey(&world_grid, pos) {
            Some((rel, _, _)) => (
                (rel.x / WORLD_MODEL_RADIUS).clamp(-1.0, 1.0),
                (rel.z / WORLD_MODEL_RADIUS).clamp(-1.0, 1.0),
                1.0,
            ),
            None => (0.0, 0.0, 0.0),
        };
        // Normalised target-distance observation. 1.0 means "no photo
        // within sensory radius"; 0.0 means "right on top of one".
        let td_norm = (organism.target_distance / SENSORY_RADIUS).clamp(0.0, 1.0);

        input_buf[in_off + 0] = organism.hunger;
        input_buf[in_off + 1] = organism.dopamine;
        input_buf[in_off + 2] = rel_x;
        input_buf[in_off + 3] = rel_z;
        input_buf[in_off + 4] = has_photo;
        input_buf[in_off + 5] = pool.prev_action[pa_off + 0];
        input_buf[in_off + 6] = pool.prev_action[pa_off + 1];
        input_buf[in_off + 7] = pool.prev_action[pa_off + 2];
        input_buf[in_off + 8] = td_norm;

        // Reward + advantage from the previous step, if we have one.
        if pool.has_prev[s] {
            // Primary reward: Δdopamine (eating, reproducing → big
            // positive spikes; idle decay → small negatives).
            let r_dopamine = organism.dopamine - pool.prev_dopamine[s];
            // Secondary reward: progress on closing the distance to
            // the nearest photo. Positive when the target got closer
            // this tick, negative when it moved away (or disappeared
            // from sensory range, which jumps `target_distance` up to
            // `SENSORY_RADIUS`).
            let r_progress = (pool.prev_target_distance[s] - organism.target_distance)
                             / SENSORY_RADIUS;
            let reward    = r_dopamine + PROGRESS_REWARD_WEIGHT * r_progress;
            let baseline  = pool.baseline[s];
            let advantage = reward - baseline;
            // Per-output broadcast of the slot's scalar advantage
            // and mask. Inactive slots stay 0 in both → contribute
            // nothing to the loss.
            for o in 0..OUT {
                adv_buf [s * OUT + o] = advantage;
                mask_buf[s * OUT + o] = 1.0;
            }
            pool.baseline[s] = BASELINE_ALPHA * baseline
                             + (1.0 - BASELINE_ALPHA) * reward;
            // Snapshot prev_state / prev_action into the GPU batch.
            for i in 0..IN  { prev_state_buf [in_off + i] = pool.prev_state [in_off + i]; }
            for i in 0..OUT { prev_action_buf[pa_off + i] = pool.prev_action[pa_off + i]; }
        }

        active.push(Active {
            entity:          e,
            slot:            slot.0,
            dopamine:        organism.dopamine,
            target_distance: organism.target_distance,
        });
    }
    if active.is_empty() { return; }

    // ── Forward pass on CURRENT state — picks this tick's μ. ──
    let cur_t = Tensor::<MyBackend, 2>::from_data(
        TensorData::new(input_buf.clone(), [n, IN]),
        &pool.device,
    );
    let mu_cur      = pool.model.forward(cur_t);
    let mu_cur_data = mu_cur.into_data().into_vec::<f32>().expect("forward output");

    // ── Sample a_t = μ_t + σ · ε, drive movement directly from a_t. ──
    let mut rng = rand::rng();
    let mut sampled_action = vec![0.0_f32; n * OUT];
    for a in active.iter() {
        let s   = a.slot as usize;
        let off = s * OUT;
        let mut act = [0.0_f32; OUT];
        for o in 0..OUT {
            let mu    = mu_cur_data[off + o];
            let noise = gaussian_noise(&mut rng);
            act[o]    = (mu + SIGMA * noise).clamp(-1.0, 1.0);
            sampled_action[off + o] = act[o];
        }

        // Network output → organism. No clamps beyond what tanh +
        // the clamp() above already provide; no hand-coded oracle
        // or wander logic.
        let speed_a = act[0];
        let dir_x   = act[1];
        let dir_z   = act[2];
        let speed   = ((speed_a + 1.0) * 0.5).clamp(0.0, 1.0) * MAX_SPEED;
        let dir_xz  = Vec3::new(dir_x, 0.0, dir_z);
        let mag     = dir_xz.length();
        if let Ok((_, mut org, _, _)) = heteros.get_mut(a.entity) {
            org.movement_speed = speed;
            // Only update direction when the network output has
            // discernible magnitude — at near-zero magnitude the
            // unit vector is undefined, keep the last commanded
            // heading so yaw doesn't snap randomly.
            if mag > 1e-3 {
                org.movement_direction = dir_xz / mag;
            }
        }
    }

    // ── REINFORCE update on the PREVIOUS step. ─────────────────
    let active_with_prev: f32 = mask_buf.iter().step_by(OUT).sum();
    if active_with_prev > 0.0 {
        let prev_t = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(prev_state_buf.clone(), [n, IN]),
            &pool.device,
        );
        let act_t  = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(prev_action_buf.clone(), [n, OUT]),
            &pool.device,
        );
        let adv_t  = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(adv_buf.clone(),  [n, OUT]),
            &pool.device,
        );
        let mask_t = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(mask_buf.clone(), [n, OUT]),
            &pool.device,
        );

        let mu_prev = pool.model.forward(prev_t);
        // loss = −log π(a | μ, σ) · advantage
        //      = 0.5 · ((a − μ) / σ)² · advantage (+ const)
        let diff    = act_t - mu_prev;
        let half_sq = diff.powf_scalar(2.0).mul_scalar(0.5 / (SIGMA * SIGMA));
        let rows    = half_sq * adv_t * mask_t;
        let denom   = active_with_prev.max(1.0);
        let loss    = rows.sum().div_scalar(denom);

        let cm = pool.model.clone();
        let gp = GradientsParams::from_grads(loss.backward(), &pool.model);
        pool.model = pool.opt.step(LR, cm, gp);
    }

    // ── Commit current → prev for the next tick. ───────────────
    for a in active.iter() {
        let s = a.slot as usize;
        let in_off = s * IN;
        let pa_off = s * OUT;
        for i in 0..IN  { pool.prev_state [in_off + i] = input_buf     [in_off + i]; }
        for i in 0..OUT { pool.prev_action[pa_off + i] = sampled_action[pa_off + i]; }
        pool.prev_dopamine[s]        = a.dopamine;
        pool.prev_target_distance[s] = a.target_distance;
        pool.has_prev[s]             = true;
    }
}
