// Intelligence Level 1 — Herbivore SUPERVISED pool.
//
// Replaces the previous `intelligence_level_1_hetero.rs` REINFORCE
// brain for the herbivore (default-heterotroph) case. The network is
// trained by supervised regression against an oracle that always
// knows the right answer:
//
//     oracle target = (speed = +1, direction = unit vector toward
//                      the nearest photoautotroph)
//
// No exploration noise, no rollouts, no policy gradient. Every brain
// tick:
//   1. Snapshot inputs (`energy_norm` + world-model) for every active
//      slot.
//   2. Forward pass on the GPU.
//   3. Apply the predicted speed + direction to the organism.
//   4. Build oracle target tensors. Slots with no photo in range get
//      a `mask = 0` row (no gradient).
//   5. Masked MSE loss + Adam step.
//
// Because the oracle is correct by construction, the network
// converges in seconds rather than the minutes-to-hours the
// REINFORCE pipeline needed.
//
// Per-organism per-slot weights — recycled-on-despawn the same way
// every other pool in this codebase does. The pool only enrols
// heterotrophs at `IntelligenceLevel::Level1` that are NOT marked
// `Carnivore` (carnivores use IL2 / IL3 instead).

use bevy::prelude::*;
use burn::module::{Initializer, Module, Param};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use burn_cuda::CudaDevice;
use std::collections::HashMap;

use crate::colony::{Carnivore, IntelligenceLevel, Organism, Heterotroph};
use crate::energy::get_max_energy;
use crate::rl_helpers::{BrainInheritance, MyBackend};
use crate::simulation_settings::OrganismPoolSize;
use crate::world_model::{WorldModelGrid, WORLD_MODEL_RADIUS, nearest_prey};


// ── Architecture ────────────────────────────────────────────────────────────
//
// Inputs (4): explicit nearest-photo block + self energy.
//
//   [0]  energy / max_energy                        (∈ [0, 1])
//   [1]  nearest_photo.rel_x / WORLD_MODEL_RADIUS   (∈ [-1, 1])
//   [2]  nearest_photo.rel_z / WORLD_MODEL_RADIUS   (∈ [-1, 1])
//   [3]  has_photo                                   (0 or 1)
//
// The old design fed the 4-nearest world-model block here and used
// `nearest_prey` for the oracle target — but the two operate over
// DIFFERENT subsets (nearest 4 of any type vs nearest photo in range),
// so the inputs frequently lacked information about the very photo
// the oracle was pointing at. The supervised gradient was then
// learning "given these heteros, output a direction" — uncorrelated
// noise that collapses the policy toward the mean (≈ zero direction).
//
// With the explicit nearest-photo block, the input-output mapping is
// trivial: output `(1, x, z)` when input is `(_, x, z, 1)`. A
// 4-hidden-unit network would learn it in seconds; we keep 8 hidden
// units for a little headroom against the input noise.
const IN:        usize = 4;
const HIDDEN:    usize = 8;
const OUT:       usize = 3;                      // speed_a, dir_x, dir_z
const MAX_SPEED: f32   = 20.0;
const LR:        f64   = 1e-3;


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
    model: PoolMlpHerbivore1<MyBackend>,
    opt:   Box<dyn BrainOptHerbivore1>,
    free:  Vec<u32>,
    map:   HashMap<Entity, u32>,
    /// Static batch dimension. Set once at construction from
    /// `OrganismPoolSize`; matches the row count of every weight
    /// tensor and is the only correct length to use when building
    /// per-tick input / target / mask tensors. `Vec::capacity()` is
    /// NOT a substitute — it's the Vec's allocation size and drifts
    /// from the static N as items are pushed/popped (Bevy 0.18's
    /// allocator policy can grow capacity past the initial value).
    n:     usize,
    pub device: CudaDevice,
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
    let m = PoolMlpHerbivore1::<MyBackend>::new(device, n);
    let mut o: Box<dyn BrainOptHerbivore1> = Box::new(AdamConfig::new().init());
    let i = Tensor::<MyBackend, 2>::zeros([n, IN], device);
    let mu = m.forward(i.clone());
    let target = Tensor::<MyBackend, 2>::zeros([n, OUT], device);
    let mask   = Tensor::<MyBackend, 2>::zeros([n, OUT], device);
    let diff = mu - target;
    let loss = (diff.powf_scalar(2.0) * mask).sum().div_scalar(1.0_f32);
    let g = GradientsParams::from_grads(loss.backward(), &m);
    let _ = o.step(LR, m, g);
}


// ── Slot assignment ─────────────────────────────────────────────────────────

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


// ── Apply tick (forward + supervised MSE backward) ──────────────────────────

pub fn apply_intelligence_level_herbivore_1(
    time:        Res<Time<Virtual>>,
    world_grid:  Res<WorldModelGrid>,
    mut pool:    NonSendMut<BrainPoolHerbivore1>,
    mut heteros: Query<(Entity, &mut Organism, &Transform, &BrainSlotHerbivore1), (With<Heterotroph>, Without<Carnivore>)>,
    mut input_buf:  Local<Vec<f32>>,
    mut target_buf: Local<Vec<f32>>,
    mut mask_buf:   Local<Vec<f32>>,
) {
    if time.is_paused() { return; }
    // Static batch dimension — must match the weight tensors' row
    // count exactly. See `BrainPoolHerbivore1::n` for why this is
    // stored explicitly instead of derived from `free.capacity()`.
    let n = pool.n();

    input_buf.clear();  input_buf.resize(n * IN,  0.0);
    target_buf.clear(); target_buf.resize(n * OUT, 0.0);
    mask_buf.clear();   mask_buf.resize(n * OUT, 0.0);

    // Per-slot transient: (entity, slot, pos).
    let mut active: Vec<(Entity, u32, Vec3)> = Vec::new();

    for (e, organism, transform, slot) in heteros.iter() {
        let s = slot.0 as usize;
        if s >= n { continue; }
        let pos = transform.translation;

        let max_e    = get_max_energy(&organism).max(1.0);
        let energy_n = (organism.energy / max_e).clamp(0.0, 1.0);

        let off  = s * IN;
        let toff = s * OUT;
        input_buf[off] = energy_n;

        // Single oracle lookup powers BOTH the input block and the
        // target. Inputs and target now reference the same photo, so
        // the supervised mapping is well-defined: the network sees
        // exactly the position it must point at.
        if let Some((rel, _, _)) = nearest_prey(&world_grid, pos) {
            let mag2 = rel.x * rel.x + rel.z * rel.z;
            if mag2 > 1e-6 {
                let rel_x_n = rel.x / WORLD_MODEL_RADIUS;
                let rel_z_n = rel.z / WORLD_MODEL_RADIUS;
                input_buf[off + 1] = rel_x_n;
                input_buf[off + 2] = rel_z_n;
                input_buf[off + 3] = 1.0;          // has_photo flag

                let inv = mag2.sqrt().recip();
                target_buf[toff + 0] = 1.0;        // speed = full
                target_buf[toff + 1] = rel.x * inv;
                target_buf[toff + 2] = rel.z * inv;
                mask_buf[toff + 0]   = 1.0;
                mask_buf[toff + 1]   = 1.0;
                mask_buf[toff + 2]   = 1.0;
            }
        }
        // else: input dims [1..3] stay zero, has_photo stays zero,
        // mask stays zero → no gradient.

        active.push((e, slot.0, pos));
    }
    if active.is_empty() { return; }

    // ── Forward inference. ──────────────────────────────────────
    let cur_t  = Tensor::<MyBackend, 2>::from_data(TensorData::new(input_buf.clone(),  [n, IN]),  &pool.device);
    let mu_cur = pool.model.forward(cur_t.clone());
    let mu_data = mu_cur.clone().into_data().into_vec::<f32>().expect("forward output");

    // ── Apply predicted action to each active organism. ──────────
    for &(entity, slot, _pos) in active.iter() {
        let s = slot as usize;
        let off = s * OUT;
        let speed_a = mu_data[off + 0].clamp(-1.0, 1.0).max(0.0);
        let dx      = mu_data[off + 1];
        let dz      = mu_data[off + 2];

        let Ok((_, mut org, _, _)) = heteros.get_mut(entity) else { continue };
        org.movement_speed = speed_a * MAX_SPEED;
        let mag2 = dx * dx + dz * dz;
        if mag2 > 1e-6 {
            let inv = mag2.sqrt().recip();
            org.movement_direction = Vec3::new(dx * inv, 0.0, dz * inv);
        }
    }

    // ── Supervised MSE update. ──────────────────────────────────
    let target_t = Tensor::<MyBackend, 2>::from_data(TensorData::new(target_buf.clone(), [n, OUT]), &pool.device);
    let mask_t   = Tensor::<MyBackend, 2>::from_data(TensorData::new(mask_buf.clone(),   [n, OUT]), &pool.device);
    let diff   = mu_cur - target_t;
    let sq     = diff.powf_scalar(2.0) * mask_t;
    let denom  = (active.len() as f32).max(1.0);
    let loss   = sq.sum().div_scalar(denom);

    let cm = pool.model.clone();
    let gp = GradientsParams::from_grads(loss.backward(), &pool.model);
    pool.model = pool.opt.step(LR, cm, gp);
}
