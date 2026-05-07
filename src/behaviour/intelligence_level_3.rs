use bevy::prelude::*;
use burn::backend::Autodiff;
use burn::module::{Initializer, Module, Param};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use burn_cuda::{Cuda, CudaDevice};
use std::collections::HashMap;

use crate::colony::{Organism, Heterotroph, Photoautotroph, MAXIMUM_ORGANISMS};

pub type MyBackend = Autodiff<Cuda>;

const N:         usize = MAXIMUM_ORGANISMS as usize;
const IN:        usize = 3;
const HIDDEN:    usize = 16;
const OUT:       usize = 4;
const RADIUS:    f32   = 30.0;
const MAX_SPEED: f32   = 20.0;
const LR:        f64   = 1e-3;


#[derive(Component, Clone, Copy)]
pub struct BrainSlot(pub u32);


// Per-organism private weights. The leading `N` dimension means every slot
// owns its own MLP — there is no weight sharing across organisms. The
// batched matmul (`unsqueeze → matmul → squeeze`) evaluates all N MLPs in
// a single GPU op, so isolation costs no throughput.
#[derive(Module, Debug)]
pub struct PoolMlp<B: Backend> {
    w1: Param<Tensor<B, 3>>,    // [N, IN, HIDDEN]
    b1: Param<Tensor<B, 2>>,    // [N, HIDDEN]
    w2: Param<Tensor<B, 3>>,    // [N, HIDDEN, OUT]
    b2: Param<Tensor<B, 2>>,    // [N, OUT]
}

impl<B: Backend> PoolMlp<B> {
    fn new(device: &B::Device) -> Self {
        let w = Initializer::Uniform { min: -0.5, max: 0.5 };
        let z = Initializer::Zeros;
        Self {
            w1: w.init([N, IN, HIDDEN], device),
            b1: z.init([N, HIDDEN], device),
            w2: w.init([N, HIDDEN, OUT], device),
            b2: z.init([N, OUT], device),
        }
    }

    // x: [N, IN] → [N, OUT]. Each row uses its own [IN, HIDDEN] / [HIDDEN, OUT]
    // weight matrices via batched matmul.
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = x.unsqueeze_dim::<3>(1).matmul(self.w1.val()).squeeze::<2>() + self.b1.val();
        let x = burn::tensor::activation::relu(x);
        let x = x.unsqueeze_dim::<3>(1).matmul(self.w2.val()).squeeze::<2>() + self.b2.val();
        burn::tensor::activation::tanh(x)
    }
}


pub trait BrainOpt {
    fn step(&mut self, lr: f64, m: PoolMlp<MyBackend>, g: GradientsParams) -> PoolMlp<MyBackend>;
}

impl<O: Optimizer<PoolMlp<MyBackend>, MyBackend>> BrainOpt for O {
    fn step(&mut self, lr: f64, m: PoolMlp<MyBackend>, g: GradientsParams) -> PoolMlp<MyBackend> {
        Optimizer::step(self, lr, m, g)
    }
}


pub struct BrainPool {
    model:       PoolMlp<MyBackend>,
    opt:         Box<dyn BrainOpt>,
    free:        Vec<u32>,
    map:         HashMap<Entity, u32>,
    pub device:  CudaDevice,
}

impl BrainPool {
    fn new(device: CudaDevice) -> Self {
        Self {
            model: PoolMlp::<MyBackend>::new(&device),
            opt:   Box::new(AdamConfig::new().init()),
            free:  (0..N as u32).rev().collect(),
            map:   HashMap::with_capacity(N),
            device,
        }
    }
}

impl FromWorld for BrainPool {
    fn from_world(_: &mut World) -> Self {
        let device = CudaDevice::default();
        warmup(&device);
        Self::new(device)
    }
}


// One full forward → MSE → backward → step at the maximum batch shape so
// CUDA caches the kernels. Subsequent ticks reuse the same shape and avoid
// recompilation.
fn warmup(device: &CudaDevice) {
    let m = PoolMlp::<MyBackend>::new(device);
    let mut o: Box<dyn BrainOpt> = Box::new(AdamConfig::new().init());
    let i      = Tensor::<MyBackend, 2>::zeros([N, IN],  device);
    let target = Tensor::<MyBackend, 2>::zeros([N, OUT], device);
    let mask   = Tensor::<MyBackend, 2>::zeros([N, 1],   device);
    let out    = m.forward(i);
    let sq     = (out - target).powf_scalar(2.0).sum_dim(1);   // [N, 1]
    let loss   = (sq * mask).sum().div_scalar(1.0_f32);
    let g      = GradientsParams::from_grads(loss.backward(), &m);
    let _      = o.step(LR, m, g);
}


pub fn assign_brains(
    mut pool: NonSendMut<BrainPool>,
    new: Query<Entity, (With<Heterotroph>, Without<BrainSlot>)>,
    mut commands: Commands,
) {
    for e in new.iter() {
        let Some(slot) = pool.free.pop() else { continue };
        // NOTE: The slot's weights are intentionally NOT reset on reuse.
        // A new organism inheriting the previous occupant's trained MLP is
        // analogous to inherited instinct — and a freshly-spawned predator
        // converges to correct pursuit within a few ticks of MSE training
        // anyway, so any residual bias is quickly overwritten.
        pool.map.insert(e, slot);
        // try_insert silently no-ops on dead entities — see the matching
        // comment in `intelligence_level_1.rs::assign_brains_l1`. Same race
        // applies here: a heterotroph can starve in Update between this
        // PreUpdate query and the apply_deferred flush. RemovedComponents
        // self-heals the slot bookkeeping in the next tick.
        commands.entity(e).try_insert(BrainSlot(slot));
    }
}

pub fn free_brains(
    mut pool: NonSendMut<BrainPool>,
    mut removed: RemovedComponents<Heterotroph>,
) {
    for e in removed.read() {
        if let Some(slot) = pool.map.remove(&e) { pool.free.push(slot); }
    }
}


// MSE supervision against an analytically-computed ideal action.
//
// For every active heterotroph with prey within RADIUS, the oracle is:
//   ideal_speed = +1.0  (tanh-space; remaps to MAX_SPEED on application)
//   ideal_dir   = unit vector from predator to nearest prey
//
// The batch is fixed at shape [N, IN]/[N, OUT] so CUDA kernel caches stay
// hot. Inactive slots (no prey in range, or unassigned) get zero input,
// zero target, and zero mask — their per-row squared error is multiplied
// by 0 before reduction, so they contribute exactly zero gradient.
//
// Predators with no prey in range are not just zero-loss but also skipped
// at apply time: their movement_direction / movement_speed are left
// untouched, so they keep cruising on their last command instead of being
// jerked toward (0, 0, 0).
pub fn apply_intelligence_level_3(
    time: Res<Time<Virtual>>,
    mut pool: NonSendMut<BrainPool>,
    mut heteros: Query<(Entity, &mut Organism, &Transform, &BrainSlot), With<Heterotroph>>,
    photos: Query<&Transform, With<Photoautotroph>>,
) {
    if time.is_paused() { return; }

    // Flat row-major scratch buffers for the fixed [N, _] tensors.
    let mut input  = vec![0.0_f32; N * IN];
    let mut target = vec![0.0_f32; N * OUT];
    let mut mask   = vec![0.0_f32; N];

    // (entity, slot) for every predator that gets its action applied this tick.
    let mut active: Vec<(Entity, u32)> = Vec::with_capacity(N);
    let mut count  = 0.0_f32;

    for (entity, _organism, transform, slot) in heteros.iter() {
        let s = slot.0 as usize;
        if s >= N { continue; }
        let pos = transform.translation;

        // Find nearest prey within RADIUS.
        let mut best_d = f32::INFINITY;
        let mut best_p = pos;
        let mut found  = false;
        for pt in photos.iter() {
            let d = pos.distance(pt.translation);
            if d <= RADIUS && d < best_d {
                best_d = d;
                best_p = pt.translation;
                found  = true;
            }
        }

        // Always fill the input row (zeros if no prey, normalized rel pos otherwise).
        // Whether this row's gradient counts is controlled by `mask`, not by
        // skipping the input — keeping the tensor shape fixed at [N, IN] is
        // what makes the GPU kernel cache stay hot.
        let rel = if found { (best_p - pos) / RADIUS } else { Vec3::ZERO };
        let i_off = s * IN;
        input[i_off    ] = rel.x;
        input[i_off + 1] = rel.y;
        input[i_off + 2] = rel.z;

        if found {
            // Oracle: head straight at the prey at full speed.
            let ideal_dir = (best_p - pos).normalize_or_zero();
            let t_off = s * OUT;
            target[t_off    ] = 1.0;            // ideal speed (tanh-space)
            target[t_off + 1] = ideal_dir.x;
            target[t_off + 2] = ideal_dir.y;
            target[t_off + 3] = ideal_dir.z;
            mask[s] = 1.0;
            count  += 1.0;
            active.push((entity, slot.0));
        }
        // If no prey found: mask stays 0, target stays 0, and we do NOT
        // push to `active`, so this organism's movement is not overwritten
        // and it keeps cruising on its previous command.
    }

    // No predators have prey in range → nothing to train, nothing to apply.
    // We still skip the GPU work entirely (a [N, IN] forward pass on all
    // zeros would just waste cycles).
    if count == 0.0 { return; }

    // Build tensors and train.
    let i_t = Tensor::<MyBackend, 2>::from_data(TensorData::new(input,  [N, IN]),  &pool.device);
    let t_t = Tensor::<MyBackend, 2>::from_data(TensorData::new(target, [N, OUT]), &pool.device);
    let m_t = Tensor::<MyBackend, 2>::from_data(TensorData::new(mask,   [N, 1]),   &pool.device);

    let out = pool.model.forward(i_t);
    // Per-row squared error → mask out inactive slots → mean over active count.
    let sq   = (out.clone() - t_t).powf_scalar(2.0).sum_dim(1);   // [N, 1]
    let loss = (sq * m_t).sum().div_scalar(count);

    let cm = pool.model.clone();
    let gp = GradientsParams::from_grads(loss.backward(), &pool.model);
    pool.model = pool.opt.step(LR, cm, gp);

    // Read outputs back and apply only to active predators.
    let out_data = out.into_data().into_vec::<f32>().expect("brain output");

    for (entity, slot) in active {
        let off = slot as usize * OUT;
        let speed = out_data[off];
        let dir   = Vec3::new(out_data[off + 1], out_data[off + 2], out_data[off + 3]);

        let Ok((_, mut org, _, _)) = heteros.get_mut(entity) else { continue };
        if dir.length_squared() > 0.01 { org.movement_direction = dir.normalize(); }
        org.movement_speed = ((speed + 1.0) * 0.5).clamp(0.0, 1.0) * MAX_SPEED;
    }
}
