// Intelligence Level 1 — photoautotroph brain pool.
//
// Architecture mirrors `intelligence_level_3.rs` (heterotrophs): every
// photoautotroph owns a private MLP, all `N` MLPs share the same tensor
// shape and execute as a single batched matmul on the GPU. There is no
// weight sharing — slot `k`'s weights live in row `k` of `w1`/`w2`/`b1`/
// `b2` and only ever receive gradients from row `k`'s loss.
//
// Inputs (5):
//   - `in_sunlight`              (0 / 1)
//   - normalised position xyz   (∈ [0, 1])
//   - normalised energy          (∈ [0, 1])
//
// Hidden: 4 (per spec). Outputs: tanh-squashed (speed, dir.x, dir.y, dir.z).
//
// Oracle target ("move out of shadows, stay in sunlight, but each slot
// has its own preferred heading"):
//   in shadow → speed = +1, direction = `sun_xz` rotated by `slot_yaw[s]`
//   in sun    → speed = -1, direction = same per-slot rotated `sun_xz`
//
// Per-slot diversity is the load-bearing detail. A previous version used a
// SHARED constant `sun_xz` direction target across every slot — and because
// masked MSE has a single global minimum at that target, every per-slot
// MLP converged to indistinguishable outputs in ~10–30 seconds, producing
// a "river of algae" streaming along the sun vector. (An even earlier
// version used a position-encoded `(cos θ, 0, sin θ)` flow field for
// diversity, but its multi-period sinusoid is unlearnable by a 4-hidden-
// unit ReLU MLP, so photoautotrophs ended up wandering on essentially
// fixed random per-slot directions and never escaping shadows.)
//
// The current scheme: `BrainPoolL1::slot_yaw[s]` holds a per-slot angular
// bias drawn uniformly from `[-π/2, +π/2]` at pool init, fixed for the
// slot's lifetime. The training target for slot `s` is `sun_xz` rotated
// around the Y axis by `slot_yaw[s]`. Each slot still trains against a
// constant (so a 4-unit MLP fits trivially in a few hundred steps) — but
// each slot's constant is *different*, so the converged output is unique
// per slot. The yaw range is half-circle so every slot's target keeps a
// non-negative dot product with `sun_xz` — no "stragglers that head
// permanently away from the sun and pile up at the world boundary".
// Slot weights AND `slot_yaw[s]` both persist across reuse, identical to
// the "instinct survives" semantics of Level 3.

use bevy::prelude::*;
use burn::backend::Autodiff;
use burn::module::{Initializer, Module, Param};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use rand::prelude::*;
use burn_cuda::{Cuda, CudaDevice};
use std::collections::HashMap;

use crate::colony::{Organism, Photoautotroph, MAXIMUM_ORGANISMS};
use crate::energy::get_max_energy;
use crate::photosynthesis::SUN_DIRECTION;
use crate::world_geometry::{MAP_MAX_X, MAP_MAX_Z};

type MyBackend = Autodiff<Cuda>;

const N:         usize = MAXIMUM_ORGANISMS;
const IN:        usize = 5;
const HIDDEN:    usize = 4;
const OUT:       usize = 4;
const MAX_SPEED: f32   = 20.0;
const LR:        f64   = 1e-3;

/// Vertical extent used to normalise `pos.y` into roughly `[0, 1]`. Picked
/// generously so the brain's position channel stays bounded across all
/// plausible terrain heights.
const Y_NORMALISATION: f32 = 50.0;


// ── Brain slot component ────────────────────────────────────────────────────

/// Component pointing each photoautotroph at its private row in `BrainPoolL1`.
#[derive(Component, Clone, Copy)]
pub struct BrainSlotL1(pub u32);


// ── Per-organism MLP ────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct PoolMlpL1<B: Backend> {
    w1: Param<Tensor<B, 3>>,    // [N, IN, HIDDEN]
    b1: Param<Tensor<B, 2>>,    // [N, HIDDEN]
    w2: Param<Tensor<B, 3>>,    // [N, HIDDEN, OUT]
    b2: Param<Tensor<B, 2>>,    // [N, OUT]
}

impl<B: Backend> PoolMlpL1<B> {
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
    // weight matrices via batched matmul, so the N MLPs run in a single GPU op.
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = x.unsqueeze_dim::<3>(1).matmul(self.w1.val()).squeeze::<2>() + self.b1.val();
        let x = burn::tensor::activation::relu(x);
        let x = x.unsqueeze_dim::<3>(1).matmul(self.w2.val()).squeeze::<2>() + self.b2.val();
        burn::tensor::activation::tanh(x)
    }
}


// Erases the optimiser's generic state from `BrainPoolL1`'s public type so
// it can be stored as a `Box<dyn ...>` inside a non-send resource.
trait BrainOptL1 {
    fn step(&mut self, lr: f64, m: PoolMlpL1<MyBackend>, g: GradientsParams) -> PoolMlpL1<MyBackend>;
}

impl<O: Optimizer<PoolMlpL1<MyBackend>, MyBackend>> BrainOptL1 for O {
    fn step(&mut self, lr: f64, m: PoolMlpL1<MyBackend>, g: GradientsParams) -> PoolMlpL1<MyBackend> {
        Optimizer::step(self, lr, m, g)
    }
}


// ── Brain pool resource ─────────────────────────────────────────────────────

pub struct BrainPoolL1 {
    model:      PoolMlpL1<MyBackend>,
    opt:        Box<dyn BrainOptL1>,
    free:       Vec<u32>,
    map:        HashMap<Entity, u32>,
    /// Per-slot angular bias around the Y axis applied to the `sun_xz`
    /// training target. Drawn uniformly from `[-π/2, +π/2]` at pool init,
    /// fixed per slot for life. Index by `BrainSlotL1.0`. This is what
    /// breaks the "all photoautotrophs converge to the same output"
    /// pathology — see the file header for the full rationale.
    slot_yaw:   Vec<f32>,
    pub device: CudaDevice,
}

impl BrainPoolL1 {
    fn new(device: CudaDevice) -> Self {
        // Per-slot yaw bias. Half-circle range keeps every slot's target
        // sun-aligned in expectation (dot with `sun_xz` ≥ 0) so the
        // population still has a directional bias toward sunlit terrain
        // — just no longer a single coherent stream.
        let mut rng = rand::rng();
        let half = std::f32::consts::FRAC_PI_2;
        let slot_yaw: Vec<f32> = (0..N).map(|_| rng.random_range(-half..half)).collect();

        Self {
            model: PoolMlpL1::<MyBackend>::new(&device),
            opt:   Box::new(AdamConfig::new().init()),
            free:  (0..N as u32).rev().collect(),
            map:   HashMap::with_capacity(N),
            slot_yaw,
            device,
        }
    }
}

impl FromWorld for BrainPoolL1 {
    fn from_world(_: &mut World) -> Self {
        let device = CudaDevice::default();
        warmup(&device);
        Self::new(device)
    }
}


// One full forward → MSE → backward → step at the maximum batch shape so
// CubeCL caches every kernel the runtime tick will hit. Without this the
// first real tick stalls for several seconds while kernels JIT.
fn warmup(device: &CudaDevice) {
    let m = PoolMlpL1::<MyBackend>::new(device);
    let mut o: Box<dyn BrainOptL1> = Box::new(AdamConfig::new().init());
    let i      = Tensor::<MyBackend, 2>::zeros([N, IN],  device);
    let target = Tensor::<MyBackend, 2>::zeros([N, OUT], device);
    let mask   = Tensor::<MyBackend, 2>::zeros([N, 1],   device);
    let out    = m.forward(i);
    let sq     = (out - target).powf_scalar(2.0).sum_dim(1);
    let loss   = (sq * mask).sum().div_scalar(1.0_f32);
    let g      = GradientsParams::from_grads(loss.backward(), &m);
    let _      = o.step(LR, m, g);
}


// ── Slot allocation systems ─────────────────────────────────────────────────

pub fn assign_brains_l1(
    mut pool:     NonSendMut<BrainPoolL1>,
    new:          Query<Entity, (With<Photoautotroph>, Without<BrainSlotL1>)>,
    mut commands: Commands,
) {
    for e in new.iter() {
        let Some(slot) = pool.free.pop() else { continue };
        // Slot weights are intentionally NOT reset on reuse — a recycled
        // slot inherits the previous occupant's trained MLP as "instinct",
        // identical to the Level 3 pool behaviour.
        pool.map.insert(e, slot);
        // try_insert (Bevy 0.18 EntityCommands) silently no-ops if `e` was
        // despawned between this query iteration and the deferred command
        // flush. This is the canonical Bevy 0.18 idiom for "insert on a
        // possibly-dead entity" (cf. issue #10166 and the `state_scoped`
        // example) and prevents the panic that the panic-on-error default
        // handler raises for plain `insert`. Race scenario:
        // `predation_system` (Update) despawns `e` after this query
        // iterator captured it; at the next apply_deferred the entity is
        // gone. With try_insert the missed insert leaves no `BrainSlotL1`
        // on a dead entity (no harm) and `pool.map`'s stale entry is
        // reclaimed by `free_brains_l1` next tick via RemovedComponents.
        commands.entity(e).try_insert(BrainSlotL1(slot));
    }
}

pub fn free_brains_l1(
    mut pool:    NonSendMut<BrainPoolL1>,
    mut removed: RemovedComponents<Photoautotroph>,
) {
    for e in removed.read() {
        if let Some(slot) = pool.map.remove(&e) { pool.free.push(slot); }
    }
}


// ── Main brain tick ─────────────────────────────────────────────────────────

pub fn apply_intelligence_level_1(
    time:       Res<Time<Virtual>>,
    mut pool:   NonSendMut<BrainPoolL1>,
    mut photos: Query<(Entity, &mut Organism, &Transform, &BrainSlotL1), With<Photoautotroph>>,
) {
    if time.is_paused() { return; }

    // Flat row-major scratch buffers for the fixed [N, _] tensors. Every
    // slot is filled, even unassigned ones (their `mask` row is 0 so they
    // contribute zero gradient) — this keeps the GPU kernel cache hot by
    // never varying the tensor shape.
    let mut input  = vec![0.0_f32; N * IN];
    let mut target = vec![0.0_f32; N * OUT];
    let mut mask   = vec![0.0_f32; N];

    let mut active: Vec<(Entity, u32)> = Vec::with_capacity(N);
    let mut count  = 0.0_f32;

    // Toward-sun heading projected onto the xz plane — the *base* direction
    // we want shadowed photoautotrophs to head in. Each slot's actual
    // training target is this vector rotated around the Y axis by that
    // slot's `slot_yaw` (see the file header for why this matters).
    let sun_xz = {
        let v = Vec3::new(SUN_DIRECTION.x, 0.0, SUN_DIRECTION.z);
        v.normalize_or_zero()
    };

    for (entity, organism, transform, slot) in photos.iter() {
        let s = slot.0 as usize;
        if s >= N { continue; }
        let pos = transform.translation;

        let in_sun_f = if organism.in_sunlight { 1.0 } else { 0.0 };
        let max_e    = get_max_energy(organism).max(1.0);
        let energy   = (organism.energy / max_e).clamp(0.0, 1.0);

        let i_off = s * IN;
        input[i_off    ] = in_sun_f;
        input[i_off + 1] = (pos.x / MAP_MAX_X).clamp(0.0, 1.0);
        input[i_off + 2] = (pos.y / Y_NORMALISATION).clamp(0.0, 1.0);
        input[i_off + 3] = (pos.z / MAP_MAX_Z).clamp(0.0, 1.0);
        input[i_off + 4] = energy;

        // Speed target in tanh-space: +1 saturates to 1.0 (apply remaps to
        // MAX_SPEED), -1 saturates to -1.0 (apply remaps to 0).
        let speed_target = if organism.in_sunlight { -1.0 } else { 1.0 };

        // Per-slot rotated sun-xz. Each slot has its own `slot_yaw[s]` so
        // the converged target is unique per slot — that's what breaks
        // the global-minimum collapse and produces visibly diverse
        // movement instead of a single-direction river.
        let yaw = pool.slot_yaw[s];
        let (sin_y, cos_y) = yaw.sin_cos();
        let dir_x = sun_xz.x * cos_y - sun_xz.z * sin_y;
        let dir_z = sun_xz.x * sin_y + sun_xz.z * cos_y;

        let t_off = s * OUT;
        target[t_off    ] = speed_target;
        target[t_off + 1] = dir_x;
        target[t_off + 2] = 0.0; // Y stays 0 — photoautotroph motion is xz-planar.
        target[t_off + 3] = dir_z;
        mask[s] = 1.0;
        count  += 1.0;
        active.push((entity, slot.0));
    }

    if count == 0.0 { return; }

    let i_t = Tensor::<MyBackend, 2>::from_data(TensorData::new(input,  [N, IN]),  &pool.device);
    let t_t = Tensor::<MyBackend, 2>::from_data(TensorData::new(target, [N, OUT]), &pool.device);
    let m_t = Tensor::<MyBackend, 2>::from_data(TensorData::new(mask,   [N, 1]),   &pool.device);

    let out  = pool.model.forward(i_t);
    // Per-row squared error → mask out unassigned slots → mean over active count.
    let sq   = (out.clone() - t_t).powf_scalar(2.0).sum_dim(1);
    let loss = (sq * m_t).sum().div_scalar(count);

    let cm = pool.model.clone();
    let gp = GradientsParams::from_grads(loss.backward(), &pool.model);
    pool.model = pool.opt.step(LR, cm, gp);

    let out_data = out.into_data().into_vec::<f32>().expect("brain output");

    for (entity, slot) in active {
        let off   = slot as usize * OUT;
        let speed = out_data[off];
        let dir   = Vec3::new(out_data[off + 1], out_data[off + 2], out_data[off + 3]);

        let Ok((_, mut org, _, _)) = photos.get_mut(entity) else { continue };
        if dir.length_squared() > 0.01 { org.movement_direction = dir.normalize(); }
        org.movement_speed = ((speed + 1.0) * 0.5).clamp(0.0, 1.0) * MAX_SPEED;
    }
}
