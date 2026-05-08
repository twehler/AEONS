// Intelligence Level 2 — Heterotroph REINFORCE pool, medium hidden width.
//
// Same architecture as `intelligence_level_1_hetero.rs` but with
// `HIDDEN = 32` (vs L1's 16). Inputs / outputs / training loop are
// identical — see the L1 hetero file for the full REINFORCE derivation
// and per-tick narration. The only thing the level number selects in
// the new RL world is the network's hidden width: more capacity at the
// cost of more GPU compute per tick.
//
// Inputs (17): energy_norm + 16-dim world model.
// Outputs (4): tanh-squashed (speed, dir.x, dir.y, dir.z).

use bevy::prelude::*;
use burn::module::{Initializer, Module, Param};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use burn_cuda::CudaDevice;
use std::collections::HashMap;

use crate::colony::{IntelligenceLevel, Organism, Heterotroph, MAXIMUM_ORGANISMS};
use crate::energy::get_max_energy;
use crate::rl_helpers::{BrainInheritance, BrainRestore, MyBackend, PoolSnapshot, gaussian_noise};
use crate::world_model::{WorldModelGrid, WORLD_MODEL_DIMS, fill_world_model};


const N:         usize = MAXIMUM_ORGANISMS;
const IN:        usize = 1 + WORLD_MODEL_DIMS;
const HIDDEN:    usize = 32;
const OUT:       usize = 4;
const MAX_SPEED: f32   = 20.0;
const LR:        f64   = 1e-3;

const SIGMA:          f32 = 0.2;
const BASELINE_ALPHA: f32 = 0.05;
const REWARD_CLAMP:   f32 = 1.0;


#[derive(Component, Clone, Copy)]
pub struct BrainSlotL2(pub u32);


#[derive(Module, Debug)]
pub struct PoolMlpL2<B: Backend> {
    w1: Param<Tensor<B, 3>>,
    b1: Param<Tensor<B, 2>>,
    w2: Param<Tensor<B, 3>>,
    b2: Param<Tensor<B, 2>>,
}

impl<B: Backend> PoolMlpL2<B> {
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

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = x.unsqueeze_dim::<3>(1).matmul(self.w1.val()).squeeze::<2>() + self.b1.val();
        let x = burn::tensor::activation::relu(x);
        let x = x.unsqueeze_dim::<3>(1).matmul(self.w2.val()).squeeze::<2>() + self.b2.val();
        burn::tensor::activation::tanh(x)
    }
}


trait BrainOptL2 {
    fn step(&mut self, lr: f64, m: PoolMlpL2<MyBackend>, g: GradientsParams) -> PoolMlpL2<MyBackend>;
}

impl<O: Optimizer<PoolMlpL2<MyBackend>, MyBackend>> BrainOptL2 for O {
    fn step(&mut self, lr: f64, m: PoolMlpL2<MyBackend>, g: GradientsParams) -> PoolMlpL2<MyBackend> {
        Optimizer::step(self, lr, m, g)
    }
}


pub struct BrainPoolL2 {
    model:        PoolMlpL2<MyBackend>,
    opt:          Box<dyn BrainOptL2>,
    free:         Vec<u32>,
    map:          HashMap<Entity, u32>,
    prev_state:   Vec<f32>,
    prev_action:  Vec<f32>,
    prev_energy:  Vec<f32>,
    has_prev:     Vec<bool>,
    baseline:     Vec<f32>,
    pub device:   CudaDevice,
}

impl BrainPoolL2 {
    fn new(device: CudaDevice) -> Self {
        Self {
            model:       PoolMlpL2::<MyBackend>::new(&device),
            opt:         Box::new(AdamConfig::new().init()),
            free:        (0..N as u32).rev().collect(),
            map:         HashMap::with_capacity(N),
            prev_state:  vec![0.0; N * IN],
            prev_action: vec![0.0; N * OUT],
            prev_energy: vec![0.0; N],
            has_prev:    vec![false; N],
            baseline:    vec![0.0; N],
            device,
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

impl FromWorld for BrainPoolL2 {
    fn from_world(_: &mut World) -> Self {
        let device = CudaDevice::default();
        warmup(&device);
        Self::new(device)
    }
}


fn warmup(device: &CudaDevice) {
    let m = PoolMlpL2::<MyBackend>::new(device);
    let mut o: Box<dyn BrainOptL2> = Box::new(AdamConfig::new().init());
    let i      = Tensor::<MyBackend, 2>::zeros([N, IN],  device);
    let action = Tensor::<MyBackend, 2>::zeros([N, OUT], device);
    let mask   = Tensor::<MyBackend, 2>::zeros([N, 1],   device);
    let adv    = Tensor::<MyBackend, 2>::zeros([N, 1],   device);
    let out    = m.forward(i);
    let diff   = action - out;
    let sum_sq = diff.powf_scalar(2.0).sum_dim(1);
    let scale  = 0.5_f32 / (SIGMA * SIGMA);
    let loss   = (sum_sq * adv * mask).sum().mul_scalar(scale).div_scalar(1.0_f32);
    let g      = GradientsParams::from_grads(loss.backward(), &m);
    let _      = o.step(LR, m, g);
}


pub fn assign_brains_l2(
    mut pool:     NonSendMut<BrainPoolL2>,
    new:          Query<(Entity, &Organism, Option<&BrainInheritance>, Option<&BrainRestore>), (
        With<Heterotroph>,
        Without<BrainSlotL2>,
    )>,
    mut commands: Commands,
) {
    for (e, organism, inheritance, restore) in new.iter() {
        if !matches!(organism.intelligence_level, IntelligenceLevel::Level2) { continue; }

        let Some(slot) = pool.free.pop() else { continue };
        let s = slot as usize;

        let mut restored = false;
        if let Some(r) = restore {
            match pool.restore_slot(s, r) {
                Ok(())   => { restored = true; }
                Err(err) => error!("L2 brain restore failed for {e:?}: {err} — using fresh slot"),
            }
        } else if let Some(BrainInheritance(parent)) = inheritance {
            if let Some(&parent_slot) = pool.map.get(parent) {
                pool.inherit_row(parent_slot as usize, s);
            }
        }

        if !restored {
            pool.has_prev[s]    = false;
            pool.baseline[s]    = 0.0;
            pool.prev_energy[s] = organism.energy;
        }

        pool.map.insert(e, slot);
        commands.entity(e).try_insert(BrainSlotL2(slot));
        commands.entity(e).try_remove::<BrainInheritance>();
        commands.entity(e).try_remove::<BrainRestore>();
    }
}

pub fn free_brains_l2(
    mut pool:    NonSendMut<BrainPoolL2>,
    mut removed: RemovedComponents<Heterotroph>,
) {
    for e in removed.read() {
        if let Some(slot) = pool.map.remove(&e) {
            let s = slot as usize;
            pool.has_prev[s] = false;
            pool.free.push(slot);
        }
    }
}


pub fn apply_intelligence_level_2(
    time:           Res<Time<Virtual>>,
    world_grid:     Res<WorldModelGrid>,
    mut pool:       NonSendMut<BrainPoolL2>,
    mut heteros:    Query<(Entity, &mut Organism, &Transform, &BrainSlotL2), With<Heterotroph>>,
    mut input_buf:  Local<Vec<f32>>,
    mut adv_buf:    Local<Vec<f32>>,
    mut mask_buf:   Local<Vec<f32>>,
    mut active_buf: Local<Vec<(Entity, u32, f32)>>,
) {
    if time.is_paused() { return; }

    input_buf.clear();   input_buf.resize(N * IN, 0.0);
    adv_buf.clear();     adv_buf.resize(N, 0.0);
    mask_buf.clear();    mask_buf.resize(N, 0.0);
    active_buf.clear();

    for (e, organism, transform, slot) in heteros.iter() {
        let s = slot.0 as usize;
        if s >= N { continue; }
        let pos = transform.translation;

        let max_e    = get_max_energy(&organism).max(1.0);
        let energy_n = (organism.energy / max_e).clamp(0.0, 1.0);

        let off = s * IN;
        input_buf[off] = energy_n;
        let wm_slice = &mut input_buf[off + 1 .. off + 1 + WORLD_MODEL_DIMS];
        fill_world_model(&world_grid, pos, wm_slice);

        active_buf.push((e, slot.0, organism.energy));
    }
    if active_buf.is_empty() { return; }

    let cur_t = Tensor::<MyBackend, 2>::from_data(
        TensorData::new(input_buf.clone(), [N, IN]),
        &pool.device,
    );
    let mu_cur  = pool.model.forward(cur_t);
    let mu_data = mu_cur.into_data().into_vec::<f32>().expect("forward output");

    let mut count = 0.0_f32;
    for &(_, slot, energy_now) in active_buf.iter() {
        let s = slot as usize;
        if pool.has_prev[s] {
            let raw = energy_now - pool.prev_energy[s];
            let r   = raw.clamp(-REWARD_CLAMP, REWARD_CLAMP);
            pool.baseline[s] = (1.0 - BASELINE_ALPHA) * pool.baseline[s] + BASELINE_ALPHA * r;
            adv_buf[s]   = r - pool.baseline[s];
            mask_buf[s]  = 1.0;
            count       += 1.0;
        }
    }

    if count > 0.0 {
        let prev_state_t = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(pool.prev_state.clone(), [N, IN]),
            &pool.device,
        );
        let prev_action_t = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(pool.prev_action.clone(), [N, OUT]),
            &pool.device,
        );
        let adv_t = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(adv_buf.clone(), [N, 1]),
            &pool.device,
        );
        let mask_t = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(mask_buf.clone(), [N, 1]),
            &pool.device,
        );

        let mu_prev = pool.model.forward(prev_state_t);
        let diff    = prev_action_t - mu_prev;
        let sum_sq  = diff.powf_scalar(2.0).sum_dim(1);
        let scale   = 0.5_f32 / (SIGMA * SIGMA);
        let loss    = (sum_sq * adv_t * mask_t).sum().mul_scalar(scale).div_scalar(count);

        let cm = pool.model.clone();
        let gp = GradientsParams::from_grads(loss.backward(), &pool.model);
        pool.model = pool.opt.step(LR, cm, gp);
    }

    let mut rng = rand::rng();
    for &(entity, slot, energy_now) in active_buf.iter() {
        let s   = slot as usize;
        let off = s * OUT;

        let mut action = [0.0_f32; OUT];
        for i in 0..OUT {
            action[i] = mu_data[off + i] + SIGMA * gaussian_noise(&mut rng);
        }

        let speed_a = action[0].clamp(-1.0, 1.0);
        let dir     = Vec3::new(action[1], 0.0, action[3]);
        let Ok((_, mut org, _, _)) = heteros.get_mut(entity) else { continue };
        if dir.length_squared() > 0.01 { org.movement_direction = dir.normalize(); }
        org.movement_speed = ((speed_a + 1.0) * 0.5).clamp(0.0, 1.0) * MAX_SPEED;

        let in_off = s * IN;
        for i in 0..IN  { pool.prev_state [in_off + i] = input_buf[in_off + i]; }
        for i in 0..OUT { pool.prev_action[off    + i] = action[i]; }
        pool.prev_energy[s] = energy_now;
        pool.has_prev[s]    = true;
    }
}
