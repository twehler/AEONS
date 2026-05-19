// Intelligence Level 1 — Photoautotroph REINFORCE pool.
//
// One private MLP per (mobile) photoautotroph. All `N` MLPs share the
// same `[N, IN, HIDDEN]` / `[N, HIDDEN, OUT]` tensor shape and run as
// a single batched matmul on the GPU; row `s` is slot `s`'s private
// brain. There is no weight sharing — slot `s`'s gradients only ever
// come from row `s`'s loss term.
//
// Inputs (5):
//   * `in_sunlight`              (0 / 1)
//   * normalised position xyz   (∈ [0, 1])
//   * normalised energy          (∈ [0, 1])
//
// Outputs (4): tanh-squashed (speed, dir.x, dir.y, dir.z). Speed is
// remapped to `[0, MAX_SPEED]`, dir.y is forced to 0 on apply (motion
// is xz-planar for photoautotrophs), dir is renormalised when its
// length is non-trivial. The output of the MLP is the *mean* `μ` of
// a Gaussian policy `π(a|s) = N(μ(s), σ²I)` — the actual action
// applied to the organism is `a = μ + σ · ε` where `ε ~ N(0, 1)` is
// drawn fresh each tick. That stochasticity is what gives REINFORCE
// a non-zero policy gradient.
//
// Training (REINFORCE with EMA baseline):
//   * Reward `r_s = energy_now[s] - energy_prev[s]`, clamped to
//     `[-REWARD_CLAMP, REWARD_CLAMP]` so single-frame predation /
//     reproduction spikes can't blow up the gradient. Reproduction
//     halves the parent's energy in `reproduction.rs`, which would
//     otherwise look like a catastrophic negative reward; clamping
//     bounds the per-step damage and the EMA baseline absorbs the
//     residual.
//   * Per-slot baseline `b_s` updated as
//     `b_s ← (1 − α) · b_s + α · r_s`. Advantage `A_s = r_s − b_s`.
//   * Loss `L = (0.5 / σ²) · sum_s [A_s · mask_s · ‖a_prev,s − μ(s_prev,s)‖²] / count_active`.
//     This is the standard fixed-σ Gaussian-policy REINFORCE: the
//     log-likelihood gradient reduces to `(a − μ) / σ²` which, dotted
//     with the advantage, yields the squared-error scaled by the
//     advantage. Positive advantage pulls μ toward the sampled action
//     (was good, do it more); negative advantage pushes μ away.
//   * One Adam step at `LR = 1e-3` per brain tick.
//
// Per-tick flow (for slots that are currently assigned to a live
// organism):
//   1. Read current state into `input_buf` (CPU side).
//   2. GPU forward pass over the full `[N, IN]` batch → `μ_cur`.
//   3. Sample `a_cur = μ_cur + σ · ε` per slot (CPU Box–Muller).
//   4. Apply `a_cur` to the organism: `movement_speed`,
//      `movement_direction`.
//   5. Build advantage / mask vectors from the *previously* stored
//      `(prev_state, prev_action, prev_energy)` against the current
//      energy — only slots that have a `has_prev[s] == true` row
//      contribute to the loss.
//   6. GPU forward + backward + optimiser step on the prev-state
//      input. Skipped entirely when no slot has a `prev` (saves the
//      whole `[N, IN]` autodiff round-trip).
//   7. Save `(current_state, sampled_action, current_energy)` into
//      the slot's `prev_*` rows for the next tick to train against.
//
// Slot assignment (`assign_brains_l1_photo`) — runs in PreUpdate:
//   * Pops a free slot index for any photoautotroph that doesn't
//     have a `BrainSlotL1Photo` yet AND isn't sessile (`Without<
//     BrainLevel0>`) AND isn't on a non-Level1 pool. Level matching
//     is implicit: photoautotrophs that are mobile are always
//     Level 1 by `IntelligenceLevel::for_initial_spawn` and
//     reproduction inherits, so the only L1 photoautotrophs in the
//     world end up here.
//   * Honours `BrainInheritance(parent)`: if the offspring carries
//     it AND the parent is in this same pool, copies the parent's
//     `[1, IN, HIDDEN]` / `[1, HIDDEN, OUT]` / `[1, HIDDEN]` /
//     `[1, OUT]` rows into the new slot before clearing
//     `has_prev / baseline` for that slot. Without inheritance the
//     slot keeps whatever weights its previous tenant trained — the
//     "instinct survives" behaviour.

use bevy::prelude::*;
use burn::module::{Initializer, Module, Param};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use burn_cuda::CudaDevice;
use std::collections::HashMap;

use crate::colony::{IntelligenceLevel, Organism, Photoautotroph};
use crate::energy::get_max_energy;
use crate::rl_helpers::{BrainInheritance, BrainRestore, MyBackend, PoolSnapshot, gaussian_noise};
use crate::simulation_settings::OrganismPoolSize;
use crate::world_geometry::MapSize;


// ── Architecture constants ──────────────────────────────────────────────────

const IN:        usize = 5;
const HIDDEN:    usize = 16;
const OUT:       usize = 4;
const MAX_SPEED: f32   = 40.0;
const LR:        f64   = 1e-3;

/// Vertical extent used to normalise `pos.y` into roughly `[0, 1]`. Picked
/// generously so the brain's position channel stays bounded across all
/// plausible terrain heights.
const Y_NORMALISATION: f32 = 50.0;

/// Standard deviation of the Gaussian exploration noise added to the MLP's
/// `μ` output to produce the actual action. Larger σ → more exploration,
/// noisier movement; smaller σ → faster convergence to a deterministic
/// policy but risk of premature commitment. 0.2 is a defensible default
/// — the tanh-squashed `μ` is in `[-1, 1]` so the noise injects roughly
/// 20% of the action range as exploration.
const SIGMA: f32 = 0.2;

/// EMA decay for the per-slot reward baseline used to compute advantage.
/// Smaller α → slower-moving baseline (stabler advantage but slow to
/// adapt to environment changes); larger α → faster but noisier.
const BASELINE_ALPHA: f32 = 0.05;

/// Hard clip on per-tick reward magnitude. Reproduction halves the
/// parent's energy in a single tick, so an unclamped raw delta could
/// be a multi-unit negative spike that dominates the gradient. The
/// clamp turns it into a bounded penalty that the baseline absorbs
/// over a few subsequent ticks instead.
const REWARD_CLAMP: f32 = 1.0;


// ── Slot component ──────────────────────────────────────────────────────────

/// Per-photoautotroph pointer into `BrainPoolL1Photo`'s row dimension.
#[derive(Component, Clone, Copy)]
pub struct BrainSlotL1Photo(pub u32);


// ── Per-organism MLP ────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct PoolMlpL1Photo<B: Backend> {
    w1: Param<Tensor<B, 3>>,    // [N, IN, HIDDEN]
    b1: Param<Tensor<B, 2>>,    // [N, HIDDEN]
    w2: Param<Tensor<B, 3>>,    // [N, HIDDEN, OUT]
    b2: Param<Tensor<B, 2>>,    // [N, OUT]
}

impl<B: Backend> PoolMlpL1Photo<B> {
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

    /// `x: [N, IN] → [N, OUT]`. Each row uses its own row of `w1` /
    /// `w2` / `b1` / `b2` via batched matmul, so the `N` MLPs run as
    /// a single GPU op. Output is tanh-squashed into `[-1, 1]` per
    /// dim — the *mean* `μ` of the Gaussian policy.
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = x.unsqueeze_dim::<3>(1).matmul(self.w1.val()).squeeze::<2>() + self.b1.val();
        let x = burn::tensor::activation::relu(x);
        let x = x.unsqueeze_dim::<3>(1).matmul(self.w2.val()).squeeze::<2>() + self.b2.val();
        burn::tensor::activation::tanh(x)
    }
}


// Erases the optimiser's generic state from `BrainPoolL1Photo`'s public
// type so it can live as a `Box<dyn …>` inside a non-send resource.
trait BrainOptL1Photo {
    fn step(
        &mut self,
        lr: f64,
        m:  PoolMlpL1Photo<MyBackend>,
        g:  GradientsParams,
    ) -> PoolMlpL1Photo<MyBackend>;
}

impl<O: Optimizer<PoolMlpL1Photo<MyBackend>, MyBackend>> BrainOptL1Photo for O {
    fn step(
        &mut self,
        lr: f64,
        m:  PoolMlpL1Photo<MyBackend>,
        g:  GradientsParams,
    ) -> PoolMlpL1Photo<MyBackend> {
        Optimizer::step(self, lr, m, g)
    }
}


// ── Pool resource ───────────────────────────────────────────────────────────

pub struct BrainPoolL1Photo {
    model:        PoolMlpL1Photo<MyBackend>,
    opt:          Box<dyn BrainOptL1Photo>,
    free:         Vec<u32>,
    map:          HashMap<Entity, u32>,
    /// Last-tick state for slot `s`, flat row-major `[N * IN]`. Used
    /// as the input to the REINFORCE training forward pass.
    prev_state:   Vec<f32>,
    /// Last-tick *sampled* action (NOT just the MLP mean) for slot
    /// `s`, flat row-major `[N * OUT]`. The squared error
    /// `(prev_action − μ(prev_state))²` is the kernel of the
    /// log-likelihood gradient.
    prev_action:  Vec<f32>,
    /// Last-tick energy reading for slot `s`, length `N`. Reward is
    /// `current_energy − prev_energy[s]`, clamped.
    prev_energy:  Vec<f32>,
    /// Whether `prev_*` rows for slot `s` are populated. Set `true`
    /// at the end of each apply tick, reset to `false` whenever a
    /// slot is (re)assigned to a different organism.
    has_prev:     Vec<bool>,
    /// EMA baseline for slot `s` over reward signal. Subtracted from
    /// the raw reward to give the advantage that scales the loss.
    baseline:     Vec<f32>,
    pub device:   CudaDevice,
    pub n:        usize,
}

impl BrainPoolL1Photo {
    fn new(device: CudaDevice, n: usize) -> Self {
        Self {
            model:       PoolMlpL1Photo::<MyBackend>::new(&device, n),
            opt:         Box::new(AdamConfig::new().init()),
            free:        (0..n as u32).rev().collect(),
            map:         HashMap::with_capacity(n),
            prev_state:  vec![0.0; n * IN],
            prev_action: vec![0.0; n * OUT],
            prev_energy: vec![0.0; n],
            has_prev:    vec![false; n],
            baseline:    vec![0.0; n],
            device,
            n,
        }
    }

    /// CPU snapshot of every weight tensor + REINFORCE state.
    /// Used by the save system to extract per-organism brain rows
    /// without re-syncing each row from the GPU. Four GPU→CPU
    /// transfers (one per Param's `Tensor::into_data`) — cheap on
    /// the rare save path, intolerable per-organism.
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

    /// Write `r`'s saved per-slot weights + REINFORCE state into
    /// row `slot` of the pool. Used during colony load when an
    /// entity arrives carrying a `BrainRestore` component. Returns
    /// an error string on size mismatch (e.g. save file produced
    /// against a different `HIDDEN`); the caller should log and
    /// fall back to fresh slot init.
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

    /// Deep-copy slot `parent`'s rows of every weight tensor into
    /// slot `child`'s rows. Preserves each `Param`'s `ParamId`
    /// (via `Param::clone().map(...)`) so the optimiser's Adam
    /// moments — keyed by ParamId — stay valid after the copy.
    /// Adam moments themselves are NOT copied; the new slot starts
    /// learning from zero momentum, which is fine (the moments at
    /// the parent's slot belong to a *different row* of the same
    /// `Param`, so they're irrelevant to the child's row).
    fn inherit_row(&mut self, parent: usize, child: usize) {
        // w1: [N, IN, HIDDEN] → row is [1, IN, HIDDEN].
        let p = self.model.w1.val().slice([parent..parent+1, 0..IN, 0..HIDDEN]);
        self.model.w1 = self.model.w1.clone().map(|t| {
            t.slice_assign([child..child+1, 0..IN, 0..HIDDEN], p)
        });
        // b1: [N, HIDDEN] → row is [1, HIDDEN].
        let p = self.model.b1.val().slice([parent..parent+1, 0..HIDDEN]);
        self.model.b1 = self.model.b1.clone().map(|t| {
            t.slice_assign([child..child+1, 0..HIDDEN], p)
        });
        // w2: [N, HIDDEN, OUT] → row is [1, HIDDEN, OUT].
        let p = self.model.w2.val().slice([parent..parent+1, 0..HIDDEN, 0..OUT]);
        self.model.w2 = self.model.w2.clone().map(|t| {
            t.slice_assign([child..child+1, 0..HIDDEN, 0..OUT], p)
        });
        // b2: [N, OUT] → row is [1, OUT].
        let p = self.model.b2.val().slice([parent..parent+1, 0..OUT]);
        self.model.b2 = self.model.b2.clone().map(|t| {
            t.slice_assign([child..child+1, 0..OUT], p)
        });
    }
}

impl FromWorld for BrainPoolL1Photo {
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


/// Forward + backward + Adam step at the maximum batch shape so
/// CubeCL caches every kernel the runtime tick will hit. Without
/// this the first real tick stalls for several seconds JIT-ing.
fn warmup(device: &CudaDevice, n: usize) {
    let m = PoolMlpL1Photo::<MyBackend>::new(device, n);
    let mut o: Box<dyn BrainOptL1Photo> = Box::new(AdamConfig::new().init());
    let i        = Tensor::<MyBackend, 2>::zeros([n, IN],  device);
    let action   = Tensor::<MyBackend, 2>::zeros([n, OUT], device);
    let mask     = Tensor::<MyBackend, 2>::zeros([n, 1],   device);
    let adv      = Tensor::<MyBackend, 2>::zeros([n, 1],   device);
    let out      = m.forward(i);
    let diff     = action - out;
    let sum_sq   = diff.powf_scalar(2.0).sum_dim(1);
    let scale    = 0.5_f32 / (SIGMA * SIGMA);
    let loss     = (sum_sq * adv * mask).sum().mul_scalar(scale).div_scalar(1.0_f32);
    let g        = GradientsParams::from_grads(loss.backward(), &m);
    let _        = o.step(LR, m, g);
}


// ── Slot allocation systems ─────────────────────────────────────────────────

pub fn assign_brains_l1_photo(
    mut pool:     NonSendMut<BrainPoolL1Photo>,
    new:          Query<
        (Entity, &Organism, Option<&BrainInheritance>, Option<&BrainRestore>),
        (
            With<Photoautotroph>,
            Without<BrainSlotL1Photo>,
            Without<crate::intelligence_level_0::BrainLevel0>,
        ),
    >,
    mut commands: Commands,
) {
    for (e, organism, inheritance, restore) in new.iter() {
        // Only enroll Level1 photoautotrophs in this pool. Higher
        // levels for photoautotrophs aren't currently produced by
        // `IntelligenceLevel::for_initial_spawn`, but inheritance
        // could in principle propagate one if a future spawn site
        // sets it — guard explicitly to keep the pool's rows pure.
        if !matches!(organism.intelligence_level, IntelligenceLevel::Level1) { continue; }

        let Some(slot) = pool.free.pop() else { continue };
        let s = slot as usize;

        // Three sources for the new slot's contents, in priority
        // order:
        //   1. `BrainRestore` — colony load. Restores weights AND
        //      REINFORCE prev_* state to the exact save-time
        //      values; skip the prev_* reset below.
        //   2. `BrainInheritance` — reproduction. Copies parent's
        //      weight rows; resets prev_* (parent's prev_* belong
        //      to the parent's tick, not the child's).
        //   3. Default — recycled slot. Keeps the previous
        //      tenant's weights (the "instinct survives recycling"
        //      behaviour); resets prev_*.
        let mut restored = false;
        if let Some(r) = restore {
            match pool.restore_slot(s, r) {
                Ok(())   => { restored = true; }
                Err(err) => error!("L1 photo brain restore failed for {e:?}: {err} — using fresh slot"),
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
        // try_insert / try_remove silently no-op on dead entities
        // (cf. issue #10166 / Bevy's state_scoped pattern). The
        // markers are removed unconditionally — they've done their
        // work and lingering past slot assignment would just
        // confuse later passes.
        commands.entity(e).try_insert(BrainSlotL1Photo(slot));
        commands.entity(e).try_remove::<BrainInheritance>();
        commands.entity(e).try_remove::<BrainRestore>();
    }
}

pub fn free_brains_l1_photo(
    mut pool:    NonSendMut<BrainPoolL1Photo>,
    mut removed: RemovedComponents<Photoautotroph>,
) {
    for e in removed.read() {
        if let Some(slot) = pool.map.remove(&e) {
            let s = slot as usize;
            pool.has_prev[s] = false;
            pool.free.push(slot);
        }
    }
}


// ── Apply / train tick ──────────────────────────────────────────────────────

pub fn apply_intelligence_level_1_photo(
    time:           Res<Time<Virtual>>,
    map_size:       Res<MapSize>,
    mut pool:       NonSendMut<BrainPoolL1Photo>,
    mut photos:     Query<(Entity, &mut Organism, &Transform, &BrainSlotL1Photo), With<Photoautotroph>>,
    mut input_buf:  Local<Vec<f32>>,
    mut adv_buf:    Local<Vec<f32>>,
    mut mask_buf:   Local<Vec<f32>>,
    mut active_buf: Local<Vec<(Entity, u32, f32)>>,
) {
    if time.is_paused() { return; }

    // ── Step 1: read current state into the flat scratch buffer.
    input_buf.clear();   input_buf.resize(pool.n * IN, 0.0);
    adv_buf.clear();     adv_buf.resize(pool.n, 0.0);
    mask_buf.clear();    mask_buf.resize(pool.n, 0.0);
    active_buf.clear();

    for (e, organism, transform, slot) in photos.iter() {
        let s = slot.0 as usize;
        if s >= pool.n { continue; }
        let pos = transform.translation;

        let in_sun_f = if organism.in_sunlight { 1.0 } else { 0.0 };
        let max_e    = get_max_energy(&organism).max(1.0);
        let energy_n = (organism.energy / max_e).clamp(0.0, 1.0);

        let off = s * IN;
        input_buf[off    ] = in_sun_f;
        input_buf[off + 1] = (pos.x / map_size.x).clamp(0.0, 1.0);
        input_buf[off + 2] = (pos.y / Y_NORMALISATION).clamp(0.0, 1.0);
        input_buf[off + 3] = (pos.z / map_size.z).clamp(0.0, 1.0);
        input_buf[off + 4] = energy_n;

        active_buf.push((e, slot.0, organism.energy));
    }
    if active_buf.is_empty() { return; }

    // ── Step 2: GPU forward over current state → μ_cur.
    let cur_t = Tensor::<MyBackend, 2>::from_data(
        TensorData::new(input_buf.clone(), [pool.n, IN]),
        &pool.device,
    );
    let mu_cur = pool.model.forward(cur_t);
    let mu_data = mu_cur.into_data().into_vec::<f32>().expect("forward output");

    // ── Step 3: build advantage / mask from prev_* against current energy.
    let mut count = 0.0_f32;
    for &(_, slot, energy_now) in active_buf.iter() {
        let s = slot as usize;
        if pool.has_prev[s] {
            let raw    = energy_now - pool.prev_energy[s];
            let r      = raw.clamp(-REWARD_CLAMP, REWARD_CLAMP);
            pool.baseline[s] = (1.0 - BASELINE_ALPHA) * pool.baseline[s] + BASELINE_ALPHA * r;
            adv_buf[s]   = r - pool.baseline[s];
            mask_buf[s]  = 1.0;
            count       += 1.0;
        }
    }

    // ── Step 4: REINFORCE backward + Adam step (only when any slot has
    // prev_* — the very first apply tick after pool init has nothing to
    // train against).
    if count > 0.0 {
        let prev_state_t = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(pool.prev_state.clone(), [pool.n, IN]),
            &pool.device,
        );
        let prev_action_t = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(pool.prev_action.clone(), [pool.n, OUT]),
            &pool.device,
        );
        let adv_t = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(adv_buf.clone(), [pool.n, 1]),
            &pool.device,
        );
        let mask_t = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(mask_buf.clone(), [pool.n, 1]),
            &pool.device,
        );

        let mu_prev = pool.model.forward(prev_state_t);
        let diff    = prev_action_t - mu_prev;
        let sum_sq  = diff.powf_scalar(2.0).sum_dim(1);  // [N, 1]
        let scale   = 0.5_f32 / (SIGMA * SIGMA);
        let loss    = (sum_sq * adv_t * mask_t).sum().mul_scalar(scale).div_scalar(count);

        let cm = pool.model.clone();
        let gp = GradientsParams::from_grads(loss.backward(), &pool.model);
        pool.model = pool.opt.step(LR, cm, gp);
        // Force burn-fusion to flush the queued backward + Adam ops so
        // the lazy stream doesn't accumulate across ticks (see L1-hetero
        // train_dqn for the full rationale).
        let _ = pool.model.b2.val().into_data();
    }

    // ── Step 5: sample action, apply to organism, and stash prev_* for
    // next tick. Done in a single pass so we can write to `pool.prev_*`
    // without holding a borrow through the GPU ops above.
    let mut rng = rand::rng();
    for &(entity, slot, energy_now) in active_buf.iter() {
        let s   = slot as usize;
        let off = s * OUT;

        // a = μ + σ · ε, ε ~ N(0, 1) per dim.
        let mut action = [0.0_f32; OUT];
        for i in 0..OUT {
            action[i] = mu_data[off + i] + SIGMA * gaussian_noise(&mut rng);
        }

        // Apply to organism. dir.y is forced 0 — the photoautotroph
        // moves only on the xz plane and a non-zero Y output would
        // perturb the body's yaw computation in apply_movement.
        let speed_a = action[0].clamp(-1.0, 1.0);
        let dir     = Vec3::new(action[1], 0.0, action[3]);
        let Ok((_, mut org, _, _)) = photos.get_mut(entity) else { continue };
        if dir.length_squared() > 0.01 { org.movement_direction = dir.normalize(); }
        org.movement_speed = ((speed_a + 1.0) * 0.5).clamp(0.0, 1.0) * MAX_SPEED;

        // Save next-tick training inputs.
        let in_off = s * IN;
        for i in 0..IN  { pool.prev_state [in_off + i] = input_buf[in_off + i]; }
        for i in 0..OUT { pool.prev_action[off    + i] = action[i]; }
        pool.prev_energy[s] = energy_now;
        pool.has_prev[s]    = true;
    }
}
