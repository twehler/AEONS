// SHARED-POLICY PPO engine for the SWIMMING brain pool
// (intelligence_level_1_swimming).
//
// EXPERIMENT (vs. the per-organism `src_individual_learning` variant): ALL
// swimmers share ONE actor + ONE critic. Every organism's experience trains
// the SAME weights, so the population learns as a single agent with ~N× the
// data — far more sample-efficient than independent per-organism brains. The
// biological reading: individuals of a species behave roughly the same, so one
// shared policy is a fair model; individuality is preserved only as momentary
// variation via each organism's OWN exploration noise (see `noise_state`), not
// as distinct learned weights.
//
// Swimming organisms are dynamic per-part bodies with neutral buoyancy whose
// limbs hang on BALL (spherical) joints (`rapier_setup::spherical_data`) — the
// policy commands THREE target angles per joint and locomotion must emerge from
// the anisotropic blade-element fluid drag (`apply_fluid_drag`): a limb stroked
// broadside pushes water, the reaction propels/turns the body.
//
// A SMALL net (`IN=116 → HIDDEN=64 → OUT=24`) over a 3D observation built around
// two ORACLES:
//   * TARGET ORACLE — body-local 3D unit direction toward the nearest
//     phototroph (+ normalised distance). WHERE the target is.
//   * ROTATION ORACLE — body-local rotation vector (axis · angle / π) turning
//     the body's +Z onto the target direction. HOW to rotate to face it.
//
// Reward (see `simulation_settings`): a BIG sparse eat reward (`K_SWIM_EAT` per
// Δpredations — eating is by proximity, the swimmer passes through prey), a
// SMALL dense reward per world-unit of 3D distance closed (`K_SWIM_PROGRESS`),
// a rotation-toward-target objective (`K_SWIM_ALIGN_GAIN` rectified
// facing-alignment gain + `K_SWIM_ALIGN` small absolute facing bonus), and a
// mild anti-corkscrew spin penalty.
//
// EXPLORATION (unchanged from the individual variant — still per organism so
// the shared policy sees diverse trajectories): larger σ (`SWIM_LOG_STD_INIT`),
// temporally-correlated AR(1)/OU noise (`SWIM_NOISE_CORR`) so exploratory
// strokes are HELD ~1 s and produce net thrust, and a vigorous oscillatory
// warm-start (`SWIM_WARMSTART_AMP`) giving the one shared net a rhythmic gait
// prior.
//
// Swim brains are NOT persisted to `.colony`; loaded/imported swimmers re-init
// fresh and any stale `BrainRestoreLimb` payload is dropped at enrol.

use bevy::prelude::*;
use burn::module::{Initializer, Module, Param};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Tensor, TensorData, backend::Backend, activation::{relu, tanh}};
use burn_cuda::CudaDevice;
use std::collections::{HashMap, VecDeque};

use crate::colony::Organism;
use crate::energy::MAX_ENERGY_PER_CELL;
use crate::rapier_setup::SwimJointTargets;
use crate::rl_helpers::{MyBackend, gaussian_noise};
// Reuse the limb pool's training-step row type so a future dataset-export hook
// can serve both pools with one CSV schema.
pub use crate::limb_ppo::LimbTrainingStep as SwimTrainingStep;


// ── Architecture constants ────────────────────────────────────────────────────

pub use crate::simulation_settings::{
    ROLLOUT_LEN, PPO_EPOCHS, CLIP_EPS, GAMMA, LAMBDA, LR,
    VALUE_LOSS_COEF, ENTROPY_COEF, GAIT_FREQUENCY_HZ,
    MAX_LIMB_JOINTS,
};
pub use crate::simulation_settings::{
    K_SWIM_EAT, K_SWIM_PROGRESS, K_SWIM_ALIGN_GAIN, K_SWIM_ALIGN, K_SWIM_SPIN,
};
// Swimmer-specific exploration knobs (NOT the terrestrial LOG_STD_INIT —
// swimmers explore limb rotation harder; see simulation_settings for why).
pub use crate::simulation_settings::{
    SWIM_LOG_STD_INIT, SWIM_NOISE_CORR, SWIM_WARMSTART_AMP,
};

/// Observation dimension. Full-3D layout (`MAX_LIMB_JOINTS = 8`):
///   * `obs[0]`           — energy_norm                                  (1)
///   * `obs[1..4]`        — TARGET ORACLE: body-local unit direction
///                          toward the nearest phototroph (zero-length
///                          when none in range — distinguishable)        (3)
///   * `obs[4]`           — target distance / SWIM_SENSORY_RADIUS
///                          (1.0 when no target)                         (1)
///   * `obs[5..8]`        — ROTATION ORACLE: body-local rotation vector
///                          (axis · angle / π) turning body +Z onto the
///                          target direction                             (3)
///   * `obs[8..56]`       — per-joint angles: 8 joints × (sin,cos of
///                          Euler-XYZ relative to base) = 8×6           (48)
///   * `obs[56..80]`      — per-joint angular velocity: 8 × 3           (24)
///   * `obs[80..83]`      — base linear velocity (body-local)            (3)
///   * `obs[83..86]`      — base angular velocity (body-local)           (3)
///   * `obs[86..90]`      — base orientation quaternion (x, y, z, w)     (4)
///   * `obs[90..114]`     — prev_action recurrence (OUT)                (24)
///   * `obs[114..116]`    — phase clock (sin, cos)                       (2)
pub const IN:     usize = 116;
/// Hidden layer width — one ReLU layer. 64 clears the capacity bottleneck
/// (116→H→24 plus the 2 hidden units the oscillatory warm-start reserves as
/// sin/cos phase carriers). With a SHARED policy the per-organism
/// sample-efficiency argument for keeping it small is weaker (one net trains on
/// all data), but 64 is still ample for this oracle-fed motor-control task.
pub const HIDDEN: usize = 64;
/// Action dimension: THREE target angles (joint-frame X/Y/Z, each ∈ [-1,1] ×
/// `LIMB_SWING_LIMIT`) per ball joint, for up to `MAX_LIMB_JOINTS` joints.
/// Output `[3k .. 3k+3]` drives body-part `k+1`'s ball joint
/// (`rapier_setup::drive_swim_motors`, parts beyond 8 wrap modulo).
pub const OUT:    usize = MAX_LIMB_JOINTS * 3;


// ── Networks (SHARED: one net, no per-organism batch dim) ──────────────────────

/// Actor MLP — a SINGLE shared `IN → HIDDEN → OUT` net returning `μ` for the
/// diagonal-Gaussian policy (tanh-bounded). Weights are 2-D (`[IN, HIDDEN]`,
/// `[HIDDEN, OUT]`) and biases / `log_std` are `[1, ·]` so a forward over a
/// batch of `B` observations is a plain `[B, IN] · [IN, HIDDEN]` matmul with
/// broadcast bias. The same `forward` serves both inference (one row per active
/// organism) and the PPO update (all pooled transitions as one batch).
#[derive(Module, Debug)]
pub struct SwimActor<B: Backend> {
    pub w1: Param<Tensor<B, 2>>,  // [IN, HIDDEN]
    pub b1: Param<Tensor<B, 2>>,  // [1, HIDDEN]
    pub w2: Param<Tensor<B, 2>>,  // [HIDDEN, OUT]
    pub b2: Param<Tensor<B, 2>>,  // [1, OUT]
    pub log_std: Param<Tensor<B, 2>>,  // [1, OUT]
}

impl<B: Backend> SwimActor<B> {
    /// Forward: `obs [B, IN] → μ [B, OUT]` (shared weights, broadcast bias).
    pub fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = relu(obs.matmul(self.w1.val()) + self.b1.val());  // [B,H] + [1,H]
        tanh(h.matmul(self.w2.val()) + self.b2.val())             // [B,OUT] + [1,OUT]
    }
}

/// Critic MLP — a SINGLE shared `IN → HIDDEN → 1` net returning `V(s)`.
#[derive(Module, Debug)]
pub struct SwimCritic<B: Backend> {
    pub w1: Param<Tensor<B, 2>>,  // [IN, HIDDEN]
    pub b1: Param<Tensor<B, 2>>,  // [1, HIDDEN]
    pub w2: Param<Tensor<B, 2>>,  // [HIDDEN, 1]
    pub b2: Param<Tensor<B, 2>>,  // [1, 1]
}

impl<B: Backend> SwimCritic<B> {
    /// Forward: `obs [B, IN] → V [B, 1]`.
    pub fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = relu(obs.matmul(self.w1.val()) + self.b1.val());  // [B,H] + [1,H]
        h.matmul(self.w2.val()) + self.b2.val()                   // [B,1] + [1,1]
    }
}


// ── Rollout buffer ────────────────────────────────────────────────────────────

/// One step in an agent's rollout (CPU-side; see `limb_ppo::RolloutEntry`).
#[derive(Clone, Debug)]
pub struct SwimRolloutEntry {
    pub obs:      [f32; IN],
    pub action:   [f32; OUT],
    pub log_prob: f32,
    pub value:    f32,
    pub reward:   f32,
    pub done:     bool,
}

impl Default for SwimRolloutEntry {
    fn default() -> Self {
        Self {
            obs:      [0.0; IN],
            action:   [0.0; OUT],
            log_prob: 0.0,
            value:    0.0,
            reward:   0.0,
            done:     false,
        }
    }
}


// ── Slot trait + observation inputs ───────────────────────────────────────────

/// Marker-component trait so the shared `apply_step` can extract a slot index
/// regardless of which per-level swimming slot component the query carries.
pub trait SwimSlot {
    fn slot(&self) -> u32;
}

/// Physics-derived observation inputs gathered per SWIMMING organism by
/// `gather_swim_obs_inputs`. Defaults are the neutral "no info" state.
#[derive(Clone, Copy)]
pub struct SwimObsInputs {
    /// Sin/cos pairs of each ball joint's relative-rotation Euler XYZ
    /// (relative to the BASE body's rotation), 6 per joint at offset `j*6`.
    pub joint_sincos:  [f32; 48],   // 8 joints × 6
    /// Per-joint world-frame angular velocity, 3 per joint at offset `j*3`.
    pub joint_angvel:  [f32; 24],   // 8 joints × 3
    /// Base body's world-frame rotation. Source frame for the two oracles and
    /// for the body-local velocity encoding.
    pub base_rot:      Quat,
    /// Base body's world-frame linear velocity.
    pub base_lin_vel:  Vec3,
    /// Base body's world-frame angular velocity.
    pub base_ang_vel:  Vec3,
    /// Nearest phototroph: world-frame relative position (prey − self), 3D
    /// distance, and the prey entity (so the reward can zero its deltas when
    /// the nearest target CHANGES between ticks). `None` when nothing is in
    /// `SWIM_SENSORY_RADIUS`.
    pub target:        Option<(Vec3, f32, Entity)>,
}

impl Default for SwimObsInputs {
    fn default() -> Self {
        Self {
            joint_sincos: [0.0; 48],
            joint_angvel: [0.0; 24],
            base_rot:     Quat::IDENTITY,
            base_lin_vel: Vec3::ZERO,
            base_ang_vel: Vec3::ZERO,
            target:       None,
        }
    }
}

/// 3D ROTATION ORACLE: the body-local rotation vector (axis · angle) that
/// rotates the body's forward axis (`base_rot · +Z`) onto `target_dir_world`.
/// Returned scaled by 1/π so each component is ∈ [-1, 1]. The brain can read
/// it as "rotate this much about my local X/Y/Z to face the target".
/// Anti-parallel targets (angle ≈ π, axis degenerate) fall back to the body's
/// local +Y as the turn axis — any consistent perpendicular works.
pub fn rotation_to_target_local(base_rot: Quat, target_dir_world: Vec3) -> Vec3 {
    let fwd = (base_rot * Vec3::Z).normalize_or_zero();
    let dir = target_dir_world.normalize_or_zero();
    if fwd.length_squared() < 1e-6 || dir.length_squared() < 1e-6 {
        return Vec3::ZERO;
    }
    let dot = fwd.dot(dir).clamp(-1.0, 1.0);
    let angle = dot.acos();                         // ∈ [0, π]
    let axis_world = fwd.cross(dir);
    let axis_world = if axis_world.length_squared() > 1e-8 {
        axis_world.normalize()
    } else if dot < 0.0 {
        // Facing exactly away: turn about local up (any perpendicular works).
        (base_rot * Vec3::Y).normalize_or_zero()
    } else {
        return Vec3::ZERO;                          // already facing it
    };
    // World → body-local, normalised by π.
    (base_rot.inverse() * (axis_world * angle)) / std::f32::consts::PI
}

/// Assemble the per-tick observation for one swimming organism.
pub fn build_observation(
    organism:    &Organism,
    prev_action: &[f32; OUT],
    physics:     &SwimObsInputs,
    phase:       f32,
) -> [f32; IN] {
    let mut obs = [0.0_f32; IN];
    let max_energy = organism.grown_cell_count() as f32 * MAX_ENERGY_PER_CELL;
    if max_energy > 0.0 {
        obs[0] = (organism.energy / max_energy).clamp(0.0, 1.0);
    }

    let inv_rot = physics.base_rot.inverse();

    // TARGET ORACLE (obs[1..5]) + ROTATION ORACLE (obs[5..8]). A real bearing
    // is a unit vector; no target ⇒ zero-length direction + maxed distance,
    // distinguishable from any real target.
    match physics.target {
        Some((rel, dist, _)) if rel.length_squared() > 1e-6 => {
            let dir_world = rel.normalize();
            let dir_local = inv_rot * dir_world;
            obs[1] = dir_local.x;
            obs[2] = dir_local.y;
            obs[3] = dir_local.z;
            obs[4] = (dist / crate::simulation_settings::SWIM_SENSORY_RADIUS).clamp(0.0, 1.0);
            let rotvec = rotation_to_target_local(physics.base_rot, dir_world);
            obs[5] = rotvec.x;
            obs[6] = rotvec.y;
            obs[7] = rotvec.z;
        }
        _ => {
            obs[4] = 1.0;
        }
    }

    obs[8..56].copy_from_slice(&physics.joint_sincos);    // 8 joints × 6
    obs[56..80].copy_from_slice(&physics.joint_angvel);   // 8 joints × 3

    // Base velocities in BODY-LOCAL frame (full 3D rotation, not just yaw —
    // a swimmer can be in any orientation).
    let lv = inv_rot * physics.base_lin_vel;
    let av = inv_rot * physics.base_ang_vel;
    obs[80] = lv.x; obs[81] = lv.y; obs[82] = lv.z;
    obs[83] = av.x; obs[84] = av.y; obs[85] = av.z;

    // Base orientation as a quaternion — smooth, singularity-free (sin/cos
    // Euler would gimbal-wrap for free-tumbling 3D bodies).
    obs[86] = physics.base_rot.x;
    obs[87] = physics.base_rot.y;
    obs[88] = physics.base_rot.z;
    obs[89] = physics.base_rot.w;

    obs[90..114].copy_from_slice(prev_action);

    // Phase clock — the one rhythm aid (see limb_ppo).
    obs[114] = phase.sin();
    obs[115] = phase.cos();
    obs
}


/// Walk every limb-physics body-part entity once and build a
/// `HashMap<organism_root, SwimObsInputs>`. Mirrors
/// `limb_ppo::gather_limb_obs_inputs` but 3D: no ground contacts, no
/// heightmap clearance; the nearest-prey target keeps its full 3D relative
/// position (the spatial hash is XZ-bucketed, but entries carry `Vec3` and
/// `nearest_prey` ranks by 3D distance, so the water column is covered).
pub fn gather_swim_obs_inputs(
    body_parts: &bevy::ecs::system::Query<(
        &bevy::prelude::ChildOf,
        &crate::cell::BodyPartIndex,
        &bevy::prelude::GlobalTransform,
        &bevy_rapier3d::prelude::Velocity,
    )>,
    world_grid: &crate::world_model::WorldModelGrid,
) -> HashMap<Entity, SwimObsInputs> {
    let mut out:      HashMap<Entity, SwimObsInputs> = HashMap::new();
    let mut base_rot: HashMap<Entity, Quat>          = HashMap::new();
    let mut base_pos: HashMap<Entity, Vec3>          = HashMap::new();

    // Pass 1: base bodies (BodyPartIndex 0).
    for (child_of, idx, gt, vel) in body_parts.iter() {
        if idx.0 != 0 { continue; }
        let root = child_of.parent();
        let rot = gt.rotation();
        let entry = out.entry(root).or_default();
        entry.base_rot     = rot;
        entry.base_lin_vel = vel.linear;
        entry.base_ang_vel = vel.angular;
        base_rot.insert(root, rot);
        base_pos.insert(root, gt.translation());
    }

    // Pass 2: limbs — per-joint relative rotation + angular velocity.
    for (child_of, idx, gt, vel) in body_parts.iter() {
        if idx.0 == 0 || idx.0 > MAX_LIMB_JOINTS { continue; }
        let root = child_of.parent();
        let Some(base_r) = base_rot.get(&root) else { continue };
        let j = idx.0 - 1;
        let rel = base_r.inverse() * gt.rotation();
        let (ex, ey, ez) = rel.to_euler(bevy::math::EulerRot::XYZ);
        let entry = out.entry(root).or_default();
        let base = j * 6;
        entry.joint_sincos[base    ] = ex.sin();
        entry.joint_sincos[base + 1] = ex.cos();
        entry.joint_sincos[base + 2] = ey.sin();
        entry.joint_sincos[base + 3] = ey.cos();
        entry.joint_sincos[base + 4] = ez.sin();
        entry.joint_sincos[base + 5] = ez.cos();
        let av_base = j * 3;
        entry.joint_angvel[av_base    ] = vel.angular.x;
        entry.joint_angvel[av_base + 1] = vel.angular.y;
        entry.joint_angvel[av_base + 2] = vel.angular.z;
    }

    // Pass 3: nearest phototroph (full 3D rel + distance + entity).
    for (root, entry) in out.iter_mut() {
        if let Some(pos) = base_pos.get(root) {
            entry.target = crate::world_model::nearest_prey_within(
                world_grid, *pos, crate::simulation_settings::SWIM_SENSORY_RADIUS,
            );
        }
    }

    out
}


// ── Pool ──────────────────────────────────────────────────────────────────────

/// SHARED-policy engine wrapped by the swimming pool resource. Holds the ONE
/// actor + ONE critic, their optimisers, per-slot rollout buffers + bookkeeping,
/// the free-slot list, and the entity↔slot map. `n` is the max concurrent
/// swimmers (sizes the per-slot bookkeeping arrays), NOT a per-organism weight
/// count — the weights are shared across all slots.
pub struct BrainPoolSwim {
    pub n:      usize,
    pub device: CudaDevice,
    /// Per-species shared nets. Every organism of a species uses AND trains its
    /// species' net; different species hold different weights, so behaviour
    /// diverges by species while same-species individuals act alike (momentary
    /// variation comes only from each organism's own exploration noise). Keyed
    /// by `Organism::species_id` (`UNCLASSIFIED` until first classification);
    /// created lazily, warm-started with the gait prior.
    pub species: HashMap<u32, SpeciesBrain>,
    pub rollouts: Vec<VecDeque<SwimRolloutEntry>>,
    pub free: Vec<u32>,
    pub map: HashMap<Entity, u32>,
    pub prev_action: Vec<[f32; OUT]>,
    /// Per-slot `Organism::predations` as of the last apply tick (eat-reward
    /// delta). Reset to 0 in `enrol`.
    pub prev_predations: Vec<u8>,
    /// Per-slot `(target_entity, 3D distance)` as of the previous apply tick.
    /// Progress is paid only when the SAME entity is still the target.
    pub prev_target: Vec<Option<(Entity, f32)>>,
    /// Per-slot `(target_entity, facing alignment ∈ [-1,1])` as of the previous
    /// apply tick — the rectified rotation-toward-target reward.
    pub prev_alignment: Vec<Option<(Entity, f32)>>,
    /// Per-slot, per-dim TEMPORALLY-CORRELATED exploration noise (AR(1)/OU).
    /// Kept PER ORGANISM even under the shared policy, so the swimmers explore
    /// diverse trajectories (better coverage for the one shared learner) and
    /// move non-identically moment-to-moment despite identical weights.
    pub noise_state: Vec<[f32; OUT]>,
    /// Per-slot species key as of the last apply tick. Lets `train` group slots
    /// by species without an organism query (written in `apply_step`).
    pub slot_species: Vec<u32>,
    pub ticks_since_train: usize,
    pub training_step: u64,
    pub training_history: VecDeque<SwimTrainingStep>,
    /// DIAGNOSTIC (logged, not used in learning): mean & min nearest-prey
    /// distance over active swimmers as of the last apply tick. The
    /// time-series `target_distance` column is the SLIDING sensory field and
    /// is stale for swimmers, so this is the only honest "are they closing on
    /// prey?" signal. A falling `dbg_mean_tgt` over training = pursuit is being
    /// learned; `dbg_min_tgt` near `EAT_RADIUS` (6) = a swimmer is at biting range.
    pub dbg_mean_tgt: f32,
    pub dbg_min_tgt:  f32,
}

/// Cap on the in-memory training-step ring buffer.
pub const SWIM_TRAINING_HISTORY_CAP: usize = 1024;

/// Type-erased optimisers (mirrors `limb_ppo::BrainOptActor`/`Critic`).
pub trait SwimOptActor {
    fn step(&mut self, lr: f64, m: SwimActor<MyBackend>, g: GradientsParams) -> SwimActor<MyBackend>;
}
impl<O: Optimizer<SwimActor<MyBackend>, MyBackend>> SwimOptActor for O {
    fn step(&mut self, lr: f64, m: SwimActor<MyBackend>, g: GradientsParams) -> SwimActor<MyBackend> {
        Optimizer::step(self, lr, m, g)
    }
}
pub trait SwimOptCritic {
    fn step(&mut self, lr: f64, m: SwimCritic<MyBackend>, g: GradientsParams) -> SwimCritic<MyBackend>;
}
impl<O: Optimizer<SwimCritic<MyBackend>, MyBackend>> SwimOptCritic for O {
    fn step(&mut self, lr: f64, m: SwimCritic<MyBackend>, g: GradientsParams) -> SwimCritic<MyBackend> {
        Optimizer::step(self, lr, m, g)
    }
}


/// Sentinel species key for organisms not yet classified by the speciation
/// system (real `species_id`s start at 1). All unclassified swimmers share this
/// one transitional net until the ~1 Hz speciation tick assigns them; the next
/// apply tick then re-points them at their real species' net automatically
/// (the per-tick species lookup makes reclassification a no-op to handle).
pub const UNCLASSIFIED: u32 = 0;

/// One species' shared actor + critic + their optimisers, plus that species'
/// own PPO step counter. Created lazily by `new_species_brain` and warm-started
/// with the oscillatory gait prior.
pub struct SpeciesBrain {
    pub actor:         SwimActor<MyBackend>,
    pub critic:        SwimCritic<MyBackend>,
    pub opt_actor:     Box<dyn SwimOptActor>,
    pub opt_critic:    Box<dyn SwimOptCritic>,
    pub training_step: u64,
}

/// Build one freshly-initialised species brain. Biases zero; `log_std`
/// initialised to `SWIM_LOG_STD_INIT`; weights fan-in-scaled so the tanh `μ`
/// doesn't saturate at init, with the OSCILLATORY WARM-START applied: two
/// hidden carrier units (0 = sin, 1 = cos) fed only the phase-clock inputs so
/// each output starts as `A·sin(phase + φ_k)` — a TRAINABLE rhythmic gait
/// prior, not a CPG. Per-output phase `φ_k` is randomised so the 8 joints don't
/// stroke in lockstep. (Each new species gets its OWN warm-started prior.)
pub fn new_species_brain(device: &CudaDevice) -> SpeciesBrain {
    let z = Initializer::Zeros;
    let log_std_init = Initializer::Constant { value: SWIM_LOG_STD_INIT as f64 };

    const WARMSTART_PHASE_GAIN: f32 = 2.0;
    const WARMSTART_BIAS:       f32 = 3.0;
    const PHASE_SIN_IDX: usize = 114;
    const PHASE_COS_IDX: usize = 115;
    use rand::RngExt as _;
    let mut rng = rand::rng();
    let rand_small = |rng: &mut rand::rngs::ThreadRng| rng.random_range(-0.093_f32..0.093);

    // Single shared net: w1 [IN,HIDDEN], w2 [HIDDEN,OUT] (no batch dim).
    let mut w1v = vec![0.0_f32; IN * HIDDEN];
    for v in w1v.iter_mut() { *v = rand_small(&mut rng); }
    let mut b1v = vec![0.0_f32; HIDDEN];
    let mut w2v = vec![0.0_f32; HIDDEN * OUT];
    for v in w2v.iter_mut() { *v = rand_small(&mut rng); }
    let mut b2v = vec![0.0_f32; OUT];

    // Carrier units 0 (sin) and 1 (cos): clear their input columns, then
    // feed ONLY the phase-clock inputs.
    for inp in 0..IN {
        w1v[inp * HIDDEN + 0] = 0.0;
        w1v[inp * HIDDEN + 1] = 0.0;
    }
    w1v[PHASE_SIN_IDX * HIDDEN + 0] = WARMSTART_PHASE_GAIN;
    w1v[PHASE_COS_IDX * HIDDEN + 1] = WARMSTART_PHASE_GAIN;
    b1v[0] = WARMSTART_BIAS;
    b1v[1] = WARMSTART_BIAS;
    for k in 0..OUT {
        let phi: f32 = rng.random_range(0.0..std::f32::consts::TAU);
        let (s, c) = (phi.sin(), phi.cos());
        w2v[0 * OUT + k] = SWIM_WARMSTART_AMP * c; // h0 (sin) → out k
        w2v[1 * OUT + k] = SWIM_WARMSTART_AMP * s; // h1 (cos) → out k
        b2v[k] = -WARMSTART_BIAS * SWIM_WARMSTART_AMP * (c + s); // cancel constant
    }

    let actor = SwimActor::<MyBackend> {
        w1:      Param::from_tensor(Tensor::<MyBackend, 2>::from_data(TensorData::new(w1v, [IN, HIDDEN]), device)),
        b1:      Param::from_tensor(Tensor::<MyBackend, 2>::from_data(TensorData::new(b1v, [1, HIDDEN]),  device)),
        w2:      Param::from_tensor(Tensor::<MyBackend, 2>::from_data(TensorData::new(w2v, [HIDDEN, OUT]), device)),
        b2:      Param::from_tensor(Tensor::<MyBackend, 2>::from_data(TensorData::new(b2v, [1, OUT]),     device)),
        log_std: log_std_init.init([1, OUT], device),
    };
    let w = Initializer::Uniform { min: -0.093, max: 0.093 };
    let critic = SwimCritic::<MyBackend> {
        w1: w.init([IN, HIDDEN], device),
        b1: z.init([1, HIDDEN],  device),
        w2: w.init([HIDDEN, 1],  device),
        b2: z.init([1, 1],       device),
    };
    let opt_actor:  Box<dyn SwimOptActor>  = Box::new(AdamConfig::new().init());
    let opt_critic: Box<dyn SwimOptCritic> = Box::new(AdamConfig::new().init());
    SpeciesBrain { actor, critic, opt_actor, opt_critic, training_step: 0 }
}

impl BrainPoolSwim {
    /// Allocate the pool. `n` sizes the per-slot bookkeeping arrays (max
    /// concurrent swimmers). Per-species nets are NOT built here — they're
    /// created lazily (warm-started) by `ensure_species` as species appear, so
    /// the pool starts with no nets and grows one per live species.
    pub fn new(n: usize, device: CudaDevice) -> Self {
        let mut rollouts = Vec::with_capacity(n);
        for _ in 0..n { rollouts.push(VecDeque::with_capacity(ROLLOUT_LEN)); }
        let free: Vec<u32> = (0..n as u32).rev().collect();
        Self {
            n,
            device,
            species: HashMap::new(),
            rollouts,
            free,
            map: HashMap::new(),
            prev_action:     vec![[0.0; OUT]; n],
            prev_predations: vec![0u8; n],
            prev_target:     vec![None; n],
            prev_alignment:  vec![None; n],
            noise_state:     vec![[0.0; OUT]; n],
            slot_species:    vec![UNCLASSIFIED; n],
            ticks_since_train: 0,
            training_step: 0,
            training_history: VecDeque::with_capacity(SWIM_TRAINING_HISTORY_CAP),
            dbg_mean_tgt: 0.0,
            dbg_min_tgt:  0.0,
        }
    }

    /// Read-only accessor for a future dataset-export hook.
    pub fn training_history(&self) -> &VecDeque<SwimTrainingStep> {
        &self.training_history
    }

    /// Lazily create a species' shared brain (warm-started gait prior) on first
    /// sighting. Idempotent. New species fork off existing ones via the
    /// speciation system; their net simply starts fresh from the prior — within
    /// a species the shared net is already trained, so a newborn of an existing
    /// species is competent immediately (it just uses the species net).
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

    /// Pop a fresh slot for an organism. `None` once the pool is full. No
    /// weight init — every slot uses the shared net; the slot is pure
    /// bookkeeping (rollout buffer + reward/noise state).
    pub fn enrol(&mut self, e: Entity) -> Option<u32> {
        let s = self.free.pop()?;
        self.map.insert(e, s);
        self.rollouts[s as usize].clear();
        self.prev_action[s as usize]     = [0.0; OUT];
        self.prev_predations[s as usize] = 0;
        self.prev_target[s as usize]     = None;
        self.prev_alignment[s as usize]  = None;
        self.noise_state[s as usize]     = [0.0; OUT];
        self.slot_species[s as usize]    = UNCLASSIFIED;
        Some(s)
    }

    /// Release a slot when its organism is removed.
    pub fn release(&mut self, e: Entity, s: u32) {
        self.map.remove(&e);
        if (s as usize) < self.rollouts.len() {
            self.rollouts[s as usize].clear();
        }
        self.free.push(s);
    }

    /// GAE advantages + returns for one agent's rollout (see limb_ppo).
    pub fn compute_gae(
        buf:         &VecDeque<SwimRolloutEntry>,
        values_tail: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let t_len = buf.len();
        let mut advantages = vec![0.0_f32; t_len];
        let mut last_gae   = 0.0_f32;
        let mut next_value = values_tail;
        let mut next_nontermi = 1.0_f32;
        for t in (0..t_len).rev() {
            let r = &buf[t];
            let nontermi = if r.done { 0.0 } else { next_nontermi };
            let delta = r.reward + GAMMA * next_value * nontermi - r.value;
            last_gae = delta + GAMMA * LAMBDA * nontermi * last_gae;
            advantages[t] = last_gae;
            next_value    = r.value;
            next_nontermi = if r.done { 0.0 } else { 1.0 };
        }
        let returns: Vec<f32> = advantages.iter().zip(buf.iter())
            .map(|(a, e)| a + e.value)
            .collect();
        (advantages, returns)
    }

    /// Per-tick apply step. Builds each active organism's observation, runs ONE
    /// batched forward through the SHARED actor + critic, samples a correlated
    /// action, writes it to `SwimJointTargets`, computes the reward, and appends
    /// a rollout entry to that organism's slot buffer. Trains every
    /// `ROLLOUT_LEN` ticks on the POOLED transitions of all slots.
    pub fn apply_step<S: Component + Copy + SwimSlot>(
        &mut self,
        mut organisms:     Query<(Entity, &Organism, &mut SwimJointTargets, &S)>,
        obs_inputs:        &HashMap<Entity, SwimObsInputs>,
        virtual_time_secs: f32,
    ) {
        let phase = virtual_time_secs * std::f32::consts::TAU * GAIT_FREQUENCY_HZ;

        // ── 1. Collect active slots + observations, grouped by SPECIES. Each
        //       organism's CURRENT species (read fresh every tick, so a
        //       reclassification by the speciation system needs no handling) is
        //       recorded in `slot_species` for `train`. ──
        let mut input_buf: Vec<f32> = vec![0.0; self.n * IN];
        let mut active: Vec<(Entity, u32)> = Vec::new();
        let mut groups: HashMap<u32, Vec<usize>> = HashMap::new();
        let default_phys = SwimObsInputs::default();
        for (e, organism, _, slot) in organisms.iter() {
            let s = slot.slot() as usize;
            if s >= self.n { continue; }
            let prev_action = self.prev_action[s];
            let phys = obs_inputs.get(&e).unwrap_or(&default_phys);
            let obs = build_observation(organism, &prev_action, phys, phase);
            let base = s * IN;
            input_buf[base..base + IN].copy_from_slice(&obs);
            let key = organism.species_id.unwrap_or(UNCLASSIFIED);
            self.slot_species[s] = key;
            groups.entry(key).or_default().push(s);
            active.push((e, slot.slot()));
        }
        if active.is_empty() { return; }

        // ── 2. One batched forward PER SPECIES through that species' shared net;
        //       results scatter back into per-slot `mu_buf`/`v_buf` so the
        //       sampling/reward loop below is identical to the single-net case. ──
        let device = self.device.clone();
        let mut mu_buf = vec![0.0_f32; self.n * OUT];
        let mut v_buf  = vec![0.0_f32; self.n];
        for (key, slots) in &groups {
            self.ensure_species(*key);
            let brain = self.species.get(key).expect("species ensured above");
            let cnt = slots.len();
            let mut rows = vec![0.0_f32; cnt * IN];
            for (i, &s) in slots.iter().enumerate() {
                rows[i * IN..i * IN + IN].copy_from_slice(&input_buf[s * IN..s * IN + IN]);
            }
            let obs_t = Tensor::<MyBackend, 2>::from_data(TensorData::new(rows, [cnt, IN]), &device);
            let mu = brain.actor.forward(obs_t.clone()).into_data().into_vec::<f32>().expect("swim actor forward");
            let v  = brain.critic.forward(obs_t).into_data().into_vec::<f32>().expect("swim critic forward");
            for (i, &s) in slots.iter().enumerate() {
                mu_buf[s * OUT..s * OUT + OUT].copy_from_slice(&mu[i * OUT..i * OUT + OUT]);
                v_buf[s] = v[i];
            }
        }
        let sigma  = SWIM_LOG_STD_INIT.exp();

        // ── 3. Sample per slot — TEMPORALLY-CORRELATED exploration (AR(1)/OU,
        //       variance-preserving so the action marginal stays N(μ,σ²) and
        //       the recorded log-prob is exact; PPO's clip bounds the temporal
        //       correlation bias). Held excursions stroke water → net thrust. ──
        let corr    = SWIM_NOISE_CORR;
        let diffuse = (1.0 - corr * corr).max(0.0).sqrt();
        let mut rng = rand::rng();
        let mut actions: Vec<(Entity, u32, [f32; OUT], f32)> = Vec::with_capacity(active.len());
        for &(e, s) in &active {
            let su = s as usize;
            let base = su * OUT;
            let mu = &mu_buf[base..base + OUT];
            let mut action = [0.0_f32; OUT];
            let mut log_prob = 0.0_f32;
            for i in 0..OUT {
                let eps = gaussian_noise(&mut rng);
                let nz = corr * self.noise_state[su][i] + diffuse * eps;
                self.noise_state[su][i] = nz;
                action[i] = (mu[i] + nz * sigma).clamp(-1.0, 1.0);
                let diff = (action[i] - mu[i]) / sigma;
                log_prob += -0.5 * diff * diff
                            - sigma.ln()
                            - 0.5 * (2.0 * std::f32::consts::PI).ln();
            }
            actions.push((e, s, action, log_prob));
        }

        // ── 4. Write actions, compute reward, log rollout entries. ──
        // (Diagnostic: track nearest-prey distance across active swimmers.)
        let mut tgt_sum = 0.0_f32; let mut tgt_min = f32::MAX; let mut tgt_cnt = 0u32;
        for (e, s, action, log_prob) in actions {
            let Ok((_, organism, mut targets, _)) = organisms.get_mut(e) else { continue };
            targets.0 = action;

            let su = s as usize;
            // BIG sparse eat reward.
            let pred_delta = organism.predations.saturating_sub(self.prev_predations[su]) as f32;
            let event_reward = K_SWIM_EAT * pred_delta;

            let phys = obs_inputs.get(&e).unwrap_or(&default_phys);
            if let Some((_, d, _)) = phys.target {
                tgt_sum += d; tgt_cnt += 1; if d < tgt_min { tgt_min = d; }
            }

            // TARGET objective: dense reward per world-unit of 3D distance closed
            // (rectified, same-entity guarded) + facing alignment for the
            // ROTATION objective.
            let (progress, alignment) = match phys.target {
                Some((rel, _dist, ent)) => {
                    // CLOSING SPEED toward the target: the body's own velocity
                    // projected onto the unit direction to the prey (signed —
                    // moving away is penalised). Smooth, available every tick,
                    // and a clean per-tick gradient (see K_SWIM_PROGRESS).
                    // Clamped to the linear-speed governor so it can't spike.
                    let dir = rel.normalize_or_zero();
                    let closing = phys.base_lin_vel.dot(dir)
                        .clamp(-crate::simulation_settings::MAX_LIMB_LINEAR_SPEED,
                                crate::simulation_settings::MAX_LIMB_LINEAR_SPEED);
                    self.prev_target[su] = Some((ent, _dist));
                    let fwd = (phys.base_rot * Vec3::Z).normalize_or_zero();
                    (closing, Some((ent, fwd.dot(dir))))
                }
                None => {
                    self.prev_target[su] = None;
                    (0.0, None)
                }
            };

            // ROTATION objective: rectified tick-over-tick gain in facing
            // alignment (turning toward pays; turning away is free) + a small
            // absolute facing bonus for HOLDING the front on target.
            let align_gain = match (self.prev_alignment[su], alignment) {
                (Some((pe, pa)), Some((ce, ca))) if pe == ce => (ca - pa).max(0.0),
                _                                            => 0.0,
            };
            self.prev_alignment[su] = alignment;
            let align_abs = alignment.map(|(_, a)| a.max(0.0)).unwrap_or(0.0);

            // Anti-corkscrew: mild base spin penalty.
            let spin = phys.base_ang_vel.length();

            let dense_reward = K_SWIM_PROGRESS   * progress
                             + K_SWIM_ALIGN_GAIN * align_gain
                             + K_SWIM_ALIGN      * align_abs
                             - K_SWIM_SPIN       * spin;

            let reward = event_reward + dense_reward;
            self.prev_predations[su] = organism.predations;
            self.prev_action[su]     = action;

            let obs_base = su * IN;
            let mut obs_row = [0.0_f32; IN];
            obs_row.copy_from_slice(&input_buf[obs_base..obs_base + IN]);
            let entry = SwimRolloutEntry {
                obs:      obs_row,
                action,
                log_prob,
                value:    v_buf[su],
                reward,
                done:     false,
            };
            let buf = &mut self.rollouts[su];
            if buf.len() >= ROLLOUT_LEN { buf.pop_front(); }
            buf.push_back(entry);
        }

        if tgt_cnt > 0 {
            self.dbg_mean_tgt = tgt_sum / tgt_cnt as f32;
            self.dbg_min_tgt  = tgt_min;
        }

        self.ticks_since_train += 1;
        if self.ticks_since_train >= ROLLOUT_LEN {
            self.train(virtual_time_secs);
            self.ticks_since_train = 0;
        }
    }

    /// PER-SPECIES PPO update. Groups every active slot by its CURRENT species
    /// (`slot_species`, written each apply tick) and, for each species, pools
    /// only THAT species' transitions into one flat batch `[M, ·]`, computes
    /// per-trajectory GAE, normalises advantages within the species, and runs
    /// `PPO_EPOCHS` clipped-surrogate + value-MSE + entropy steps on that
    /// species' own actor/critic. All members of a species update the same
    /// weights (so they converge to one policy); different species are trained
    /// independently (so they diverge).
    pub fn train(&mut self, virtual_time_secs: f32) {
        self.prune_species();
        // Group active slots (non-empty rollout) by their current species.
        let mut groups: HashMap<u32, Vec<usize>> = HashMap::new();
        for s in 0..self.n {
            if !self.rollouts[s].is_empty() {
                groups.entry(self.slot_species[s]).or_default().push(s);
            }
        }
        let device = self.device.clone();
        let log_2pi = (2.0_f32 * std::f32::consts::PI).ln();
        let entropy_const = 0.5_f32 * (2.0_f32 * std::f32::consts::PI * std::f32::consts::E).ln();

        for (key, slots) in groups {
            self.ensure_species(key);
            let cnt = slots.len();

            // ── 0. V(s_T) bootstrap per slot (batched critic forward). ──
            let mut last_obs_buf: Vec<f32> = vec![0.0; cnt * IN];
            for (i, &s) in slots.iter().enumerate() {
                if let Some(last) = self.rollouts[s].back() {
                    last_obs_buf[i * IN..i * IN + IN].copy_from_slice(&last.obs);
                }
            }
            let bootstrap_vec = {
                let brain = self.species.get(&key).expect("species ensured above");
                let last_obs_t = Tensor::<MyBackend, 2>::from_data(
                    TensorData::new(last_obs_buf, [cnt, IN]), &device);
                brain.critic.forward(last_obs_t)
                    .into_data().into_vec::<f32>().expect("swim bootstrap critic forward")
            };

            // ── 1. Per-slot GAE → POOLED flat buffers for THIS species. ──
            let mut states:  Vec<f32> = Vec::new();
            let mut actions: Vec<f32> = Vec::new();
            let mut old_lp:  Vec<f32> = Vec::new();
            let mut adv:     Vec<f32> = Vec::new();
            let mut returns: Vec<f32> = Vec::new();
            for (i, &s) in slots.iter().enumerate() {
                let buf = &self.rollouts[s];
                if buf.is_empty() { continue; }
                let (advantages, rets) = Self::compute_gae(buf, bootstrap_vec[i]);
                for (ti, entry) in buf.iter().enumerate() {
                    states.extend_from_slice(&entry.obs);
                    actions.extend_from_slice(&entry.action);
                    old_lp.push(entry.log_prob);
                    adv.push(advantages[ti]);
                    returns.push(rets[ti]);
                }
            }
            let m = old_lp.len();
            if m == 0 { continue; }

            // ── 1-bis. Per-species advantage normalisation + return summary. ──
            let adv_mean: f32 = adv.iter().sum::<f32>() / m as f32;
            let adv_var:  f32 = adv.iter().map(|a| (a - adv_mean) * (a - adv_mean)).sum::<f32>() / m as f32;
            let adv_std = adv_var.sqrt().max(1e-8);
            for a in adv.iter_mut() { *a = (*a - adv_mean) / adv_std; }

            let ret_mean: f64 = returns.iter().map(|&r| r as f64).sum::<f64>() / m as f64;
            let ret_var:  f64 = returns.iter().map(|&r| (r as f64 - ret_mean) * (r as f64 - ret_mean)).sum::<f64>() / m as f64;
            let mean_return = ret_mean as f32;
            let return_var  = ret_var as f32;

            // ── 2. GPU tensors ([M, ·]). ──
            let states_t  = Tensor::<MyBackend, 2>::from_data(TensorData::new(states,  [m, IN]),  &device);
            let actions_t = Tensor::<MyBackend, 2>::from_data(TensorData::new(actions, [m, OUT]), &device);
            let old_lp_t  = Tensor::<MyBackend, 2>::from_data(TensorData::new(old_lp,  [m, 1]),   &device);
            let adv_t     = Tensor::<MyBackend, 2>::from_data(TensorData::new(adv,     [m, 1]),   &device);
            let returns_t = Tensor::<MyBackend, 2>::from_data(TensorData::new(returns, [m, 1]),   &device);

            // ── 3. PPO_EPOCHS updates over THIS species' net. ──
            let mut last_actor_loss:  f32 = 0.0;
            let mut last_critic_loss: f32 = 0.0;
            let mut last_entropy:     f32 = 0.0;
            let mut last_total_loss:  f32 = 0.0;

            let species_step = {
                let brain = self.species.get_mut(&key).expect("species ensured above");
                for epoch in 0..PPO_EPOCHS {
                    let mu = brain.actor.forward(states_t.clone());          // [M, OUT]
                    let log_std_b = brain.actor.log_std.val();               // [1, OUT]
                    let sigma_b   = log_std_b.clone().exp();                 // [1, OUT]
                    let inv_sigma_sq = sigma_b.powf_scalar(-2.0);            // [1, OUT]
                    let diff = actions_t.clone() - mu;                       // [M, OUT]
                    let new_lp = (diff.clone() * diff)
                        .mul(inv_sigma_sq)
                        .mul_scalar(-0.5_f32)
                        .sub(log_std_b.clone())
                        .sub_scalar(0.5_f32 * log_2pi)
                        .sum_dim(1);                                         // [M, 1]

                    let ratio = (new_lp - old_lp_t.clone()).exp();           // [M, 1]
                    let surr1 = ratio.clone() * adv_t.clone();
                    let surr2 = ratio.clamp(1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_t.clone();
                    let policy_loss = surr1.min_pair(surr2).neg();           // [M, 1]

                    let v = brain.critic.forward(states_t.clone());          // [M, 1]
                    let v_diff = v - returns_t.clone();
                    let value_loss = (v_diff.clone() * v_diff).mul_scalar(0.5);

                    let entropy = log_std_b.add_scalar(entropy_const).sum_dim(1).mean();

                    let loss = (policy_loss.clone() + value_loss.clone().mul_scalar(VALUE_LOSS_COEF)).mean()
                        - entropy.clone().mul_scalar(ENTROPY_COEF);

                    if epoch == PPO_EPOCHS - 1 {
                        last_actor_loss = crate::limb_ppo::scalar_of(policy_loss.mean());
                        last_total_loss = crate::limb_ppo::scalar_of(loss.clone());
                        last_entropy    = crate::limb_ppo::scalar_of(entropy);
                    }

                    let actor_clone = brain.actor.clone();
                    let actor_grads = GradientsParams::from_grads(loss.backward(), &brain.actor);
                    brain.actor = brain.opt_actor.step(LR, actor_clone, actor_grads);

                    // Critic: fresh graph (the one above was consumed by backward).
                    let v2 = brain.critic.forward(states_t.clone());
                    let v2_diff = v2 - returns_t.clone();
                    let critic_loss = (v2_diff.clone() * v2_diff).mul_scalar(0.5).mean();
                    if epoch == PPO_EPOCHS - 1 {
                        last_critic_loss = crate::limb_ppo::scalar_of(critic_loss.clone());
                    }
                    let critic_clone = brain.critic.clone();
                    let critic_grads = GradientsParams::from_grads(critic_loss.backward(), &brain.critic);
                    brain.critic = brain.opt_critic.step(LR, critic_clone, critic_grads);
                }
                brain.training_step += 1;
                brain.training_step
            };

            // ── 4. Log this species' update. ──
            self.training_step += 1;
            if self.training_history.len() >= SWIM_TRAINING_HISTORY_CAP {
                self.training_history.pop_front();
            }
            self.training_history.push_back(SwimTrainingStep {
                step:               self.training_step,
                virtual_time_secs,
                n_active:           cnt as u32,
                actor_loss:         last_actor_loss,
                critic_loss:        last_critic_loss,
                entropy:            last_entropy,
                total_loss:         last_total_loss,
                mean_return,
                return_var,
                supervised_loss:    0.0,
            });
            // Periodic progress line: a rising `mean_return` for a species = its
            // shared policy is learning to pursue/eat.
            if self.training_step % 20 == 0 {
                info!(
                    "swim per-species PPO | species {} step {} | t={:.0}s | members {} | mean_return {:.3} | \
                     actor_loss {:.4} critic_loss {:.4} entropy {:.3} | \
                     prey_dist mean {:.1} min {:.1}",
                    key, species_step, virtual_time_secs, cnt,
                    mean_return, last_actor_loss, last_critic_loss, last_entropy,
                    self.dbg_mean_tgt, self.dbg_min_tgt,
                );
            }
        }

        // Clear ALL rollout buffers (every active slot was consumed above).
        for r in &mut self.rollouts { r.clear(); }
    }
}
