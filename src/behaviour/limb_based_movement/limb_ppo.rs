// Shared PPO engine for the limb-based brain pools (herbivore_1_limb,
// L2_limb, L3_limb).
//
// Per-organism PPO with no shared weights:
//
//   * Every organism has its OWN actor and critic MLPs (rows in the
//     pool's batched weight tensors). Forward passes run as batched
//     `matmul`s — one call per layer, regardless of population.
//   * Actor outputs `μ` (mean target joint angle, one per DOF) and the
//     pool stores a separate per-organism `log_std` tensor — so the
//     policy is a Diagonal Gaussian `N(μ_i, exp(log_std_i)²)`.
//   * Critic outputs a single scalar `V(s)`.
//   * Rollouts are stored per-organism in fixed-length ring buffers; at
//     `ROLLOUT_LEN` we compute per-agent GAE advantages + returns and
//     run `PPO_EPOCHS` epochs of the clipped-surrogate update against
//     the buffered (state, action, log_prob, advantage, return) tuples.
//
// This file owns the network architecture and the math primitives
// (forward, sampling, GAE, PPO loss). Per-level files in the same folder
// wrap this engine in their own `Resource` / `Component` newtypes,
// supply the enrolment filter (`IntelligenceLevel` + `!sliding_movement`
// + carnivore-vs-herbivore split), and delegate assign / free / apply
// to the shared helpers here.
//
// EMERGENT per-joint control: the brain directly commands each limb hinge's
// target angle and walking must EMERGE from RL — there is no CPG generating the
// rhythm and no pursuit force aiming the body. The exact observation/action
// layout is documented on the `IN` / `OUT` consts below; in summary:
//   * Observation (`IN`): energy, body-local world-model neighbours, per-joint
//     hinge angle (sin/cos) + angular velocity for up to MAX_LIMB_JOINTS joints,
//     base pose/velocity + up-vector, per-limb foot contacts, prev action,
//     nearest-prey bearing, and a phase clock (sin/cos) — the one rhythm aid.
//   * Action (`OUT = MAX_LIMB_JOINTS`): one target hinge angle per limb body
//     part (output k → body-part k+1). The actor outputs tanh-bounded `μ`;
//     sampling adds Gaussian noise scaled by `exp(LOG_STD_INIT)`; the value is
//     scaled by `LIMB_SWING_LIMIT` and tracked by the in-solver hinge motor.

use bevy::prelude::*;
use burn::module::{Initializer, Module, Param};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Tensor, TensorData, backend::Backend, activation::{relu, tanh}};
use burn_cuda::CudaDevice;
use std::collections::{HashMap, VecDeque};

use crate::colony::Organism;
use crate::energy::MAX_ENERGY_PER_CELL;
use crate::rl_helpers::{MyBackend, gaussian_noise};


// ── Architecture constants ────────────────────────────────────────────────────

/// Observation dimension. EMERGENT per-joint layout (`MAX_LIMB_JOINTS = 8`):
///   * `obs[0]`           — energy_norm                          (1)
///   * `obs[1..25]`       — world-model neighbours (body-local)  (24)
///   * `obs[25..73]`      — per-joint angles: 8 joints × (sin,cos
///                          of Euler-XYZ relative to base) = 8×6 (48)
///   * `obs[73..97]`      — per-joint angular velocity: 8 × 3    (24)
///   * `obs[97..109]`     — base orientation (sin/cos × 3 axes)
///                          + angular vel (3) + linear vel (3)   (12)
///   * `obs[109..112]`    — base up-vector `base_rot · +Y`        (3)
///   * `obs[112..121]`    — per-limb contact (base + 8 limbs)    (9)
///   * `obs[121..129]`    — prev_action recurrence (OUT)         (8)
///   * `obs[129..132]`    — nearest-prey bearing (body-local
///                          dir_x, dir_z, dist_norm)             (3)
///   * `obs[132..134]`    — phase clock (sin, cos)               (2)
///
/// Orientations are encoded as `(sin, cos)` pairs of Euler-XYZ angles (not raw
/// angles) so the ±π wrap is continuous. The **up-vector** (`base_rot · +Y`) is
/// the singularity-free tilt/fall signal; its `.y` is exactly the `uprightness`
/// term the reward grades on. The nearest-prey bearing closes the
/// observation/reward gap for `K_HEADING` / `K_PROGRESS`. The **phase clock** is
/// the only "rhythm aid": `sin/cos(2π·GAIT_FREQUENCY_HZ·t)` of virtual time,
/// handed to the brain so a feedforward policy can phase-lock a sustained gait
/// oscillation — the brain still generates every joint command itself (no CPG).
///
/// `OUT = MAX_LIMB_JOINTS`: one target hinge angle per limb body part, so every
/// joint (each hip AND knee, each Bilateral mirror half) is independently
/// controlled and an alternating gait can EMERGE. `drive_limb_motors` maps
/// `out[i-1] → body-part i`'s hinge motor target.
pub const IN:     usize = 134;
pub const HIDDEN: usize = 128;
pub const OUT:    usize = 8;

// `OUT` is the per-joint action dimension; it MUST equal `MAX_LIMB_JOINTS`
// (the physics-side joint bound used by `drive_limb_motors`). A mismatch would
// silently drop or wrap joint commands.
const _: () = assert!(OUT == crate::simulation_settings::MAX_LIMB_JOINTS);

pub use crate::simulation_settings::{
    ROLLOUT_LEN, PPO_EPOCHS, CLIP_EPS, GAMMA, LAMBDA, LR,
    VALUE_LOSS_COEF, ENTROPY_COEF, LOG_STD_INIT, GAIT_FREQUENCY_HZ,
};

// ── Reward shaping ──
//
// Sparse primary signals (event-driven, same shape as the sliding pools)
// plus dense locomotion-intrinsic terms designed to AVOID the freeze
// local optimum that the first reward draft fell into:
//
//   1. Forward velocity — `lin_vel_xz · heading_xz` (signed projection
//      onto the body's facing direction). The bootstrap version of this
//      term rewarded speed MAGNITUDE (`|lin_vel_xz|`, any direction) to
//      get organisms moving at all when forward velocity was ~0 under
//      random actions. That worked — but once they moved, magnitude
//      rewarded spinning/circling just as much as directed travel, and
//      the policy converged to a spin-drift local optimum. Projecting
//      onto the heading pays only for travel in the facing direction:
//      pure spin nets ~0, backward drift is mildly penalised. Combined
//      with `K_HEADING` (turn to face prey) this composes into directed
//      pursuit — face the target, then move forward.
//   2. Uprightness GATED on motion — `(rot · +Y).y * min(1, speed / IDLE_THRESH)`.
//      Bare uprightness was paying the freeze policy a constant alive
//      bonus. Gating it on motion means standing still upright scores 0;
//      the term only kicks in once the organism is actually locomoting.
//   3. Idle penalty — `−K_IDLE * max(0, 1 − speed / IDLE_THRESH)`.
//      Standing still loses reward every tick. Pushes the policy
//      gradient *out* of the freeze basin.
//
// Action smoothness (`K_SMOOTH`) was removed: it's the right tool to
// clean up jitter once a gait exists, but during bootstrap it directly
// rewards constant outputs, which is exactly the freeze policy.
pub use crate::simulation_settings::{
    K_EAT, K_REPRO, K_FWD, K_UP, K_IDLE, K_PROGRESS, K_HEADING, K_STEP,
    IDLE_THRESH, K_MOVE, SPEED_REWARD_CAP, K_SPIN,
};


// ── Networks ──────────────────────────────────────────────────────────────────

/// Actor MLP — per-organism `IN → HIDDEN → OUT` returning `μ` for the
/// diagonal-Gaussian policy. `log_std` lives on the pool (not inside
/// the network module) so it can be sampled / mutated by the lifecycle
/// systems without going through Burn's autodiff path.
#[derive(Module, Debug)]
pub struct LimbActor<B: Backend> {
    pub w1: Param<Tensor<B, 3>>,  // [N, IN, HIDDEN]
    pub b1: Param<Tensor<B, 2>>,  // [N, HIDDEN]
    pub w2: Param<Tensor<B, 3>>,  // [N, HIDDEN, OUT]
    pub b2: Param<Tensor<B, 2>>,  // [N, OUT]
    /// Per-organism `log_std` for each action dim. Trainable.
    pub log_std: Param<Tensor<B, 2>>,  // [N, OUT]
}

impl<B: Backend> LimbActor<B> {
    /// Inference forward: `obs [N, IN] → μ [N, OUT]`. One batched matmul
    /// per layer; every organism uses its own slice of the weight
    /// tensors. Called once per brain tick to sample fresh actions.
    pub fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        // [N, IN] → [N, 1, IN]
        let x = obs.unsqueeze_dim::<3>(1);
        // [N, 1, IN] · [N, IN, HIDDEN] → [N, 1, HIDDEN] → [N, HIDDEN]
        let h = x.matmul(self.w1.val()).squeeze_dim::<2>(1) + self.b1.val();
        let h = relu(h);
        let h = h.unsqueeze_dim::<3>(1);
        let mu = h.matmul(self.w2.val()).squeeze_dim::<2>(1) + self.b2.val();
        // Bound the policy mean to [-1, 1] so `limb_targets` can't run
        // away during training. The PD controller in `avian_setup` then
        // scales this into a target joint angle.
        tanh(mu)
    }

    /// Rollout forward: `states [N, T, IN] → μ [N, T, OUT]`. Called
    /// inside the PPO train step to replay the buffered trajectories
    /// through the (slightly newer) policy weights for the gradient
    /// computation. Mirrors the sliding L3 brain's
    /// `forward_rollout` pattern.
    pub fn forward_rollout(&self, states: Tensor<B, 3>) -> Tensor<B, 3> {
        // states: [N, T, IN] @ w1 [N, IN, HIDDEN] = [N, T, HIDDEN]
        let h_pre = states.matmul(self.w1.val());
        let b1_b  = self.b1.val().unsqueeze_dim::<3>(1);   // [N, 1, HIDDEN]
        let h     = relu(h_pre + b1_b);
        let mu_pre = h.matmul(self.w2.val());              // [N, T, OUT]
        let b2_b   = self.b2.val().unsqueeze_dim::<3>(1);  // [N, 1, OUT]
        // Match `forward`: tanh-bound μ so the rollout replay sees the
        // same squashed distribution the sampler did. PPO ratio remains
        // consistent across collection and update.
        tanh(mu_pre + b2_b)
    }
}

/// Critic MLP — per-organism `IN → HIDDEN → 1` returning `V(s)`.
#[derive(Module, Debug)]
pub struct LimbCritic<B: Backend> {
    pub w1: Param<Tensor<B, 3>>,  // [N, IN, HIDDEN]
    pub b1: Param<Tensor<B, 2>>,  // [N, HIDDEN]
    pub w2: Param<Tensor<B, 3>>,  // [N, HIDDEN, 1]
    pub b2: Param<Tensor<B, 2>>,  // [N, 1]
}

impl<B: Backend> LimbCritic<B> {
    /// Inference forward: `obs [N, IN] → V [N, 1]`.
    pub fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = obs.unsqueeze_dim::<3>(1);
        let h = x.matmul(self.w1.val()).squeeze_dim::<2>(1) + self.b1.val();
        let h = relu(h);
        let h = h.unsqueeze_dim::<3>(1);
        let v = h.matmul(self.w2.val()).squeeze_dim::<2>(1) + self.b2.val();
        v
    }

    /// Rollout forward: `states [N, T, IN] → V [N, T, 1]`.
    pub fn forward_rollout(&self, states: Tensor<B, 3>) -> Tensor<B, 3> {
        let h_pre = states.matmul(self.w1.val());
        let b1_b  = self.b1.val().unsqueeze_dim::<3>(1);
        let h     = relu(h_pre + b1_b);
        let v_pre = h.matmul(self.w2.val());
        let b2_b  = self.b2.val().unsqueeze_dim::<3>(1);
        v_pre + b2_b
    }
}


// ── Rollout buffer ────────────────────────────────────────────────────────────

/// One step in an agent's rollout. Stored CPU-side so PPO updates can
/// be assembled into the right batched tensors without round-trips.
#[derive(Clone, Debug)]
pub struct RolloutEntry {
    pub obs:        [f32; IN],
    pub action:     [f32; OUT],
    /// Log-prob of `action` under the policy at sample time
    /// (`old_log_prob` for the PPO importance ratio).
    pub log_prob:   f32,
    /// `V(s_t)` from the critic at sample time, used by GAE.
    pub value:      f32,
    /// Reward attributed to this step at the end of the next tick (so
    /// `train` can compute returns + advantages over the buffer).
    pub reward:     f32,
    /// Episode-boundary marker (e.g. organism died, or a hard reset).
    /// For long-lived organisms this stays `false`; useful for future
    /// curriculum / episodic schemes.
    pub done:       bool,
}

impl Default for RolloutEntry {
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


// ── Slot trait + observation builder ──────────────────────────────────────────

/// Marker-component trait so the shared `apply_step` can extract a slot
/// index regardless of which per-level component (BrainSlotHerbivore1Limb,
/// BrainSlotL2Limb, BrainSlotL3Limb) the query carries.
pub trait LimbSlot {
    fn slot(&self) -> u32;
}

/// Bundle of physics-derived observation inputs gathered per organism
/// by `apply_step`'s pre-pass. Fields default to zero so partial
/// gathers (e.g. an organism missing a second limb) leave their slots
/// at their natural neutral value.
#[derive(Clone, Copy)]
pub struct LimbObsInputs {
    pub world_model:    [f32; 24],
    /// Sin/cos pairs of each limb joint's relative-rotation Euler XYZ
    /// (relative to the BASE body's rotation), for up to `MAX_LIMB_JOINTS`
    /// joints. Layout per joint `j` (= body-part index `j+1`):
    /// `[sin x, cos x, sin y, cos y, sin z, cos z]` at offset `j*6`.
    /// Absent joints stay at the zero default (a constant the net ignores).
    pub joint_sincos:   [f32; 48],   // 8 joints × 6
    /// Per-limb-joint world-frame angular velocity, 3 scalars per joint
    /// (× up to `MAX_LIMB_JOINTS`), at offset `j*3`.
    pub joint_angvel:   [f32; 24],   // 8 joints × 3
    /// Base body's orientation as `(sin, cos)` pairs of Euler-XYZ (6)
    /// + angular velocity (3) + linear velocity (3). Sin/cos rather
    /// than raw angles so the ±π wrap is continuous (matching
    /// `joint_sincos`). Layout: `[sin rx, cos rx, sin ry, cos ry,
    /// sin rz, cos rz, ωx, ωy, ωz, vx, vy, vz]`.
    pub base_pose_vel:  [f32; 12],
    /// Base up-vector in WORLD frame (`base_rot · +Y`). The
    /// singularity-free tilt/fall signal: `(0,1,0)` upright, `y≈0` on
    /// its side, `y≈−1` upside-down. `.y` equals the reward's
    /// `uprightness` term.
    pub base_up:        [f32; 3],
    /// Contact flags for [base, limb1, limb2, …, limb8]. `1.0` while
    /// touching, `0.0` otherwise. Index 0 is the base; index `i` (1..=8) is
    /// body-part `i`. Foot-ground contact is the core gait feedback signal.
    pub limb_contact:   [f32; 9],
    /// Base body's world-frame rotation, kept verbatim. Used by the
    /// world-model body-local encoder and by the reward shaper
    /// (forward-velocity projection, uprightness term). Not consumed
    /// as a network input directly — `joint_sincos` and `base_pose_vel`
    /// already carry the relevant pose information for the network.
    pub base_rot:       Quat,
    /// Base body's world-frame linear velocity. Same provenance as
    /// `base_pose_vel[6..9]` but stored as a `Vec3` so the reward
    /// shaper can project onto the heading without re-deriving.
    pub base_lin_vel:   Vec3,
    /// XZ distance from the base body to the nearest photoautotroph
    /// inside `WORLD_MODEL_RADIUS`, or `None` if no prey is in range.
    /// Used by the reward shaper to credit progress toward food.
    pub nearest_prey_dist: Option<f32>,
    /// World-frame XZ unit vector pointing from the base body toward
    /// the nearest photoautotroph, or `None` when no prey is in range
    /// (or the prey is directly underfoot). Used by the reward shaper
    /// to credit the body for turning its heading toward the target.
    pub nearest_prey_dir_xz: Option<Vec2>,
}

// Manual `Default` (the derive only supports arrays up to length 32, and
// `joint_sincos` is `[f32; 48]`). All-zero arrays / identity rotation / `None`
// — the neutral "no info" state used for organisms absent from the gather pass.
impl Default for LimbObsInputs {
    fn default() -> Self {
        Self {
            world_model:         [0.0; 24],
            joint_sincos:        [0.0; 48],
            joint_angvel:        [0.0; 24],
            base_pose_vel:       [0.0; 12],
            base_up:             [0.0; 3],
            limb_contact:        [0.0; 9],
            base_rot:            Quat::IDENTITY,
            base_lin_vel:        Vec3::ZERO,
            nearest_prey_dist:   None,
            nearest_prey_dir_xz: None,
        }
    }
}

/// Assemble the per-tick observation for one organism. Combines its
/// `Organism`-side state (energy, prev_action) with the physics-derived
/// inputs the caller gathered from Avian.
/// Pull a single f32 off a 0-D or 1-element tensor by syncing it to
/// CPU. Used by `train` to record loss/entropy scalars in
/// `training_history` for the dataset exporter. One sync per scalar
/// per training step — cheap on the scale of a full PPO update.
pub fn scalar_of<const D: usize>(t: Tensor<MyBackend, D>) -> f32 {
    let data = t.into_data();
    data.into_vec::<f32>().expect("scalar_of: f32 dtype").first().copied().unwrap_or(0.0)
}


pub fn build_observation(
    organism:     &Organism,
    prev_action:  &[f32; OUT],
    physics:      &LimbObsInputs,
    phase:        f32,
) -> [f32; IN] {
    let mut obs = [0.0_f32; IN];
    let max_energy = organism.grown_cell_count() as f32 * MAX_ENERGY_PER_CELL;
    if max_energy > 0.0 {
        obs[0] = (organism.energy / max_energy).clamp(0.0, 1.0);
    }
    obs[1..25].copy_from_slice(&physics.world_model);
    obs[25..73].copy_from_slice(&physics.joint_sincos);     // 8 joints × 6
    obs[73..97].copy_from_slice(&physics.joint_angvel);     // 8 joints × 3
    obs[97..109].copy_from_slice(&physics.base_pose_vel);   // 6 sincos + 3 angvel + 3 linvel
    obs[109..112].copy_from_slice(&physics.base_up);        // up-vector
    obs[112..121].copy_from_slice(&physics.limb_contact);   // base + 8 limbs
    obs[121..129].copy_from_slice(prev_action);             // OUT = 8

    // Nearest-prey bearing in BODY-LOCAL frame (obs[129..132]):
    // (dir_x, dir_z, dist_norm). The stored direction is world-frame;
    // rotate it through the base body's inverse yaw so it lands in the
    // same frame as the body-local world-model block. A real bearing
    // is a unit vector (length 1); when no prey is in range we emit a
    // zero-length direction (distinguishable as "no target") and a
    // maxed-out distance.
    match (physics.nearest_prey_dir_xz, physics.nearest_prey_dist) {
        (Some(dir_world), Some(dist)) => {
            let fwd = physics.base_rot * Vec3::Z;
            let len = (fwd.x * fwd.x + fwd.z * fwd.z).sqrt();
            let (sin_h, cos_h) = if len > 1e-6 { (fwd.x / len, fwd.z / len) } else { (0.0, 1.0) };
            // Inverse-yaw rotation, identical convention to
            // `world_model::encode_neighbours_body_local`.
            obs[129] = dir_world.x * cos_h - dir_world.y * sin_h;
            obs[130] = dir_world.x * sin_h + dir_world.y * cos_h;
            obs[131] = (dist / crate::world_model::WORLD_MODEL_RADIUS).clamp(0.0, 1.0);
        }
        _ => {
            obs[129] = 0.0;
            obs[130] = 0.0;
            obs[131] = 1.0;
        }
    }

    // Phase clock (obs[132..134]): sin/cos of the shared virtual-time phase
    // oscillator. NOT a motor command — the brain reads it and learns to map
    // phase → coordinated joint angles, the one rhythm aid that lets a
    // feedforward policy sustain a gait oscillation while still generating
    // every joint command itself.
    obs[132] = phase.sin();
    obs[133] = phase.cos();
    obs
}


/// Walk every limb-based body-part entity once, build a
/// `HashMap<organism_root_entity, LimbObsInputs>` populated from
/// Avian's per-body Position/Rotation/AngularVelocity/LinearVelocity
/// and the `LimbContact` flag. Joint angles are extracted as
/// `parent.rotation.inverse() * limb.rotation → Euler XYZ`, then
/// stored as `(sin, cos)` pairs per axis. World-model neighbours come
/// from the global `WorldModelGrid` keyed on the BASE body's
/// position (Avian's authoritative pose for the organism).
///
/// Two passes:
///   1. Find each organism's BASE body part (`BodyPartIndex(0)`), cache
///      its rotation + write base pose / velocity / base-contact
///      into the inputs map.
///   2. Find each limb body part (`BodyPartIndex ∈ 1..=MAX_LIMB_JOINTS`),
///      compute its rotation relative to the cached base rotation, and fill
///      its OWN per-joint `joint_sincos` / `joint_angvel` / contact slot
///      (joint `idx-1`). Every joint — each hip AND knee, each Bilateral
///      mirror half — is observed independently, matching the per-joint
///      action layout the brain controls.
pub fn gather_limb_obs_inputs(
    body_parts:  &bevy::ecs::system::Query<(
        &bevy::prelude::ChildOf,
        &crate::cell::BodyPartIndex,
        &avian3d::prelude::Position,
        &avian3d::prelude::Rotation,
        &avian3d::prelude::AngularVelocity,
        &avian3d::prelude::LinearVelocity,
        Option<&crate::avian_setup::LimbContact>,
    )>,
    world_grid:  &crate::world_model::WorldModelGrid,
) -> HashMap<Entity, LimbObsInputs> {
    let mut out: HashMap<Entity, LimbObsInputs> = HashMap::new();
    let mut base_rot: HashMap<Entity, Quat>     = HashMap::new();
    let mut base_pos: HashMap<Entity, Vec3>     = HashMap::new();

    // Pass 1: base bodies — record base pose+vel + base contact + cache
    // rotation/position for the limb pass.
    for (child_of, idx, pos, rot, ang_vel, lin_vel, contact) in body_parts.iter() {
        if idx.0 != 0 { continue; }
        let root = child_of.parent();
        let (rx, ry, rz) = rot.0.to_euler(bevy::math::EulerRot::XYZ);
        let entry = out.entry(root).or_default();
        // Orientation as sin/cos pairs (continuous across the ±π wrap).
        entry.base_pose_vel[0]  = rx.sin();
        entry.base_pose_vel[1]  = rx.cos();
        entry.base_pose_vel[2]  = ry.sin();
        entry.base_pose_vel[3]  = ry.cos();
        entry.base_pose_vel[4]  = rz.sin();
        entry.base_pose_vel[5]  = rz.cos();
        entry.base_pose_vel[6]  = ang_vel.0.x;
        entry.base_pose_vel[7]  = ang_vel.0.y;
        entry.base_pose_vel[8]  = ang_vel.0.z;
        entry.base_pose_vel[9]  = lin_vel.0.x;
        entry.base_pose_vel[10] = lin_vel.0.y;
        entry.base_pose_vel[11] = lin_vel.0.z;
        // Up-vector: where the body's +Y points in world frame.
        let up = rot.0 * Vec3::Y;
        entry.base_up = [up.x, up.y, up.z];
        entry.limb_contact[0] = if contact.is_some_and(|c| c.in_contact) { 1.0 } else { 0.0 };
        entry.base_rot     = rot.0;
        entry.base_lin_vel = lin_vel.0;
        base_rot.insert(root, rot.0);
        base_pos.insert(root, pos.0);
    }

    // Pass 2: limbs (idx 1..=MAX_LIMB_JOINTS). Each limb body part fills its
    // OWN joint slot (joint = idx-1), measured relative to the base rotation.
    // No pair/half collapsing — every joint is observed and controlled
    // independently, so an alternating gait can emerge.
    for (child_of, idx, _pos, rot, ang_vel, _lin_vel, contact) in body_parts.iter() {
        if idx.0 == 0 || idx.0 > OUT { continue; }
        let root = child_of.parent();
        let Some(base_r) = base_rot.get(&root) else { continue };
        let j = idx.0 - 1;   // 0-based joint slot
        let rel = base_r.inverse() * rot.0;
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
        entry.joint_angvel[av_base    ] = ang_vel.0.x;
        entry.joint_angvel[av_base + 1] = ang_vel.0.y;
        entry.joint_angvel[av_base + 2] = ang_vel.0.z;
        // Contact slot: base is 0, limb idx maps to slot idx (1..=8).
        entry.limb_contact[idx.0] = if contact.is_some_and(|c| c.in_contact) { 1.0 } else { 0.0 };
    }

    // World-model neighbours, keyed on the base body's Avian position
    // and rotated into the base body's local frame so the brain sees
    // "in front of me / to my left" instead of "world-X / world-Z".
    // The nearest photoautotroph distance is also captured here for
    // the reward shaper (progress toward food).
    for (root, entry) in out.iter_mut() {
        if let Some(pos) = base_pos.get(root) {
            let neighbours = crate::world_model::collect_neighbours(world_grid, *pos);
            crate::world_model::encode_neighbours_body_local(
                &neighbours, entry.base_rot, &mut entry.world_model,
            );
            // Nearest photoautotroph: capture both the XZ distance
            // (K_PROGRESS reward) and the world-frame XZ unit direction
            // toward it (K_HEADING reward — see the reward shaper).
            if let Some((prey_pos, dist, _)) = crate::world_model::nearest_prey(world_grid, *pos) {
                entry.nearest_prey_dist = Some(dist);
                let rel = prey_pos - *pos;
                let rel_xz = Vec2::new(rel.x, rel.z);
                entry.nearest_prey_dir_xz =
                    (rel_xz.length() > 1e-6).then(|| rel_xz.normalize());
            }
        }
    }

    out
}


// ── Save/load (.colony persistence) ───────────────────────────────────────────

/// Brain-weight payload attached to an organism at `.colony` load time
/// so the limb-pool's `assign_brains_*` system can restore the
/// learned weights instead of re-initialising from scratch. One per
/// organism — the assign system consumes the component on read and
/// removes it. Carries both actor and critic weights so a saved-and-
/// loaded organism resumes training exactly where it left off
/// (modulo Adam optimiser moments, which we deliberately don't
/// persist — they re-warm quickly).
#[derive(bevy::ecs::component::Component, Clone, Debug)]
pub struct BrainRestoreLimb {
    pub actor_w1:      Vec<f32>,   // [IN * HIDDEN]
    pub actor_b1:      Vec<f32>,   // [HIDDEN]
    pub actor_w2:      Vec<f32>,   // [HIDDEN * OUT]
    pub actor_b2:      Vec<f32>,   // [OUT]
    pub actor_log_std: Vec<f32>,   // [OUT]
    pub critic_w1:     Vec<f32>,   // [IN * HIDDEN]
    pub critic_b1:     Vec<f32>,   // [HIDDEN]
    pub critic_w2:     Vec<f32>,   // [HIDDEN]
    pub critic_b2:     Vec<f32>,   // [1]
}

/// Full read-only snapshot of one limb pool's GPU weight state, taken
/// once per save before iterating organisms. Mirrors the sliding-pool
/// `PoolSnapshot` pattern: one GPU→CPU sync per tensor (10 total per
/// pool), then per-organism extraction is pure CPU indexing.
pub struct LimbPoolSnapshot {
    pub actor_w1:      Vec<f32>,
    pub actor_b1:      Vec<f32>,
    pub actor_w2:      Vec<f32>,
    pub actor_b2:      Vec<f32>,
    pub actor_log_std: Vec<f32>,
    pub critic_w1:     Vec<f32>,
    pub critic_b1:     Vec<f32>,
    pub critic_w2:     Vec<f32>,
    pub critic_b2:     Vec<f32>,
    pub map:           HashMap<Entity, u32>,
}

impl LimbPoolSnapshot {
    /// Slice the appropriate `[IN * HIDDEN]` etc. range for one slot
    /// and assemble it into a `BrainRestoreLimb` ready to serialise.
    pub fn extract(&self, e: Entity) -> Option<BrainRestoreLimb> {
        let s = *self.map.get(&e)? as usize;
        Some(BrainRestoreLimb {
            actor_w1:      self.actor_w1[s * IN * HIDDEN .. (s + 1) * IN * HIDDEN].to_vec(),
            actor_b1:      self.actor_b1[s * HIDDEN     .. (s + 1) * HIDDEN]     .to_vec(),
            actor_w2:      self.actor_w2[s * HIDDEN * OUT .. (s + 1) * HIDDEN * OUT].to_vec(),
            actor_b2:      self.actor_b2[s * OUT        .. (s + 1) * OUT]        .to_vec(),
            actor_log_std: self.actor_log_std[s * OUT   .. (s + 1) * OUT]        .to_vec(),
            critic_w1:     self.critic_w1[s * IN * HIDDEN .. (s + 1) * IN * HIDDEN].to_vec(),
            critic_b1:     self.critic_b1[s * HIDDEN    .. (s + 1) * HIDDEN]     .to_vec(),
            critic_w2:     self.critic_w2[s * HIDDEN    .. (s + 1) * HIDDEN]     .to_vec(),
            critic_b2:     self.critic_b2[s ..= s].to_vec(),  // [1] per slot
        })
    }
}

impl BrainPoolLimb {
    /// Pull every weight tensor off the GPU into one shared snapshot.
    /// One sync per tensor; the per-organism `extract` step is pure CPU
    /// indexing.
    pub fn snapshot(&self) -> LimbPoolSnapshot {
        LimbPoolSnapshot {
            actor_w1:      self.actor.w1.val().clone().into_data().into_vec::<f32>().expect("actor.w1"),
            actor_b1:      self.actor.b1.val().clone().into_data().into_vec::<f32>().expect("actor.b1"),
            actor_w2:      self.actor.w2.val().clone().into_data().into_vec::<f32>().expect("actor.w2"),
            actor_b2:      self.actor.b2.val().clone().into_data().into_vec::<f32>().expect("actor.b2"),
            actor_log_std: self.actor.log_std.val().clone().into_data().into_vec::<f32>().expect("actor.log_std"),
            critic_w1:     self.critic.w1.val().clone().into_data().into_vec::<f32>().expect("critic.w1"),
            critic_b1:     self.critic.b1.val().clone().into_data().into_vec::<f32>().expect("critic.b1"),
            critic_w2:     self.critic.w2.val().clone().into_data().into_vec::<f32>().expect("critic.w2"),
            critic_b2:     self.critic.b2.val().clone().into_data().into_vec::<f32>().expect("critic.b2"),
            map:           self.map.clone(),
        }
    }

    /// Write a single slot's weights into the (batched) Param tensors.
    /// Called by `assign_brains_*_limb` when the new entity carries a
    /// `BrainRestoreLimb` payload.
    pub fn restore_slot(&mut self, slot: u32, r: &BrainRestoreLimb) {
        use burn::tensor::Tensor;
        // Architecture guard. A saved payload's tensor lengths must
        // match the CURRENT IN/HIDDEN/OUT or `TensorData::new` below
        // would panic on the shape/length mismatch. When they don't
        // match (e.g. a save from before IN was bumped 61 → 64), skip
        // restoration entirely and leave the slot's fresh-init weights
        // — the organism trains from scratch rather than crashing the
        // run. Only `actor_w1` / `critic_w1` depend on IN, so checking
        // every tensor's expected length covers all dimension changes.
        let shapes_ok = r.actor_w1.len()      == IN * HIDDEN
            && r.actor_b1.len()      == HIDDEN
            && r.actor_w2.len()      == HIDDEN * OUT
            && r.actor_b2.len()      == OUT
            && r.actor_log_std.len() == OUT
            && r.critic_w1.len()     == IN * HIDDEN
            && r.critic_b1.len()     == HIDDEN
            && r.critic_w2.len()     == HIDDEN
            && r.critic_b2.len()     == 1;
        if !shapes_ok {
            warn!(
                "limb brain restore skipped: payload shapes don't match current \
                 architecture (IN={IN}, HIDDEN={HIDDEN}, OUT={OUT}); slot keeps \
                 fresh-init weights"
            );
            return;
        }
        let s = slot as usize;
        let device = &self.device;
        // Write each saved tensor into row `s` of the batched Param via
        // `Param::map(|t| t.slice_assign(...))`. We must use `map` and
        // NOT `Param::from_tensor(val().slice_assign(...))`: on the
        // `Autodiff` backend the result of `slice_assign` is a NON-LEAF
        // tensor (it's the output of an op), and `Param::from_tensor`
        // requires a leaf — feeding it a non-leaf panics with
        // "Can't convert a non leaf tensor into a tracked tensor"
        // (the load-time crash this fixes). `Param::map` rebuilds the
        // parameter as a proper trainable leaf. Mirrors the working
        // pattern in the sliding pools' `restore_slot`.

        // Actor weights.
        let t = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(r.actor_w1.clone(), [1, IN, HIDDEN]), device,
        );
        self.actor.w1 = self.actor.w1.clone().map(|x| x.slice_assign([s..s + 1, 0..IN, 0..HIDDEN], t));

        let t = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(r.actor_b1.clone(), [1, HIDDEN]), device,
        );
        self.actor.b1 = self.actor.b1.clone().map(|x| x.slice_assign([s..s + 1, 0..HIDDEN], t));

        let t = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(r.actor_w2.clone(), [1, HIDDEN, OUT]), device,
        );
        self.actor.w2 = self.actor.w2.clone().map(|x| x.slice_assign([s..s + 1, 0..HIDDEN, 0..OUT], t));

        let t = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(r.actor_b2.clone(), [1, OUT]), device,
        );
        self.actor.b2 = self.actor.b2.clone().map(|x| x.slice_assign([s..s + 1, 0..OUT], t));

        let t = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(r.actor_log_std.clone(), [1, OUT]), device,
        );
        self.actor.log_std = self.actor.log_std.clone().map(|x| x.slice_assign([s..s + 1, 0..OUT], t));

        // Critic weights.
        let t = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(r.critic_w1.clone(), [1, IN, HIDDEN]), device,
        );
        self.critic.w1 = self.critic.w1.clone().map(|x| x.slice_assign([s..s + 1, 0..IN, 0..HIDDEN], t));

        let t = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(r.critic_b1.clone(), [1, HIDDEN]), device,
        );
        self.critic.b1 = self.critic.b1.clone().map(|x| x.slice_assign([s..s + 1, 0..HIDDEN], t));

        let t = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(r.critic_w2.clone(), [1, HIDDEN, 1]), device,
        );
        self.critic.w2 = self.critic.w2.clone().map(|x| x.slice_assign([s..s + 1, 0..HIDDEN, 0..1], t));

        let t = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(r.critic_b2.clone(), [1, 1]), device,
        );
        self.critic.b2 = self.critic.b2.clone().map(|x| x.slice_assign([s..s + 1, 0..1], t));
    }

    /// Copy slot `src`'s actor + critic weight rows into slot `dst` — a fresh
    /// occupant INHERITS a trained brain ("instinct") instead of starting from
    /// the untrained warm-start. Used by the `assign_*_limb` systems so a newly
    /// spawned (or reproduced) limb organism joins an already-trained population
    /// competent — without it, a freshly placed Crawler just collapses and does
    /// nothing until it learns from scratch. GPU-side row copy, no CPU
    /// round-trip. Mirrors the sliding pools' inheritance path.
    pub fn inherit_slot(&mut self, dst: u32, src: u32) {
        use burn::tensor::Tensor as _;
        if dst == src { return; }
        let (d, s) = (dst as usize, src as usize);
        self.actor.w1 = self.actor.w1.clone().map(|t| { let r = t.clone().slice([s..s+1, 0..IN, 0..HIDDEN]); t.slice_assign([d..d+1, 0..IN, 0..HIDDEN], r) });
        self.actor.b1 = self.actor.b1.clone().map(|t| { let r = t.clone().slice([s..s+1, 0..HIDDEN]);       t.slice_assign([d..d+1, 0..HIDDEN], r) });
        self.actor.w2 = self.actor.w2.clone().map(|t| { let r = t.clone().slice([s..s+1, 0..HIDDEN, 0..OUT]); t.slice_assign([d..d+1, 0..HIDDEN, 0..OUT], r) });
        self.actor.b2 = self.actor.b2.clone().map(|t| { let r = t.clone().slice([s..s+1, 0..OUT]);          t.slice_assign([d..d+1, 0..OUT], r) });
        self.actor.log_std = self.actor.log_std.clone().map(|t| { let r = t.clone().slice([s..s+1, 0..OUT]); t.slice_assign([d..d+1, 0..OUT], r) });
        self.critic.w1 = self.critic.w1.clone().map(|t| { let r = t.clone().slice([s..s+1, 0..IN, 0..HIDDEN]); t.slice_assign([d..d+1, 0..IN, 0..HIDDEN], r) });
        self.critic.b1 = self.critic.b1.clone().map(|t| { let r = t.clone().slice([s..s+1, 0..HIDDEN]);        t.slice_assign([d..d+1, 0..HIDDEN], r) });
        self.critic.w2 = self.critic.w2.clone().map(|t| { let r = t.clone().slice([s..s+1, 0..HIDDEN, 0..1]);   t.slice_assign([d..d+1, 0..HIDDEN, 0..1], r) });
        self.critic.b2 = self.critic.b2.clone().map(|t| { let r = t.clone().slice([s..s+1, 0..1]);             t.slice_assign([d..d+1, 0..1], r) });
    }
}

/// Serialise one `BrainRestoreLimb` to a byte buffer. Layout: a
/// `u32` length-prefix per vector, then the f32 little-endian bytes.
pub fn encode_brain_restore_limb(buf: &mut Vec<u8>, r: &BrainRestoreLimb) {
    fn put_vec(buf: &mut Vec<u8>, v: &[f32]) {
        buf.extend_from_slice(&(v.len() as u32).to_le_bytes());
        for x in v { buf.extend_from_slice(&x.to_le_bytes()); }
    }
    put_vec(buf, &r.actor_w1);
    put_vec(buf, &r.actor_b1);
    put_vec(buf, &r.actor_w2);
    put_vec(buf, &r.actor_b2);
    put_vec(buf, &r.actor_log_std);
    put_vec(buf, &r.critic_w1);
    put_vec(buf, &r.critic_b1);
    put_vec(buf, &r.critic_w2);
    put_vec(buf, &r.critic_b2);
}

/// Deserialise one `BrainRestoreLimb`, advancing `c` past the read
/// bytes. Returns `Err` on truncation or malformed length prefixes.
pub fn decode_brain_restore_limb(bytes: &[u8], c: &mut usize) -> std::io::Result<BrainRestoreLimb> {
    fn read_vec(bytes: &[u8], c: &mut usize) -> std::io::Result<Vec<f32>> {
        if bytes.len() < *c + 4 {
            return Err(std::io::Error::other("limb brain payload truncated (len prefix)"));
        }
        let n = u32::from_le_bytes(bytes[*c..*c + 4].try_into().unwrap()) as usize;
        *c += 4;
        if bytes.len() < *c + n * 4 {
            return Err(std::io::Error::other("limb brain payload truncated (vec body)"));
        }
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(f32::from_le_bytes(bytes[*c..*c + 4].try_into().unwrap()));
            *c += 4;
        }
        Ok(v)
    }
    Ok(BrainRestoreLimb {
        actor_w1:      read_vec(bytes, c)?,
        actor_b1:      read_vec(bytes, c)?,
        actor_w2:      read_vec(bytes, c)?,
        actor_b2:      read_vec(bytes, c)?,
        actor_log_std: read_vec(bytes, c)?,
        critic_w1:     read_vec(bytes, c)?,
        critic_b1:     read_vec(bytes, c)?,
        critic_w2:     read_vec(bytes, c)?,
        critic_b2:     read_vec(bytes, c)?,
    })
}


// ── Optimiser trait objects ───────────────────────────────────────────────────

/// Type-erases the burn `Optimizer<LimbActor<MyBackend>, MyBackend>`
/// implementation so the pool can store it without leaking the concrete
/// generic (mirrors the `BrainOptL3` pattern in the sliding pools).
pub trait BrainOptActor {
    fn step(
        &mut self,
        lr: f64,
        m:  LimbActor<MyBackend>,
        g:  GradientsParams,
    ) -> LimbActor<MyBackend>;
}

impl<O: Optimizer<LimbActor<MyBackend>, MyBackend>> BrainOptActor for O {
    fn step(
        &mut self,
        lr: f64,
        m:  LimbActor<MyBackend>,
        g:  GradientsParams,
    ) -> LimbActor<MyBackend> {
        Optimizer::step(self, lr, m, g)
    }
}

pub trait BrainOptCritic {
    fn step(
        &mut self,
        lr: f64,
        m:  LimbCritic<MyBackend>,
        g:  GradientsParams,
    ) -> LimbCritic<MyBackend>;
}

impl<O: Optimizer<LimbCritic<MyBackend>, MyBackend>> BrainOptCritic for O {
    fn step(
        &mut self,
        lr: f64,
        m:  LimbCritic<MyBackend>,
        g:  GradientsParams,
    ) -> LimbCritic<MyBackend> {
        Optimizer::step(self, lr, m, g)
    }
}


// ── Pool ──────────────────────────────────────────────────────────────────────

/// Shared engine wrapped by each per-level pool resource. Holds the
/// actor + critic networks, their optimisers, the per-slot rollout
/// buffers, the free-slot list, and the entity↔slot map.
///
/// The per-level pool resources (e.g. `BrainPoolHerbivore1Limb`) own
/// one of these and forward every operation to it.
pub struct BrainPoolLimb {
    pub n:      usize,
    pub device: CudaDevice,
    pub actor:  LimbActor<MyBackend>,
    pub critic: LimbCritic<MyBackend>,
    // Optimisers are intentionally NOT stored here in Phase 3. The
    // sliding pools dispatch through a per-pool `BrainOpt*` trait
    // object (`Box<dyn BrainOptL3>`) so the concrete `Optimizer`
    // generics stay local to the train function. Phase 4 will do the
    // same for the limb pools when wiring the PPO gradient code; for
    // now the structure compiles with no optimiser state at all,
    // since `train_one` is a stub.
    /// Per-slot rollout buffers (length ROLLOUT_LEN once full).
    pub rollouts: Vec<VecDeque<RolloutEntry>>,
    /// Slot indices currently free for new enrolments. Pop from the
    /// end (LIFO) — most-recently-freed slot is reused first.
    pub free: Vec<u32>,
    /// Entity → slot map for save/load + diagnostics.
    pub map: HashMap<Entity, u32>,
    /// Cached `prev_action` for the recurrence input (last 6 of the
    /// IN=55 observation).
    pub prev_action: Vec<[f32; OUT]>,
    /// Per-slot `Organism::predations` counter as of the last apply
    /// tick. Used to compute the per-tick eat reward as a delta. Reset
    /// to 0 in `enrol`.
    pub prev_predations: Vec<u8>,
    /// Per-slot `Organism::reproductions` counter as of the last apply
    /// tick. Reset to 0 in `enrol`.
    pub prev_reproductions: Vec<u8>,
    /// Per-slot XZ distance to the nearest photoautotroph as of the
    /// previous apply tick. Used to compute the K_PROGRESS reward as
    /// `max(0, prev − curr)`. `None` when the last tick had no prey in
    /// range (so first-frame and post-respawn ticks don't fabricate
    /// progress).
    pub prev_nearest_prey_dist: Vec<Option<f32>>,
    /// Per-slot heading-alignment toward the nearest prey as of the
    /// previous apply tick, used to compute the rectified K_HEADING
    /// reward `max(0, align_now − align_prev)`. `None` when the last
    /// tick had no prey direction (so first-frame and prey-just-
    /// appeared ticks don't fabricate a turn reward).
    pub prev_heading_alignment: Vec<Option<f32>>,
    /// Per-slot limb-contact flags `[base, limb1, …, limb8]` as of the
    /// previous apply tick. Diffed against the current flags to count
    /// stepping transitions for the `K_STEP` reward.
    pub prev_limb_contact: Vec<[f32; 9]>,
    /// Adam optimiser for the actor's parameters.
    pub opt_actor:  Box<dyn BrainOptActor>,
    /// Adam optimiser for the critic's parameters.
    pub opt_critic: Box<dyn BrainOptCritic>,
    /// Apply-tick counter since the last PPO update. When this hits
    /// `ROLLOUT_LEN` the pool's `train()` runs and the counter resets.
    pub ticks_since_train: usize,
    /// Monotonic training-step counter. Incremented each time `train()`
    /// completes a PPO update. Used as the `step` field on
    /// `LimbTrainingStep` entries for the dataset exporter.
    pub training_step: u64,
    /// Ring buffer of recent training steps for offline analysis.
    /// Capped at `LIMB_TRAINING_HISTORY_CAP` so memory stays bounded.
    pub training_history: VecDeque<LimbTrainingStep>,
}

/// One PPO update's logged scalars. Mirrors the sliding pool's
/// `TrainingStep` schema field-for-field so the same `training_stats`
/// CSV layout can serve both. Channels that don't apply to the limb
/// pool (`supervised_loss`) are zeroed.
#[derive(Clone, Debug)]
pub struct LimbTrainingStep {
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

/// Cap on the in-memory training-step ring buffer.
pub const LIMB_TRAINING_HISTORY_CAP: usize = 1024;

impl BrainPoolLimb {
    /// Allocate a fresh pool with batch dim `n`. Biases zero; per-organism
    /// `log_std` initialised to `LOG_STD_INIT`.
    ///
    /// Weights use **Xavier** init, NOT the sliding pools' flat `Uniform ±0.5`.
    /// With `IN = 134` a flat `±0.5` makes the pre-tanh activation enormous
    /// (~134 terms × 0.5) → the actor's `μ` SATURATES at ±1 from the very first
    /// step → every joint target is pinned at ±`LIMB_SWING_LIMIT` against its
    /// mechanical stop → the legs are FROZEN at their extremes and never
    /// oscillate, so no stroke, no propulsion, nothing for RL to reward
    /// (confirmed in data: body `base_ang_vel ≈ 0` despite σ=0.5 exploration —
    /// the legs weren't even moving). Fan-in-scaled Xavier keeps the initial
    /// output near 0 (legs near rest), so exploration noise actually moves the
    /// limbs and the policy can discover propulsive strokes. (The small sliding
    /// nets, `IN = 31`, didn't saturate at ±0.5, which is why they were fine.)
    pub fn new(n: usize, device: CudaDevice) -> Self {
        // ~1/sqrt(fan_in) for fan_in≈IN=134 → ±0.086. Small, fan-scaled, and
        // predictable regardless of how Burn infers fan-in on the batched
        // [N, IN, HIDDEN] weight. Keeps initial activations unit-scale so `μ`
        // does NOT saturate.
        let w = Initializer::Uniform { min: -0.086, max: 0.086 };
        let z = Initializer::Zeros;
        let log_std_init = Initializer::Constant { value: LOG_STD_INIT as f64 };

        // ── Actor OSCILLATORY WARM-START ───────────────────────────────────
        // Pure per-tick Gaussian exploration almost never stumbles onto a
        // COHERENT periodic propulsive stroke (data: legs jitter incoherently →
        // net-zero force → body never moves → flat reward → no gait learned).
        // So we INITIALISE the actor so the phase-clock input (obs[132]=sin,
        // obs[133]=cos of the gait phase) drives each joint's target as
        // `μ_k ≈ A·sin(phase + φ_k)` — i.e. the legs rhythmically stroke out of
        // the box, with a per-joint phase offset so they don't all bounce in
        // unison. This is a TRAINABLE init, NOT a CPG: the weights are ordinary
        // policy parameters, so PPO is free to reshape, redirect, or abandon the
        // oscillation as the reward dictates — locomotion is still learned and
        // emergent, just bootstrapped near a rhythmic prior (standard practice
        // for learned legged gaits). Mechanism: two hidden units are biased
        // into relu's linear region (`b1 = WARMSTART_BIAS`) and fed only the
        // sin/cos phase inputs (gain `WARMSTART_PHASE_GAIN`), so they carry
        // `C ± g·sin` / `C ± g·cos`; the output layer recombines them as
        // `A·sin(phase+φ_k)` via `cos(φ_k)·sin + sin(φ_k)·cos`, and `b2` cancels
        // the resulting constant. Every other weight stays at the small random
        // init so the rest of the policy (steering toward prey, posture, etc.)
        // learns normally.
        const WARMSTART_PHASE_GAIN: f32 = 2.0;  // g: sin/cos input → carrier hidden unit
        const WARMSTART_BIAS:       f32 = 3.0;  // C: keeps carrier units in relu's linear region (C>g)
        // a: output amplitude gain (final swing ≈ a·g·sin). Cut 0.25 → 0.10
        // (2026-06-03) so the legs mostly HOLD a planted weight-bearing stance
        // and only gently modulate — a ±37° swing was lifting the feet so far
        // each cycle that the light body dropped onto its belly (unnatural,
        // penetrating). A small swing keeps the feet planted (natural standing
        // posture); the policy can grow the stride from there if rewarded.
        const WARMSTART_AMP:        f32 = 0.25;
        use rand::RngExt as _;
        let mut rng = rand::rng();
        let rand_small = |rng: &mut rand::rngs::ThreadRng| rng.random_range(-0.086_f32..0.086);
        let phi = |k: usize| (k as f32) * std::f32::consts::FRAC_PI_2; // per-joint phase offset

        let mut w1v = vec![0.0_f32; n * IN * HIDDEN];
        for v in w1v.iter_mut() { *v = rand_small(&mut rng); }
        let mut b1v = vec![0.0_f32; n * HIDDEN];
        let mut w2v = vec![0.0_f32; n * HIDDEN * OUT];
        for v in w2v.iter_mut() { *v = rand_small(&mut rng); }
        let mut b2v = vec![0.0_f32; n * OUT];
        for nn in 0..n {
            // Hidden units 0 (sin carrier) and 1 (cos carrier): clear their
            // input columns, then feed ONLY the phase inputs.
            for inp in 0..IN {
                w1v[nn * IN * HIDDEN + inp * HIDDEN + 0] = 0.0;
                w1v[nn * IN * HIDDEN + inp * HIDDEN + 1] = 0.0;
            }
            w1v[nn * IN * HIDDEN + 132 * HIDDEN + 0] = WARMSTART_PHASE_GAIN; // sin → h0
            w1v[nn * IN * HIDDEN + 133 * HIDDEN + 1] = WARMSTART_PHASE_GAIN; // cos → h1
            b1v[nn * HIDDEN + 0] = WARMSTART_BIAS;
            b1v[nn * HIDDEN + 1] = WARMSTART_BIAS;
            for k in 0..OUT {
                let (s, c) = (phi(k).sin(), phi(k).cos());
                w2v[nn * HIDDEN * OUT + 0 * OUT + k] = WARMSTART_AMP * c; // h0 (sin) → out
                w2v[nn * HIDDEN * OUT + 1 * OUT + k] = WARMSTART_AMP * s; // h1 (cos) → out
                b2v[nn * OUT + k] = -WARMSTART_BIAS * WARMSTART_AMP * (c + s); // cancel constant
            }
        }
        let actor = LimbActor::<MyBackend> {
            w1:      Param::from_tensor(Tensor::<MyBackend, 3>::from_data(TensorData::new(w1v, [n, IN, HIDDEN]), &device)),
            b1:      Param::from_tensor(Tensor::<MyBackend, 2>::from_data(TensorData::new(b1v, [n, HIDDEN]),     &device)),
            w2:      Param::from_tensor(Tensor::<MyBackend, 3>::from_data(TensorData::new(w2v, [n, HIDDEN, OUT]), &device)),
            b2:      Param::from_tensor(Tensor::<MyBackend, 2>::from_data(TensorData::new(b2v, [n, OUT]),        &device)),
            log_std: log_std_init.init([n, OUT], &device),
        };
        let critic = LimbCritic::<MyBackend> {
            w1: w.init([n, IN, HIDDEN], &device),
            b1: z.init([n, HIDDEN],     &device),
            w2: w.init([n, HIDDEN, 1],  &device),
            b2: z.init([n, 1],          &device),
        };
        let mut rollouts = Vec::with_capacity(n);
        for _ in 0..n { rollouts.push(VecDeque::with_capacity(ROLLOUT_LEN)); }
        let free: Vec<u32> = (0..n as u32).rev().collect();
        let opt_actor:  Box<dyn BrainOptActor>  = Box::new(AdamConfig::new().init());
        let opt_critic: Box<dyn BrainOptCritic> = Box::new(AdamConfig::new().init());
        Self {
            n,
            device,
            actor,
            critic,
            rollouts,
            free,
            map: HashMap::new(),
            prev_action:        vec![[0.0; OUT]; n],
            prev_predations:    vec![0u8; n],
            prev_reproductions: vec![0u8; n],
            prev_nearest_prey_dist: vec![None; n],
            prev_heading_alignment: vec![None; n],
            prev_limb_contact:      vec![[0.0; 9]; n],
            opt_actor,
            opt_critic,
            ticks_since_train: 0,
            training_step: 0,
            training_history: VecDeque::with_capacity(LIMB_TRAINING_HISTORY_CAP),
        }
    }

    /// Read-only accessor for the dataset exporter.
    pub fn training_history(&self) -> &VecDeque<LimbTrainingStep> {
        &self.training_history
    }

    /// Pop a fresh slot for an organism. `None` once the pool is full.
    pub fn enrol(&mut self, e: Entity) -> Option<u32> {
        let s = self.free.pop()?;
        self.map.insert(e, s);
        self.rollouts[s as usize].clear();
        self.prev_action[s as usize] = [0.0; OUT];
        self.prev_predations[s as usize] = 0;
        self.prev_reproductions[s as usize] = 0;
        self.prev_nearest_prey_dist[s as usize] = None;
        self.prev_heading_alignment[s as usize] = None;
        self.prev_limb_contact[s as usize] = [0.0; 9];
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

    /// Compute Generalized Advantage Estimation advantages and returns
    /// for one agent's filled rollout buffer.
    ///
    /// `values_tail` is `V(s_T)` — the critic's estimate for the state
    /// right after the buffer's last action (used as the bootstrap when
    /// the rollout isn't terminated by a `done=true` flag).
    ///
    /// Returns `(advantages, returns)` aligned with `buf`'s order.
    pub fn compute_gae(
        buf:         &VecDeque<RolloutEntry>,
        values_tail: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let t_len = buf.len();
        let mut advantages = vec![0.0_f32; t_len];
        let mut last_gae   = 0.0_f32;
        let mut next_value = values_tail;
        let mut next_nontermi = 1.0_f32;
        // Walk the buffer in reverse so the recurrence accumulates
        // future-into-past, the standard GAE direction.
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

    /// Per-tick apply step shared by all three limb-based brain pools.
    ///
    /// For each `(entity, &mut Organism, slot_u32)` tuple:
    ///   * Builds an observation vector (Phase 4 fills `energy_norm` +
    ///     `prev_action`; the remaining 48 dims are placeholder zeros
    ///     until physics observation wiring lands in a follow-up).
    ///   * Runs ONE batched actor forward pass for every active slot.
    ///   * Samples an action `μ + ε * exp(LOG_STD_INIT)` per slot,
    ///     using `gaussian_noise` for `ε`.
    ///   * Writes the action to `Organism::limb_targets`.
    ///   * Appends a `RolloutEntry` (reward = 0 placeholder; value = 0
    ///     until the critic forward is wired).
    ///   * When a slot's rollout reaches `ROLLOUT_LEN`, calls
    ///     `train_one` (currently stubbed — see Phase 4-bis) and clears
    ///     the buffer.
    pub fn apply_step<'w, 's, S: Component + Copy + LimbSlot>(
        &mut self,
        mut organisms:     Query<(Entity, &mut Organism, &S)>,
        obs_inputs:        &HashMap<Entity, LimbObsInputs>,
        virtual_time_secs: f32,
    ) {
        // Shared phase-clock value fed to every brain this tick (sin/cos
        // computed in `build_observation`). Derived from virtual time so it
        // is invariant to TimeSpeed / frame rate, matching the brain cadence.
        let phase = virtual_time_secs * std::f32::consts::TAU * GAIT_FREQUENCY_HZ;

        // ── 1. Collect active slots + observations. ─────────────────
        let mut input_buf: Vec<f32> = vec![0.0; self.n * IN];
        let mut active: Vec<(Entity, u32)> = Vec::new();
        let default_phys = LimbObsInputs::default();
        for (e, organism, slot) in organisms.iter() {
            let s = slot.slot() as usize;
            if s >= self.n { continue; }
            let prev_action = self.prev_action[s];
            let phys = obs_inputs.get(&e).unwrap_or(&default_phys);
            let obs = build_observation(&organism, &prev_action, phys, phase);
            let base = s * IN;
            input_buf[base..base + IN].copy_from_slice(&obs);
            active.push((e, slot.slot()));
        }
        if active.is_empty() { return; }

        // ── 2. Batched actor + critic forward. ──────────────────────
        let obs_t = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(input_buf.clone(), [self.n, IN]),
            &self.device,
        );
        let mu_t  = self.actor.forward(obs_t.clone());
        let v_t   = self.critic.forward(obs_t);
        let mu_buf = mu_t.into_data().into_vec::<f32>().expect("actor forward");
        let v_buf  = v_t.into_data().into_vec::<f32>().expect("critic forward");
        let sigma  = (LOG_STD_INIT as f32).exp();

        // ── 3. Sample + apply per slot. ─────────────────────────────
        let mut rng = rand::rng();
        // We can't borrow self immutably (rollouts) while iterating an
        // active list and also borrow it mutably below; collect actions
        // first, then mutate.
        let mut actions: Vec<(Entity, u32, [f32; OUT], f32)> = Vec::with_capacity(active.len());
        for &(e, s) in &active {
            let base = (s as usize) * OUT;
            let mu = &mu_buf[base..base + OUT];
            let mut action = [0.0_f32; OUT];
            let mut log_prob = 0.0_f32;
            for i in 0..OUT {
                let eps = gaussian_noise(&mut rng);
                // μ is tanh-bounded to [-1, 1]; clamp the noisy sample to
                // the same range so the PD controller can't see runaway
                // commands. log_prob is still computed against the
                // pre-clamp Gaussian — the PPO ratio's clip handles the
                // small bias this introduces.
                action[i] = (mu[i] + eps * sigma).clamp(-1.0, 1.0);
                let diff = (action[i] - mu[i]) / sigma;
                log_prob += -0.5 * diff * diff
                            - sigma.ln()
                            - 0.5 * (2.0 * std::f32::consts::PI).ln();
            }
            actions.push((e, s, action, log_prob));
        }

        // ── 4. Write actions, compute reward, log rollout entries. ──
        for (e, s, action, log_prob) in actions {
            let Ok((_, mut organism, _)) = organisms.get_mut(e) else { continue };
            organism.limb_targets = action;

            // Sparse event rewards (same shape as the sliding pools).
            let su = s as usize;
            let pred_delta  = organism.predations    .saturating_sub(self.prev_predations[su])    as f32;
            let repro_delta = organism.reproductions .saturating_sub(self.prev_reproductions[su]) as f32;
            let event_reward = K_EAT * pred_delta + K_REPRO * repro_delta;

            // Dense locomotion-intrinsic rewards. Physics inputs are
            // already gathered for this organism; if absent (organism
            // skipped the gather pass for some reason) the defaults
            // collapse every term to zero so the event reward stands
            // alone.
            let phys = obs_inputs.get(&e).unwrap_or(&default_phys);
            let lin_xz = Vec2::new(phys.base_lin_vel.x, phys.base_lin_vel.z);
            let speed  = lin_xz.length();
            // Forward velocity: signed projection of XZ velocity onto the
            // body's facing direction (`base_rot · +Z`). Rewards directed
            // travel, not undirected spin (which nets ~0 here). See the
            // `K_FWD` note above.
            let fwd = phys.base_rot * Vec3::Z;
            let heading_xz = Vec2::new(fwd.x, fwd.z).normalize_or_zero();
            let forward_speed = lin_xz.dot(heading_xz);
            // Motion gate ∈ [0, 1]: 0 when stationary, 1 once speed ≥
            // IDLE_THRESH. Used to suppress the uprightness alive-bonus
            // when the brain isn't actually moving and to scale the
            // idle penalty inversely. Uses speed MAGNITUDE (not forward
            // velocity) — "is it moving at all" is the right gate for
            // the liveness terms, independent of direction.
            let motion_gate = (speed / IDLE_THRESH).clamp(0.0, 1.0);
            // Uprightness: world-Y component of the body's local +Y axis.
            let uprightness = (phys.base_rot * Vec3::Y).y;

            // Progress-toward-prey: positive credit for closing XZ
            // distance to the nearest photoautotroph since the last
            // tick. Only fires when prey was in range on BOTH ticks
            // (so first-frame and just-respawned organisms don't
            // fabricate progress from None → Some).
            let progress = match (self.prev_nearest_prey_dist[su], phys.nearest_prey_dist) {
                (Some(prev), Some(curr)) => (prev - curr).max(0.0),
                _                        => 0.0,
            };
            self.prev_nearest_prey_dist[su] = phys.nearest_prey_dist;

            // Heading-alignment-toward-prey: rectified improvement in
            // how directly the body faces the nearest prey, tick-over-
            // tick. `alignment = dot(heading_xz, prey_dir_xz) ∈ [-1, 1]`
            // where heading is the body's local +Z projected to XZ.
            // Reward only the increase (turning toward), never penalise
            // turning away — same rectified philosophy as `progress`.
            let alignment = phys.nearest_prey_dir_xz.map(|dir| {
                let fwd = phys.base_rot * Vec3::Z;
                let fwd_xz = Vec2::new(fwd.x, fwd.z);
                let h = fwd_xz.normalize_or_zero();
                h.dot(dir)
            });
            let heading_gain = match (self.prev_heading_alignment[su], alignment) {
                (Some(prev), Some(curr)) => (curr - prev).max(0.0),
                _                        => 0.0,
            };
            self.prev_heading_alignment[su] = alignment;

            // Stepping: count limb-contact transitions (a foot lifting
            // or planting) since the last tick over ALL limb flags
            // (indices 1..=8 — index 0 is the base, deliberately
            // excluded). Each flip of a 0/1 flag contributes 1.0.
            // Rewards cycling feet on/off the ground, the core of a gait.
            let prev_lc = self.prev_limb_contact[su];
            let step_events: f32 = (1..phys.limb_contact.len())
                .map(|k| (phys.limb_contact[k] - prev_lc[k]).abs())
                .sum();
            self.prev_limb_contact[su] = phys.limb_contact;

            // Base angular speed (spin/tumble), for the anti-degenerate
            // penalty — keeps the "reward movement" gradient pointed at
            // translation rather than the spin/flight the data showed.
            let base_ang_speed = Vec3::new(
                phys.base_pose_vel[6], phys.base_pose_vel[7], phys.base_pose_vel[8],
            ).length();

            // Movement reward is GATED on uprightness: a controlled, upright
            // body that translates earns it; a tumbling / ballistic body
            // (uprightness near 0 or negative while airborne) earns ~nothing.
            // This is what makes controlled walking out-score the high-speed
            // flight/spin degenerate the data showed (which otherwise wins
            // even capped, because it racks up raw speed).
            let upright_pos = uprightness.clamp(0.0, 1.0);
            let dense_reward = K_MOVE     * speed.min(SPEED_REWARD_CAP) * upright_pos
                             + K_FWD      * forward_speed
                             + K_UP       * uprightness * motion_gate
                             - K_IDLE     * (1.0 - motion_gate)
                             + K_PROGRESS * progress
                             + K_HEADING  * heading_gain
                             + K_STEP     * step_events
                             - K_SPIN     * base_ang_speed;

            let reward = event_reward + dense_reward;
            self.prev_predations[su]    = organism.predations;
            self.prev_reproductions[su] = organism.reproductions;
            self.prev_action[su]        = action;

            // Pull this slot's real observation row out of the flat
            // input_buf so the PPO train step can re-forward through
            // the (slightly newer) policy weights.
            let obs_base = (s as usize) * IN;
            let mut obs_row = [0.0_f32; IN];
            obs_row.copy_from_slice(&input_buf[obs_base..obs_base + IN]);
            let entry = RolloutEntry {
                obs:      obs_row,
                action,
                log_prob,
                value:    v_buf[s as usize],
                reward,
                done:     false,
            };
            let buf = &mut self.rollouts[s as usize];
            if buf.len() >= ROLLOUT_LEN { buf.pop_front(); }
            buf.push_back(entry);

            // We DON'T train per-slot here; the global counter below
            // triggers one batched train across every slot whose buffer
            // has anything in it, with a per-slot mask that zeroes out
            // contributions from slots that didn't accumulate a full
            // rollout this window.
        }

        self.ticks_since_train += 1;
        if self.ticks_since_train >= ROLLOUT_LEN {
            self.train(virtual_time_secs);
            self.ticks_since_train = 0;
        }
    }

    /// Global synchronous PPO update.
    ///
    /// Triggered by `apply_step` every `ROLLOUT_LEN` ticks. Builds
    /// batched `[N, T, …]` tensors from the per-slot rollout buffers,
    /// runs GAE per slot, normalises advantages, then loops
    /// `PPO_EPOCHS` times over the clipped-surrogate + value-MSE +
    /// entropy-bonus loss. Slots whose buffer didn't accumulate any
    /// entries this window are masked out (mask = 0 for those rows ×
    /// timesteps), so their per-row gradients vanish and their Adam
    /// state is left effectively untouched.
    ///
    /// All N organisms share the optimiser step, but each one's
    /// gradient flows only through its own row of the batched-MLP
    /// weight tensors — `forward_rollout` does a per-row matmul, so
    /// no cross-organism leakage.
    pub fn train(&mut self, virtual_time_secs: f32) {
        let n = self.n;
        let t = ROLLOUT_LEN;

        // ── 0. V(s_T) bootstrap. ─────────────────────────────────────
        //
        // For each slot with a non-empty rollout, forward the LATEST
        // critic on the last observed state (`buf.back().obs`) to get
        // a fresh value estimate. This becomes `values_tail` in
        // `compute_gae`, replacing the earlier `0.0` bias. We batch
        // every slot's last-obs into a single `[N, IN]` tensor and
        // forward once — slots with empty buffers contribute a row of
        // zeros that we never read.
        let mut last_obs_buf: Vec<f32> = vec![0.0; n * IN];
        for s in 0..n {
            if let Some(last) = self.rollouts[s].back() {
                let base = s * IN;
                last_obs_buf[base..base + IN].copy_from_slice(&last.obs);
            }
        }
        let last_obs_t = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(last_obs_buf, [n, IN]),
            &self.device,
        );
        let bootstrap_t = self.critic.forward(last_obs_t);   // [N, 1]
        let bootstrap_vec = bootstrap_t.into_data().into_vec::<f32>()
            .expect("bootstrap critic forward");

        // ── 1. Per-slot GAE + flat buffers. ─────────────────────────
        let mut states_buf:    Vec<f32> = vec![0.0; n * t * IN];
        let mut actions_buf:   Vec<f32> = vec![0.0; n * t * OUT];
        let mut old_lp_buf:    Vec<f32> = vec![0.0; n * t];
        let mut adv_buf:       Vec<f32> = vec![0.0; n * t];
        let mut returns_buf:   Vec<f32> = vec![0.0; n * t];
        let mut mask_buf:      Vec<f32> = vec![0.0; n * t];
        let mut total_count   = 0.0_f32;
        // Per-slot return summaries for the training-step CSV. We need
        // the mean and variance across all (slot × timestep) entries
        // that actually contributed to this window.
        let mut active_slots: u32 = 0;

        for s in 0..n {
            let buf = &self.rollouts[s];
            let count = buf.len();
            if count == 0 { continue; }
            active_slots += 1;

            // Per-slot bootstrap from the batched critic forward above.
            let bootstrap = bootstrap_vec[s];
            let (advantages, returns) = Self::compute_gae(buf, bootstrap);

            // Per-slot advantage normalisation (zero-mean, unit-var)
            // — standard PPO trick to keep the policy gradient
            // well-scaled across slots with different reward
            // magnitudes.
            let mean: f32 = advantages.iter().sum::<f32>() / count as f32;
            let var: f32 = advantages.iter()
                .map(|a| (a - mean) * (a - mean))
                .sum::<f32>() / count as f32;
            let std = var.sqrt().max(1e-8);

            for ti in 0..count {
                let entry = &buf[ti];
                let s_base = (s * t + ti) * IN;
                states_buf[s_base..s_base + IN].copy_from_slice(&entry.obs);
                let a_base = (s * t + ti) * OUT;
                actions_buf[a_base..a_base + OUT].copy_from_slice(&entry.action);
                old_lp_buf [s * t + ti] = entry.log_prob;
                adv_buf    [s * t + ti] = (advantages[ti] - mean) / std;
                returns_buf[s * t + ti] = returns[ti];
                mask_buf   [s * t + ti] = 1.0;
                total_count += 1.0;
            }
        }

        if total_count < 1.0 {
            // No accumulated data — skip this update window.
            for r in &mut self.rollouts { r.clear(); }
            return;
        }

        // ── 1-bis. Return summary (computed before the buffers move
        //          into GPU tensors). ──────────────────────────────
        let mut sum_r = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        let mut count_r: u32 = 0;
        for i in 0..(n * t) {
            if mask_buf[i] > 0.5 {
                let r = returns_buf[i] as f64;
                sum_r  += r;
                sum_sq += r * r;
                count_r += 1;
            }
        }
        let mean_return = if count_r > 0 { (sum_r / count_r as f64) as f32 } else { 0.0 };
        let return_var  = if count_r > 0 {
            ((sum_sq / count_r as f64) - (sum_r / count_r as f64).powi(2)) as f32
        } else { 0.0 };

        // ── 2. Build GPU tensors. ──────────────────────────────────
        let states_t = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(states_buf, [n, t, IN]),
            &self.device,
        );
        let actions_t = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(actions_buf, [n, t, OUT]),
            &self.device,
        );
        let old_lp_t = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(old_lp_buf, [n, t, 1]),
            &self.device,
        );
        let adv_t = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(adv_buf, [n, t, 1]),
            &self.device,
        );
        let returns_t = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(returns_buf, [n, t, 1]),
            &self.device,
        );
        let mask_t = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(mask_buf, [n, t, 1]),
            &self.device,
        );

        // ── 3. PPO_EPOCHS updates. Each epoch runs the policy + value
        //       loss together → backward → Adam step on each network. ─
        let log_2pi = (2.0_f32 * std::f32::consts::PI).ln();
        // Entropy of `N(μ, σ)` per dim is `log σ + 0.5·ln(2πe)`. With
        // per-organism `log_std`, total entropy per slot = sum over OUT
        // dims of `log_std[s, d] + entropy_const`. Constant doesn't
        // affect gradients, kept for interpretability.
        let entropy_const = 0.5_f32 * (2.0_f32 * std::f32::consts::PI * std::f32::consts::E).ln();

        // Scalars captured from the FINAL PPO epoch and pushed onto
        // `training_history` after the loop completes. Default to 0
        // so an empty rollout window still produces a sensible row.
        let mut last_actor_loss:  f32 = 0.0;
        let mut last_critic_loss: f32 = 0.0;
        let mut last_entropy:     f32 = 0.0;
        let mut last_total_loss:  f32 = 0.0;

        for epoch in 0..PPO_EPOCHS {
            // Actor forward over rollout: [N, T, OUT] μ.
            let mu = self.actor.forward_rollout(states_t.clone());
            // log_std: [N, OUT] → broadcast to [N, T, OUT].
            let log_std_nt = self.actor.log_std.val()
                .unsqueeze_dim::<3>(1);                    // [N, 1, OUT]
            let sigma_nt = log_std_nt.clone().exp();
            let inv_sigma_sq = sigma_nt.powf_scalar(-2.0);
            // New log_prob per (slot, t, dim), summed over dim → [N, T, 1].
            let diff = actions_t.clone() - mu;             // [N, T, OUT]
            let new_lp_per_dim = (diff.clone() * diff)
                .mul(inv_sigma_sq)
                .mul_scalar(-0.5_f32)
                .sub(log_std_nt.clone())
                .sub_scalar(0.5_f32 * log_2pi);            // [N, T, OUT]
            let new_lp = new_lp_per_dim.sum_dim(2);        // [N, T, 1]

            // Ratio + clipped surrogate.
            let ratio = (new_lp - old_lp_t.clone()).exp();
            let surr1 = ratio.clone() * adv_t.clone();
            let surr2 = ratio.clamp(1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_t.clone();
            // PPO maximises the surrogate → loss is `-min(s1, s2)`.
            let policy_loss = (surr1.min_pair(surr2)).neg();   // [N, T, 1]

            // Critic forward + MSE on returns (same forward used to
            // produce a value loss whose gradient updates the critic
            // — actor branch is independent because they share no
            // parameters).
            let v = self.critic.forward_rollout(states_t.clone()); // [N, T, 1]
            let v_diff = v - returns_t.clone();
            let value_loss = (v_diff.clone() * v_diff).mul_scalar(0.5);

            // Entropy bonus per slot: `Σ_d (log_std[s,d] + const)`.
            let entropy_per_slot = log_std_nt
                .add_scalar(entropy_const)
                .sum_dim(2);                                // [N, 1, 1]
            // Mean over slots (the rollout window is constant length
            // per slot, so mean over N is the right normaliser).
            let entropy_mean = entropy_per_slot.mean();

            let combined = (policy_loss.clone() + value_loss.clone().mul_scalar(VALUE_LOSS_COEF))
                * mask_t.clone();
            let loss = combined.sum().div_scalar(total_count)
                - entropy_mean.clone().mul_scalar(ENTROPY_COEF);

            // Telemetry: on the last epoch, pull scalars off the GPU
            // before backward consumes them. The clones below are cheap
            // (handle copies on Burn tensors); the actual GPU→CPU
            // transfer is one synchronisation per scalar.
            if epoch == PPO_EPOCHS - 1 {
                last_actor_loss  = scalar_of(policy_loss.clone().mean());
                last_total_loss  = scalar_of(loss.clone());
                last_entropy     = scalar_of(entropy_mean.clone());
            }

            // Backward + Adam step on the actor. Clone is cheap (handle
            // copies; underlying tensors are reference-counted).
            let actor_clone = self.actor.clone();
            let actor_grads = GradientsParams::from_grads(loss.backward(), &self.actor);
            self.actor = self.opt_actor.step(LR, actor_clone, actor_grads);

            // Critic update: re-forward to build a fresh graph (the
            // graph from above was consumed by `loss.backward()`),
            // then its own backward + step.
            let v2 = self.critic.forward_rollout(states_t.clone());
            let v2_diff = v2 - returns_t.clone();
            let critic_loss_raw = (v2_diff.clone() * v2_diff).mul_scalar(0.5);
            let critic_loss = (critic_loss_raw * mask_t.clone()).sum()
                .div_scalar(total_count);
            if epoch == PPO_EPOCHS - 1 {
                last_critic_loss = scalar_of(critic_loss.clone());
            }
            let critic_clone = self.critic.clone();
            let critic_grads = GradientsParams::from_grads(critic_loss.backward(), &self.critic);
            self.critic = self.opt_critic.step(LR, critic_clone, critic_grads);
        }

        // ── 4-bis. Push a TrainingStep row onto the ring buffer. ──
        // `mean_return` / `return_var` were computed above (before the
        // buffers moved into GPU tensors).
        self.training_step += 1;
        if self.training_history.len() >= LIMB_TRAINING_HISTORY_CAP {
            self.training_history.pop_front();
        }
        self.training_history.push_back(LimbTrainingStep {
            step:               self.training_step,
            virtual_time_secs,
            n_active:           active_slots,
            actor_loss:         last_actor_loss,
            critic_loss:        last_critic_loss,
            entropy:            last_entropy,
            total_loss:         last_total_loss,
            mean_return,
            return_var,
            supervised_loss:    0.0,
        });

        // ── 4. Reset rollout buffers for the next window. ─────────
        for r in &mut self.rollouts { r.clear(); }
    }
}
