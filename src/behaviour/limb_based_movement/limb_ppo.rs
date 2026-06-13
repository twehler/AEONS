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
// supply the enrolment filter (`IntelligenceLevel` + `!movement_mode.is_sliding()`
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
/// `OUT = MAX_LIMB_JOINTS + N_LIMB_TWIST_GROUPS`: `out[0..MAX_LIMB_JOINTS]` are
/// per-limb hinge SWING targets (`drive_limb_motors`); the last
/// `N_LIMB_TWIST_GROUPS` outputs are GROUPED TWIST efforts about the limbs' long
/// axes (`drive_limb_twist`, limb i → group `(i-1) % N`). So limbs move in 3D
/// (swing + grouped twist), not just one fore/aft plane. Twist is grouped (few
/// outputs) because per-limb twist doubled the action space and hurt locomotion
/// learning. (The observation echoes only the SWING half of the previous action,
/// so `IN` is unchanged — the brain re-observes actual 3D joint rotation via
/// `joint_sincos`, which already captures twist.)
pub const IN:     usize = 134;
pub const HIDDEN: usize = 128;
pub const OUT:    usize = MAX_LIMB_JOINTS + N_LIMB_TWIST_GROUPS;

pub use crate::simulation_settings::{
    ROLLOUT_LEN, PPO_EPOCHS, CLIP_EPS, GAMMA, LAMBDA, LR,
    VALUE_LOSS_COEF, ENTROPY_COEF, LOG_STD_INIT, GAIT_FREQUENCY_HZ,
    MAX_LIMB_JOINTS, N_LIMB_TWIST_GROUPS,
};

// ── Reward shaping ──
//
// Sparse event signals (same shape as the sliding pools) plus dense
// locomotion-intrinsic terms designed to avoid the freeze local optimum:
//
//   1. Forward velocity — `lin_vel_xz · heading_xz` (signed projection onto
//      the body's facing direction). Pays only for travel in the facing
//      direction: pure spin nets ~0, backward drift mildly penalised. With
//      `K_HEADING` this composes into directed pursuit.
//   2. Uprightness GATED on motion — `(rot · +Y).y * min(1, speed / IDLE_THRESH)`.
//      Standing still upright scores 0; the term only counts while locomoting.
//   3. Idle penalty — `−K_IDLE * max(0, 1 − speed / IDLE_THRESH)`, pushing the
//      gradient out of the freeze basin.
pub use crate::simulation_settings::{
    K_EAT, K_REPRO, K_FWD, K_UP, K_IDLE, K_PROGRESS, K_HEADING, K_STEP,
    IDLE_THRESH, K_MOVE, SPEED_REWARD_CAP, K_SPIN, GROUND_GATE_MIN_FEET, K_VERT, K_AIR, K_TWIST,
    K_BELLY,
    // Standing task.
    STANDING_TASK, K_STAND, STAND_HEIGHT_TARGET, STAND_MIN_FEET, K_TILT, K_DRIFT,
    K_TORQUE_REG, K_ACTRATE, K_ALIVE, STAND_UPRIGHT_MIN,
};


// ── Networks (SHARED PER SPECIES: one net, no per-organism batch dim) ───────────

/// Actor MLP — a SINGLE shared `IN → HIDDEN → OUT` net (per species) returning
/// `μ` for the diagonal-Gaussian policy (tanh-bounded). Weights are 2-D
/// (`[IN, HIDDEN]`, `[HIDDEN, OUT]`) and biases / `log_std` are `[1, ·]` so a
/// forward over a batch of `B` observations is a plain `[B, IN] · [IN, HIDDEN]`
/// matmul with broadcast bias. The same `forward` serves both inference (one row
/// per active organism of the species) and the PPO update (all pooled
/// transitions of the species as one batch). `log_std` is trainable.
#[derive(Module, Debug)]
pub struct LimbActor<B: Backend> {
    pub w1: Param<Tensor<B, 2>>,  // [IN, HIDDEN]
    pub b1: Param<Tensor<B, 2>>,  // [1, HIDDEN]
    pub w2: Param<Tensor<B, 2>>,  // [HIDDEN, OUT]
    pub b2: Param<Tensor<B, 2>>,  // [1, OUT]
    /// Shared `log_std` for each action dim. Trainable.
    pub log_std: Param<Tensor<B, 2>>,  // [1, OUT]
}

impl<B: Backend> LimbActor<B> {
    /// Forward: `obs [B, IN] → μ [B, OUT]` (shared weights, broadcast bias).
    /// Bound the policy mean to [-1, 1] so `limb_targets` can't run away during
    /// training. The PD controller in `rapier_setup` then scales this into a
    /// target joint angle.
    pub fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = relu(obs.matmul(self.w1.val()) + self.b1.val());  // [B,H] + [1,H]
        tanh(h.matmul(self.w2.val()) + self.b2.val())             // [B,OUT] + [1,OUT]
    }
}

/// Critic MLP — a SINGLE shared `IN → HIDDEN → 1` net (per species) returning
/// `V(s)`.
#[derive(Module, Debug)]
pub struct LimbCritic<B: Backend> {
    pub w1: Param<Tensor<B, 2>>,  // [IN, HIDDEN]
    pub b1: Param<Tensor<B, 2>>,  // [1, HIDDEN]
    pub w2: Param<Tensor<B, 2>>,  // [HIDDEN, 1]
    pub b2: Param<Tensor<B, 2>>,  // [1, 1]
}

impl<B: Backend> LimbCritic<B> {
    /// Forward: `obs [B, IN] → V [B, 1]`.
    pub fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = relu(obs.matmul(self.w1.val()) + self.b1.val());  // [B,H] + [1,H]
        h.matmul(self.w2.val()) + self.b2.val()                   // [B,1] + [1,1]
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
    /// Base body's clearance above the terrain (`base_pos.y − heightmap`), the
    /// height signal for the STANDING reward's tall-posture term. `f32::MAX` (a
    /// huge sentinel) when no heightmap is available, so the height factor reads
    /// as "fully tall" and the term is inert rather than zeroing the reward.
    pub base_clearance: f32,
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
            base_clearance:      f32::MAX,
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
    obs[121..129].copy_from_slice(&prev_action[0..MAX_LIMB_JOINTS]); // SWING half only (keeps IN=134)

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
            obs[131] = (dist / crate::simulation_settings::PREY_SCAN_RADIUS).clamp(0.0, 1.0);
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
        &bevy::prelude::GlobalTransform,
        &bevy_rapier3d::prelude::Velocity,
        Option<&crate::rapier_setup::LimbContact>,
    )>,
    world_grid:  &crate::world_model::WorldModelGrid,
    heightmap:   Option<&crate::world_geometry::HeightmapSampler>,
) -> HashMap<Entity, LimbObsInputs> {
    let mut out: HashMap<Entity, LimbObsInputs> = HashMap::new();
    let mut base_rot: HashMap<Entity, Quat>     = HashMap::new();
    let mut base_pos: HashMap<Entity, Vec3>     = HashMap::new();

    // Pass 1: base bodies — record base pose+vel + base contact + cache
    // rotation/position for the limb pass. Pose comes from the world-space
    // `GlobalTransform`; world-frame velocity from Rapier's `Velocity`.
    for (child_of, idx, gt, vel, contact) in body_parts.iter() {
        if idx.0 != 0 { continue; }
        let root = child_of.parent();
        let rot = gt.rotation();
        let (rx, ry, rz) = rot.to_euler(bevy::math::EulerRot::XYZ);
        let entry = out.entry(root).or_default();
        // Orientation as sin/cos pairs (continuous across the ±π wrap).
        entry.base_pose_vel[0]  = rx.sin();
        entry.base_pose_vel[1]  = rx.cos();
        entry.base_pose_vel[2]  = ry.sin();
        entry.base_pose_vel[3]  = ry.cos();
        entry.base_pose_vel[4]  = rz.sin();
        entry.base_pose_vel[5]  = rz.cos();
        entry.base_pose_vel[6]  = vel.angular.x;
        entry.base_pose_vel[7]  = vel.angular.y;
        entry.base_pose_vel[8]  = vel.angular.z;
        entry.base_pose_vel[9]  = vel.linear.x;
        entry.base_pose_vel[10] = vel.linear.y;
        entry.base_pose_vel[11] = vel.linear.z;
        // Up-vector: where the body's +Y points in world frame.
        let up = rot * Vec3::Y;
        entry.base_up = [up.x, up.y, up.z];
        entry.limb_contact[0] = if contact.is_some_and(|c| c.in_contact) { 1.0 } else { 0.0 };
        entry.base_rot     = rot;
        entry.base_lin_vel = vel.linear;
        base_rot.insert(root, rot);
        base_pos.insert(root, gt.translation());
    }

    // Pass 2: limbs (idx 1..=MAX_LIMB_JOINTS). Each limb body part fills its
    // OWN joint slot (joint = idx-1), measured relative to the base rotation.
    // No pair/half collapsing — every joint is observed and controlled
    // independently, so an alternating gait can emerge.
    for (child_of, idx, gt, vel, contact) in body_parts.iter() {
        if idx.0 == 0 || idx.0 > MAX_LIMB_JOINTS { continue; }
        let root = child_of.parent();
        let Some(base_r) = base_rot.get(&root) else { continue };
        let j = idx.0 - 1;   // 0-based joint slot
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
            // Base clearance above terrain → the STANDING height-target signal.
            if let Some(hm) = heightmap {
                entry.base_clearance = pos.y - hm.height_at(pos.x, pos.z);
            }
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
    /// Pull every per-species weight tensor off the GPU and FAN them OUT into a
    /// per-SLOT flat snapshot: for each occupied slot, that slot's CURRENT
    /// species' net weights are written into the slot's row of the flat buffers.
    /// This keeps the snapshot's flat layout (`[n * IN * HIDDEN]` etc., indexed
    /// by slot) identical to the old per-organism pool, so the dataset exporter
    /// (which slot-indexes `actor_log_std`) and `extract(entity)` (which slices
    /// the entity's slot row) both work unchanged. Unoccupied slots stay zero
    /// (never read). One GPU→CPU sync per tensor per live species.
    pub fn snapshot(&self) -> LimbPoolSnapshot {
        let n = self.n;
        let mut snap = LimbPoolSnapshot {
            actor_w1:      vec![0.0; n * IN * HIDDEN],
            actor_b1:      vec![0.0; n * HIDDEN],
            actor_w2:      vec![0.0; n * HIDDEN * OUT],
            actor_b2:      vec![0.0; n * OUT],
            actor_log_std: vec![0.0; n * OUT],
            critic_w1:     vec![0.0; n * IN * HIDDEN],
            critic_b1:     vec![0.0; n * HIDDEN],
            critic_w2:     vec![0.0; n * HIDDEN],
            critic_b2:     vec![0.0; n],
            map:           self.map.clone(),
        };
        // Build entity→slot reverse lookup grouped by species via slot_species:
        // for each occupied slot, write its species' flat weights into that row.
        // Pull each species' tensors to CPU once, then scatter to all its slots.
        let mut cache: HashMap<u32, BrainRestoreLimb> = HashMap::new();
        for (_e, &slot) in self.map.iter() {
            let s = slot as usize;
            if s >= n { continue; }
            let key = self.slot_species[s];
            let flat = cache.entry(key).or_insert_with(|| {
                let brain = self.species.get(&key);
                match brain {
                    Some(b) => BrainRestoreLimb {
                        actor_w1:      b.actor.w1.val().clone().into_data().into_vec::<f32>().expect("actor.w1"),
                        actor_b1:      b.actor.b1.val().clone().into_data().into_vec::<f32>().expect("actor.b1"),
                        actor_w2:      b.actor.w2.val().clone().into_data().into_vec::<f32>().expect("actor.w2"),
                        actor_b2:      b.actor.b2.val().clone().into_data().into_vec::<f32>().expect("actor.b2"),
                        actor_log_std: b.actor.log_std.val().clone().into_data().into_vec::<f32>().expect("actor.log_std"),
                        critic_w1:     b.critic.w1.val().clone().into_data().into_vec::<f32>().expect("critic.w1"),
                        critic_b1:     b.critic.b1.val().clone().into_data().into_vec::<f32>().expect("critic.b1"),
                        critic_w2:     b.critic.w2.val().clone().into_data().into_vec::<f32>().expect("critic.w2"),
                        critic_b2:     b.critic.b2.val().clone().into_data().into_vec::<f32>().expect("critic.b2"),
                    },
                    // No net for this species yet (slot enrolled but never
                    // applied): leave zeros (a future load degrades to fresh init).
                    None => BrainRestoreLimb {
                        actor_w1:      vec![0.0; IN * HIDDEN],
                        actor_b1:      vec![0.0; HIDDEN],
                        actor_w2:      vec![0.0; HIDDEN * OUT],
                        actor_b2:      vec![0.0; OUT],
                        actor_log_std: vec![0.0; OUT],
                        critic_w1:     vec![0.0; IN * HIDDEN],
                        critic_b1:     vec![0.0; HIDDEN],
                        critic_w2:     vec![0.0; HIDDEN],
                        critic_b2:     vec![0.0; 1],
                    },
                }
            });
            snap.actor_w1     [s * IN * HIDDEN .. (s + 1) * IN * HIDDEN].copy_from_slice(&flat.actor_w1);
            snap.actor_b1     [s * HIDDEN      .. (s + 1) * HIDDEN]     .copy_from_slice(&flat.actor_b1);
            snap.actor_w2     [s * HIDDEN * OUT .. (s + 1) * HIDDEN * OUT].copy_from_slice(&flat.actor_w2);
            snap.actor_b2     [s * OUT         .. (s + 1) * OUT]        .copy_from_slice(&flat.actor_b2);
            snap.actor_log_std[s * OUT         .. (s + 1) * OUT]        .copy_from_slice(&flat.actor_log_std);
            snap.critic_w1    [s * IN * HIDDEN .. (s + 1) * IN * HIDDEN].copy_from_slice(&flat.critic_w1);
            snap.critic_b1    [s * HIDDEN      .. (s + 1) * HIDDEN]     .copy_from_slice(&flat.critic_b1);
            snap.critic_w2    [s * HIDDEN      .. (s + 1) * HIDDEN]     .copy_from_slice(&flat.critic_w2);
            snap.critic_b2    [s ..= s]                                 .copy_from_slice(&flat.critic_b2);
        }
        snap
    }

    /// Overwrite a SPECIES' shared net from saved flat weights (one 2-D net's
    /// flat layout — same as `BrainRestoreLimb`). Called by `assign_brains_*_limb`
    /// when a loaded organism of species `key` carries a `BrainRestoreLimb`
    /// payload. Shape-validated: on a length mismatch (e.g. a save from before
    /// IN was bumped) the saved payload is rejected and the species is left to
    /// lazily warm-start a fresh net — degrade, don't panic. Idempotent per save:
    /// the first loaded member of a species writes the net; later members hit the
    /// same (now-correct) weights and overwrite identically.
    pub fn restore_species(&mut self, key: u32, r: &BrainRestoreLimb) {
        // Architecture guard. Lengths must match the CURRENT IN/HIDDEN/OUT or
        // `TensorData::new` below would panic on the shape/length mismatch.
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
                 architecture (IN={IN}, HIDDEN={HIDDEN}, OUT={OUT}); species {key} \
                 keeps / lazily warm-starts fresh weights"
            );
            // Make sure the species at least has a (warm-start) net to use.
            self.ensure_species(key);
            return;
        }
        let device = self.device.clone();
        let actor = LimbActor::<MyBackend> {
            w1:      Param::from_tensor(Tensor::<MyBackend, 2>::from_data(TensorData::new(r.actor_w1.clone(),      [IN, HIDDEN]), &device)),
            b1:      Param::from_tensor(Tensor::<MyBackend, 2>::from_data(TensorData::new(r.actor_b1.clone(),      [1, HIDDEN]),  &device)),
            w2:      Param::from_tensor(Tensor::<MyBackend, 2>::from_data(TensorData::new(r.actor_w2.clone(),      [HIDDEN, OUT]), &device)),
            b2:      Param::from_tensor(Tensor::<MyBackend, 2>::from_data(TensorData::new(r.actor_b2.clone(),      [1, OUT]),     &device)),
            log_std: Param::from_tensor(Tensor::<MyBackend, 2>::from_data(TensorData::new(r.actor_log_std.clone(), [1, OUT]),     &device)),
        };
        let critic = LimbCritic::<MyBackend> {
            w1: Param::from_tensor(Tensor::<MyBackend, 2>::from_data(TensorData::new(r.critic_w1.clone(), [IN, HIDDEN]), &device)),
            b1: Param::from_tensor(Tensor::<MyBackend, 2>::from_data(TensorData::new(r.critic_b1.clone(), [1, HIDDEN]),  &device)),
            w2: Param::from_tensor(Tensor::<MyBackend, 2>::from_data(TensorData::new(r.critic_w2.clone(), [HIDDEN, 1]),  &device)),
            b2: Param::from_tensor(Tensor::<MyBackend, 2>::from_data(TensorData::new(r.critic_b2.clone(), [1, 1]),       &device)),
        };
        let opt_actor:  Box<dyn BrainOptActor>  = Box::new(AdamConfig::new().init());
        let opt_critic: Box<dyn BrainOptCritic> = Box::new(AdamConfig::new().init());
        self.species.insert(key, SpeciesBrain {
            actor, critic, opt_actor, opt_critic, training_step: 0,
        });
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
    /// Per-species shared nets. Every organism of a species uses AND trains its
    /// species' net; different species hold different weights, so behaviour
    /// diverges by species while same-species individuals act alike (momentary
    /// variation comes only from each organism's own exploration noise). Keyed
    /// by `Organism::species_id` (`UNCLASSIFIED` until first classification);
    /// created lazily, warm-started with the oscillatory gait prior.
    pub species: HashMap<u32, SpeciesBrain>,
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
    /// Per-slot species key as of the last apply tick. Lets `train` group slots
    /// by species without an organism query (written in `apply_step`), and lets
    /// `snapshot` resolve each slot's species net.
    pub slot_species: Vec<u32>,
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

/// Sentinel species key for organisms not yet classified by the speciation
/// system (real `species_id`s start at 1). All unclassified limb organisms
/// share this one transitional net until the ~1 Hz speciation tick assigns
/// them; the next apply tick then re-points them at their real species' net
/// automatically (the per-tick species lookup makes reclassification a no-op).
pub const UNCLASSIFIED: u32 = 0;

/// One species' shared actor + critic + their optimisers, plus that species'
/// own PPO step counter. Created lazily by `new_species_brain` and warm-started
/// with the oscillatory gait prior.
pub struct SpeciesBrain {
    pub actor:         LimbActor<MyBackend>,
    pub critic:        LimbCritic<MyBackend>,
    pub opt_actor:     Box<dyn BrainOptActor>,
    pub opt_critic:    Box<dyn BrainOptCritic>,
    pub training_step: u64,
}

/// Build one freshly-initialised species brain. Biases zero; shared `log_std`
/// initialised to `LOG_STD_INIT`.
///
/// Weights use fan-in-scaled (Xavier-style) init, NOT the sliding pools'
/// flat `Uniform ±0.5`. With `IN = 134` a flat `±0.5` saturates the actor's
/// tanh `μ` at ±1 from step one, pinning every joint at its mechanical stop
/// (frozen legs, no propulsion to reward). Fan-in scaling keeps the initial
/// output near 0 so exploration noise actually moves the limbs. The OSCILLATORY
/// WARM-START is applied (unless STANDING_TASK): two hidden carrier units fed
/// only the phase-clock inputs make each SWING output start as
/// `A·sin(phase + φ_k)` — a TRAINABLE rhythmic prior, not a CPG. (Each new
/// species gets its OWN warm-started prior.)
pub fn new_species_brain(device: &CudaDevice) -> SpeciesBrain {
    // ~1/sqrt(fan_in) for fan_in≈IN=134 → ±0.086. Keeps initial activations
    // unit-scale so `μ` does NOT saturate.
    let w = Initializer::Uniform { min: -0.086, max: 0.086 };
    let z = Initializer::Zeros;
    let log_std_init = Initializer::Constant { value: LOG_STD_INIT as f64 };

    // ── Actor OSCILLATORY WARM-START ───────────────────────────────────
    // Pure Gaussian exploration almost never stumbles onto a coherent
    // periodic propulsive stroke, so the actor is initialised so the
    // phase-clock inputs (obs[132]=sin, obs[133]=cos) drive each joint as
    // `μ_k ≈ A·sin(phase + φ_k)`, with a per-joint phase offset. This is a
    // TRAINABLE init, NOT a CPG: the weights are ordinary policy parameters
    // PPO can reshape or abandon — locomotion stays learned, just
    // bootstrapped near a rhythmic prior. Mechanism: two hidden units are
    // biased into relu's linear region (`b1 = WARMSTART_BIAS`) and fed only
    // the sin/cos phase inputs (gain `WARMSTART_PHASE_GAIN`), carrying
    // `C ± g·sin` / `C ± g·cos`; the output layer recombines them as
    // `A·sin(phase+φ_k)` and `b2` cancels the constant. Every other weight
    // stays at the small random init.
    const WARMSTART_PHASE_GAIN: f32 = 2.0;  // g: sin/cos input → carrier hidden unit
    const WARMSTART_BIAS:       f32 = 3.0;  // C: keeps carrier units in relu's linear region (C>g)
    // a: output amplitude gain (final swing ≈ a·g·sin). Kept small so the
    // legs mostly hold a planted weight-bearing stance and only gently
    // modulate; a larger swing lifts ALL feet together and seeds the
    // ballistic hop on the light body. Lowered to keep more organisms out of
    // that basin from step 0. The policy grows the stride from here if rewarded.
    const WARMSTART_AMP:        f32 = 0.15;
    use rand::RngExt as _;
    let mut rng = rand::rng();
    let rand_small = |rng: &mut rand::rngs::ThreadRng| rng.random_range(-0.086_f32..0.086);
    let phi = |k: usize| (k as f32) * std::f32::consts::FRAC_PI_2; // per-joint phase offset

    // Single shared net: w1 [IN,HIDDEN], w2 [HIDDEN,OUT] (no batch dim).
    let mut w1v = vec![0.0_f32; IN * HIDDEN];
    for v in w1v.iter_mut() { *v = rand_small(&mut rng); }
    let mut b1v = vec![0.0_f32; HIDDEN];
    let mut w2v = vec![0.0_f32; HIDDEN * OUT];
    for v in w2v.iter_mut() { *v = rand_small(&mut rng); }
    let mut b2v = vec![0.0_f32; OUT];
    // The STANDING task wants a STILL stance, not a gait: skip the oscillatory
    // warm-start so μ≈0 (legs hold near-neutral). Gentler initial dynamics →
    // far less joint separation, and no rhythmic prior fighting a static stand.
    if !STANDING_TASK {
        // Hidden units 0 (sin carrier) and 1 (cos carrier): clear their
        // input columns, then feed ONLY the phase inputs.
        for inp in 0..IN {
            w1v[inp * HIDDEN + 0] = 0.0;
            w1v[inp * HIDDEN + 1] = 0.0;
        }
        w1v[132 * HIDDEN + 0] = WARMSTART_PHASE_GAIN; // sin → h0
        w1v[133 * HIDDEN + 1] = WARMSTART_PHASE_GAIN; // cos → h1
        b1v[0] = WARMSTART_BIAS;
        b1v[1] = WARMSTART_BIAS;
        // Oscillatory warm-start applies to the SWING outputs only
        // (0..MAX_LIMB_JOINTS). The TWIST outputs (MAX_LIMB_JOINTS..OUT) keep
        // their small random init → twist starts near zero and is learned
        // from scratch (no rhythmic prior imposed on the new DOF).
        for k in 0..MAX_LIMB_JOINTS {
            let (s, c) = (phi(k).sin(), phi(k).cos());
            w2v[0 * OUT + k] = WARMSTART_AMP * c; // h0 (sin) → out
            w2v[1 * OUT + k] = WARMSTART_AMP * s; // h1 (cos) → out
            b2v[k] = -WARMSTART_BIAS * WARMSTART_AMP * (c + s); // cancel constant
        }
    }
    // Uniform exploration σ for all outputs (swing + twist). Twist stays
    // controllable by keeping `MAX_LIMB_TWIST_TORQUE` low (so noise can't
    // fling the light body) rather than by suppressing twist exploration —
    // a low twist-σ froze the twist outputs and made the `K_TWIST` opt-in
    // cost unable to push gratuitous twist back down.
    let actor = LimbActor::<MyBackend> {
        w1:      Param::from_tensor(Tensor::<MyBackend, 2>::from_data(TensorData::new(w1v, [IN, HIDDEN]), device)),
        b1:      Param::from_tensor(Tensor::<MyBackend, 2>::from_data(TensorData::new(b1v, [1, HIDDEN]),  device)),
        w2:      Param::from_tensor(Tensor::<MyBackend, 2>::from_data(TensorData::new(w2v, [HIDDEN, OUT]), device)),
        b2:      Param::from_tensor(Tensor::<MyBackend, 2>::from_data(TensorData::new(b2v, [1, OUT]),     device)),
        log_std: log_std_init.init([1, OUT], device),
    };
    let critic = LimbCritic::<MyBackend> {
        w1: w.init([IN, HIDDEN], device),
        b1: z.init([1, HIDDEN],  device),
        w2: w.init([HIDDEN, 1],  device),
        b2: z.init([1, 1],       device),
    };
    let opt_actor:  Box<dyn BrainOptActor>  = Box::new(AdamConfig::new().init());
    let opt_critic: Box<dyn BrainOptCritic> = Box::new(AdamConfig::new().init());
    SpeciesBrain { actor, critic, opt_actor, opt_critic, training_step: 0 }
}

impl BrainPoolLimb {
    /// Allocate the pool. `n` sizes the per-slot bookkeeping arrays (max
    /// concurrent limb organisms). Per-species nets are NOT built here —
    /// they're created lazily (warm-started) by `ensure_species` as species
    /// appear, so the pool starts with no nets and grows one per live species.
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
            prev_action:        vec![[0.0; OUT]; n],
            prev_predations:    vec![0u8; n],
            prev_reproductions: vec![0u8; n],
            prev_nearest_prey_dist: vec![None; n],
            prev_heading_alignment: vec![None; n],
            prev_limb_contact:      vec![[0.0; 9]; n],
            slot_species:           vec![UNCLASSIFIED; n],
            ticks_since_train: 0,
            training_step: 0,
            training_history: VecDeque::with_capacity(LIMB_TRAINING_HISTORY_CAP),
        }
    }

    /// Lazily create a species' shared brain (warm-started gait prior) on first
    /// sighting. Idempotent. New species fork off existing ones via the
    /// speciation system; their net simply starts fresh from the prior.
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
        self.slot_species[s as usize] = UNCLASSIFIED;
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
    ///   * Builds an observation vector.
    ///   * Runs ONE batched actor + critic forward pass for every active slot.
    ///   * Samples an action `μ + ε * exp(LOG_STD_INIT)` per slot.
    ///   * Writes the action to `Organism::limb_targets`.
    ///   * Appends a `RolloutEntry` (reward + critic value).
    ///   * Triggers a batched `train` every `ROLLOUT_LEN` ticks.
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

        // ── 1. Collect active slots + observations, grouped by SPECIES. Each
        //       organism's CURRENT species (read fresh every tick, so a
        //       reclassification by the speciation system needs no handling) is
        //       recorded in `slot_species` for `train`/`snapshot`. ──
        let mut input_buf: Vec<f32> = vec![0.0; self.n * IN];
        let mut active: Vec<(Entity, u32)> = Vec::new();
        let mut groups: HashMap<u32, Vec<usize>> = HashMap::new();
        let default_phys = LimbObsInputs::default();
        for (e, organism, slot) in organisms.iter() {
            let s = slot.slot() as usize;
            if s >= self.n { continue; }
            let prev_action = self.prev_action[s];
            let phys = obs_inputs.get(&e).unwrap_or(&default_phys);
            let obs = build_observation(&organism, &prev_action, phys, phase);
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
            let mu = brain.actor.forward(obs_t.clone()).into_data().into_vec::<f32>().expect("limb actor forward");
            let v  = brain.critic.forward(obs_t).into_data().into_vec::<f32>().expect("limb critic forward");
            for (i, &s) in slots.iter().enumerate() {
                mu_buf[s * OUT..s * OUT + OUT].copy_from_slice(&mu[i * OUT..i * OUT + OUT]);
                v_buf[s] = v[i];
            }
        }
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
            // Ground gate ∈ [0,1]: movement only counts as locomotion when feet
            // are planted. A ballistic hop has high XZ speed while AIRBORNE (no
            // feet down, planted_frac→0); without this gate it out-scores a
            // grounded crawl on raw XZ speed, so the policy learns to hop
            // (data: net travel coincided exactly with feet leaving the ground).
            // Gating the speed terms pays only for translating WITH feet on the
            // ground = walking. Morphology-general: counts whatever feet exist.
            let planted_count: f32 = phys.limb_contact[1..].iter().sum();
            let ground_gate = (planted_count / GROUND_GATE_MIN_FEET).clamp(0.0, 1.0);

            // (Swimming organisms are NOT in the limb pools — they train in
            // their own 3D pool, `swimming_movement/swim_ppo.rs`.)
            let dense_reward = if STANDING_TASK {
                // ── STANDING-UPRIGHT reward (dm_control "stand" + legged_gym
                // + quadruped fall-recovery recipe). PRIMARY term is the
                // MULTIPLICATIVE product tall × level × feet-planted, so the
                // policy is paid only when the base is held HIGH (forcing
                // knee/sub-limb extension — the whole-leg requirement), LEVEL,
                // and supported on planted legs. Penalties kill the known
                // failure modes; the gated alive bonus rewards holding it.

                // Uprightness ∈ [0,1]: world-Y of the body's local +Y axis,
                // remapped (1 = perfectly upright, 0.5 = on its side, 0 = inverted).
                let upright01 = (0.5 * (1.0 + phys.base_up[1])).clamp(0.0, 1.0);
                // Tall-posture factor ∈ [0,1]: linear ramp of base clearance toward
                // the leg-length-scaled target (sentinel f32::MAX ⇒ 1, inert).
                let tall = (phys.base_clearance / STAND_HEIGHT_TARGET).clamp(0.0, 1.0);
                // Whole-leg support ∈ [0,1]: planted limb segments incl. sub-limbs.
                let foot_support = (planted_count / STAND_MIN_FEET).clamp(0.0, 1.0);
                // Primary: scores only when tall AND level AND supported.
                let stand = tall * upright01 * foot_support;

                // Action-rate smoothness (mean |Δaction| over swing joints) and a
                // light torque/energy regulariser (mean squared swing action).
                let prev_a = &self.prev_action[su];
                let mut act_rate = 0.0f32;
                let mut torque_reg = 0.0f32;
                for k in 0..MAX_LIMB_JOINTS {
                    act_rate   += (action[k] - prev_a[k]).abs();
                    torque_reg += action[k] * action[k];
                }
                act_rate   /= MAX_LIMB_JOINTS as f32;
                torque_reg /= MAX_LIMB_JOINTS as f32;

                let belly = phys.limb_contact[0];
                let alive = if upright01 > STAND_UPRIGHT_MIN && belly < 0.5 { 1.0 } else { 0.0 };

                  K_STAND      * stand
                - K_BELLY      * belly                          // guiderail: no base-floor contact
                - K_TILT       * (1.0 - upright01)              // guiderail: penalise non-horizontal base
                - K_VERT       * phys.base_lin_vel.y.abs()      // anti-bounce/hover
                - K_SPIN       * base_ang_speed                 // anti-tumble/wobble
                - K_DRIFT      * speed                          // a stand is quiet, not wandering
                - K_TORQUE_REG * torque_reg                     // energy / anti-jamming
                - K_ACTRATE    * act_rate                       // anti-trembling smoothness
                + K_ALIVE      * alive                          // hold the upright stance
            } else {
                K_MOVE     * speed.min(SPEED_REWARD_CAP) * upright_pos * ground_gate
                             + K_FWD      * forward_speed * ground_gate
                             + K_UP       * uprightness * motion_gate
                             - K_IDLE     * (1.0 - motion_gate)
                             + K_PROGRESS * progress
                             + K_HEADING  * heading_gain
                             + K_STEP     * step_events
                             - K_SPIN     * base_ang_speed
                             - K_VERT     * phys.base_lin_vel.y.max(0.0)
                             - K_AIR      * (1.0 - planted_count).max(0.0)
                             // STAND, don't collapse: penalise the BASE (belly)
                             // touching the ground. Belly-resting is otherwise
                             // "free" so the policy never learns to hold the body
                             // UP on its legs (the user saw dogs/runners collapse).
                             // Composes with the airborne penalty + planted-feet
                             // gate → optimum is "feet down, belly up" = walking.
                             - K_BELLY    * phys.limb_contact[0]
                             // Twist is OPT-IN: a mild cost on twist magnitude so the
                             // brain only twists when it buys enough locomotion reward
                             // to offset it (gratuitous twist destabilises the light
                             // body). The twist DOF stays available; it just isn't free.
                             - K_TWIST    * (action[MAX_LIMB_JOINTS..].iter()
                                               .map(|t| t.abs()).sum::<f32>()
                                               / N_LIMB_TWIST_GROUPS as f32)
            };

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

            // Training is not per-slot: the global counter below triggers one
            // batched train across every non-empty slot, with a per-slot mask
            // zeroing out slots that didn't accumulate a full rollout.
        }

        self.ticks_since_train += 1;
        if self.ticks_since_train >= ROLLOUT_LEN {
            self.train(virtual_time_secs);
            self.ticks_since_train = 0;
        }
    }

    /// PER-SPECIES PPO update.
    ///
    /// Triggered by `apply_step` every `ROLLOUT_LEN` ticks. Groups every active
    /// slot by its CURRENT species (`slot_species`, written each apply tick)
    /// and, for each species, pools only THAT species' transitions into one flat
    /// batch `[M, ·]`, computes per-trajectory GAE, normalises advantages within
    /// the species, and runs `PPO_EPOCHS` clipped-surrogate + value-MSE +
    /// entropy steps on that species' own actor/critic. All members of a species
    /// update the same weights (so they converge to one policy); different
    /// species are trained independently (so they diverge). The forward is the
    /// plain 2-D `forward` over the pooled `[M, IN]` batch.
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
        // Entropy of `N(μ, σ)` per dim is `log σ + 0.5·ln(2πe)`. Constant
        // doesn't affect gradients, kept for interpretability.
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
                    .into_data().into_vec::<f32>().expect("limb bootstrap critic forward")
            };

            // ── 1. Per-slot GAE → POOLED flat buffers for THIS species.
            //       Per-slot advantage normalisation is replaced by a single
            //       per-species normalisation (see 1-bis) — the pooled batch is
            //       this species' whole experience window. ──
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

            {
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
                        last_actor_loss = scalar_of(policy_loss.mean());
                        last_total_loss = scalar_of(loss.clone());
                        last_entropy    = scalar_of(entropy);
                    }

                    let actor_clone = brain.actor.clone();
                    let actor_grads = GradientsParams::from_grads(loss.backward(), &brain.actor);
                    brain.actor = brain.opt_actor.step(LR, actor_clone, actor_grads);

                    // Critic: fresh graph (the one above was consumed by backward).
                    let v2 = brain.critic.forward(states_t.clone());
                    let v2_diff = v2 - returns_t.clone();
                    let critic_loss = (v2_diff.clone() * v2_diff).mul_scalar(0.5).mean();
                    if epoch == PPO_EPOCHS - 1 {
                        last_critic_loss = scalar_of(critic_loss.clone());
                    }
                    let critic_clone = brain.critic.clone();
                    let critic_grads = GradientsParams::from_grads(critic_loss.backward(), &brain.critic);
                    brain.critic = brain.opt_critic.step(LR, critic_clone, critic_grads);
                }
                brain.training_step += 1;
            }

            // ── 4. Log this species' update. ──
            self.training_step += 1;
            if self.training_history.len() >= LIMB_TRAINING_HISTORY_CAP {
                self.training_history.pop_front();
            }
            self.training_history.push_back(LimbTrainingStep {
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
        }

        // Clear ALL rollout buffers (every active slot was consumed above).
        for r in &mut self.rollouts { r.clear(); }
    }
}
