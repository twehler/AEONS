// Avian3d physics plugin glue.
//
// This module owns the Avian3d setup for AEONS: registering the
// physics plugin group, spawning a heightfield collider for the
// terrain once `HeightmapSampler` is available, and (later, see
// `colony.rs::spawn_organism`) inserting per-body-part rigid-body /
// collider components.
//
// Per the locomotion plan (Phase 2):
//
//   * Sliding organisms (`Organism::sliding_movement == true`) get a
//     `RigidBody::Kinematic` on the OrganismRoot with a compound
//     collider built from every cell across every body part. The
//     existing `apply_movement` continues to write the root's
//     transform; Avian's kinematic sync follows it so other physics
//     bodies (limb-based organisms) can collide with sliders.
//
//   * Limb-based organisms (`Organism::sliding_movement == false`) get
//     a `RigidBody::Dynamic` PER BODY PART, each with its own compound
//     collider (cells of that part), and a `SphericalJoint` between
//     attached parts (anchor1 = `attachment.origin_local` in the
//     parent, anchor2 = ZERO on the child since limb parts are rebased
//     to a first-cell pivot). The base body is the root of the chain;
//     limbs hang off it through spherical (3-DOF) joints. Brains
//     drive PD target angles into those joints in Phase 4.
//
// Gravity is the Avian default (`Vec3::new(0.0, -9.81, 0.0)`), which
// matches AEONS's `+Y`-up world.

use avian3d::prelude::*;
use bevy::prelude::*;

use crate::world_geometry::{HeightmapSampler, HEIGHTMAP_CELL_SIZE};


pub struct AvianSetupPlugin;

/// XPBD solver substeps per physics step. Avian default is 6. This was
/// briefly raised to 16 to keep the rigid `SphericalJoint`s convergent
/// under high PD torque, but substeps multiply the ENTIRE solver cost
/// linearly and dominated the frame time once dozens of limb organisms
/// (each = several dynamic bodies + joints + contacts) were alive.
/// Lowered to 8 now that joint stability comes from the small joint
/// compliance (`LIMB_JOINT_COMPLIANCE`), proper mass properties, and
/// the damping splits rather than from brute-force substepping — 8 is
/// half the cost of 16 while still well above the default. If joints
/// visibly drift again under fast commands, nudge back up to 10–12
/// before reaching for 16.
use crate::simulation_settings::LIMB_SOLVER_SUBSTEPS;

pub use crate::simulation_settings::LIMB_JOINT_COMPLIANCE;

impl Plugin for AvianSetupPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(PhysicsPlugins::default());
        app.insert_resource(SubstepCount(LIMB_SOLVER_SUBSTEPS));
        // One-shot terrain collider — fires the first time
        // `HeightmapSampler` exists AND no collider has been spawned yet.
        app.add_systems(
            Update,
            spawn_terrain_collider.run_if(
                resource_exists::<HeightmapSampler>
                    .and(not(resource_exists::<TerrainColliderSpawned>)),
            ),
        );
        // PD-style torque application for limb-based organisms — reads
        // the brain's `Organism::limb_targets` and applies torques to
        // the dynamic body-part rigid bodies. Runs in `FixedUpdate`,
        // NOT `Update`: Avian steps the physics in `FixedPostUpdate`,
        // which runs once per `FixedMain` iteration (N times per real
        // frame, scaling with `TimeSpeed`). Avian clears applied forces
        // after each step, so a torque applied once per real frame in
        // `Update` only actuated ~1/N of the physics steps at high
        // speed — the limbs coasted the rest. Running here re-applies
        // the brain's target torque on EVERY physics step, so control
        // authority is invariant to `TimeSpeed` / frame rate.
        // `FixedUpdate` runs before `FixedPostUpdate` within each
        // iteration, so the torque is in place when the step solves.
        app.add_systems(FixedUpdate, apply_limb_pd_torques);
        // Per-entity collision flags for limb-based body parts. Reads
        // `CollisionStart` / `CollisionEnd` messages emitted by Avian's
        // narrow phase.
        app.add_systems(Update, update_limb_contacts);
        // Translate Avian collision events involving limb body parts
        // into the `OrganismContactEvent`s that predation consumes.
        // This is the predation source for every limb-involved contact;
        // `organism_collision.rs` skips those pairs (its `root × local`
        // geometry is invalid for dynamic limb bodies). Ordered before
        // `predation_system` would be ideal, but `OrganismContactEvent`
        // is double-buffered so a one-frame lag (matching the custom
        // collision path, which runs in `Last`) is harmless.
        app.add_systems(Update, emit_limb_contact_events);
    }
}


/// Marker resource: present once the terrain heightfield collider has
/// been spawned. The spawn system run-condition flips off on insertion
/// so the system never fires again.
#[derive(Resource)]
struct TerrainColliderSpawned;

/// Marker component on the terrain collider entity. Useful for tests
/// and for despawning the collider if the world ever changes (the
/// current pipeline loads the world exactly once at startup, so this
/// stays around for the lifetime of the run).
#[derive(Component)]
pub struct TerrainCollider;


/// Event-driven contact flag attached to each limb-based body-part
/// entity at spawn. `true` while the body part is in contact with at
/// least one other collider. The limb-brain observation system reads
/// the flag at brain-tick time as one of the 3 contact dimensions.
///
/// Updated by `update_limb_contacts` from Avian's `CollisionStart` and
/// `CollisionEnd` events. Maintains a tiny per-entity counter so
/// simultaneous contacts with multiple colliders (terrain + another
/// organism + ...) only flip back to `false` once ALL contacts end.
#[derive(Component, Default)]
pub struct LimbContact {
    pub in_contact: bool,
    /// Reference count: how many distinct colliders we're currently in
    /// contact with. `in_contact` mirrors `count > 0`.
    pub count:      u32,
}


/// Last torque commanded by `apply_limb_pd_torques` for this body
/// part, in world frame. Updated every brain-tick frame; read by
/// `dataset_export` to log per-organism `torque_norm`. Pure telemetry
/// — does not influence physics. Inserted at spawn on every limb-
/// based body part (base + limbs).
#[derive(Component, Default, Clone, Copy)]
pub struct LastAppliedTorque(pub Vec3);


/// DIAGNOSTIC OVERRIDE (Tier 1 experiment, May 2026).
///
/// When `Some(v)`, `apply_limb_pd_torques` ignores the brain's
/// `Organism::limb_targets` and applies `v` (world-frame torque, N·m)
/// to every limb body part every frame — base body included. The PD
/// math is skipped entirely. This is the cheapest way to verify
/// whether the force-injection pipeline (`Forces::apply_torque` →
/// Avian integrator → body motion) is wired correctly.
///
/// Expected outcome when `Some(Vec3::new(50.0, 0.0, 0.0))`:
///   * If organisms tumble around, the brain output isn't reaching
///     the controller / Avian is fine → the bug is in the brain →
///     controller path.
///   * If organisms still don't move, `Forces::apply_torque` isn't
///     reaching the integrator → the bug is in the physics setup.
///
/// Set back to `None` once the experiment is done.
///
/// Outcome (May 2026): with `MassPropertiesBundle::from_shape(...)`
/// now wired in at spawn, the test produced visible body rotation —
/// confirming the force-injection pipeline works and the original
/// "frozen body" bug was the uninitialised inverse-inertia default.
/// Kept here (set to `None`) as a quick switch for any future
/// regression where the bodies appear to stop responding to torque.
const CONSTANT_TORQUE_TEST: Option<Vec3> = None;

/// Read Avian's collision start/end messages and toggle each
/// participating body part's `LimbContact` flag. Both colliders in an
/// event get incremented / decremented so the flag captures the body
/// part's perspective regardless of which collider was listed first.
pub fn update_limb_contacts(
    mut started: MessageReader<CollisionStart>,
    mut ended:   MessageReader<CollisionEnd>,
    mut q:       Query<&mut LimbContact>,
) {
    for ev in started.read() {
        if let Ok(mut c) = q.get_mut(ev.collider1) {
            c.count = c.count.saturating_add(1);
            c.in_contact = c.count > 0;
        }
        if let Ok(mut c) = q.get_mut(ev.collider2) {
            c.count = c.count.saturating_add(1);
            c.in_contact = c.count > 0;
        }
    }
    for ev in ended.read() {
        if let Ok(mut c) = q.get_mut(ev.collider1) {
            c.count = c.count.saturating_sub(1);
            c.in_contact = c.count > 0;
        }
        if let Ok(mut c) = q.get_mut(ev.collider2) {
            c.count = c.count.saturating_sub(1);
            c.in_contact = c.count > 0;
        }
    }
}


/// Translate Avian `CollisionStart` events into `OrganismContactEvent`s
/// for every contact that involves a limb-based organism's body part.
///
/// Each limb body part is its own collider with `CollisionEventsEnabled`,
/// so Avian fires `CollisionStart` with the limb part as `collider1`
/// (events are emitted per-events-enabled collider, and only limb parts
/// carry the component — sliding organisms and the terrain do not). The
/// limb part carries `ChildOf` (→ organism root) and `BodyPartIndex`,
/// which is exactly the `(root, body_part)` pair `OrganismContactEvent`
/// needs.
///
/// `collider2` is the other party:
///   * another limb body part → resolved the same way (full granularity);
///   * a sliding/sessile organism's root collider → resolved to its root
///     with a fallback body-part index (the first alive part), since the
///     sliding compound collider has no per-part identity;
///   * the terrain collider (or anything without an organism identity)
///     → skipped.
///
/// Predation only ever reads the PREY's body-part index, so the
/// fallback on the sliding side is sufficient: a sliding prey loses
/// gradient-consumption granularity (always its first alive part), which
/// is acceptable for the photo/sessile organisms that make up that case.
pub fn emit_limb_contact_events(
    mut started:   MessageReader<CollisionStart>,
    limb_parts:    Query<(&bevy::prelude::ChildOf, &crate::cell::BodyPartIndex)>,
    roots:         Query<&crate::colony::Organism, With<crate::colony::OrganismRoot>>,
    mut out:       MessageWriter<crate::organism_collision::OrganismContactEvent>,
) {
    // Resolve a collider entity to `(organism_root, body_part_index)`.
    // Returns `None` for non-organism colliders (e.g., terrain).
    let resolve = |e: Entity| -> Option<(Entity, usize)> {
        if let Ok((child_of, bp_idx)) = limb_parts.get(e) {
            // Limb body part: root is its parent, index is exact.
            return Some((child_of.parent(), bp_idx.0));
        }
        if let Ok(org) = roots.get(e) {
            // Sliding/sessile organism: the collider sits on the root,
            // which has no per-part identity. Fall back to the first
            // alive body part. (Predation tolerates a stale index via
            // its own `is_alive()` guard, but first-alive avoids
            // wasting the contact on an already-eaten slot.)
            let bp = org.body_parts.iter().position(|b| b.is_alive()).unwrap_or(0);
            return Some((e, bp));
        }
        None
    };

    for ev in started.read() {
        let Some((root_a, bp_a)) = resolve(ev.collider1) else { continue };
        let Some((root_b, bp_b)) = resolve(ev.collider2) else { continue };
        // Self-contact (two body parts of the same organism touching)
        // is not predation — skip. The brain still "feels" it via the
        // `LimbContact` flag updated in `update_limb_contacts`.
        if root_a == root_b { continue; }
        out.write(crate::organism_collision::OrganismContactEvent {
            a: root_a,
            b: root_b,
            body_part_a: bp_a,
            body_part_b: bp_b,
        });
    }
}


/// Build a 3D heightfield collider from `HeightmapSampler` and spawn it
/// as a single static rigid body covering the world's XZ footprint.
///
/// Avian3d's heightfield takes a `Vec<Vec<Scalar>>` of heights and a
/// `Vector` scale that defines the full XZ extent of the field. Heights
/// land between `±0.5 * scale.y` *around the entity's Y* — i.e. the
/// field is centered on its entity's position. We position the entity
/// so the heightfield aligns with the world coordinates the rest of
/// the codebase uses: XZ center at the middle of the heightmap, Y
/// chosen so the field's surface matches the original heights.
fn spawn_terrain_collider(
    mut commands: Commands,
    heightmap:    Res<HeightmapSampler>,
) {
    if heightmap.width == 0 || heightmap.depth == 0 { return; }

    let w = heightmap.width  as usize;
    let d = heightmap.depth  as usize;

    // Heights are stored as a flat row-major buffer indexed by
    // `zi * width + xi`. Convert to the `Vec<Vec<f32>>` layout Avian
    // expects (outer = rows, inner = columns).
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(d);
    for zi in 0..d {
        let mut row = Vec::with_capacity(w);
        for xi in 0..w {
            row.push(heightmap.heights[zi * w + xi]);
        }
        rows.push(row);
    }

    // The heightfield covers world space `(0, 0)..(w*HCS, d*HCS)` after
    // the world-normalisation step (which translates the AABB min
    // corner to the origin). Avian centres the field on its entity,
    // so place the entity at the XZ midpoint and `Y = 0` (the heights
    // themselves carry the vertical positioning).
    let scale_x = w as f32 * HEIGHTMAP_CELL_SIZE;
    let scale_z = d as f32 * HEIGHTMAP_CELL_SIZE;
    let entity_pos = Vec3::new(scale_x * 0.5, 0.0, scale_z * 0.5);
    let scale = Vec3::new(scale_x, 1.0, scale_z);

    commands.spawn((
        TerrainCollider,
        RigidBody::Static,
        Collider::heightfield(rows, scale),
        Transform::from_translation(entity_pos),
    ));
    commands.insert_resource(TerrainColliderSpawned);
}


// ── PD-on-angle torque controller ────────────────────────────────────────────

/// Position gain — torque-per-radian. Multiplies the joint-angle error
/// (target − current) to produce the restoring torque component.
///
/// HIGH-gain PD position control (legged_gym / Isaac-Gym style): the
/// gain must be large enough that, when the brain commands a target
/// angle "into the ground" and the contact blocks the joint, the
/// residual-error torque can actually drive the body upward — i.e.
/// `KP × max_error ≳ body weight × lever`. At the old `KP = 12` the
/// max spring torque (12 × MAX_JOINT_ANGLE ≈ 25 N·m) was below the
/// organism's weight, so a planted limb could never press the body
/// up — a concrete reason locomotion never emerged. `KP = 40` gives
/// ≈ 84 N·m of stance authority, comfortably above a density-0.2
/// organism's weight.
use crate::simulation_settings::KP_TORQUE;

use crate::simulation_settings::KD_TORQUE;

use crate::simulation_settings::MAX_JOINT_ANGLE;

/// Read `Organism::limb_targets` on each limb-based organism and apply
/// PD-on-angle torques to its dynamic limb rigid bodies. The base body
/// (`BodyPartIndex(0)`) is skipped — the brain only commands appendages.
///
/// Joint angle is computed as the relative rotation between the limb
/// and its logical parent body-part (resolved via
/// `Organism::body_parts[i].attachment.parent_idx`). The brain's
/// target unit-vector is scaled by `MAX_JOINT_ANGLE` to get the desired
/// Euler-XYZ angle in the parent's local frame. Torque is then
/// `KP * (target − current) − KD * ω`, rotated back into world frame by
/// the parent's orientation before `apply_torque` (which is world-frame
/// in Avian).
///
/// Mapping from body-part index to `limb_targets` slice:
///
///   * idx 1, 2 → pair 0 (`limb_targets[0..3]`) — first appendage pair
///   * idx 3, 4 → pair 1 (`limb_targets[3..6]`) — second appendage pair
///
/// The LEFT half of a pair (odd offset within a pair) gets its X-axis
/// target sign-flipped so a mirrored pair swings in opposite directions.
pub fn apply_limb_pd_torques(
    sim_running: Res<crate::simulation_settings::SimulationRunning>,
    organisms: Query<&crate::colony::Organism>,
    bp_rot:    Query<(&crate::cell::BodyPartIndex, &bevy::prelude::ChildOf, &Rotation), With<RigidBody>>,
    mut bp_q:  Query<(
        Entity,
        &crate::cell::BodyPartIndex,
        &bevy::prelude::ChildOf,
        &Rotation,
        Forces,
    ), With<RigidBody>>,
    mut torque_log: Query<&mut LastAppliedTorque>,
) {
    // Do NOT apply torque while the simulation is paused (e.g. the
    // user switched to the Colony / Species editor, which pauses
    // `Time<Virtual>` and thereby freezes Avian's physics step).
    // `Forces::apply_torque` ACCUMULATES into each body's integration
    // buffer and that buffer is only cleared when a physics step runs.
    // With the step frozen, every paused frame adds another dose of
    // angular acceleration; the first resumed step then integrates the
    // whole accumulated pile at once and the organisms explode outward.
    // Gating on `SimulationRunning` keeps the buffer from accumulating
    // while paused. (The brain apply systems are already paused — they
    // run on a virtual-time `on_timer`.)
    if !sim_running.0 { return; }

    // ── Tier 1 diagnostic shortcut. If `CONSTANT_TORQUE_TEST` is on,
    // bypass the brain + PD math entirely and inject a fixed world-
    // frame torque into every limb body part (including the base, so
    // we get a strong directly-observable spin if Avian is integrating
    // anything). The early return below skips the entire normal
    // pipeline so we can be sure the only forces in play are the
    // constants we wrote here.
    if let Some(test_torque) = CONSTANT_TORQUE_TEST {
        for (e, _idx, parent, _rot, mut forces) in &mut bp_q {
            let Ok(organism) = organisms.get(parent.parent()) else { continue };
            if organism.sliding_movement { continue; }
            forces.apply_torque(test_torque);
            if let Ok(mut last) = torque_log.get_mut(e) { last.0 = test_torque; }
        }
        return;
    }

    // (organism_root, body_part_idx) → world rotation. Built in one pass
    // so the actuator loop can look up each limb's logical parent
    // without traversing the Bevy hierarchy (limbs are siblings of the
    // base under `OrganismRoot`, joints are separate entities).
    let mut rot_map: std::collections::HashMap<(Entity, usize), Quat> =
        std::collections::HashMap::new();
    for (idx, parent, rot) in &bp_rot {
        rot_map.insert((parent.parent(), idx.0), rot.0);
    }

    for (e, idx, parent, child_rot, mut forces) in &mut bp_q {
        let i = idx.0;
        if i == 0 {
            // Base body — record zero torque (we never command it).
            if let Ok(mut last) = torque_log.get_mut(e) { last.0 = Vec3::ZERO; }
            continue;
        }
        let Ok(organism) = organisms.get(parent.parent()) else { continue };
        if organism.sliding_movement { continue; }
        if i >= organism.body_parts.len() { continue; }
        let Some(attach) = organism.body_parts[i].attachment.as_ref() else { continue };
        let Some(&parent_rot) = rot_map.get(&(parent.parent(), attach.parent_idx))
            else { continue };

        // Relative rotation in the parent's frame, decomposed to
        // Euler-XYZ. Note: Euler decomposition has well-known
        // singularities; for limb swings within ±π/2 the XYZ extraction
        // is well-conditioned.
        let q_rel = parent_rot.inverse() * child_rot.0;
        let (cx, cy, cz) = q_rel.to_euler(EulerRot::XYZ);
        let current = Vec3::new(cx, cy, cz);

        let pair_idx = ((i - 1) / 2).min(1);
        let half_idx = (i - 1) % 2;
        let base = pair_idx * 3;
        let mut target_unit = Vec3::new(
            organism.limb_targets[base],
            organism.limb_targets[base + 1],
            organism.limb_targets[base + 2],
        );
        if half_idx == 1 { target_unit.x = -target_unit.x; }
        let target = target_unit * MAX_JOINT_ANGLE;

        let ang_vel_world = forces.angular_velocity();
        // Spring term lives in parent's local frame; rotate to world.
        // KD damping is applied directly on world-frame ω.
        let local_spring = KP_TORQUE * (target - current);
        let torque = parent_rot * local_spring - KD_TORQUE * ang_vel_world;
        forces.apply_torque(torque);
        // Telemetry: record the world-frame torque magnitude so the
        // dataset exporter can answer "is the controller producing
        // anything at all?" without instrumenting Avian itself.
        if let Ok(mut last) = torque_log.get_mut(e) { last.0 = torque; }
    }
}
