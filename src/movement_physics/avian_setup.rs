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
//     collider (cells of that part), and a `RevoluteJoint` (1-DOF hinge
//     + in-solver spring-damper motor) between attached parts. The joint
//     is anchored at the limb's FIRST cell centre (anchor2 = ZERO on the
//     child since limb parts are rebased to that pivot; anchor1 = the same
//     point in the parent's frame), so the limb rotates about — and can
//     never separate from — its attachment point. The base body is the
//     root of the chain; limbs hang off it through these hinges. The brain
//     drives each hinge's target angle directly (`drive_limb_motors`), so
//     walking EMERGES from learned per-joint motion.
//
// Gravity is the Avian default (`Vec3::new(0.0, -9.81, 0.0)`), which
// matches AEONS's `+Y`-up world.

use avian3d::prelude::*;
use bevy::prelude::*;
use bevy::ecs::system::SystemParam;

use crate::world_geometry::{HeightmapSampler, HEIGHTMAP_CELL_SIZE};


/// Avian collision-filter hook: reject contacts between two colliders that
/// belong to the SAME organism. Every body part is a flat child of its
/// `OrganismRoot` (`commands.entity(root).add_child(child)` at spawn), so
/// two body parts of one organism share the same `ChildOf` parent. A
/// creature's own parts are held together by the joint chain; letting them
/// ALSO collide makes the contact solver fight the joints (contact pushes
/// apart, joint pulls together), and on the very light limb bodies that
/// conflict erupts into explosive linear+angular velocity (the "flying"
/// bug) and floods Avian's contact graph (the `prepare_contact_constraints`
/// panic). Filtering same-organism pairs removes both at the source while
/// leaving inter-organism collisions and limb↔terrain contacts intact.
///
/// Activated per collider by the `ActiveCollisionHooks::FILTER_PAIRS`
/// component (added to every limb body part at spawn); registered via
/// `PhysicsPlugins::with_collision_hooks`.
#[derive(SystemParam)]
pub struct SelfCollisionFilter<'w, 's> {
    parents: Query<'w, 's, &'static ChildOf>,
}

impl CollisionHooks for SelfCollisionFilter<'_, '_> {
    fn filter_pairs(&self, collider1: Entity, collider2: Entity, _commands: &mut Commands) -> bool {
        match (self.parents.get(collider1), self.parents.get(collider2)) {
            // Same organism root → same creature → don't collide.
            (Ok(p1), Ok(p2)) => p1.parent() != p2.parent(),
            // One/both have no parent (e.g. terrain, or a sliding organism's
            // single root collider) → allow the contact.
            _ => true,
        }
    }
}


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
        app.add_plugins(PhysicsPlugins::default().with_collision_hooks::<SelfCollisionFilter>());
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
        // Sweep dangling/leaked limb joints BEFORE the physics step so the
        // solver never processes a joint whose body was despawned.
        app.add_systems(PreUpdate, cleanup_orphaned_limb_joints);
        // EMERGENT locomotion: the brain drives each limb hinge's target angle
        // directly (no CPG generating the rhythm, no pursuit force aiming the
        // body). `drive_limb_motors` just copies `Organism::limb_targets` onto
        // the hinge motors. Runs in `FixedUpdate` (NOT `Update`): Avian steps
        // the physics in `FixedPostUpdate`, N times per real frame at high
        // `TimeSpeed`, and `FixedUpdate` runs first within each iteration — so
        // the latest setpoint is in place when each step solves, invariant to
        // `TimeSpeed` / frame rate. (The motor setpoint also persists on the
        // joint, so Avian re-applies it every substep regardless.)
        app.add_systems(FixedUpdate, drive_limb_motors);
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
        // Per-part non-penetration floor: pushes only the body parts that dip
        // below the terrain back up to the surface (legs fold; nothing is
        // hoisted), so the world is solid concrete AND the posture stays natural
        // (belly resting at the surface, no parts hanging in the air). Runs
        // after Avian's step is synced for the frame; gated on the heightmap.
        app.add_systems(
            PostUpdate,
            enforce_limb_floor.run_if(resource_exists::<HeightmapSampler>),
        );
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


/// Last torque commanded by `drive_limb_motors` for this body
/// part, in world frame. Updated every brain-tick frame; read by
/// `dataset_export` to log per-organism `torque_norm`. Pure telemetry
/// — does not influence physics. Inserted at spawn on every limb-
/// based body part (base + limbs).
#[derive(Component, Default, Clone, Copy)]
pub struct LastAppliedTorque(pub Vec3);


/// Marker on each limb `RevoluteJoint` entity linking it back to the
/// organism + body part it drives. `drive_limb_motors` queries these
/// to set the joint motor's `target_position` from the brain's
/// `Organism::limb_targets` each tick. `limb_entity` is the limb body
/// part's entity (for `LastAppliedTorque` telemetry); `body_part` is its
/// `BodyPartIndex` (used to pick the matching brain output `limb_targets[body_part-1]`).
#[derive(Component, Clone, Copy)]
pub struct LimbJointDrive {
    pub organism:    Entity,
    pub body_part:   usize,
    pub limb_entity: Entity,
}

/// Despawn limb `RevoluteJoint` entities whose constrained bodies no longer
/// exist. The joints are standalone entities (NOT children of the organism
/// root), so when a body part is eaten (`predation`) or an organism dies
/// (`energy`), nothing despawns the joints that referenced those bodies:
/// they LEAK (accumulating forever over a long run) and DANGLE (pointing at
/// despawned entities). A dangling joint feeds Avian's solver a stale body
/// reference, which under heavy collider churn corrupts the contact graph
/// and panics in `prepare_contact_constraints`. This sweep removes any limb
/// joint with a missing endpoint. It runs in `PreUpdate` — before the
/// `FixedMain` physics step — so the next solve never sees a dangling joint.
pub fn cleanup_orphaned_limb_joints(
    mut commands: Commands,
    joints:       Query<(Entity, &avian3d::prelude::RevoluteJoint), With<LimbJointDrive>>,
    bodies:       Query<(), With<RigidBody>>,
) {
    for (joint_entity, joint) in &joints {
        // Both endpoints are limb body parts (Dynamic rigid bodies). If
        // either no longer has a RigidBody, it was despawned → orphaned.
        if bodies.get(joint.body1).is_err() || bodies.get(joint.body2).is_err() {
            commands.entity(joint_entity).try_despawn();
        }
    }
}


/// Hard non-penetration floor for limb-based organisms — the world behaves like
/// SOLID CONCRETE: no body part may sink below the terrain surface.
///
/// Avian's contact solver alone cannot guarantee this here: a limb organism
/// rests its belly on the terrain while its RIGID leg joints (point_compliance
/// = 0) demand the feet sit a fixed offset BELOW the body — geometrically below
/// the surface. The belly contact (pinning the body down) and the foot contacts
/// (pushing up) are mutually incompatible, so the finite-substep solver leaves
/// the feet penetrating ~0.8–1.1 units (measured). That sinks the feet into the
/// "concrete" and impairs locomotion.
///
/// This system enforces non-penetration PER PART (not a whole-body hoist): for
/// each body part whose collider bottom is below the terrain, it lifts ONLY that
/// part up to the surface and zeroes its downward velocity. A whole-organism
/// rigid lift (the earlier version) raised the belly far above the ground to
/// keep the lowest foot at the surface — leaving the body and other feet
/// "hanging in the air" (the unnatural posture). Lifting each penetrating part
/// individually instead lets the body settle NATURALLY: the belly rests at the
/// surface and the lower legs/feet — which would otherwise jut below it — are
/// pushed up to the surface too, the hinge joints simply folding to absorb the
/// motion (they can't separate — the joint is a positional constraint, not a
/// translation). Result: nothing penetrates AND nothing floats; the creature
/// lies/stands with all its lowest points resting on the ground, gravity
/// respected. A small margin keeps the collider bottoms a hair above the surface
/// so they read as resting on it rather than flickering in/out of contact.
///
/// Runs in `PostUpdate` (after Avian syncs the step's results), mirroring the
/// sliding organisms' `apply_floor_collision`. Gated on `HeightmapSampler`.
pub fn enforce_limb_floor(
    heightmap: Res<HeightmapSampler>,
    mut parts: Query<
        (&crate::cell::BodyPartIndex, &ColliderAabb, &mut Position, &mut LinearVelocity),
        With<LimbContact>,
    >,
) {
    const FLOOR_MARGIN: f32 = -0.08;
    // Cap the per-frame correction so the floor nudges a sunk part up gently
    // over a few frames rather than TELEPORTING it (a large instantaneous jump
    // injects energy through the joint and can fling a part — the rare deep
    // outlier seen in the data). At the physics rate a few-frame convergence is
    // imperceptible while staying non-penetrating.
    const MAX_LIFT_PER_STEP: f32 = 3.0;
    // The BASE belly gets a deadzone: it may graze / lightly contact the ground
    // (its low-friction slide is the locomotion substrate and must stay
    // undisturbed), and is only pushed back when it dips DEEPER than this — so
    // the belly never deeply sinks but its sliding dynamics, hence the directed
    // limb-propelled movement, are preserved. The LIMB feet (the "sub-limbs"
    // the user saw sinking) are floored hard to the surface (deadzone 0).
    const BASE_DEADZONE: f32 = 0.2;
    for (bp_idx, aabb, mut position, mut linvel) in parts.iter_mut() {
        let deadzone = if bp_idx.0 == 0 { BASE_DEADZONE } else { 0.0 };
        let cx = 0.5 * (aabb.min.x + aabb.max.x);
        let cz = 0.5 * (aabb.min.z + aabb.max.z);
        let terrain = heightmap.height_at(cx, cz);
        let pen = (terrain + FLOOR_MARGIN) - aabb.min.y - deadzone;   // > 0 ⇒ below the allowed level
        if pen > 0.0 {
            position.0.y += pen.min(MAX_LIFT_PER_STEP);
            // Only cancel a DOWNWARD velocity that is faster than the gentle
            // settle, so the leg's propulsion stroke (and the body weight
            // pressing the feet for grip) is preserved — we don't freeze the
            // part, just stop it driving further into the ground.
            if linvel.0.y < -0.5 { linvel.0.y = -0.5; }
        }
    }
}


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


// ── Limb hinge-motor target driver (EMERGENT, brain-driven) ──────────────────

use crate::simulation_settings::{LIMB_SWING_LIMIT, MAX_LIMB_JOINTS};

/// Drive each limb's hinge toward the brain's target angle by writing the
/// `RevoluteJoint` **angular motor**'s `target_position` DIRECTLY from the
/// brain's per-joint output. The motor itself (a stable
/// `MotorModel::SpringDamper`, configured at spawn) runs inside Avian's XPBD
/// solver every substep and tracks the setpoint; this system only refreshes
/// the setpoint each physics step.
///
/// This is the EMERGENT-locomotion contract: there is NO central pattern
/// generator producing the rhythm and NO pursuit force aiming the body. The
/// brain must learn — per joint — a phase-locked angle trajectory whose ground
/// reaction nets forward thrust (the reward in `limb_ppo.rs` shapes this). The
/// brain is handed a phase-clock signal as an *observation* (see
/// `build_observation`) so a feedforward policy can phase-lock a sustained
/// oscillation, but it generates every joint command itself.
///
/// Mapping: brain output `limb_targets[i-1]` (∈ [-1, 1]) → the hinge of
/// body-part index `i`, scaled to a target hinge angle in
/// `[-LIMB_SWING_LIMIT, +LIMB_SWING_LIMIT]`. Each runtime limb part (including
/// every Bilateral-mirror half and every multi-segment knee) thus gets its own
/// independent command, so an alternating / coordinated gait can emerge rather
/// than being imposed. Parts beyond `MAX_LIMB_JOINTS` wrap modulo (rare).
///
/// The in-solver motor acts on the hinge axis ONLY (1-DOF — structurally cannot
/// push off-hinge), uses implicit-Euler integration (unconditionally stable),
/// and reads the true hinge angle via `atan2` (no Euler decomposition → no
/// gimbal singularity), which is why it has none of the failure modes of the
/// old external `Forces::apply_torque` PD-on-Euler controller.
pub fn drive_limb_motors(
    sim_running: Res<crate::simulation_settings::SimulationRunning>,
    organisms:   Query<&crate::colony::Organism>,
    mut joints:  Query<(&LimbJointDrive, &mut avian3d::prelude::RevoluteJoint)>,
    mut torque_log: Query<&mut LastAppliedTorque>,
) {
    // Skip while paused: the motor setpoint persists on the joint component
    // (Avian re-applies it every substep), so there is nothing to refresh.
    if !sim_running.0 { return; }

    for (drive, mut joint) in &mut joints {
        let Ok(organism) = organisms.get(drive.organism) else { continue };
        if organism.sliding_movement { continue; }
        let i = drive.body_part;
        if i == 0 || i >= organism.body_parts.len() { continue; }

        // Pick this joint's brain output: body-part i ↔ output (i-1), wrapping
        // beyond MAX_LIMB_JOINTS so deeper parts still get a (shared) command.
        let out_idx = (i - 1) % MAX_LIMB_JOINTS;
        let cmd = organism.limb_targets[out_idx].clamp(-1.0, 1.0);
        let target = cmd * LIMB_SWING_LIMIT;

        joint.motor.target_position = target;
        joint.motor.enabled = true;

        // Telemetry proxy: record the commanded hinge angle.
        if let Ok(mut last) = torque_log.get_mut(drive.limb_entity) {
            last.0 = Vec3::new(target, 0.0, 0.0);
        }
    }
}
