// bevy_rapier3d physics glue (migrated from Avian/XPBD). Registers the plugin,
// spawns the terrain heightfield collider once `HeightmapSampler` exists, and
// (in `colony.rs::spawn_organism`) inserts per-body-part rigid-body/collider
// components.
//
//   * Sliding organisms (`movement_mode.is_sliding()`): `RigidBody::KinematicPositionBased`
//     on the root + compound collider over all cells. `apply_movement` writes
//     the root transform; Rapier's kinematic sync follows it.
//   * Limb organisms (`!movement_mode.is_sliding()`): `RigidBody::Dynamic` PER
//     body part, connected to their attachment parent by a joint chosen by
//     movement mode + part kind:
//       - WALKERS/standers: a maximal-coordinate `ImpulseJoint` (revolute hinge
//         + in-solver position motor), driven by `drive_limb_motors`. (Walkers
//         stay on impulse joints: the reduced-coordinate path amplified a
//         low-gravity standing warm-start fling — see the architecture doc.)
//       - SWIMMERS: a REDUCED-COORDINATE `MultibodyJoint` (spherical, 3-DOF ball
//         + per-axis position motors), driven by `drive_swim_motors`. In a
//         Featherstone articulation the child's pose is a function of (parent
//         pose, joint angle) with NO translational DOF in the state vector, so
//         the joint anchors are mathematically INCAPABLE of separating under any
//         force — the definitive fix for chained segments drifting apart.
//       - STATIC parts: a rigid `FixedJoint` (no motor, no brain link).
//     `LimbJointDrive` is inserted alongside every joint (motors + telemetry).
//
// The joint is a COMPONENT on the CHILD limb entity (referencing the parent
// body entity), unlike Avian's standalone joint entities.
//
// Gravity is the Rapier default (-9.81 Y), matching the +Y-up world.

use bevy::prelude::*;
use bevy::ecs::system::SystemParam;
use bevy_rapier3d::prelude::*;

use crate::world_geometry::{HeightmapSampler, HEIGHTMAP_CELL_SIZE};

pub use crate::simulation_settings::LIMB_JOINT_COMPLIANCE;
use crate::simulation_settings::{
    LIMB_SWING_LIMIT, MAX_LIMB_JOINTS, N_LIMB_TWIST_GROUPS,
    LIMB_MOTOR_STIFFNESS, LIMB_MOTOR_DAMPING,
    MAX_LIMB_LINEAR_SPEED, MAX_LIMB_ANGULAR_SPEED,
    LIMB_SOLVER_ITERATIONS, LIMB_STABILIZATION_ITERATIONS,
};
use crate::simulation_settings::{STANDING_TASK, STANDING_GRAVITY_START, STANDING_GRAVITY_FADE_SECS};
use crate::simulation_settings::{
    WATER_DENSITY, SWIM_DRAG_CD, SWIM_ANGULAR_DRAG_COEF,
    SWIM_BORDER_STIFFNESS, SWIM_CONFINE_MAX_FORCE,
    SWIM_DRAG_MAX_FORCE, SWIM_DRAG_MAX_TORQUE,
    SWIM_MAX_LIMB_TORQUE,
};
use crate::cell::{Cell, CELL_COLLISION_RADIUS};
use crate::environment::WaterLevel;
use crate::world_geometry::MapSize;


/// Contact-filter hook: reject contacts between two colliders of the SAME
/// organism (parts share a `ChildOf` root). Without this the contact solver
/// fights the joint chain (contact pushes apart, joint pulls together), which
/// on light limb bodies erupts into explosive velocity. Inter-organism and
/// limb↔terrain contacts stay intact. Activated per collider via
/// `ActiveHooks::FILTER_CONTACT_PAIRS`.
#[derive(SystemParam)]
pub struct SelfCollisionFilter<'w, 's> {
    parents: Query<'w, 's, &'static ChildOf>,
    orgs:    Query<'w, 's, (&'static crate::colony::Organism, bevy::prelude::Has<crate::colony::Heterotroph>)>,
    photos:  Query<'w, 's, (), bevy::prelude::With<crate::colony::Photoautotroph>>,
}

impl SelfCollisionFilter<'_, '_> {
    /// Resolve a collider entity to its organism root (limb part → parent; a sliding
    /// root collider → itself).
    fn root_of(&self, e: bevy::prelude::Entity) -> bevy::prelude::Entity {
        self.parents.get(e).map(|c| c.parent()).unwrap_or(e)
    }
    /// A limb (dynamic) heterotroph — the predators that cluster on prey.
    fn is_limb_hetero(&self, root: bevy::prelude::Entity) -> bool {
        self.orgs.get(root).map(|(o, h)| h && !o.movement_mode.is_sliding()).unwrap_or(false)
    }
    fn is_photo(&self, root: bevy::prelude::Entity) -> bool { self.photos.get(root).is_ok() }
}

impl BevyPhysicsHooks for SelfCollisionFilter<'_, '_> {
    fn filter_contact_pair(&self, context: PairFilterContextView) -> Option<SolverFlags> {
        let e1 = context.collider1();
        let e2 = context.collider2();
        let r1 = self.root_of(e1);
        let r2 = self.root_of(e2);
        // Same organism → don't collide (intra-body parts).
        if r1 == r2 { return None; }
        // Two different limb herbivores → don't collide. They cluster densely when
        // they all seek the same prey; the O(n²) contact pile craters the frame rate
        // and they don't need to physically block each other (only prey + terrain do).
        if self.is_limb_hetero(r1) && self.is_limb_hetero(r2) { return None; }
        // Limb herbivore ↔ phototroph → don't physically collide. Eating is by
        // PROXIMITY (emit_proximity_predation), so the herbivore passes through prey
        // instead of forming a dense limb↔prey contact pile at feeding clusters that
        // craters the frame rate. Predation still happens (proximity event).
        if (self.is_limb_hetero(r1) && self.is_photo(r2)) || (self.is_limb_hetero(r2) && self.is_photo(r1)) {
            return None;
        }
        // Otherwise (limb↔terrain for the floor, sliding↔sliding, etc.) → solve.
        Some(SolverFlags::COMPUTE_IMPULSES)
    }
}


pub struct RapierSetupPlugin;

impl Plugin for RapierSetupPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(
            RapierPhysicsPlugin::<SelfCollisionFilter>::default()
                .in_fixed_schedule(),
        );
        // Attach deferred joints once bodies are registered: walkers/standers get
        // an `ImpulseJoint` per part; swimmers get a single atomic reduced-coordinate
        // `MultibodyJoint` articulation (separation impossible by construction);
        // statics get a rigid `FixedJoint`.
        app.add_systems(PreUpdate, (
            attach_pending_limb_joints,
            attach_pending_swimmer_multibody,
            attach_pending_fixed_joints,
        ));
        // BULLETPROOF multibody despawn teardown — tears out a whole articulation
        // in one idempotent call before bevy_rapier's per-body removal hits
        // rapier3d 0.32's buggy per-joint `remove` (issue #382, the
        // `multibody_joint_set.rs:223` `.unwrap()` panic). `sync_removals` runs in
        // TWO places when physics is `.in_fixed_schedule()`: in `PhysicsSet::SyncBackend`
        // (FixedUpdate) AND, because removal-detection must not miss events, in
        // PostUpdate (bevy_rapier plugin.rs:292-297). A despawn in `Update` (e.g. the
        // colony editor's CLEAR, predation, the limb cull) is processed by the
        // PostUpdate copy that SAME frame — before any FixedUpdate runs — so the net
        // MUST precede BOTH copies. (An earlier FixedUpdate-only net "passed" the cull
        // only by luck: the cull fires ~1 frame post-spawn, often before the atomic
        // multibody attach, so the culled bodies had no articulation yet.)
        app.add_systems(FixedUpdate, teardown_multibody_on_removal.before(PhysicsSet::SyncBackend));
        // Non-finite safety net — remove any organism whose body went NaN/inf
        // BEFORE the solver can touch it (it would otherwise panic). Runs before
        // the physics step.
        app.add_systems(FixedUpdate, cull_nonfinite_bodies.before(PhysicsSet::SyncBackend));
        app.add_systems(
            PostUpdate,
            teardown_multibody_on_removal.before(bevy_rapier3d::plugin::systems::sync_removals),
        );
        // EXPLICIT multibody-link transform sync: overwrite swimmer parts' Bevy
        // transforms from the authoritative Rapier pose AFTER bevy_rapier's
        // (unreliable, for nested multibody children) writeback and BEFORE Bevy
        // propagates GlobalTransform — so the rendered/colliding pose matches the
        // articulation and the parts visibly stay pinned to their joints.
        app.add_systems(
            PostUpdate,
            sync_multibody_link_transforms
                .after(PhysicsSet::Writeback)
                .before(bevy::transform::TransformSystems::Propagate),
        );
        // Zero the persistent ExternalForce at the TOP of the step so the twist
        // couple + swimmer drag + confinement ACCUMULATE onto a clean slate.
        // Safety net first: scrub any non-finite swimmer velocity before anything
        // (forces, then the physics step) can read it and panic the solver.
        app.add_systems(FixedUpdate, sanitize_swimmer_velocity.before(reset_external_forces));
        // MUST run before EVERY ExternalForce writer (`drive_limb_twist`,
        // `apply_fluid_drag`, `confine_swimmers`), not just `drive_limb_motors`
        // (which writes joint motors, NOT ExternalForce). Ordering only against
        // drive_limb_motors left the reset unordered vs the force accumulators, so
        // the scheduler could zero the force AFTER it was accumulated — silently
        // dropping the drag on some frames. `.before(drive_limb_twist)` puts it
        // ahead of the whole chain (twist → drag → confine are already chained).
        app.add_systems(FixedUpdate,
            reset_external_forces
                .before(drive_limb_motors)
                .before(drive_limb_twist));
        // EMERGENT locomotion: copy `limb_targets` onto the hinge motors and
        // apply the twist couple; `drive_swim_motors` does the same for swimmer
        // BALL joints from `SwimJointTargets`. FixedUpdate so the setpoint is
        // fresh per step.
        app.add_systems(FixedUpdate, (drive_limb_motors, drive_swim_motors, drive_limb_twist));
        // SWIMMING: anisotropic blade-element fluid drag (reaction propels/turns
        // the body) then soft XZ world borders. After the twist driver (so they
        // ADD to ef.torque), before the velocity clamp.
        app.add_systems(FixedUpdate,
            apply_fluid_drag
                .after(drive_limb_twist)
                .before(clamp_limb_velocity));
        // Confinement runs after the drag force is applied and before the velocity
        // clamp. (No edge vs `steer_base_toward_prey` is needed: that system
        // excludes swimmers via `Without<SwimmerBody>`.)
        app.add_systems(FixedUpdate,
            confine_swimmers
                .after(apply_fluid_drag)
                .before(clamp_limb_velocity));
        // WATER-GATED GRAVITY (replaces the old water-plane ceiling spring):
        // swimmer parts above the surface get GravityScale 1 (fall back in),
        // submerged parts 0 (neutral buoyancy). Before SyncBackend so a
        // crossing takes effect the same step; writes only on change.
        app.add_systems(FixedUpdate,
            apply_water_gravity.before(PhysicsSet::SyncBackend));
        // STANDING fall-reset: teleport collapsed organisms back to the standing
        // pose. MUST run before SyncBackend so the Transform/Velocity writes are
        // pushed into Rapier this step (a post-Writeback write gets overwritten).
        app.add_systems(FixedUpdate, reset_fallen_standers.before(PhysicsSet::SyncBackend));
        // Heading-steering ASSIST (terrestrial walkers only; swimmers excluded):
        // aim the emergent crawl at prey by a throttled rigid-yaw TELEPORT of the
        // base (writes Transform + rotates Velocity), run before SyncBackend so the
        // writes are pushed into Rapier this step. Ordered after drive_limb_twist
        // only to keep it in the post-motor phase.
        app.add_systems(FixedUpdate, steer_base_toward_prey
            .after(drive_limb_twist)
            .before(PhysicsSet::SyncBackend));
        // Velocity governor (Rapier has no built-in max-speed): clamp rare
        // joint-instability spikes without throttling normal motion.
        app.add_systems(FixedUpdate, clamp_limb_velocity.after(drive_limb_motors));
        // SOLID FLOOR: one THICK static cuboid at terrain height. A dynamic body's
        // pose is owned by Rapier (a PostUpdate Transform clamp gets overwritten by
        // the next-step writeback), so the floor MUST be a real collider. A thick
        // cuboid (vs. the thin heightfield) can't be tunnelled by the near-massless
        // bodies. Flat-world appropriate (standing task); top face = terrain height.
        app.add_systems(
            Update,
            spawn_terrain_floor.run_if(
                resource_exists::<HeightmapSampler>
                    .and(not(resource_exists::<TerrainColliderSpawned>)),
            ),
        );
        // Ground-contact flags (foot/belly planted) from REAL Rapier collisions
        // with the floor, plus the predation event source.
        app.add_systems(Update, (handle_limb_collisions, configure_solver, assign_collision_groups));
        // SWIMDIAG: size↔speed↔stroke diagnostic for the "big = slow" investigation.
        app.add_systems(
            Update,
            swim_size_speed_diag
                .run_if(bevy::time::common_conditions::on_timer(std::time::Duration::from_secs(1))),
        );
        // STANDING curriculum: ramp limb gravity from light → full.
        app.add_systems(Update, fade_standing_gravity_assist);
    }
}

/// STANDING "training wheels": scale limb-body gravity from `STANDING_GRAVITY_START`
/// up to 1.0 over `STANDING_GRAVITY_FADE_SECS` of virtual time, so the policy first
/// learns to brace/balance the tall stance under light load, then maintains it as
/// true weight is restored. No-op unless `STANDING_TASK`.
fn fade_standing_gravity_assist(
    sim_running: Res<crate::simulation_settings::SimulationRunning>,
    time:        Res<bevy::prelude::Time<bevy::prelude::Virtual>>,
    // `Without<SwimmerBody>`: swimmers carry GravityScale(0.0) (neutral buoyancy);
    // the standing gravity ramp must never touch them.
    mut q:       Query<&mut GravityScale, (With<LastAppliedTorque>, Without<SwimmerBody>)>,
) {
    if !STANDING_TASK || !sim_running.0 { return; }
    let frac = (time.elapsed_secs() / STANDING_GRAVITY_FADE_SECS).clamp(0.0, 1.0);
    let g = STANDING_GRAVITY_START + (1.0 - STANDING_GRAVITY_START) * frac;
    for mut gs in &mut q {
        if (gs.0 - g).abs() > 1e-3 { gs.0 = g; }
    }
}

/// Raise Rapier's solver iteration counts once at startup so chained impulse
/// joints on the near-massless limb bodies stay convergent (no separation).
fn configure_solver(
    mut ctx:  Query<&mut RapierContextSimulation>,
    mut done: Local<bool>,
) {
    if *done { return; }
    if let Ok(mut c) = ctx.single_mut() {
        // Perf sweep: env overrides A/B the solver cost WITHOUT a rebuild
        // (`AEONS_SOLVER_ITERS=4 cargo run …`). The constraint + contact solve
        // scales ~linearly with num_solver_iterations — the prime lever for the
        // snake-multibody StepSimulation cost. Unset ⇒ the tuned defaults.
        let iters = std::env::var("AEONS_SOLVER_ITERS").ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(LIMB_SOLVER_ITERATIONS)
            .max(1);
        let stab = std::env::var("AEONS_STAB_ITERS").ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(LIMB_STABILIZATION_ITERATIONS);
        c.integration_parameters.num_solver_iterations = iters;
        c.integration_parameters.num_internal_stabilization_iterations = stab;
        *done = true;
        info!(
            "rapier solver configured: num_solver_iterations={} (default {}), stabilization_iterations={} (default {})",
            iters, LIMB_SOLVER_ITERATIONS, stab, LIMB_STABILIZATION_ITERATIONS
        );
    }
}


// ── Collision groups (perf: drop useless contact manifolds) ─────────────────
//
// Profiling pinned the FPS bottleneck to the rapier StepSimulation phase
// (~14.7ms/step, stable). With NO collision groups, every collider interacted
// with every other — including the 150 sessile photoautotroph (ball_plankton)
// compound colliders generating persistent photo↔photo and photo↔terrain
// contact manifolds, re-solved every iteration of every step. Photoautotrophs
// need NO rapier collision: sliding floor handling is mesh/heightmap-based (not
// rapier) and predation is proximity/event-based. So we partition colliders
// into three groups and let photoautotrophs collide ONLY with creatures (to
// keep the kinematic-prey collision-EVENT predation path), never with each
// other or the terrain.
//
//   G1 = terrain · G2 = photoautotroph · G3 = creature (heterotrophs incl. snakes)
fn photo_collision_groups() -> CollisionGroups {
    // photo ↔ creature only — drops photo↔photo and photo↔terrain manifolds.
    CollisionGroups::new(Group::GROUP_2, Group::GROUP_3)
}
fn creature_collision_groups() -> CollisionGroups {
    CollisionGroups::new(Group::GROUP_3, Group::GROUP_1 | Group::GROUP_2 | Group::GROUP_3)
}
fn terrain_collision_groups() -> CollisionGroups {
    CollisionGroups::new(Group::GROUP_1, Group::GROUP_3)
}

/// Assign a `CollisionGroups` to every collider that lacks one, by kind:
/// terrain → G1, photoautotroph → G2, everything else (sliding heterotrophs,
/// limb/swim body parts) → G3. Runs each frame but only touches NEW
/// colliders (`Without<CollisionGroups>`), so it is a cheap one-shot per spawned
/// collider. This is what removes the photo↔photo / photo↔terrain manifold cost
/// that dominated StepSimulation. Photoautotroph markers live on the (sliding)
/// root, which is also the collider entity; limb/swim parts are child collider
/// entities with no trophic marker → correctly default to creature.
fn assign_collision_groups(
    mut commands: Commands,
    q: Query<
        (Entity, Has<crate::colony::Photoautotroph>, Has<TerrainCollider>),
        (With<Collider>, Without<CollisionGroups>),
    >,
) {
    for (e, is_photo, is_terrain) in &q {
        let groups = if is_terrain {
            terrain_collision_groups()
        } else if is_photo {
            photo_collision_groups()
        } else {
            creature_collision_groups()
        };
        commands.entity(e).try_insert(groups);
    }
}




/// Per-limb terrain-contact flag (read by the limb-brain observation and the
/// standing reward). Driven from terrain CLEARANCE in
/// `enforce_limb_floor_and_contacts` — a part is "in contact" when its first-cell
/// origin is within `GROUND_CONTACT_EPS` of the terrain surface. (Robust and
/// terrain-specific, vs. counting arbitrary Rapier contact pairs.)
#[derive(Component, Default)]
pub struct LimbContact {
    pub in_contact: bool,
    /// Distinct Rapier collision pairs currently touching (telemetry only).
    pub count:      u32,
}


/// Present once the static floor is spawned; gates the spawn system off.
#[derive(Resource)]
struct TerrainColliderSpawned;

/// Marker on the static floor collider entity (used to recognise floor contacts).
#[derive(Component)]
pub struct TerrainCollider;


/// Last commanded hinge target angle (telemetry only, read by `dataset_export`
/// and `limb_time_series_log`). Inserted on every limb-based body part.
#[derive(Component, Default, Clone, Copy)]
pub struct LastAppliedTorque(pub Vec3);


/// Marks a body part belonging to a SWIMMING organism. ALL of the swimmer-only
/// systems (fluid drag, water-gated gravity, soft world borders, neutral
/// buoyancy, standing-curriculum exclusion) gate on this so walking/standing
/// limb organisms stay byte-identical. Inserted in `colony::insert_limb_physics`
/// on every part (incl. the base) when the organism is `is_swimming()`.
#[derive(Component)]
pub struct SwimmerBody;

/// Per-part presented cross-sectional AREA per LOCAL axis, used by the
/// anisotropic blade-element drag. For a cell cloud with local-space extents
/// `(ex, ey, ez)`, the area presented when moving along +X is the YZ face
/// `ey·ez`, along +Y the XZ face `ex·ez`, along +Z the XY face `ex·ey` — so a
/// flat fin pushes far more fluid edge-on than face-on, and the reaction force
/// propels/turns the body. Computed in `drag_shape_from_cells`.
#[derive(Component, Clone, Copy)]
pub struct LimbDragShape {
    pub area_local: Vec3,
}

/// Build a part's `LimbDragShape` from its cells. The per-axis span is
/// `max−min` over `cell.local_pos` PLUS `2·CELL_COLLISION_RADIUS` (the same
/// radius the limb colliders inflate each cell by in `limb_part_collider`), so
/// the presented areas match the physics footprint. A degenerate (empty) part
/// falls back to a unit cube.
pub fn drag_shape_from_cells(cells: &[Cell]) -> LimbDragShape {
    if cells.is_empty() {
        return LimbDragShape { area_local: Vec3::ONE };
    }
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for c in cells {
        lo = lo.min(c.local_pos);
        hi = hi.max(c.local_pos);
    }
    let pad = 2.0 * CELL_COLLISION_RADIUS;
    let ext = (hi - lo) + Vec3::splat(pad);
    let (ex, ey, ez) = (ext.x, ext.y, ext.z);
    LimbDragShape { area_local: Vec3::new(ey * ez, ex * ez, ex * ey) }
}


/// Pristine spawn-local `Transform` of a limb body part (the intended STANDING
/// pose), captured at spawn in `colony::insert_limb_physics`. The STANDING
/// fall-reset (`reset_fallen_standers`) teleports a collapsed organism's parts
/// back to this pose so the policy gets a fresh standing attempt.
#[derive(Component, Clone, Copy)]
pub struct LimbRestPose(pub Transform);


/// Links a limb's `MultibodyJoint` (on this same child entity) back to the
/// organism + body part it drives, and stores the joint geometry so the motor
/// driver can rebuild the joint data with a fresh target each tick.
///
/// `anchor1`/`anchor2` are the joint anchor points in the parent / limb LOCAL
/// frames; the telemetry measures joint SEPARATION as the world distance
/// between them (with multibody joints this must stay ~0 by construction).
#[derive(Component, Clone, Copy)]
pub struct LimbJointDrive {
    pub organism:    Entity,
    pub body_part:   usize,
    pub limb_entity: Entity,
    pub parent:      Entity,
    pub anchor1:     Vec3,
    pub anchor2:     Vec3,
    pub axis:        Vec3,
}

/// Links a limb BODY to its organism + body part + PARENT body for brain-driven
/// TWIST. `drive_limb_twist` applies a momentum-conserving torque COUPLE
/// (+T on the limb, −T on `parent`) about the limb's world-frame long axis.
#[derive(Component, Clone, Copy)]
pub struct LimbTwistDrive {
    pub organism:   Entity,
    pub body_part:  usize,
    pub axis_local: Vec3,
    pub parent:     Entity,
}


/// Deferred limb joint: holds the joint spec until BOTH endpoint bodies are
/// registered in Rapier. Adding a `MultibodyJoint` the same frame its bodies are
/// spawned panics inside Rapier (`multibody_joint_set` unwraps the not-yet-known
/// parent body handle). `attach_pending_limb_joints` converts this into the real
/// `MultibodyJoint` once both entities carry a `RapierRigidBodyHandle`.
#[derive(Component, Clone, Copy)]
pub struct PendingLimbJoint(pub LimbJointDrive);

/// Marker for a STATIC body part's deferred RIGID joint. Carries just the parent
/// body + the two anchor points (same scheme as the limb joints: `anchor2` is the
/// part's first cell, `anchor1` the same point in the parent's frame). Converted
/// to a real `FixedJoint` by `attach_pending_fixed_joints` once both bodies are
/// Rapier-registered. No motor and no `LimbJointDrive`, so the brain never moves it.
#[derive(Component, Clone, Copy)]
pub struct PendingFixedJoint {
    pub parent:  Entity,
    pub anchor1: Vec3,
    pub anchor2: Vec3,
}

/// Convert `PendingLimbJoint`s into real `ImpulseJoint`s once both the child
/// and its parent are registered Rapier bodies (`RapierRigidBodyHandle` present).
/// If the parent never becomes a body (e.g. a body-less part), the pending joint
/// simply stays unattached rather than crashing the solver.
pub fn attach_pending_limb_joints(
    mut commands: Commands,
    pending:      Query<(Entity, &PendingLimbJoint)>,
    registered:   Query<(), With<RapierRigidBodyHandle>>,
    organisms:    Query<&crate::colony::Organism>,
) {
    for (e, pj) in &pending {
        let parent = pj.0.parent;
        // Self-loop guard: a part must never joint to itself (a malformed root
        // attachment).
        if parent == e { commands.entity(e).try_remove::<PendingLimbJoint>(); continue; }
        // SWIMMERS are handled by `attach_pending_swimmer_multibody` (atomic,
        // reduced-coordinate). Only WALKERS/standers get an `ImpulseJoint` here.
        let is_swimmer = organisms.get(pj.0.organism)
            .map(|o| o.movement_mode.is_swimming()).unwrap_or(false);
        if is_swimmer { continue; }
        if registered.get(e).is_ok() && registered.get(parent).is_ok() {
            commands.entity(e)
                .try_insert(build_limb_joint(&pj.0, false))
                .try_remove::<PendingLimbJoint>();
        }
    }
}

/// Attach a SWIMMER organism's chain as a single REDUCED-COORDINATE articulation:
/// each part's joint becomes a `MultibodyJoint` (spherical, 3-DOF ball). In a
/// Featherstone articulation the child's pose is a function of (parent pose, joint
/// angle) — the translational DOFs do not exist in the state vector, so the joint
/// anchors are **mathematically incapable of separating**, regardless of motor
/// torque, spin, mass ratio or contact. This is the definitive fix for chained
/// segments drifting apart.
///
/// ATOMIC + ordered: a partial / out-of-order multibody insert forms a degenerate
/// articulation, so we wait until EVERY one of the organism's pending joints has
/// both bodies Rapier-registered, then insert them **parent-before-child** (sorted
/// by body-part index — body_parts is BFS-ordered, so parent index < child index)
/// in one frame. `LimbJointDrive` is kept (motors + separation telemetry).
pub fn attach_pending_swimmer_multibody(
    mut commands: Commands,
    pending:      Query<(Entity, &PendingLimbJoint)>,
    registered:   Query<(), With<RapierRigidBodyHandle>>,
    organisms:    Query<&crate::colony::Organism>,
) {
    use std::collections::HashMap;
    // Group this organism's pending swimmer joints together.
    let mut by_org: HashMap<Entity, Vec<(Entity, LimbJointDrive)>> = HashMap::new();
    for (e, pj) in &pending {
        if pj.0.parent == e { commands.entity(e).try_remove::<PendingLimbJoint>(); continue; }
        let swims = organisms.get(pj.0.organism)
            .map(|o| o.movement_mode.is_swimming()).unwrap_or(false);
        if !swims { continue; }
        by_org.entry(pj.0.organism).or_default().push((e, pj.0));
    }

    for (_org, mut joints) in by_org {
        // Gate: hold until the whole tree is registered (every child AND every
        // parent — the parent set includes the base body, which has no pending
        // joint of its own). One missing handle ⇒ defer the entire organism.
        let all_ready = joints.iter().all(|(child, d)|
            registered.get(*child).is_ok() && registered.get(d.parent).is_ok());
        if !all_ready { continue; }

        // Insert root-to-leaf so each child's parent is already a multibody link.
        joints.sort_by_key(|(_, d)| d.body_part);
        for (child, d) in joints {
            commands.entity(child)
                .try_insert(MultibodyJoint::new(d.parent, spherical_data(&d, [0.0; 3])))
                .try_remove::<PendingLimbJoint>();
        }
    }
}

/// BULLETPROOF multibody teardown for despawns. `rapier3d` 0.32's per-joint
/// `MultibodyJointSet::remove` has a root-link bug (issue #382, fixed in 0.33):
/// when a whole organism despawns atomically, bevy_rapier's `sync_removals`
/// removes each body in turn and the second link hits a stale arena entry →
/// panic. We pre-empt it: BEFORE `PhysicsSet::SyncBackend`, for every entity that
/// just lost its `RapierRigidBodyHandle` (i.e. is despawning), tear out its whole
/// articulation in ONE call (`remove_multibody_articulations`, which clears all
/// `rb2mb` mappings without the buggy sub-tree rebuild). It is idempotent (guarded
/// by `rb2mb.get`), so calling it for several links of the same organism is safe,
/// and it makes BOTH of bevy_rapier's later removal paths (`bodies.remove` and the
/// multibody-joint-handle removal) find nothing live → no-op. This catches EVERY
/// despawn path (predation, out-of-bounds, startup cull, future) because it hooks
/// the component removal itself, not individual call sites. Non-multibody bodies
/// (walkers, sliders) are no-ops here.
pub fn teardown_multibody_on_removal(
    mut removed: RemovedComponents<RapierRigidBodyHandle>,
    mut ctx:     Query<(&mut RapierContextJoints, &RapierRigidBodySet)>,
) {
    let Ok((mut joints, rb_set)) = ctx.single_mut() else { return };
    for entity in removed.read() {
        if let Some(&handle) = rb_set.entity2body().get(&entity) {
            joints.multibody_joints.remove_multibody_articulations(handle, true);
        }
    }
}

/// HARD SAFETY NET: a body whose Rapier state has gone non-finite (NaN/inf) will
/// panic the solver next step (`f32::clamp` on a NaN joint coordinate inside the
/// multibody motor constraint). Before each physics step we scan all body parts;
/// any organism with a non-finite body has its multibody articulation removed
/// IMMEDIATELY (so this step can't touch it) and the whole organism is despawned.
/// Self-contained (doesn't depend on the deferred despawn/teardown timing) → it
/// GUARANTEES no NaN ever reaches `step_simulation`, regardless of the underlying
/// numerical instability. Losing the odd unstable organism is acceptable; a crash
/// is not.
pub fn cull_nonfinite_bodies(
    mut commands:  Commands,
    mut joints_q:  Query<&mut RapierContextJoints>,
    rb_set_q:      Query<&RapierRigidBodySet>,
    parts:         Query<(&bevy::prelude::ChildOf, &RapierRigidBodyHandle)>,
    roots:         Query<(), With<crate::colony::OrganismRoot>>,
) {
    let (Ok(mut joints), Ok(rb_set)) = (joints_q.single_mut(), rb_set_q.single()) else { return };
    use std::collections::HashMap;
    let mut kill: HashMap<bevy::prelude::Entity, bevy_rapier3d::rapier::dynamics::RigidBodyHandle> = HashMap::new();
    for (child_of, handle) in &parts {
        if let Some(b) = rb_set.bodies.get(handle.0) {
            // Check translation, ORIENTATION, and both velocities. Orientation is
            // essential and was previously missed: the spherical motor's `curr_pos`
            // is derived from the joint's relative ORIENTATION, so a body whose
            // quaternion drifts to NaN (while translation/velocity stay finite)
            // slipped past the old position-only check, and the NEXT step's motor
            // constraint read a NaN `curr_pos` → `f32::clamp(min=NaN,max=NaN)` panic
            // inside `step_simulation` (the rare ~hours-in crash). Culling here on a
            // non-finite orientation removes the body before that step runs.
            let pose = bevy_rapier3d::utils::iso_to_transform(b.position());
            if !pose.translation.is_finite() || !pose.rotation.is_finite()
                || !b.linvel().is_finite() || !b.angvel().is_finite() {
                kill.entry(child_of.parent()).or_insert(handle.0);
            }
        }
    }
    for (root, handle) in kill {
        // Remove the whole articulation NOW so this step never solves a NaN motor.
        joints.multibody_joints.remove_multibody_articulations(handle, true);
        if roots.get(root).is_ok() {
            commands.entity(root).despawn();
            warn!("culled organism {:?}: non-finite physics state (multibody instability)", root);
        }
    }
}

/// EXPLICIT multibody-link transform sync (the render/gameplay fix).
///
/// bevy_rapier 0.34's `writeback_rigid_bodies` does not reliably write a
/// REDUCED-COORDINATE multibody link's pose onto its Bevy `Transform` when the
/// link is a nested child entity: the physics (the Rapier body isometry == the
/// articulation's `local_to_world`) is perfectly coupled, but the Bevy
/// `GlobalTransform` desyncs by a large per-organism offset. Everything that
/// reads `GlobalTransform` — the rendered mesh, predation `EAT_RADIUS`, the world
/// model, navigators — then sees the parts floating away from their joints even
/// though the joint physically holds. (Diagnosed from data: `|GT − rb|` up to
/// hundreds of units while `rb == mblink` and inter-part spacing stays the snake's
/// true shape.)
///
/// Fix: for every swimmer body part, overwrite its local `Transform` from the
/// AUTHORITATIVE Rapier body world pose each frame, AFTER bevy_rapier's writeback
/// and BEFORE Bevy's transform propagation, so the propagated `GlobalTransform`
/// equals the true articulated pose. Walkers/sliders are untouched (their
/// writeback is correct).
pub fn sync_multibody_link_transforms(
    rb_set_q:   Query<&RapierRigidBodySet>,
    globals:    Query<&GlobalTransform>,
    // `With<SwimmerBody>`: only swimmer organisms use reduced-coordinate
    // multibody links, and the marker is on every swimmer body part (incl.
    // the base). Filtering at the query level skips the per-frame
    // `organisms.get(root) → continue` over every walker/slider part.
    mut parts:  Query<(&bevy::prelude::ChildOf, &RapierRigidBodyHandle, &mut Transform),
                      (With<crate::cell::BodyPartIndex>, With<SwimmerBody>)>,
) {
    let Ok(rb_set) = rb_set_q.single() else { return };
    for (child_of, handle, mut transform) in &mut parts {
        let root = child_of.parent();
        let Some(rb) = rb_set.bodies.get(handle.0) else { continue };
        let Ok(parent_gt) = globals.get(root) else { continue };
        // World pose from the physics body → local transform under the (static)
        // root, so propagation yields GlobalTransform == the true articulated pose.
        let world = GlobalTransform::from(bevy_rapier3d::utils::iso_to_transform(rb.position()));
        let new_local = world.reparented_to(parent_gt);
        if *transform != new_local { *transform = new_local; }
    }
}

/// Convert `PendingFixedJoint`s (STATIC parts) into real `ImpulseJoint`s with a
/// `FixedJoint` once both bodies are Rapier-registered — rigidly welding the part
/// to its parent (no articulation, no motor). Mirrors the deferral in
/// `attach_pending_limb_joints` (same unknown-handle panic otherwise).
pub fn attach_pending_fixed_joints(
    mut commands: Commands,
    pending:      Query<(Entity, &PendingFixedJoint)>,
    registered:   Query<(), With<RapierRigidBodyHandle>>,
) {
    for (e, pj) in &pending {
        if pj.parent == e { commands.entity(e).try_remove::<PendingFixedJoint>(); continue; }
        if registered.get(e).is_ok() && registered.get(pj.parent).is_ok() {
            let data = FixedJointBuilder::new()
                .local_anchor1(pj.anchor1)
                .local_anchor2(pj.anchor2)
                .build();
            commands.entity(e)
                .try_insert(ImpulseJoint::new(pj.parent, data))
                .try_remove::<PendingFixedJoint>();
        }
    }
}

/// Build the revolute multibody-joint data for a limb hinge with the motor set
/// to `target` (radians). Rebuilt each tick by `drive_limb_motors` — cheap and
/// version-robust (avoids depending on internal mutable-accessor APIs).
/// TERRESTRIAL (walking/standing) only — swimmers get `spherical_data`.
fn revolute_data(drive: &LimbJointDrive, target: f32) -> TypedJoint {
    let (stiffness, damping, max_torque) =
        (LIMB_MOTOR_STIFFNESS, LIMB_MOTOR_DAMPING, crate::simulation_settings::MAX_LIMB_TORQUE);
    RevoluteJointBuilder::new(drive.axis)
        .local_anchor1(drive.anchor1)
        .local_anchor2(drive.anchor2)
        .limits([-LIMB_SWING_LIMIT, LIMB_SWING_LIMIT])
        .motor_position(target, stiffness, damping)
        // G2 (limbs never separate): bound the per-joint motor force so an
        // aggressive command can't spike the impulse-joint constraint into a
        // visible separation (iter9 saw a rare 0.52u spike under steering-driven
        // maneuvers with the uncapped ForceBased motor). MAX_LIMB_TORQUE ≫ the
        // torque a crawl needs, so propulsion is unaffected; only violent spikes clip.
        .motor_max_force(max_torque)
        // FORCE-based: motor stiffness/damping produce real TORQUE (F = k·err +
        // d·vel_err), so the planted foot pushes against the ground reaction and
        // PROPELS the body — locomotion. AccelerationBased (mass-independent) was
        // chosen for the heavy-body low-gravity STANDING curriculum, but it can't
        // deliver propulsion against an external (ground) constraint: it sizes the
        // impulse for the light limb's inertia alone, so the foot slips and the
        // body never translates (measured: feet never alternate, spd ~0.007,
        // reward variance ~0 → PPO actor_loss 0). Force-based risk is light-limb
        // over-drive / joint separation — bounded here by the lighter-than-Avian
        // BODY_PART_DENSITY=0.012 + the MAX_LIMB_*_SPEED governor; watch joint_sep.
        .motor_model(MotorModel::ForceBased)
        .build()
        .into()
}

/// Build the SPHERICAL (ball) joint data for a SWIMMING limb with per-axis
/// position-motor targets `target` (radians about the joint frame's X/Y/Z).
/// Rebuilt each tick by `drive_swim_motors`.
///
/// Same anchor scheme as the revolute hinge — `anchor2 = ZERO` is the limb's
/// FIRST PLACED CELL (the limb is rebased so that cell sits at its local
/// origin) and `anchor1` is the same world point in the parent's frame — so the
/// rotation point IS the first cell's centre and the point constraint keeps the
/// joint at its original position relative to the parent body part. A ball
/// joint frees all three rotational DOF (no hinge axis), letting limbs stroke
/// freely underwater; per-axis limits keep a limb from folding through the
/// body, and the brain commands all three target angles.
/// Swimmer scale-tuning multipliers (env-overridable, read ONCE per run).
///
/// "Play with constants" knobs for the micrometre-scale feel: large swimmers
/// read as ponderous giants because the `AccelerationBased` joint motor is slow
/// (`LIMB_MOTOR_STIFFNESS` ⇒ natural freq ≈ 0.4 Hz) and its force cap
/// (`SWIM_MAX_LIMB_TORQUE`) saturates on large-inertia limbs (`τ = I·α`,
/// `I ∝ size⁵`). These let a run be A/B'd WITHOUT a rebuild, e.g.
/// `AEONS_SWIM_MOTOR_STIFFNESS_MULT=6 AEONS_SWIM_TORQUE_MULT=4 cargo run …`.
/// Default 1.0 (= unchanged); env is constant per run so it is cached. Restart
/// to change. All are swim-only (gated on `SwimmerBody`/the spherical motor), so
/// walkers/standers stay byte-identical.
pub struct SwimTuning {
    /// ×`LIMB_MOTOR_STIFFNESS` for the swim joint motor — the primary
    /// "snappier strokes / less giant" knob (damping is scaled by its √ to keep
    /// the critically-damped ratio, so higher = faster WITHOUT overshoot/jitter).
    pub motor_stiffness_mult: f32,
    /// ×`SWIM_MAX_LIMB_TORQUE` motor force cap — raise so large-inertia limbs
    /// stop saturating (the size-dependent part of the "giant" feel).
    pub motor_torque_mult:    f32,
    /// ×`SWIM_DRAG_CD` / angular-drag coef — the literal "water resistance". NOTE
    /// this drag is ALSO the propulsion source, so cutting it cuts thrust too.
    pub drag_mult:            f32,
    /// ×`SWIM_LINK_INERTIA_FLOOR` — lower = snappier thin links (mostly affects
    /// small limbs, which the floor over-weights).
    pub inertia_mult:         f32,
}
pub fn swim_tuning() -> &'static SwimTuning {
    static T: std::sync::OnceLock<SwimTuning> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let m = |k: &str| std::env::var(k).ok()
            .and_then(|s| s.parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(1.0);
        let t = SwimTuning {
            motor_stiffness_mult: m("AEONS_SWIM_MOTOR_STIFFNESS_MULT"),
            motor_torque_mult:    m("AEONS_SWIM_TORQUE_MULT"),
            drag_mult:            m("AEONS_SWIM_DRAG_MULT"),
            inertia_mult:         m("AEONS_SWIM_INERTIA_MULT"),
        };
        info!(
            "swim tuning: motor_stiffness×{:.2} motor_torque×{:.2} drag×{:.2} inertia×{:.2}",
            t.motor_stiffness_mult, t.motor_torque_mult, t.drag_mult, t.inertia_mult,
        );
        t
    })
}

/// SWIMDIAG (1 Hz) — size↔speed↔stroke diagnostic to pin the "big = slow" cause.
/// Logs each swimmer's cell count (size proxy) and base XZ speed, plus the global
/// limb angular-speed distribution. Reading:
///   * big `cells` + low `base_speed` + HIGH global `limb_ang` ⇒ thrust/mass-bound
///     (limbs whip, but the heavy body ∝size³ won't move / drag thrust is capped).
///   * big `cells` + low `base_speed` + LOW `limb_ang` ⇒ motor/inertia-bound.
pub fn swim_size_speed_diag(
    roots: Query<(&crate::colony::Organism, &Velocity), With<SwimmerBody>>,
    parts: Query<&Velocity, (With<SwimmerBody>, Without<crate::colony::Organism>)>,
) {
    let (mut amax, mut asum, mut n) = (0.0f32, 0.0f32, 0u32);
    for v in &parts {
        let a = v.angular.length();
        if a.is_finite() { if a > amax { amax = a; } asum += a; n += 1; }
    }
    let mut logged = 0;
    for (org, v) in &roots {
        let cells = org.photo_cell_count + org.non_photo_cell_count;
        let base_xz = (v.linear.x * v.linear.x + v.linear.z * v.linear.z).sqrt();
        info!("SWIMDIAG cells={} base_speed={:.2}", cells, base_xz);
        logged += 1;
        if logged >= 12 { break; }   // bound log volume
    }
    if n > 0 {
        info!("SWIMDIAG limb_ang mean={:.2} max={:.2} over {} parts",
              asum / n as f32, amax, n);
    }
}

fn spherical_data(drive: &LimbJointDrive, target: [f32; 3]) -> TypedJoint {
    let tune = swim_tuning();
    let mut b = SphericalJointBuilder::new()
        .local_anchor1(drive.anchor1)
        .local_anchor2(drive.anchor2);
    for (axis, t) in [
        (JointAxis::AngX, target[0]),
        (JointAxis::AngY, target[1]),
        (JointAxis::AngZ, target[2]),
    ] {
        b = b
            .limits(axis, [-LIMB_SWING_LIMIT, LIMB_SWING_LIMIT])
            // ACCELERATION-based (mass-invariant): the motor commands a target
            // angular ACCELERATION, so the solved impulse scales with each link's
            // ACTUAL articulated inertia. This is essential for reduced-coordinate
            // multibody chains with thin/small links (e.g. a 2-cell mirrored limb,
            // tiny inertia about its long axis): a ForceBased motor applied a
            // fixed ×83 mass-ratio-scaled torque through that tiny inertia →
            // runaway angular acceleration → NaN joint coordinate → solver panic
            // (`unit_joint_motor_constraint` clamp on a NaN `curr_pos`). Because
            // the gains are now mass-invariant, the ×SWIM_MASS_RATIO scaling is
            // DROPPED — base gains apply uniformly regardless of body mass.
            // Stiffness × the env knob (faster strokes); damping × its √ to hold
            // the critically-damped ratio so a higher multiplier never overshoots.
            .motor_position(
                axis, t,
                LIMB_MOTOR_STIFFNESS * tune.motor_stiffness_mult,
                LIMB_MOTOR_DAMPING * tune.motor_stiffness_mult.sqrt(),
            )
            // Cap kept high (heavy neutral-buoyancy bodies need force ∝ mass to
            // hit a target accel); with accel-based motors a small-inertia link
            // only ever draws a small impulse, so the cap never drives overflow.
            // ×motor_torque_mult un-saturates large-inertia limbs (the size-
            // dependent part of the "giant" feel).
            .motor_max_force(axis, SWIM_MAX_LIMB_TORQUE * tune.motor_torque_mult)
            .motor_model(axis, MotorModel::AccelerationBased);
    }
    b.into()
}

/// Public helper used by `attach_pending_limb_joints` to build the initial joint
/// component (target 0 = rest pose). Swimmers get a BALL (spherical) joint —
/// free 3D limb rotation underwater; walkers/standers keep the revolute hinge.
///
/// IMPULSE joints with a FORCE-BASED motor. Impulse joints are robust to
/// despawn (each is independent — no articulation to double-free) and don't crash.
///
/// MULTIBODY (reduced-coordinate) joints were re-tested for the standing task and
/// REJECTED by data: the rigid articulation does not crash (once the whole tree
/// is attached in one frame), but at the standing curriculum's low gravity it
/// AMPLIFIES the warm-start/motor fling — a body tunnelled through the floor to
/// `base_clearance ≈ −11390` and joint_sep median hit 7.0 (vs impulse's steady
/// ~0.01). The impulse point constraint is the more stable of the two; the real
/// separation cause is the low-gravity FLING, fixed at its source (see the
/// gravity-curriculum / motor settings), not by the joint formulation.
pub fn build_limb_joint(drive: &LimbJointDrive, is_swimmer: bool) -> ImpulseJoint {
    if is_swimmer {
        ImpulseJoint::new(drive.parent, spherical_data(drive, [0.0; 3]))
    } else {
        ImpulseJoint::new(drive.parent, revolute_data(drive, 0.0))
    }
}


/// Handler for Rapier collision events involving limb parts:
///   * GROUND contact (foot/belly planted): a part touching the static FLOOR sets
///     that part's `LimbContact` (ref-counted) — the physically-correct planted
///     signal (fires on the part's actual collider geometry, incl. the distal
///     foot). Drives the standing reward's foot-support term + belly penalty
///     (base = index 0).
///   * PREDATION: a contact between two DIFFERENT organisms emits an
///     `OrganismContactEvent`.
/// The same-organism contact filter means intra-body pairs never reach here.
pub fn handle_limb_collisions(
    mut events:    MessageReader<CollisionEvent>,
    terrain:       Query<(), With<TerrainCollider>>,
    limb_parts:    Query<(&bevy::prelude::ChildOf, &crate::cell::BodyPartIndex)>,
    roots:         Query<&crate::colony::Organism, With<crate::colony::OrganismRoot>>,
    mut contacts:  Query<&mut LimbContact>,
    mut out:       MessageWriter<crate::organism_collision::OrganismContactEvent>,
) {
    let resolve = |e: Entity| -> Option<(Entity, usize)> {
        if let Ok((child_of, bp_idx)) = limb_parts.get(e) {
            return Some((child_of.parent(), bp_idx.0));
        }
        if let Ok(org) = roots.get(e) {
            let bp = org.body_parts.iter().position(|b| b.is_alive()).unwrap_or(0);
            return Some((e, bp));
        }
        None
    };

    for ev in events.read() {
        let (a, b, started) = match ev {
            CollisionEvent::Started(a, b, _) => (*a, *b, true),
            CollisionEvent::Stopped(a, b, _) => (*a, *b, false),
        };
        let a_floor = terrain.get(a).is_ok();
        let b_floor = terrain.get(b).is_ok();

        // Ground contact: exactly one side is the floor → the other is a limb part.
        if a_floor != b_floor {
            let part = if a_floor { b } else { a };
            if let Ok(mut c) = contacts.get_mut(part) {
                if started { c.count = c.count.saturating_add(1); }
                else       { c.count = c.count.saturating_sub(1); }
                c.in_contact = c.count > 0;
            }
            continue;
        }
        if a_floor && b_floor { continue; }

        // Organism-organism contact → predation event (on start only).
        if started {
            let Some((root_a, bp_a)) = resolve(a) else { continue };
            let Some((root_b, bp_b)) = resolve(b) else { continue };
            if root_a == root_b { continue; }
            out.write(crate::organism_collision::OrganismContactEvent {
                a: root_a, b: root_b, body_part_a: bp_a, body_part_b: bp_b,
            });
        }
    }
}


/// Velocity governor: clamp each limb body's linear/angular speed to the caps
/// (Rapier has no built-in `MaxLinearSpeed`/`MaxAngularSpeed`). Caps rare
/// joint-instability spikes; normal motion stays well under them.
pub fn clamp_limb_velocity(
    mut q: Query<&mut Velocity, With<LastAppliedTorque>>,
) {
    for mut v in &mut q {
        let ls = v.linear.length();
        if ls > MAX_LIMB_LINEAR_SPEED { v.linear *= MAX_LIMB_LINEAR_SPEED / ls; }
        let as_ = v.angular.length();
        if as_ > MAX_LIMB_ANGULAR_SPEED { v.angular *= MAX_LIMB_ANGULAR_SPEED / as_; }
    }
}


/// Spawn the static floor as ONE THICK cuboid whose top face is at the terrain
/// height. A thin heightfield lets collapsing/thrashing creatures punch through
/// and fall to the kill-floor; a deep cuboid is tunnel-proof. The standing world
/// (`world_superflat.glb`) is flat (uniform clearances confirm it), so a single
/// height is correct everywhere. Covers the whole map in XZ and extends
/// `FLOOR_THICKNESS` downward.
fn spawn_terrain_floor(
    mut commands: Commands,
    heightmap:    Res<HeightmapSampler>,
) {
    if heightmap.width == 0 || heightmap.depth == 0 { return; }
    const FLOOR_THICKNESS: f32 = 100.0;

    let span_x = heightmap.width as f32 * HEIGHTMAP_CELL_SIZE;
    let span_z = heightmap.depth as f32 * HEIGHTMAP_CELL_SIZE;
    let cx = span_x * 0.5;
    let cz = span_z * 0.5;
    let top = heightmap.height_at(cx, cz);   // flat world ⇒ uniform

    let hy = FLOOR_THICKNESS * 0.5;
    let centre = Vec3::new(cx, top - hy, cz);   // top face at `top`

    commands.spawn((
        TerrainCollider,
        RigidBody::Fixed,
        Collider::cuboid(span_x, hy, span_z),    // oversize XZ; deep in Y
        Transform::from_translation(centre),
    ));
    commands.insert_resource(TerrainColliderSpawned);
    info!("terrain floor (thick cuboid) spawned: top y={:.2}, span {:.0}×{:.0}", top, span_x, span_z);
}


// ── Limb hinge-motor target driver (EMERGENT, brain-driven) ──────────────────

/// Refresh each limb `MultibodyJoint`'s revolute position-motor target from the
/// brain's per-joint output each physics step. Rebuilds the joint data so the
/// in-solver position motor (stiffness/damping PD) tracks the new setpoint.
///
/// Mapping: `limb_targets[i-1]` (∈ [-1,1]) → hinge of body-part `i`, scaled to
/// `[-LIMB_SWING_LIMIT, +LIMB_SWING_LIMIT]`. Parts beyond `MAX_LIMB_JOINTS`
/// wrap modulo.
pub fn drive_limb_motors(
    sim_running: Res<crate::simulation_settings::SimulationRunning>,
    organisms:   Query<&crate::colony::Organism>,
    mut joints:  Query<(&LimbJointDrive, &mut ImpulseJoint)>,
    mut torque_log: Query<&mut LastAppliedTorque>,
) {
    if !sim_running.0 { return; }

    for (drive, mut joint) in &mut joints {
        let Ok(organism) = organisms.get(drive.organism) else { continue };
        if organism.movement_mode.is_sliding() { continue; }
        // Swimmers carry BALL joints driven by `drive_swim_motors` from the
        // swimming brain's 3-axis targets — never rebuild them as hinges here.
        if organism.movement_mode.is_swimming() { continue; }
        let i = drive.body_part;
        if i == 0 || i >= organism.body_parts.len() { continue; }

        let out_idx = (i - 1) % MAX_LIMB_JOINTS;
        let cmd = organism.limb_targets[out_idx].clamp(-1.0, 1.0);
        let target = cmd * LIMB_SWING_LIMIT;

        joint.data = revolute_data(drive, target);

        if let Ok(mut last) = torque_log.get_mut(drive.limb_entity) {
            last.0 = Vec3::new(target, 0.0, 0.0);
        }
    }
}

/// Per-joint 3-axis target angles for a SWIMMING organism's ball joints,
/// written by the swimming brain (`swim_ppo::apply_step`) and consumed by
/// `drive_swim_motors`. Lives on the ORGANISM ROOT entity (inserted in
/// `colony::insert_limb_physics` for swimmers). Layout: joint `j` (= body-part
/// `j+1`, modulo `MAX_LIMB_JOINTS`) owns `[3*j .. 3*j+3]` = target rotation
/// about the joint frame's (X, Y, Z), each ∈ [-1, 1] (× `LIMB_SWING_LIMIT`).
/// Kept OFF `Organism` (separate component) so the save format and every
/// `Organism` construction site stay untouched.
#[derive(Component, Clone, Copy)]
pub struct SwimJointTargets(pub [f32; MAX_LIMB_JOINTS * 3]);

impl Default for SwimJointTargets {
    fn default() -> Self { Self([0.0; MAX_LIMB_JOINTS * 3]) }
}

/// Swimming analogue of `drive_limb_motors`: refresh each swimmer ball joint's
/// THREE per-axis position motors from the swimming brain's `SwimJointTargets`
/// each physics step. The joint anchors are immutable across the rebuild, so
/// the rotation point stays pinned at the limb's first cell.
pub fn drive_swim_motors(
    sim_running: Res<crate::simulation_settings::SimulationRunning>,
    organisms:   Query<(&crate::colony::Organism, &SwimJointTargets)>,
    // Swimmer ball joints are REDUCED-COORDINATE `MultibodyJoint`s (separation
    // impossible by construction); walkers keep `ImpulseJoint` (driven by
    // `drive_limb_motors`). Rebuilding `joint.data` retargets the 3 per-axis
    // position motors; the spherical DOF count is constant (3), so the
    // multibody user-change path stays well-defined.
    mut joints:  Query<(&LimbJointDrive, &mut MultibodyJoint)>,
    mut torque_log: Query<&mut LastAppliedTorque>,
) {
    if !sim_running.0 { return; }

    for (drive, mut joint) in &mut joints {
        let Ok((organism, targets)) = organisms.get(drive.organism) else { continue };
        if !organism.movement_mode.is_swimming() { continue; }
        let i = drive.body_part;
        if i == 0 || i >= organism.body_parts.len() { continue; }

        let base = ((i - 1) % MAX_LIMB_JOINTS) * 3;
        let target = [
            targets.0[base    ].clamp(-1.0, 1.0) * LIMB_SWING_LIMIT,
            targets.0[base + 1].clamp(-1.0, 1.0) * LIMB_SWING_LIMIT,
            targets.0[base + 2].clamp(-1.0, 1.0) * LIMB_SWING_LIMIT,
        ];
        joint.data = spherical_data(drive, target);

        if let Ok(mut last) = torque_log.get_mut(drive.limb_entity) {
            last.0 = Vec3::new(target[0], target[1], target[2]);
        }
    }
}

/// Brain-driven TWIST: apply a momentum-conserving torque couple about each
/// limb's CURRENT world-frame long axis (+T on the limb, −T on its parent), via
/// the persistent `ExternalForce.torque`. `limb_targets[MAX_LIMB_JOINTS + grp]`
/// (∈ [-1,1]) scales `axis · cmd · MAX_LIMB_TWIST_TORQUE`.
pub fn drive_limb_twist(
    sim_running: Res<crate::simulation_settings::SimulationRunning>,
    organisms:   Query<&crate::colony::Organism>,
    limbs:       Query<(Entity, &LimbTwistDrive, &GlobalTransform)>,
    mut forces:  Query<(Entity, &mut ExternalForce)>,
) {
    if !sim_running.0 { return; }
    let mut acc: std::collections::HashMap<Entity, Vec3> = std::collections::HashMap::new();
    for (limb_e, drive, gt) in &limbs {
        let Ok(organism) = organisms.get(drive.organism) else { continue };
        if organism.movement_mode.is_sliding() { continue; }
        // Swimmers: the ball joint's three position motors already command
        // every rotational DOF (incl. twist about the long axis); the external
        // twist couple would only fight them. Skip.
        if organism.movement_mode.is_swimming() { continue; }
        let i = drive.body_part;
        if i == 0 || i >= organism.body_parts.len() { continue; }
        let out_idx = MAX_LIMB_JOINTS + (i - 1) % N_LIMB_TWIST_GROUPS;
        let cmd = organism.limb_targets[out_idx].clamp(-1.0, 1.0);
        let axis_world = (gt.rotation() * drive.axis_local).normalize_or_zero();
        let t = axis_world * (cmd * crate::simulation_settings::MAX_LIMB_TWIST_TORQUE);
        *acc.entry(limb_e).or_default() += t;
        *acc.entry(drive.parent).or_default() -= t;
    }
    for (e, mut ef) in &mut forces {
        // ACCUMULATE (not assign): `reset_external_forces` zeroes the force/torque
        // first this step, and `apply_fluid_drag` / `confine_swimmers` add later, so
        // every contributor composes instead of clobbering.
        ef.torque += acc.get(&e).copied().unwrap_or(Vec3::ZERO);
    }
}

/// Zero every limb body's persistent `ExternalForce` at the START of FixedUpdate
/// so the per-step contributors (`drive_limb_twist` torque couple, swimmer fluid
/// drag, swimmer confinement) ACCUMULATE onto a clean slate instead of letting a
/// stale force integrate forever. Runs before `drive_limb_motors`.
pub fn reset_external_forces(
    mut q: Query<&mut ExternalForce, With<LastAppliedTorque>>,
) {
    for mut ef in &mut q {
        ef.force  = Vec3::ZERO;
        ef.torque = Vec3::ZERO;
    }
}

/// ANISOTROPIC blade-element fluid drag for swimmer parts. In each part's LOCAL
/// frame the drag is per-axis quadratic, scaled by the presented AREA on that
/// axis (`LimbDragShape`), so a fin edge-on pushes little fluid and face-on
/// pushes a lot — the difference, transmitted through the joints, both PROPELS
/// and ROTATES the whole organism (emergent locomotion). Force applied at the
/// COM; net body rotation emerges from the differing drag on off-axis parts (no
/// centre-of-pressure offset is modelled). Plus an isotropic quadratic angular
/// drag. Runs AFTER the motor/twist drivers, BEFORE the velocity clamp.
///
/// SUBMERSION-GATED (per part): drag is applied ONLY to parts BELOW the water
/// surface. A part above the surface gets no fluid force, so a swimmer spawned
/// (or flung) into the air falls with ordinary "dry-land" physics — full
/// gravity (`apply_water_gravity` turns gravity ON above the surface at the
/// same threshold) and no phantom in-air drag — until it re-enters the water.
/// Gating per part (not per organism) keeps a partially-breaching swimmer
/// physical: only its submerged parts feel the water.
pub fn apply_fluid_drag(
    sim_running: Res<crate::simulation_settings::SimulationRunning>,
    water:       Res<WaterLevel>,
    mut q: Query<(&GlobalTransform, &Velocity, &mut ExternalForce, &LimbDragShape), With<SwimmerBody>>,
) {
    if !sim_running.0 { return; }
    let water_y = water.0;
    for (gt, vel, mut ext, drag) in &mut q {
        // Above the surface → no water to push against; fall with dry-land
        // physics. (Same threshold as `apply_water_gravity`'s gravity toggle.)
        if gt.translation().y > water_y { continue; }
        // Never compute drag from a non-finite velocity (would propagate NaN
        // straight into the solver and panic the physics step).
        if !vel.linear.is_finite() { continue; }
        let rot = gt.rotation();
        let v_local = rot.inverse() * vel.linear;
        let a = drag.area_local;
        // "Water resistance" env knob (×SWIM_DRAG_CD / angular coef). NOTE this
        // drag is ALSO the blade-element propulsion source, so it scales thrust
        // and resistance together.
        let cd = SWIM_DRAG_CD * swim_tuning().drag_mult;
        // Per-axis quadratic drag, CLAMPED: the v² term is explosive on a light
        // body if velocity spikes, so cap each axis force. (With SWIM_BODY_DENSITY
        // the body is heavy enough that this cap is rarely hit, but it guarantees
        // the force can never diverge.)
        let f_local = Vec3::new(
            -0.5 * WATER_DENSITY * cd * a.x * v_local.x.abs() * v_local.x,
            -0.5 * WATER_DENSITY * cd * a.y * v_local.y.abs() * v_local.y,
            -0.5 * WATER_DENSITY * cd * a.z * v_local.z.abs() * v_local.z,
        ).clamp(Vec3::splat(-SWIM_DRAG_MAX_FORCE), Vec3::splat(SWIM_DRAG_MAX_FORCE));
        let fw = rot * f_local;
        if fw.is_finite() { ext.force += fw; }
        let w = vel.angular;
        if w.is_finite() {
            let mut t = -SWIM_ANGULAR_DRAG_COEF * swim_tuning().drag_mult * w.length() * w;
            let tl = t.length();
            if tl > SWIM_DRAG_MAX_TORQUE { t *= SWIM_DRAG_MAX_TORQUE / tl; }
            ext.torque += t;
        }
    }
}

/// Keep swimmers inside the map's XZ bounds with SOFT restoring forces (no
/// hard collider). The Y axis needs no spring any more: WATER-GATED GRAVITY
/// (`apply_water_gravity`) pulls a breaching body back physically, and the
/// terrain FLOOR is the existing static cuboid collider. Runs after
/// `apply_fluid_drag`, before the velocity clamp.
pub fn confine_swimmers(
    sim_running: Res<crate::simulation_settings::SimulationRunning>,
    map:         Res<MapSize>,
    mut q:       Query<(&GlobalTransform, &mut ExternalForce), With<SwimmerBody>>,
) {
    if !sim_running.0 { return; }
    let r = CELL_COLLISION_RADIUS;
    for (gt, mut ext) in &mut q {
        let pos = gt.translation();
        // XZ soft borders: clamped springs (`stiffness × overshoot`, capped at
        // SWIM_CONFINE_MAX_FORCE so a far-out body can't get an explosive
        // impulse — it is pushed back firmly but finitely).
        if pos.x < r {
            ext.force.x += (SWIM_BORDER_STIFFNESS * (r - pos.x)).min(SWIM_CONFINE_MAX_FORCE);
        } else if pos.x > map.x - r {
            ext.force.x -= (SWIM_BORDER_STIFFNESS * (pos.x - (map.x - r))).min(SWIM_CONFINE_MAX_FORCE);
        }
        if pos.z < r {
            ext.force.z += (SWIM_BORDER_STIFFNESS * (r - pos.z)).min(SWIM_CONFINE_MAX_FORCE);
        } else if pos.z > map.z - r {
            ext.force.z -= (SWIM_BORDER_STIFFNESS * (pos.z - (map.z - r))).min(SWIM_CONFINE_MAX_FORCE);
        }
    }
}


/// WATER-GATED GRAVITY for water-based DYNAMIC bodies (`SwimmerBody` — every
/// dynamic water-based organism is a swimmer; water-based phototrophs are
/// kinematic and handled in `movement::apply_gravity`): a part ABOVE the
/// water surface gets full gravity (it falls back into the water); a
/// submerged part is neutrally buoyant (`GravityScale(0)`, the spawn
/// default). This replaces the old artificial ceiling spring with the real
/// physics the water surface implies — a breaching swimmer arcs and falls
/// back instead of being rubber-banded down.
///
/// Efficiency: one compare per part; `GravityScale` is WRITTEN only when the
/// part actually crosses the surface (almost never per step), so Rapier's
/// change-detection sync stays idle in the steady state.
pub fn apply_water_gravity(
    sim_running: Res<crate::simulation_settings::SimulationRunning>,
    water:       Res<WaterLevel>,
    mut q:       Query<(&GlobalTransform, &mut GravityScale), With<SwimmerBody>>,
) {
    if !sim_running.0 { return; }
    let water_y = water.0;
    for (gt, mut gs) in &mut q {
        let target = if gt.translation().y > water_y { 1.0 } else { 0.0 };
        if gs.0 != target { gs.0 = target; }
    }
}


/// Safety net: zero any non-finite (NaN/∞) velocity on a swimmer body BEFORE the
/// physics step. If a transient instability ever produces a bad velocity, this
/// stops it propagating into Rapier's solver (which panics on non-finite state)
/// — the swimmer momentarily halts instead of crashing the whole simulation.
pub fn sanitize_swimmer_velocity(
    mut q: Query<&mut Velocity, With<SwimmerBody>>,
) {
    for mut v in &mut q {
        if !v.linear.is_finite()  { v.linear  = Vec3::ZERO; }
        if !v.angular.is_finite() { v.angular = Vec3::ZERO; }
    }
}

// ── Heading-steering ASSIST toward prey (locomotion task) ────────────────────

/// Aim each limb organism's emergent crawl at the nearest perceived prey with a PD
/// YAW TORQUE on the base (`ExternalForce`, Rapier-integrated). Force-based steering
/// is stable: it does NOT teleport dynamic-body transforms (which inject energy and
/// separated joints / collapsed the base — G1/G2), and the joints carry the limbs
/// around with the base. Proportional term turns the base so its TRAVEL points at
/// prey; the derivative term (−Kd·yaw_rate, plus the base's angular damping) damps
/// the turn so the low-inertia body converges instead of spinning. The leg GAIT
/// stays fully brain-driven — only the heading is assisted (the brain doesn't learn
/// to steer the long-slab, high-yaw-inertia Crawler on its own). Off during standing.
/// Runs AFTER `drive_limb_twist` (which sets `ef.torque` absolutely) so it ADDs.
pub fn steer_base_toward_prey(
    sim_running: Res<crate::simulation_settings::SimulationRunning>,
    grid:        Option<Res<crate::world_model::WorldModelGrid>>,
    organisms:   Query<&crate::colony::Organism>,
    // `Without<SwimmerBody>`: this scripted yaw-teleport is a terrestrial aiming
    // assist. Swimmers turn purely via emergent blade-element drag + joint
    // dynamics, so they are excluded — which also removes the `&mut Velocity`
    // aliasing with `confine_swimmers` (no ordering edge needed between them).
    mut parts:   Query<(&bevy::prelude::ChildOf, &crate::cell::BodyPartIndex, &GlobalTransform, &mut Transform, &mut Velocity), Without<SwimmerBody>>,
    mut tick:    bevy::prelude::Local<u32>,
) {
    use crate::simulation_settings::{STANDING_TASK, STEER_ASSIST_GAIN, STEER_ASSIST_MAX_YAW, STEER_INTERVAL};
    if STANDING_TASK || !sim_running.0 { return; }
    let Some(grid) = grid else { return };
    // Throttle: the rigid-yaw teleport writes Transform on every dynamic limb part,
    // forcing a Rapier re-sync — costly when many organisms actively seek. Running it
    // every Nth tick (still tens of Hz) cuts that cost ~N× with negligible aiming loss.
    *tick = tick.wrapping_add(1);
    if *tick % STEER_INTERVAL != 0 { return; }

    // Pass 1 (read base): per limb-organism root, the small yaw Δθ that turns its
    // TRAVEL direction toward the nearest prey (aim travel, not the +Z heading the
    // gait propels off), plus the base local pivot for the coherent rotation.
    use std::collections::HashMap;
    let mut plan: HashMap<bevy::prelude::Entity, (Vec3, f32)> = HashMap::new();
    for (co, idx, gt, tf, vel) in &parts {
        if idx.0 != 0 { continue; }
        let Ok(organism) = organisms.get(co.parent()) else { continue };
        if organism.movement_mode.is_sliding() { continue; }
        let pos = gt.translation();
        let Some((rel, _d, _e)) = crate::world_model::nearest_prey(&grid, pos) else { continue };
        let vxz = Vec2::new(vel.linear.x, vel.linear.z);
        let d = if vxz.length() > 0.05 {
            vxz.normalize()
        } else {
            let fwd = gt.rotation() * Vec3::Z; Vec2::new(fwd.x, fwd.z).normalize_or_zero()
        };
        let p = Vec2::new(rel.x, rel.z);
        if d.length_squared() < 1e-6 || p.length_squared() < 1e-6 { continue; }
        let p = p.normalize();
        let cross_y = d.y * p.x - d.x * p.y;       // (travel × prey)·Y, .y holds z
        let angle   = cross_y.atan2(d.dot(p));     // ∈ [-π,π]; 0 ⇒ travelling at prey
        let step    = (angle * STEER_ASSIST_GAIN).clamp(-STEER_ASSIST_MAX_YAW, STEER_ASSIST_MAX_YAW);
        let dtheta  = if angle >= 0.0 { step.min(angle) } else { step.max(angle) };
        plan.insert(co.parent(), (tf.translation, dtheta));
    }

    // Pass 2 (write): rigidly yaw every part of a planned organism about its base
    // pivot — translation, orientation, AND velocities — so the whole creature
    // reorients coherently (joints stay satisfied; the gait continues, now aimed).
    for (co, _idx, _gt, mut tf, mut vel) in &mut parts {
        let Some(&(pivot, dtheta)) = plan.get(&co.parent()) else { continue };
        if dtheta.abs() < 1e-5 { continue; }
        let q = Quat::from_rotation_y(dtheta);
        tf.translation = pivot + q * (tf.translation - pivot);
        tf.rotation    = q * tf.rotation;
        vel.linear     = q * vel.linear;
        vel.angular    = q * vel.angular;
    }
}


// ── STANDING fall-reset (episode reset) ──────────────────────────────────────

/// Detect collapsed limb organisms and teleport them back to their standing
/// spawn pose, the legged-RL "reset on fall" so the policy keeps experiencing the
/// standing region (see `STAND_RESET_*` in simulation_settings). Without it a
/// fallen Runner belly-crawls forever and never relearns standing.
///
/// Runs in `FixedUpdate` BEFORE `PhysicsSet::SyncBackend`: SyncBackend propagates
/// the written `Transform` to `GlobalTransform` and then `apply_rigid_body_user_changes`
/// teleports the Rapier body — so the pose write STICKS (a PostUpdate write, after
/// Writeback, would be overwritten). Velocities are zeroed the same way
/// (`Changed<Velocity>` is applied in the same set).
pub fn reset_fallen_standers(
    sim_running: Res<crate::simulation_settings::SimulationRunning>,
    time:        Res<bevy::prelude::Time<bevy::prelude::Virtual>>,
    base_q:      Query<(&bevy::prelude::ChildOf, &GlobalTransform, &LimbContact, &crate::cell::BodyPartIndex), Without<SwimmerBody>>,
    // `Without<SwimmerBody>`: never teleport a swimmer to a terrestrial rest pose.
    mut parts:   Query<(&bevy::prelude::ChildOf, &LimbRestPose, &mut Transform, &mut Velocity), Without<SwimmerBody>>,
    mut fall:     Local<std::collections::HashMap<Entity, f32>>,
    mut cooldown: Local<std::collections::HashMap<Entity, f32>>,
) {
    use crate::simulation_settings::{STANDING_TASK, STAND_RESET_UPRIGHT, STAND_RESET_GRACE_SECS, STAND_RESET_COOLDOWN_SECS};
    if !STANDING_TASK || !sim_running.0 { return; }
    let dt = time.delta_secs();

    // Per-organism base (body-part 0) uprightness + belly contact.
    let mut base: std::collections::HashMap<Entity, (f32, bool)> = std::collections::HashMap::new();
    for (co, gt, contact, idx) in &base_q {
        if idx.0 != 0 { continue; }
        let up = (gt.rotation() * Vec3::Y).y;
        base.insert(co.parent(), (up, contact.in_contact));
    }

    // Decide which roots to reset (sustained fall, not in cooldown).
    let mut to_reset: std::collections::HashSet<Entity> = std::collections::HashSet::new();
    for (&root, &(up, belly)) in &base {
        if let Some(cd) = cooldown.get_mut(&root) {
            if *cd > 0.0 { *cd -= dt; fall.insert(root, 0.0); continue; }
        }
        // Fallen = tipped past the upright threshold OR belly on the floor. Resetting
        // on tilt alone (not only after a full belly-flop) keeps experience in the
        // standing region rather than letting it drift down and learn belly behaviour.
        let fallen = belly || up < STAND_RESET_UPRIGHT;
        let t = fall.entry(root).or_insert(0.0);
        if fallen { *t += dt; } else { *t = 0.0; }
        if *t >= STAND_RESET_GRACE_SECS {
            to_reset.insert(root);
            *t = 0.0;
            cooldown.insert(root, STAND_RESET_COOLDOWN_SECS);
        }
    }
    if to_reset.is_empty() { return; }

    // Teleport every part of a fallen organism back to its standing spawn pose.
    for (co, rest, mut tf, mut v) in &mut parts {
        if to_reset.contains(&co.parent()) {
            *tf = rest.0;
            v.linear  = Vec3::ZERO;
            v.angular = Vec3::ZERO;
        }
    }
}
