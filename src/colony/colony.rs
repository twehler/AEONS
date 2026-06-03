use crate::cell::*;
use crate::frontend::ShowGizmo;
use crate::volumetric_growth::{build_mesh_from_ocg, build_smoothed_mesh_from_ocg};
use crate::world_geometry::{HeightmapSampler, MapSize};
use crate::movement::DirectionTimer;
use bevy::prelude::*;
use rand::prelude::*;

// Re-export the Organism module so existing `use crate::colony::*;` imports
// keep finding Organism, OrganismRoot, OrganismKind, Photoautotroph, etc.
pub use crate::organism::*;

// Re-export the binary `.colony` save/load module (split out of this file)
// so external paths like `crate::colony::SaveRequested` keep resolving and
// the plugin / spawn_colony can reference its items unqualified.
pub use crate::colony_save_load::*;

// Initial cohort sizes are derived at spawn-time from three
// resources (all launcher-set, defaulting to the matching
// `DEFAULT_*` constants):
//   * herbivores        = StartHeterotrophs    (launcher: "Start Heterotroph Number")
//   * photoautotrophs   = MaxOrganisms − MaxHerbivores
// The running-population cap is `MaxOrganisms` (and herbivore-only
// cap is `MaxHerbivores`), both editable at runtime via the
// statistics panel. Keeping the START count decoupled from the
// CAP lets the user seed a small starter cohort and let reproduction
// fill the room up to the cap. See `spawn_colony`.

pub use crate::simulation_settings::INITIAL_KRISHI;

use crate::simulation_settings::{
    SPAWN_CRAWLER_PATH, SPAWN_CRAWLER_COUNT,
    SPAWN_STRIDER_PATH, SPAWN_STRIDER_COUNT,
    SPAWN_ALGAE_PATH, SPAWN_ALGAE_COUNT,
};

use crate::simulation_settings::BASE_ANGULAR_DAMPING;

use crate::simulation_settings::LIMB_ANGULAR_DAMPING;

use crate::simulation_settings::LIMB_LINEAR_DAMPING;

use crate::simulation_settings::BODY_PART_DENSITY;



/// Build a SINGLE convex-hull collider for a limb-based body part from
/// its cell cloud, replacing the old `Collider::compound` of one sphere
/// per cell. Collider count drives Avian's narrow-phase cost (every
/// shape-pair is tested), so collapsing ~5 spheres per part into one
/// hull cuts the per-organism collider count ~5× and the pair count
/// quadratically — the dominant lever for limb-organism physics
/// performance.
///
/// Each cell contributes its six axis-extreme points (`center ± r` on
/// X/Y/Z). Hulling those instead of bare centers (a) keeps the hull at
/// roughly the true cell extent so body parts don't sink into the
/// terrain by a cell radius, and (b) guarantees a non-degenerate 3D
/// point set even for a 1-cell part (six points → an octahedron), so
/// `convex_hull` never returns `None` in practice — the sphere fallback
/// is purely defensive.
fn limb_part_collider(cells: &[Cell]) -> Option<avian3d::prelude::Collider> {
    if cells.is_empty() { return None; }
    let r = crate::cell::CELL_COLLISION_RADIUS;
    let mut pts: Vec<Vec3> = Vec::with_capacity(cells.len() * 6);
    for c in cells {
        let p = c.local_pos;
        pts.push(p + Vec3::X * r); pts.push(p - Vec3::X * r);
        pts.push(p + Vec3::Y * r); pts.push(p - Vec3::Y * r);
        pts.push(p + Vec3::Z * r); pts.push(p - Vec3::Z * r);
    }
    Some(
        avian3d::prelude::Collider::convex_hull(pts)
            .unwrap_or_else(|| avian3d::prelude::Collider::sphere(r)),
    )
}


/// Compute the (parent_anchor, limb_anchor) pair for a limb's
/// `SphericalJoint`, placing the pivot at the FACE MIDPOINT between
/// the parent cell adjacent to the attachment and the limb's first
/// cell. With the limb rebased so its first cell sits at the limb's
/// local origin, this gives:
///
///   * `anchor1` = `(parent_cell_pos + pivot) / 2`
///       — point on the parent body, on the parent-cell ↔ limb-first-
///       cell shared rhombic face.
///   * `anchor2` = `(parent_cell_pos − pivot) / 2`
///       — same point expressed in the limb's local frame; the limb's
///       first cell sits at `(0, 0, 0)` and the anchor lies between
///       the limb origin and the (offset) direction of the parent
///       cell.
///
/// The two anchors coincide in world space so the cells sit on
/// opposite sides of the joint — like a shoulder joint at the body
/// surface rather than a rotation around the limb's own centroid.
///
/// "Parent cell adjacent to the attachment" is the parent cell
/// closest to `pivot`; by construction of the species-editor's
/// candidate-driven placement, that's exactly one RD lattice step
/// away. Falls back to `Vec3::ZERO` when the parent body has no
/// cells (defensive — should never happen for a valid attachment).
fn limb_joint_anchors(
    parent_origin: Vec3,   // parent body-part entity origin, in the root frame
    limb_cells:    &[Cell],
    pivot:         Vec3,   // first-cell position in the root frame (= attach.origin_local)
) -> (Vec3, Vec3, Vec3) {
    // The limb rotates/swings about the centre of its FIRST placed cell, which
    // (cells are rebased so the first cell sits at local origin) is exactly the
    // limb entity's origin. So:
    //   anchor2 (limb-local)   = ZERO            — the rotation point IS the first cell
    //   anchor1 (parent-local) = pivot − parent_origin — that same point in the parent's frame
    // Both resolve to one world point, so the limb is pinned at its first cell
    // and can only ROTATE about it — it can never translate away from the
    // parent, no matter the force. (Body parts are a flat hierarchy under the
    // root, so a part's entity origin equals its own `attachment.origin_local`,
    // hence `parent_origin` is the parent part's attachment origin.)
    let anchor1 = pivot - parent_origin;
    let anchor2 = Vec3::ZERO;

    // Hinge axis: ⟂ the limb's out-direction and world-up, so the primary swing
    // is in a vertical plane (lift/push for locomotion). `out_dir` = the limb's
    // rest long axis (centroid of its rebased cells points along the limb).
    let centroid = if limb_cells.is_empty() {
        Vec3::ZERO
    } else {
        limb_cells.iter().map(|c| c.local_pos).sum::<Vec3>() / limb_cells.len() as f32
    };
    let mut out_dir = centroid.normalize_or_zero();
    if out_dir.length_squared() <= 1e-6 {
        out_dir = Vec3::Z;
    }
    let mut hinge_axis = out_dir.cross(Vec3::Y).normalize_or_zero();
    if hinge_axis.length_squared() <= 1e-6 {
        hinge_axis = out_dir.cross(Vec3::X).normalize_or_zero();
    }
    if hinge_axis.length_squared() <= 1e-6 {
        hinge_axis = Vec3::Z;
    }

    (anchor1, anchor2, hinge_axis)
}


pub struct ColonyPlugin;

impl Plugin for ColonyPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SaveRequested>();
        // `init_resource` is idempotent — it only inserts when no
        // resource of this type already exists. main.rs sets the load
        // path BEFORE add_plugins runs, so this is just the no-load
        // fallback for callers that wire ColonyPlugin without setting
        // a path. Using `insert_resource` here would unconditionally
        // overwrite main.rs's value.
        app.init_resource::<ColonyLoadPath>();
        app.init_resource::<crate::simulation_settings::AutoSpawnHeteros>();
        app.init_resource::<crate::simulation_settings::StartHeterotrophs>();
        app.init_resource::<crate::simulation_settings::StartPhotoautotrophs>();
        app.init_resource::<crate::simulation_settings::MinHeteroCount>();
        app.init_resource::<crate::simulation_settings::MinHeteroCountEditState>();
        app.init_resource::<AutosaveTimer>();
        app.init_resource::<crate::dataset_export::ExportDatasetRequested>();
        app.init_resource::<crate::dataset_export::AutoExportSchedule>();
        // Rotate prior-run dataset artefacts ONCE at process start,
        // before any auto-export fires (the earliest milestone is
        // 5 virtual minutes, so the Startup schedule has all the
        // time it needs).
        app.add_systems(Startup, |_w: &mut World| {
            crate::dataset_export::rotate_existing_datasets();
        });
        app.init_resource::<crate::time_series_log::TimeSeriesLogger>();
        app.init_resource::<crate::limb_time_series_log::LimbTimeSeriesLogger>();
        app.add_systems(Update, spawn_colony.run_if(resource_exists::<HeightmapSampler>));
        app.add_systems(Update, animate_limbs);
        app.add_systems(Update, save_colony_system);
        app.add_systems(Update, autosave_system);
        app.add_systems(Update, auto_spawn_heteros);
        app.add_systems(Update, crate::dataset_export::tick_auto_export_schedule);
        app.add_systems(Update, crate::dataset_export::export_dataset_system);
        app.add_systems(Update, crate::time_series_log::tick_time_series_logger);
        app.add_systems(Update, crate::limb_time_series_log::tick_limb_time_series_logger);
        // Default lineage record attachment. Fires once per organism
        // — on the tick AFTER spawn (Added<OrganismRoot> filter).
        // Offspring of reproduction already get a LineageRecord
        // explicitly inserted by `reproduction.rs` (with `parent_id`
        // populated), so the `Without<LineageRecord>` filter skips
        // them here.
        app.add_systems(Update, assign_lineage_records);
    }
}


/// Default attach: every OrganismRoot that doesn't already have a
/// `LineageRecord` gets one minted as `new_initial` (parent_id =
/// None, spawn_time = current virtual time). Covers initial cohort,
/// auto-spawn, editor placement, and loaded organisms. Reproduction
/// offspring already carry a `new_offspring` record by the time
/// this runs.
fn assign_lineage_records(
    mut commands:  Commands,
    virtual_time:  Res<Time<Virtual>>,
    new_q:         Query<Entity, (Added<OrganismRoot>, Without<LineageRecord>)>,
) {
    let t = virtual_time.elapsed_secs();
    for e in &new_q {
        commands.entity(e).try_insert(LineageRecord::new_initial(t));
    }
}


/// Optional path to a `.colony` file. When `Some(path)`, the very first
/// `spawn_colony` invocation reads that file and spawns the saved
/// organisms instead of generating a fresh colony. When `None`, the
/// fresh-colony path runs as before. Set by `main.rs` from the second
/// positional CLI argument; defaults to `None` (`Option::default()`)
/// when no caller provides a value.
#[derive(Resource, Default)]
pub struct ColonyLoadPath(pub Option<String>);




/// Materialise a loaded organism record. Mirrors `spawn_organism`'s entity
/// hierarchy construction (root + body-part children, branches parented
/// to their attachment-target body part) but uses the saved Transform
/// and the fully-populated Organism component AS-IS — no fresh randomised
/// movement direction, no rebuilt cell-count caches, no `recompute_body_parts`
/// call (the saved cells already carry their neighbour counts and the
/// photo cache was reconstructed during decoding).
fn spawn_loaded_organism(
    record:    LoadedRecord,
    smoothing: bool,
    commands:  &mut Commands,
    meshes:    &mut ResMut<Assets<Mesh>>,
    materials: &OrganismMaterials,
    rng:       &mut impl rand::Rng,
) -> Entity {
    let LoadedRecord { pos, rotation, kind, organism, brain, brain_limb } = record;
    if organism.body_parts.is_empty() {
        // Defensive — skip organisms with no body parts (should never
        // happen on a valid save).
        return commands.spawn_empty().id();
    }

    let body_parts_snapshot = organism.body_parts.clone();
    // Capture intelligence level + adult flag + movement paradigm
    // before `organism` is moved into the spawn bundle. The Avian
    // physics block below needs `sliding_movement`.
    let intelligence_level = organism.intelligence_level;
    let adult              = organism.adult;
    let sliding_movement   = organism.sliding_movement;
    let direction_interval = 1.0 + rng.random::<f32>() * 9.0;

    let mut root_cmd = commands.spawn((
        Transform { translation: pos, rotation, ..default() },
        Visibility::Visible,
        OrganismRoot,
        organism,
        DirectionTimer::new(direction_interval),
    ));
    match kind {
        OrganismKind::Photoautotroph => { root_cmd.insert(Photoautotroph); }
        OrganismKind::Heterotroph    => { root_cmd.insert(Heterotroph); }
    }
    match intelligence_level {
        IntelligenceLevel::Level0 => {
            root_cmd.insert(crate::intelligence_level_0::BrainLevel0);
        }
        IntelligenceLevel::Level1
        | IntelligenceLevel::Level2
        | IntelligenceLevel::Level3 => {}
    }
    // Attach the saved brain weights (if any) so the matching pool's
    // `assign_brains_*` writes them into the slot it picks. Skipped
    // for Level 0 (no pool) and for v001/v002 saves (no brain bytes).
    if let Some(b) = brain {
        if !matches!(intelligence_level, IntelligenceLevel::Level0) {
            root_cmd.insert(b);
        }
    }
    // v007 limb-brain payload: attached as a component so the matching
    // limb pool's `assign_brains_*_limb` writes the weights into the
    // freshly-allocated row. Routing is implicit — each pool's assign
    // filter (`intelligence_level + !sliding_movement`) picks exactly
    // one consumer.
    if let Some(b) = brain_limb {
        root_cmd.insert(b);
    }
    let root = root_cmd.id();

    let mut bp_entities: Vec<Entity> = Vec::with_capacity(body_parts_snapshot.len());
    for (idx, bp) in body_parts_snapshot.iter().enumerate() {
        // Adult-at-load → use the smoothed mesh, but only when
        // `smoothing` is on; otherwise the faceted build that
        // `continuous_growth` will eventually re-smooth (or leave
        // faceted, depending on the flag at that future tick).
        let mesh_handle = if bp.regrowable && !bp.ocg.is_empty() {
            let mesh = if adult && smoothing {
                build_smoothed_mesh_from_ocg(&bp.ocg)
            } else {
                build_mesh_from_ocg(&bp.ocg)
            };
            Some(meshes.add(mesh))
        } else {
            None
        };

        let transform = match &bp.attachment {
            Some(a) => Transform {
                translation: a.origin_local,
                rotation:    a.rotation,
                ..Default::default()
            },
            None => Transform::from_translation(Vec3::ZERO),
        };

        let mut child_cmd = commands.spawn((
            transform,
            Visibility::Visible,
            BodyPartIndex(idx),
            ShowGizmo,
        ));
        if let Some(mh) = mesh_handle {
            let mat = materials.handle_for(kind, bp);
            child_cmd.insert((
                Mesh3d(mh),
                MeshMaterial3d(mat),
                OrganismMesh,
                // Per-organism meshes don't cast shadows: the shadow
                // pass would otherwise re-extract every body-part
                // entity per cascade per frame, dominating render cost.
                // Terrain still casts shadows; organism shadows are
                // negligible at our cell scale.
                bevy::light::NotShadowCaster,
            ));
        }
        let child = child_cmd.id();

        // Flat hierarchy — see the matching block in `spawn_organism`
        // for the rationale (phantom-organism fix on body-part despawn
        // cascade).
        let _ = &bp.attachment;
        commands.entity(root).add_child(child);

        bp_entities.push(child);
    }

    // ── Avian3d physics + LimbAnimation (same code path as
    //   `spawn_organism`). Without this, loaded limb-based organisms
    //   would have body-part entities but no rigid bodies, no joints,
    //   no contact flags, no animation — the brain runs but the PD
    //   torque step has nothing to torque, so the organism freezes
    //   at its saved pose and looks "limbless / unmoving" compared
    //   to its fresh-spawned counterpart. Sliding organisms also
    //   need their kinematic compound collider so other physics
    //   bodies can collide with them.
    if sliding_movement {
        // Single compound on the root.
        let mut shapes: Vec<(avian3d::prelude::Position, avian3d::prelude::Rotation, avian3d::prelude::Collider)> = Vec::new();
        for bp in &body_parts_snapshot {
            if !bp.is_alive() { continue; }
            let part_offset = bp.attachment.as_ref()
                .map_or(Vec3::ZERO, |a| a.origin_local);
            for cell in &bp.cells {
                let pos = part_offset + cell.local_pos;
                shapes.push((
                    avian3d::prelude::Position(pos),
                    avian3d::prelude::Rotation::default(),
                    avian3d::prelude::Collider::sphere(
                        crate::cell::CELL_COLLISION_RADIUS,
                    ),
                ));
            }
        }
        if !shapes.is_empty() {
            commands.entity(root).insert((
                avian3d::prelude::RigidBody::Kinematic,
                avian3d::prelude::Collider::compound(shapes),
            ));
        }
        // LimbAnimation on procedural-style limb body parts (kind =
        // Limb). Mirrors the in-loop insertion in spawn_organism.
        for (idx, bp) in body_parts_snapshot.iter().enumerate() {
            if !bp.is_alive() { continue; }
            if !matches!(bp.kind, crate::cell::BodyPartKind::Limb) { continue; }
            use rand::RngExt;
            let mirror = bp.attachment.as_ref()
                .is_some_and(|a| a.origin_local.x < 0.0);
            let two_pi = std::f32::consts::TAU;
            let la = LimbAnimation {
                freqs:  [
                    rng.random_range(1.5..5.0),
                    rng.random_range(1.5..5.0),
                    rng.random_range(1.5..5.0),
                ],
                phases: [
                    rng.random::<f32>() * two_pi,
                    rng.random::<f32>() * two_pi,
                    rng.random::<f32>() * two_pi,
                ],
                amps:   [
                    rng.random_range(0.3..0.7),
                    rng.random_range(0.3..0.7),
                    rng.random_range(0.3..0.7),
                ],
                mirror,
            };
            commands.entity(bp_entities[idx]).insert(la);
        }
    } else {
        // Limb-based: per-part Dynamic + joints + contact flags.
        for (idx, bp) in body_parts_snapshot.iter().enumerate() {
            if !bp.is_alive() { continue; }
            // Single convex-hull collider per body part (see
            // `limb_part_collider`) instead of a compound of one sphere
            // per cell — far cheaper for Avian's narrow phase.
            if let Some(collider) = limb_part_collider(&bp.cells) {
                let ang_damping = if idx == 0 { BASE_ANGULAR_DAMPING } else { LIMB_ANGULAR_DAMPING };
                // Friction for LIMB-PROPELLED crawling: the LIMB feet GRIP
                // (high μ, Max combine) so a leg stroke pushes off the ground
                // (the limbs do the propulsion — the guiderail), while the BASE
                // belly is LOW-friction (Min combine) so the gripping-foot push
                // slides the body forward instead of pinning it. The per-part
                // floor (`enforce_limb_floor`) keeps every part at/above the
                // surface so this stays non-penetrating and natural-looking.
                let (fric_coeff, fric_rule) = if idx == 0 {
                    (crate::simulation_settings::BASE_FRICTION_COEFFICIENT,
                     avian3d::prelude::CoefficientCombine::Min)
                } else {
                    (crate::simulation_settings::LIMB_FRICTION_COEFFICIENT,
                     avian3d::prelude::CoefficientCombine::Max)
                };
                commands.entity(bp_entities[idx]).insert((
                    avian3d::prelude::RigidBody::Dynamic,
                    collider.clone(),
                    // Avian's auto-mass-compute pipeline silently fails
                    // to populate `ComputedMass` / `ComputedAngularInertia`
                    // for these compound colliders. Defaults of those
                    // types store INVERSE values that init to zero
                    // (representing infinite mass/inertia), so every
                    // `Forces::apply_force` / `apply_torque` call
                    // computes `acceleration = inverse_mass * force = 0`
                    // and the body never moves. Explicitly deriving the
                    // mass properties from the compound's shape ×
                    // `BODY_PART_DENSITY` bypasses the broken auto-
                    // compute path. Diagnosed May 2026 via the
                    // constant-torque experiment in `avian_setup`.
                    avian3d::prelude::MassPropertiesBundle::from_shape(&collider, BODY_PART_DENSITY),
                    // Per-part friction (see `fric_coeff`/`fric_rule` above):
                    // grippy feet (Max combine, μ→1.0) anchor the leg stroke;
                    // slippery belly (Min combine, μ→0.05) slides instead of
                    // pinning the body — together they propel an emergent crawl.
                    avian3d::prelude::Friction::new(fric_coeff)
                        .with_combine_rule(fric_rule),
                    avian3d::prelude::CollisionEventsEnabled,
                    crate::avian_setup::LimbContact::default(),
                    // Base body gets heavy damping to absorb joint-
                    // reaction torques; limbs get lighter damping so
                    // the PD controller can produce dynamic swings.
                    avian3d::prelude::AngularDamping(ang_damping),
                    avian3d::prelude::LinearDamping(LIMB_LINEAR_DAMPING),
                    // Avian's default sleep plugin freezes low-velocity
                    // bodies, and the brain's `Forces::apply_torque`
                    // writes don't wake them — every limb organism
                    // settled, slept, and never moved again. Marker
                    // keeps these bodies permanently integrated so the
                    // PD controller's torques actually take effect.
                    avian3d::prelude::SleepingDisabled,
                    // Telemetry: the PD controller writes the last
                    // commanded torque here every frame; the dataset
                    // exporter reads it to log torque_norm per organism.
                    crate::avian_setup::LastAppliedTorque::default(),
                    // Filter out self-collision: a creature's own body parts
                    // must not collide with each other. The joint chain holds
                    // them together; self-contacts fight that (contact pushes
                    // apart, joint pulls together) and the solver pumps that
                    // conflict into explosive linear+angular velocity on the
                    // light bodies. `SelfCollisionFilter` rejects same-organism
                    // pairs; this marker activates the filter hook for them.
                    avian3d::prelude::ActiveCollisionHooks::FILTER_PAIRS,
                    // Velocity governor: pursuit locomotion is steady (~2.5 u/s)
                    // and the gait is gentle, both well under these caps — so
                    // the caps only clip the rare joint-instability spike,
                    // bounding limb separation and preventing any fly-off,
                    // without throttling normal motion.
                    avian3d::prelude::MaxLinearSpeed(crate::simulation_settings::MAX_LIMB_LINEAR_SPEED),
                    avian3d::prelude::MaxAngularSpeed(crate::simulation_settings::MAX_LIMB_ANGULAR_SPEED),
                ));
            }
        }
        // Spherical joints — one per appendage. The pivot is placed on
        // the FACE between the parent cell adjacent to the attachment
        // and the limb's first cell (see `limb_joint_anchors`), so the
        // limb rotates like an anatomical shoulder joint rather than
        // around its own centroid. Compliance is set to a tiny non-
        // zero value (`LIMB_JOINT_COMPLIANCE`) — strictly-rigid joints
        // (compliance = 0) are numerically fragile under the high PD
        // torques the brain commands, and a microscopic give lets the
        // solver converge cleanly.
        for (idx, bp) in body_parts_snapshot.iter().enumerate() {
            if !bp.is_alive() { continue; }
            let Some(attach) = bp.attachment.as_ref() else { continue; };
            // Defensive: a stale / out-of-range `parent_idx` (possible
            // after sim-time predation soft-deletes + branch growth, or
            // a hand-edited save) would otherwise panic on the indexing
            // below. Skip the joint instead — the limb stays a free
            // dynamic body rather than crashing the spawn.
            let pidx = attach.parent_idx;
            if pidx >= body_parts_snapshot.len() || !body_parts_snapshot[pidx].is_alive() {
                warn!("limb joint skipped: parent_idx {pidx} out of range or consumed");
                continue;
            }
            let parent_e = bp_entities[pidx];
            let child_e  = bp_entities[idx];
            let parent_origin = body_parts_snapshot[pidx].attachment.as_ref()
                .map_or(Vec3::ZERO, |a| a.origin_local);
            let (anchor1, anchor2, hinge_axis) =
                limb_joint_anchors(parent_origin, &body_parts_snapshot[idx].cells, attach.origin_local);
            commands.spawn((
                // 1-DOF hinge: the limb swings in a single (vertical) plane and
                // physically CANNOT orbit the body — the two non-hinge axes are
                // held by a robust bilateral align constraint, unlike a ball
                // joint's fragile, singular swing/twist limits that the limb
                // kept spinning through. `align_compliance` leaves a small give
                // so the limb can twist a little at the attachment (like a real
                // shoulder/hip). Hinge axis ⟂ limb (vertical swing plane); see
                // `limb_joint_anchors`.
                avian3d::prelude::RevoluteJoint::new(parent_e, child_e)
                    .with_local_anchor1(anchor1)
                    .with_local_anchor2(anchor2)
                    .with_hinge_axis(hinge_axis)
                    .with_point_compliance(crate::avian_setup::LIMB_JOINT_COMPLIANCE)
                    .with_align_compliance(crate::simulation_settings::LIMB_HINGE_ALIGN_COMPLIANCE)
                    .with_angle_limits(
                        -crate::simulation_settings::LIMB_SWING_LIMIT,
                        crate::simulation_settings::LIMB_SWING_LIMIT,
                    )
                    // Built-in spring-damper angular motor (solved in-step,
                    // 1-DOF on the hinge axis). The brain sets its
                    // `target_position` each tick via `drive_limb_motors`;
                    // this is what replaced the unstable external PD torque.
                    .with_motor(
                        avian3d::prelude::AngularMotor::new(
                            avian3d::prelude::MotorModel::SpringDamper {
                                frequency:     crate::simulation_settings::LIMB_MOTOR_FREQUENCY,
                                damping_ratio: crate::simulation_settings::LIMB_MOTOR_DAMPING_RATIO,
                            },
                        )
                        .with_max_torque(crate::simulation_settings::MAX_LIMB_TORQUE),
                    ),
                // Disable collision between this limb and its direct
                // attachment parent: they overlap at the joint and would
                // otherwise generate persistent contacts every substep —
                // wasted narrow-phase work AND a solver fight (the joint
                // pulls them together while the contact pushes them
                // apart). Collision with NON-adjacent body parts of the
                // same organism is unaffected, so intra-body interactions
                // are preserved.
                avian3d::prelude::JointCollisionDisabled,
                // Link the joint back to its organism + limb so the motor
                // driver can set this hinge's target angle each tick.
                crate::avian_setup::LimbJointDrive {
                    organism:    root,
                    body_part:   idx,
                    limb_entity: child_e,
                },
            ));
        }
    }

    root
}




// Organism + markers + OrganismKind moved to `organism.rs`. They're
// re-exported above via `pub use crate::organism::*;` so existing imports
// continue to find them through `crate::colony::*`.

// ── Spawning ─────────────────────────────────────────────────────────────────

fn spawn_colony(
    mut commands:    Commands,
    mut meshes:      ResMut<Assets<Mesh>>,
    mut materials:   ResMut<Assets<StandardMaterial>>,
    heightmap:       Res<HeightmapSampler>,
    load_path:       Res<ColonyLoadPath>,
    smoothing:       Res<crate::simulation_settings::Smoothing>,
    map_size:        Res<MapSize>,
    // NOTE: the launcher start-count / cap resources no longer drive
    // the fresh cohort (it spawns fixed `.species` counts), but
    // `--max-herbivores` still sizes the GPU brain pools in `main.rs`.
    mut virtual_time:    ResMut<Time<Virtual>>,
    mut spawned:     Local<bool>,
) {
    if *spawned { return; }
    *spawned = true;

    let materials = OrganismMaterials::new(&mut materials);
    // Mirror the OrganismMaterials into a Resource so runtime spawners
    // (`auto_spawn_heteros`) can reuse the same handles without
    // rebuilding the StandardMaterials. Clone is cheap — each field is
    // a Handle<StandardMaterial>.
    commands.insert_resource(OrganismMaterials {
        photo:      materials.photo.clone(),
        hetero:     materials.hetero.clone(),
        debug_blue: materials.debug_blue.clone(),
    });
    let mut rng = rand::rng();

    // If a colony save file was supplied on the command line, try to
    // restore it. On any failure (missing file, malformed bytes, version
    // mismatch) we log the reason and fall through to fresh generation
    // so the run still produces something visible.
    if let Some(path) = &load_path.0 {
        match load_colony_from_file(path) {
            Ok((notation, records)) => {
                let n = records.len();
                for record in records {
                    spawn_loaded_organism(record, smoothing.0, &mut commands, &mut meshes, &materials, &mut rng);
                }
                // Resume the virtual clock at the saved point. Two
                // `advance_by` calls: the first bumps `elapsed` to the
                // target; the second (ZERO) resets THIS frame's `delta`
                // back to 0, so no delta-integrating system (physics
                // accumulator, movement) sees the jump as a single
                // giant timestep. Bevy's normal per-frame update resumes
                // from the new elapsed next frame.
                let total = notation.total_secs();
                if total > 0 {
                    virtual_time.advance_by(std::time::Duration::from_secs(total as u64));
                    virtual_time.advance_by(std::time::Duration::ZERO);
                }
                info!("loaded colony from {} — {} organisms restored, virtual time resumed at {}h{}m{}s",
                      path, n, notation.hours, notation.minutes, notation.seconds);
                return;
            }
            Err(e) => {
                error!("failed to load colony from {}: {} — falling back to fresh generation", path, e);
            }
        }
    }

    // Fresh-start cohort: seed the world from authored `.species`
    // files instead of procedurally-generated starter organisms. Each
    // file is loaded once and spawned `count` times at random
    // positions. Counts and paths are fixed (see the `SPAWN_*`
    // constants); the launcher's start-count fields no longer drive
    // the fresh cohort, though `--max-herbivores` still sizes the GPU
    // brain pools in `main.rs`.
    spawn_species_cohort(SPAWN_ALGAE_PATH,   SPAWN_ALGAE_COUNT,   &heightmap, &map_size, smoothing.0, &mut commands, &mut meshes, &materials, &mut rng);
    spawn_species_cohort(SPAWN_CRAWLER_PATH, SPAWN_CRAWLER_COUNT, &heightmap, &map_size, smoothing.0, &mut commands, &mut meshes, &materials, &mut rng);
    spawn_species_cohort(SPAWN_STRIDER_PATH, SPAWN_STRIDER_COUNT, &heightmap, &map_size, smoothing.0, &mut commands, &mut meshes, &materials, &mut rng);
}


/// Build a STATIC appendage `BodyPart` from a full OCG (cells render at
/// their authored positions; no per-frame rotation). Mirrors
/// `colony_editor::placement::appendage_body_part`.
fn appendage_body_part_from_ocg(ocg: Vec<(usize, Vec3, CellType)>) -> BodyPart {
    let cells = ocg.iter().map(|(_, p, ct)| Cell::new(*p, *ct)).collect();
    BodyPart {
        kind:         BodyPartKind::Organ,
        local_offset: Vec3::ZERO,
        cells,
        ocg,
        attachment:   Some(crate::body_part::Attachment {
            parent_idx: 0, origin_local: Vec3::ZERO, rotation: Quat::IDENTITY,
        }),
        consumed:   false,
        debug_blue: false,
        regrowable: true,
    }
}

/// Build a LIMB `BodyPart` from a full OCG: cells are rebased so the
/// first cell sits at the limb's local origin and the attachment pivot
/// is that first cell, so the limb rotates around its base. Mirrors
/// `colony_editor::placement::limb_body_part`.
fn limb_body_part_from_ocg(ocg: Vec<(usize, Vec3, CellType)>) -> BodyPart {
    let pivot = ocg.first().map(|(_, p, _)| *p).unwrap_or(Vec3::ZERO);
    let shifted: Vec<(usize, Vec3, CellType)> =
        ocg.iter().map(|(i, p, ct)| (*i, *p - pivot, *ct)).collect();
    let cells = shifted.iter().map(|(_, p, ct)| Cell::new(*p, *ct)).collect();
    BodyPart {
        kind:         BodyPartKind::Limb,
        local_offset: Vec3::ZERO,
        cells,
        ocg:          shifted,
        attachment:   Some(crate::body_part::Attachment {
            parent_idx: 0, origin_local: pivot, rotation: Quat::IDENTITY,
        }),
        consumed:   false,
        debug_blue: false,
        regrowable: true,
    }
}

/// Load a `.species` file and spawn `count` instances at random
/// positions across the world. Used by `spawn_colony`'s fresh-start
/// path. Body-part assembly (root + appendages, bilateral mirroring,
/// limb-pivot rebasing) mirrors
/// `colony_editor::placement::spawn_real_organism`, so a species spawns
/// identically whether placed in the editor or seeded here. Each spawn
/// gets an `ImportedSpeciesOrigin` (filename stem) so it founds its own
/// lineage in the speciation registry. Errors (missing/bad file) are
/// logged and the cohort is skipped.
#[allow(clippy::too_many_arguments)]
fn spawn_species_cohort(
    path:      &str,
    count:     usize,
    heightmap: &HeightmapSampler,
    map_size:  &MapSize,
    smoothing: bool,
    commands:  &mut Commands,
    meshes:    &mut ResMut<Assets<Mesh>>,
    materials: &OrganismMaterials,
    rng:       &mut impl rand::Rng,
) {
    let species = match crate::species_editor::save::load_species(std::path::Path::new(path)) {
        Ok(s)  => s,
        Err(e) => { error!("spawn_colony: failed to load species {path}: {e}"); return; }
    };
    if species.body_parts.is_empty() {
        error!("spawn_colony: species {path} has no body parts — skipped");
        return;
    }
    let name = std::path::Path::new(path).file_stem()
        .and_then(|s| s.to_str()).unwrap_or("species").to_string();
    let kind = match species.metabolism {
        crate::species_editor::session::Metabolism::Photoautotroph => OrganismKind::Photoautotroph,
        crate::species_editor::session::Metabolism::Heterotroph    => OrganismKind::Heterotroph,
    };
    let is_carnivore = matches!(
        species.classification,
        crate::species_editor::session::Classification::Carnivore
    );
    let sliding = species.movement.is_sliding();
    let margin  = crate::world_geometry::WORLD_SAFETY_MARGIN;

    for _ in 0..count {
        // Body-part list rebuilt per spawn (spawn_organism consumes it):
        // root from part 0; each later part is an appendage (bilateral
        // → mirrored pair, NoSymmetry → single).
        let mut body_parts = vec![root_body_part_from_ocg(&species.body_parts[0].ocg)];
        for lbp in &species.body_parts[1..] {
            let make = |o: Vec<(usize, Vec3, CellType)>| -> BodyPart {
                if lbp.is_limb { limb_body_part_from_ocg(o) } else { appendage_body_part_from_ocg(o) }
            };
            match species.symmetry {
                Symmetry::Bilateral => {
                    body_parts.push(make(lbp.ocg.clone()));
                    body_parts.push(make(crate::body_part::mirror_right_to_left(&lbp.ocg)));
                }
                Symmetry::NoSymmetry => body_parts.push(make(lbp.ocg.clone())),
            }
        }
        let cell_count = body_parts.iter().map(|bp| bp.cells.len()).sum::<usize>() as f32;
        let initial_energy = cell_count * crate::energy::MAX_ENERGY_PER_CELL * 0.5;

        let x = rng.random_range(margin..(map_size.x - margin));
        let z = rng.random_range(margin..(map_size.z - margin));
        let y = heightmap.height_at(x, z) + 1.0;

        let entity = spawn_organism(
            Vec3::new(x, y, z),
            body_parts,
            kind,
            species.symmetry,
            species.has_variable_form,
            species.is_sessile,
            species.intelligence,
            smoothing,
            initial_energy,
            sliding,
            commands,
            meshes,
            materials,
            rng,
        );
        if is_carnivore {
            commands.entity(entity).try_insert(Carnivore);
        }
        commands.entity(entity).try_insert(
            crate::lineages::species::ImportedSpeciesOrigin { name: name.clone() },
        );
        if let Some(b) = &species.brain {
            commands.entity(entity).try_insert(b.clone());
        }
    }
    info!("spawn_colony: spawned {count} × {name} from {path}");
}


/// Build the canonical root body part from a flat OCG. Cells mirror the OCG
/// positions; the part is `regrowable` (mutation can extend it) and not in
/// debug-blue mode.
pub fn root_body_part_from_ocg(ocg: &[(usize, Vec3, CellType)]) -> BodyPart {
    let cells = ocg.iter()
        .map(|(_, pos, ct)| Cell::new(*pos, *ct))
        .collect();
    BodyPart {
        kind:         BodyPartKind::Body,
        local_offset: Vec3::ZERO,
        cells,
        ocg:          ocg.to_vec(),
        attachment:   None,
        consumed:     false,
        debug_blue:   false,
        regrowable:   true,
    }
}


/// Set of shared StandardMaterial handles passed to `spawn_organism`. Sharing
/// one handle per (kind × debug-flag) pair keeps GPU bind-group churn minimal.
///
/// Stored as a `Resource` so runtime spawners (`auto_spawn_heteros`,
/// editor-mode placement) can reuse the same handles without rebuilding
/// the `StandardMaterial`s. `spawn_colony` populates it on first run.
#[derive(Resource)]
pub struct OrganismMaterials {
    pub photo:       Handle<StandardMaterial>,
    pub hetero:      Handle<StandardMaterial>,
    pub debug_blue:  Handle<StandardMaterial>,
}

impl OrganismMaterials {
    pub fn new(materials: &mut Assets<StandardMaterial>) -> Self {
        // The trophic material is now WHITE so the mesh's per-vertex
        // colours (assigned per cell by `build_mesh_from_ocg`) come
        // through unmultiplied — every cell shows its own colour
        // regardless of the body part's overall trophic kind. The
        // `photo` and `hetero` handles share one underlying material.
        let body = materials.add(StandardMaterial {
            base_color: Color::WHITE,
            ..default()
        });
        Self {
            photo:      body.clone(),
            hetero:     body,
            // `debug_blue` is kept as a full-part override for
            // procedural reproduction appendages (which seed from a
            // non-Placeholder cell and would otherwise look identical
            // to the base body). Set on the BodyPart's `debug_blue`
            // flag at creation time.
            debug_blue: materials.add(StandardMaterial {
                base_color: Color::srgb(0.2, 0.4, 0.95),
                ..default()
            }),
        }
    }

    /// Pick the material handle for one body part, given the trophic kind of
    /// its owning organism. Per-cell colouring lives on the mesh itself
    /// (`Mesh::ATTRIBUTE_COLOR`), so this only needs to choose between the
    /// shared white body material and the debug-blue full-part override.
    pub fn handle_for(&self, _kind: OrganismKind, bp: &BodyPart) -> Handle<StandardMaterial> {
        if bp.debug_blue {
            self.debug_blue.clone()
        } else {
            // Same handle as `self.hetero`; `kind` is unused now that
            // colour comes from per-vertex data.
            self.photo.clone()
        }
    }
}


/// Per-limb erratic-rotation parameters. Attached at spawn to every
/// body-part entity whose `BodyPart::kind == BodyPartKind::Limb`. Each
/// axis (X / Y / Z) oscillates independently with its own random
/// frequency, phase and amplitude — incommensurate frequencies make the
/// compound rotation read as chaotic rather than periodic. `mirror`
/// inverts all three angles so the left half of a bilateral limb pair
/// swings opposite to its right twin.
#[derive(Component, Clone, Debug)]
pub struct LimbAnimation {
    pub freqs:  [f32; 3],
    pub phases: [f32; 3],
    pub amps:   [f32; 3],
    pub mirror: bool,
}

/// Apply the per-limb erratic rotation each frame. Runs in `Update`;
/// `TransformSystems::Propagate` (PostUpdate) picks it up the same
/// frame. Reads the default `Time` (virtual clock), so animation freezes
/// in lock-step with the rest of the simulation when paused.
pub fn animate_limbs(
    time:  Res<Time>,
    mut q: Query<(&LimbAnimation, &mut Transform)>,
) {
    let t = time.elapsed_secs();
    for (la, mut tr) in &mut q {
        let sign = if la.mirror { -1.0 } else { 1.0 };
        let x = sign * la.amps[0] * (t * la.freqs[0] + la.phases[0]).sin();
        let y = sign * la.amps[1] * (t * la.freqs[1] + la.phases[1]).sin();
        let z = sign * la.amps[2] * (t * la.freqs[2] + la.phases[2]).sin();
        tr.rotation = Quat::from_euler(EulerRot::XYZ, x, y, z);
    }
}


/// Construct + register an organism from a list of body parts at world
/// position `pos`. Each body part owns its own OCG; the mesh for each
/// regrowable part is built by replaying the part's OCG through
/// `build_mesh_from_ocg`. Used by both initial colony spawn and
/// reproduction.
///
/// Hierarchy produced:
///   OrganismRoot (transform = pos, has Organism + trophic marker)
///   ├── body-part-0 child (Mesh3d, parent of any branches that attach to it)
///   │   └── body-part-1 child (if attached to part 0; rotates around its
///   │                          attachment.origin_local in part-0's frame)
///   └── ...
pub fn spawn_organism(
    pos:                Vec3,
    mut body_parts:     Vec<BodyPart>,
    kind:               OrganismKind,
    symmetry:           Symmetry,
    has_variable_form:  bool,
    is_sessile:         bool,
    intelligence_level: IntelligenceLevel,
    smoothing:          bool,
    initial_energy:     f32,
    // Movement paradigm — see `Organism::sliding_movement`. Pass `true`
    // for legacy / sliding organisms (current behaviour); `false` to
    // spawn the organism into Avian's physics world for limb-based
    // locomotion. Defaults to `true` at every current call site.
    sliding_movement:   bool,
    commands:           &mut Commands,
    meshes:             &mut ResMut<Assets<Mesh>>,
    materials:          &OrganismMaterials,
    rng:                &mut impl rand::Rng,
) -> Entity {
    // Enforce the invariant: variable-form organisms are always sessile and
    // always NoSymmetry. Caller bugs that violate this are silently fixed
    // up here so downstream systems can trust the fields.
    let symmetry  = if has_variable_form { Symmetry::NoSymmetry } else { symmetry };
    let is_sessile = is_sessile || has_variable_form;
    // Sessile organisms MUST be on the sliding-movement path: their root
    // is a `RigidBody::Kinematic` (immovable to physics; `apply_movement`
    // skips it). Spawning a sessile organism into the limb-based Dynamic
    // body path would let gravity, joint reactions, and collisions push
    // it around — which presents to the player as a "sliding" sessile
    // organism, the exact thing the sessile flag is supposed to prevent.
    let sliding_movement = sliding_movement || is_sessile;
    if body_parts.is_empty() {
        // Defensive — callers should always provide at least the root part.
        // Returning a bogus entity would silently corrupt downstream queries.
        panic!("spawn_organism called with empty body_parts");
    }

    // Bring per-cell physiology caches in sync with the assembled cell list
    // before the Organism component is built. This populates each cell's
    // `neighbour_count` and (for Photo cells) `PhotosyntheticCell::*`. The
    // photosynthesis tick reads those caches per-frame; if we skipped this
    // step every cell would produce zero energy.
    crate::physiology::recompute_body_parts(&mut body_parts);

    let angle     = rng.random::<f32>() * std::f32::consts::TAU;
    let direction = Vec3::new(angle.cos(), 0.0, angle.sin());
    let speed     = match kind {
        OrganismKind::Photoautotroph => 0.0,
        OrganismKind::Heterotroph    => 15.0 + rng.random::<f32>() * 10.0,
    };

    // Adult at spawn for any organism that won't grow during its own
    // lifetime. Variable-form organisms grow via `continuous_growth`
    // and only become adult once they reach `MAX_CELLS`; everything
    // else is born "fully grown" (its body plan is whatever it was
    // born with — reproduction grows the OFFSPRING by one cell, not
    // the parent).
    let adult = !has_variable_form;

    let mut organism = Organism {
        body_parts: body_parts.clone(),
        symmetry,
        intelligence_level,
        is_sessile,
        has_variable_form,
        sliding_movement,
        limb_targets: [0.0; 8],
        adult,
        photo_cell_count:     0,
        non_photo_cell_count: 0,
        energy: initial_energy.max(0.0),
        in_sunlight: false,
        reproduced: false,
        reproductions: 0,
        predations: 0,
        hunger: 0.0,
        dopamine: 0.0,
        target_distance: crate::sensory::SENSORY_RADIUS,
        movement_speed: speed,
        movement_direction: direction,
        velocity: Vec3::ZERO,
        is_climbing: false,
        climb_energy_debt: 0.0,
        // Populated by recompute_cell_counts below.
        cached_bounding_radius: 0.0,
        // Structural DNA slots — filled here so newly-reproduced /
        // user-placed organisms enter the speciation system with a
        // valid vector from frame 1. Brain-gene slots stay 0 until
        // `sync_dna_from_brain_pool` runs.
        dna: crate::lineages::dna::structural_dna(
            kind,
            symmetry,
            has_variable_form,
            is_sessile,
            intelligence_level,
        ),
        // Inherited from parent (when reproducing) in a separate
        // post-spawn step by `reproduction.rs`; left `None` for
        // initial-cohort spawns + editor placements so the
        // classification tick assigns them.
        species_id: None,
    };
    organism.recompute_cell_counts();

    let direction_interval = 1.0 + rng.random::<f32>() * 9.0;
    let spawn_rotation     = Quat::from_rotation_y(angle);

    let mut root_cmd = commands.spawn((
        Transform::from_translation(pos).with_rotation(spawn_rotation),
        Visibility::Visible,
        OrganismRoot,
        organism,
        DirectionTimer::new(direction_interval),
    ));
    match kind {
        OrganismKind::Photoautotroph => { root_cmd.insert(Photoautotroph); }
        OrganismKind::Heterotroph    => { root_cmd.insert(Heterotroph); }
    }
    // Wire intelligence-pool markers off the stored `intelligence_level`
    // field. Level 0 → `BrainLevel0` keeps the entity out of the L1/L3
    // assign queries. Levels 1 and 3 don't need a marker at spawn —
    // their `assign_*` systems pick the entity up via the trophic
    // marker (`Photoautotroph` / `Heterotroph`) and assign a slot.
    match intelligence_level {
        IntelligenceLevel::Level0 => {
            root_cmd.insert(crate::intelligence_level_0::BrainLevel0);
        }
        IntelligenceLevel::Level1
        | IntelligenceLevel::Level2
        | IntelligenceLevel::Level3 => {}
    }
    let root = root_cmd.id();

    // Spawn one mesh child per body part. Branches are parented to their
    // attachment-target body part's entity so Bevy's transform propagation
    // gives rotation-around-origin for free. We assume body parts are
    // ordered such that any branch's parent_idx < its own index, which is
    // the order callers naturally produce (parts cloned then appended).
    let mut bp_entities: Vec<Entity> = Vec::with_capacity(body_parts.len());
    for (idx, bp) in body_parts.iter().enumerate() {
        // Skip mesh creation for non-regrowable / empty parts (e.g. Krishi).
        // Adult organisms get the smoothed mesh straight away (when
        // `smoothing` is on); growing (variable-form) organisms get the
        // faceted mesh that `continuous_growth` will replace once they
        // reach MAX_CELLS. With `smoothing` off the faceted mesh is
        // used unconditionally.
        let mesh_handle = if bp.regrowable && !bp.ocg.is_empty() {
            let mesh = if adult && smoothing {
                build_smoothed_mesh_from_ocg(&bp.ocg)
            } else {
                build_mesh_from_ocg(&bp.ocg)
            };
            Some(meshes.add(mesh))
        } else {
            None
        };

        // Branch transform: translate by origin_local in parent's frame,
        // apply attachment.rotation. Root: identity.
        let transform = match &bp.attachment {
            Some(a) => Transform {
                translation: a.origin_local,
                rotation:    a.rotation,
                ..Default::default()
            },
            None => Transform::from_translation(Vec3::ZERO),
        };

        let mut child_cmd = commands.spawn((
            transform,
            Visibility::Visible,
            BodyPartIndex(idx),
            ShowGizmo,
        ));
        if let Some(mh) = mesh_handle {
            let mat = materials.handle_for(kind, bp);
            child_cmd.insert((
                Mesh3d(mh),
                MeshMaterial3d(mat),
                OrganismMesh,
                bevy::light::NotShadowCaster,
            ));
        }
        let child = child_cmd.id();

        // Limb animation: kind=Limb body parts get the `LimbAnimation`
        // marker with randomized per-axis oscillator parameters. Mirror
        // twins (entity origin on the −X side) get the inverted-sign
        // animation so the pair swings opposite, like a real animal's
        // limbs. The actual rotation update lives in `animate_limbs`.
        //
        // Limb-based organisms drive their joints via the physics
        // engine (Phase 4 hooks brain PD targets into Avian), so the
        // kinematic rotation marker is only inserted for sliding
        // organisms — physics-driven limbs would fight the animation.
        if sliding_movement
            && matches!(bp.kind, crate::cell::BodyPartKind::Limb) {
            use rand::RngExt;
            let mirror = bp.attachment.as_ref()
                .is_some_and(|a| a.origin_local.x < 0.0);
            let two_pi = std::f32::consts::TAU;
            let la = LimbAnimation {
                freqs:  [
                    rng.random_range(1.5..5.0),
                    rng.random_range(1.5..5.0),
                    rng.random_range(1.5..5.0),
                ],
                phases: [
                    rng.random::<f32>() * two_pi,
                    rng.random::<f32>() * two_pi,
                    rng.random::<f32>() * two_pi,
                ],
                amps:   [
                    rng.random_range(0.3..0.7),
                    rng.random_range(0.3..0.7),
                    rng.random_range(0.3..0.7),
                ],
                mirror,
            };
            commands.entity(child).insert(la);
        }

        // Parent EVERY body-part entity directly under the OrganismRoot
        // — branches included. Earlier this nested branches under their
        // parent body-part entity for "rotation-around-pivot via
        // transform propagation", but body-part transforms are identity
        // in practice (Quat::IDENTITY rotation, Vec3::ZERO translation
        // on the root part) so siblings-under-root gives the same
        // world transform. The flat layout fixes a phantom-organism bug:
        // when predation despawned body_parts[0]'s entity, Bevy's
        // recursive try_despawn cascaded to every branch nested under
        // it, even though `prey_dead == false` because other body
        // parts were still alive in the Organism data — the root
        // entity then survived with zero children and the data
        // continued to claim alive body parts, becoming an invisible
        // ghost that still photosynthesised and reproduced.
        let _ = &bp.attachment; // attachment kept on the data side for
                                // continuous_growth's branch geometry;
                                // the entity hierarchy is flat.
        commands.entity(root).add_child(child);

        bp_entities.push(child);
    }

    // ── Avian3d physics components ─────────────────────────────────────
    //
    // Sliding organisms (current behaviour for everything):
    //   * `RigidBody::Kinematic` on the OrganismRoot — its transform is
    //     written by `apply_movement`; Avian's kinematic sync follows.
    //   * One compound collider on the root built from every cell
    //     across every body part, with cell positions offset by the
    //     body part's attachment origin (so the collider matches the
    //     world positions the rest of the codebase already uses).
    //
    // Limb-based organisms (none until Phase 5's species-editor toggle):
    //   * Each body part entity gets `RigidBody::Dynamic` + its own
    //     compound collider built from THAT part's cells.
    //   * For each appendage (`attachment.is_some()`) we spawn a
    //     `SphericalJoint` entity connecting the parent body-part
    //     entity to the appendage entity, anchored at
    //     `attachment.origin_local` on the parent and `Vec3::ZERO` on
    //     the child (limb body parts are rebased so their local origin
    //     sits at the first-cell pivot).
    if sliding_movement {
        let mut shapes: Vec<(avian3d::prelude::Position, avian3d::prelude::Rotation, avian3d::prelude::Collider)> = Vec::new();
        for bp in &body_parts {
            if !bp.is_alive() { continue; }
            let part_offset = bp.attachment.as_ref()
                .map_or(Vec3::ZERO, |a| a.origin_local);
            for cell in &bp.cells {
                let pos = part_offset + cell.local_pos;
                shapes.push((
                    avian3d::prelude::Position(pos),
                    avian3d::prelude::Rotation::default(),
                    avian3d::prelude::Collider::sphere(
                        crate::cell::CELL_COLLISION_RADIUS,
                    ),
                ));
            }
        }
        if !shapes.is_empty() {
            commands.entity(root).insert((
                avian3d::prelude::RigidBody::Kinematic,
                avian3d::prelude::Collider::compound(shapes),
            ));
        }
    } else {
        // Limb-based: per-part dynamic bodies.
        for (idx, bp) in body_parts.iter().enumerate() {
            if !bp.is_alive() { continue; }
            // Single convex-hull collider per body part (see
            // `limb_part_collider`) — matches the spawn_organism path.
            if let Some(collider) = limb_part_collider(&bp.cells) {
                let ang_damping = if idx == 0 { BASE_ANGULAR_DAMPING } else { LIMB_ANGULAR_DAMPING };
                // Friction for LIMB-PROPELLED crawling: the LIMB feet GRIP
                // (high μ, Max combine) so a leg stroke pushes off the ground
                // (the limbs do the propulsion — the guiderail), while the BASE
                // belly is LOW-friction (Min combine) so the gripping-foot push
                // slides the body forward instead of pinning it. The per-part
                // floor (`enforce_limb_floor`) keeps every part at/above the
                // surface so this stays non-penetrating and natural-looking.
                let (fric_coeff, fric_rule) = if idx == 0 {
                    (crate::simulation_settings::BASE_FRICTION_COEFFICIENT,
                     avian3d::prelude::CoefficientCombine::Min)
                } else {
                    (crate::simulation_settings::LIMB_FRICTION_COEFFICIENT,
                     avian3d::prelude::CoefficientCombine::Max)
                };
                commands.entity(bp_entities[idx]).insert((
                    avian3d::prelude::RigidBody::Dynamic,
                    collider.clone(),
                    // Density lowered from 1.0 so total organism mass
                    // stays light enough that ground-contact friction
                    // can be overcome by the brain's PD torques. See
                    // `BODY_PART_DENSITY` for rationale.
                    avian3d::prelude::MassPropertiesBundle::from_shape(&collider, BODY_PART_DENSITY),
                    // Per-part friction (see `fric_coeff`/`fric_rule` above):
                    // grippy feet (Max combine, μ→1.0) anchor the leg stroke;
                    // slippery belly (Min combine, μ→0.05) slides instead of
                    // pinning the body — together they propel an emergent crawl.
                    avian3d::prelude::Friction::new(fric_coeff)
                        .with_combine_rule(fric_rule),
                    // Per-entity event toggle so Avian's narrow phase
                    // emits `CollisionStart` / `CollisionEnd` messages
                    // for this body part — consumed by
                    // `update_limb_contacts` to populate `LimbContact`.
                    avian3d::prelude::CollisionEventsEnabled,
                    crate::avian_setup::LimbContact::default(),
                    // Base body gets heavy damping to absorb joint-
                    // reaction torques; limbs get lighter damping so
                    // the PD controller can produce dynamic swings.
                    avian3d::prelude::AngularDamping(ang_damping),
                    avian3d::prelude::LinearDamping(LIMB_LINEAR_DAMPING),
                    // Avian's default sleep plugin freezes low-velocity
                    // bodies, and the brain's `Forces::apply_torque`
                    // writes don't wake them — every limb organism
                    // settled, slept, and never moved again. Marker
                    // keeps these bodies permanently integrated so the
                    // PD controller's torques actually take effect.
                    avian3d::prelude::SleepingDisabled,
                    // Telemetry: the PD controller writes the last
                    // commanded torque here every frame; the dataset
                    // exporter reads it to log torque_norm per organism.
                    crate::avian_setup::LastAppliedTorque::default(),
                    // Filter out self-collision: a creature's own body parts
                    // must not collide with each other. The joint chain holds
                    // them together; self-contacts fight that (contact pushes
                    // apart, joint pulls together) and the solver pumps that
                    // conflict into explosive linear+angular velocity on the
                    // light bodies. `SelfCollisionFilter` rejects same-organism
                    // pairs; this marker activates the filter hook for them.
                    avian3d::prelude::ActiveCollisionHooks::FILTER_PAIRS,
                    // Velocity governor: pursuit locomotion is steady (~2.5 u/s)
                    // and the gait is gentle, both well under these caps — so
                    // the caps only clip the rare joint-instability spike,
                    // bounding limb separation and preventing any fly-off,
                    // without throttling normal motion.
                    avian3d::prelude::MaxLinearSpeed(crate::simulation_settings::MAX_LIMB_LINEAR_SPEED),
                    avian3d::prelude::MaxAngularSpeed(crate::simulation_settings::MAX_LIMB_ANGULAR_SPEED),
                ));
            }
        }
        // Joints — one spherical joint per appendage. Skips the base
        // body, which has no attachment. See `limb_joint_anchors` for
        // pivot-placement details (joint sits on the cell-to-cell
        // face) and `LIMB_JOINT_COMPLIANCE` for the small non-zero
        // compliance that keeps the XPBD solver well-conditioned.
        for (idx, bp) in body_parts.iter().enumerate() {
            if !bp.is_alive() { continue; }
            let Some(attach) = bp.attachment.as_ref() else { continue; };
            // Defensive bounds/aliveness guard — see the parallel
            // `spawn_organism` joint loop. A loaded colony that has been
            // through sim time (predation, growth, reproduction) can
            // carry a stale `parent_idx`; without this guard the index
            // below panics, which is the load-time crash this fixes.
            let pidx = attach.parent_idx;
            if pidx >= body_parts.len() || !body_parts[pidx].is_alive() {
                warn!("limb joint skipped on load: parent_idx {pidx} out of range or consumed");
                continue;
            }
            let parent_e = bp_entities[pidx];
            let child_e  = bp_entities[idx];
            let parent_origin = body_parts[pidx].attachment.as_ref()
                .map_or(Vec3::ZERO, |a| a.origin_local);
            let (anchor1, anchor2, hinge_axis) =
                limb_joint_anchors(parent_origin, &body_parts[idx].cells, attach.origin_local);
            commands.spawn((
                // 1-DOF hinge: the limb swings in a single (vertical) plane and
                // physically CANNOT orbit the body — the two non-hinge axes are
                // held by a robust bilateral align constraint, unlike a ball
                // joint's fragile, singular swing/twist limits that the limb
                // kept spinning through. `align_compliance` leaves a small give
                // so the limb can twist a little at the attachment (like a real
                // shoulder/hip). Hinge axis ⟂ limb (vertical swing plane); see
                // `limb_joint_anchors`.
                avian3d::prelude::RevoluteJoint::new(parent_e, child_e)
                    .with_local_anchor1(anchor1)
                    .with_local_anchor2(anchor2)
                    .with_hinge_axis(hinge_axis)
                    .with_point_compliance(crate::avian_setup::LIMB_JOINT_COMPLIANCE)
                    .with_align_compliance(crate::simulation_settings::LIMB_HINGE_ALIGN_COMPLIANCE)
                    .with_angle_limits(
                        -crate::simulation_settings::LIMB_SWING_LIMIT,
                        crate::simulation_settings::LIMB_SWING_LIMIT,
                    )
                    // Built-in spring-damper angular motor (solved in-step,
                    // 1-DOF on the hinge axis). The brain sets its
                    // `target_position` each tick via `drive_limb_motors`;
                    // this is what replaced the unstable external PD torque.
                    .with_motor(
                        avian3d::prelude::AngularMotor::new(
                            avian3d::prelude::MotorModel::SpringDamper {
                                frequency:     crate::simulation_settings::LIMB_MOTOR_FREQUENCY,
                                damping_ratio: crate::simulation_settings::LIMB_MOTOR_DAMPING_RATIO,
                            },
                        )
                        .with_max_torque(crate::simulation_settings::MAX_LIMB_TORQUE),
                    ),
                // See spawn_organism: suppress the persistent self-contact
                // between a limb and its joint-adjacent parent. Non-adjacent
                // intra-body collisions are preserved.
                avian3d::prelude::JointCollisionDisabled,
                // Link the joint back to its organism + limb so the motor
                // driver can set this hinge's target angle each tick.
                crate::avian_setup::LimbJointDrive {
                    organism:    root,
                    body_part:   idx,
                    limb_entity: child_e,
                },
            ));
        }
    }

    root
}


// ── Auto-spawn heterotrophs ─────────────────────────────────────────────────
//
// Tops the heterotroph population up to `MinHeteroCount` whenever a
// heterotroph death event fires AND `AutoSpawnHeteros(true)`.
//
// Zero-overhead in the steady state: `RemovedComponents<Heterotroph>`
// is empty between actual death events, so the early-return after the
// flag check costs nothing. When the flag is off we drain the reader
// once to avoid event backlog growth.
//
// Newly spawned organisms have no `BrainInheritance` component, so the
// brain pool's slot-assign path samples fresh random hyperparameter
// genes (see `intelligence_level_1_hetero::assign_brains_l1_hetero`)
// rather than inheriting from any parent.
pub fn auto_spawn_heteros(
    auto:           Res<crate::simulation_settings::AutoSpawnHeteros>,
    min_count:      Res<crate::simulation_settings::MinHeteroCount>,
    mut removed:    RemovedComponents<Heterotroph>,
    heteros:        Query<(), With<Heterotroph>>,
    org_mats:       Option<Res<OrganismMaterials>>,
    heightmap:      Option<Res<HeightmapSampler>>,
    map_size:       Option<Res<MapSize>>,
    smoothing:      Option<Res<crate::simulation_settings::Smoothing>>,
    mut commands:   Commands,
    mut meshes:     ResMut<Assets<Mesh>>,
) {
    if !auto.0 {
        for _ in removed.read() {}
        return;
    }

    let dead = removed.read().count();
    if dead == 0 { return; }

    let (Some(org_mats), Some(heightmap), Some(map_size), Some(smoothing)) =
        (org_mats, heightmap, map_size, smoothing) else { return };

    let current = heteros.iter().count();
    if current >= min_count.0 { return; }
    let to_spawn = min_count.0 - current;

    let mut rng = rand::rng();
    for _ in 0..to_spawn {
        // Spawn strictly inside the WORLD_SAFETY_MARGIN inset so the
        // organism is born inside the same XZ band that
        // `apply_world_bounds` keeps it within at runtime.
        let x = rng.random_range(
            crate::world_geometry::WORLD_SAFETY_MARGIN
                ..(map_size.x - crate::world_geometry::WORLD_SAFETY_MARGIN),
        );
        let z = rng.random_range(
            crate::world_geometry::WORLD_SAFETY_MARGIN
                ..(map_size.z - crate::world_geometry::WORLD_SAFETY_MARGIN),
        );
        let y = heightmap.height_at(x, z) + 1.0;

        let right_seed = vec![(
            0usize,
            Vec3::new(crate::body_part::MIN_X_BILATERAL, 0.0, 0.0),
            CellType::NonPhoto,
        )];
        let body_parts = vec![
            crate::body_part::bilateral_body_part_from_right_ocg(&right_seed)
        ];
        let max_e = 2.0 * crate::energy::MAX_ENERGY_PER_CELL;
        let intel = IntelligenceLevel::for_initial_spawn(
            OrganismKind::Heterotroph,
            false,
            &mut rng,
        );
        spawn_organism(
            Vec3::new(x, y, z),
            body_parts,
            OrganismKind::Heterotroph,
            Symmetry::Bilateral,
            false, // has_variable_form
            false, // is_sessile
            intel,
            smoothing.0,
            max_e * 0.5,
            true,  // sliding_movement — auto-spawned heteros use legacy sliding
            &mut commands,
            &mut meshes,
            &org_mats,
            &mut rng,
        );
    }
}
