use crate::cell::*;
use crate::frontend::ShowGizmo;
use crate::volumetric_growth::{build_mesh_from_ocg, build_smoothed_mesh_from_ocg};
use crate::world_geometry::{HeightmapSampler, MapSize};
use crate::movement::DirectionTimer;
use bevy::prelude::*;
use rand::prelude::*;

// Re-exports so `use crate::colony::*;` resolves Organism etc. and
// save/load items (`crate::colony::SaveRequested`).
pub use crate::organism::*;
pub use crate::colony_save_load::*;

pub use crate::simulation_settings::INITIAL_KRISHI;

use crate::simulation_settings::{
    SPAWN_SWIMMER_PATH, SPAWN_SWIMMER_COUNT,
    SPAWN_STRIDER_PATH, SPAWN_STRIDER_COUNT,
    SPAWN_ALGAE_PATH, SPAWN_ALGAE_COUNT,
    SWIM_SPAWN_CLEARANCE,
};

use crate::simulation_settings::BASE_ANGULAR_DAMPING;

use crate::simulation_settings::LIMB_ANGULAR_DAMPING;

use crate::simulation_settings::LIMB_LINEAR_DAMPING;

use crate::simulation_settings::BODY_PART_DENSITY;



/// Single convex-hull collider for a limb body part from its cell cloud.
/// One hull per part (vs. one sphere per cell) cuts narrow-phase pair
/// count — the dominant lever for limb-organism physics performance.
/// Each cell contributes six axis-extreme points (`center ± r`): keeps
/// the hull at true cell extent (no terrain sinking) and guarantees a
/// non-degenerate 3D point set even for a 1-cell part, so the sphere
/// fallback is purely defensive.
fn limb_part_collider(cells: &[Cell]) -> Option<bevy_rapier3d::prelude::Collider> {
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
        bevy_rapier3d::prelude::Collider::convex_hull(&pts)
            .unwrap_or_else(|| bevy_rapier3d::prelude::Collider::ball(r)),
    )
}

/// The limb's long axis in its LOCAL frame: direction from the part origin to
/// the centroid of its cells, normalised. Used as the brain-driven TWIST (roll)
/// axis (`rapier_setup::drive_limb_twist`). Falls back to +Y for a degenerate
/// (single cell at origin) limb.
fn limb_long_axis_local(cells: &[Cell]) -> Vec3 {
    if cells.is_empty() { return Vec3::Y; }
    let mut mean = Vec3::ZERO;
    for c in cells { mean += c.local_pos; }
    mean /= cells.len() as f32;
    mean.try_normalize().unwrap_or(Vec3::Y)
}


/// Compute the (anchor1, anchor2, hinge_axis) for a limb joint. The
/// limb is rebased so its first cell sits at its local origin, so the
/// rotation point IS that first cell: anchor2 = ZERO (limb-local),
/// anchor1 = pivot − parent_origin (same world point in parent frame).
/// Pinning at the first cell means the limb can only rotate, never
/// translate away. Defensive ZERO fallback when a body has no cells.
fn limb_joint_anchors(
    parent_origin: Vec3,   // parent body-part entity origin, in the root frame
    limb_cells:    &[Cell],
    pivot:         Vec3,   // first-cell position in the root frame (= attach.origin_local)
) -> (Vec3, Vec3, Vec3) {
    // Body parts are a flat hierarchy under the root, so `parent_origin`
    // is the parent part's own `attachment.origin_local`.
    let anchor1 = pivot - parent_origin;
    let anchor2 = Vec3::ZERO;

    // Hinge axis ⟂ limb out-direction and world-up, so the primary swing
    // is in a vertical plane (lift/push for locomotion). `out_dir` = the
    // limb's rest long axis (centroid of its rebased cells).
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


/// Insert the SLIDING physics representation onto the organism root: a
/// kinematic body + one compound collider over every alive cell (offset by its
/// part's attachment origin to match world positions). `apply_movement` writes
/// the root `Transform`; Rapier's kinematic sync follows it.
fn insert_sliding_collider(commands: &mut Commands, root: Entity, parts: &[BodyPart]) {
    use bevy_rapier3d::prelude::*;
    let mut shapes: Vec<(Vec3, Quat, Collider)> = Vec::new();
    for bp in parts {
        if !bp.is_alive() { continue; }
        let part_offset = bp.attachment.as_ref().map_or(Vec3::ZERO, |a| a.origin_local);
        for cell in &bp.cells {
            shapes.push((
                part_offset + cell.local_pos,
                Quat::IDENTITY,
                Collider::ball(crate::cell::CELL_COLLISION_RADIUS),
            ));
        }
    }
    if !shapes.is_empty() {
        commands.entity(root).insert((
            RigidBody::KinematicPositionBased,
            Collider::compound(shapes),
            // DIAG/fix: ensure a dynamic limb herbivore that touches this
            // (kinematic) prey generates a collision event the predation system
            // can act on. ActiveCollisionTypes::all() enables KINEMATIC-DYNAMIC
            // event generation regardless of which body carries the flag.
            ActiveEvents::COLLISION_EVENTS,
            ActiveCollisionTypes::all(),
        ));
    }
}

/// Insert the LIMB physics representation: per-body-part `RigidBody::Dynamic`
/// connected to their attachment parent by a reduced-coordinate `MultibodyJoint`
/// (revolute hinge + position motor). The multibody formulation makes joint
/// separation impossible by construction (the reason for the Rapier migration).
/// Asymmetric friction (grippy feet, slippery belly) makes leg strokes propel
/// the body; the per-organism contact-filter hook rejects self-collisions.
fn insert_limb_physics(
    commands:    &mut Commands,
    root:        Entity,
    bp_entities: &[Entity],
    parts:       &[BodyPart],
    is_swimmer:  bool,
) {
    use bevy_rapier3d::prelude::*;
    use crate::simulation_settings::{SWIM_LINEAR_DAMPING, SWIM_ANGULAR_DAMPING};
    // Swimmer roots carry the ball-joint target buffer the swimming brain
    // writes (`drive_swim_motors` reads it each step).
    if is_swimmer {
        commands.entity(root).insert(crate::rapier_setup::SwimJointTargets::default());
    }
    for (idx, bp) in parts.iter().enumerate() {
        if !bp.is_alive() { continue; }
        let Some(collider) = limb_part_collider(&bp.cells) else { continue; };
        let ang_damping = if idx == 0 { BASE_ANGULAR_DAMPING } else { LIMB_ANGULAR_DAMPING };
        let (fric_coeff, fric_rule) = if idx == 0 {
            (crate::simulation_settings::BASE_FRICTION_COEFFICIENT, CoefficientCombineRule::Min)
        } else {
            (crate::simulation_settings::LIMB_FRICTION_COEFFICIENT, CoefficientCombineRule::Max)
        };
        // SWIMMER specialisation (gated so walkers stay byte-identical): neutral
        // buoyancy (GravityScale 0) + small isotropic damping (the explicit
        // blade-element drag in `rapier_setup::apply_fluid_drag` dominates).
        let (gravity_scale, damping) = if is_swimmer {
            (
                GravityScale(0.0),
                Damping { linear_damping: SWIM_LINEAR_DAMPING, angular_damping: SWIM_ANGULAR_DAMPING },
            )
        } else {
            (
                // Per-body gravity scale — the STANDING curriculum ramps this from a
                // low value to 1.0 (`rapier_setup::fade_standing_gravity_assist`).
                GravityScale(1.0),
                // Base: heavy angular damping to absorb joint-reaction torques;
                // limbs: lighter for dynamic swings.
                Damping { linear_damping: LIMB_LINEAR_DAMPING, angular_damping: ang_damping },
            )
        };
        // Walkers use a near-massless density so weak motors beat ground friction;
        // swimmers need realistic (neutral-buoyancy) mass or the quadratic fluid
        // drag divides by ~nothing and the integrator explodes to NaN.
        let density = if is_swimmer {
            crate::simulation_settings::SWIM_BODY_DENSITY
        } else {
            BODY_PART_DENSITY
        };
        commands.entity(bp_entities[idx]).insert((
            RigidBody::Dynamic,
            collider.clone(),
            ColliderMassProperties::Density(density),
            Friction { coefficient: fric_coeff, combine_rule: fric_rule },
            // Generate collision events (→ predation) + activate the
            // same-organism contact-filter hook.
            ActiveEvents::COLLISION_EVENTS,
            ActiveCollisionTypes::all(), // incl. KINEMATIC-DYNAMIC so limb↔phototroph (kinematic) contacts register
            ActiveHooks::FILTER_CONTACT_PAIRS,
            crate::rapier_setup::LimbContact::default(),
            damping,
            // Readable/writable velocity + persistent twist torque accumulator.
            Velocity::zero(),
            ExternalForce::default(),
            // Never sleep: motor torque writes must always integrate.
            Sleeping::disabled(),
            gravity_scale,
            crate::rapier_setup::LastAppliedTorque::default(),
        ));
        // Swimmer parts (incl. the base) carry the marker + their anisotropic
        // drag footprint so the swimmer-only systems apply blade-element drag,
        // water-plane confinement, and neutral-buoyancy curriculum exclusion.
        if is_swimmer {
            use bevy_rapier3d::prelude::{AdditionalMassProperties, MassProperties};
            commands.entity(bp_entities[idx]).insert((
                crate::rapier_setup::SwimmerBody,
                crate::rapier_setup::drag_shape_from_cells(&bp.cells),
                // Isotropic inertia floor: conditions the Featherstone solve so a
                // thin (e.g. 2-cell mirrored) link's tiny long-axis inertia can't
                // make the 3-axis spherical multibody motor integrate to a NaN.
                AdditionalMassProperties::MassProperties(MassProperties {
                    local_center_of_mass: Vec3::ZERO,
                    mass: 0.0,
                    principal_inertia: Vec3::splat(crate::simulation_settings::SWIM_LINK_INERTIA_FLOOR),
                    principal_inertia_local_frame: Quat::IDENTITY,
                }),
            ));
        }
        // STANDING fall-reset: remember this part's pristine spawn-local pose (the
        // intended standing configuration) so `reset_fallen_standers` can teleport
        // a collapsed organism back to a fresh standing attempt. Computed exactly
        // as the spawn child Transform (attachment origin/rotation, else identity).
        let rest = match bp.attachment.as_ref() {
            Some(a) => Transform { translation: a.origin_local, rotation: a.rotation, ..Default::default() },
            None    => Transform::from_translation(Vec3::ZERO),
        };
        commands.entity(bp_entities[idx]).insert(crate::rapier_setup::LimbRestPose(rest));
        // Static parts are rigidly fixed (no motor, no brain twist couple).
        if idx > 0 && !matches!(bp.kind, BodyPartKind::Static) {
            if let Some(parent_e) = bp.attachment.as_ref().map(|a| a.parent_idx)
                .filter(|&p| p < bp_entities.len()).map(|p| bp_entities[p])
            {
                commands.entity(bp_entities[idx]).insert(crate::rapier_setup::LimbTwistDrive {
                    organism:   root,
                    body_part:  idx,
                    axis_local: limb_long_axis_local(&bp.cells),
                    parent:     parent_e,
                });
            }
        }
    }
    // One joint per appendage, inserted as a component on the CHILD limb entity
    // (referencing the parent body): a revolute hinge for walkers/standers, a
    // BALL (spherical) joint for swimmers — both pivoting at the limb's first
    // placed cell (`anchor2 = ZERO` on the rebased limb), so the joint stays at
    // its original position relative to the parent body part.
    for (idx, bp) in parts.iter().enumerate() {
        if !bp.is_alive() { continue; }
        let Some(attach) = bp.attachment.as_ref() else { continue; };
        let pidx = attach.parent_idx;
        if pidx >= parts.len() || !parts[pidx].is_alive() {
            warn!("limb joint skipped: parent_idx {pidx} out of range or consumed");
            continue;
        }
        let parent_e = bp_entities[pidx];
        let child_e  = bp_entities[idx];
        let parent_origin = parts[pidx].attachment.as_ref().map_or(Vec3::ZERO, |a| a.origin_local);
        let (anchor1, anchor2, hinge_axis) =
            limb_joint_anchors(parent_origin, &parts[idx].cells, attach.origin_local);
        // STATIC: a rigid fixed joint — no motor, no `LimbJointDrive`, so neither
        // `drive_limb_motors`/`drive_swim_motors` nor the brain can move it. It
        // rides the body as one rigid unit. Deferred-attached for the same reason
        // as limb joints (both bodies must be Rapier-registered first).
        if matches!(bp.kind, BodyPartKind::Static) {
            commands.entity(child_e).insert(crate::rapier_setup::PendingFixedJoint {
                parent: parent_e,
                anchor1,
                anchor2,
            });
            continue;
        }
        let drive = crate::rapier_setup::LimbJointDrive {
            organism:    root,
            body_part:   idx,
            limb_entity: child_e,
            parent:      parent_e,
            anchor1,
            anchor2,
            axis:        hinge_axis,
        };
        // Deferred: the real MultibodyJoint is attached by
        // `rapier_setup::attach_pending_limb_joints` once BOTH bodies are
        // registered in Rapier — adding it the same frame the bodies spawn
        // panics the solver (unknown parent body handle).
        commands.entity(child_e).insert((
            crate::rapier_setup::PendingLimbJoint(drive),
            drive,
        ));
    }
}


/// Keep the (sparse, FPS-capped, non-respawning) prey field PERCEIVABLE: relocate
/// any phototroph farther than `MAINTAIN_RANGE` from every limb herbivore to near a
/// random herbivore. Without this, herbivores disperse into permanently prey-empty
/// space (prey don't respawn — only heteros auto-spawn — and the cull only trims to
/// cap), so the steering assist loses its target (`prey_found` froze in the data).
/// Relocation (not respawn) keeps the prey COUNT fixed → FPS-neutral. XZ only (Y kept
/// → no floor re-clamp needed on the flat training world). Throttled per tick.
/// Locomotion-only (skipped during the standing task).
pub fn maintain_prey_near_herbivores(
    sim_running: Res<crate::simulation_settings::SimulationRunning>,
    heightmap:   Option<Res<HeightmapSampler>>,
    bases:       Query<(&bevy::prelude::ChildOf, &crate::cell::BodyPartIndex, &GlobalTransform)>,
    orgs:        Query<&Organism>,
    heteros:     Query<(), With<Heterotroph>>,
    mut photos:  Query<(&mut Transform, &Organism), (With<Photoautotroph>, With<OrganismRoot>)>,
) {
    if crate::simulation_settings::STANDING_TASK || !sim_running.0 { return; }
    let Some(heightmap) = heightmap else { return };
    const MAINTAIN_RANGE: f32 = 220.0; // XZ dist from all herbivores beyond which a prey is relocated
    const MOVE_PER_TICK:  usize = 6;   // throttle relocations to avoid transform spikes
    const PREY_Y_OFFSET:  f32 = 0.5;   // sit just on the terrain
    use rand::Rng;

    // Limb-herbivore base (body-part 0) world positions = where prey must stay near.
    let mut hp: Vec<Vec2> = Vec::new();
    for (co, idx, gt) in &bases {
        if idx.0 != 0 { continue; }
        let root = co.parent();
        if !heteros.contains(root) { continue; }
        if orgs.get(root).map(|o| o.movement_mode.is_sliding()).unwrap_or(true) { continue; }
        let t = gt.translation();
        hp.push(Vec2::new(t.x, t.z));
    }
    if hp.is_empty() { return; }

    let mut rng = rand::rng();
    let r2 = MAINTAIN_RANGE * MAINTAIN_RANGE;
    let mut moved = 0;
    for (mut tf, org) in &mut photos {
        // Relocate prey that drifted (or loaded) far from every herbivore to near a
        // random one, throttled. Prey don't respawn + the loaded field is sparse, so
        // without this herbivores disperse into permanently prey-empty space.
        let pxz = Vec2::new(tf.translation.x, tf.translation.z);
        let mut best = f32::MAX;
        for h in &hp { best = best.min(pxz.distance_squared(*h)); }
        if best > r2 && moved < MOVE_PER_TICK {
            let h = hp[rng.random_range(0..hp.len())];
            let ang = rng.random_range(0.0..std::f32::consts::TAU);
            // Spread prey out (within the 250u sensing range but NOT piled on the
            // herbivore): a dense local prey cluster created a huge herbivore↔prey
            // contact count that craters FPS when seekers converge.
            let r = rng.random_range(120.0..210.0);
            tf.translation.x = h.x + r * ang.cos();
            tf.translation.z = h.y + r * ang.sin();
            moved += 1;
        }
        // CLAMP GROUND-BASED prey to the terrain surface. The loaded phototrophs sit
        // ABOVE the (superflat) terrain and gravity-fall (apply_movement) to y<-500,
        // where the kill-floor despawns them — the whole prey field vanished in ~30 s.
        // Pinning their Y to the heightmap keeps them a stable, reachable food source.
        // WATER-BASED prey (floating algae) are EXCLUDED: water-gated gravity already
        // holds them at their depth (neither sinking nor surfacing), and clamping them
        // to the terrain is exactly the bug that made them sink to the bottom.
        if org.ground_based {
            let surf = heightmap.height_at(tf.translation.x, tf.translation.z);
            tf.translation.y = surf + PREY_Y_OFFSET;
        }
    }
}


pub struct ColonyPlugin;

impl Plugin for ColonyPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SaveRequested>();
        // `init_resource` (not `insert_resource`) preserves the load path
        // main.rs sets before add_plugins; this is the no-load fallback.
        app.init_resource::<ColonyLoadPath>();
        app.init_resource::<crate::simulation_settings::AutoSpawnHeteros>();
        // Default false; phase 5 inserts the real value from argv. `init_resource`
        // (not insert) preserves any argv-set value that ran before the plugin.
        app.init_resource::<crate::simulation_settings::AdjustColonyDimensions>();
        // WaterLevel may be inserted from argv before this plugin; init (not
        // insert) keeps that value while guaranteeing presence for `spawn_colony`.
        app.init_resource::<crate::environment::WaterLevel>();
        app.init_resource::<crate::simulation_settings::StartHeterotrophs>();
        app.init_resource::<crate::simulation_settings::StartPhotoautotrophs>();
        app.init_resource::<crate::simulation_settings::MinHeteroCount>();
        app.init_resource::<crate::simulation_settings::MinHeteroCountEditState>();
        app.init_resource::<AutosaveTimer>();
        app.init_resource::<crate::simulation_settings::RunElapsed>();
        app.init_resource::<crate::dataset_export::ExportDatasetRequested>();
        app.init_resource::<crate::dataset_export::AutoExportSchedule>();
        // Rotate prior-run dataset artefacts once at process start,
        // before any auto-export fires.
        app.add_systems(Startup, |_w: &mut World| {
            crate::dataset_export::rotate_existing_datasets();
        });
        app.init_resource::<crate::time_series_log::TimeSeriesLogger>();
        app.init_resource::<crate::limb_time_series_log::LimbTimeSeriesLogger>();
        app.init_resource::<crate::limb_force_probe::LimbForceProbe>();
        app.add_systems(Update, spawn_colony.run_if(resource_exists::<HeightmapSampler>));
        // Swimmer spawn-Y safety net: PreUpdate, so a frame-N spawn is
        // re-seated out of the floor cuboid BEFORE its first physics step.
        app.add_systems(PreUpdate, reseat_new_swimmers);
        app.add_systems(Update, cull_excess_limb_organisms_for_standing);
        // PostUpdate so the prey Y-clamp is the LAST write each frame (apply_movement
        // gravity-falls sliding prey in Update; clamping after pins them to terrain).
        app.add_systems(PostUpdate, maintain_prey_near_herbivores);
        app.add_systems(Update, animate_limbs);
        app.add_systems(Update, tick_run_elapsed);
        app.add_systems(Update, save_colony_system);
        app.add_systems(Update, autosave_system);
        app.add_systems(Update, auto_spawn_heteros);
        // Always-present prey floor: top plankton up to MIN_PLANKTON_COUNT,
        // throttled to ~1 Hz (count-and-spawn isn't needed per frame).
        app.add_systems(Update, auto_spawn_plankton
            .run_if(bevy::time::common_conditions::on_timer(std::time::Duration::from_secs(1))));
        app.add_systems(Update, crate::dataset_export::tick_auto_export_schedule);
        app.add_systems(Update, crate::dataset_export::export_dataset_system);
        app.add_systems(Update, crate::time_series_log::tick_time_series_logger);
        app.add_systems(Update, crate::limb_time_series_log::tick_limb_time_series_logger);
        app.add_systems(Update, crate::limb_force_probe::tick_limb_force_probe);
        // Default lineage record, one per organism on the tick after spawn.
        // Reproduction offspring already carry one (with `parent_id`), so
        // the `Without<LineageRecord>` filter skips them.
        app.add_systems(Update, assign_lineage_records);
    }
}


/// Accumulate WALL-clock seconds into `RunElapsed` from the uncapped real-time
/// clock while the sim is running. This is the elapsed-time source for exports
/// + save headers: unlike `Time<Virtual>` (whose `max_delta` cap makes it
/// under-count wall time across slow frames), it counts a multi-second freeze
/// in full, and unlike a raw `Time<Real>` read it pauses with the sim and is
/// seeded on load so a resumed colony continues its age.
fn tick_run_elapsed(
    real_time:   Res<Time<bevy::time::Real>>,
    running:     Res<crate::simulation_settings::SimulationRunning>,
    mut elapsed: ResMut<crate::simulation_settings::RunElapsed>,
) {
    if running.0 {
        elapsed.0 += real_time.delta().as_secs_f64();
    }
}


/// Mint a `new_initial` LineageRecord for any OrganismRoot lacking one
/// (initial cohort, auto-spawn, editor, loaded). Reproduction offspring
/// already carry a `new_offspring` record.
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


/// Optional `.colony` file path. When `Some`, the first `spawn_colony`
/// loads it instead of generating a fresh colony. Set by `main.rs` from
/// the second positional CLI argument; defaults to `None`.
#[derive(Resource, Default)]
pub struct ColonyLoadPath(pub Option<String>);




/// Materialise a loaded organism record. Mirrors `spawn_organism`'s
/// hierarchy build but uses the saved Transform and Organism AS-IS — no
/// fresh movement direction, no `recompute_body_parts` (saved cells
/// already carry neighbour counts; photo cache rebuilt during decode).
fn spawn_loaded_organism(
    record:    LoadedRecord,
    smoothing: bool,
    commands:  &mut Commands,
    meshes:    &mut ResMut<Assets<Mesh>>,
    materials: &OrganismMaterials,
    rng:       &mut impl rand::Rng,
) -> Entity {
    let LoadedRecord { pos, rotation, kind, organism, brain, brain_limb, species_name } = record;
    if organism.body_parts.is_empty() {
        // Defensive — should never happen on a valid save.
        return commands.spawn_empty().id();
    }

    let body_parts_snapshot = organism.body_parts.clone();
    // Capture before `organism` is moved into the spawn bundle.
    let intelligence_level = organism.intelligence_level;
    let adult              = organism.adult;
    let movement_mode      = organism.movement_mode;
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
    // Attach saved brain weights so the matching pool's `assign_brains_*`
    // restores them. Skipped for Level 0 (no pool).
    if let Some(b) = brain {
        if !matches!(intelligence_level, IntelligenceLevel::Level0) {
            root_cmd.insert(b);
        }
    }
    // Limb-brain payload: each limb pool's assign filter
    // (`intelligence_level + !movement_mode.is_sliding()`) picks exactly one consumer.
    if let Some(b) = brain_limb {
        root_cmd.insert(b);
    }
    let root = root_cmd.id();

    // v011: pin species identity at spawn from the saved name. Resolve (or
    // create) the named species in the registry and set `species_id` NOW, so the
    // per-species brain restore in the pools' assign step lands in the right
    // shared net instead of the transient UNCLASSIFIED one (which the ~1 Hz
    // speciation reclassification would otherwise strand). Same-name organisms
    // collapse to one id, so a reloaded colony keeps its species and the trained
    // shared policy. Older saves carry no name → classified fresh, as before.
    if let Some(name) = species_name {
        commands.queue(move |world: &mut World| {
            let id = {
                let Some(mut registry) =
                    world.get_resource_mut::<crate::lineages::species::SpeciesRegistry>()
                else { return };
                let existing = registry.find_alive_by_name(&name).map(|s| s.id);
                existing.unwrap_or_else(|| registry.create_with_name(name.clone(), Vec::new(), None))
            };
            if let Ok(mut e) = world.get_entity_mut(root) {
                if let Some(mut org) = e.get_mut::<Organism>() {
                    if org.species_id.is_none() {
                        org.species_id = Some(id);
                    }
                }
            }
        });
    }

    let mut bp_entities: Vec<Entity> = Vec::with_capacity(body_parts_snapshot.len());
    for (idx, bp) in body_parts_snapshot.iter().enumerate() {
        // Adult + smoothing on → smoothed mesh, else faceted.
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
                // No shadow cast: the shadow pass would re-extract every
                // body-part entity per cascade per frame, dominating render
                // cost; organism shadows are negligible at cell scale.
                bevy::light::NotShadowCaster,
            ));
        }
        let child = child_cmd.id();

        // Flat hierarchy — see `spawn_organism` for the phantom-organism
        // despawn-cascade rationale.
        let _ = &bp.attachment;
        commands.entity(root).add_child(child);

        bp_entities.push(child);
    }

    // Avian3d physics + LimbAnimation (same path as `spawn_organism`).
    // Loaded organisms need this or they have no rigid bodies / joints
    // and freeze at their saved pose; sliding organisms need the
    // kinematic compound collider so other bodies can collide with them.
    if movement_mode.is_sliding() {
        insert_sliding_collider(commands, root, &body_parts_snapshot);
        // LimbAnimation on Limb body parts; mirrors spawn_organism.
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
        insert_limb_physics(commands, root, &bp_entities, &body_parts_snapshot, movement_mode.is_swimming());
    }

    root
}




// ── Spawning ─────────────────────────────────────────────────────────────────

fn spawn_colony(
    mut commands:    Commands,
    mut meshes:      ResMut<Assets<Mesh>>,
    mut materials:   ResMut<Assets<StandardMaterial>>,
    heightmap:       Res<HeightmapSampler>,
    load_path:       Res<ColonyLoadPath>,
    smoothing:       Res<crate::simulation_settings::Smoothing>,
    map_size:        Res<MapSize>,
    // Loaded .colony files carry a water level; apply it unless the user
    // asked to keep launcher dimensions (--adjust-colony-dimensions).
    mut water_level:        ResMut<crate::environment::WaterLevel>,
    adjust_dimensions:      Res<crate::simulation_settings::AdjustColonyDimensions>,
    // Launcher start-count / cap resources no longer drive the fresh
    // cohort (fixed `.species` counts); `--max-herbivores` still sizes
    // the GPU brain pools in `main.rs`.
    mut virtual_time:    ResMut<Time<Virtual>>,
    mut run_elapsed:     ResMut<crate::simulation_settings::RunElapsed>,
    mut spawned:     Local<bool>,
) {
    if *spawned { return; }
    *spawned = true;

    let materials = OrganismMaterials::new(&mut materials);
    // Mirror into a Resource so runtime spawners (`auto_spawn_heteros`)
    // reuse the same handles. Clone is cheap (each field is a Handle).
    commands.insert_resource(OrganismMaterials {
        photo:      materials.photo.clone(),
        hetero:     materials.hetero.clone(),
        debug_blue: materials.debug_blue.clone(),
    });
    let mut rng = rand::rng();

    // If a save file was supplied, try to restore it; on any failure
    // fall through to fresh generation so the run still produces output.
    if let Some(path) = &load_path.0 {
        match load_colony_from_file(path) {
            Ok((notation, loaded_water_level, records)) => {
                let n = records.len();
                // Apply the file's water level unless the user requested the
                // launcher dimensions be kept (--adjust-colony-dimensions).
                if !adjust_dimensions.0 {
                    water_level.0 = loaded_water_level;
                    info!("colony water level set to {} from save file", loaded_water_level);
                }
                for mut record in records {
                    // Swimmers were authored/saved at terrain level (editor placement
                    // bakes `terrain + 0.5`); re-seat them submerged relative to the
                    // now-applied water level so they don't spawn above the surface.
                    if record.organism.movement_mode.is_swimming() {
                        let terrain = heightmap.height_at(record.pos.x, record.pos.z);
                        record.pos.y = submerged_spawn_y(terrain, limb_floor_top(&heightmap), water_level.0);
                    }
                    spawn_loaded_organism(record, smoothing.0, &mut commands, &mut meshes, &materials, &mut rng);
                }
                // Resume the virtual clock at the saved point. First
                // `advance_by` bumps `elapsed`; the second (ZERO) resets
                // this frame's `delta` to 0 so no delta-integrating system
                // sees the jump as one giant timestep.
                let total = notation.total_secs();
                if total > 0 {
                    virtual_time.advance_by(std::time::Duration::from_secs(total as u64));
                    virtual_time.advance_by(std::time::Duration::ZERO);
                }
                // Seed the wall-clock run age so exports/saves continue from the
                // saved elapsed time (the header is written from `RunElapsed`).
                run_elapsed.0 = total as f64;
                info!("loaded colony from {} — {} organisms restored, virtual time resumed at {}h{}m{}s",
                      path, n, notation.hours, notation.minutes, notation.seconds);
                return;
            }
            Err(e) => {
                error!("failed to load colony from {}: {} — falling back to fresh generation", path, e);
            }
        }
    }

    // Fresh-start cohort: seed from authored `.species` files at fixed
    // counts (`SPAWN_*` constants).
    spawn_species_cohort(SPAWN_ALGAE_PATH,   SPAWN_ALGAE_COUNT,   &heightmap, &map_size, water_level.0, smoothing.0, &mut commands, &mut meshes, &materials, &mut rng);
    spawn_species_cohort(SPAWN_SWIMMER_PATH, SPAWN_SWIMMER_COUNT, &heightmap, &map_size, water_level.0, smoothing.0, &mut commands, &mut meshes, &materials, &mut rng);
    spawn_species_cohort(SPAWN_STRIDER_PATH, SPAWN_STRIDER_COUNT, &heightmap, &map_size, water_level.0, smoothing.0, &mut commands, &mut meshes, &materials, &mut rng);
}


/// Build an appendage `BodyPart` of the given `kind` from a full OCG.
/// `parent_idx` is the RUNTIME index of the part this one attaches to (0 = base;
/// another appendage index makes it a sub-part of that part).
///
/// REBASED to the first cell: the entity sits at the first-cell pivot and the
/// cells are shifted relative to it, so the joint pivots at the CONTACT POINT
/// and the body's centre of mass sits near its origin. (Appendages used to keep
/// `origin_local = ZERO` with cells at their authored positions, pinning each at
/// the ROOT ORIGIN with its COM up to ~9 units away — a huge lever arm that
/// turned fluid-drag/motor forces into violent spin, the "freshly-spawned
/// sub-limbed swimmer flung into the air" bug; rebasing collapses the lever to
/// part-scale.) `kind` decides the runtime wiring in `insert_limb_physics`:
/// `Limb`/`Organ`/`Segment` get a motorized joint driven by the brain; `Static`
/// gets a rigid fixed joint with no brain connection. Mirrors
/// `colony_editor::placement::{limb,appendage}_body_part`.
pub fn kinded_appendage_from_ocg(ocg: Vec<(usize, Vec3, CellType)>, parent_idx: usize, kind: BodyPartKind) -> BodyPart {
    let pivot = ocg.first().map(|(_, p, _)| *p).unwrap_or(Vec3::ZERO);
    let shifted: Vec<(usize, Vec3, CellType)> =
        ocg.iter().map(|(i, p, ct)| (*i, *p - pivot, *ct)).collect();
    let cells = shifted.iter().map(|(_, p, ct)| Cell::new(*p, *ct)).collect();
    BodyPart {
        kind,
        local_offset: Vec3::ZERO,
        cells,
        ocg:          shifted,
        attachment:   Some(crate::body_part::Attachment {
            parent_idx, origin_local: pivot, rotation: Quat::IDENTITY,
        }),
        consumed:   false,
        debug_blue: false,
        regrowable: true,
    }
}

/// Combine a Bilateral right-half OCG with its mirror into ONE midline part's
/// OCG (right cells, then their mirror, renumbered). Used for `Segment`/`Static`
/// appendages so they DON'T split into a pair. The right half's first cell stays
/// at index 0, so `kinded_appendage_from_ocg` still pivots about it. Midline
/// cells (`x ≈ 0`) are their own mirror and are not duplicated
/// (`mirror_right_to_left` skips them), matching the base-body fuse.
pub fn fuse_bilateral_ocg(right_ocg: &[(usize, Vec3, CellType)]) -> Vec<(usize, Vec3, CellType)> {
    let left = crate::body_part::mirror_right_to_left(right_ocg);
    right_ocg.iter().chain(left.iter()).enumerate()
        .map(|(i, (_, p, ct))| (i, *p, *ct)).collect()
}

/// Build a limb `BodyPart` from a full OCG: cells rebased so the first
/// cell sits at local origin and is the attachment pivot, so the limb
/// rotates around its base. Mirrors `colony_editor::placement::limb_body_part`.
/// One-shot (when `STANDING_TASK`): after the colony loads, despawn limb-based
/// organisms beyond `STANDING_MAX_LIMB_ORGS` so the heavy multi-part Runners
/// don't crater the frame rate. Runs once, the first frame limb organisms exist.
pub fn cull_excess_limb_organisms_for_standing(
    mut commands: Commands,
    mut done:     Local<bool>,
    q:            Query<(Entity, &Organism), With<OrganismRoot>>,
) {
    if *done { return; }
    let limb: Vec<Entity> = q.iter()
        .filter(|(_, o)| !o.movement_mode.is_sliding())
        .map(|(e, _)| e)
        .collect();
    if limb.is_empty() { return; } // organisms not loaded/spawned yet
    *done = true;
    // Each limb organism is many dynamic bodies + joints (CPU physics), so the full
    // loaded cohort craters the frame rate. A handful of independent PPO learners is
    // plenty. Tighter cap for the heavy multi-part Runners under the standing task.
    let keep = if crate::simulation_settings::STANDING_TASK {
        crate::simulation_settings::STANDING_MAX_LIMB_ORGS
    } else {
        crate::simulation_settings::LOCOMOTION_MAX_LIMB_ORGS
    };
    let mut culled = 0;
    for &e in limb.iter().skip(keep) {
        commands.entity(e).despawn();
        culled += 1;
    }
    info!("limb cull: kept {} limb organisms, despawned {} (FPS budget)",
          keep.min(limb.len()), culled);
}

/// Load a `.species` file and spawn `count` instances at random positions.
/// Body-part assembly mirrors `colony_editor::placement::spawn_real_organism`
/// so a species spawns identically in editor or here. Each spawn gets an
/// `ImportedSpeciesOrigin` (filename stem) so it founds its own lineage.
/// Errors are logged and the cohort skipped.
#[allow(clippy::too_many_arguments)]
/// Top Y of the limb-physics floor collider — a single flat `Fixed` cuboid whose
/// top face sits at the map-CENTRE terrain height (see `rapier_setup::spawn_terrain_floor`,
/// which assumes a flat world). Swimmers must spawn ABOVE this, not just above the
/// local heightmap height, or they spawn inside the cuboid and Rapier ejects them.
pub fn limb_floor_top(heightmap: &HeightmapSampler) -> f32 {
    use crate::world_geometry::HEIGHTMAP_CELL_SIZE;
    let cx = heightmap.width as f32 * HEIGHTMAP_CELL_SIZE * 0.5;
    let cz = heightmap.depth as f32 * HEIGHTMAP_CELL_SIZE * 0.5;
    heightmap.height_at(cx, cz)
}

/// Spawn-time Y for a SWIMMING organism: submerged in the water column —
/// `SWIM_SPAWN_CLEARANCE` below the surface and above the floor — so no body part
/// breaches the water plane (which would fire the ceiling restoring force) NOR
/// penetrates the floor cuboid (which would make Rapier eject the near-massless
/// body upward). The floor reference is `max(local_terrain, floor_top)`: the actual
/// physics floor is the flat cuboid at `floor_top`, but never go below the visible
/// local terrain either. Centred in the column when deep enough; if the column is
/// too shallow (water at/below the floor), falls back to just above the floor — the
/// now-bounded confinement force settles it rather than launching it.
pub fn submerged_spawn_y(local_terrain: f32, floor_top: f32, water_level: f32) -> f32 {
    let floor = local_terrain.max(floor_top);
    let min_y = floor + SWIM_SPAWN_CLEARANCE;
    let max_y = water_level - SWIM_SPAWN_CLEARANCE;
    if max_y > min_y { 0.5 * (min_y + max_y) } else { min_y }
}

/// SAFETY NET for every swimmer spawn path: re-seat a newly added SWIMMING
/// organism whose root Y is outside the safe water column
/// `[max(local_terrain, floor_top) + clearance, water − clearance]`.
///
/// The limb-physics floor is ONE FLAT cuboid whose top is the map-CENTRE
/// terrain height (`rapier_setup::spawn_terrain_floor`) — on a non-flat map,
/// any spawner using the walker convention (`local terrain + ~1`) places a
/// swimmer INSIDE that cuboid wherever the local terrain is below the centre
/// height, and Rapier's depenetration ejects the body violently upward (the
/// "flung away on spawn" bug). The cohort / load / reproduction paths call
/// `submerged_spawn_y` directly; this system catches the rest (editor/live
/// placement, undo-restore, future spawners) — and runs in `PreUpdate`, so a
/// frame-N spawn is re-seated BEFORE its first physics step in frame N+1's
/// FixedUpdate (the parts' `GlobalTransform`s follow the root via the
/// SyncBackend propagation, exactly like the initial spawn placement).
pub fn reseat_new_swimmers(
    heightmap: Option<Res<HeightmapSampler>>,
    water:     Option<Res<crate::environment::WaterLevel>>,
    mut roots: Query<(&Organism, &mut Transform), (Added<Organism>, With<OrganismRoot>)>,
) {
    let (Some(hm), Some(water)) = (heightmap, water) else { return };
    let floor_top = limb_floor_top(&hm);
    for (org, mut tf) in &mut roots {
        if !org.movement_mode.is_swimming() { continue; }
        let terrain = hm.height_at(tf.translation.x, tf.translation.z);
        let min_y = terrain.max(floor_top) + SWIM_SPAWN_CLEARANCE;
        let max_y = water.0 - SWIM_SPAWN_CLEARANCE;
        let y = tf.translation.y;
        // Inside the column (or exactly at the shallow-column fallback) → leave
        // the spawner's placement alone.
        let ok = if max_y > min_y {
            (min_y..=max_y).contains(&y)
        } else {
            (y - min_y).abs() < 0.01
        };
        if !ok {
            let safe = submerged_spawn_y(terrain, floor_top, water.0);
            info!(
                "re-seated swimmer spawn y {:.1} → {:.1} (floor_top {:.1}, terrain {:.1}, water {:.1})",
                y, safe, floor_top, terrain, water.0
            );
            tf.translation.y = safe;
        }
    }
}

/// Spawn ONE organism from an already-loaded `.species` at a random in-bounds
/// position. Owns the per-instance body-part assembly (the Bilateral right/left
/// + NoSymmetry parent mapping), the water-based/swimming-submerged vs
/// on-the-floor spawn height, and the Carnivore / `ImportedSpeciesOrigin` /
/// brain tagging. Shared by the startup `spawn_species_cohort` and the runtime
/// `auto_spawn_plankton`. Returns the spawned `OrganismRoot`.
#[allow(clippy::too_many_arguments)]
fn spawn_species_instance(
    species:     &crate::species_editor::save::LoadedSpecies,
    name:        &str,
    heightmap:   &HeightmapSampler,
    map_size:    &MapSize,
    water_level: f32,
    smoothing:   bool,
    commands:    &mut Commands,
    meshes:      &mut ResMut<Assets<Mesh>>,
    materials:   &OrganismMaterials,
    rng:         &mut impl rand::Rng,
) -> Option<Entity> {
    if species.body_parts.is_empty() { return None; }
    let kind = match species.metabolism {
        crate::species_editor::session::Metabolism::Photoautotroph => OrganismKind::Photoautotroph,
        crate::species_editor::session::Metabolism::Heterotroph    => OrganismKind::Heterotroph,
    };
    let is_carnivore = matches!(
        species.classification,
        crate::species_editor::session::Classification::Carnivore
    );
    let margin = crate::world_geometry::WORLD_SAFETY_MARGIN;

    // Rebuilt per spawn (spawn_organism consumes it): root from part 0, later
    // parts as appendages. Each part's RUNTIME `parent_idx` is mapped from the
    // species file's editor parent index, mirroring `placement::spawn_real_organism`.
    // NoSymmetry is 1:1. Bilateral: a `Limb` expands into a right+left PAIR tracked
    // separately (a sub-limb's right half attaches to its parent's right half);
    // `Segment`/`Static` FUSE their two halves into ONE midline part (no split),
    // with both `right_of`/`left_of` pointing at that single part so a child of a
    // segment attaches correctly regardless of which half it was authored on.
    // Base body: NoSymmetry uses the OCG verbatim; Bilateral FUSES the stored
    // right half with its mirror into one symmetric part (the `.species` stores
    // only the right half), matching the colony-editor's `expand()`.
    let base_part = match species.symmetry {
        Symmetry::Bilateral  => crate::body_part::bilateral_body_part_from_right_ocg(&species.body_parts[0].ocg),
        Symmetry::NoSymmetry => root_body_part_from_ocg(&species.body_parts[0].ocg),
    };
    let mut body_parts = vec![base_part];
    match species.symmetry {
        Symmetry::Bilateral => {
            let mut right_of: Vec<usize> = vec![0];
            let mut left_of:  Vec<usize> = vec![0];
            for lbp in &species.body_parts[1..] {
                let p_right = right_of.get(lbp.parent).copied().unwrap_or(0);
                let p_left  = left_of.get(lbp.parent).copied().unwrap_or(0);
                match lbp.kind {
                    BodyPartKind::Segment | BodyPartKind::Static => {
                        // Single fused midline part — never split.
                        let idx = body_parts.len();
                        body_parts.push(kinded_appendage_from_ocg(
                            fuse_bilateral_ocg(&lbp.ocg), p_right, lbp.kind));
                        right_of.push(idx);
                        left_of.push(idx);
                    }
                    kind => {
                        // Limb / Organ → mirrored right+left pair.
                        let r_idx = body_parts.len();
                        body_parts.push(kinded_appendage_from_ocg(lbp.ocg.clone(), p_right, kind));
                        let l_idx = body_parts.len();
                        body_parts.push(kinded_appendage_from_ocg(
                            crate::body_part::mirror_right_to_left(&lbp.ocg), p_left, kind));
                        right_of.push(r_idx);
                        left_of.push(l_idx);
                    }
                }
            }
        }
        Symmetry::NoSymmetry => {
            for (i, lbp) in species.body_parts[1..].iter().enumerate() {
                let parent = if lbp.parent <= i { lbp.parent } else { 0 };
                body_parts.push(kinded_appendage_from_ocg(lbp.ocg.clone(), parent, lbp.kind));
            }
        }
    }
    let cell_count = body_parts.iter().map(|bp| bp.cells.len()).sum::<usize>() as f32;
    let initial_energy = cell_count * crate::energy::MAX_ENERGY_PER_CELL * 0.5;

    let x = rng.random_range(margin..(map_size.x - margin));
    let z = rng.random_range(margin..(map_size.z - margin));
    let terrain = heightmap.height_at(x, z);
    // WATER-BASED organisms (swimmers + floating phototrophs like plankton)
    // start submerged in the water column; everyone else on the floor.
    let y = if species.movement.is_swimming() || !species.ground_based {
        submerged_spawn_y(terrain, limb_floor_top(heightmap), water_level)
    } else {
        terrain + 1.0
    };

    let entity = spawn_organism(
        Vec3::new(x, y, z), body_parts, kind, species.symmetry,
        species.has_variable_form, species.is_sessile, species.intelligence,
        smoothing, initial_energy, species.movement, species.ground_based,
        commands, meshes, materials, rng,
    );
    if is_carnivore {
        commands.entity(entity).try_insert(Carnivore);
    }
    commands.entity(entity).try_insert(
        crate::lineages::species::ImportedSpeciesOrigin { name: name.to_string() },
    );
    // PERSISTENT floor identity for the prey field. `ImportedSpeciesOrigin` is
    // stripped by speciation (~1 Hz) on first classification, so it cannot be
    // used to recognise the auto-spawned plankton floor past the first second —
    // doing so made `auto_spawn_plankton` (count) and `apply_max_phototrophs_cull`
    // (exclusion) both go blind, driving perpetual spawn/despawn/respawn churn.
    // This marker survives classification. Reproduced descendants don't spawn
    // through here, so they stay unmarked → cullable, as intended.
    if name == "ball_plankton" {
        commands.entity(entity).try_insert(BallPlankton);
    }
    if let Some(brain) = &species.brain {
        // Attach the matching restore COMPONENT. The PPO payload (`BrainRestoreLimb`)
        // serves both walkers and swimmers; the organism's movement mode routes it
        // to the limb or swim pool's assign step.
        match brain {
            crate::species_editor::save::LoadedBrain::Sliding(b) => {
                commands.entity(entity).try_insert(b.clone());
            }
            crate::species_editor::save::LoadedBrain::Ppo(b) => {
                commands.entity(entity).try_insert(b.clone());
            }
        }
        // This import carries a TRAINED brain. Pin its species identity NOW
        // (at spawn) instead of waiting for the ~1 Hz speciation tick: the
        // per-species brain pools restore saved weights keyed by `species_id`
        // in their assign step, so a still-`None` species_id would restore the
        // trained weights into the transient UNCLASSIFIED net — and they'd be
        // dropped when the clone is later reclassified into its real (fresh)
        // net. Resolving the named species here gives every same-name clone one
        // shared id, so the restore lands in the right shared net and all clones
        // boot from — and keep sharing — the trained policy. (Mirrors the
        // name-routing in `speciation::classify_organisms`; only brain-carrying
        // imports take this path, so plankton and brainless imports are
        // unaffected and still classify lazily.)
        let species_name = name.to_string();
        commands.queue(move |world: &mut World| {
            let id = {
                let Some(mut registry) =
                    world.get_resource_mut::<crate::lineages::species::SpeciesRegistry>()
                else { return };
                let existing = registry.find_alive_by_name(&species_name).map(|s| s.id);
                existing.unwrap_or_else(|| {
                    registry.create_with_name(species_name.clone(), Vec::new(), None)
                })
            };
            if let Ok(mut e) = world.get_entity_mut(entity) {
                if let Some(mut org) = e.get_mut::<Organism>() {
                    if org.species_id.is_none() {
                        org.species_id = Some(id);
                    }
                }
            }
        });
    }
    Some(entity)
}

fn spawn_species_cohort(
    path:        &str,
    count:       usize,
    heightmap:   &HeightmapSampler,
    map_size:    &MapSize,
    water_level: f32,
    smoothing:   bool,
    commands:    &mut Commands,
    meshes:      &mut ResMut<Assets<Mesh>>,
    materials:   &OrganismMaterials,
    rng:         &mut impl rand::Rng,
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

    for _ in 0..count {
        spawn_species_instance(
            &species, &name, heightmap, map_size, water_level, smoothing,
            commands, meshes, materials, rng,
        );
    }
    info!("spawn_colony: spawned {count} × {name} from {path}");
}


/// Persistent marker for an auto-spawned `ball_plankton` floor organism. Set at
/// spawn (`spawn_species_instance`) and NEVER removed, unlike the transient
/// `ImportedSpeciesOrigin` tag that speciation strips on first classification.
/// Both the floor top-up (`auto_spawn_plankton`) and the cap-cull exclusion
/// (`apply_max_phototrophs_cull`) key off this so the prey field is recognised
/// for the whole run. Reproduced descendants don't carry it (different spawn
/// path) → they count as bonus prey and stay cullable.
#[derive(Component)]
pub struct BallPlankton;


/// Maintain a minimum of `MIN_PLANKTON_COUNT` `ball_plankton`: whenever the
/// live count drops below the floor, spawn the deficit from the species file.
/// A reliable prey field (e.g. food for swimming heterotrophs).
///
/// Counts living organisms carrying the persistent `BallPlankton` marker (the
/// auto-spawned ones); reproduced descendants are unmarked and count as bonus
/// prey. The species file is loaded once (cached in a `Local`). Throttled to
/// ~1 Hz so the count-and-spawn isn't per-frame; always on (not gated by
/// AI-training mode). The `MaxPhotoautotrophs` cap-cull
/// (`apply_max_phototrophs_cull`) explicitly EXCLUDES `BallPlankton` from the
/// random cull, so the cap and this floor no longer compete — that competition
/// was the cause of constant plankton spawn/despawn/respawn churn (cull randomly
/// kills a plankton → drops below floor → respawn → over cap → cull again). With
/// a sane cap (≥ the floor) the total still trims to the cap by culling only
/// non-floor phototrophs; if the cap is set below the floor the floor wins.
/// Reproduced (unmarked) plankton descendants remain cullable like any other
/// phototroph.
pub fn auto_spawn_plankton(
    heightmap:  Option<Res<HeightmapSampler>>,
    map_size:   Option<Res<MapSize>>,
    water:      Option<Res<crate::environment::WaterLevel>>,
    smoothing:  Option<Res<crate::simulation_settings::Smoothing>>,
    org_mats:   Option<Res<OrganismMaterials>>,
    existing:   Query<(), (With<BallPlankton>, With<Photoautotroph>)>,
    mut species_cache: Local<Option<crate::species_editor::save::LoadedSpecies>>,
    mut commands: Commands,
    mut meshes:   ResMut<Assets<Mesh>>,
) {
    use crate::simulation_settings::{MIN_PLANKTON_COUNT, PLANKTON_SPECIES_PATH};
    let (Some(heightmap), Some(map_size), Some(water), Some(smoothing), Some(org_mats)) =
        (heightmap, map_size, water, smoothing, org_mats) else { return };

    let current = existing.iter().count();
    if current >= MIN_PLANKTON_COUNT { return; }

    // Lazy-load + cache the species (one disk read for the run).
    if species_cache.is_none() {
        match crate::species_editor::save::load_species(std::path::Path::new(PLANKTON_SPECIES_PATH)) {
            Ok(s)  => *species_cache = Some(s),
            Err(e) => { error!("auto_spawn_plankton: failed to load {PLANKTON_SPECIES_PATH}: {e}"); return; }
        }
    }
    let species = species_cache.as_ref().unwrap();

    let mut rng = rand::rng();
    let to_spawn = MIN_PLANKTON_COUNT - current;
    for _ in 0..to_spawn {
        spawn_species_instance(
            species, "ball_plankton", &heightmap, &map_size, water.0, smoothing.0,
            &mut commands, &mut meshes, &org_mats, &mut rng,
        );
    }
    info!("auto_spawn_plankton: topped up plankton {current} → {MIN_PLANKTON_COUNT}");
}


/// Build the canonical root body part from a flat OCG; `regrowable` so
/// mutation can extend it.
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


/// Shared StandardMaterial handles for `spawn_organism`. Sharing one
/// handle per (kind × debug-flag) pair keeps GPU bind-group churn minimal.
/// Stored as a `Resource` so runtime spawners reuse the same handles;
/// `spawn_colony` populates it on first run.
#[derive(Resource)]
pub struct OrganismMaterials {
    pub photo:       Handle<StandardMaterial>,
    pub hetero:      Handle<StandardMaterial>,
    pub debug_blue:  Handle<StandardMaterial>,
}

impl OrganismMaterials {
    pub fn new(materials: &mut Assets<StandardMaterial>) -> Self {
        // White so per-vertex cell colours (from `build_mesh_from_ocg`)
        // come through unmultiplied. `photo` and `hetero` share it.
        let body = materials.add(StandardMaterial {
            base_color: Color::WHITE,
            ..default()
        });
        Self {
            photo:      body.clone(),
            hetero:     body,
            // Full-part override for procedural reproduction appendages,
            // selected via the BodyPart's `debug_blue` flag.
            debug_blue: materials.add(StandardMaterial {
                base_color: Color::srgb(0.2, 0.4, 0.95),
                ..default()
            }),
        }
    }

    /// Material handle for one body part. Per-cell colour lives on the mesh
    /// (`Mesh::ATTRIBUTE_COLOR`), so this only chooses between the shared
    /// white body material and the debug-blue full-part override.
    pub fn handle_for(&self, _kind: OrganismKind, bp: &BodyPart) -> Handle<StandardMaterial> {
        if bp.debug_blue {
            self.debug_blue.clone()
        } else {
            self.photo.clone()
        }
    }
}


/// Per-limb erratic-rotation parameters (sliding limbs only). Each axis
/// oscillates independently; incommensurate frequencies read as chaotic
/// rather than periodic. `mirror` inverts all three angles so a bilateral
/// limb pair swings in opposition.
#[derive(Component, Clone, Debug)]
pub struct LimbAnimation {
    pub freqs:  [f32; 3],
    pub phases: [f32; 3],
    pub amps:   [f32; 3],
    pub mirror: bool,
}

/// Apply the per-limb erratic rotation each frame. Reads the virtual
/// clock, so animation freezes in lock-step when the sim is paused.
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


/// Construct + register an organism from a list of body parts at `pos`.
/// Each body part owns its own OCG; regrowable-part meshes are built via
/// `build_mesh_from_ocg`. Used by initial spawn and reproduction.
/// Hierarchy: OrganismRoot (Organism + trophic marker) with one mesh
/// child entity per body part (flat — see the parenting note below).
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
    // Movement paradigm — see `Organism::movement_mode`. `is_sliding()` =
    // kinematic root; otherwise limb-based (Avian dynamics).
    movement_mode:      MovementMode,
    // Ground- vs water-based — see `Organism::ground_based`. Only phototrophs
    // may override the movement-mode default (floating algae); heterotrophs
    // are coerced to `movement_mode.default_ground_based()` below.
    ground_based:       bool,
    commands:           &mut Commands,
    meshes:             &mut ResMut<Assets<Mesh>>,
    materials:          &OrganismMaterials,
    rng:                &mut impl rand::Rng,
) -> Entity {
    // Invariants: variable-form ⇒ NoSymmetry + sessile; sessile ⇒ sliding
    // (sessile root is Kinematic; a Dynamic body would get pushed around).
    let symmetry  = if has_variable_form { Symmetry::NoSymmetry } else { symmetry };
    let is_sessile = is_sessile || has_variable_form;
    let movement_mode = if is_sessile { MovementMode::Sliding } else { movement_mode };
    // Invariant: only PHOTOTROPHS may opt OUT of the movement-mode default
    // (a floating, water-based alga is sessile/sliding yet not ground-based);
    // heterotrophs always derive it (swimmers false, sliders/walkers true),
    // and nobody can claim ground-based while on a fluid movement mode.
    let ground_based = match kind {
        OrganismKind::Photoautotroph => ground_based && movement_mode.default_ground_based(),
        OrganismKind::Heterotroph    => movement_mode.default_ground_based(),
    };
    if body_parts.is_empty() {
        panic!("spawn_organism called with empty body_parts");
    }

    // Sync per-cell physiology caches (neighbour_count, PhotosyntheticCell)
    // before building the Organism; the photosynthesis tick reads them, so
    // skipping this would make every cell produce zero energy.
    crate::physiology::recompute_body_parts(&mut body_parts);

    let angle     = rng.random::<f32>() * std::f32::consts::TAU;
    let direction = Vec3::new(angle.cos(), 0.0, angle.sin());
    let speed     = match kind {
        OrganismKind::Photoautotroph => 0.0,
        OrganismKind::Heterotroph    => 15.0 + rng.random::<f32>() * 10.0,
    };

    // Born adult unless variable-form (those grow via `continuous_growth`
    // and become adult only at `MAX_CELLS`).
    let adult = !has_variable_form;

    let mut organism = Organism {
        body_parts: body_parts.clone(),
        symmetry,
        intelligence_level,
        is_sessile,
        has_variable_form,
        movement_mode,
        ground_based,
        limb_targets: [0.0; 10],
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
        // Structural DNA slots filled now (valid vector from frame 1);
        // brain-gene slots stay 0 until `sync_dna_from_brain_pool` runs.
        dna: crate::lineages::dna::structural_dna(
            kind,
            symmetry,
            has_variable_form,
            is_sessile,
            intelligence_level,
        ),
        // Set post-spawn by `reproduction.rs` when reproducing; `None`
        // otherwise, so the classification tick assigns it.
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
    // Level0 → `BrainLevel0` keeps the entity out of the L1-L3 assign
    // queries; L1-L3 are picked up by their `assign_*` via trophic marker.
    match intelligence_level {
        IntelligenceLevel::Level0 => {
            root_cmd.insert(crate::intelligence_level_0::BrainLevel0);
        }
        IntelligenceLevel::Level1
        | IntelligenceLevel::Level2
        | IntelligenceLevel::Level3 => {}
    }
    let root = root_cmd.id();

    // One mesh child entity per body part. Assumes any branch's parent_idx
    // < its own index (the order callers naturally produce).
    let mut bp_entities: Vec<Entity> = Vec::with_capacity(body_parts.len());
    for (idx, bp) in body_parts.iter().enumerate() {
        // Skip mesh for non-regrowable / empty parts. Adult + smoothing on
        // → smoothed mesh, else faceted (continuous_growth re-smooths later).
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

        // Branch transform: origin_local + attachment.rotation; root identity.
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

        // Sliding-only: Limb parts get `LimbAnimation` (mirror twins on
        // the −X side swing in opposition). Limb-based organisms drive
        // joints via physics, which would fight this animation.
        if movement_mode.is_sliding()
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

        // Flat layout: parent EVERY body-part entity directly under the
        // root. Body-part transforms are identity in practice, so this
        // matches nesting; it also avoids a phantom-organism bug where
        // despawning body_parts[0] cascaded to nested branches (leaving a
        // ghost root whose data still claimed alive parts).
        let _ = &bp.attachment; // kept on the data side for continuous_growth
        commands.entity(root).add_child(child);

        bp_entities.push(child);
    }

    // Avian3d physics. Sliding: Kinematic root (transform written by
    // `apply_movement`) + one compound collider from all cells (offset by
    // attachment origin to match world positions). Limb-based: per-part
    // Dynamic body + collider, plus one hinge joint per appendage.
    if movement_mode.is_sliding() {
        insert_sliding_collider(commands, root, &body_parts);
    } else {
        insert_limb_physics(commands, root, &bp_entities, &body_parts, movement_mode.is_swimming());
    }

    root
}


// ── Auto-spawn heterotrophs ─────────────────────────────────────────────────
//
// Tops heterotrophs up to `MinHeteroCount` on a death event when
// `AutoSpawnHeteros(true)`. Zero-overhead in steady state (no removed
// events); drain the reader when the flag is off to avoid backlog. New
// organisms carry no `BrainInheritance`, so the pool samples fresh genes.
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
        // Inside the WORLD_SAFETY_MARGIN inset — the XZ band
        // `apply_world_bounds` keeps organisms within at runtime.
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
            MovementMode::Sliding,  // auto-spawned heteros use legacy sliding
            true,                   // sliding ⇒ ground-based
            &mut commands,
            &mut meshes,
            &org_mats,
            &mut rng,
        );
    }
}
