// Movement, gravity, floor collision and world-bounds clamping (sliding
// organisms; limb organisms are driven by Avian).
//
// Wall/climb queries iterate every body part's cells. A cell's world position
// is `transform.transform_point(body_part.local_offset + cell.local_pos)` —
// body parts are flat children of the root with no per-part rotation, so the
// root Transform applies to every cell.

use bevy::prelude::*;
use bevy::transform::TransformSystems;

use crate::cell::*;
use crate::colony::*;
use crate::environment::WaterLevel;
use crate::organism_collision;
use crate::world_geometry::{HeightmapSampler, MapSize, WorldMesh, WORLD_SAFETY_MARGIN};
use bevy::math::Mat3;

use crate::simulation_settings::{MIN_DIRECTION_INTERVAL, MAX_DIRECTION_INTERVAL, PHOTO_WANDER_MAX_SPEED};

use crate::simulation_settings::{GRAVITY, MAX_CLIMB_HEIGHT};
use crate::simulation_settings::SURFACE_ADHESION_SEARCH;

use crate::simulation_settings::ORGANISM_DESPAWN_Y;


/// Re-roll cadence for photoautotroph wander direction (initial interval in
/// `[MIN_DIRECTION_INTERVAL, MAX_DIRECTION_INTERVAL]`).
#[derive(Component)]
pub struct DirectionTimer {
    pub timer: Timer,
}

impl DirectionTimer {
    pub fn new(duration: f32) -> Self {
        Self { timer: Timer::from_seconds(duration, TimerMode::Repeating) }
    }
}


/// One-shot placement marker. Present once `place_sessile_organisms` has seated
/// an organism on the terrain. SESSILE life forms are placed on the floor ONCE
/// and then receive NO per-frame work (no gravity, no surface adhesion, no
/// floor/wall collision) — they are inert scenery that just gets eaten. Mobile
/// organisms are marked too (so they fall out of the placement query) but keep
/// their own per-frame movement systems.
///
/// The marker is REMOVED by the only two things that move/reshape a sessile
/// body — continuous growth (`grow_variable_form_organisms`) and prey relocation
/// (`maintain_prey_near_herbivores`) — so the placement re-runs exactly once per
/// such event, never per frame.
#[derive(Component)]
pub struct SurfacePlaced;


pub struct MovementPlugin;

impl Plugin for MovementPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(organism_collision::OrganismCollisionTimer::default());
        // Swimmer water-plane ceiling reads `WaterLevel`. `init_resource` only
        // seeds the default if absent, so phase-5 launcher/.colony wiring (which
        // may run `insert_resource` earlier) is preserved.
        app.init_resource::<WaterLevel>();

        // Wall/climb queries need the heightmap (airborne early-out) + the
        // triangle mesh (full test); both land asynchronously, so gate on them.
        app.add_systems(PostUpdate,
            apply_movement
                .before(TransformSystems::Propagate)
                .run_if(resource_exists::<HeightmapSampler>)
                .run_if(resource_exists::<WorldMesh>)
        );

        // MOBILE ground-based sliders crawl arbitrary surfaces like snails
        // (orient to the surface normal + glue + tangential travel). Sessile
        // organisms are EXCLUDED — they are seated once by
        // `place_sessile_organisms` and never re-adhered. Runs in the same
        // PostUpdate window as `apply_movement` (disjoint organism set:
        // ground-based vs water-based) and before Propagate so the
        // collider/global transforms pick up the new pose.
        app.add_systems(PostUpdate,
            apply_surface_adhesion
                .before(TransformSystems::Propagate)
                .run_if(resource_exists::<HeightmapSampler>)
                .run_if(resource_exists::<WorldMesh>)
        );

        // One-shot terrain seating for SESSILE life forms (and a no-op mark for
        // mobile ones, so they leave the query). Runs before Propagate so the
        // collider/global transform pick up the placement the same frame. The
        // `Without<SurfacePlaced>` filter makes this empty once everything is
        // seated — zero per-frame cost for the settled population.
        app.add_systems(PostUpdate,
            place_sessile_organisms
                .before(TransformSystems::Propagate)
                .run_if(resource_exists::<HeightmapSampler>)
        );

        app.add_systems(PostUpdate, (
            apply_floor_collision.run_if(resource_exists::<HeightmapSampler>),
            apply_world_bounds.run_if(resource_exists::<HeightmapSampler>),
            // After Propagate so trunk GlobalTransforms are current. Ungated:
            // pure Y test, query empty until organisms exist.
            despawn_fallen_organisms,
        ).chain().after(TransformSystems::Propagate));

        app.add_systems(Last, organism_collision::apply_organism_collision);

        // Photoautotroph wander has always been 2D (the project only ever ran
        // the old `TwoD` mode); gravity is now applied per-organism inside
        // `apply_gravity` (kinematic-sliding organisms only).
        app.add_systems(PostUpdate, random_2d_direction.before(apply_movement));
        app.add_systems(PostUpdate,
            apply_gravity
                .after(random_2d_direction)
                .before(apply_movement));
    }
}


// ── Random direction (photoautotrophs) ──────────────────────────────────────

fn random_2d_direction(
    time: Res<Time>,
    mut query: Query<(&mut Organism, &mut DirectionTimer), (With<OrganismRoot>, With<Photoautotroph>)>,
) {
    let dt = time.delta();

    for (mut organism, mut timer) in &mut query {
        // Active wander is only for organisms that actually translate under it:
        // GROUND-BASED, non-sessile photoautotrophs. Sessile ones are skipped by
        // `apply_movement` (they never move), and water-based floaters drift via
        // buoyancy rather than propelling. Assigning EITHER a wander speed is a
        // phantom — it never moves them, but `energy.rs` still charges its
        // movement cost. For a SUBMERGED floater (e.g. sessile water-based
        // `ball_plankton`) that's the speed³ fluid-drag term, which dwarfs
        // photosynthesis and starves it to 0 → despawn (then the plankton floor
        // respawns it → perpetual sporadic despawn/respawn). Keep their speed 0.
        if organism.is_sessile || !organism.ground_based {
            if organism.movement_speed != 0.0 { organism.movement_speed = 0.0; }
            continue;
        }

        timer.timer.tick(dt);
        if !timer.timer.just_finished() { continue; }

        let angle = rand::random::<f32>() * std::f32::consts::TAU;
        organism.movement_direction = Vec3::new(angle.cos(), 0.0, angle.sin());
        // Bounded small: a large wander speed makes the submerged `speed³` fluid
        // drag dwarf photosynthesis and starve the organism (see
        // `PHOTO_WANDER_MAX_SPEED`).
        organism.movement_speed     = rand::random::<f32>() * PHOTO_WANDER_MAX_SPEED;

        let next = MIN_DIRECTION_INTERVAL
            + rand::random::<f32>() * (MAX_DIRECTION_INTERVAL - MIN_DIRECTION_INTERVAL);
        timer.timer = Timer::from_seconds(next, TimerMode::Repeating);
    }
}


// ── Apply movement (wall collision via WorldMesh) ────────────────────────────

/// Per-cell AABB-vs-mesh test for a proposed XZ step from `pose`. Returns
/// `(blocked, worst_climb_needed)`: `blocked` is true if any cell would
/// intersect the world mesh after the step, and `worst_climb_needed` is the
/// largest surface-top rise above a cell's bottom across all hits. Shared by
/// the straight-ahead test and the wall-slide re-test in `apply_movement`.
///
/// The AABB bottom is inset by `RD_HALF_SIZE` so a settled cell reads as
/// resting rather than wall-blocked — without it, a settled cell triggers a
/// permanent climb-vs-gravity oscillation (terrain jitter).
fn move_blocked_climb(
    pose:        &Transform,
    body_parts:  &[BodyPart],
    move_xz:     Vec3,
    world_mesh:  &WorldMesh,
    heightmap:   &HeightmapSampler,
    half:        f32,
    scratch:     &mut Vec<u32>,
) -> (bool, f32) {
    let mut blocked = false;
    let mut climb_needed = 0.0_f32;
    for body_part in body_parts {
        for cell in &body_part.cells {
            let local_pos = body_part.local_offset + cell.local_pos;
            let world_pos = pose.transform_point(local_pos);
            let next_pos  = world_pos + move_xz;

            let cell_min = Vec3::new(next_pos.x - half, next_pos.y, next_pos.z - half);
            let cell_max = next_pos + Vec3::splat(half);

            if world_mesh.aabb_intersects_with(cell_min, cell_max, scratch) {
                blocked = true;
                let top = world_mesh
                    .max_y_in_xz_with(cell_min, cell_max, scratch)
                    .unwrap_or_else(|| heightmap.height_at(next_pos.x, next_pos.z));
                let needed = top - (world_pos.y - half);
                if needed > climb_needed { climb_needed = needed; }
            }
        }
    }
    (blocked, climb_needed)
}

/// Step each NON-ground-based sliding organism in its commanded direction
/// (ground-based sliders adhere to surfaces via `apply_surface_adhesion`
/// instead and are skipped here).
///   1. Broad-phase: airborne (well above heightmap) ⇒ skip triangle queries.
///   2. Per cell, test a post-step AABB against `WorldMesh`.
///   3. On a hit: a surmountable rise (≤ `MAX_CLIMB_HEIGHT`) is climbed;
///      a steeper wall blocks the step.
fn apply_movement(
    time:            Res<Time>,
    world_mesh:      Res<WorldMesh>,
    heightmap:       Res<HeightmapSampler>,
    mut query:       Query<(&mut Transform, &mut Organism), With<OrganismRoot>>,
) {
    let dt = time.delta_secs();
    let half_size = RD_HALF_SIZE;

    // Water-based sliders move independently (own Transform + Organism, read-only
    // shared mesh/heightmap), so the per-cell wall/climb mesh tests fan out over
    // `ComputeTaskPool`. The triangle-candidate scratch is per-closure (a shared
    // `Local` would race across threads).
    query.par_iter_mut().for_each(|(mut transform, mut organism)| {
        let mut tri_scratch: Vec<u32> = Vec::new();
        // Sessile organisms never move (brain writes ignored here); floor/
        // bounds clamping still apply via their own systems.
        if organism.is_sessile { return; }
        // Limb organisms get their pose from Avian — don't fight the solver.
        // Only kinematic Sliding is processed here; LimbBasedWalking / Swimming
        // / Flying take the dynamic path and are skipped.
        if !organism.movement_mode.is_sliding() { return; }
        // Ground/ocean-floor sliders adhere to surfaces (snails) and are driven
        // by `apply_surface_adhesion`; only water-based sliders fall through to
        // this gravity-free wall-collision path.
        if organism.ground_based { return; }

        // Yaw so local +Z ("front") points along the XZ movement direction.
        // Y-axis only (no pitch/roll) keeps the AABB tests axis-aligned. Snap
        // (no slerp): brains can change heading any tick, and smoothing would
        // lag the visual behind the collision geometry.
        if organism.movement_speed > 0.0 {
            let dir = organism.movement_direction;
            let dir_xz_len_sq = dir.x * dir.x + dir.z * dir.z;
            if dir_xz_len_sq > 1e-6 {
                // Y-axis yaw θ maps local +Z to world (sin θ, 0, cos θ);
                // aligning with dir gives θ = atan2(dir.x, dir.z).
                let yaw = dir.x.atan2(dir.z);
                transform.rotation = Quat::from_rotation_y(yaw);
            }
        }

        let move_vector = organism.movement_direction * organism.movement_speed * dt;

        // Airborne early-out: well above the heightmap → no triangles to hit.
        let floor_y = heightmap.height_at(transform.translation.x, transform.translation.z);
        if transform.translation.y - organism.bounding_radius() > floor_y + 2.0 {
            organism.is_climbing = false;
            transform.translation.x += move_vector.x;
            transform.translation.z += move_vector.z;
            return;
        }

        let move_xz = Vec3::new(move_vector.x, 0.0, move_vector.z);
        let pos     = transform.translation;
        let pose    = *transform;   // pre-move pose snapshot for the collision tests

        let mut is_blocked   = false;
        let mut climb_needed = 0.0_f32;

        if move_xz.x != 0.0 || move_xz.z != 0.0 {
            // Whole-organism broad-phase: one AABB-vs-mesh test before the
            // per-cell work. The cached bounding radius pads beyond the farthest
            // cell by ≥ 2·RD_HALF_SIZE, so a radius box at the proposed position
            // conservatively encloses every per-cell AABB; a miss ⇒ no cell can
            // hit ⇒ move freely.
            let radius      = organism.bounding_radius();
            let centre_next = pos + move_xz;
            let org_min = centre_next - Vec3::splat(radius);
            let org_max = centre_next + Vec3::splat(radius);

            if !world_mesh.aabb_intersects_with(org_min, org_max, &mut tri_scratch) {
                organism.is_climbing = false;
                transform.translation.x += move_vector.x;
                transform.translation.z += move_vector.z;
                return;
            }

            let (b, c) = move_blocked_climb(
                &pose, &organism.body_parts, move_xz,
                &world_mesh, &heightmap, half_size, &mut tri_scratch,
            );
            is_blocked   = b;
            climb_needed = c;
        }

        if is_blocked && climb_needed > 0.0 && climb_needed <= MAX_CLIMB_HEIGHT {
            // Surmountable rise: scramble up it (Y only; floor-follow settles).
            let climb_amount = (organism.movement_speed * dt).min(climb_needed);
            transform.translation.y += climb_amount;
            organism.climb_energy_debt += climb_amount;
            organism.is_climbing = true;
        } else if is_blocked {
            organism.is_climbing = false;
        } else {
            // Clear ahead: take the full step.
            transform.translation.x += move_vector.x;
            transform.translation.z += move_vector.z;
            organism.is_climbing = false;
        }
    });
}


// ── Surface adhesion (MOBILE ground-based sliders crawl like snails) ─────────

/// Kinematic surface adhesion for MOBILE GROUND-BASED sliding organisms. For
/// them this REPLACES gravity + heightmap floor-lift + wall collision: each tick
/// we find the nearest mesh surface, glue the root to it (offset out along the
/// surface normal), orient the body's up-axis to that normal, and crawl along
/// the surface tangent. The result is snail-like travel across arbitrary angled
/// surfaces (floors, walls, even overhangs) with NO physics solver — one
/// localized closest-point query per organism. Visual exactness is not a goal;
/// staying glued and never getting stuck is.
///
/// SESSILE organisms are excluded: they never move, so they are seated ONCE by
/// `place_sessile_organisms` (re-seated on growth/relocation) and never re-adhere
/// — no per-frame query work for the settled plant/prey population.
fn apply_surface_adhesion(
    time:         Res<Time>,
    world_mesh:   Res<WorldMesh>,
    heightmap:    Res<HeightmapSampler>,
    mut query:    Query<(&mut Transform, &mut Organism), With<OrganismRoot>>,
) {
    let dt = time.delta_secs();

    // Each crawler re-adheres independently (its own Transform + Organism, read-only
    // shared `WorldMesh`/`HeightmapSampler`), so the per-organism mesh closest-point
    // query fans out over `ComputeTaskPool`. The triangle-candidate `scratch` must be
    // per-closure (a shared `Local` would be a data race across threads).
    query.par_iter_mut().for_each(|(mut transform, mut organism)| {
        let mut scratch: Vec<u32> = Vec::new();
        if organism.is_sessile { return; }       // seated once by place_sessile_organisms
        if !organism.movement_mode.is_sliding() { return; }
        if !organism.ground_based { return; }   // water-based: handled elsewhere

        let pos = transform.translation;

        // Surface we're adhering to: nearest mesh point + normal. Off-mesh
        // (farther than the search radius) ⇒ fall back to the heightmap as a
        // flat floor, so the body never floats away or falls forever.
        let (foot, normal) = world_mesh
            .closest_surface(pos, SURFACE_ADHESION_SEARCH, &mut scratch)
            .unwrap_or_else(|| {
                let h = heightmap.height_at(pos.x, pos.z);
                (Vec3::new(pos.x, h, pos.z), Vec3::Y)
            });

        // Tangential crawl: commanded move projected onto the surface plane.
        // The into-surface component is dropped, so the body never drives itself
        // into the wall.
        let dir   = organism.movement_direction;
        let speed = organism.movement_speed;
        let tangent_step = if speed > 0.0 {
            let step = dir * (speed * dt);
            step - normal * step.dot(normal)
        } else {
            Vec3::ZERO
        };

        // Offset the root so the LOWEST cell's bottom sits on the surface. Local
        // +Y maps to the surface normal, so a cell's distance along the normal
        // from the root equals its local Y. The cell half-extent MUST be the
        // GEOMETRY-SCALED half-spacing (`CELL_SPACING = GEOMETRY_SCALE`), NOT the
        // canonical `RD_HALF_SIZE = 0.5`: at the live 0.1 geometry scale a body's
        // cells are ~0.1 apart, so using 0.5 floated every organism a fixed
        // ~0.5u (≈5× its own size) above the ground — the constant hover. This
        // tracks `GEOMETRY_SCALE` automatically.
        let mut min_local_y = f32::INFINITY;
        for bp in &organism.body_parts {
            for cell in &bp.cells {
                let ly = bp.local_offset.y + cell.local_pos.y;
                if ly < min_local_y { min_local_y = ly; }
            }
        }
        let cell_half = CELL_SPACING * 0.5;
        let offset = cell_half - if min_local_y.is_finite() { min_local_y } else { 0.0 };

        // Glue to the surface (foot + normal offset) plus the tangential step;
        // next tick re-snaps from the new position so it tracks the surface
        // (any over-step toward a steeper wall self-corrects then).
        transform.translation = foot + normal * offset + tangent_step;
        organism.is_climbing = true;   // adhered → never free-fall

        // Orient: local +Y → surface normal, local +Z → travel direction
        // (projected onto the surface). Cheap orthonormal basis → quat.
        let up = normal;
        let mut fwd = if tangent_step.length_squared() > 1e-10 {
            tangent_step
        } else {
            // Not travelling this tick: keep the current facing re-projected
            // onto the new surface plane (so a stopped body doesn't spin).
            let cur = transform.rotation * Vec3::Z;
            cur - up * cur.dot(up)
        };
        if fwd.length_squared() < 1e-10 {
            // Degenerate (facing ∥ normal): any tangent will do.
            fwd = up.cross(Vec3::X);
            if fwd.length_squared() < 1e-10 { fwd = up.cross(Vec3::Z); }
        }
        let fwd   = fwd.normalize();
        let right = up.cross(fwd).normalize_or_zero();   // local X = up × fwd (right-handed)
        if right != Vec3::ZERO {
            transform.rotation = Quat::from_mat3(&Mat3::from_cols(right, up, fwd));
        }
    });
}


// ── One-shot sessile seating ─────────────────────────────────────────────────

/// Seat SESSILE organisms on the ocean floor / terrain ONCE, then leave them
/// completely alone. This is the whole runtime cost of a sedentary life form:
/// placed on the surface at spawn (and re-placed only when it grows or is
/// relocated), it then gets NO gravity, NO surface adhesion, NO floor/wall
/// collision — inert scenery that only gets eaten.
///
/// Placement uses the SAME terrain source the sliders rest on — the actual world
/// mesh (`WorldMesh::closest_surface`), NOT the heightmap (whose per-cell MAX
/// reads above the rendered surface and floats plants) — falling back to the
/// heightmap only when off-mesh. The `min_local_y` term sits the body's lowest
/// cell's bottom on the surface (lift scaled by `CELL_SPACING`).
///
/// The `Without<SurfacePlaced>` filter means this is a true one-shot: every
/// organism is processed the frame after it spawns and then drops out of the
/// query, so the settled population costs nothing. Mobile organisms are marked
/// here too (so they leave the query) but are NOT moved — their height is owned
/// by `apply_surface_adhesion` / `apply_gravity`. Re-seating after growth or
/// relocation is triggered by removing the marker (see `SurfacePlaced`).
fn place_sessile_organisms(
    mut commands: Commands,
    world_mesh:   Option<Res<WorldMesh>>,
    heightmap:    Res<HeightmapSampler>,
    mut query:    Query<(Entity, &mut Transform, &Organism), (With<OrganismRoot>, Without<SurfacePlaced>)>,
    mut scratch:  Local<Vec<u32>>,
) {
    for (entity, mut transform, organism) in &mut query {
        // Mark every organism so it leaves the query; only sessile ones are moved.
        commands.entity(entity).try_insert(SurfacePlaced);
        if !organism.is_sessile { continue; }

        let min_local_y = organism.body_parts.iter()
            .flat_map(|bp| bp.cells.iter().map(move |c| bp.local_offset.y + c.local_pos.y))
            .fold(f32::INFINITY, f32::min);
        let lift = CELL_SPACING * 0.5 - if min_local_y.is_finite() { min_local_y } else { 0.0 };

        let pos = transform.translation;
        let surface_y = world_mesh.as_ref()
            .and_then(|wm| wm.closest_surface(pos, SURFACE_ADHESION_SEARCH, &mut scratch))
            .map(|(foot, _normal)| foot.y)
            .unwrap_or_else(|| heightmap.height_at(pos.x, pos.z));
        transform.translation.y = surface_y + lift;
    }
}


// ── Gravity ──────────────────────────────────────────────────────────────────

/// Manual gravity for KINEMATIC (sliding) organisms, gated by
/// `Organism::ground_based`:
///   * ground-based (sliders/walkers) — gravity always applies (unchanged);
///   * WATER-BASED (floating phototrophs) — gravity applies ONLY while the
///     root is ABOVE the water surface (it falls back in); submerged it
///     holds its depth — neither sinking to the bottom nor surfacing.
/// Cost: one extra compare per organism per frame (the water level is read
/// once); no per-cell work.
fn apply_gravity(
    time:  Res<Time>,
    water: Res<WaterLevel>,
    mut query: Query<(&mut Transform, &mut Organism), With<OrganismRoot>>,
) {
    let dt = time.delta_secs();
    let water_y = water.0;

    for (mut transform, mut organism) in &mut query {
        // Sessile life forms are seated on the floor once and never fall.
        if organism.is_sessile { continue; }
        // Kinematic-sliding only: LimbBasedWalking / Swimming / Flying are all
        // dynamic bodies and get gravity (or not) from rapier, not Organism.velocity.
        if !organism.movement_mode.is_sliding() { continue; }
        // Ground/ocean-floor sliders are glued to surfaces by
        // `apply_surface_adhesion` (no free-fall); only water-based sliders get
        // gravity here.
        if organism.ground_based { continue; }
        // Water-based + submerged → neutral buoyancy: zero any residual
        // vertical velocity (e.g. from the fall that brought it under) and
        // skip the gravity integration entirely.
        if !organism.ground_based && transform.translation.y <= water_y {
            if organism.velocity.y != 0.0 { organism.velocity.y = 0.0; }
            continue;
        }
        if !organism.is_climbing {
            organism.velocity.y += -GRAVITY * dt;
            transform.translation.y += organism.velocity.y * dt;
        }
    }
}


// ── Floor collision ──────────────────────────────────────────────────────────

/// Lift the organism until no grown cell penetrates the heightmap (cell world
/// positions come directly from the root transform).
fn apply_floor_collision(
    heightmap: Res<HeightmapSampler>,
    mut query: Query<(&mut Transform, &mut Organism), With<OrganismRoot>>,
) {
    let half = RD_HALF_SIZE;
    for (mut transform, mut organism) in &mut query {
        // Sessile life forms are seated on the floor once; no per-frame lift.
        if organism.is_sessile { continue; }
        // Kinematic-sliding only (mirrors `apply_gravity`): limb/swimmer floor
        // handling lives in `rapier_setup::enforce_limb_floor_and_contacts`.
        // This system runs `.after(TransformSystems::Propagate)` — i.e. after
        // `sync_multibody_link_transforms` — so without this guard it would
        // clobber Rapier's authoritative pose for any penetrating dynamic body.
        if !organism.movement_mode.is_sliding() { continue; }
        // Ground/ocean-floor sliders are placed by `apply_surface_adhesion`;
        // this heightmap floor-lift only catches water-based sliders.
        if organism.ground_based { continue; }
        let mut max_pen = 0.0_f32;

        for body_part in &organism.body_parts {
            for cell in &body_part.cells {
                let local_pos   = body_part.local_offset + cell.local_pos;
                let world_pos   = transform.transform_point(local_pos);
                let cell_bottom = world_pos.y - half;
                let floor_y     = heightmap.height_at(world_pos.x, world_pos.z);
                let pen         = floor_y - cell_bottom;
                if pen > max_pen { max_pen = pen; }
            }
        }

        if max_pen > 0.0 {
            transform.translation.y += max_pen;
            if organism.velocity.y < 0.0 { organism.velocity.y = 0.0; }
        }
    }
}


// ── World boundaries ─────────────────────────────────────────────────────────

fn apply_world_bounds(
    map_size:  Res<MapSize>,
    mut query: Query<&mut Transform, With<OrganismRoot>>,
) {
    // Bounds = map area inset by WORLD_SAFETY_MARGIN per edge. The terrain is
    // normalised ≥ MapSize, so this clamp stays inside the geometry; the
    // margin keeps organisms off border cliffs / the world AABB.
    let min_x = WORLD_SAFETY_MARGIN;
    let max_x = (map_size.x - WORLD_SAFETY_MARGIN).max(WORLD_SAFETY_MARGIN);
    let min_z = WORLD_SAFETY_MARGIN;
    let max_z = (map_size.z - WORLD_SAFETY_MARGIN).max(WORLD_SAFETY_MARGIN);

    for mut transform in &mut query {
        transform.translation.x = transform.translation.x.clamp(min_x, max_x);
        transform.translation.z = transform.translation.z.clamp(min_z, max_z);
    }
}


// ── Kill-floor ─────────────────────────────────────────────────────────────────

/// Despawn organisms fallen below `ORGANISM_DESPAWN_Y` (slipped off the edge /
/// through a mesh gap — they'd fall forever, wasting cycles).
///
/// Keys off Avian's `Position` (authoritative), NOT `GlobalTransform`, which
/// has been observed reading a large constant Y offset and wrongly despawning
/// the whole colony. Two cases by movement mode:
///   * Sliding/sessile: kinematic `RigidBody` on the root → root has `Position`.
///   * Limb: no root rigid body → read the trunk part's (`BodyPartIndex(0)`).
///
/// Despawning the root cascades to children and frees the brain slot exactly
/// as a starvation death does.
fn despawn_fallen_organisms(
    mut commands: Commands,
    // Only sliding/sessile organisms are kill-floored here. LIMB organisms are
    // kept alive deliberately: on a tunnel-proof floor they can't fall away, and
    // a stumble during STANDING training must NOT delete the learner — the policy
    // has to recover the stance, not get one shot and vanish.
    // Phototrophs are excluded: they're the prey food source. Sessile ones are
    // seated on the floor by `place_sessile_organisms` and get no gravity, so
    // they never fall; this exclusion also protects any mobile phototroph from a
    // transient drop, so the herbivores never lose their food source.
    roots: Query<(Entity, &GlobalTransform, &crate::colony::Organism),
                 (With<OrganismRoot>, Without<crate::colony::Photoautotroph>)>,
) {
    for (root, gt, org) in &roots {
        if !org.movement_mode.is_sliding() { continue; }
        if gt.translation().y < ORGANISM_DESPAWN_Y {
            commands.entity(root).despawn();
        }
    }
}


// ── Tests (headless; no CUDA) ────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::{BodyPart, BodyPartKind, Cell, CellType, CELL_SPACING};
    use crate::colony::{IntelligenceLevel, MovementMode, Organism, Symmetry};

    /// 1×1 heightmap: `height_at` returns `h` everywhere.
    fn flat_heightmap(h: f32) -> HeightmapSampler {
        HeightmapSampler { heights: vec![h], width: 1, depth: 1, min_x: 0, min_z: 0, max_height: h }
    }

    /// Minimal phototroph: one body part, one cell at local +Y = `cell_y`.
    fn test_photo(movement_mode: MovementMode, ground_based: bool, cell_y: f32) -> Organism {
        let cell = Cell {
            local_pos: Vec3::new(0.0, cell_y, 0.0),
            cell_type: CellType::Photo,
            cell_energy: 0.0,
            neighbour_count: 0,
            photo: None,
        };
        let bp = BodyPart {
            kind: BodyPartKind::Body,
            local_offset: Vec3::ZERO,
            cells: vec![cell],
            ocg: vec![],
            attachment: None,
            consumed: false,
            debug_blue: false,
            regrowable: true,
        };
        Organism {
            body_parts: vec![bp],
            symmetry: Symmetry::NoSymmetry,
            intelligence_level: IntelligenceLevel::Level0,
            is_sessile: true,
            has_variable_form: true,
            movement_mode,
            ground_based,
            limb_targets: [0.0; 10],
            adult: false,
            photo_cell_count: 1,
            non_photo_cell_count: 0,
            upkeep_weight: 0.0,
            energy: 0.0,
            in_sunlight: false,
            reproduced: false,
            reproductions: 0,
            predations: 0,
            hunger: 0.0,
            dopamine: 0.0,
            target_distance: 0.0,
            movement_speed: 0.0,
            movement_direction: Vec3::ZERO,
            velocity: Vec3::ZERO,
            is_climbing: false,
            climb_energy_debt: 0.0,
            cached_bounding_radius: 1.0,
            dna: vec![],
            species_id: None,
        }
    }

    /// Run `place_sessile_organisms` once on a world holding one sessile
    /// phototroph spawned far above a flat terrain; return its resulting Y.
    fn run_place(org: Organism, spawn_y: f32, terrain: f32) -> f32 {
        let mut world = World::new();
        world.insert_resource(flat_heightmap(terrain));
        let e = world
            .spawn((Transform::from_xyz(5.0, spawn_y, 7.0), Photoautotroph, OrganismRoot, org))
            .id();
        let mut sched = Schedule::default();
        sched.add_systems(place_sessile_organisms);
        sched.run(&mut world);
        world.get::<Transform>(e).unwrap().translation.y
    }

    #[test]
    fn places_ground_phototroph_onto_floor() {
        // Lowest cell at local 0 ⇒ root sits at terrain + CELL_SPACING/2 so the
        // cell's BOTTOM rests on the floor.
        let y = run_place(test_photo(MovementMode::Sliding, true, 0.0), 100.0, 10.0);
        let expected = 10.0 + CELL_SPACING * 0.5;
        assert!((y - expected).abs() < 1e-4, "ground photo y={y}, expected≈{expected}");
    }

    #[test]
    fn places_water_phototroph_onto_floor_too() {
        // Sedentary life forms ALL sit on the floor — even a water-based one.
        let y = run_place(test_photo(MovementMode::Sliding, false, 0.0), 250.0, 10.0);
        let expected = 10.0 + CELL_SPACING * 0.5;
        assert!((y - expected).abs() < 1e-4, "water photo y={y}, expected≈{expected}");
    }

    #[test]
    fn placement_accounts_for_body_extent_below_origin() {
        // Lowest cell 0.3 BELOW the origin ⇒ root lifts higher so that cell's
        // bottom still lands on the floor.
        let y = run_place(test_photo(MovementMode::Sliding, true, -0.3), 100.0, 10.0);
        let expected = 10.0 + CELL_SPACING * 0.5 - (-0.3);
        assert!((y - expected).abs() < 1e-4, "y={y}, expected≈{expected}");
    }

    /// Exercises the REAL sim path: with a `WorldMesh` present, placement must use
    /// `closest_surface` (the mesh, where sliders sit) — NOT the heightmap.
    /// A flat mesh quad at Y=8 sits BELOW a heightmap that reports Y=10, so a
    /// correct placement lands the plant on the MESH (8), proving it ignores the
    /// (higher) heightmap when the mesh is available.
    #[test]
    fn placement_uses_world_mesh_surface_not_heightmap() {
        use crate::world_geometry::build_world_mesh;

        let mut world = World::new();
        world.insert_resource(flat_heightmap(10.0)); // heightmap claims Y=10 …
        // … but the actual mesh surface is at Y=8 (a big flat quad around origin).
        let quad = vec![
            [Vec3::new(-50.0, 8.0, -50.0), Vec3::new(50.0, 8.0, -50.0), Vec3::new(50.0, 8.0, 50.0)],
            [Vec3::new(-50.0, 8.0, -50.0), Vec3::new(50.0, 8.0, 50.0), Vec3::new(-50.0, 8.0, 50.0)],
        ];
        world.insert_resource(build_world_mesh(quad));

        let org = test_photo(MovementMode::Sliding, true, 0.0);
        let e = world
            .spawn((Transform::from_xyz(5.0, 9.0, 7.0), Photoautotroph, OrganismRoot, org))
            .id();
        let mut sched = Schedule::default();
        sched.add_systems(place_sessile_organisms);
        sched.run(&mut world);

        let y = world.get::<Transform>(e).unwrap().translation.y;
        let expected = 8.0 + CELL_SPACING * 0.5; // mesh surface, not heightmap's 10
        assert!((y - expected).abs() < 1e-4, "y={y}, expected≈{expected} (mesh, not heightmap)");
    }
}
