// Movement, gravity, floor collision and world-bounds clamping.
//
// New architecture: organisms are made of one or more body parts, and each
// body part has a list of grown vertex-cells. Wall collision and climb
// queries iterate the cells of every body part rather than the old single
// `active_cells` list. World position of a cell is
// `transform.transform_point(body_part.local_offset + cell.local_pos)` —
// body parts are flat children of the root entity (no per-body-part
// rotation), so the same root Transform applies to every cell.

use bevy::prelude::*;
use bevy::transform::TransformSystems;

use crate::cell::*;
use crate::colony::*;
use crate::organism_collision;
use crate::world_geometry::{HeightmapSampler, MapSize, WorldMesh, WORLD_SAFETY_MARGIN};

const MIN_DIRECTION_INTERVAL: f32 = 1.0;
const MAX_DIRECTION_INTERVAL: f32 = 10.0;

const GRAVITY:          f32 = 9.8;
const MAX_CLIMB_HEIGHT: f32 = 4.0;


/// Re-roll cadence for photoautotroph wander direction. Each spawn rolls its
/// own initial interval in `[MIN_DIRECTION_INTERVAL, MAX_DIRECTION_INTERVAL]`.
#[derive(Component)]
pub struct DirectionTimer {
    pub timer: Timer,
}

impl DirectionTimer {
    pub fn new(duration: f32) -> Self {
        Self { timer: Timer::from_seconds(duration, TimerMode::Repeating) }
    }
}


pub struct MovementPlugin {
    pub mode: MovementMode,
}

impl MovementPlugin {
    pub fn with_mode(mode: MovementMode) -> Self { Self { mode } }
}

impl Plugin for MovementPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(self.mode);
        app.insert_resource(organism_collision::OrganismCollisionTimer::default());

        // Wall + climb queries need both the heightmap (broad-phase
        // airborne early-out) and the triangle mesh (full SAT test). Both
        // resources land in the world asynchronously when the .glb finishes
        // decoding, so the system stays gated until they exist.
        app.add_systems(PostUpdate,
            apply_movement
                .before(TransformSystems::Propagate)
                .run_if(resource_exists::<HeightmapSampler>)
                .run_if(resource_exists::<WorldMesh>)
        );

        app.add_systems(PostUpdate, (
            apply_floor_collision.run_if(resource_exists::<HeightmapSampler>),
            apply_world_bounds.run_if(resource_exists::<HeightmapSampler>),
        ).chain().after(TransformSystems::Propagate));

        app.add_systems(Last, organism_collision::apply_organism_collision);

        match self.mode {
            MovementMode::TwoD => {
                app.add_systems(PostUpdate, random_2d_direction.before(apply_movement));
                app.add_systems(PostUpdate,
                    apply_gravity
                        .after(random_2d_direction)
                        .before(apply_movement));
            }
            MovementMode::ThreeD => {
                app.add_systems(PostUpdate, random_3d_direction.before(apply_movement));
            }
        }
    }
}

#[derive(Resource, PartialEq, Clone, Copy, Default)]
pub enum MovementMode {
    /// XZ-plane movement with gravity. Used by terrestrial heterotrophs.
    #[default]
    TwoD,
    /// Full XYZ movement, no gravity. Used by aquatic / aerial creatures.
    ThreeD,
}


// ── Random direction (photoautotrophs) ──────────────────────────────────────

fn random_2d_direction(
    time: Res<Time>,
    mut query: Query<(&mut Organism, &mut DirectionTimer), (With<OrganismRoot>, With<Photoautotroph>)>,
) {
    let dt = time.delta();

    for (mut organism, mut timer) in &mut query {
        timer.timer.tick(dt);
        if !timer.timer.just_finished() { continue; }

        let angle = rand::random::<f32>() * std::f32::consts::TAU;
        organism.movement_direction = Vec3::new(angle.cos(), 0.0, angle.sin());
        organism.movement_speed     = rand::random::<f32>() * 20.0;

        let next = MIN_DIRECTION_INTERVAL
            + rand::random::<f32>() * (MAX_DIRECTION_INTERVAL - MIN_DIRECTION_INTERVAL);
        timer.timer = Timer::from_seconds(next, TimerMode::Repeating);
    }
}

fn random_3d_direction(
    time: Res<Time>,
    mut query: Query<(&mut Organism, &mut DirectionTimer), (With<OrganismRoot>, With<Photoautotroph>)>,
) {
    let dt = time.delta();

    for (mut organism, mut timer) in &mut query {
        timer.timer.tick(dt);
        if !timer.timer.just_finished() { continue; }

        let theta = rand::random::<f32>() * std::f32::consts::TAU;
        let phi   = rand::random::<f32>() * std::f32::consts::PI;
        let dir   = Vec3::new(
            theta.cos() * phi.sin(),
            phi.cos(),
            theta.sin() * phi.sin(),
        );

        organism.movement_direction = dir;
        organism.movement_speed     = rand::random::<f32>() * 20.0;

        let next = MIN_DIRECTION_INTERVAL
            + rand::random::<f32>() * (MAX_DIRECTION_INTERVAL - MIN_DIRECTION_INTERVAL);
        timer.timer = Timer::from_seconds(next, TimerMode::Repeating);
    }
}


// ── Apply movement (wall collision via WorldMesh) ────────────────────────────

/// Step every organism in its commanded direction, except where a body-part
/// cell would intersect the world mesh.
///
/// Strategy:
///   1. Cheap broad-phase: if every cell of the organism is well above the
///      heightmap, skip the expensive triangle queries entirely.
///   2. For each grown cell of every body part, build a cell-AABB at the
///      post-step world position and test against `WorldMesh`. Any hit
///      blocks the XZ step; the maximum required climb height across all
///      hits decides whether the organism can scramble up.
///
/// The AABB's bottom face is inset by `RD_HALF_SIZE` so a settled cell
/// reads as resting on terrain rather than wall-blocked — without that
/// inset, a single settled cell triggers a permanent
/// climb-vs-gravity oscillation (visible terrain jitter).
fn apply_movement(
    time:            Res<Time>,
    world_mesh:      Res<WorldMesh>,
    heightmap:       Res<HeightmapSampler>,
    mut query:       Query<(&mut Transform, &mut Organism), With<OrganismRoot>>,
    mut tri_scratch: Local<Vec<u32>>,
) {
    let dt = time.delta_secs();
    let half_size = RD_HALF_SIZE;

    for (mut transform, mut organism) in &mut query {
        // Sessile organisms never move (plant-like body plan). Brain
        // systems may still write `movement_speed` / `movement_direction`
        // — those writes are simply ignored here. Floor-collision and
        // world-bounds clamping run in their own systems and continue to
        // apply, so a sessile cell still settles onto the heightmap.
        if organism.is_sessile { continue; }

        // Yaw the organism so its local +Z axis (defined as "front" by
        // the body-plan convention) points along the XZ projection of
        // its movement direction. Body never pitches or rolls — Y-axis
        // rotation only — which keeps mesh-vs-world AABB tests in the
        // axis-aligned regime they were tuned for. Snap (no slerp) is
        // intentional: brains can change heading at any tick rate, and
        // a smoothing layer would just lag the visual behind the
        // collision geometry that already follows the new heading.
        if organism.movement_speed > 0.0 {
            let dir = organism.movement_direction;
            let dir_xz_len_sq = dir.x * dir.x + dir.z * dir.z;
            if dir_xz_len_sq > 1e-6 {
                // For a Y-axis yaw θ: local +Z maps to world
                // (sin θ, 0, cos θ). Solving for θ that aligns this
                // with (dir.x, 0, dir.z) gives θ = atan2(dir.x, dir.z).
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
            continue;
        }

        let mut is_blocked   = false;
        let mut climb_needed = 0.0_f32;

        if move_vector.x != 0.0 || move_vector.z != 0.0 {
            // Iterate every cell of every body part. AABB-vs-mesh test per
            // cell — same scheme as before, but indexed via body parts so
            // multi-part organisms pay collision cost proportional to their
            // total grown cell count. We don't early-exit on first hit
            // because the climb-height decision needs the *worst* required
            // climb across all blocking cells.
            for body_part in &organism.body_parts {
                for cell in &body_part.cells {
                    let local_pos = body_part.local_offset + cell.local_pos;
                    let world_pos = transform.transform_point(local_pos);
                    let next_pos  = world_pos
                        + Vec3::new(move_vector.x, 0.0, move_vector.z);

                    let cell_min = Vec3::new(
                        next_pos.x - half_size,
                        next_pos.y,
                        next_pos.z - half_size,
                    );
                    let cell_max = next_pos + Vec3::splat(half_size);

                    if world_mesh.aabb_intersects_with(cell_min, cell_max, &mut tri_scratch) {
                        is_blocked = true;
                        let top = world_mesh
                            .max_y_in_xz_with(cell_min, cell_max, &mut tri_scratch)
                            .unwrap_or_else(|| heightmap.height_at(next_pos.x, next_pos.z));
                        let needed = top - (world_pos.y - half_size);
                        if needed > climb_needed { climb_needed = needed; }
                    }
                }
            }
        }

        organism.is_climbing = is_blocked;

        if is_blocked && climb_needed > 0.0 && climb_needed <= MAX_CLIMB_HEIGHT {
            let climb_amount = (organism.movement_speed * dt).min(climb_needed);
            transform.translation.y += climb_amount;
            organism.climb_energy_debt += climb_amount;
        } else if is_blocked {
            organism.is_climbing = false;
        } else {
            transform.translation.x += move_vector.x;
            transform.translation.z += move_vector.z;
        }
    }
}


// ── Gravity ──────────────────────────────────────────────────────────────────

fn apply_gravity(
    mode: Res<MovementMode>,
    time: Res<Time>,
    mut query: Query<(&mut Transform, &mut Organism), With<OrganismRoot>>,
) {
    if *mode == MovementMode::ThreeD { return; }

    let dt = time.delta_secs();

    for (mut transform, mut organism) in &mut query {
        if !organism.is_climbing {
            organism.velocity.y += -GRAVITY * dt;
            transform.translation.y += organism.velocity.y * dt;
        }
    }
}


// ── Floor collision ──────────────────────────────────────────────────────────

/// Lift the organism until no grown cell penetrates the heightmap. Body
/// parts are flat children of the root, so their world-space cell positions
/// are computed directly from the root transform (no per-body-part global
/// transform query is needed any more).
fn apply_floor_collision(
    heightmap: Res<HeightmapSampler>,
    mut query: Query<(&mut Transform, &mut Organism), With<OrganismRoot>>,
) {
    let half = RD_HALF_SIZE;
    for (mut transform, mut organism) in &mut query {
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
    // Operative bounds are the user-defined map area inset by
    // WORLD_SAFETY_MARGIN on each edge. The underlying terrain mesh
    // is normalised to be ≥ MapSize, so this clamp is always inside
    // the world geometry; the margin then keeps organisms from
    // wandering all the way to the visible edge (e.g. falling off
    // border cliffs or sticking to the world AABB).
    let min_x = WORLD_SAFETY_MARGIN;
    let max_x = (map_size.x - WORLD_SAFETY_MARGIN).max(WORLD_SAFETY_MARGIN);
    let min_z = WORLD_SAFETY_MARGIN;
    let max_z = (map_size.z - WORLD_SAFETY_MARGIN).max(WORLD_SAFETY_MARGIN);

    for mut transform in &mut query {
        transform.translation.x = transform.translation.x.clamp(min_x, max_x);
        transform.translation.z = transform.translation.z.clamp(min_z, max_z);
    }
}
