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

use crate::simulation_settings::{MIN_DIRECTION_INTERVAL, MAX_DIRECTION_INTERVAL};

use crate::simulation_settings::{GRAVITY, MAX_CLIMB_HEIGHT};

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
        organism.movement_speed     = rand::random::<f32>() * 20.0;

        let next = MIN_DIRECTION_INTERVAL
            + rand::random::<f32>() * (MAX_DIRECTION_INTERVAL - MIN_DIRECTION_INTERVAL);
        timer.timer = Timer::from_seconds(next, TimerMode::Repeating);
    }
}


// ── Apply movement (wall collision via WorldMesh) ────────────────────────────

/// Step each sliding organism in its commanded direction, blocked where a
/// body-part cell would intersect the world mesh.
///   1. Broad-phase: airborne (well above heightmap) ⇒ skip triangle queries.
///   2. Per cell, test a post-step AABB against `WorldMesh`; a hit blocks the
///      XZ step, and the max required climb height decides scramble-up.
///
/// The AABB bottom is inset by `RD_HALF_SIZE` so a settled cell reads as
/// resting rather than wall-blocked — without it, a settled cell triggers a
/// permanent climb-vs-gravity oscillation (terrain jitter).
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
        // Sessile organisms never move (brain writes ignored here); floor/
        // bounds clamping still apply via their own systems.
        if organism.is_sessile { continue; }
        // Limb organisms get their pose from Avian — don't fight the solver.
        // Only kinematic Sliding is processed here; LimbBasedWalking / Swimming
        // / Flying take the dynamic path and are skipped.
        if !organism.movement_mode.is_sliding() { continue; }

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
            continue;
        }

        let mut is_blocked   = false;
        let mut climb_needed = 0.0_f32;

        if move_vector.x != 0.0 || move_vector.z != 0.0 {
            // AABB-vs-mesh test per cell of every body part. No early-exit:
            // the climb decision needs the WORST required climb across all hits.
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
        // Kinematic-sliding only: LimbBasedWalking / Swimming / Flying are all
        // dynamic bodies and get gravity (or not) from rapier, not Organism.velocity.
        if !organism.movement_mode.is_sliding() { continue; }
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
        // Kinematic-sliding only (mirrors `apply_gravity`): limb/swimmer floor
        // handling lives in `rapier_setup::enforce_limb_floor_and_contacts`.
        // This system runs `.after(TransformSystems::Propagate)` — i.e. after
        // `sync_multibody_link_transforms` — so without this guard it would
        // clobber Rapier's authoritative pose for any penetrating dynamic body.
        if !organism.movement_mode.is_sliding() { continue; }
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
    // Phototrophs are excluded: they're the prey food source, kept alive on the
    // terrain by `maintain_prey_near_herbivores` (the loaded field sits above the
    // flat world and would otherwise gravity-fall through to the kill-floor and
    // vanish, starving the herbivores of anything to eat).
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
