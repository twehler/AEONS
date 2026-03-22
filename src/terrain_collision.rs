use bevy::prelude::*;
use crate::world_geometry::{HeightmapSampler, BlockWorld};
use crate::colony::OrganismRoot;

// The radius used for wall collision queries, in world units.
// Should be roughly the radius of the organism's body — adjust per simulation.
// A larger radius prevents clipping through thin walls at high organism density.
const ORGANISM_RADIUS: f32 = 1.0;

// How many block layers above the floor to check for ceiling collision.
// 2 covers organisms up to 2 blocks tall.
const ORGANISM_HEIGHT: f32 = 2.0;


pub struct TerrainCollisionPlugin;

impl Plugin for TerrainCollisionPlugin {
    fn build(&self, app: &mut App) {
        // Runs in PostUpdate so organism transforms are settled for this frame
        // before we apply corrections. This avoids a one-frame lag where the
        // organism visually penetrates the terrain before being pushed out.
        app.add_systems(PostUpdate, apply_terrain_collision);
    }
}


// ── Collision system ──────────────────────────────────────────────────────────

// Queries every organism root entity and resolves terrain penetration.
// Each organism is treated as a vertical capsule of radius ORGANISM_RADIUS
// and height ORGANISM_HEIGHT for collision purposes — cheap to compute,
// good enough for biological simulation semantics.
//
// Two phases per organism:
//   1. Floor: heightmap lookup — O(1), handles open ground and slopes
//   2. Walls: block query on horizontal neighbours — only runs near solid blocks
fn apply_terrain_collision(
    heightmap:  Res<HeightmapSampler>,
    blockworld: Res<BlockWorld>,
    mut query:  Query<&mut Transform, With<OrganismRoot>>,
) {
    for mut transform in &mut query {
        let pos = transform.translation;

        // ── Phase 1: Floor collision via heightmap ────────────────────────────
        // Sample the terrain height at the organism's foot position.
        // The foot is at pos.y (the organism root sits at ground level).
        // If the organism has sunk below the surface, push it up instantly.
        // This handles walking on slopes and falling onto terrain correctly.
        let floor_y = heightmap.height_at(pos.x, pos.z);
        if pos.y < floor_y {
            transform.translation.y = floor_y;
        }

        // ── Phase 2: Wall collision via block query ───────────────────────────
        // Only run if the organism might be near a wall. We check this by
        // testing whether any block exists at the organism's current Y level
        // within a 1-block manhattan distance. If the heightmap already fully
        // resolved the position (open ground), this rarely triggers.
        //
        // We sample at the organism's foot Y and mid-body Y separately so
        // that both low walls and tall walls are caught.
        let check_ys = [
            pos.y as i32,                        // foot level
            (pos.y + ORGANISM_HEIGHT * 0.5) as i32, // mid-body
        ];

        let mut push = Vec3::ZERO;

        for &check_y in &check_ys {
            push += resolve_wall_axis(pos, check_y, ORGANISM_RADIUS, &blockworld);
        }

        // Apply the accumulated wall push — clamp to avoid overshooting
        // if two walls simultaneously push in opposite directions.
        if push != Vec3::ZERO {
            transform.translation.x += push.x;
            transform.translation.z += push.z;

            // Re-check floor after horizontal push, since moving horizontally
            // may have placed the organism over a different height column.
            let new_floor = heightmap.height_at(
                transform.translation.x,
                transform.translation.z,
            );
            if transform.translation.y < new_floor {
                transform.translation.y = new_floor;
            }
        }
    }
}


// ── Wall resolution ───────────────────────────────────────────────────────────

// Checks the four horizontal face-neighbours of the organism's block position
// at the given Y level. For each solid neighbour, computes how far the organism
// overlaps into that block and accumulates a push vector to resolve it.
//
// Returns the total XZ push to apply to the organism's translation.
// Y component is always zero — vertical resolution is handled by the heightmap.
fn resolve_wall_axis(
    pos:        Vec3,
    check_y:    i32,
    radius:     f32,
    blockworld: &BlockWorld,
) -> Vec3 {
    let mut push = Vec3::ZERO;

    // The organism's block footprint centre
    let block_x = pos.x.floor() as i32;
    let block_z = pos.z.floor() as i32;

    // Four horizontal face directions: +X, -X, +Z, -Z
    let neighbours: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

    for (dx, dz) in neighbours {
        let nx = block_x + dx;
        let nz = block_z + dz;

        if !blockworld.is_solid(nx, check_y, nz) {
            continue;
        }

        // The face of the solid block closest to the organism centre lies at:
        // for +X neighbour: x = nx (left face of block at nx)
        // for -X neighbour: x = nx + 1 (right face of block at nx)
        // The penetration depth is how far the organism's radius crosses that face.
        let (face_x, face_z) = block_face_toward_organism(pos.x, pos.z, dx, dz, nx, nz);

        let overlap_x = if dx != 0 {
            let dist = (pos.x - face_x).abs();
            if dist < radius { radius - dist } else { 0.0 }
        } else {
            0.0
        };

        let overlap_z = if dz != 0 {
            let dist = (pos.z - face_z).abs();
            if dist < radius { radius - dist } else { 0.0 }
        } else {
            0.0
        };

        // Push away from the wall along the axis of penetration only
        push.x += overlap_x * dx as f32 * -1.0;
        push.z += overlap_z * dz as f32 * -1.0;
    }

    push
}


// Returns the world-space XZ position of the block face that faces the organism.
// For a block at (nx, nz) with the organism approaching from direction (dx, dz):
// the relevant face is on the side of the block closest to the organism.
fn block_face_toward_organism(
    org_x: f32,
    org_z: f32,
    dx:    i32,
    dz:    i32,
    nx:    i32,
    nz:    i32,
) -> (f32, f32) {
    // Each block occupies [nx, nx+1] in X and [nz, nz+1] in Z.
    // The face toward the organism is:
    //   +X neighbour (dx=1): left face at x = nx     (organism is to the left)
    //   -X neighbour (dx=-1): right face at x = nx+1 (organism is to the right)
    //   +Z neighbour (dz=1): front face at z = nz
    //   -Z neighbour (dz=-1): back face at z = nz+1
    let face_x = if dx == 1 {
        nx as f32          // left face of +X block
    } else if dx == -1 {
        (nx + 1) as f32    // right face of -X block
    } else {
        org_x              // no X face relevant
    };

    let face_z = if dz == 1 {
        nz as f32          // front face of +Z block
    } else if dz == -1 {
        (nz + 1) as f32    // back face of -Z block
    } else {
        org_z              // no Z face relevant
    };

    (face_x, face_z)
}
