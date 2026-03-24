use bevy::prelude::*;
use crate::world_geometry::{HeightmapSampler, BlockWorld};
use crate::colony::*;
use crate::cell::*;
use std::collections::HashMap;

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

/* OLD
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

        info!("pos={:.1?} floor_y={:.1} min_x={} min_z={} width={} depth={}",
            pos, floor_y, heightmap.min_x, heightmap.min_z,
            heightmap.width, heightmap.depth);


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
*/

pub fn apply_terrain_collision(
    heightmap: Res<HeightmapSampler>,
    blockworld: Res<BlockWorld>,
    mut root_query: Query<(&mut Transform, &OrganismRoot)>,
    mut joint_query: Query<&mut Transform, Without<OrganismRoot>>,
    organisms: Query<&Organism>,
) {
    // Get all organism roots with their transforms
    for (mut root_transform, root_entity) in root_query.iter_mut() {
        // Find the Organism component associated with this root
        let organism = match organisms.get(root_entity) {
            Ok(org) => org,
            Err(_) => continue,
        };

        let mut max_penetration = 0.0f32;
        let mut wall_push = Vec3::ZERO;
        
        // Track which cells need correction
        let mut cell_corrections: Vec<(Vec3, f32)> = Vec::new();

        // Get the collection transforms relative to root
        let mut collection_transforms: HashMap<CollectionId, Transform> = HashMap::new();
        
        // Build transforms for each collection by walking the hierarchy
        for (&collection_id, collection) in &organism.collections {
            let mut transform = Transform::from_translation(collection.starter_cell_position);
            
            // Apply parent transforms if any
            let mut current_parent = collection.parent;
            while let Some(parent_id) = current_parent {
                if let Some(parent_collection) = organism.collections.get(&parent_id) {
                    transform.translation += parent_collection.starter_cell_position;
                    current_parent = parent_collection.parent;
                } else {
                    break;
                }
            }
            
            collection_transforms.insert(collection_id, transform);
        }

        // Iterate through every cell in the OCG
        for entry in &organism.ocg {
            // Get the collection transform for this cell
            if let Some(collection_transform) = collection_transforms.get(&entry.collection_id) {
                // Calculate cell's local position relative to organism root
                let cell_local_pos = collection_transform.translation + Vec3::from(entry.offset);
                
                // Transform to world space using root transform
                let cell_world_pos = root_transform.transform_point(cell_local_pos);
                
                // Check heightmap collision (floor/ceiling)
                let floor_y = heightmap.height_at(cell_world_pos.x, cell_world_pos.z);
                let cell_bottom = cell_world_pos.y;
                let cell_top = cell_bottom + 1.0; // Assume cell height is 1 unit
                
                // Check if cell is penetrating terrain
                if cell_bottom < floor_y {
                    // Cell is underground - calculate how much to lift
                    let penetration = floor_y - cell_bottom;
                    if penetration > max_penetration {
                        max_penetration = penetration;
                    }
                }
                
                // Check wall collision for this cell
                let wall_correction = resolve_cell_wall_collision(cell_world_pos, 0.5, &blockworld);
                if wall_correction != Vec3::ZERO {
                    wall_push += wall_correction;
                    cell_corrections.push((cell_world_pos, wall_correction.length()));
                }
            }
        }

        // Apply vertical correction (floor/ceiling)
        if max_penetration > 0.0 {
            root_transform.translation.y += max_penetration;
            
            // Re-check heightmap after vertical movement to ensure we're exactly on surface
            let new_pos = root_transform.translation;
            let floor_y = heightmap.height_at(new_pos.x, new_pos.z);
            if root_transform.translation.y < floor_y {
                root_transform.translation.y = floor_y;
            }
        }

        // Apply horizontal correction (walls)
        if wall_push != Vec3::ZERO {
            // Average the wall pushes to get smooth movement
            let avg_push = if !cell_corrections.is_empty() {
                let total_weight: f32 = cell_corrections.iter().map(|(_, weight)| weight).sum();
                let weighted_push = cell_corrections.iter()
                    .fold(Vec3::ZERO, |acc, (pos, weight)| acc + *pos * *weight);
                (weighted_push / total_weight).normalize() * wall_push.length()
            } else {
                wall_push
            };
            
            // Apply horizontal movement
            root_transform.translation.x += avg_push.x;
            root_transform.translation.z += avg_push.z;
            
            // Re-check floor after horizontal movement
            let new_pos = root_transform.translation;
            let floor_y = heightmap.height_at(new_pos.x, new_pos.z);
            if root_transform.translation.y < floor_y {
                root_transform.translation.y = floor_y;
            }
        }
    }
}

// Helper function to resolve wall collisions for individual cells
fn resolve_cell_wall_collision(
    cell_pos: Vec3,
    radius: f32,
    blockworld: &BlockWorld,
) -> Vec3 {
    let mut push = Vec3::ZERO;
    
    // Get the cell's footprint block coordinates
    let block_x = (cell_pos.x - radius).floor() as i32;
    let block_z = (cell_pos.z - radius).floor() as i32;
    let block_x_max = (cell_pos.x + radius).floor() as i32;
    let block_z_max = (cell_pos.z + radius).floor() as i32;
    
    // Check all blocks that the cell might intersect with
    for bx in block_x..=block_x_max {
        for bz in block_z..=block_z_max {
            // Check at multiple Y levels (cell's Y and surrounding Y levels)
            let cell_y_level = cell_pos.y.floor() as i32;
            for dy in -1..=1 {
                let check_y = cell_y_level + dy;
                
                if blockworld.is_solid(bx, check_y, bz) {
                    // Calculate push from this block
                    let block_min_x = bx as f32;
                    let block_max_x = (bx + 1) as f32;
                    let block_min_z = bz as f32;
                    let block_max_z = (bz + 1) as f32;
                    
                    // Check X-axis penetration
                    if cell_pos.x + radius > block_min_x && cell_pos.x - radius < block_max_x {
                        let overlap_x = if cell_pos.x < block_min_x {
                            // Cell is to the left of block
                            (cell_pos.x + radius) - block_min_x
                        } else if cell_pos.x > block_max_x {
                            // Cell is to the right of block
                            block_max_x - (cell_pos.x - radius)
                        } else {
                            // Cell center is inside block horizontally
                            let left_overlap = (cell_pos.x + radius) - block_min_x;
                            let right_overlap = block_max_x - (cell_pos.x - radius);
                            if left_overlap < right_overlap {
                                -left_overlap
                            } else {
                                right_overlap
                            }
                        };
                        
                        // Check Z-axis penetration
                        let overlap_z = if cell_pos.z < block_min_z {
                            (cell_pos.z + radius) - block_min_z
                        } else if cell_pos.z > block_max_z {
                            block_max_z - (cell_pos.z - radius)
                        } else {
                            let front_overlap = (cell_pos.z + radius) - block_min_z;
                            let back_overlap = block_max_z - (cell_pos.z - radius);
                            if front_overlap < back_overlap {
                                -front_overlap
                            } else {
                                back_overlap
                            }
                        };
                        
                        // Apply push along the axis with greater penetration
                        if overlap_x.abs() > overlap_z.abs() && overlap_x > 0.0 {
                            let direction = if overlap_x > 0.0 {
                                if cell_pos.x < block_min_x { -1.0 } else { 1.0 }
                            } else {
                                0.0
                            };
                            push.x += overlap_x * direction;
                        } else if overlap_z > 0.0 {
                            let direction = if overlap_z > 0.0 {
                                if cell_pos.z < block_min_z { -1.0 } else { 1.0 }
                            } else {
                                0.0
                            };
                            push.z += overlap_z * direction;
                        }
                    }
                }
            }
        }
    }
    
    push
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
