use bevy::prelude::*;
use crate::world_geometry::{HeightmapSampler, BlockWorld};
use crate::colony::*;
use crate::cell::*;
use std::collections::{HashMap, HashSet};

pub struct MovementPlugin;

impl Plugin for MovementPlugin {
    fn build(&self, app: &mut App) {
        // Runs in PostUpdate so organism transforms are settled for this frame
        // before we apply corrections. This avoids a one-frame lag where the
        // organism visually penetrates the terrain before being pushed out.
        app.add_systems(PostUpdate, (apply_movement, apply_terrain_collision).chain());
    }
}



// Movement speed in world units per second
const MOVEMENT_SPEED: f32 = 15.0;

// Maximum step height in world units (1.0 = one full block)
const MAX_STEP_HEIGHT: f32 = 0.5;


fn apply_movement(
    time: Res<Time>,
    mut query: Query<&mut Transform, With<OrganismRoot>>,
) {
    // Move in negative Z direction (forward in many 3D coordinate systems)
    let movement = Vec3::new(0.0, 0.0, MOVEMENT_SPEED * time.delta_secs());
    
    for mut transform in &mut query {
        transform.translation += movement;
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
    mut query:  Query<(&mut Transform, &Organism), With<OrganismRoot>>,
) {
    for (mut transform, organism) in &mut query {


        // ── Phase 1: Floor (per-cell)
        let mut max_penetration = 0.0;
        for (local_pos, cell_type) in &organism.active_cells {
            let world_pos = transform.transform_point(*local_pos);
            let floor_y = heightmap.height_at(world_pos.x, world_pos.z);
            let half_height = cell_type.size() / 2.0;
            let cell_bottom = world_pos.y - half_height;
            if cell_bottom < floor_y {
                let penetration = floor_y - cell_bottom;
                if penetration > max_penetration {
                    max_penetration = penetration;
                }
            }
        }
        if max_penetration > 0.0 {
            transform.translation.y += max_penetration;
        }


        // ── Phase 2: Walls (grid-based with step climbing) ─────────────────────
        let occupied_cells = get_occupied_grid_cells(&transform, &organism.active_cells);

        let mut push_x: f32 = 0.0;
        let mut push_z: f32 = 0.0;
        let mut x_count = 0;
        let mut z_count = 0;
        let mut step_up_height = 0.0;

        // Find the central cell for step detection
        if let Some((central_local_pos, central_cell_type)) = find_central_cell(&transform, organism) {
            let central_world_pos = transform.transform_point(central_local_pos);
            let central_half_height = central_cell_type.size() / 2.0;
            let central_bottom = central_world_pos.y - central_half_height;

            // Check the direction we're moving (currently only -Z)
            let move_dir_z = 1.0; // Moving in negative Z

            // Calculate the block in front of the central cell
            let central_grid = [
                central_world_pos.x.floor() as i32,
                central_world_pos.y.floor() as i32,
                central_world_pos.z.floor() as i32,
            ];

            let front_block_x = central_grid[0];
            let front_block_z = central_grid[2] + (move_dir_z as i32);

            // Check if there's a solid block directly in front
            if blockworld.is_solid(front_block_x, central_grid[1], front_block_z) {
                // Calculate step height needed
                let block_top = central_grid[1] as f32 + 1.0;
                let step_needed = block_top - central_bottom;

                // Check if it's climbable (step height <= max AND space above block is empty)
                if step_needed <= MAX_STEP_HEIGHT {
                    let above_block_x = front_block_x;
                    let above_block_y = central_grid[1] + 1;
                    let above_block_z = front_block_z;

                    // Check if the space above the block is empty
                    if !blockworld.is_solid(above_block_x, above_block_y, above_block_z) {
                        step_up_height = step_needed;
                    } else {
                        // Block above exists - treat as wall, push back
                        if move_dir_z == -1.0 {
                            push_z += 1.0; // Push backward
                            z_count += 1;
                        }
                    }
                } else {
                    // Too high to step - treat as wall, push back
                    if move_dir_z == 1.0 {
                        push_z += -1.0; // Push backward
                        z_count += 1;
                    }
                }
            }
        }

        // Apply step-up first (before horizontal push)
        if step_up_height > 0.0 {
            transform.translation.y += step_up_height;
        }

        // Now handle regular wall collisions for all cells (including central)
        // We need to recompute occupied cells after potential step-up
        let occupied_cells_after_step = get_occupied_grid_cells(&transform, &organism.active_cells);

        for grid_pos in &occupied_cells_after_step {
            let [x, y, z] = *grid_pos;


            // Check four cardinal directions or world borders
            if blockworld.is_solid(x + 1, y, z) || x < heightmap.min_x {
                push_x -= 1.0;
                x_count += 1;
            }
            if blockworld.is_solid(x - 1, y, z) || x > heightmap.min_x + heightmap.width as i32 {
                push_x += 1.0;
                x_count += 1;
            }
            if blockworld.is_solid(x, y, z + 1) || z < heightmap.min_z {
                push_z -= 1.0;
                z_count += 1;
            }
            if blockworld.is_solid(x, y, z - 1) || z > heightmap.min_x + heightmap.depth as i32 {
                push_z += 1.0;
                z_count += 1;
            }
        }

        // Apply push based on majority direction
        if x_count > 0 {
            let x_direction = push_x.signum();
            transform.translation.x += x_direction * 0.1;
        }
        if z_count > 0 {
            let z_direction = push_z.signum();
            transform.translation.z += z_direction * 0.1;
        }
    }
}

fn get_occupied_grid_cells(
    transform: &Transform,
    active_cells: &[(Vec3, CellType)],
) -> HashSet<[i32; 3]> {
    let mut occupied = HashSet::new();
    for (local_pos, _) in active_cells {
        let world_pos = transform.transform_point(*local_pos);
        let grid_key = [
            world_pos.x.floor() as i32,
            world_pos.y.floor() as i32,
            world_pos.z.floor() as i32,
        ];
        occupied.insert(grid_key);
    }
    occupied
}

// Helper to find the central cell (the one at the root collection's origin)
fn find_central_cell(
    transform: &Transform,
    organism: &Organism,
) -> Option<(Vec3, CellType)> {
    // Find the root collection (the one with no parent)
    let root_collection_id = organism.collections
        .iter()
        .find(|(_, coll)| coll.parent.is_none())
        .map(|(id, _)| *id)?;
    
    // Find the cell that belongs to the root collection and has offset [0,0,0]
    organism.active_cells.iter().find(|(local_pos, _)| {
        // Check if this cell is at the root collection's starter position
        if let Some(coll) = organism.collections.get(&root_collection_id) {
            *local_pos == coll.starter_cell_position
        } else {
            false
        }
    }).map(|(pos, cell_type)| (*pos, *cell_type))
}
