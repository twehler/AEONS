use bevy::prelude::*;
use crate::colony::*;
use crate::cell::GLOBAL_CELL_SIZE;
use std::collections::{HashMap, HashSet};

// Constants for organism collision
const ORGANISM_COLLISION_RADIUS: f32 = 5.0;
const CELL_COLLECTION_COLLISION_RADIUS: f32 = 5.0;
const COLLISION_RESPONSE_FORCE: f32 = 0.5; // How strongly they push apart

// Resource to control collision check interval
#[derive(Resource)]
pub struct OrganismCollisionTimer {
    pub timer: Timer,
}

impl Default for OrganismCollisionTimer {
    fn default() -> Self {
        Self {
            timer: Timer::from_seconds(0.2, TimerMode::Repeating), // Every 200ms
        }
    }
}

// Data for spatial hashing of organisms
#[derive(Clone)]
struct OrganismSpatialData {
    entity: Entity,
    position: Vec3,
    movement_direction: Vec3,
    movement_speed: f32,
}

// Data for cell-collection spatial hashing
#[derive(Clone)]
struct CellCollectionSpatialData {
    collection_id: CollectionId,
    organism_entity: Entity,
    position: Vec3,  // Position of the collection's starter cell in world space
}

// Get grid key for organism position
fn get_grid_key(pos: Vec3, cell_size: f32) -> [i32; 3] {
    [
        (pos.x / cell_size).floor() as i32,
        (pos.y / cell_size).floor() as i32,
        (pos.z / cell_size).floor() as i32,
    ]
}

// Get neighboring grid cells (3x3x3 cube)
fn get_neighbor_keys(key: [i32; 3]) -> Vec<[i32; 3]> {
    let mut neighbors = Vec::with_capacity(27);
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                neighbors.push([key[0] + dx, key[1] + dy, key[2] + dz]);
            }
        }
    }
    neighbors
}

// Build spatial hash of cell-collections for a specific organism
fn build_cell_collection_spatial_hash(
    organism: &Organism,
    transform: &Transform,
    organism_entity: Entity,
) -> HashMap<[i32; 3], Vec<CellCollectionSpatialData>> {
    let mut spatial_hash: HashMap<[i32; 3], Vec<CellCollectionSpatialData>> = HashMap::new();
    
    for (collection_id, collection) in &organism.collections {
        let world_pos = transform.transform_point(collection.starter_cell_position);
        let grid_key = get_grid_key(world_pos, CELL_COLLECTION_COLLISION_RADIUS);
        
        let data = CellCollectionSpatialData {
            collection_id: *collection_id,
            organism_entity,
            position: world_pos,
        };
        
        spatial_hash.entry(grid_key).or_insert_with(Vec::new).push(data);
    }
    
    spatial_hash
}

// Mark cell-collections that are within collision radius
fn mark_target_cell_collections(
    current_organism: &Organism,
    current_transform: &Transform,
    target_organism_entity: Entity,
    target_organism: &Organism,
    target_transform: &Transform,
    target_cell_collections: &mut HashSet<(Entity, CollectionId)>, // (organism_entity, collection_id)
) {
    // Build spatial hash for target organism's cell-collections
    let target_spatial_hash = build_cell_collection_spatial_hash(
        target_organism,
        target_transform,
        target_organism_entity,
    );
    
    // Check each collection in current organism against target's collections
    for (current_collection_id, current_collection) in &current_organism.collections {
        let current_world_pos = current_transform.transform_point(current_collection.starter_cell_position);
        let current_grid_key = get_grid_key(current_world_pos, CELL_COLLECTION_COLLISION_RADIUS);
        let neighbor_keys = get_neighbor_keys(current_grid_key);
        
        // Check neighboring grid cells in target's spatial hash
        for neighbor_key in neighbor_keys {
            if let Some(target_collections) = target_spatial_hash.get(&neighbor_key) {
                for target_collection in target_collections {
                    // Calculate distance between collection centers
                    let distance = current_world_pos.distance(target_collection.position);
                    
                    // If within collision radius, mark the target collection
                    if distance < CELL_COLLECTION_COLLISION_RADIUS {
                        target_cell_collections.insert((target_collection.organism_entity, target_collection.collection_id));
                    }
                }
            }
        }
    }
}


// Main collision system
pub fn apply_organism_collision(
    time: Res<Time>,
    mut timer: ResMut<OrganismCollisionTimer>,
    mut query: Query<(&mut Organism, &Transform, Entity), With<OrganismRoot>>,
) {
    // Tick the timer
    timer.timer.tick(time.delta());

    // Only run collision checks when timer finishes
    if !timer.timer.just_finished() {
        return;
    }

    // Build spatial hash of all organisms
    let mut spatial_hash: HashMap<[i32; 3], Vec<OrganismSpatialData>> = HashMap::new();

    for (organism, transform, entity) in query.iter() {
        let position = transform.translation;
        let grid_key = get_grid_key(position, ORGANISM_COLLISION_RADIUS);

        let data = OrganismSpatialData {
            entity,
            position,
            movement_direction: organism.movement_direction,
            movement_speed: organism.movement_speed,
        };

        spatial_hash.entry(grid_key).or_insert_with(Vec::new).push(data);
    }

    // Store accumulated pushes per organism
    let mut pushes: HashMap<Entity, Vec<Vec3>> = HashMap::new();
    let mut processed_pairs: HashSet<(Entity, Entity)> = HashSet::new();
    
    // Temporary marker sets
    let mut target_organisms: HashSet<Entity> = HashSet::new();
    let mut target_cell_collections: HashSet<(Entity, CollectionId)> = HashSet::new();

    // Detect collisions between organisms
    for (organism, transform, entity) in query.iter() {
        let position = transform.translation;
        let current_key = get_grid_key(position, ORGANISM_COLLISION_RADIUS);
        let neighbor_keys = get_neighbor_keys(current_key);

        // Check all potential neighboring organisms
        for neighbor_key in neighbor_keys {
            if let Some(organisms_in_cell) = spatial_hash.get(&neighbor_key) {
                for other in organisms_in_cell {
                    // Skip self
                    if other.entity == entity {
                        continue;
                    }

                    // Avoid processing the same pair twice
                    let pair_key = if entity < other.entity {
                        (entity, other.entity)
                    } else {
                        (other.entity, entity)
                    };

                    if processed_pairs.contains(&pair_key) {
                        continue;
                    }
                    processed_pairs.insert(pair_key);

                    // Calculate distance between organisms
                    let distance = position.distance(other.position);

                    // Check if within collision radius
                    if distance < ORGANISM_COLLISION_RADIUS {
                        // Mark both organisms as targets
                        target_organisms.insert(entity);
                        target_organisms.insert(other.entity);
                        
                        // Now check cell-collections for this pair
                        if let Ok((other_organism, other_transform, other_entity)) = query.get(other.entity) {
                            // Clear previous collection markers for this pair
                            let mut pair_collections: HashSet<(Entity, CollectionId)> = HashSet::new();
                            
                            // Mark target cell-collections in the other organism
                            mark_target_cell_collections(
                                organism,
                                transform,
                                other_entity,
                                other_organism,
                                other_transform,
                                &mut pair_collections,
                            );
                            
                            // Also mark target cell-collections in the current organism
                            mark_target_cell_collections(
                                other_organism,
                                other_transform,
                                entity,
                                organism,
                                transform,
                                &mut pair_collections,
                            );
                            
                            // Add to global target set
                            target_cell_collections.extend(pair_collections.clone());
                            
                            // NOW: Perform cell-level collision between marked collections
                            // We need to get the actual cell positions for marked collections
                            for (marked_entity, marked_collection_id) in pair_collections.iter() {
                                // Get the organism and transform for the marked collection
                                if let Ok((marked_organism, marked_transform, _)) = query.get(*marked_entity) {
                                    if let Some(marked_collection) = marked_organism.collections.get(marked_collection_id) {
                                        // Build spatial hash of cells in the marked collection
                                        let mut cell_spatial_hash: HashMap<[i32; 3], Vec<(Vec3, Entity)>> = HashMap::new();
                                        
                                        // Collect all active cells from this collection
                                        for (cell_local_pos, cell_type) in &marked_organism.active_cells {
                                            // Check if this cell belongs to the marked collection
                                            // We need to determine which collection this cell belongs to
                                            let cell_world_pos = marked_transform.transform_point(*cell_local_pos);
                                            let cell_grid_key = get_grid_key(cell_world_pos, GLOBAL_CELL_SIZE);
                                            
                                            // Find which collection this cell belongs to
                                            for (coll_id, coll) in &marked_organism.collections {
                                                if coll.starter_cell_position == *cell_local_pos {
                                                    if *coll_id == *marked_collection_id {
                                                        cell_spatial_hash.entry(cell_grid_key)
                                                            .or_insert_with(Vec::new)
                                                            .push((cell_world_pos, *marked_entity));
                                                    }
                                                    break;
                                                }
                                            }
                                        }
                                        
                                        // Now check cells from the other organism's marked collection
                                        // Determine which organism is the "other" one
                                        let other_entity = if *marked_entity == entity { other_entity } else { entity };
                                        let other_organism_data = if *marked_entity == entity { 
                                            (other_organism, other_transform, other_entity)
                                        } else { 
                                            (organism, transform, entity)
                                        };
                                        
                                        if let Some(other_collection) = other_organism_data.0.collections.get(marked_collection_id) {
                                            // Check each cell in the other collection against the spatial hash
                                            for (other_cell_local_pos, other_cell_type) in &other_organism_data.0.active_cells {
                                                // Check if this cell belongs to the marked collection in the other organism
                                                let other_cell_world_pos = other_organism_data.1.transform_point(*other_cell_local_pos);
                                                
                                                // Determine if this cell belongs to the other marked collection
                                                let mut is_in_target_collection = false;
                                                for (coll_id, coll) in &other_organism_data.0.collections {
                                                    if coll.starter_cell_position == *other_cell_local_pos {
                                                        if *coll_id == *marked_collection_id {
                                                            is_in_target_collection = true;
                                                        }
                                                        break;
                                                    }
                                                }
                                                
                                                if is_in_target_collection {
                                                    let other_cell_grid_key = get_grid_key(other_cell_world_pos, GLOBAL_CELL_SIZE);
                                                    let neighbor_cell_keys = get_neighbor_keys(other_cell_grid_key);
                                                    
                                                    // Check neighboring cells in the spatial hash
                                                    for neighbor_key in neighbor_cell_keys {
                                                        if let Some(cells) = cell_spatial_hash.get(&neighbor_key) {
                                                            for (cell_world_pos, cell_owner) in cells {
                                                                let distance = other_cell_world_pos.distance(*cell_world_pos);
                                                                
                                                                // Check if cells are colliding
                                                                if distance < GLOBAL_CELL_SIZE {
                                                                    // Calculate push direction
                                                                    let push_dir = (other_cell_world_pos - *cell_world_pos).normalize();
                                                                    
                                                                    // Calculate push strength (based on movement speed)
                                                                    let other_speed = other_organism_data.0.movement_speed;
                                                                    let current_speed = if *cell_owner == entity { 
                                                                        organism.movement_speed 
                                                                    } else { 
                                                                        other_organism_data.0.movement_speed 
                                                                    };
                                                                    
                                                                    let avg_speed = (other_speed + current_speed) / 2.0;
                                                                    let strength = 0.5 * (1.0 - distance / GLOBAL_CELL_SIZE);
                                                                    
                                                                    // Apply pushes to both organisms
                                                                    let push_self = push_dir * strength * avg_speed;
                                                                    let push_other = -push_dir * strength * avg_speed;
                                                                    
                                                                    pushes.entry(*cell_owner).or_insert_with(Vec::new).push(push_self);
                                                                    pushes.entry(other_entity).or_insert_with(Vec::new).push(push_other);
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        
                        // Calculate push direction (away from each other) - this is the organism-level push
                        let push_dir = (position - other.position).normalize();
                        let strength = COLLISION_RESPONSE_FORCE * (1.0 - distance / ORGANISM_COLLISION_RADIUS);
                        let push_self = push_dir * strength * organism.movement_speed;
                        let push_other = -push_dir * strength * other.movement_speed;
                        
                        // Record organism-level pushes
                        pushes.entry(entity).or_insert_with(Vec::new).push(push_self);
                        pushes.entry(other.entity).or_insert_with(Vec::new).push(push_other);
                    }
                }
            }
        }
    }

    // Apply accumulated pushes to movement direction
    apply_collision_pushes(query, &pushes);
}

fn apply_collision_pushes(
    mut query: Query<(&mut Organism, &Transform, Entity), With<OrganismRoot>>,
    pushes: &HashMap<Entity, Vec<Vec3>>,
) {
    for (mut organism, _, entity) in query.iter_mut() {
        if let Some(push_list) = pushes.get(&entity) {
            if !push_list.is_empty() {
                // Sum all pushes
                let mut total_push = Vec3::ZERO;
                for push in push_list {
                    total_push += *push;
                }

                // Average the pushes
                let avg_push = total_push / push_list.len() as f32;

                // Apply to movement direction
                let new_direction = organism.movement_direction + avg_push;

                // Normalize to maintain unit length
                if new_direction.length_squared() > 0.0 {
                    organism.movement_direction = new_direction.normalize();
                }
            }
        }
    }
}
