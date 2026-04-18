use bevy::prelude::*;
use crate::colony::*;
use crate::cell::{CellType, GLOBAL_CELL_SIZE, MeshCell, merge_organism_mesh, OrganismMesh};
use crate::viewport_settings::ShowGizmo;
use std::collections::HashMap;
use rand::prelude::*;

// ── Constants ────────────────────────────────────────────────────────────────

const REPRODUCTION_CHECK_INTERVAL: f32 = 2.0;
const REPRODUCTION_ENERGY_FRACTION: f32 = 0.8;  // must have 80% of max energy
const OFFSPRING_ENERGY_FRACTION: f32 = 0.4;     // each offspring gets 40%
const MAX_ENERGY_PER_CELL: f32 = 10.0;
const SPAWN_OFFSET: f32 = 5.0;                  // how far apart offspring spawn

// ── Evolution Limits & Biases ────────────────────────────────────────────────
const MUTATION_RATE: f32 = 0.85;                // 85% chance for a mutation event to occur per birth
const GROWTH_BIAS: f32 = 0.65;                  // 65% chance to grow vs 35% chance to shrink/change
const MAX_COLLECTIONS: usize = 64;              // Maximum 64 cell collections
const MAX_CELLS_PER_COLLECTION: usize = 32;     // Maximum 32 cells per collection

// ── Timer resource ───────────────────────────────────────────────────────────

#[derive(Resource)]
pub struct ReproductionTimer {
    pub timer: Timer,
}

impl Default for ReproductionTimer {
    fn default() -> Self {
        Self {
            timer: Timer::from_seconds(REPRODUCTION_CHECK_INTERVAL, TimerMode::Repeating),
        }
    }
}

// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct ReproductionPlugin;

impl Plugin for ReproductionPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ReproductionTimer::default());
        app.add_systems(Update, reproduction_system);
    }
}

// ── Mutation ─────────────────────────────────────────────────────────────────

const ALL_CELL_TYPES: [CellType; 11] = [
    CellType::BlueCell, CellType::RedCell, CellType::GreenCell,
    CellType::YellowCell, CellType::OrangeCell, CellType::LightBlueCell,
    CellType::PhotoCell, CellType::GutCell, CellType::HardCell,
    CellType::FootCell, CellType::FinCell,
];

fn mutate_ocg(ocg: &[OcgEntry], collections: &HashMap<CollectionId, CellCollection>) -> (Vec<OcgEntry>, HashMap<CollectionId, CellCollection>) {
    let mut rng = rand::rng();
    let mut new_ocg = ocg.to_vec();
    let mut new_collections = collections.clone();

    // 1. Check if a mutation event happens at all
    if rng.random::<f32>() > MUTATION_RATE {
        return (new_ocg, new_collections);
    }

    // 2. Decide if the mutation adds to the organism (Growth) or modifies/removes (Shrinkage)
    if rng.random::<f32>() < GROWTH_BIAS {
        // --- GROWTH EVENT ---
        
        // Count how many cells are in each collection
        let mut coll_counts = HashMap::new();
        for entry in &new_ocg {
            *coll_counts.entry(entry.collection_id).or_insert(0) += 1;
        }

        // Pick a random existing collection to branch off of
        if let Some(&coll_id) = new_collections.keys().choose(&mut rng) {
            let count = coll_counts.get(&coll_id).copied().unwrap_or(0);
            
            // If the collection has space, add a cell to it
            if count < MAX_CELLS_PER_COLLECTION {
                let parents: Vec<_> = new_ocg.iter().filter(|e| e.collection_id == coll_id).collect();
                if let Some(parent) = parents.choose(&mut rng) {
                    let directions = [
                        Vec3::new(GLOBAL_CELL_SIZE, 0.0, 0.0),
                        Vec3::new(-GLOBAL_CELL_SIZE, 0.0, 0.0),
                        Vec3::new(0.0, GLOBAL_CELL_SIZE, 0.0),
                        Vec3::new(0.0, -GLOBAL_CELL_SIZE, 0.0),
                        Vec3::new(0.0, 0.0, GLOBAL_CELL_SIZE),
                        Vec3::new(0.0, 0.0, -GLOBAL_CELL_SIZE),
                    ];
                    let dir = directions.choose(&mut rng).unwrap();
                    let new_type = ALL_CELL_TYPES[rng.random_range(0..ALL_CELL_TYPES.len())];
                    
                    new_ocg.push(OcgEntry {
                        collection_id: coll_id,
                        cell_type: new_type,
                        offset: parent.offset + *dir,
                    });
                }
            } 
            // If the collection is full, try to add a brand new collection instead
            else if new_collections.len() < MAX_COLLECTIONS {
                // Generate a safe new ID
                let next_id_val = new_collections.keys().map(|id| id.0).max().unwrap_or(0) + 1;
                let new_id = CollectionId(next_id_val);
                
                new_collections.insert(new_id, CellCollection {
                    starter_cell_position: Vec3::new(
                        rng.random_range(-2.0..2.0), 
                        rng.random_range(-2.0..2.0), 
                        rng.random_range(-2.0..2.0)
                    ),
                    parent: Some(coll_id),
                });
                
                new_ocg.push(OcgEntry {
                    collection_id: new_id,
                    cell_type: ALL_CELL_TYPES[rng.random_range(0..ALL_CELL_TYPES.len())],
                    offset: Vec3::ZERO,
                });
            }
        }
    } else {
        // --- SHRINKAGE / MODIFICATION EVENT ---
        if new_ocg.len() > 1 {
            let idx = rng.random_range(0..new_ocg.len());
            // 50% chance to delete the cell entirely, 50% chance to just change its type
            if rng.random::<bool>() {
                new_ocg.remove(idx);
            } else {
                new_ocg[idx].cell_type = ALL_CELL_TYPES[rng.random_range(0..ALL_CELL_TYPES.len())];
            }
        }
    }

    // Safety fallback: Ensure at least 1 cell survives
    if new_ocg.is_empty() && !ocg.is_empty() {
        new_ocg.push(ocg[0].clone());
    }

    (new_ocg, new_collections)
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn compute_floor_cells(ocg: &[OcgEntry], collections: &HashMap<CollectionId, CellCollection>) -> Vec<(CollectionId, Vec3)> {
    let mut min_y_per_collection: HashMap<CollectionId, (Vec3, f32)> = HashMap::new();
    for entry in ocg {
        if let Some(coll) = collections.get(&entry.collection_id) {
            let world_y = coll.starter_cell_position.y + entry.offset.y;
            match min_y_per_collection.entry(entry.collection_id) {
                std::collections::hash_map::Entry::Vacant(e) => { e.insert((entry.offset, world_y)); }
                std::collections::hash_map::Entry::Occupied(mut e) => {
                    if world_y < e.get().1 { e.insert((entry.offset, world_y)); }
                }
            }
        }
    }
    min_y_per_collection.into_iter().map(|(id, (offset, _))| (id, offset)).collect()
}

fn compute_bounding_radius(ocg: &[OcgEntry], collections: &HashMap<CollectionId, CellCollection>) -> f32 {
    ocg.iter().map(|entry| {
        if let Some(coll) = collections.get(&entry.collection_id) {
            (coll.starter_cell_position + entry.offset).length()
        } else {
            0.0
        }
    }).fold(0.0f32, f32::max) + GLOBAL_CELL_SIZE
}

fn build_mesh_for_ocg(
    ocg: &[OcgEntry],
    collections: &HashMap<CollectionId, CellCollection>,
    meshes: &mut ResMut<Assets<Mesh>>,
) -> Handle<Mesh> {
    let mesh_cells: Vec<MeshCell> = ocg.iter().filter_map(|entry| {
        collections.get(&entry.collection_id).map(|coll| {
            MeshCell {
                mesh_space_pos: coll.starter_cell_position + entry.offset,
                cell_type: entry.cell_type,
            }
        })
    }).collect();
    meshes.add(merge_organism_mesh(mesh_cells))
}

// ── System ───────────────────────────────────────────────────────────────────

fn reproduction_system(
    time: Res<Time>,
    mut timer: ResMut<ReproductionTimer>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut query: Query<(Entity, &mut Organism, &Transform), With<OrganismRoot>>,
    pop_cap: Res<crate::colony::PopulationCap>,
) {
    timer.timer.tick(time.delta());
    if !timer.timer.just_finished() {
        return;
    }

    let current_pop = query.iter().count();
    if current_pop >= pop_cap.max {
        return;
    }
    let spawn_budget = pop_cap.max - current_pop;

    let mut births: Vec<(Vec3, Vec<OcgEntry>, HashMap<CollectionId, CellCollection>, f32)> = Vec::new();

    for (_entity, mut organism, transform) in &mut query {
        let cell_count = organism.grown_cell_count as f32;
        if cell_count < 1.0 { continue; }

        let max_energy = cell_count * MAX_ENERGY_PER_CELL;
        let threshold = max_energy * REPRODUCTION_ENERGY_FRACTION;

        if organism.energy >= threshold {
            // Parent loses energy for reproduction
            let offspring_energy = max_energy * OFFSPRING_ENERGY_FRACTION;
            organism.energy -= offspring_energy * 2.0; // energy for both offspring (one replaces parent conceptually)

            // Mutate OCG for offspring
            let (child_ocg, child_collections) = mutate_ocg(&organism.ocg, &organism.collections);

            // Spawn offset perpendicular to movement direction
            let perp = Vec3::new(-organism.movement_direction.z, 0.0, organism.movement_direction.x);
            let spawn_pos = transform.translation + perp * SPAWN_OFFSET;

            births.push((spawn_pos, child_ocg, child_collections, offspring_energy));
            if births.len() >= spawn_budget {
                break;
            }
        }
    }

    births.truncate(spawn_budget);

    // Spawn offspring outside the query borrow
    let shared_material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        ..default()
    });

    for (pos, ocg, collections, energy) in births {
        let mut rng = rand::rng();
        let angle = rng.random::<f32>() * std::f32::consts::TAU;
        let direction = Vec3::new(angle.cos(), 0.0, angle.sin()).normalize();
        let speed = rng.random::<f32>() * 20.0;

        let floor_cells = compute_floor_cells(&ocg, &collections);
        let bounding_radius = compute_bounding_radius(&ocg, &collections);

        let active_cells: Vec<(Vec3, CellType)> = ocg.iter().filter_map(|entry| {
            collections.get(&entry.collection_id).map(|coll| {
                (coll.starter_cell_position + entry.offset, entry.cell_type)
            })
        }).collect();

        let grown_cell_count = ocg.len();
        let mesh_handle = build_mesh_for_ocg(&ocg, &collections, &mut meshes);

        // Spawn joint entities
        let joint_entities: HashMap<CollectionId, Entity> = collections.iter()
            .map(|(&id, coll)| {
                let entity = commands.spawn((
                    Transform::from_translation(coll.starter_cell_position),
                    Visibility::Visible,
                )).id();
                (id, entity)
            })
            .collect();

        let organism = Organism {
            collections: collections.clone(),
            pos,
            energy,
            growth_speed: 1.0,
            adult: false,
            ocg,
            joint_entities: joint_entities.clone(),
            active_cells,
            grown_cell_count,
            is_climbing: false,
            movement_speed: speed,
            movement_direction: direction,
            velocity: Vec3::ZERO,
            floor_cells,
            bounding_radius,
        };

        let random_interval = 1.0 + rng.random::<f32>() * 9.0;

        let organism_root = commands.spawn((
            Transform::from_translation(pos),
            Visibility::Visible,
            OrganismRoot,
            organism,
            DirectionTimer::new(random_interval),
        )).id();

        // Wire joint hierarchy
        for (&id, coll) in &collections {
            let joint_entity = joint_entities[&id];
            match coll.parent {
                None => { commands.entity(organism_root).add_child(joint_entity); }
                Some(parent_id) => {
                    if let Some(&parent_entity) = joint_entities.get(&parent_id) {
                        commands.entity(parent_entity).add_child(joint_entity);
                    }
                }
            }
        }

        // Spawn mesh
        let mesh_entity = commands.spawn((
            Mesh3d(mesh_handle),
            MeshMaterial3d(shared_material.clone()),
            Transform::IDENTITY,
            OrganismMesh,
            ShowGizmo,
        )).id();
        commands.entity(organism_root).add_child(mesh_entity);
    }
}
