use crate::cell::*;
use crate::viewport_settings::*;
use crate::world_geometry::HeightmapSampler;
use bevy::prelude::*;
use std::collections::HashMap;
use rand::RngExt;

pub struct ColonyPlugin;

impl Plugin for ColonyPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(PopulationCap::default());
        app.add_systems(Update, spawn_colony);
    }
}

// ── Core data structures ─────────────────────────────────────────────────────

pub struct Colony {
    organisms: Vec<Organism>,
}

#[derive(Component, Clone)]
pub struct Organism {
    pub collections: HashMap<CollectionId, CellCollection>,
    pub pos:         Vec3,
    pub energy: f32,
    pub growth_speed: f32,
    pub adult: bool,

    pub ocg: Vec<OcgEntry>,
    pub joint_entities: HashMap<CollectionId, Entity>,
    pub active_cells: Vec<(Vec3, CellType)>,
    pub grown_cell_count: usize,
    pub is_climbing: bool,
    pub movement_speed: f32,
    pub movement_direction: Vec3,
    pub velocity: Vec3,
    pub floor_cells: Vec<(CollectionId, Vec3)>,
    pub bounding_radius: f32,
    
    // NEW: Rotation fields
    pub target_rotation: Quat,
    pub rotation_speed: f32,
}

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub struct CollectionId(pub u32);

#[derive(Clone, Debug)]
pub struct OcgEntry {
    pub collection_id: CollectionId,
    pub cell_type:     CellType,
    pub offset: Vec3,
}

#[derive(Clone)]
pub struct CellCollection {
    pub starter_cell_position: Vec3,
    pub parent: Option<CollectionId>,
}

#[derive(Component)]
pub struct OrganismRoot;

#[derive(Component)]
pub struct DirectionTimer {
    pub timer: Timer,
}

impl DirectionTimer {
    pub fn new(interval: f32) -> Self {
        Self {
            timer: Timer::from_seconds(interval, TimerMode::Repeating),
        }
    }
}

// NEW: Timer specifically for choosing new rotations
#[derive(Component)]
pub struct RotationTimer {
    pub timer: Timer,
}

impl RotationTimer {
    pub fn new(interval: f32) -> Self {
        Self {
            timer: Timer::from_seconds(interval, TimerMode::Repeating),
        }
    }
}

#[derive(Resource)]
pub struct PopulationCap {
    pub max: usize,
}

impl Default for PopulationCap {
    fn default() -> Self {
        Self { max: 1500 }
    }
}

// ── Plugin entry point ───────────────────────────────────────────────────────

fn spawn_colony(
    mut commands:      Commands,
    mut meshes:        ResMut<Assets<Mesh>>,
    mut materials:     ResMut<Assets<StandardMaterial>>,
    heightmap:         Res<HeightmapSampler>,
    mut spawned:       Local<bool>,
) {
    if *spawned { return; }
    *spawned = true;

    let shared_material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        ..default()
    });

    let mut rng = rand::rng();

    // ── Template 1: Flat Raft (Producer) — 350 organisms ─────────────────
    let raft_cc = CollectionId(1);
    let mut raft_collections = HashMap::new();
    raft_collections.insert(raft_cc, CellCollection { starter_cell_position: Vec3::ZERO, parent: None });
    let raft_ocg = vec![
        OcgEntry { collection_id: raft_cc, cell_type: CellType::PhotoCell, offset: Vec3::new(0.0, 0.0, 0.0) },
        OcgEntry { collection_id: raft_cc, cell_type: CellType::PhotoCell, offset: Vec3::new(GLOBAL_CELL_SIZE, 0.0, 0.0) },
        OcgEntry { collection_id: raft_cc, cell_type: CellType::PhotoCell, offset: Vec3::new(-GLOBAL_CELL_SIZE, 0.0, 0.0) },
        OcgEntry { collection_id: raft_cc, cell_type: CellType::PhotoCell, offset: Vec3::new(0.0, 0.0, GLOBAL_CELL_SIZE) },
        OcgEntry { collection_id: raft_cc, cell_type: CellType::RedCell, offset: Vec3::new(GLOBAL_CELL_SIZE, 0.0, GLOBAL_CELL_SIZE) },
        OcgEntry { collection_id: raft_cc, cell_type: CellType::RedCell, offset: Vec3::new(-GLOBAL_CELL_SIZE, 0.0, GLOBAL_CELL_SIZE) },
        OcgEntry { collection_id: raft_cc, cell_type: CellType::RedCell, offset: Vec3::new(GLOBAL_CELL_SIZE, 0.0, -GLOBAL_CELL_SIZE) },
        OcgEntry { collection_id: raft_cc, cell_type: CellType::RedCell, offset: Vec3::new(-GLOBAL_CELL_SIZE, 0.0, -GLOBAL_CELL_SIZE) },
    ];

    // ── Template 2: Drifter (Aquatic Producer) — 300 organisms ───────────
    let drift_cc = CollectionId(1);
    let mut drift_collections = HashMap::new();
    drift_collections.insert(drift_cc, CellCollection { starter_cell_position: Vec3::ZERO, parent: None });
    let drift_ocg = vec![
        OcgEntry { collection_id: drift_cc, cell_type: CellType::PhotoCell, offset: Vec3::new(0.0, 0.0, 0.0) },
        OcgEntry { collection_id: drift_cc, cell_type: CellType::PhotoCell, offset: Vec3::new(GLOBAL_CELL_SIZE, 0.0, 0.0) },
        OcgEntry { collection_id: drift_cc, cell_type: CellType::PhotoCell, offset: Vec3::new(-GLOBAL_CELL_SIZE, 0.0, 0.0) },
        OcgEntry { collection_id: drift_cc, cell_type: CellType::FinCell, offset: Vec3::new(0.0, 0.0, GLOBAL_CELL_SIZE) },
        OcgEntry { collection_id: drift_cc, cell_type: CellType::FinCell, offset: Vec3::new(0.0, 0.0, -GLOBAL_CELL_SIZE) },
        OcgEntry { collection_id: drift_cc, cell_type: CellType::YellowCell, offset: Vec3::new(0.0, -GLOBAL_CELL_SIZE, 0.0) },
        OcgEntry { collection_id: drift_cc, cell_type: CellType::YellowCell, offset: Vec3::new(0.0, -2.0 * GLOBAL_CELL_SIZE, 0.0) },
    ];

    // ── Template 3: Photospine (Vertical Producer) — 150 organisms ───────
    let spine_cc = CollectionId(1);
    let mut spine_collections = HashMap::new();
    spine_collections.insert(spine_cc, CellCollection { starter_cell_position: Vec3::ZERO, parent: None });
    let spine_ocg = vec![
        OcgEntry { collection_id: spine_cc, cell_type: CellType::PhotoCell, offset: Vec3::new(0.0, 2.0 * GLOBAL_CELL_SIZE, 0.0) },
        OcgEntry { collection_id: spine_cc, cell_type: CellType::PhotoCell, offset: Vec3::new(GLOBAL_CELL_SIZE, 2.0 * GLOBAL_CELL_SIZE, 0.0) },
        OcgEntry { collection_id: spine_cc, cell_type: CellType::PhotoCell, offset: Vec3::new(-GLOBAL_CELL_SIZE, 2.0 * GLOBAL_CELL_SIZE, 0.0) },
        OcgEntry { collection_id: spine_cc, cell_type: CellType::YellowCell, offset: Vec3::new(0.0, GLOBAL_CELL_SIZE, 0.0) },
        OcgEntry { collection_id: spine_cc, cell_type: CellType::YellowCell, offset: Vec3::new(0.0, 0.0, 0.0) },
        OcgEntry { collection_id: spine_cc, cell_type: CellType::FinCell, offset: Vec3::new(GLOBAL_CELL_SIZE, 0.0, 0.0) },
        OcgEntry { collection_id: spine_cc, cell_type: CellType::FinCell, offset: Vec3::new(-GLOBAL_CELL_SIZE, 0.0, 0.0) },
        OcgEntry { collection_id: spine_cc, cell_type: CellType::OrangeCell, offset: Vec3::new(0.0, -GLOBAL_CELL_SIZE, 0.0) },
        OcgEntry { collection_id: spine_cc, cell_type: CellType::OrangeCell, offset: Vec3::new(0.0, -2.0 * GLOBAL_CELL_SIZE, 0.0) },
    ];

    // ── Template 4: Crawler (Pre-predator) — 100 organisms ───────────────
    let crawl_cc = CollectionId(1);
    let mut crawl_collections = HashMap::new();
    crawl_collections.insert(crawl_cc, CellCollection { starter_cell_position: Vec3::ZERO, parent: None });
    let crawl_ocg = vec![
        OcgEntry { collection_id: crawl_cc, cell_type: CellType::PhotoCell, offset: Vec3::new(0.0, GLOBAL_CELL_SIZE, 0.0) },
        OcgEntry { collection_id: crawl_cc, cell_type: CellType::PhotoCell, offset: Vec3::new(GLOBAL_CELL_SIZE, GLOBAL_CELL_SIZE, 0.0) },
        OcgEntry { collection_id: crawl_cc, cell_type: CellType::PhotoCell, offset: Vec3::new(-GLOBAL_CELL_SIZE, GLOBAL_CELL_SIZE, 0.0) },
        OcgEntry { collection_id: crawl_cc, cell_type: CellType::GutCell, offset: Vec3::new(0.0, 0.0, GLOBAL_CELL_SIZE) },
        OcgEntry { collection_id: crawl_cc, cell_type: CellType::GutCell, offset: Vec3::new(0.0, 0.0, 2.0 * GLOBAL_CELL_SIZE) },
        OcgEntry { collection_id: crawl_cc, cell_type: CellType::FootCell, offset: Vec3::new(GLOBAL_CELL_SIZE, -GLOBAL_CELL_SIZE, 0.0) },
        OcgEntry { collection_id: crawl_cc, cell_type: CellType::FootCell, offset: Vec3::new(-GLOBAL_CELL_SIZE, -GLOBAL_CELL_SIZE, 0.0) },
        OcgEntry { collection_id: crawl_cc, cell_type: CellType::LightBlueCell, offset: Vec3::new(0.0, 0.0, 0.0) },
        OcgEntry { collection_id: crawl_cc, cell_type: CellType::LightBlueCell, offset: Vec3::new(0.0, 0.0, -GLOBAL_CELL_SIZE) },
        OcgEntry { collection_id: crawl_cc, cell_type: CellType::LightBlueCell, offset: Vec3::new(0.0, -GLOBAL_CELL_SIZE, 0.0) },
    ];

    // ── Template 5: Armored Blob (Defensive) — 100 organisms ─────────────
    let armor_cc = CollectionId(1);
    let mut armor_collections = HashMap::new();
    armor_collections.insert(armor_cc, CellCollection { starter_cell_position: Vec3::ZERO, parent: None });
    let armor_ocg = vec![
        OcgEntry { collection_id: armor_cc, cell_type: CellType::PhotoCell, offset: Vec3::new(0.0, 0.0, 0.0) },
        OcgEntry { collection_id: armor_cc, cell_type: CellType::PhotoCell, offset: Vec3::new(GLOBAL_CELL_SIZE, 0.0, 0.0) },
        OcgEntry { collection_id: armor_cc, cell_type: CellType::PhotoCell, offset: Vec3::new(-GLOBAL_CELL_SIZE, 0.0, 0.0) },
        OcgEntry { collection_id: armor_cc, cell_type: CellType::PhotoCell, offset: Vec3::new(0.0, 0.0, GLOBAL_CELL_SIZE) },
        OcgEntry { collection_id: armor_cc, cell_type: CellType::HardCell, offset: Vec3::new(2.0 * GLOBAL_CELL_SIZE, 0.0, 0.0) },
        OcgEntry { collection_id: armor_cc, cell_type: CellType::HardCell, offset: Vec3::new(-2.0 * GLOBAL_CELL_SIZE, 0.0, 0.0) },
        OcgEntry { collection_id: armor_cc, cell_type: CellType::HardCell, offset: Vec3::new(0.0, 0.0, 2.0 * GLOBAL_CELL_SIZE) },
        OcgEntry { collection_id: armor_cc, cell_type: CellType::HardCell, offset: Vec3::new(0.0, 0.0, -2.0 * GLOBAL_CELL_SIZE) },
        OcgEntry { collection_id: armor_cc, cell_type: CellType::OrangeCell, offset: Vec3::new(GLOBAL_CELL_SIZE, 0.0, GLOBAL_CELL_SIZE) },
        OcgEntry { collection_id: armor_cc, cell_type: CellType::OrangeCell, offset: Vec3::new(-GLOBAL_CELL_SIZE, 0.0, -GLOBAL_CELL_SIZE) },
        OcgEntry { collection_id: armor_cc, cell_type: CellType::RedCell, offset: Vec3::new(GLOBAL_CELL_SIZE, 0.0, -GLOBAL_CELL_SIZE) },
        OcgEntry { collection_id: armor_cc, cell_type: CellType::RedCell, offset: Vec3::new(-GLOBAL_CELL_SIZE, 0.0, GLOBAL_CELL_SIZE) },
    ];

    let templates: Vec<(u32, &HashMap<CollectionId, CellCollection>, &Vec<OcgEntry>, f32, f32)> = vec![
        (350, &raft_collections, &raft_ocg, 38.0, 42.0),
        (300, &drift_collections, &drift_ocg, 35.0, 45.0),
        (150, &spine_collections, &spine_ocg, 38.0, 42.0),
        (100, &crawl_collections, &crawl_ocg, 20.0, 30.0),
        (100, &armor_collections, &armor_ocg, 38.0, 42.0),
    ];

    for (count, collections, ocg, min_y_offset, max_y_offset) in &templates {
        let template_mesh_cells: Vec<MeshCell> = ocg.iter().map(|entry| {
            let coll = &collections[&entry.collection_id];
            MeshCell { mesh_space_pos: coll.starter_cell_position + entry.offset, cell_type: entry.cell_type }
        }).collect();
        let mesh_handle = meshes.add(merge_organism_mesh(template_mesh_cells));

        for _ in 0..*count {
            let x = rng.random_range(0.0..1024.0);
            let z = rng.random_range(0.0..1024.0);
            let terrain_y = heightmap.height_at(x, z);
            let y_offset = min_y_offset + rng.random_range(0.0..(max_y_offset - min_y_offset));
            let y = terrain_y + y_offset;
            let mut org = create_organism(Vec3::new(x, y, z), collections, ocg);
            spawn_organism(&mut org, &mut commands, mesh_handle.clone(), shared_material.clone());
        }
    }
}

fn create_organism(
    pos: Vec3,
    collections: &HashMap<CollectionId, CellCollection>,
    ocg: &Vec<OcgEntry>,
) -> Organism {
    let angle = rand::random::<f32>() * std::f32::consts::TAU;
    let direction = Vec3::new(angle.cos(), 0.0, angle.sin()).normalize();
    let speed = rand::random::<f32>() * 20.0;

    // NEW: Initialize target rotation and rotation speed
    let rot_angle = rand::random::<f32>() * std::f32::consts::TAU;
    let target_rotation = Quat::from_rotation_y(rot_angle);
    let rotation_speed = 1.0 + rand::random::<f32>() * 2.0;

    let mut min_y_per_collection: HashMap<CollectionId, (Vec3, f32)> = HashMap::new();
    for entry in ocg {
        let coll = &collections[&entry.collection_id];
        let world_y = coll.starter_cell_position.y + entry.offset.y;
        match min_y_per_collection.entry(entry.collection_id) {
            std::collections::hash_map::Entry::Vacant(e) => { e.insert((entry.offset, world_y)); }
            std::collections::hash_map::Entry::Occupied(mut e) => {
                if world_y < e.get().1 { e.insert((entry.offset, world_y)); }
            }
        }
    }
    let floor_cells: Vec<(CollectionId, Vec3)> = min_y_per_collection.into_iter()
        .map(|(id, (offset, _))| (id, offset))
        .collect();

    let bounding_radius = ocg.iter().map(|entry| {
        let coll = &collections[&entry.collection_id];
        let pos = coll.starter_cell_position + entry.offset;
        pos.length()
    }).fold(0.0f32, f32::max) + GLOBAL_CELL_SIZE;

    Organism {
        collections: collections.clone(),
        pos,
        energy: (ocg.len() as f32) * 10.0 * 0.5,
        growth_speed: 1.0,
        adult: false,
        ocg: ocg.clone(),
        joint_entities: HashMap::new(),
        active_cells: Vec::new(),
        grown_cell_count: ocg.len(),
        is_climbing: false,
        movement_speed: speed,
        movement_direction: direction,
        velocity: Vec3::ZERO,
        floor_cells,
        bounding_radius,
        target_rotation,
        rotation_speed,
    }
}

fn spawn_organism(
    organism:         &mut Organism,
    commands:         &mut Commands,
    mesh_handle:      Handle<Mesh>,
    material_handle:  Handle<StandardMaterial>,
) {
    let active_cells: Vec<(Vec3, CellType)> = organism.ocg.iter().map(|entry| {
        let own_collection = &organism.collections[&entry.collection_id];
        let mesh_space_pos = own_collection.starter_cell_position + entry.offset;
        (mesh_space_pos, entry.cell_type)
    }).collect();

    organism.active_cells = active_cells;

    let joint_entities: HashMap<CollectionId, Entity> = organism
        .collections
        .iter()
        .map(|(&id, coll)| {
            let entity = commands.spawn((
                Transform::from_translation(coll.starter_cell_position),
                Visibility::Visible,
            )).id();
            (id, entity)
        })
        .collect();

    organism.joint_entities = joint_entities.clone(); 

    let dir_interval = 1.0 + rand::random::<f32>() * 9.0;
    let rot_interval = 1.0 + rand::random::<f32>() * 9.0;

    let organism_root = commands.spawn((
        // NEW: Spawn with initial rotation matching target
        Transform::from_translation(organism.pos).with_rotation(organism.target_rotation),
        Visibility::Visible,
        OrganismRoot,
        organism.clone(),
        DirectionTimer::new(dir_interval),
        RotationTimer::new(rot_interval), // Attach rotation timer
    )).id();

    for (&id, coll) in &organism.collections {
        let joint_entity = joint_entities[&id];
        match coll.parent {
            None            => { commands.entity(organism_root).add_child(joint_entity); }
            Some(parent_id) => { commands.entity(joint_entities[&parent_id]).add_child(joint_entity); }
        }
    }

    let mesh_entity = commands.spawn((
        Mesh3d(mesh_handle),
        MeshMaterial3d(material_handle),
        Transform::IDENTITY,
        OrganismMesh,
        ShowGizmo,
    )).id();
    commands.entity(organism_root).add_child(mesh_entity);
}
