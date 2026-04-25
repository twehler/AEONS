use crate::cell::*;
use crate::viewport_settings::*;
use crate::world_geometry::HeightmapSampler;
use crate::movement::*;
use bevy::prelude::*;
use std::collections::HashMap;
use rand::RngExt;
use rand::prelude::*;

pub struct ColonyPlugin;

impl Plugin for ColonyPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(PopulationCap::default());
        app.add_systems(Update, spawn_colony.run_if(resource_exists::<HeightmapSampler>));
    }
}

// ── Core data structures ──────────────────────────────────────────────────────

#[derive(Component, Clone)]
pub struct Organism {
    pub collections:        HashMap<CollectionId, CellCollection>,
    pub pos:                Vec3,
    pub energy:             f32,
    pub growth_speed:       f32,
    pub adult:              bool,
    pub ocg:                Vec<OcgEntry>,
    pub joint_entities:     HashMap<CollectionId, Entity>,
    pub active_cells:       Vec<(Vec3, CellType)>,
    pub grown_cell_count:   usize,
    pub is_climbing:        bool,
    pub movement_speed:     f32,
    pub movement_direction: Vec3,
    pub velocity:           Vec3,
    pub floor_cells:        Vec<(CollectionId, Vec3)>,
    pub bounding_radius:    f32,
    pub target_rotation:    Quat,
    pub rotation_speed:     f32,
}

/// Marks an organism as a photoautotroph (energy from photosynthesis).
#[derive(Component, Clone, Copy)]
pub struct Photoautotroph;

/// Marks an organism as a heterotroph (energy from consuming other organisms).
#[derive(Component, Clone, Copy)]
pub struct Heterotroph;

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub struct CollectionId(pub u32);

#[derive(Clone, Debug)]
pub struct OcgEntry {
    pub collection_id: CollectionId,
    pub cell_type:     CellType,
    pub offset:        Vec3,
}

#[derive(Clone)]
pub struct CellCollection {
    pub starter_cell_position: Vec3,
    pub parent:                Option<CollectionId>,
}

#[derive(Component)]
pub struct OrganismRoot;


#[derive(Resource)]
pub struct PopulationCap {
    pub max: usize,
}
impl Default for PopulationCap {
    fn default() -> Self { Self { max: 1500 } }
}

// ── Organism templates ────────────────────────────────────────────────────────

struct TemplateData {
    count:       u32,
    collections: HashMap<CollectionId, CellCollection>,
    ocg:         Vec<OcgEntry>,
    spawn_y:     f32,
}

/// Photoautotroph — Single-cell photosynthetic organism.
fn photoautotroph_template() -> TemplateData {
    let cc = CollectionId(1);
    let mut collections = HashMap::new();
    collections.insert(cc, CellCollection { starter_cell_position: Vec3::ZERO, parent: None });

    let ocg = vec![
        OcgEntry { collection_id: cc, cell_type: CellType::PhotoCell, offset: Vec3::ZERO },
    ];

    TemplateData { count: 900, collections, ocg, spawn_y: 1.0 }
}

/// Heterotroph — Compact 18-cell predatory blob.
fn heterotroph_template() -> TemplateData {
    let cc = CollectionId(1);
    let mut collections = HashMap::new();
    collections.insert(cc, CellCollection { starter_cell_position: Vec3::ZERO, parent: None });

    let g = GLOBAL_CELL_SIZE;
    let ocg = vec![
        // Core digestion (2 cells)
        OcgEntry { collection_id: cc, cell_type: CellType::GutCell,  offset: Vec3::new( 0.0,  0.0,  0.0) },
        OcgEntry { collection_id: cc, cell_type: CellType::GutCell,  offset: Vec3::new( 0.0,   g,   0.0) },
        
        // Horizontal Fins for movement (4 cells)
        OcgEntry { collection_id: cc, cell_type: CellType::FinCell,  offset: Vec3::new(  g,  0.0,  0.0) },
        OcgEntry { collection_id: cc, cell_type: CellType::FinCell,  offset: Vec3::new( -g,  0.0,  0.0) },
        OcgEntry { collection_id: cc, cell_type: CellType::FinCell,  offset: Vec3::new( 0.0, 0.0,   g) },
        OcgEntry { collection_id: cc, cell_type: CellType::FinCell,  offset: Vec3::new( 0.0, 0.0,  -g) },
        
        // Red Body Mass - Top layer (4 cells)
        OcgEntry { collection_id: cc, cell_type: CellType::RedCell,  offset: Vec3::new(  g,   g,  0.0) },
        OcgEntry { collection_id: cc, cell_type: CellType::RedCell,  offset: Vec3::new( -g,   g,  0.0) },
        OcgEntry { collection_id: cc, cell_type: CellType::RedCell,  offset: Vec3::new( 0.0,  g,   g) },
        OcgEntry { collection_id: cc, cell_type: CellType::RedCell,  offset: Vec3::new( 0.0,  g,  -g) },
        
        // Red Body Mass - Bottom layer (4 cells)
        OcgEntry { collection_id: cc, cell_type: CellType::RedCell,  offset: Vec3::new(  g,  -g,  0.0) },
        OcgEntry { collection_id: cc, cell_type: CellType::RedCell,  offset: Vec3::new( -g,  -g,  0.0) },
        OcgEntry { collection_id: cc, cell_type: CellType::RedCell,  offset: Vec3::new( 0.0, -g,   g) },
        OcgEntry { collection_id: cc, cell_type: CellType::RedCell,  offset: Vec3::new( 0.0, -g,  -g) },
        
        // Red Body Mass - Middle horizontal corners (4 cells)
        OcgEntry { collection_id: cc, cell_type: CellType::RedCell,  offset: Vec3::new(  g, 0.0,   g) },
        OcgEntry { collection_id: cc, cell_type: CellType::RedCell,  offset: Vec3::new( -g, 0.0,   g) },
        OcgEntry { collection_id: cc, cell_type: CellType::RedCell,  offset: Vec3::new(  g, 0.0,  -g) },
        OcgEntry { collection_id: cc, cell_type: CellType::RedCell,  offset: Vec3::new( -g, 0.0,  -g) },
    ];

    TemplateData { count: 20, collections, ocg, spawn_y: 1.0 }
}

// ── Spawning ──────────────────────────────────────────────────────────────────

fn spawn_colony(
    mut commands:  Commands,
    mut meshes:    ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    heightmap:     Res<HeightmapSampler>,
    mut spawned:   Local<bool>,
) {
    if *spawned { return; }
    *spawned = true;

    let shared_material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        ..default()
    });

    let mut rng = rand::rng();

    let photo_template = photoautotroph_template();
    let hetero_template = heterotroph_template();

    let photo_mesh = {
        let cells: Vec<MeshCell> = photo_template.ocg.iter().map(|e| {
            MeshCell {
                mesh_space_pos: photo_template.collections[&e.collection_id].starter_cell_position + e.offset,
                cell_type: e.cell_type,
            }
        }).collect();
        meshes.add(merge_organism_mesh(cells))
    };

    let hetero_mesh = {
        let cells: Vec<MeshCell> = hetero_template.ocg.iter().map(|e| {
            MeshCell {
                mesh_space_pos: hetero_template.collections[&e.collection_id].starter_cell_position + e.offset,
                cell_type: e.cell_type,
            }
        }).collect();
        meshes.add(merge_organism_mesh(cells))
    };

    // ── Spawn photoautotrophs ─────────────────────────────────────────────────
    for _ in 0..photo_template.count {
        let x = rng.random_range(0.0_f32..1024.0);
        let z = rng.random_range(0.0_f32..1024.0);
        let y = heightmap.height_at(x, z) + photo_template.spawn_y;

        let mut org = create_organism(
            Vec3::new(x, y, z),
            &photo_template.collections,
            &photo_template.ocg,
            OrganismKind::Photoautotroph,
            &mut rng,
        );
        spawn_organism(
            &mut org,
            &mut commands,
            photo_mesh.clone(),
            shared_material.clone(),
            OrganismKind::Photoautotroph,
        );
    }

    // ── Spawn heterotrophs ────────────────────────────────────────────────────
    for _ in 0..hetero_template.count {
        let x = rng.random_range(0.0_f32..1024.0);
        let z = rng.random_range(0.0_f32..1024.0);
        let y = heightmap.height_at(x, z) + hetero_template.spawn_y;

        let mut org = create_organism(
            Vec3::new(x, y, z),
            &hetero_template.collections,
            &hetero_template.ocg,
            OrganismKind::Heterotroph,
            &mut rng,
        );
        spawn_organism(
            &mut org,
            &mut commands,
            hetero_mesh.clone(),
            shared_material.clone(),
            OrganismKind::Heterotroph,
        );
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum OrganismKind {
    Photoautotroph,
    Heterotroph,
}

fn create_organism(
    pos:        Vec3,
    collections: &HashMap<CollectionId, CellCollection>,
    ocg:        &Vec<OcgEntry>,
    kind:       OrganismKind,
    rng:        &mut impl rand::Rng,
) -> Organism {
    let angle     = rng.random::<f32>() * std::f32::consts::TAU;
    let direction = Vec3::new(angle.cos(), 0.0, angle.sin()).normalize();

    let movement_speed = match kind {
        OrganismKind::Photoautotroph => 0.0,
        OrganismKind::Heterotroph    => 15.0 + rng.random::<f32>() * 10.0,
    };

    let target_rotation = Quat::from_rotation_y(angle);

    let rotation_speed = match kind {
        OrganismKind::Photoautotroph => 0.0,
        OrganismKind::Heterotroph    => 1.0 + rng.random::<f32>() * 2.0,
    };

    let mut min_y_per_coll: HashMap<CollectionId, (Vec3, f32)> = HashMap::new();
    for entry in ocg {
        let coll    = &collections[&entry.collection_id];
        let world_y = coll.starter_cell_position.y + entry.offset.y;
        min_y_per_coll
            .entry(entry.collection_id)
            .and_modify(|e| { if world_y < e.1 { *e = (entry.offset, world_y); } })
            .or_insert((entry.offset, world_y));
    }
    let floor_cells: Vec<(CollectionId, Vec3)> = min_y_per_coll
        .into_iter()
        .map(|(id, (off, _))| (id, off))
        .collect();

    let bounding_radius = ocg.iter()
        .map(|e| (collections[&e.collection_id].starter_cell_position + e.offset).length())
        .fold(0.0_f32, f32::max) + GLOBAL_CELL_SIZE;

    let max_energy = (ocg.len() as f32) * 10.0;

    Organism {
        collections:        collections.clone(),
        pos,
        energy:             max_energy * 0.5,
        growth_speed:       1.0,
        adult:              false,
        ocg:                ocg.clone(),
        joint_entities:     HashMap::new(),
        active_cells:       Vec::new(),
        grown_cell_count:   ocg.len(),
        is_climbing:        false,
        movement_speed,
        movement_direction: direction,
        velocity:           Vec3::ZERO,
        floor_cells,
        bounding_radius,
        target_rotation,
        rotation_speed,
    }
}

fn spawn_organism(
    organism:        &mut Organism,
    commands:        &mut Commands,
    mesh_handle:     Handle<Mesh>,
    material_handle: Handle<StandardMaterial>,
    kind:            OrganismKind,
) {
    organism.active_cells = organism.ocg.iter().map(|e| {
        (
            organism.collections[&e.collection_id].starter_cell_position + e.offset,
            e.cell_type,
        )
    }).collect();

    let joint_entities: HashMap<CollectionId, Entity> = organism.collections.iter().map(|(&id, coll)| {
        let e = commands.spawn((
            Transform::from_translation(coll.starter_cell_position),
            Visibility::Visible,
        )).id();
        (id, e)
    }).collect();
    organism.joint_entities = joint_entities.clone();

    let mut rng = rand::rng();

    let random_interval_dir = 1.0 + rng.random::<f32>() * 9.0;
    let random_interval_rot = 1.0 + rng.random::<f32>() * 9.0; 

    let root = match kind {
        OrganismKind::Photoautotroph => commands.spawn((
            Transform::from_translation(organism.pos).with_rotation(organism.target_rotation),
            Visibility::Visible,
            OrganismRoot,
            Photoautotroph,
            organism.clone(),
            DirectionTimer::new(random_interval_dir),
            RotationTimer::new(random_interval_rot), 

        )).id(),
        OrganismKind::Heterotroph => commands.spawn((
            Transform::from_translation(organism.pos).with_rotation(organism.target_rotation),
            Visibility::Visible,
            OrganismRoot,
            Heterotroph,
            organism.clone(),
            DirectionTimer::new(random_interval_dir),
            RotationTimer::new(random_interval_rot), 

        )).id(),
    };

    for (&id, coll) in &organism.collections {
        let joint = joint_entities[&id];
        match coll.parent {
            None         => { commands.entity(root).add_child(joint); }
            Some(par_id) => { commands.entity(joint_entities[&par_id]).add_child(joint); }
        }
    }

    let mesh_entity = commands.spawn((
        Mesh3d(mesh_handle),
        MeshMaterial3d(material_handle),
        Transform::IDENTITY,
        OrganismMesh,
        ShowGizmo,
    )).id();
    commands.entity(root).add_child(mesh_entity);
}
