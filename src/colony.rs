use crate::cell::*;
use crate::viewport_settings::*;
use crate::world_geometry::HeightmapSampler;
use bevy::prelude::*;
use std::collections::HashMap;
use rand::RngExt;

pub struct ColonyPlugin;

impl Plugin for ColonyPlugin {
    fn build(&self, app: &mut App) {
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

    // The OCG-vector is the organism's complete positional DNA.
    // Each entry addresses one cell and which collection it belongs to,
    // what type it is, and where it sits (absolute offset from that
    // collection's starter cell).
    // Growth order = index order.
    // Mutation = inserting, removing, or modifying entries --> leads to change of physiology
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
}

// Stable identifier for a CellCollection within an organism.
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub struct CollectionId(pub u32);

// One entry in the OCG — one cell in the organism's body.
#[derive(Clone, Debug)]
pub struct OcgEntry {
    pub collection_id: CollectionId,
    pub cell_type:     CellType,

    // Absolute offset from the collection's starter cell (local bone space).
    // Convention: every offset must place the cell adjacent (≤ 2 units) to
    // at least one already-placed cell in the same collection, so the body
    // is always a connected graph.
    pub offset: Vec3,
}

#[derive(Clone)]
pub struct CellCollection {
    // Position of this collection's starter cell relative to the organism root.
    // Becomes the joint entity's Transform translation at rest.
    // Rotating this joint rotates the whole collection around this point.
    pub starter_cell_position: Vec3,

    // Optional parent collection for the joint hierarchy.
    // None = direct child of organism root.
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
    let cc1 = CollectionId(1); // body core  — root, no parent
    let cc2 = CollectionId(2); // right limb — child of cc1
    let cc3 = CollectionId(3); // left limb  — child of cc1
    let cc4 = CollectionId(4);
    let cc5 = CollectionId(5);

    let mut collections = HashMap::new();
    collections.insert(cc1, CellCollection {
        starter_cell_position: Vec3::new(0.0, 0.0, 0.0),
        parent: None,
    });
    collections.insert(cc2, CellCollection {
        starter_cell_position: Vec3::new(2.0 * GLOBAL_CELL_SIZE, 0.0, 0.0),
        parent: Some(cc1),
    });
    collections.insert(cc3, CellCollection {
        starter_cell_position: Vec3::new(2.0 * -GLOBAL_CELL_SIZE, 0.0, 0.0),
        parent: Some(cc1),
    });
    collections.insert(cc4, CellCollection {
        starter_cell_position: Vec3::new(0.0, 0.0, 2.0 * GLOBAL_CELL_SIZE),
        parent: Some(cc1),
    });
    collections.insert(cc5, CellCollection {
        starter_cell_position: Vec3::new(0.0, 0.0, 2.0 * -GLOBAL_CELL_SIZE),
        parent: Some(cc1),
    });




    // GLOBAL_CELL_SIZE is a constant in cell.rs
    let half_cell_size = GLOBAL_CELL_SIZE / 2.0;


    let ocg: Vec<OcgEntry> = vec![
        // starter cells for cell-collections
        OcgEntry { collection_id: cc1, cell_type: CellType::RedCell,       offset: Vec3::new(0.0,  0.0,  0.0) },
        OcgEntry { collection_id: cc2, cell_type: CellType::RedCell,      offset: Vec3::new(0.0, 0.0, 0.0) },
        OcgEntry { collection_id: cc3, cell_type: CellType::RedCell,     offset: Vec3::new(0.0, 0.0, 0.0) },
        OcgEntry { collection_id: cc4, cell_type: CellType::RedCell,     offset: Vec3::new(0.0, 0.0, 0.0) },
        OcgEntry { collection_id: cc5, cell_type: CellType::RedCell,     offset: Vec3::new(0.0, 0.0, 0.0) },



        // regular cells

        OcgEntry { collection_id: cc1, cell_type: CellType::OrangeCell,    offset: Vec3::new(0.0,  GLOBAL_CELL_SIZE, 0.0) },
        OcgEntry { collection_id: cc1, cell_type: CellType::LightBlueCell, offset: Vec3::new(0.0, -GLOBAL_CELL_SIZE, 0.0) },
        OcgEntry { collection_id: cc1, cell_type: CellType::YellowCell,    offset: Vec3::new(0.0,  0.0,  GLOBAL_CELL_SIZE) },
        OcgEntry { collection_id: cc1, cell_type: CellType::YellowCell,    offset: Vec3::new(0.0,  0.0,  -GLOBAL_CELL_SIZE) },
        OcgEntry { collection_id: cc1, cell_type: CellType::YellowCell,    offset: Vec3::new(GLOBAL_CELL_SIZE,  0.0,  0.0) },
        OcgEntry { collection_id: cc1, cell_type: CellType::YellowCell,    offset: Vec3::new(-GLOBAL_CELL_SIZE,  0.0,  0.0) },

        OcgEntry { collection_id: cc1, cell_type: CellType::OrangeCell,    offset: Vec3::new(0.0,  half_cell_size, half_cell_size) },
        OcgEntry { collection_id: cc1, cell_type: CellType::LightBlueCell, offset: Vec3::new(half_cell_size, half_cell_size,  0.0) },
        OcgEntry { collection_id: cc1, cell_type: CellType::YellowCell,    offset: Vec3::new(half_cell_size,  0.0, half_cell_size) },
        OcgEntry { collection_id: cc1, cell_type: CellType::YellowCell,    offset: Vec3::new(0.0, -half_cell_size, -half_cell_size) },
        OcgEntry { collection_id: cc1, cell_type: CellType::YellowCell,    offset: Vec3::new(-half_cell_size, -half_cell_size, 0.0) },
        OcgEntry { collection_id: cc1, cell_type: CellType::YellowCell,    offset: Vec3::new(-half_cell_size, 0.0, -half_cell_size) },

        OcgEntry { collection_id: cc1, cell_type: CellType::OrangeCell,    offset: Vec3::new(0.0,  -half_cell_size, half_cell_size) },
        OcgEntry { collection_id: cc1, cell_type: CellType::LightBlueCell, offset: Vec3::new(half_cell_size, -half_cell_size,  0.0) },
        OcgEntry { collection_id: cc1, cell_type: CellType::YellowCell,    offset: Vec3::new(-half_cell_size, 0.0, half_cell_size) },
        OcgEntry { collection_id: cc1, cell_type: CellType::YellowCell,    offset: Vec3::new(0.0, half_cell_size, -half_cell_size) },
        OcgEntry { collection_id: cc1, cell_type: CellType::YellowCell,    offset: Vec3::new(-half_cell_size, half_cell_size,  0.0) },
        OcgEntry { collection_id: cc1, cell_type: CellType::YellowCell,    offset: Vec3::new(half_cell_size, 0.0, -half_cell_size) },



        // creating cells for first limb
        OcgEntry { collection_id: cc2, cell_type: CellType::YellowCell,    offset: Vec3::new(GLOBAL_CELL_SIZE * 1.0, 0.0, 0.0) },
        OcgEntry { collection_id: cc2, cell_type: CellType::YellowCell,    offset: Vec3::new(GLOBAL_CELL_SIZE * 2.0, 0.0, 0.0) },
        OcgEntry { collection_id: cc2, cell_type: CellType::YellowCell,    offset: Vec3::new(GLOBAL_CELL_SIZE * 2.0, 0.0, GLOBAL_CELL_SIZE) },
        OcgEntry { collection_id: cc2, cell_type: CellType::YellowCell,    offset: Vec3::new(GLOBAL_CELL_SIZE * 2.0, 0.0, GLOBAL_CELL_SIZE * 2.0) },
        OcgEntry { collection_id: cc2, cell_type: CellType::YellowCell,    offset: Vec3::new(GLOBAL_CELL_SIZE * 2.0, 0.0, GLOBAL_CELL_SIZE * 3.0) },
        OcgEntry { collection_id: cc2, cell_type: CellType::YellowCell,    offset: Vec3::new(GLOBAL_CELL_SIZE * 2.0, 0.0, GLOBAL_CELL_SIZE * 4.0) },





        // creating cells for second limb
        OcgEntry { collection_id: cc3, cell_type: CellType::YellowCell,    offset: Vec3::new(GLOBAL_CELL_SIZE * -1.0, 0.0, 0.0) },
        OcgEntry { collection_id: cc3, cell_type: CellType::YellowCell,    offset: Vec3::new(GLOBAL_CELL_SIZE * -2.0, 0.0, 0.0) },
        OcgEntry { collection_id: cc3, cell_type: CellType::YellowCell,    offset: Vec3::new(GLOBAL_CELL_SIZE * -2.0, 0.0, GLOBAL_CELL_SIZE) },
        OcgEntry { collection_id: cc3, cell_type: CellType::YellowCell,    offset: Vec3::new(GLOBAL_CELL_SIZE * -2.0, 0.0, GLOBAL_CELL_SIZE * 2.0) },
        OcgEntry { collection_id: cc3, cell_type: CellType::YellowCell,    offset: Vec3::new(GLOBAL_CELL_SIZE * -2.0, 0.0, GLOBAL_CELL_SIZE * 3.0) },
        OcgEntry { collection_id: cc3, cell_type: CellType::YellowCell,    offset: Vec3::new(GLOBAL_CELL_SIZE * -2.0, 0.0, GLOBAL_CELL_SIZE * 4.0) },




        
        // third limb
        OcgEntry { collection_id: cc4, cell_type: CellType::YellowCell,    offset: Vec3::new(0.0, 0.0, GLOBAL_CELL_SIZE * 1.0) },
        OcgEntry { collection_id: cc4, cell_type: CellType::YellowCell,    offset: Vec3::new(0.0, 0.0, GLOBAL_CELL_SIZE * 2.0) },
        OcgEntry { collection_id: cc4, cell_type: CellType::YellowCell,    offset: Vec3::new(GLOBAL_CELL_SIZE * 1.0, 0.0, GLOBAL_CELL_SIZE * 2.0) },
        OcgEntry { collection_id: cc4, cell_type: CellType::YellowCell,    offset: Vec3::new(GLOBAL_CELL_SIZE * 2.0, 0.0, GLOBAL_CELL_SIZE * 2.0) },
        OcgEntry { collection_id: cc4, cell_type: CellType::YellowCell,    offset: Vec3::new(GLOBAL_CELL_SIZE * 3.0, 0.0, GLOBAL_CELL_SIZE * 2.0) },




        // fourth limb
        OcgEntry { collection_id: cc5, cell_type: CellType::YellowCell,    offset: Vec3::new(0.0, 0.0, GLOBAL_CELL_SIZE * -1.0) },
        OcgEntry { collection_id: cc5, cell_type: CellType::YellowCell,    offset: Vec3::new(0.0, 0.0, GLOBAL_CELL_SIZE * -2.0) },

        ];



    let mut orgs = vec![];

    let mut rng = rand::rng();

    for _i in 0..1000 {
        let x = rng.random_range(0.0..1024.0);
        let z = rng.random_range(0.0..1024.0);
        let y = heightmap.height_at(x, z) + 10.0; // spawn 10 units above terrain
        let org = create_organism(Vec3::new(x, y, z), &collections, &ocg);
        orgs.push(org);
    }



    // Build mesh once from the template OCG (same body plan for all organisms)
    let template_mesh_cells: Vec<MeshCell> = ocg.iter().map(|entry| {
        let own_collection = &collections[&entry.collection_id];
        let mesh_space_pos = own_collection.starter_cell_position + entry.offset;
        MeshCell { mesh_space_pos, cell_type: entry.cell_type }
    }).collect();
    let shared_mesh_handle = meshes.add(merge_organism_mesh(template_mesh_cells));
    let shared_material_handle = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        ..default()
    });

    let mut colony = Colony { organisms: orgs };

    for organism in &mut colony.organisms {
        spawn_organism(organism, &mut commands, shared_mesh_handle.clone(), shared_material_handle.clone());
    }
}




fn create_organism(
    pos: Vec3,
    collections: &HashMap<CollectionId, CellCollection>,
    ocg: &Vec<OcgEntry>,
) -> Organism {
    // Generate random initial direction and speed
    let angle = rand::random::<f32>() * std::f32::consts::TAU;
    let direction = Vec3::new(angle.cos(), 0.0, angle.sin()).normalize();
    let speed = rand::random::<f32>() * 20.0;

    // Compute floor_cells: the cell with the lowest Y offset per collection
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

    // Compute bounding radius from OCG (max distance from origin)
    let bounding_radius = ocg.iter().map(|entry| {
        let coll = &collections[&entry.collection_id];
        let pos = coll.starter_cell_position + entry.offset;
        pos.length()
    }).fold(0.0f32, f32::max) + GLOBAL_CELL_SIZE;

    Organism {
        collections: collections.clone(),
        pos,
        energy: (ocg.len() as f32) * 10.0 * 0.5, // start at 50% of max capacity
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
    }
}






// ── Organism spawn ───────────────────────────────────────────────────────────

fn spawn_organism(
    organism:         &mut Organism,
    commands:         &mut Commands,
    mesh_handle:      Handle<Mesh>,
    material_handle:  Handle<StandardMaterial>,
) {
    // ── Step 1: Build active_cells from OCG ─────────────────────────────────
    let active_cells: Vec<(Vec3, CellType)> = organism.ocg.iter().map(|entry| {
        let own_collection = &organism.collections[&entry.collection_id];
        let mesh_space_pos = own_collection.starter_cell_position + entry.offset;
        (mesh_space_pos, entry.cell_type)
    }).collect();

    organism.active_cells = active_cells;

    // ── Step 3: Spawn joint entities ─────────────────────────────────────────
    // Each joint's Transform is its starter_cell_position relative to its parent.
    // For root joints (parent = None) this is relative to the organism root entity.
    // For child joints (parent = Some) this is relative to the parent joint entity.
    // Bevy's transform propagation computes world transforms automatically.
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

    // Store the joint entities in the Organism struct
    organism.joint_entities = joint_entities.clone();  // Clone because we need it for hierarchy wiring


    // random float between 1 and 10
    let random_interval = 1.0 + rand::random::<f32>() * 9.0;

    let organism_root = commands.spawn((
        Transform::from_translation(organism.pos),
        Visibility::Visible,
        OrganismRoot,
        organism.clone(),
        DirectionTimer::new(random_interval), // Add this
    )).id();





    // ── Step 6: Wire joint hierarchy ─────────────────────────────────────────
    for (&id, coll) in &organism.collections {
        let joint_entity = joint_entities[&id];
        match coll.parent {
            None            => { commands.entity(organism_root).add_child(joint_entity); }
            Some(parent_id) => { commands.entity(joint_entities[&parent_id]).add_child(joint_entity); }
        }
    }



    // ── Step 8: Spawn mesh entity ────────────────────────────────────
    // Transform::IDENTITY — the mesh sits at the organism root's origin.

    let mesh_entity = commands.spawn((
    Mesh3d(mesh_handle),
    MeshMaterial3d(material_handle),
    Transform::IDENTITY,
    OrganismMesh,
    ShowGizmo,
)).id();
    commands.entity(organism_root).add_child(mesh_entity);
}



