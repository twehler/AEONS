use crate::cell::*;
use crate::viewport_settings::*;
use bevy::prelude::*;
use std::collections::HashMap;
use bevy::mesh::skinning::{SkinnedMesh, SkinnedMeshInverseBindposes};

pub struct ColonyPlugin;

impl Plugin for ColonyPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_colony)
        .add_systems(Update, apply_gravity);
    }
}

const GRAVITY: f32 = 9.8;

fn apply_gravity(
    time:      Res<Time>,
    mut query: Query<&mut Transform, With<OrganismRoot>>,
) {
    for mut transform in &mut query {
        transform.translation.y -= GRAVITY * time.delta_secs();
    }
}




// ── Core data structures ─────────────────────────────────────────────────────

pub struct Colony {
    organisms: Vec<Organism>,
}

#[derive(Component)]
pub struct Organism {
    pub collections: HashMap<CollectionId, CellCollection>,
    pos:         Vec3,
    orientation: Quat,
    energy: f32,
    growth_speed: f32,
    adult: bool,


    // The OCG is the organism's complete positional DNA.
    // Each entry addresses one cell: which collection it belongs to,
    // what type it is, and where it sits (absolute offset from that
    // collection's starter cell). Growth order = index order.
    // Mutation = inserting, removing, or modifying entries.
    pub ocg: Vec<OcgEntry>,
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
    pub offset: [f32; 3],
}

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





// ── Blend zone constant ──────────────────────────────────────────────────────

const BLEND_RADIUS: f32 = 2.0;


// ── Plugin entry point ───────────────────────────────────────────────────────

fn spawn_colony(
    mut commands:      Commands,
    mut meshes:        ResMut<Assets<Mesh>>,
    mut materials:     ResMut<Assets<StandardMaterial>>,
    mut inv_bindposes: ResMut<Assets<SkinnedMeshInverseBindposes>>,
) {
    let cc1 = CollectionId(1); // body core  — root, no parent
    let cc2 = CollectionId(2); // right limb — child of cc1
    let cc3 = CollectionId(3); // left limb  — child of cc1

    let mut collections = HashMap::new();
    collections.insert(cc1, CellCollection {
        starter_cell_position: Vec3::new(0.0, 0.0, 0.0),
        parent: None,
    });
    collections.insert(cc2, CellCollection {
        starter_cell_position: Vec3::new(4.0, 0.0, 0.0),
        parent: Some(cc1),
    });
    collections.insert(cc3, CellCollection {
        starter_cell_position: Vec3::new(-4.0, 0.0, 0.0),
        parent: Some(cc1),
    });

    let ocg: Vec<OcgEntry> = vec![
        // starter cells
        OcgEntry { collection_id: cc1, cell_type: CellType::RedCell,       offset: [0.0,  0.0,  0.0] },
        //OcgEntry { collection_id: cc2, cell_type: CellType::RedCell,      offset: [0.0,  0.0,  0.0] },
        //OcgEntry { collection_id: cc3, cell_type: CellType::RedCell,     offset: [0.0,  0.0,  0.0] },

        // regular cells

        OcgEntry { collection_id: cc1, cell_type: CellType::OrangeCell,    offset: [0.0,  1.0,  0.0] },
        OcgEntry { collection_id: cc1, cell_type: CellType::LightBlueCell, offset: [0.0, -1.0,  0.0] },
        OcgEntry { collection_id: cc1, cell_type: CellType::YellowCell,    offset: [0.0,  0.0,  1.0] },
        OcgEntry { collection_id: cc1, cell_type: CellType::YellowCell,    offset: [0.0,  0.0,  -1.0] },
        OcgEntry { collection_id: cc1, cell_type: CellType::YellowCell,    offset: [1.0,  0.0,  0.0] },
        OcgEntry { collection_id: cc1, cell_type: CellType::YellowCell,    offset: [-1.0,  0.0,  0.0] },

        OcgEntry { collection_id: cc1, cell_type: CellType::OrangeCell,    offset: [0.0,  0.5,  0.5] },
        OcgEntry { collection_id: cc1, cell_type: CellType::LightBlueCell, offset: [0.5, 0.5,  0.0] },
        OcgEntry { collection_id: cc1, cell_type: CellType::YellowCell,    offset: [0.5,  0.0, 0.5] },
        OcgEntry { collection_id: cc1, cell_type: CellType::YellowCell,    offset: [0.0, -0.5, -0.5] },
        OcgEntry { collection_id: cc1, cell_type: CellType::YellowCell,    offset: [-0.5, -0.5,  0.0] },
        OcgEntry { collection_id: cc1, cell_type: CellType::YellowCell,    offset: [-0.5, 0.0, -0.5] },

        OcgEntry { collection_id: cc1, cell_type: CellType::OrangeCell,    offset: [0.0,  -0.5,  0.5] },
        OcgEntry { collection_id: cc1, cell_type: CellType::LightBlueCell, offset: [0.5, -0.5,  0.0] },
        OcgEntry { collection_id: cc1, cell_type: CellType::YellowCell,    offset: [-0.5,  0.0, 0.5] },
        OcgEntry { collection_id: cc1, cell_type: CellType::YellowCell,    offset: [0.0, 0.5, -0.5] },
        OcgEntry { collection_id: cc1, cell_type: CellType::YellowCell,    offset: [-0.5, 0.5,  0.0] },
        OcgEntry { collection_id: cc1, cell_type: CellType::YellowCell,    offset: [0.5, 0.0, -0.5] },

        ];

    let o1 = Organism {
        collections,
        pos:         Vec3::new(100.0, 50.0, 100.0),
        orientation: Quat::IDENTITY,
        energy: 1.0,
        growth_speed: 1.0,
        adult: false,
        ocg,
    };

    let colony = Colony { organisms: vec![o1] };

    for organism in &colony.organisms {
        spawn_organism(organism, &mut commands, &mut meshes, &mut materials, &mut inv_bindposes);
    }
}


// ── Organism spawn ───────────────────────────────────────────────────────────

fn spawn_organism(
    organism:      &Organism,
    commands:      &mut Commands,
    meshes:        &mut ResMut<Assets<Mesh>>,
    materials:     &mut ResMut<Assets<StandardMaterial>>,
    inv_bindposes: &mut ResMut<Assets<SkinnedMeshInverseBindposes>>,
) {
    // ── Step 1: Assign stable joint indices ──────────────────────────────────
    // Sort by CollectionId for deterministic ordering — HashMap iteration
    // order is not guaranteed, so we must sort explicitly.
    // The inverse_bindposes array and SkinnedMesh.joints must match this order.
    let mut sorted_ids: Vec<CollectionId> = organism.collections.keys().copied().collect();
    sorted_ids.sort_by_key(|id| id.0);

    let joint_index_map: HashMap<CollectionId, u16> = sorted_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i as u16))
        .collect();

    // ── Step 2: Build MeshCell list from OCG ─────────────────────────────────
    // mesh_space_pos = starter_cell_position + offset
    // This places each cell correctly in the mesh's rest pose coordinate space,
    // relative to the organism root. The skinning shader deforms from this rest
    // position using the joint transforms and inverse bindposes.
    let mesh_cells: Vec<MeshCell> = organism.ocg.iter().map(|entry| {
        let own_collection  = &organism.collections[&entry.collection_id];
        let own_joint       = joint_index_map[&entry.collection_id];
        let mesh_space_pos  = own_collection.starter_cell_position + Vec3::from(entry.offset);

        // Blend zone: find the nearest other collection's starter within BLEND_RADIUS
        let nearest_neighbour = organism.collections
            .iter()
            .filter(|(id, _)| **id != entry.collection_id)
            .map(|(id, coll)| {
                let dist = mesh_space_pos.distance(coll.starter_cell_position);
                (*id, dist)
            })
            .filter(|(_, dist)| *dist < BLEND_RADIUS)
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());

        let skinning = match nearest_neighbour {
            None => CellSkinning::single(own_joint),
            Some((neighbour_id, neighbour_dist)) => {
                let own_dist       = Vec3::from(entry.offset).length().max(0.001);
                let primary_weight = (neighbour_dist / (own_dist + neighbour_dist))
                    .clamp(0.5, 1.0);
                CellSkinning::blended(own_joint, joint_index_map[&neighbour_id], primary_weight)
            }
        };

        MeshCell { mesh_space_pos, cell_type: entry.cell_type, skinning }
    }).collect();

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

    // ── Step 4: Compute inverse bindposes ────────────────────────────────────
    // The inverse bindpose is the inverse of each bone's world-space rest transform.
    // World-space rest = organism.pos + starter_cell_position (for all joints,
    // since at rest no joint has any rotation and the hierarchy is purely additive).
    // We use Mat4::inverse() for correctness — it handles any future rotation at rest.
    let inverse_bindposes: Vec<Mat4> = sorted_ids
    .iter()
    .map(|id| {
        let coll = &organism.collections[id];
        // The mesh entity is a child of organism_root, so its local space
        // IS the organism root's space. The joint's rest position in that
        // space is simply starter_cell_position — no organism.pos needed.
        Mat4::from_translation(coll.starter_cell_position).inverse()
    })
    .collect();

    let bindpose_handle = inv_bindposes.add(
        SkinnedMeshInverseBindposes::from(inverse_bindposes)
    );

    // ── Step 5: Spawn organism root ──────────────────────────────────────────
    let organism_root = commands.spawn((
        Transform::from_translation(organism.pos).with_rotation(organism.orientation),
        Visibility::Visible,
        OrganismRoot, // ← add this
    )).id();

    // ── Step 6: Wire joint hierarchy ─────────────────────────────────────────
    for (&id, coll) in &organism.collections {
        let joint_entity = joint_entities[&id];
        match coll.parent {
            None            => { commands.entity(organism_root).add_child(joint_entity); }
            Some(parent_id) => { commands.entity(joint_entities[&parent_id]).add_child(joint_entity); }
        }
    }

    // ── Step 7: Build ordered joints list for SkinnedMesh ────────────────────
    // Must be in the same order as joint_index_map (sorted by CollectionId).
    let joints_vec: Vec<Entity> = sorted_ids
        .iter()
        .map(|id| joint_entities[id])
        .collect();

    // ── Step 8: Spawn skinned mesh entity ────────────────────────────────────
    // Transform::IDENTITY — the mesh sits at the organism root's origin.
    // The bones drive all deformation from the rest pose.
    let mesh   = merge_organism_mesh(mesh_cells);
    let handle = meshes.add(mesh);

    let mesh_entity = commands.spawn((
        Mesh3d(handle),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            ..default()
        })),
        Transform::IDENTITY,
        SkinnedMesh {
            inverse_bindposes: bindpose_handle,
            joints:            joints_vec,
        },
        OrganismMesh,
        ShowGizmo,
    )).id();

    commands.entity(organism_root).add_child(mesh_entity);
}



