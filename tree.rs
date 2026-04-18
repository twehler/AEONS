// tree.rs — Tree organism: stationary, branching upward growth up to height 100.0
use bevy::prelude::*;
use rand::Rng;
use rand::prelude::*; 
use std::collections::{HashMap, HashSet};

use crate::cell::{CellType, MeshCell, merge_organism_mesh, OrganismMesh, GLOBAL_CELL_SIZE};
use crate::colony::{
    CellCollection, CollectionId, DirectionTimer, OcgEntry, Organism, OrganismRoot,
};
use crate::world_geometry::HeightmapSampler;
use crate::viewport_settings::ShowGizmo;

// ── Constants ────────────────────────────────────────────────────────────────

const MAX_TREE_HEIGHT: f32 = 100.0;
const MIN_BRANCH_LEN: u32 = 3;
const MAX_BRANCH_LEN: u32 = 10;
const MIN_FORK_COUNT: usize = 2;
const MAX_FORK_COUNT: usize = 3;
const MAX_FORK_ANGLE: f32 = std::f32::consts::FRAC_PI_4; 
const MAX_PLACE_ATTEMPTS: usize = 8;
const MIN_UP_BIAS: f32 = 0.3;
const TRUNK_HEIGHT_CELLS: u32 = 5;
const TREE_CONSTANT_ENERGY: f32 = 1_000_000.0;
const MAX_RECURSION_DEPTH: u32 = 12; 

/// New Constant: Total cell limit per tree
const MAX_TOTAL_CELLS: usize = 100; 

// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct TreePlugin;

impl Plugin for TreePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (
            spawn_tree.run_if(resource_exists::<HeightmapSampler>), 
            keep_tree_energy_constant,
            freeze_trees_on_ground.run_if(resource_exists::<HeightmapSampler>),
        ));
    }
}

// ── Energy and Physics maintenance ──────────────────────────────────────────

fn keep_tree_energy_constant(
    mut query: Query<&mut Organism, (With<OrganismRoot>, With<TreeMarker>)>,
) {
    for mut organism in &mut query {
        organism.energy = TREE_CONSTANT_ENERGY;
    }
}

fn freeze_trees_on_ground(
    mut query: Query<(&mut Transform, &mut Organism), With<TreeMarker>>,
    heightmap: Res<HeightmapSampler>,
) {
    for (mut transform, mut organism) in &mut query {
        let floor_y = heightmap.height_at(transform.translation.x, transform.translation.z);
        if transform.translation.y > floor_y {
            transform.translation.y -= 0.5; 
        } else {
            transform.translation.y = floor_y;
            organism.velocity = Vec3::ZERO;
            organism.movement_speed = 0.0;
        }
    }
}

#[derive(Component)]
pub struct TreeMarker;

// ── Spawn ────────────────────────────────────────────────────────────────────

fn spawn_tree(
    mut commands:  Commands,
    mut meshes:    ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    heightmap:     Res<HeightmapSampler>, 
    mut spawned:   Local<bool>,
) {
    if *spawned { return; }
    *spawned = true;

    let mut rng = rand::rng();

    let x = rng.random_range(200.0_f32..800.0_f32);
    let z = rng.random_range(200.0_f32..800.0_f32);
    
    let terrain_y = heightmap.height_at(x, z);
    let pos = Vec3::new(x, terrain_y + 50.0, z);

    let (collections, ocg) = generate_tree_ocg(&mut rng);

    let seed_mesh_cells: Vec<MeshCell> = ocg[..1].iter().filter_map(|entry| {
        collections.get(&entry.collection_id).map(|coll| MeshCell {
            mesh_space_pos: coll.starter_cell_position + entry.offset,
            cell_type: entry.cell_type,
        })
    }).collect();

    let mesh_handle = meshes.add(merge_organism_mesh(seed_mesh_cells));

    let material_handle = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        ..default()
    });
    
    let rot_angle = rng.random::<f32>() * std::f32::consts::TAU;
    let target_rotation = Quat::from_rotation_y(rot_angle);
    let rotation_speed = 1.0 + rng.random::<f32>() * 2.0;

    let mut organism = Organism {
        collections: collections.clone(),
        pos,
        energy: TREE_CONSTANT_ENERGY,
        growth_speed: 1.0,
        adult: false,
        ocg: ocg.clone(),
        joint_entities: HashMap::new(),
        active_cells: Vec::new(),
        grown_cell_count: 1,
        is_climbing: false,
        movement_speed: 0.0,
        movement_direction: Vec3::Y,
        velocity: Vec3::new(0.0, -1.0, 0.0),
        floor_cells: vec![(CollectionId(1), Vec3::ZERO)],
        bounding_radius: MAX_TREE_HEIGHT * 1.5,
        target_rotation,
        rotation_speed,
    };

    let joint_entities: HashMap<CollectionId, Entity> = collections
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

    organism.active_cells = ocg[..1].iter().filter_map(|entry| {
        collections.get(&entry.collection_id).map(|coll| {
            (coll.starter_cell_position + entry.offset, entry.cell_type)
        })
    }).collect();

    let root = commands.spawn((
        Transform::from_translation(pos),
        Visibility::Visible,
        OrganismRoot,
        TreeMarker,
        organism,
        DirectionTimer::new(999999.0),
    )).id();

    for (&id, coll) in &collections {
        let joint_entity = joint_entities[&id];
        match coll.parent {
            None            => { commands.entity(root).add_child(joint_entity); }
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
    commands.entity(root).add_child(mesh_entity);

    info!("Single tree spawned (Max 100 cells) at {:?}", pos);
}

// ── OCG generation ───────────────────────────────────────────────────────────

fn generate_tree_ocg(
    rng: &mut impl Rng,
) -> (HashMap<CollectionId, CellCollection>, Vec<OcgEntry>) {
    let collection_id = CollectionId(1);
    let mut collections = HashMap::new();
    collections.insert(collection_id, CellCollection {
        starter_cell_position: Vec3::ZERO,
        parent: None,
    });

    let mut ocg: Vec<OcgEntry> = Vec::new();
    let mut occupied: HashSet<[i32; 3]> = HashSet::new();
    let mut cell_count = 0; // Local counter

    let base_pos = Vec3::ZERO;
    place_cell(&mut ocg, &mut occupied, &mut cell_count, collection_id, base_pos, CellType::HardCell);

    grow_branch(rng, &mut ocg, &mut occupied, &mut cell_count, collection_id, base_pos, Vec3::Y, 0, true);

    (collections, ocg)
}

fn quantise(pos: Vec3) -> [i32; 3] {
    [(pos.x / GLOBAL_CELL_SIZE).round() as i32, (pos.y / GLOBAL_CELL_SIZE).round() as i32, (pos.z / GLOBAL_CELL_SIZE).round() as i32]
}

fn place_cell(
    ocg:         &mut Vec<OcgEntry>,
    occupied:    &mut HashSet<[i32; 3]>,
    cell_count:  &mut usize,
    id:          CollectionId,
    offset:      Vec3,
    cell_type:   CellType,
) {
    if *cell_count >= MAX_TOTAL_CELLS { return; } // Global limit check
    
    occupied.insert(quantise(offset));
    ocg.push(OcgEntry { collection_id: id, cell_type, offset });
    *cell_count += 1;
}

fn grow_branch(
    rng:        &mut impl Rng,
    ocg:        &mut Vec<OcgEntry>,
    occupied:   &mut HashSet<[i32; 3]>,
    cell_count: &mut usize,
    id:         CollectionId,
    start_pos:  Vec3,
    direction:  Vec3,
    depth:      u32,
    is_trunk:   bool,
) {
    if depth > MAX_RECURSION_DEPTH || *cell_count >= MAX_TOTAL_CELLS { return; } // Check limit

    let cell_type = branch_cell_type(depth);
    let branch_len = rng.random_range(MIN_BRANCH_LEN..=MAX_BRANCH_LEN);
    let mut current_pos = start_pos;

    for step in 0..branch_len {
        if *cell_count >= MAX_TOTAL_CELLS { return; } // Mid-branch limit check

        let step_vec = snap_to_grid_step(direction);
        let next_pos = current_pos + step_vec;

        if next_pos.y > MAX_TREE_HEIGHT {
            try_place_with_fallback(rng, ocg, occupied, cell_count, id, current_pos, CellType::PhotoCell);
            return;
        }

        let ct = if is_trunk && step < TRUNK_HEIGHT_CELLS { CellType::HardCell } else { cell_type };

        let placed = try_place_with_fallback(rng, ocg, occupied, cell_count, id, next_pos, ct);
        if !placed { return; }

        current_pos = next_pos;
    }

    let fork_count = rng.random_range(MIN_FORK_COUNT..=MAX_FORK_COUNT);
    for _ in 0..fork_count {
        if *cell_count >= MAX_TOTAL_CELLS { break; } // Fork limit check
        let child_dir = deviate_direction(rng, direction, MAX_FORK_ANGLE);
        grow_branch(rng, ocg, occupied, cell_count, id, current_pos, child_dir, depth + 1, false);
    }
}

fn branch_cell_type(depth: u32) -> CellType {
    match depth {
        0 => CellType::HardCell,
        1..=2 => CellType::YellowCell,
        _ => CellType::PhotoCell,
    }
}

fn try_place_with_fallback(
    rng:        &mut impl Rng,
    ocg:        &mut Vec<OcgEntry>,
    occupied:   &mut HashSet<[i32; 3]>,
    cell_count: &mut usize,
    id:         CollectionId,
    pos:        Vec3,
    ct:         CellType,
) -> bool {
    if *cell_count >= MAX_TOTAL_CELLS { return false; } // Limit check

    if !occupied.contains(&quantise(pos)) {
        place_cell(ocg, occupied, cell_count, id, pos, ct);
        return true;
    }

    let offsets: [Vec3; 4] = [Vec3::X * GLOBAL_CELL_SIZE, Vec3::NEG_X * GLOBAL_CELL_SIZE, Vec3::Z * GLOBAL_CELL_SIZE, Vec3::NEG_Z * GLOBAL_CELL_SIZE];

    for _ in 0..MAX_PLACE_ATTEMPTS {
        let idx = rng.random_range(0..offsets.len());
        let candidate = pos + offsets[idx];
        if !occupied.contains(&quantise(candidate)) {
            place_cell(ocg, occupied, cell_count, id, candidate, ct);
            return true;
        }
    }
    false 
}

fn snap_to_grid_step(dir: Vec3) -> Vec3 {
    let scaled = dir * GLOBAL_CELL_SIZE;
    let mut snapped = Vec3::new((scaled.x / GLOBAL_CELL_SIZE).round() * GLOBAL_CELL_SIZE, (scaled.y / GLOBAL_CELL_SIZE).round() * GLOBAL_CELL_SIZE, (scaled.z / GLOBAL_CELL_SIZE).round() * GLOBAL_CELL_SIZE);
    if snapped == Vec3::ZERO {
        if dir.x.abs() >= dir.y.abs() && dir.x.abs() >= dir.z.abs() { snapped.x = if dir.x >= 0.0 { GLOBAL_CELL_SIZE } else { -GLOBAL_CELL_SIZE }; } 
        else if dir.y.abs() >= dir.x.abs() && dir.y.abs() >= dir.z.abs() { snapped.y = if dir.y >= 0.0 { GLOBAL_CELL_SIZE } else { -GLOBAL_CELL_SIZE }; } 
        else { snapped.z = if dir.z >= 0.0 { GLOBAL_CELL_SIZE } else { -GLOBAL_CELL_SIZE }; }
    }
    snapped
}

fn deviate_direction(rng: &mut impl Rng, dir: Vec3, max_angle: f32) -> Vec3 {
    let perp1 = if dir.x.abs() < 0.9 { dir.cross(Vec3::X).normalize() } else { dir.cross(Vec3::Y).normalize() };
    let perp2 = dir.cross(perp1).normalize();
    let phi: f32   = rng.random_range(0.0_f32..std::f32::consts::TAU);
    let theta: f32 = rng.random_range(0.0_f32..max_angle);
    let deviated = (dir * theta.cos() + perp1 * (theta.sin() * phi.cos()) + perp2 * (theta.sin() * phi.sin())).normalize();
    if deviated.y < MIN_UP_BIAS { return (deviated + dir * 0.5).normalize(); }
    deviated
}
