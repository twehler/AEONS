// tree.rs — Tree organism: thick trunk with realistic branching
//
// ── Geometry primer ───────────────────────────────────────────────────────────
//
// After the cell.rs quantise fix, cells are placed on a grid where the key for
// world position P is [round(P.x), round(P.y), round(P.z)] (divisor = GCS/2 = 1.0).
// Two cells are face-adjacent (zero gap) when their keys differ by exactly one of
// the 12 FACE_NEIGHBOURS vectors: [0,±1,±1] and [±1,±1,0].
//
// CRITICAL: a step of GLOBAL_CELL_SIZE (2.0) along a single axis, e.g. (0,2,0),
// produces a key diff of [0,2,0] which is NOT in FACE_NEIGHBOURS — that cell
// would float with a gap.  All movement must use exactly one FN vector per step.
//
// ── Trunk construction ────────────────────────────────────────────────────────
//
// Each trunk layer is a 5-cell cross: one centre cell + 4 arms at the horizontal
// face-neighbours [±1,0,±1] from the centre.  The arm cells are connected to the
// centre but not to each other, which is fine — the cross is connected.
//
// The trunk axis zigzags vertically using alternating FN steps [0,1,1] / [0,1,-1],
// so every consecutive axis cell is face-adjacent.  The cross arms follow the same
// zigzag, keeping every cell in layer N face-adjacent to at least one cell in N-1.
//
//   Layer 0  axis key (0,0,0)  arms (±1,0,±1)
//   Layer 1  axis key (0,1,1)  arms (±1,1, 0) and (±1,1, 2)
//   Layer 2  axis key (0,2,0)  arms (±1,2,±1)
//   ...alternating z=0 and z=1 for the axis...
//
// ── Branch construction ───────────────────────────────────────────────────────
//
// Branches are single-cell wide and grow using best_fn_step: at each step we pick
// the FACE_NEIGHBOURS vector whose direction best matches the desired float direction.
// Since every step is exactly one FN vector, gap-freedom is guaranteed by construction.
//
// Branches sprout from arm cells at chosen trunk layers, growing outward and upward.
// Sub-branches fork from branch tips in the same way.

use bevy::prelude::*;
use rand::Rng;
use rand::prelude::*;
use std::collections::{HashMap, HashSet};

use crate::cell::{CellType, MeshCell, merge_organism_mesh, OrganismMesh, GLOBAL_CELL_SIZE, FACE_NEIGHBOURS};
use crate::colony::{
    CellCollection, CollectionId, DirectionTimer, OcgEntry, Organism, OrganismRoot,
};
use crate::world_geometry::HeightmapSampler;
use crate::viewport_settings::ShowGizmo;

// ── Constants ────────────────────────────────────────────────────────────────

/// Half of GLOBAL_CELL_SIZE: the grid unit after the quantise fix.
const HALF: f32 = GLOBAL_CELL_SIZE / 2.0;

/// How many 5-cell cross layers the trunk has before branching.
const TRUNK_LAYERS: usize = 16;

/// Minimum angle (radians) between branch directions, to spread them evenly.
const MIN_BRANCH_SEPARATION: f32 = std::f32::consts::FRAC_PI_4; // 45°

/// How many primary branches sprout from the trunk.
const PRIMARY_BRANCH_COUNT: usize = 4;

/// Length range (in FN steps) for primary branches.
const PRIMARY_BRANCH_MIN: usize = 8;
const PRIMARY_BRANCH_MAX: usize = 14;

/// Number of sub-branches per primary branch tip.
const SUB_BRANCH_COUNT_MIN: usize = 2;
const SUB_BRANCH_COUNT_MAX: usize = 3;

/// Length range for sub-branches.
const SUB_BRANCH_MIN: usize = 4;
const SUB_BRANCH_MAX: usize = 8;

/// Maximum angular deviation when spawning sub-branches (radians).
const SUB_BRANCH_ANGLE: f32 = std::f32::consts::FRAC_PI_3; // 60°

/// Energy held constant every frame so energy.rs never starves the tree.
const TREE_CONSTANT_ENERGY: f32 = 1_000_000.0;

/// Total OCG cell cap — the tree will stop growing new cells beyond this.
const MAX_TOTAL_CELLS: usize = 200;

// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct TreePlugin;

impl Plugin for TreePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (
            spawn_tree.run_if(resource_exists::<HeightmapSampler>),
            keep_tree_energy_constant,
            suppress_tree_reproduction,
            freeze_trees_on_ground.run_if(resource_exists::<HeightmapSampler>),
        ));
    }
}

// ── Marker component ─────────────────────────────────────────────────────────

#[derive(Component)]
pub struct TreeMarker;

// ── Maintenance systems ───────────────────────────────────────────────────────

fn keep_tree_energy_constant(
    mut query: Query<&mut Organism, (With<OrganismRoot>, With<TreeMarker>)>,
) {
    for mut organism in &mut query {
        organism.energy = TREE_CONSTANT_ENERGY;
    }
}

/// Prevent reproduction.rs from treating trees as adult parents.
///
/// growth_system sets organism.adult = true once grown_cell_count >= ocg.len().
/// reproduction.rs queries for adult organisms and spawns offspring that inherit
/// the parent's full OCG — producing tree-shaped mobile copies of the tree.
/// Keeping adult = false on all TreeMarker entities every frame blocks this
/// without needing to modify reproduction.rs.
fn suppress_tree_reproduction(
    mut query: Query<&mut Organism, (With<OrganismRoot>, With<TreeMarker>)>,
) {
    for mut organism in &mut query {
        organism.adult = false;
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

// ── Spawn ─────────────────────────────────────────────────────────────────────

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
    // Spawn above terrain — freeze_trees_on_ground will settle it.
    let pos = Vec3::new(x, terrain_y + 50.0, z);

    let (collections, ocg) = generate_tree_ocg(&mut rng);

    // Seed mesh: first cell only. GrowthPlugin reveals the rest over time.
    let seed_mesh_cells: Vec<MeshCell> = ocg[..1.min(ocg.len())].iter().filter_map(|entry| {
        collections.get(&entry.collection_id).map(|coll| MeshCell {
            mesh_space_pos: coll.starter_cell_position + entry.offset,
            cell_type: entry.cell_type,
        })
    }).collect();

    let mesh_handle    = meshes.add(merge_organism_mesh(seed_mesh_cells));
    let material_handle = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        ..default()
    });

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
        bounding_radius: (TRUNK_LAYERS as f32) * HALF * 3.0,
        target_rotation: Quat::IDENTITY,  // trees never rotate
        rotation_speed: 0.0,               // trees never rotate
    };

    let joint_entities: HashMap<CollectionId, Entity> = collections
        .iter()
        .map(|(&id, coll)| {
            let e = commands.spawn((
                Transform::from_translation(coll.starter_cell_position),
                Visibility::Visible,
            )).id();
            (id, e)
        })
        .collect();

    organism.joint_entities = joint_entities.clone();
    organism.active_cells = ocg[..1.min(ocg.len())].iter().filter_map(|entry| {
        collections.get(&entry.collection_id)
            .map(|coll| (coll.starter_cell_position + entry.offset, entry.cell_type))
    }).collect();

    let root = commands.spawn((
        Transform::from_translation(pos),
        Visibility::Visible,
        OrganismRoot,
        TreeMarker,
        organism,
        DirectionTimer::new(999_999.0),
    )).id();

    for (&id, coll) in &collections {
        let joint = joint_entities[&id];
        match coll.parent {
            None            => { commands.entity(root).add_child(joint); }
            Some(parent_id) => { commands.entity(joint_entities[&parent_id]).add_child(joint); }
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

    info!("Tree spawned at ({:.1}, {:.1}, {:.1}) — {} cells in OCG", pos.x, pos.y, pos.z, ocg.len());
}

// ── OCG generation ────────────────────────────────────────────────────────────

fn generate_tree_ocg(
    rng: &mut impl Rng,
) -> (HashMap<CollectionId, CellCollection>, Vec<OcgEntry>) {
    let cid = CollectionId(1);
    let mut collections = HashMap::new();
    collections.insert(cid, CellCollection {
        starter_cell_position: Vec3::ZERO,
        parent: None,
    });

    let mut ocg: Vec<OcgEntry> = Vec::new();
    // Occupied set uses the SAME quantise convention as cell.rs after the fix:
    // key = [round(world.x / HALF), round(world.y / HALF), round(world.z / HALF)]
    let mut occupied: HashSet<[i32; 3]> = HashSet::new();

    // ── Build trunk ───────────────────────────────────────────────────────────
    // Collect the arm-cell keys for each layer so we can attach branches later.
    let mut arm_keys_per_layer: Vec<Vec<[i32; 3]>> = Vec::new();

    for layer in 0..TRUNK_LAYERS {
        if ocg.len() >= MAX_TOTAL_CELLS { break; }

        let (axis, arm_keys) = trunk_cross_keys(layer);

        // Centre cell: HardCell for lower trunk, YellowCell higher up
        let centre_type = if layer < TRUNK_LAYERS / 2 { CellType::HardCell } else { CellType::YellowCell };
        try_place(&mut ocg, &mut occupied, cid, axis, centre_type);

        // 4 arm cells
        for &ak in &arm_keys {
            if ocg.len() >= MAX_TOTAL_CELLS { break; }
            try_place(&mut ocg, &mut occupied, cid, ak, centre_type);
        }

        arm_keys_per_layer.push(arm_keys);
    }

    // ── Attach primary branches ───────────────────────────────────────────────
    // Distribute PRIMARY_BRANCH_COUNT branches across upper trunk layers,
    // picking evenly spaced layers and well-separated horizontal directions.
    let top_arm_layer = arm_keys_per_layer.len().saturating_sub(1);
    let bottom_branch_layer = top_arm_layer / 2; // branches only from upper half

    let mut branch_dirs: Vec<[f32; 3]> = Vec::new();

    for b in 0..PRIMARY_BRANCH_COUNT {
        if ocg.len() >= MAX_TOTAL_CELLS { break; }

        // Spread branches evenly across the upper trunk layers.
        let layer_idx = bottom_branch_layer
            + (b * (top_arm_layer - bottom_branch_layer)) / PRIMARY_BRANCH_COUNT;
        let layer_idx = layer_idx.min(top_arm_layer);
        let arm_keys  = &arm_keys_per_layer[layer_idx];

        // Pick a horizontal direction that is well-separated from existing branches.
        let dir = pick_spread_direction(rng, &branch_dirs, MIN_BRANCH_SEPARATION);
        branch_dirs.push(dir);

        // Find the arm cell in this layer whose world position best matches `dir`.
        // This gives a natural attachment point on the correct "side" of the trunk.
        let axis_key = trunk_axis_key(layer_idx);
        let start_key = best_arm_for_direction(arm_keys, axis_key, dir);

        // The branch direction is `dir` (horizontal) blended upward.
        let branch_dir = blend_upward(dir, 0.5);

        let branch_len = rng.random_range(PRIMARY_BRANCH_MIN..=PRIMARY_BRANCH_MAX);
        let tip_key = grow_branch_from_key(
            rng, &mut ocg, &mut occupied, cid,
            start_key, branch_dir, branch_len,
            CellType::YellowCell,
        );

        // ── Sub-branches from the primary branch tip ──────────────────────
        let sub_count = rng.random_range(SUB_BRANCH_COUNT_MIN..=SUB_BRANCH_COUNT_MAX);
        let mut sub_dirs: Vec<[f32; 3]> = Vec::new();
        for _ in 0..sub_count {
            if ocg.len() >= MAX_TOTAL_CELLS { break; }
            let sub_dir = deviate_direction(rng, branch_dir, SUB_BRANCH_ANGLE);
            sub_dirs.push(sub_dir);
            let sub_len = rng.random_range(SUB_BRANCH_MIN..=SUB_BRANCH_MAX);
            grow_branch_from_key(
                rng, &mut ocg, &mut occupied, cid,
                tip_key, sub_dir, sub_len,
                CellType::PhotoCell,
            );
        }
    }

    (collections, ocg)
}

// ── Geometry helpers ──────────────────────────────────────────────────────────

/// Returns the axis key and 4 arm keys for trunk layer `layer`.
/// Axis zigzags: even layers have z_key=0, odd layers have z_key=1.
/// Arms are the 4 horizontal face-neighbours [±1, y, z_arm] of the axis.
fn trunk_axis_key(layer: usize) -> [i32; 3] {
    let y = layer as i32;
    let z = (layer % 2) as i32;
    [0, y, z]
}

fn trunk_cross_keys(layer: usize) -> ([i32; 3], Vec<[i32; 3]>) {
    let [cx, cy, cz] = trunk_axis_key(layer);
    let axis = [cx, cy, cz];
    // The 4 horizontal face-neighbours (y-component = 0) of the axis.
    // These are [±1, cy, cz±1] — all four are face-adjacent to the axis key.
    let arms = vec![
        [cx + 1, cy, cz + 1],
        [cx - 1, cy, cz + 1],
        [cx + 1, cy, cz - 1],
        [cx - 1, cy, cz - 1],
    ];
    (axis, arms)
}

/// Convert a grid key to a world-space Vec3 offset (relative to organism root).
fn key_to_world(key: [i32; 3]) -> Vec3 {
    Vec3::new(key[0] as f32 * HALF, key[1] as f32 * HALF, key[2] as f32 * HALF)
}

/// Quantise a world-space offset to a grid key (must match cell.rs after the fix).
fn quantise(pos: Vec3) -> [i32; 3] {
    [
        (pos.x / HALF).round() as i32,
        (pos.y / HALF).round() as i32,
        (pos.z / HALF).round() as i32,
    ]
}

/// Place a cell if the key is not already occupied and the budget allows.
/// Returns true if placed.
fn try_place(
    ocg:      &mut Vec<OcgEntry>,
    occupied: &mut HashSet<[i32; 3]>,
    id:       CollectionId,
    key:      [i32; 3],
    ct:       CellType,
) -> bool {
    if ocg.len() >= MAX_TOTAL_CELLS { return false; }
    if occupied.contains(&key)       { return false; }
    occupied.insert(key);
    ocg.push(OcgEntry {
        collection_id: id,
        cell_type:     ct,
        offset:        key_to_world(key),
    });
    true
}

/// Pick the FACE_NEIGHBOURS entry whose normalised direction best matches `desired`.
fn best_fn_step(desired: [f32; 3]) -> [i32; 3] {
    let [dx, dy, dz] = desired;
    let mut best_dot = f32::NEG_INFINITY;
    let mut best_fn  = FACE_NEIGHBOURS[0];
    for &fn_vec in &FACE_NEIGHBOURS {
        let [fx, fy, fz] = fn_vec;
        let mag = ((fx*fx + fy*fy + fz*fz) as f32).sqrt();
        let dot = (dx * fx as f32 + dy * fy as f32 + dz * fz as f32) / mag;
        if dot > best_dot {
            best_dot = dot;
            best_fn  = fn_vec;
        }
    }
    best_fn
}

/// Grow a branch of `steps` cells starting from `start_key`, using `direction`.
/// Each step picks the best face-neighbour vector (guaranteed gap-free).
/// Returns the key of the final cell placed.
fn grow_branch_from_key(
    rng:       &mut impl Rng,
    ocg:       &mut Vec<OcgEntry>,
    occupied:  &mut HashSet<[i32; 3]>,
    id:        CollectionId,
    start_key: [i32; 3],
    direction: [f32; 3],
    steps:     usize,
    ct:        CellType,
) -> [i32; 3] {
    let mut current = start_key;
    let mut dir     = direction;

    for _ in 0..steps {
        if ocg.len() >= MAX_TOTAL_CELLS { break; }

        let fn_step = best_fn_step(dir);
        let next = [
            current[0] + fn_step[0],
            current[1] + fn_step[1],
            current[2] + fn_step[2],
        ];

        // If the ideal slot is occupied, try every other FN in dot-product order
        // (prefer the most upward/outward alternatives).
        let placed_key = if !occupied.contains(&next) {
            try_place(ocg, occupied, id, next, ct);
            next
        } else {
            // Sort remaining FNs by dot product with desired direction.
            let mut candidates: Vec<([i32;3], f32)> = FACE_NEIGHBOURS.iter()
                .map(|&fn_vec| {
                    let [fx,fy,fz] = fn_vec;
                    let mag = ((fx*fx+fy*fy+fz*fz) as f32).sqrt();
                    let dot = (dir[0]*fx as f32 + dir[1]*fy as f32 + dir[2]*fz as f32) / mag;
                    ([current[0]+fx, current[1]+fy, current[2]+fz], dot)
                })
                .collect();
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let mut placed = current;
            for (cand_key, _) in candidates {
                if !occupied.contains(&cand_key) {
                    try_place(ocg, occupied, id, cand_key, ct);
                    placed = cand_key;
                    break;
                }
            }
            placed
        };

        current = placed_key;

        // Apply a small random wobble to the direction each step —
        // makes branches curve naturally rather than being perfectly straight.
        dir = deviate_direction(rng, dir, 0.2);
        // Re-enforce upward bias: branches should never droop downward.
        if dir[1] < 0.1 {
            let [dx,_,dz] = dir;
            let mag = (dx*dx + 0.2*0.2 + dz*dz).sqrt();
            dir = [dx/mag, 0.2/mag, dz/mag];
        }
    }

    current
}

/// Pick a horizontal direction that is at least `min_sep` radians away from all
/// directions in `existing`.  Falls back to a random direction after 64 attempts.
fn pick_spread_direction(
    rng:     &mut impl Rng,
    existing: &[[f32; 3]],
    min_sep: f32,
) -> [f32; 3] {
    for _ in 0..64 {
        let angle: f32 = rng.random_range(0.0..std::f32::consts::TAU);
        let candidate = [angle.cos(), 0.0, angle.sin()];
        let ok = existing.iter().all(|&ex| {
            let dot = candidate[0]*ex[0] + candidate[1]*ex[1] + candidate[2]*ex[2];
            dot.clamp(-1.0, 1.0).acos() >= min_sep
        });
        if ok { return candidate; }
    }
    // Fallback: evenly spaced by index
    let angle = rng.random_range(0.0..std::f32::consts::TAU);
    [angle.cos(), 0.0, angle.sin()]
}

/// Blend a horizontal direction with an upward component.
/// `up_weight` of 0.5 means equal horizontal and vertical weight.
fn blend_upward(dir: [f32; 3], up_weight: f32) -> [f32; 3] {
    let [dx, _, dz] = dir;
    let mag = (dx*dx + up_weight*up_weight + dz*dz).sqrt();
    [dx/mag, up_weight/mag, dz/mag]
}

/// Randomly deviate a direction by up to `max_angle` radians.
fn deviate_direction(rng: &mut impl Rng, dir: [f32; 3], max_angle: f32) -> [f32; 3] {
    let [dx, dy, dz] = dir;
    // Build a local frame perpendicular to `dir`.
    let perp1 = if dx.abs() < 0.9 {
        let m = (dy*dy + dz*dz).sqrt().max(1e-6);
        [0.0, dz/m, -dy/m]
    } else {
        let m = (dx*dx + dz*dz).sqrt().max(1e-6);
        [-dz/m, 0.0, dx/m]
    };
    let perp2 = [
        dy*perp1[2] - dz*perp1[1],
        dz*perp1[0] - dx*perp1[2],
        dx*perp1[1] - dy*perp1[0],
    ];

    let phi:   f32 = rng.random_range(0.0..std::f32::consts::TAU);
    let theta: f32 = rng.random_range(0.0..max_angle);

    let nx = dx*theta.cos() + (perp1[0]*phi.cos() + perp2[0]*phi.sin())*theta.sin();
    let ny = dy*theta.cos() + (perp1[1]*phi.cos() + perp2[1]*phi.sin())*theta.sin();
    let nz = dz*theta.cos() + (perp1[2]*phi.cos() + perp2[2]*phi.sin())*theta.sin();
    let mag = (nx*nx + ny*ny + nz*nz).sqrt().max(1e-6);
    [nx/mag, ny/mag, nz/mag]
}

/// Among a set of arm keys, find the one whose world position is most aligned
/// with the given outward direction (relative to the axis).
fn best_arm_for_direction(arm_keys: &[[i32; 3]], axis: [i32; 3], dir: [f32; 3]) -> [i32; 3] {
    let mut best_dot = f32::NEG_INFINITY;
    let mut best_key = arm_keys[0];
    for &ak in arm_keys {
        let offset = [
            (ak[0] - axis[0]) as f32,
            (ak[1] - axis[1]) as f32,
            (ak[2] - axis[2]) as f32,
        ];
        let mag = (offset[0]*offset[0] + offset[1]*offset[1] + offset[2]*offset[2]).sqrt().max(1e-6);
        let dot = (offset[0]*dir[0] + offset[1]*dir[1] + offset[2]*dir[2]) / mag;
        if dot > best_dot {
            best_dot = dot;
            best_key = ak;
        }
    }
    best_key
}
