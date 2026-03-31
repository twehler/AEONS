use bevy::prelude::*;
use bevy::mesh::Indices;
pub use bevy::render::render_resource::PrimitiveTopology;
use bevy::asset::RenderAssetUsages;
use bevy::mesh::skinning::SkinnedMesh;
use bevy::mesh::VertexAttributeValues;
use crate::viewport_settings::ShowGizmo;
use std::collections::HashSet;

pub const GLOBAL_CELL_SIZE: f32 = 3.5;

#[derive(Component)]
pub struct OrganismMesh;


// ── CellType ─────────────────────────────────────────────────────────────────

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub enum CellType {
    BlueCell,
    RedCell,
    GreenCell,
    YellowCell,
    OrangeCell,
    LightBlueCell,
}

impl CellType {
    pub fn color(&self) -> Color {
        match self {
            Self::BlueCell      => Color::from(Srgba::hex("00d0ff").unwrap()),
            Self::RedCell       => Color::from(Srgba::hex("ff0000").unwrap()),
            Self::GreenCell     => Color::from(Srgba::hex("2bff00").unwrap()),
            Self::YellowCell    => Color::from(Srgba::hex("ffff00").unwrap()),
            Self::OrangeCell    => Color::from(Srgba::hex("ff8000").unwrap()),
            Self::LightBlueCell => Color::from(Srgba::hex("00ffee").unwrap()),
        }
    }

    pub fn size(&self) -> f32 {
        match self {
            Self::BlueCell      => GLOBAL_CELL_SIZE,
            Self::RedCell       => GLOBAL_CELL_SIZE,
            Self::GreenCell     => GLOBAL_CELL_SIZE,
            Self::YellowCell    => GLOBAL_CELL_SIZE,
            Self::OrangeCell    => GLOBAL_CELL_SIZE,
            Self::LightBlueCell => GLOBAL_CELL_SIZE,
        }
    }
}


// ── Skinning data ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct CellSkinning {
    pub joint_indices: [u16; 4],
    pub joint_weights: [f32; 4],
}

impl CellSkinning {
    pub fn single(joint: u16) -> Self {
        Self {
            joint_indices: [joint, 0, 0, 0],
            joint_weights: [1.0, 0.0, 0.0, 0.0],
        }
    }

    pub fn blended(primary_joint: u16, neighbour_joint: u16, primary_weight: f32) -> Self {
        Self {
            joint_indices: [primary_joint, neighbour_joint, 0, 0],
            joint_weights: [primary_weight, 1.0 - primary_weight, 0.0, 0.0],
        }
    }
}


// ── Neighbour directions ──────────────────────────────────────────────────────

// A rhombic dodecahedron has 12 faces. Each face has a unique outward direction.
// For a cell at integer grid position, these are the 12 face-neighbour offsets —
// the set of all permutations of (±1, ±1, 0) across the three axes.
// A face points toward neighbour N if N's grid position equals this cell's
// position plus the corresponding offset.
//
// These must match the order of add_face calls in generate_rhombic_dodecahedron_raw
// so that face index i corresponds to FACE_NEIGHBOURS[i].
//
// Derivation: the 12 face centres of a rhombic dodecahedron with cube-corner
// parameter s lie at (±s, ±s, 0), (±s, 0, ±s), (0, ±s, ±s).
// Normalised to grid steps, these become the vectors below.
// Face order matches the add_face call sequence: top cap (4), bottom cap (4),
// middle ring (4).



pub const FACE_NEIGHBOURS: [[i32; 3]; 12] = [
    // Top cap — tip +Z (v12)
    [ 0,  1,  1],  // add_face(12,  0, 10,  4)
    [-1,  0,  1],  // add_face(12,  4,  9,  6)
    [ 0, -1,  1],  // add_face(12,  6, 11,  2)
    [ 1,  0,  1],  // add_face(12,  2,  8,  0)
    // Bottom cap — tip -Z (v13)
    [ 0,  1, -1],  // add_face( 5, 10,  1, 13)
    [-1,  0, -1],  // add_face( 7,  9,  5, 13)
    [ 0, -1, -1],  // add_face( 3, 11,  7, 13)
    [ 1,  0, -1],  // add_face( 1,  8,  3, 13)
    // Middle ring
    [ 1,  1,  0],  // add_face( 1, 10,  0,  8)
    [-1,  1,  0],  // add_face( 5,  9,  4, 10)
    [-1, -1,  0],  // add_face( 7, 11,  6,  9)
    [ 1, -1,  0],  // add_face( 3,  8,  2, 11)
];


// The complete set of all 12 neighbour directions — used for the enclosed-cell
// check (a cell is fully enclosed if all 12 neighbours are occupied).
pub const ALL_NEIGHBOUR_OFFSETS: [[i32; 3]; 12] = FACE_NEIGHBOURS;


// ── Position quantisation ─────────────────────────────────────────────────────

// Converts a continuous mesh-space position to a grid key for HashSet lookup.
// We round to the nearest integer, with a tolerance to handle float imprecision.
// The grid step size is 1.0 (cells are placed at integer offsets in the OCG).
fn quantise(pos: Vec3) -> [i32; 3] {

    let epsilon = 1e-6;
    [
        ((pos.x + epsilon)).floor() as i32,
        ((pos.y + epsilon)).floor() as i32,
        ((pos.z + epsilon)).floor() as i32,
    ]
}


// ── Raw geometry ─────────────────────────────────────────────────────────────

pub struct RawCell {
    pub positions: Vec<[f32; 3]>,
    pub normals:   Vec<[f32; 3]>,
    pub indices:   Vec<u32>,
}

// Generates the visible faces of one rhombic dodecahedron at `pos` (mesh space).
// `occupied` is the full set of quantised cell positions in the organism.
// Faces whose neighbour direction points to an occupied cell are skipped —
// those faces are hidden between two cells and never visible.
pub fn generate_rhombic_dodecahedron_raw(
    pos:      Vec3,
    total_width: f32,
    occupied: &HashSet<[i32; 3]>,
) -> RawCell {
    let s    = total_width / 4.0;
    let self_key = quantise(pos);

    let v = [
        Vec3::new( s,  s,  s), Vec3::new( s,  s, -s),  // 0, 1
        Vec3::new( s, -s,  s), Vec3::new( s, -s, -s),  // 2, 3
        Vec3::new(-s,  s,  s), Vec3::new(-s,  s, -s),  // 4, 5
        Vec3::new(-s, -s,  s), Vec3::new(-s, -s, -s),  // 6, 7
        Vec3::new( 2.0*s,  0.0,    0.0  ),              // 8  +X tip
        Vec3::new(-2.0*s,  0.0,    0.0  ),              // 9  -X tip
        Vec3::new( 0.0,    2.0*s,  0.0  ),              // 10 +Y tip
        Vec3::new( 0.0,   -2.0*s,  0.0  ),              // 11 -Y tip
        Vec3::new( 0.0,    0.0,    2.0*s),              // 12 +Z tip
        Vec3::new( 0.0,    0.0,   -2.0*s),              // 13 -Z tip
    ];

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals:   Vec<[f32; 3]> = Vec::new();
    let mut indices:   Vec<u32>      = Vec::new();
    let mut face_idx:  usize         = 0;

    // Emits one diamond face only if the neighbour in that face's direction
    // is NOT occupied. If it is occupied, that face is internal — skip it.
    let mut add_face = |p1: usize, p2: usize, p3: usize, p4: usize,
                        positions: &mut Vec<[f32;3]>,
                        normals: &mut Vec<[f32;3]>,
                        indices: &mut Vec<u32>,
                        face_idx: usize| {
        let neighbour_offset = FACE_NEIGHBOURS[face_idx];
        let neighbour_key = [
            self_key[0] + neighbour_offset[0],
            self_key[1] + neighbour_offset[1],
            self_key[2] + neighbour_offset[2],
        ];

        // Skip this face entirely if the neighbour cell is present
        if occupied.contains(&neighbour_key) {
            return;
        }

        let pts  = [v[p1], v[p2], v[p3], v[p4]];
        let norm = (pts[1] - pts[0]).cross(pts[2] - pts[0]).normalize();
        let base = positions.len() as u32;
        for p in &pts {
            positions.push((*p + pos).into());
            normals.push(norm.into());
        }
        indices.extend([base, base+1, base+2, base, base+2, base+3]);
    };

    // Top cap
    add_face(12,  0, 10,  4, &mut positions, &mut normals, &mut indices, face_idx); face_idx += 1;
    add_face(12,  4,  9,  6, &mut positions, &mut normals, &mut indices, face_idx); face_idx += 1;
    add_face(12,  6, 11,  2, &mut positions, &mut normals, &mut indices, face_idx); face_idx += 1;
    add_face(12,  2,  8,  0, &mut positions, &mut normals, &mut indices, face_idx); face_idx += 1;
    // Bottom cap
    add_face( 5, 10,  1, 13, &mut positions, &mut normals, &mut indices, face_idx); face_idx += 1;
    add_face( 7,  9,  5, 13, &mut positions, &mut normals, &mut indices, face_idx); face_idx += 1;
    add_face( 3, 11,  7, 13, &mut positions, &mut normals, &mut indices, face_idx); face_idx += 1;
    add_face( 1,  8,  3, 13, &mut positions, &mut normals, &mut indices, face_idx); face_idx += 1;
    // Middle ring
    add_face( 1, 10,  0,  8, &mut positions, &mut normals, &mut indices, face_idx); face_idx += 1;
    add_face( 5,  9,  4, 10, &mut positions, &mut normals, &mut indices, face_idx); face_idx += 1;
    add_face( 7, 11,  6,  9, &mut positions, &mut normals, &mut indices, face_idx); face_idx += 1;
    add_face( 3,  8,  2, 11, &mut positions, &mut normals, &mut indices, face_idx);

    RawCell { positions, normals, indices }
}

// Standalone non-skinned mesh — no culling, used for collision shapes / previews.
pub fn generate_rhombic_dodecahedron(pos: Vec3, total_width: f32) -> Mesh {
    // Pass an empty occupied set — all faces visible, no neighbours assumed
    let raw = generate_rhombic_dodecahedron_raw(pos, total_width, &HashSet::new());
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, raw.positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL,   raw.normals);
    mesh.insert_indices(Indices::U32(raw.indices));
    mesh
}


// ── Mesh merging ─────────────────────────────────────────────────────────────

pub struct MeshCell {
    pub mesh_space_pos: Vec3,
    pub cell_type:      CellType,
}

pub fn merge_organism_mesh(cells: Vec<MeshCell>) -> Mesh {
    // ── Build occupied set ────────────────────────────────────────────────────
    // Quantise every cell's mesh-space position into grid coordinates.
    // This is the O(1) lookup used by each face's neighbour check.
    let occupied: HashSet<[i32; 3]> = cells
        .iter()
        .map(|c| quantise(c.mesh_space_pos))
        .collect();

    let mut positions:     Vec<[f32; 3]> = Vec::new();
    let mut normals:       Vec<[f32; 3]> = Vec::new();
    let mut colors:        Vec<[f32; 4]> = Vec::new();
    let mut indices:       Vec<u32>      = Vec::new();

    for cell in &cells {
        // ── Enclosed cell check ───────────────────────────────────────────────
        // If all 12 face-neighbours are occupied, this cell is completely
        // surrounded and has no visible faces — skip it entirely.
        let self_key = quantise(cell.mesh_space_pos);
        let fully_enclosed = ALL_NEIGHBOUR_OFFSETS.iter().all(|offset| {
            let neighbour_key = [
                self_key[0] + offset[0],
                self_key[1] + offset[1],
                self_key[2] + offset[2],
            ];
            occupied.contains(&neighbour_key)
        });

        if fully_enclosed {
            continue; // zero vertices, zero indices — cell is invisible
        }

        // ── Generate visible faces only ───────────────────────────────────────
       let base = positions.len() as u32;
        let raw  = generate_rhombic_dodecahedron_raw(
            cell.mesh_space_pos,
            cell.cell_type.size(),
            &occupied,
        );
        if raw.positions.is_empty() { continue; }

        let c     = cell.cell_type.color().to_linear();
        let color = [c.red, c.green, c.blue, 1.0];
        let vert_count = raw.positions.len();

        colors.extend(std::iter::repeat_n(color, vert_count));
        positions.extend_from_slice(&raw.positions);
        normals.extend_from_slice(&raw.normals);
        indices.extend(raw.indices.iter().map(|i| i + base));
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL,   normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR,    colors);
    mesh.insert_indices(Indices::U32(indices));
    mesh
} 
