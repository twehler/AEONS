use bevy::prelude::Vec3;

#[allow(dead_code)]
pub const VERT_COUNT: usize = 14;

// ── Lattice slot directions ───────────────────────────────────────────────────
//
// A rhombic dodecahedron tiles 3-space exactly. Neighbouring cell centres lie
// at  centre + SLOT_DIRS[i] * center_scale(edge).
//
// The 18 slots split into two groups:
//   • 6 axis-aligned  (magnitude 1.0)  – next-nearest BCC neighbours
//   • 12 face-diagonal (magnitude √0.5) – nearest  FCC neighbours
//
// All 18 together span the full coordination shell of the RD tiling.

pub const SLOT_DIRS: [Vec3; 18] = [
    // 6 axis-aligned
    Vec3::new( 1.0,  0.0,  0.0),
    Vec3::new( 0.0,  1.0,  0.0),
    Vec3::new( 0.0,  0.0,  1.0),
    Vec3::new(-1.0,  0.0,  0.0),
    Vec3::new( 0.0, -1.0,  0.0),
    Vec3::new( 0.0,  0.0, -1.0),
    // 12 face-diagonal (FCC nearest neighbours)
    Vec3::new( 0.0,  0.5,  0.5),
    Vec3::new( 0.5,  0.5,  0.0),
    Vec3::new( 0.5,  0.0,  0.5),
    Vec3::new( 0.0, -0.5,  0.5),
    Vec3::new( 0.0,  0.5, -0.5),
    Vec3::new( 0.5,  0.0, -0.5),
    Vec3::new(-0.5,  0.0,  0.5),
    Vec3::new( 0.5, -0.5,  0.0),
    Vec3::new(-0.5,  0.5,  0.0),
    Vec3::new(-0.5, -0.5,  0.0),
    Vec3::new(-0.5,  0.0, -0.5),
    Vec3::new( 0.0, -0.5, -0.5),
];

/// Actual centre-to-centre displacement = `SLOT_DIRS[i] * center_scale(edge)`.
/// Derived from the RD geometry: the 12 face-adjacent centres are at distance
/// `2 * edge * √(2/3)` and the 6 axis-adjacent ones at `4 * edge / √3`;
/// both equal `edge * center_scale(1)` when the slot direction is pre-scaled
/// so that `SLOT_DIRS` magnitudes are 1 or √0.5 respectively.
pub fn center_scale(edge: f32) -> f32 {
    4.0 * edge / 3.0_f32.sqrt()
}

// ── Seed (initial) dodecahedron ───────────────────────────────────────────────

/// 14 vertices of the seed rhombic dodecahedron (axis-aligned, centred at
/// origin). Scaled so edge length = `edge`.
/// Indices 0–7: cube corners (type A). Indices 8–13: face centres (type B).
pub fn seed_vertices(edge: f32) -> Vec<Vec3> {
    let s = edge / 3.0_f32.sqrt();
    let t = 2.0 * s;
    vec![
        Vec3::new( s,  s,  s), // 0  A0
        Vec3::new( s,  s, -s), // 1  A1
        Vec3::new( s, -s,  s), // 2  A2
        Vec3::new( s, -s, -s), // 3  A3
        Vec3::new(-s,  s,  s), // 4  A4
        Vec3::new(-s,  s, -s), // 5  A5
        Vec3::new(-s, -s,  s), // 6  A6
        Vec3::new(-s, -s, -s), // 7  A7
        Vec3::new( t, 0.0, 0.0), // 8  B+x
        Vec3::new(-t, 0.0, 0.0), // 9  B-x
        Vec3::new(0.0,  t, 0.0), // 10 B+y
        Vec3::new(0.0, -t, 0.0), // 11 B-y
        Vec3::new(0.0, 0.0,  t), // 12 B+z
        Vec3::new(0.0, 0.0, -t), // 13 B-z
    ]
}

/// 12 rhombic quads in CCW winding (viewed from outside). Relative indices 0–13.
pub fn seed_quads() -> Vec<[u32; 4]> {
    vec![
        [ 0,  8,  1, 10], // normal +x+y
        [ 0, 12,  2,  8], // normal +x+z
        [ 0, 10,  4, 12], // normal +y+z
        [ 1,  8,  3, 13], // normal +x-z
        [ 1, 13,  5, 10], // normal +y-z
        [ 2, 11,  3,  8], // normal +x-y
        [ 2, 12,  6, 11], // normal -y+z
        [ 3, 11,  7, 13], // normal -y-z
        [ 4, 10,  5,  9], // normal -x+y
        [ 4,  9,  6, 12], // normal -x+z
        [ 5, 13,  7,  9], // normal -x-z
        [ 6,  9,  7, 11], // normal -x-y
    ]
}

// ── Appended-cell dodecahedron ────────────────────────────────────────────────


/// Vertices of a cell dodecahedron placed at `center` in the canonical
/// axis-aligned orientation (same rotation as the seed). This guarantees
/// zero-gap space-filling: all cells share the same lattice frame.
pub fn cell_vertices(center: Vec3, _attach_normal: Vec3, edge: f32) -> Vec<Vec3> {
    seed_vertices(edge).iter().map(|&v| center + v).collect()
}

/// All 24 triangles of a dodecahedron cell. These use the same relative indices
/// (0–13) as `seed_quads`. Splitting each quad into two triangles.
pub fn cell_all_tris() -> Vec<[u32; 3]> {
    seed_quads()
        .iter()
        .flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]])
        .collect()
}

#[allow(dead_code)]
/// Index into `cell_all_tris()` of the triangle facing most toward
/// `-attach_normal` (the interior/base face after merging).
pub fn cell_base_tri_idx(cell_verts: &[Vec3], attach_normal: Vec3) -> usize {
    let target = -attach_normal;
    cell_all_tris()
        .iter()
        .enumerate()
        .max_by(|(_, ta), (_, tb)| {
            let na = face_normal(ta, cell_verts);
            let nb = face_normal(tb, cell_verts);
            na.dot(target).partial_cmp(&nb.dot(target)).unwrap()
        })
        .map(|(i, _)| i)
        .unwrap()
}

fn face_normal(tri: &[u32; 3], verts: &[Vec3]) -> Vec3 {
    let a = verts[tri[0] as usize];
    let b = verts[tri[1] as usize];
    let c = verts[tri[2] as usize];
    (b - a).cross(c - a).normalize_or_zero()
}
