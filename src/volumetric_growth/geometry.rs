use bevy::asset::RenderAssetUsages;
use bevy::mesh::PrimitiveTopology;
use bevy::prelude::*;

/// Fan-triangulate an array of quads into triangles.
pub fn quads_to_tris(quads: &[[u32; 4]]) -> Vec<[u32; 3]> {
    quads
        .iter()
        .flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]])
        .collect()
}

/// Returns `[a,b,c]` with winding chosen so the flat normal points away from
/// `interior` (the approximate mesh centroid).
pub fn outward_tri(a: u32, b: u32, c: u32, verts: &[Vec3], interior: Vec3) -> [u32; 3] {
    let av = verts[a as usize];
    let bv = verts[b as usize];
    let cv = verts[c as usize];
    let n = (bv - av).cross(cv - av);
    if n.dot((av + bv + cv) / 3.0 - interior) >= 0.0 {
        [a, b, c]
    } else {
        [a, c, b]
    }
}

#[allow(dead_code)]
/// Bijective best-match: returns permutation `p` of {0,1,2} s.t. `cell[i]` is
/// paired with `face[p[i]]`, minimising total squared distance.
pub fn best_matching(cell: &[Vec3], face: &[Vec3]) -> [usize; 3] {
    const PERMS: [[usize; 3]; 6] = [
        [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0],
    ];
    PERMS
        .iter()
        .copied()
        .min_by(|p, q| {
            let cp: f32 = (0..3).map(|i| (cell[i] - face[p[i]]).length_squared()).sum();
            let cq: f32 = (0..3).map(|i| (cell[i] - face[q[i]]).length_squared()).sum();
            cp.partial_cmp(&cq).unwrap()
        })
        .unwrap()
}

/// Flat-shaded, non-indexed triangle-list mesh.
pub fn build_flat_mesh(verts: &[Vec3], tris: &[[u32; 3]]) -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    for &[a, b, c] in tris {
        let av = verts[a as usize];
        let bv = verts[b as usize];
        let cv = verts[c as usize];
        let n = (bv - av).cross(cv - av).normalize_or_zero().to_array();
        for p in [av, bv, cv] {
            positions.push(p.to_array());
            normals.push(n);
        }
    }
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh
}
