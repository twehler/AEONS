use bevy::prelude::Vec3;
use std::collections::HashSet;

/// Blender-style Jacobi vertex smoothing (bmo_smooth_vert_exec).
///
/// Builds a one-ring adjacency from `tris`, then for `iterations` rounds:
///   new_pos[v] = (1 − lambda) · v + lambda · mean(neighbours(v))
/// All vertices are smoothed; the two-pass (Jacobi) structure ensures
/// every vertex in a round sees old positions, not partially-updated ones.
pub fn smooth_vertices(verts: &mut Vec<Vec3>, tris: &[[u32; 3]], lambda: f32, iterations: usize) {
    let n = verts.len();
    if n == 0 || tris.is_empty() {
        return;
    }

    // Fix E: use HashSet<u32> per vertex — O(1) insert vs O(degree) Vec::contains.
    let mut neighbours: Vec<HashSet<u32>> = vec![HashSet::new(); n];
    for &[a, b, c] in tris {
        for (p, q) in [(a, b), (b, c), (c, a)] {
            neighbours[p as usize].insert(q);
            neighbours[q as usize].insert(p);
        }
    }

    let mut new_pos = vec![Vec3::ZERO; n];

    for _ in 0..iterations {
        // First pass: compute new positions from OLD verts (Jacobi).
        for (i, nbrs) in neighbours.iter().enumerate() {
            if nbrs.is_empty() {
                new_pos[i] = verts[i];
                continue;
            }
            let centroid: Vec3 =
                nbrs.iter().map(|&j| verts[j as usize]).sum::<Vec3>() / nbrs.len() as f32;
            new_pos[i] = verts[i].lerp(centroid, lambda);
        }

        // Second pass: write back.
        verts.copy_from_slice(&new_pos);
    }
}
