//! Mesh rebuild via Ball-Pivoting + Taubin smoothing.
//!
//! Pipeline:
//!   1. Estimate per-vertex normals from a local k-NN centroid plane,
//!      sign-flipped to point away from the cloud centroid.
//!   2. Ball-pivoting reconstruction at radius `r ≈ mean_nn_distance × factor`.
//!   3. If BPA fails to close the surface, fall back to the star-shaped
//!      convex-hull rebuild (which always produces a closed manifold).
//!   4. `TAUBIN_PASSES` of Taubin smoothing (one λ + one μ each) to round out
//!      triangulation kinks without shrinking the mesh. Vertices below
//!      `frozen_vertex_count` are held in place to prevent cumulative drift
//!      across rebuilds.

use bevy::prelude::*;

use super::{
    BPA_RADIUS_FACTOR, EPS, Face, FROZEN_VERTEX_TAUBIN_RATE, NORMAL_ESTIMATION_KNN,
    TARGET_EDGE_LEN, TAUBIN_LAMBDA, TAUBIN_MU, TAUBIN_PASSES, Topology, mesh_centroid,
};
use super::rebuild_starshaped::rebuild_as_convex_hull;

pub(super) fn rebuild_mesh(topo: &mut Topology) {
    if topo.vertices.len() < 4 {
        return;
    }
    let r = mean_nearest_neighbor_distance(&topo.vertices) * BPA_RADIUS_FACTOR;
    let normals = estimate_vertex_normals(&topo.vertices, NORMAL_ESTIMATION_KNN);

    let bpa_faces = ball_pivot(&topo.vertices, &normals, r);
    if !bpa_faces.is_empty() && faces_are_closed_manifold(&bpa_faces) {
        topo.faces = bpa_faces;
    } else {
        // BPA failed to seal the cloud — fall back to the always-watertight
        // star-shaped reconstruction. This guarantees the no-holes invariant.
        rebuild_as_convex_hull(topo);
    }

    // Drop vertices that no face references after the rebuild. OCG entries
    // for the dropped vertices are *kept* (with their last-known positions)
    // so the historical ledger stays complete.
    compact_unreferenced_vertices(topo);

    let frozen_below = topo.frozen_vertex_count;
    for _ in 0..TAUBIN_PASSES {
        taubin_smooth_pass(topo, TAUBIN_LAMBDA, frozen_below);
        taubin_smooth_pass(topo, TAUBIN_MU, frozen_below);
    }
    // After this rebuild, every vertex now in the mesh becomes part of the
    // immutable old surface; only future growth (vertices appended after
    // this point) will be smoothable on the next pulse.
    topo.frozen_vertex_count = topo.vertices.len();
    // Push the new (smoothed) live positions back into the OCG ledger.
    topo.sync_ocg_positions();
    // Smoothing moved vertices, so face normals must be recomputed.
    let centroid = mesh_centroid(&topo.vertices);
    for face in &mut topo.faces {
        let pa = topo.vertices[face.vertices[0] as usize];
        let pb = topo.vertices[face.vertices[1] as usize];
        let pc = topo.vertices[face.vertices[2] as usize];
        let nrm = (pb - pa).cross(pc - pa).normalize_or_zero();
        let fc = (pa + pb + pc) / 3.0;
        if (fc - centroid).dot(nrm) < 0.0 {
            face.vertices.swap(1, 2);
            face.normal = -nrm;
        } else {
            face.normal = nrm;
        }
    }
}

/// Mean distance from each vertex to its nearest neighbor — a robust scale
/// estimate for the local sample density.
fn mean_nearest_neighbor_distance(verts: &[Vec3]) -> f32 {
    if verts.len() < 2 {
        return TARGET_EDGE_LEN;
    }
    let mut total = 0.0;
    for i in 0..verts.len() {
        let mut min_d = f32::INFINITY;
        for j in 0..verts.len() {
            if i == j {
                continue;
            }
            let d = (verts[j] - verts[i]).length();
            if d < min_d {
                min_d = d;
            }
        }
        total += min_d;
    }
    total / verts.len() as f32
}

/// k indices of the vertices closest to `verts[idx]`, excluding `idx` itself.
fn k_nearest(verts: &[Vec3], idx: usize, k: usize) -> Vec<usize> {
    let p = verts[idx];
    let mut dists: Vec<(usize, f32)> = (0..verts.len())
        .filter(|&i| i != idx)
        .map(|i| (i, (verts[i] - p).length_squared()))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.iter().take(k).map(|&(i, _)| i).collect()
}

/// Per-vertex outward normal estimate. We approximate the local-PCA "smallest
/// eigenvector" by the direction from the local k-NN centroid to the vertex,
/// then sign-flip it to point away from the global centroid. Cheap and
/// adequate for our roughly spherical clouds.
fn estimate_vertex_normals(verts: &[Vec3], k: usize) -> Vec<Vec3> {
    let global_c = mesh_centroid(verts);
    (0..verts.len())
        .map(|i| {
            let neighbors = k_nearest(verts, i, k);
            let mut sum = verts[i];
            for &j in &neighbors {
                sum += verts[j];
            }
            let local_c = sum / (neighbors.len() as f32 + 1.0);
            let outward = (verts[i] - global_c).normalize_or_zero();
            let mut n = (verts[i] - local_c).normalize_or_zero();
            if n.length_squared() < EPS {
                n = outward; // degenerate: fall back to radial direction.
            }
            if n.dot(outward) < 0.0 {
                -n
            } else {
                n
            }
        })
        .collect()
}

/// Center of the sphere of radius `r` passing through three points, on the
/// side aligned with `normal_hint`. Returns `None` if the three points'
/// circumradius exceeds `r` (no such sphere exists).
fn sphere_center(a: Vec3, b: Vec3, c: Vec3, r: f32, normal_hint: Vec3) -> Option<Vec3> {
    let ab = b - a;
    let ac = c - a;
    let n = ab.cross(ac);
    let n_sq = n.length_squared();
    if n_sq < 1.0e-10 {
        return None;
    }
    let to_cc = (n.cross(ab) * ac.length_squared() + ac.cross(n) * ab.length_squared())
        / (2.0 * n_sq);
    let cc = a + to_cc;
    let rc_sq = to_cc.length_squared();
    if rc_sq > r * r + EPS {
        return None;
    }
    let h = (r * r - rc_sq).max(0.0).sqrt();
    let nhat = n / n_sq.sqrt();
    if nhat.dot(normal_hint) >= 0.0 {
        Some(cc + nhat * h)
    } else {
        Some(cc - nhat * h)
    }
}

fn sphere_is_empty(verts: &[Vec3], center: Vec3, r: f32, exclude: [usize; 3]) -> bool {
    let r_sq = r * r;
    for (i, &v) in verts.iter().enumerate() {
        if exclude.contains(&i) {
            continue;
        }
        if (v - center).length_squared() < r_sq - EPS {
            return false;
        }
    }
    true
}

fn canonical_edge(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

fn faces_are_closed_manifold(faces: &[Face]) -> bool {
    use std::collections::HashMap;
    if faces.is_empty() {
        return false;
    }
    let mut edge_count: HashMap<(u32, u32), u32> = HashMap::new();
    for face in faces {
        let k = face.vertices.len();
        for i in 0..k {
            let a = face.vertices[i];
            let b = face.vertices[(i + 1) % k];
            *edge_count.entry(canonical_edge(a, b)).or_insert(0) += 1;
        }
    }
    edge_count.values().all(|&c| c == 2)
}

fn find_seed_triangle(verts: &[Vec3], normals: &[Vec3], r: f32) -> Option<(u32, u32, u32)> {
    let r_sq_max = 4.0 * r * r;
    let neighbor_count = NORMAL_ESTIMATION_KNN.max(8);
    for i in 0..verts.len() {
        let knn = k_nearest(verts, i, neighbor_count);
        for (idx_j, &j) in knn.iter().enumerate() {
            if (verts[j] - verts[i]).length_squared() > r_sq_max {
                continue;
            }
            if normals[i].dot(normals[j]) < 0.0 {
                continue;
            }
            for &k in knn.iter().skip(idx_j + 1) {
                if (verts[k] - verts[i]).length_squared() > r_sq_max {
                    continue;
                }
                if (verts[k] - verts[j]).length_squared() > r_sq_max {
                    continue;
                }
                if normals[i].dot(normals[k]) < 0.0 || normals[j].dot(normals[k]) < 0.0 {
                    continue;
                }
                let avg_n = (normals[i] + normals[j] + normals[k]).normalize_or_zero();
                if let Some(center) = sphere_center(verts[i], verts[j], verts[k], r, avg_n) {
                    if sphere_is_empty(verts, center, r, [i, j, k]) {
                        // Orient triangle so its geometric normal aligns with avg_n.
                        let geom_n = (verts[j] - verts[i]).cross(verts[k] - verts[i]);
                        if geom_n.dot(avg_n) >= 0.0 {
                            return Some((i as u32, j as u32, k as u32));
                        } else {
                            return Some((i as u32, k as u32, j as u32));
                        }
                    }
                }
            }
        }
    }
    None
}

/// For boundary edge `(a, b)` adjacent to face whose third vertex is
/// `c_existing`, find the next vertex to roll the ball onto. The candidate
/// minimizes the pivot angle around the edge, has an empty supporting ball,
/// and produces a triangle on the *outward* side of the existing face.
fn find_pivot_vertex(
    verts: &[Vec3],
    normals: &[Vec3],
    r: f32,
    a: u32,
    b: u32,
    c_existing: u32,
) -> Option<u32> {
    let pa = verts[a as usize];
    let pb = verts[b as usize];
    let mid = (pa + pb) * 0.5;
    let axis = (pb - pa).normalize_or_zero();

    let avg_n_existing =
        (normals[a as usize] + normals[b as usize] + normals[c_existing as usize])
            .normalize_or_zero();
    let c0 = sphere_center(pa, pb, verts[c_existing as usize], r, avg_n_existing)?;

    let r_sq_max = 4.0 * r * r;
    let mut best: Option<(u32, f32)> = None;
    for i in 0..verts.len() {
        let iu = i as u32;
        if iu == a || iu == b || iu == c_existing {
            continue;
        }
        let pv = verts[i];
        if (pv - pa).length_squared() > r_sq_max {
            continue;
        }
        if (pv - pb).length_squared() > r_sq_max {
            continue;
        }
        // The pivoted ball lives on the opposite side of the edge from `c0`.
        let to_c0 = c0 - mid;
        let to_v = pv - mid;
        let to_v_perp = to_v - axis.dot(to_v) * axis;
        if to_c0.dot(to_v_perp) > 0.0 {
            continue; // candidate is on c0's side — would re-create existing face.
        }
        let avg_n_new =
            (normals[a as usize] + normals[b as usize] + normals[i]).normalize_or_zero();
        let Some(cv) = sphere_center(pa, pb, pv, r, avg_n_new) else {
            continue;
        };
        if !sphere_is_empty(verts, cv, r, [a as usize, b as usize, i]) {
            continue;
        }
        let v0_perp = (to_c0 - axis.dot(to_c0) * axis).normalize_or_zero();
        let v1_perp = ((cv - mid) - axis.dot(cv - mid) * axis).normalize_or_zero();
        let cos_theta = v0_perp.dot(v1_perp).clamp(-1.0, 1.0);
        let cross = v0_perp.cross(v1_perp).dot(axis);
        // Signed angle around `axis` in [0, 2π); smaller = pivots first.
        let theta = if cross >= 0.0 {
            cos_theta.acos()
        } else {
            2.0 * std::f32::consts::PI - cos_theta.acos()
        };
        if best.map_or(true, |(_, bt)| theta < bt) {
            best = Some((iu, theta));
        }
    }
    best.map(|(i, _)| i)
}

/// Drop vertices that no face references, remap face indices to the
/// compacted vertex array, and shrink `ocg_id_for_vertex` and
/// `frozen_vertex_count` in lockstep. The OCG ledger itself is untouched —
/// dropped vertices keep their entry there as a historical record.
fn compact_unreferenced_vertices(topo: &mut Topology) {
    let n = topo.vertices.len();
    let mut referenced = vec![false; n];
    for face in &topo.faces {
        for &idx in &face.vertices {
            referenced[idx as usize] = true;
        }
    }
    if referenced.iter().all(|&r| r) {
        return; // common case: nothing to drop.
    }
    let mut remap: Vec<Option<u32>> = vec![None; n];
    let mut new_vertices: Vec<Vec3> = Vec::with_capacity(n);
    let mut new_ocg_id: Vec<u32> = Vec::with_capacity(n);
    let mut new_frozen_count: usize = 0;
    for i in 0..n {
        if referenced[i] {
            remap[i] = Some(new_vertices.len() as u32);
            new_vertices.push(topo.vertices[i]);
            new_ocg_id.push(topo.ocg_id_for_vertex[i]);
            if i < topo.frozen_vertex_count {
                new_frozen_count += 1;
            }
        }
    }
    for face in &mut topo.faces {
        for v in &mut face.vertices {
            *v = remap[*v as usize].expect("face references unmapped vertex");
        }
    }
    topo.vertices = new_vertices;
    topo.ocg_id_for_vertex = new_ocg_id;
    topo.frozen_vertex_count = new_frozen_count;
}

fn ball_pivot(verts: &[Vec3], normals: &[Vec3], r: f32) -> Vec<Face> {
    use std::collections::{HashMap, VecDeque};

    let Some((s0, s1, s2)) = find_seed_triangle(verts, normals, r) else {
        return Vec::new();
    };

    let avg_n = (normals[s0 as usize] + normals[s1 as usize] + normals[s2 as usize])
        .normalize_or_zero();
    let geom = (verts[s1 as usize] - verts[s0 as usize])
        .cross(verts[s2 as usize] - verts[s0 as usize])
        .normalize_or_zero();
    let face_normal = if geom.dot(avg_n) >= 0.0 { geom } else { -geom };

    let mut faces: Vec<Face> = vec![Face {
        vertices: vec![s0, s1, s2],
        normal: face_normal,
    }];
    let mut edge_count: HashMap<(u32, u32), u32> = HashMap::new();
    let bump = |ec: &mut HashMap<(u32, u32), u32>, a: u32, b: u32| {
        *ec.entry(canonical_edge(a, b)).or_insert(0) += 1;
    };
    bump(&mut edge_count, s0, s1);
    bump(&mut edge_count, s1, s2);
    bump(&mut edge_count, s2, s0);

    // Boundary queue: each entry is a directed edge (a, b) whose adjacent
    // face has `c_existing` as its third vertex.
    let mut queue: VecDeque<(u32, u32, u32)> = VecDeque::new();
    queue.push_back((s0, s1, s2));
    queue.push_back((s1, s2, s0));
    queue.push_back((s2, s0, s1));

    while let Some((a, b, c_existing)) = queue.pop_front() {
        if *edge_count.get(&canonical_edge(a, b)).unwrap_or(&0) >= 2 {
            continue;
        }
        let Some(v) = find_pivot_vertex(verts, normals, r, a, b, c_existing) else {
            continue;
        };
        // Adding triangle (b, a, v) shares edge (a, b) with the existing face.
        let pa = verts[a as usize];
        let pb = verts[b as usize];
        let pv = verts[v as usize];
        let avg_n =
            (normals[a as usize] + normals[b as usize] + normals[v as usize]).normalize_or_zero();
        let geom = (pa - pb).cross(pv - pb).normalize_or_zero();
        let nrm = if geom.dot(avg_n) >= 0.0 { geom } else { -geom };
        faces.push(Face {
            vertices: vec![b, a, v],
            normal: nrm,
        });
        bump(&mut edge_count, a, b);
        bump(&mut edge_count, b, v);
        bump(&mut edge_count, v, a);
        if *edge_count.get(&canonical_edge(b, v)).unwrap_or(&0) < 2 {
            queue.push_back((b, v, a));
        }
        if *edge_count.get(&canonical_edge(v, a)).unwrap_or(&0) < 2 {
            queue.push_back((v, a, b));
        }
    }
    faces
}

/// One-sided Laplacian smoothing pass with weight `weight` (positive for the
/// shrinking pass, negative for the anti-shrink pass — together they form
/// Taubin smoothing). Vertices with index `< frozen_below` smooth at a
/// reduced rate (`FROZEN_VERTEX_TAUBIN_RATE`) so the previously-smoothed
/// surface flexes just enough to absorb a new growth layer without forming
/// a hard seam at the freeze boundary.
fn taubin_smooth_pass(topo: &mut Topology, weight: f32, frozen_below: usize) {
    use std::collections::HashSet;
    let n = topo.vertices.len();
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut seen: Vec<HashSet<u32>> = vec![HashSet::new(); n];
    for face in &topo.faces {
        let k = face.vertices.len();
        for i in 0..k {
            let a = face.vertices[i];
            let b = face.vertices[(i + 1) % k];
            if seen[a as usize].insert(b) {
                adj[a as usize].push(b);
            }
            if seen[b as usize].insert(a) {
                adj[b as usize].push(a);
            }
        }
    }
    let new_pos: Vec<Vec3> = (0..n)
        .map(|i| {
            let neighbors = &adj[i];
            if neighbors.is_empty() {
                return topo.vertices[i];
            }
            let centroid: Vec3 = neighbors
                .iter()
                .map(|&j| topo.vertices[j as usize])
                .sum::<Vec3>()
                / neighbors.len() as f32;
            let rate = if i < frozen_below {
                FROZEN_VERTEX_TAUBIN_RATE
            } else {
                1.0
            };
            topo.vertices[i] + rate * weight * (centroid - topo.vertices[i])
        })
        .collect();
    topo.vertices = new_pos;
}
