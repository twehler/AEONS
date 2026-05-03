//! Growth algorithm: face selection and unit-distance capping.
//!
//! Each tick the system picks one face and replaces it with a fan of triangles
//! around a new apex placed at unit distance from each of the face's vertices.
//! Selection is biased by `TIP_BIAS` and `BRANCHING` (see top-level constants);
//! capping is the topological core that keeps the surface a closed manifold.

use bevy::prelude::*;

use super::{
    ADMISSIBILITY_FLOOR, BRANCHING, BRANCHING_SHARPNESS, EPS, Face, MAX_4_CANDIDATES,
    OcgEntry, TARGET_EDGE_LEN, TIP_BIAS, Topology, mesh_centroid,
};
use super::grow_to_sky::passes_upward_filter;

pub(super) struct GrowthCandidate {
    pub(super) face_idx: usize,
    pub(super) apex: Vec3,
}

/// Iterate every existing face, compute its outward unit-distance apex (if any),
/// and pick a TIP_BIAS-weighted valid candidate. Prefer 4-connect (quad) caps
/// over 3-connect.
pub(super) fn find_growth_candidate(topo: &mut Topology) -> Option<GrowthCandidate> {
    let mut cands_4: Vec<GrowthCandidate> = Vec::new();
    let mut cands_3: Vec<GrowthCandidate> = Vec::new();

    for (face_idx, face) in topo.faces.iter().enumerate() {
        let Some(apex) = compute_outward_apex(face, &topo.vertices) else {
            continue;
        };
        if !is_position_admissible(&topo.vertices, apex) {
            continue;
        }
        if !passes_upward_filter(face, &topo.vertices, apex) {
            continue;
        }
        let cand = GrowthCandidate { face_idx, apex };
        if face.vertices.len() == 4 {
            cands_4.push(cand);
            if cands_4.len() >= MAX_4_CANDIDATES {
                break;
            }
        } else {
            cands_3.push(cand);
        }
    }

    if !cands_4.is_empty() {
        return Some(pick_weighted(&mut cands_4, topo));
    }
    if !cands_3.is_empty() {
        return Some(pick_weighted(&mut cands_3, topo));
    }
    None
}

/// Weighted draw over candidate caps. The score per candidate is
/// `g = (apex - mesh_centroid) · face_normal` — the *outward radial gain* the
/// cap contributes to the advancing front. Pocket faces (whose normal points
/// back into the cluster) have `g ≤ 0`; we damp them by `BRANCHING` so they
/// vanish at `BRANCHING = 0` (sphere mode) and stay first-class at
/// `BRANCHING = 1` (current behavior). Among outward faces we use the existing
/// exp-bias on `(g - mean_g)`:
/// - `effective_bias = TIP_BIAS - (1 - BRANCHING) · BRANCHING_SHARPNESS`
/// - `BRANCHING = 1, TIP_BIAS = 0` → uniform random (current branching).
/// - `BRANCHING = 0` → strong preference for *smallest* outward gain among
///   convex faces — i.e. the slowest-advancing front face — which is the
///   isotropic-inflation criterion (cf. ZBrush DynaMesh, advancing-front
///   meshing, mean-curvature flow).
fn pick_weighted(cands: &mut Vec<GrowthCandidate>, topo: &mut Topology) -> GrowthCandidate {
    if cands.len() == 1 {
        return cands.swap_remove(0);
    }
    let effective_bias = TIP_BIAS - (1.0 - BRANCHING) * BRANCHING_SHARPNESS;
    let mesh_c = mesh_centroid(&topo.vertices);

    let gains: Vec<f32> = cands
        .iter()
        .map(|c| {
            let n = topo.faces[c.face_idx].normal;
            (c.apex - mesh_c).dot(n)
        })
        .collect();

    // Mean is taken over outward (g > 0) candidates only — keeps the
    // exponent magnitudes stable when most of the pool is convex.
    let pos_count = gains.iter().filter(|g| **g > 0.0).count();
    let mean_g = if pos_count == 0 {
        0.0
    } else {
        gains.iter().filter(|g| **g > 0.0).sum::<f32>() / pos_count as f32
    };

    let weights: Vec<f32> = gains
        .iter()
        .map(|&g| {
            let inward_factor = if g > 0.0 { 1.0 } else { BRANCHING };
            let g_eff = g.max(0.0); // guard exp() from blowing up on pockets
            inward_factor * (effective_bias * (g_eff - mean_g)).exp()
        })
        .collect();

    let total: f32 = weights.iter().sum();
    if total <= 0.0 {
        // All candidates were pockets and BRANCHING == 0: nothing to grow
        // outward this tick. Pick any to keep the manifold closed.
        let i = topo.rand_range(cands.len());
        return cands.swap_remove(i);
    }
    let mut t = topo.next_rand_unit() * total;
    for (i, w) in weights.iter().enumerate() {
        t -= *w;
        if t <= 0.0 {
            return cands.swap_remove(i);
        }
    }
    cands.swap_remove(cands.len() - 1) // float fallthrough
}

/// Find the apex P at unit distance from every vertex of `face`, on the outward
/// side defined by `face.normal`. Returns None if no such P exists (face vertices
/// are not co-spherical at unit distance, or only the inward apex exists).
fn compute_outward_apex(face: &Face, verts: &[Vec3]) -> Option<Vec3> {
    if face.vertices.len() < 3 {
        return None;
    }
    let a = verts[face.vertices[0] as usize];
    let b = verts[face.vertices[1] as usize];
    let c = verts[face.vertices[2] as usize];
    let centroid: Vec3 = face
        .vertices
        .iter()
        .map(|&i| verts[i as usize])
        .sum::<Vec3>()
        / face.vertices.len() as f32;

    for p in unit_sphere_intersection(a, b, c) {
        // Outward side test relative to face normal.
        if (p - centroid).dot(face.normal) <= EPS {
            continue;
        }
        // For quads: verify the remaining vertices are also at unit distance
        // from p (gates 4-connect on co-sphericity automatically).
        let all_unit = face.vertices[3..].iter().all(|&i| {
            ((verts[i as usize] - p).length() - TARGET_EDGE_LEN).abs() < EPS
        });
        if all_unit {
            return Some(p);
        }
    }
    None
}

fn is_position_admissible(verts: &[Vec3], p: Vec3) -> bool {
    for v in verts {
        let d = (*v - p).length();
        if d < EPS {
            return false; // duplicate of existing vertex
        }
        if d < ADMISSIBILITY_FLOOR - EPS {
            return false; // tighter than the chunkiness floor allows
        }
    }
    true
}

/// Returns 0, 1, or 2 points equidistant (=1) from the three given points.
pub(super) fn unit_sphere_intersection(a: Vec3, b: Vec3, c: Vec3) -> Vec<Vec3> {
    let ab = b - a;
    let ac = c - a;
    let n = ab.cross(ac);
    let n_sq = n.length_squared();
    if n_sq < 1.0e-10 {
        return Vec::new();
    }
    let to_center = (n.cross(ab) * ac.length_squared() + ac.cross(n) * ab.length_squared())
        / (2.0 * n_sq);
    let center = a + to_center;
    let r_sq = to_center.length_squared();
    if r_sq > TARGET_EDGE_LEN * TARGET_EDGE_LEN + EPS {
        return Vec::new();
    }
    let h_sq = (TARGET_EDGE_LEN * TARGET_EDGE_LEN - r_sq).max(0.0);
    let h = h_sq.sqrt();
    let normal = n / n_sq.sqrt();
    if h < EPS {
        vec![center]
    } else {
        vec![center + normal * h, center - normal * h]
    }
}

/// Replace a face with a fan of triangles connecting the new apex to its boundary.
/// This is the operation that keeps the surface a closed orientable manifold.
pub(super) fn cap_face(topo: &mut Topology, candidate: GrowthCandidate) {
    let new_idx = topo.vertices.len() as u32;
    topo.vertices.push(candidate.apex);
    let ocg_idx = topo.ocg.len() as u32;
    topo.ocg.push(OcgEntry {
        index: ocg_idx,
        position: candidate.apex - topo.relative_origin,
    });
    topo.ocg_id_for_vertex.push(ocg_idx);
    println!("OCG length: {}", topo.ocg.len());

    let face = topo.faces.swap_remove(candidate.face_idx);
    let k = face.vertices.len();
    for i in 0..k {
        let v_i = face.vertices[i];
        let v_next = face.vertices[(i + 1) % k];
        let p_i = topo.vertices[v_i as usize];
        let p_next = topo.vertices[v_next as usize];
        // CCW from outside: (P, v_i, v_next). Outward normal = (p_i-P)×(p_next-P).
        let normal = (p_i - candidate.apex)
            .cross(p_next - candidate.apex)
            .normalize_or_zero();
        topo.faces.push(Face {
            vertices: vec![new_idx, v_i, v_next],
            normal,
        });
    }
}
