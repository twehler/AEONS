//! Star-shaped reconstruction via spherical projection.
//!
//! Used as the watertight fallback when the BPA reconstruction can't seal the
//! cloud. Always produces a closed manifold of genus 0; capable of preserving
//! concave features when the cloud is star-shaped from its centroid.

use bevy::prelude::*;

use super::{EPS, Face, Topology, mesh_centroid};

/// Rebuild `topo.faces` as a star-shaped triangulation of the vertex cloud.
///
/// Pipeline:
/// 1. Project every vertex onto the unit sphere centered at the cloud's
///    centroid `C`. The projected directions live on a sphere, so their 3D
///    convex hull is a triangulation of that sphere. Connectivity is therefore
///    decided by *angular* proximity, not by who is furthest out — which is
///    exactly what lets concave vertices stay in the mesh as inward features.
/// 2. Run the incremental 3D convex-hull algorithm on the projected points
///    (seed tetrahedron from 4 extremal directions, then add the remaining
///    points one by one, removing visible faces and stitching new ones from
///    the horizon).
/// 3. Reuse the hull's face indices but recompute normals from the *original*
///    positions, flipping the vertex order whenever the lifted normal points
///    inward (toward `C`).
///
/// The result is a closed orientable manifold of genus 0 in which every
/// vertex appears (so no compaction is needed) and where bumps and dents
/// both survive. The only assumption is that the cloud is star-shaped from
/// `C` — every vertex visible from the centroid along a ray, which holds for
/// our roughly spherical growth.
pub(super) fn rebuild_as_convex_hull(topo: &mut Topology) {
    use std::collections::HashMap;

    let n = topo.vertices.len();
    if n < 4 {
        return;
    }
    let centroid = mesh_centroid(&topo.vertices);
    // Project to unit sphere around the centroid. The fallback for a vertex
    // coincident with `C` is arbitrary (Vec3::Y) — this is a degenerate edge
    // case that shouldn't occur during normal growth.
    let verts: Vec<Vec3> = topo
        .vertices
        .iter()
        .map(|v| {
            let d = *v - centroid;
            let len = d.length();
            if len < EPS { Vec3::Y } else { d / len }
        })
        .collect();

    // 1. Pick 4 non-coplanar seed vertices.
    let i0 = (0..n)
        .min_by(|&a, &b| verts[a].x.partial_cmp(&verts[b].x).unwrap())
        .unwrap();
    let i1 = (0..n)
        .max_by(|&a, &b| verts[a].x.partial_cmp(&verts[b].x).unwrap())
        .unwrap();
    if i0 == i1 {
        return;
    }
    let line_dir = (verts[i1] - verts[i0]).normalize_or_zero();
    let i2 = (0..n)
        .filter(|&i| i != i0 && i != i1)
        .max_by(|&a, &b| {
            let da = (verts[a] - verts[i0]).cross(line_dir).length();
            let db = (verts[b] - verts[i0]).cross(line_dir).length();
            da.partial_cmp(&db).unwrap()
        })
        .unwrap();
    let plane_n = (verts[i1] - verts[i0])
        .cross(verts[i2] - verts[i0])
        .normalize_or_zero();
    let i3 = (0..n)
        .filter(|&i| i != i0 && i != i1 && i != i2)
        .max_by(|&a, &b| {
            let da = (verts[a] - verts[i0]).dot(plane_n).abs();
            let db = (verts[b] - verts[i0]).dot(plane_n).abs();
            da.partial_cmp(&db).unwrap()
        })
        .unwrap();
    let seeds = [i0, i1, i2, i3];

    // 2. Build the seed tetrahedron with outward-facing normals.
    let tet_center = (verts[i0] + verts[i1] + verts[i2] + verts[i3]) / 4.0;
    let mut faces: Vec<Face> = Vec::new();
    let tris: [(usize, usize, usize); 4] = [
        (i0, i1, i2),
        (i0, i2, i3),
        (i0, i3, i1),
        (i1, i3, i2),
    ];
    for &(a, b, c) in &tris {
        let va = a as u32;
        let (mut vb, mut vc) = (b as u32, c as u32);
        let pa = verts[a];
        let mut nrm = (verts[b] - pa).cross(verts[c] - pa).normalize_or_zero();
        let face_centroid = (verts[a] + verts[b] + verts[c]) / 3.0;
        if (face_centroid - tet_center).dot(nrm) < 0.0 {
            std::mem::swap(&mut vb, &mut vc);
            nrm = -nrm;
        }
        faces.push(Face { vertices: vec![va, vb, vc], normal: nrm });
    }

    // 3. Incrementally add the remaining vertices.
    for i in 0..n {
        if seeds.contains(&i) {
            continue;
        }
        let p = verts[i];

        // Faces visible from p: those whose outward half-space contains p.
        let visible: Vec<usize> = faces
            .iter()
            .enumerate()
            .filter_map(|(fi, f)| {
                let v0 = verts[f.vertices[0] as usize];
                if (p - v0).dot(f.normal) > EPS {
                    Some(fi)
                } else {
                    None
                }
            })
            .collect();
        if visible.is_empty() {
            continue; // p is inside the current hull.
        }

        // Horizon: directed edges of visible faces whose mate is *not* visible.
        // Each horizon edge is collected with the orientation it had on its
        // visible face, so the new triangle (a, b, p) inherits the correct CCW.
        let visible_set: std::collections::HashSet<usize> = visible.iter().copied().collect();
        let mut edge_owner: HashMap<(u32, u32), usize> = HashMap::new();
        for &fi in &visible {
            let f = &faces[fi];
            for j in 0..3 {
                let a = f.vertices[j];
                let b = f.vertices[(j + 1) % 3];
                let key = if a < b { (a, b) } else { (b, a) };
                edge_owner.insert(key, fi);
            }
        }
        let mut horizon: Vec<(u32, u32)> = Vec::new();
        for &fi in &visible {
            let f = faces[fi].clone();
            for j in 0..3 {
                let a = f.vertices[j];
                let b = f.vertices[(j + 1) % 3];
                let key = if a < b { (a, b) } else { (b, a) };
                // Edge is on the horizon iff the *other* face sharing it is
                // not in the visible set. Walk all faces to find the mate.
                let mate = faces.iter().enumerate().find(|(other_fi, of)| {
                    if *other_fi == fi {
                        return false;
                    }
                    for k in 0..3 {
                        let oa = of.vertices[k];
                        let ob = of.vertices[(k + 1) % 3];
                        let okey = if oa < ob { (oa, ob) } else { (ob, oa) };
                        if okey == key {
                            return true;
                        }
                    }
                    false
                });
                let mate_visible = mate.map(|(mfi, _)| visible_set.contains(&mfi)).unwrap_or(false);
                if !mate_visible {
                    horizon.push((a, b));
                }
            }
        }
        let _ = edge_owner; // retained name for clarity; not used past collection.

        // Remove visible faces (descending indices to keep swap_remove safe).
        let mut to_remove = visible.clone();
        to_remove.sort_unstable_by(|a, b| b.cmp(a));
        for fi in to_remove {
            faces.swap_remove(fi);
        }

        // Stitch new faces from p to each horizon edge.
        let p_idx = i as u32;
        for &(a, b) in &horizon {
            let va = verts[a as usize];
            let vb = verts[b as usize];
            let nrm = (vb - va).cross(p - va).normalize_or_zero();
            faces.push(Face {
                vertices: vec![a, b, p_idx],
                normal: nrm,
            });
        }
    }

    // Re-emit each hull triangle with normals computed from the *original*
    // positions. Flip the vertex order whenever the lifted normal points
    // inward, so every face has a consistent outward orientation.
    for face in &mut faces {
        let pa = topo.vertices[face.vertices[0] as usize];
        let pb = topo.vertices[face.vertices[1] as usize];
        let pc = topo.vertices[face.vertices[2] as usize];
        let nrm = (pb - pa).cross(pc - pa).normalize_or_zero();
        let face_centroid = (pa + pb + pc) / 3.0;
        if (face_centroid - centroid).dot(nrm) < 0.0 {
            face.vertices.swap(1, 2);
            face.normal = -nrm;
        } else {
            face.normal = nrm;
        }
    }
    topo.faces = faces;
    // No compaction: every projected point lies on the unit sphere → on the
    // spherical hull → every original vertex is referenced by some face.
}
