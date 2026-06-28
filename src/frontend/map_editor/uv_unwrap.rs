// Map editor — UV atlas unwrap + paint-texture/material creation.
//
// On first MapEditor entry the terrain submeshes are unwrapped into ONE shared
// atlas (resolved decision §4) via xatlas, so every surface — including stacked
// cave/overhang surfaces at the same XZ — gets a unique, non-overlapping texel
// region. Each Bevy terrain mesh is CLONED and rebuilt with the atlas `UV_0`
// (never mutating the shared glb asset); a single paint `Image` (sized from the
// map extent) is created and sampled by a white-`base_color` `StandardMaterial`.
//
// Painting is VISUAL-ONLY and RUNTIME-ONLY: nothing here touches `WorldMesh`,
// the heightmap, the simulation, or any save format.

use bevy::asset::RenderAssetUsages;
use bevy::image::{Image, ImageSampler};
use bevy::mesh::{Indices, Mesh, VertexAttributeValues};
use bevy::prelude::*;
use bevy::render::render_resource::{TextureFormat, TextureUsages};

use xatlas_rs_v2::{ChartOptions, IndexData, MeshData, MeshDecl, PackOptions, Xatlas};

use crate::simulation_settings::{
    MAP_PAINT_BASE_SRGB, MAP_PAINT_TEXELS_PER_WORLD_UNIT, MAP_PAINT_TEX_MAX, MAP_PAINT_TEX_MIN,
};
use crate::world_geometry::MapSize;


// ── Pure texture-size helper ──────────────────────────────────────────────────

/// The (square) paint-texture edge for a map of XZ extent `(map_x, map_z)`.
/// `max(map_x, map_z) × MAP_PAINT_TEXELS_PER_WORLD_UNIT`, clamped to
/// `[MAP_PAINT_TEX_MIN, MAP_PAINT_TEX_MAX]`. Non-POT is fine under wgpu; we keep
/// exact proportionality and only clamp the extremes to bound VRAM.
pub fn paint_texture_edge(map_x: f32, map_z: f32) -> u32 {
    let raw = (map_x.max(map_z) * MAP_PAINT_TEXELS_PER_WORLD_UNIT).round() as i64;
    raw.clamp(MAP_PAINT_TEX_MIN as i64, MAP_PAINT_TEX_MAX as i64) as u32
}


// ── Unwrap result ─────────────────────────────────────────────────────────────

/// One unwrapped terrain submesh: the entity to swap, and its rebuilt (atlas-UV)
/// mesh handle.
pub struct RemappedMesh {
    pub entity:     Entity,
    pub new_handle: Handle<Mesh>,
}

pub struct UnwrapResult {
    pub remapped:   Vec<RemappedMesh>,
    pub atlas_edge: u32,
}


// ── Per-submesh input, owned so xatlas can borrow it across add_mesh+generate ──

struct SubmeshInput {
    entity:   Entity,
    original: Handle<Mesh>,
    pos_flat: Vec<f32>,        // xN*3
    nrm_flat: Vec<f32>,        // xN*3
    indices:  Vec<u32>,
    pos_xyz:  Vec<[f32; 3]>,   // for xref rebuild
    nrm_xyz:  Vec<[f32; 3]>,
}


// ── Public unwrap entry point ─────────────────────────────────────────────────

/// Unwrap every terrain submesh into one shared atlas and rebuild each as an
/// atlas-UV clone. Returns `None` (leaving the caller unprepared, to retry) if no
/// usable submesh data is available or xatlas fails — so we never half-swap.
pub fn unwrap_terrain(
    submeshes: &[(Entity, Handle<Mesh>)],
    meshes:    &mut Assets<Mesh>,
    map:       MapSize,
) -> Option<UnwrapResult> {
    // 1. Gather owned per-submesh input arrays (positions/normals/indices).
    let mut inputs: Vec<SubmeshInput> = Vec::new();
    for (entity, handle) in submeshes {
        let Some(mesh) = meshes.get(handle) else { continue };

        let Some(VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else { continue };
        let pos_xyz: Vec<[f32; 3]> = positions.clone();
        let n = pos_xyz.len();
        if n < 3 { continue; }

        // Normals: reuse if present, else synthesize flat per-vertex (xatlas only
        // uses them as a chart hint, so an approximation is fine).
        let nrm_xyz: Vec<[f32; 3]> = match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
            Some(VertexAttributeValues::Float32x3(nm)) if nm.len() == n => nm.clone(),
            _ => vec![[0.0, 1.0, 0.0]; n],
        };

        // Indices: U32/U16 → U32; non-indexed → 0..n.
        let indices: Vec<u32> = match mesh.indices() {
            Some(Indices::U32(v)) => v.clone(),
            Some(Indices::U16(v)) => v.iter().map(|&i| i as u32).collect(),
            None                  => (0..n as u32).collect(),
        };
        if indices.len() < 3 || indices.len() % 3 != 0 { continue; }

        let pos_flat: Vec<f32> = pos_xyz.iter().flat_map(|p| p.iter().copied()).collect();
        let nrm_flat: Vec<f32> = nrm_xyz.iter().flat_map(|p| p.iter().copied()).collect();

        inputs.push(SubmeshInput {
            entity:   *entity,
            original: handle.clone(),
            pos_flat,
            nrm_flat,
            indices,
            pos_xyz,
            nrm_xyz,
        });
    }
    if inputs.is_empty() { return None; }

    let atlas_edge = paint_texture_edge(map.x, map.z);

    // 2. Build all MeshDecls FIRST (borrowing `inputs`), then run xatlas. Decls
    //    and `inputs` must both outlive the `Xatlas` (it borrows them with 'x).
    let decls: Vec<MeshDecl> = inputs
        .iter()
        .map(|s| MeshDecl {
            vertex_position_data: MeshData::Contiguous(&s.pos_flat),
            vertex_normal_data:   Some(MeshData::Contiguous(&s.nrm_flat)),
            index_data:           Some(IndexData::U32(&s.indices)),
            face_count:           (s.indices.len() / 3) as u32,
            ..Default::default()
        })
        .collect();

    let mut atlas = Xatlas::new();
    for decl in &decls {
        if atlas.add_mesh(decl).is_err() {
            warn!("map editor: xatlas add_mesh failed; leaving terrain unprepared to retry");
            return None;
        }
    }

    let pack = PackOptions {
        resolution: atlas_edge,
        padding:    2,
        ..Default::default()
    };
    atlas.generate(&ChartOptions::default(), &pack);

    if atlas.atlas_count() == 0 {
        warn!("map editor: xatlas produced no atlas pages; leaving terrain unprepared");
        return None;
    }
    if atlas.atlas_count() != 1 {
        warn!(
            "map editor: xatlas produced {} atlas pages; the single shared paint \
             texture covers page 0 only (texture undersized for chart area).",
            atlas.atlas_count()
        );
    }

    let w = atlas.width() as f32;
    let h = atlas.height() as f32;
    if w <= 0.0 || h <= 0.0 { return None; }

    // 3. Rebuild each Bevy mesh from xatlas's seam-duplicated output via xref.
    let out_meshes = atlas.meshes();
    if out_meshes.len() != inputs.len() {
        warn!(
            "map editor: xatlas returned {} meshes for {} inputs; aborting unwrap",
            out_meshes.len(),
            inputs.len()
        );
        return None;
    }

    let mut remapped = Vec::with_capacity(inputs.len());
    for (k, m) in out_meshes.iter().enumerate() {
        let src = &inputs[k];

        let new_pos: Vec<[f32; 3]> =
            m.vertex_array.iter().map(|v| src.pos_xyz[v.xref as usize]).collect();
        let new_nrm: Vec<[f32; 3]> =
            m.vertex_array.iter().map(|v| src.nrm_xyz[v.xref as usize]).collect();
        // xatlas UVs are in texel units of the page; normalise to [0, 1].
        let new_uv: Vec<[f32; 2]> = m
            .vertex_array
            .iter()
            .map(|v| [v.uv[0] / w, v.uv[1] / h])
            .collect();
        let new_idx: Vec<u32> = m.index_array.to_vec();

        // Clone the source mesh so shared usages stay intact; overwrite geometry.
        let mut clone = meshes
            .get(&src.original)
            .map(|orig| orig.clone())
            .unwrap_or_else(|| {
                Mesh::new(
                    bevy::mesh::PrimitiveTopology::TriangleList,
                    RenderAssetUsages::default(),
                )
            });
        clone.insert_attribute(Mesh::ATTRIBUTE_POSITION, new_pos);
        clone.insert_attribute(Mesh::ATTRIBUTE_NORMAL, new_nrm);
        clone.insert_attribute(Mesh::ATTRIBUTE_UV_0, new_uv);
        clone.insert_indices(Indices::U32(new_idx));
        // The v1 per-vertex colour attribute is dead — drop it.
        clone.remove_attribute(Mesh::ATTRIBUTE_COLOR);

        let new_handle = meshes.add(clone);
        remapped.push(RemappedMesh { entity: src.entity, new_handle });
    }

    Some(UnwrapResult { remapped, atlas_edge })
}


// ── Paint image creation ──────────────────────────────────────────────────────

/// Create the shared paint texture: an `edge × edge` `Rgba8UnormSrgb` render
/// target, cleared to `MAP_PAINT_BASE_SRGB`. The Srgb format matches a
/// white-`base_color` `StandardMaterial.base_color_texture` (hardware-decoded to
/// linear on sample), so colours are NOT double-gamma'd. The Image lives in BOTH
/// the main and render worlds (`RenderAssetUsages::default()`), so CPU byte
/// writes re-upload automatically.
///
/// Sampling is NEAREST (not the default bilinear): painted texels are hard-edged
/// squares so a stroke stays crisp instead of bleeding/blurring across the
/// triangle ("washed" look). Combined with the higher atlas resolution
/// (`MAP_PAINT_TEXELS_PER_WORLD_UNIT`), this lets the brush paint sharp detail
/// INSIDE a triangle rather than flooding the whole face.
pub fn create_paint_image(edge: u32, images: &mut Assets<Image>) -> Handle<Image> {
    let mut img = Image::new_target_texture(edge, edge, TextureFormat::Rgba8UnormSrgb, None);
    // `new_target_texture` sets TEXTURE_BINDING | COPY_DST | RENDER_ATTACHMENT and
    // MAIN+RENDER usages. We only need to (a) keep COPY_DST for CPU re-uploads and
    // (b) fill the neutral base colour.
    img.texture_descriptor.usage |= TextureUsages::COPY_DST;
    img.sampler = ImageSampler::nearest();
    if let Some(data) = img.data.as_mut() {
        for px in data.chunks_exact_mut(4) {
            px.copy_from_slice(&MAP_PAINT_BASE_SRGB);
        }
    }
    images.add(img)
}


// ── Tests (pure helpers) ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_paint_texture_edge_reference_point() {
        // 250-unit map → 2048px (the documented reference).
        assert_eq!(paint_texture_edge(250.0, 250.0), 2048);
    }

    #[test]
    fn map_paint_texture_edge_scales_proportionally() {
        // Double the map → double the texture (still under the clamp).
        assert_eq!(paint_texture_edge(500.0, 500.0), 4096);
        // Uses the LARGER extent.
        assert_eq!(paint_texture_edge(500.0, 100.0), 4096);
    }

    #[test]
    fn map_paint_texture_edge_clamps_extremes() {
        assert_eq!(paint_texture_edge(1.0, 1.0), MAP_PAINT_TEX_MIN);
        assert_eq!(paint_texture_edge(100_000.0, 100_000.0), MAP_PAINT_TEX_MAX);
    }

    /// True if every UV lies within [0,1] (± epsilon).
    fn all_uvs_in_unit_range(uvs: &[[f32; 2]]) -> bool {
        const E: f32 = 1e-4;
        uvs.iter().all(|uv| {
            uv[0] >= -E && uv[0] <= 1.0 + E && uv[1] >= -E && uv[1] <= 1.0 + E
        })
    }

    /// Signed area ×2 of triangle (a, b, c) — sign gives winding.
    fn cross2(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    }

    /// True if point `p` is strictly inside triangle (a, b, c) (any winding).
    fn point_in_tri(p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> bool {
        let d1 = cross2(p, a, b);
        let d2 = cross2(p, b, c);
        let d3 = cross2(p, c, a);
        let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
        let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
        !(has_neg && has_pos)
    }

    /// True iff two UV triangles share interior area. Tested by checking each
    /// triangle's centroid against the other and edge-segment crossings — robust
    /// to the corner-to-corner AABB overlap that disjoint charts produce. This is
    /// the real cave/overhang-correctness check (acceptance #9): stacked surfaces
    /// must occupy NON-overlapping UV regions, even if their bounding boxes touch.
    fn tris_overlap(ta: [[f32; 2]; 3], tb: [[f32; 2]; 3]) -> bool {
        // Any vertex of one strictly inside the other ⇒ overlap.
        for &v in &ta {
            if point_in_tri(v, tb[0], tb[1], tb[2]) { return true; }
        }
        for &v in &tb {
            if point_in_tri(v, ta[0], ta[1], ta[2]) { return true; }
        }
        // Edge-segment crossings (catches the X-overlap with no contained vertex).
        let ea = [(ta[0], ta[1]), (ta[1], ta[2]), (ta[2], ta[0])];
        let eb = [(tb[0], tb[1]), (tb[1], tb[2]), (tb[2], tb[0])];
        for &(p1, p2) in &ea {
            for &(q1, q2) in &eb {
                if segs_cross(p1, p2, q1, q2) { return true; }
            }
        }
        false
    }

    fn segs_cross(p1: [f32; 2], p2: [f32; 2], q1: [f32; 2], q2: [f32; 2]) -> bool {
        let d1 = cross2(q1, q2, p1);
        let d2 = cross2(q1, q2, p2);
        let d3 = cross2(p1, p2, q1);
        let d4 = cross2(p1, p2, q2);
        ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
            && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
    }

    /// xatlas on a synthetic "cave" (two stacked tris at the same XZ, opposite Y)
    /// must give all UVs ∈ [0,1] and place the two triangles in NON-overlapping
    /// UV regions — proving stacked surfaces paint independently (acceptance #9).
    #[test]
    fn map_unwrap_cave_produces_disjoint_non_overlapping_uvs() {
        // Two horizontal triangles at the same XZ footprint, Y=10 (ceiling) and
        // Y=0 (floor).
        let pos: Vec<f32> = vec![
            // ceiling
            0.0, 10.0, 0.0,  10.0, 10.0, 0.0,  0.0, 10.0, 10.0,
            // floor
            0.0, 0.0, 0.0,   10.0, 0.0, 0.0,   0.0, 0.0, 10.0,
        ];
        let nrm: Vec<f32> = vec![
            0.0, 1.0, 0.0,  0.0, 1.0, 0.0,  0.0, 1.0, 0.0,
            0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0,
        ];
        let idx: Vec<u32> = vec![0, 1, 2, 3, 4, 5];

        let decl = MeshDecl {
            vertex_position_data: MeshData::Contiguous(&pos),
            vertex_normal_data:   Some(MeshData::Contiguous(&nrm)),
            index_data:           Some(IndexData::U32(&idx)),
            face_count:           2,
            ..Default::default()
        };

        let mut atlas = Xatlas::new();
        atlas.add_mesh(&decl).expect("add_mesh");
        let pack = PackOptions { resolution: 256, padding: 2, ..Default::default() };
        atlas.generate(&ChartOptions::default(), &pack);

        assert_eq!(atlas.atlas_count(), 1, "expected a single atlas page");
        let w = atlas.width() as f32;
        let h = atlas.height() as f32;
        let out = atlas.meshes();
        let m = &out[0];

        // Normalise UVs to [0,1].
        let uvs: Vec<[f32; 2]> = m.vertex_array.iter().map(|v| [v.uv[0] / w, v.uv[1] / h]).collect();
        assert!(all_uvs_in_unit_range(&uvs), "UVs out of [0,1]: {uvs:?}");

        // The output index array references the rebuilt (seam-duplicated) verts.
        assert_eq!(m.index_array.len(), 6, "two triangles expected");
        let t0 = [m.index_array[0] as usize, m.index_array[1] as usize, m.index_array[2] as usize];
        let t1 = [m.index_array[3] as usize, m.index_array[4] as usize, m.index_array[5] as usize];
        let ta = [uvs[t0[0]], uvs[t0[1]], uvs[t0[2]]];
        let tb = [uvs[t1[0]], uvs[t1[1]], uvs[t1[2]]];
        assert!(
            !tris_overlap(ta, tb),
            "stacked-surface triangles share overlapping UV area: {ta:?} vs {tb:?}"
        );
    }
}
