// Map editor — one-time terrain preparation for texture painting.
//
// On first MapEditor entry we walk the terrain `SceneRoot`'s descendant `Mesh3d`
// entities and, in one pass:
//   * UV-unwrap ALL submeshes into one shared atlas (`uv_unwrap::unwrap_terrain`),
//     rebuilding each mesh with non-overlapping atlas `UV_0` (cave/overhang-safe);
//   * create the shared paint `Image` (sized from the map extent) and a white-
//     `base_color` `StandardMaterial` sampling it;
//   * swap each terrain entity to the rebuilt mesh + paint material.
// Subsequent brush dabs (see `gpu_paint.rs`) stamp into the paint texture.
//
// Visual-only + runtime-only: nothing here touches `WorldMesh`, the heightmap,
// the simulation, or any save format. The retry-until-ready guard is preserved.

use std::collections::HashMap;

use bevy::prelude::*;
use bevy::mesh::{Indices, VertexAttributeValues};

use crate::simulation_settings::WindowMode;

use super::gpu_paint::{self, PaintState};
use super::uv_unwrap;
use crate::world_geometry::MapSize;


// ── Marker + resources ────────────────────────────────────────────────────────

/// Tag on the terrain `SceneRoot` (added at world load). Lets the paint pipeline
/// find the terrain hierarchy + read its world transform.
#[derive(Component)]
pub struct TerrainSceneRoot;

/// Bucket edge (world units) of the per-submesh world-XZ triangle grid used by the
/// brush to cull triangles to those overlapping the brush footprint (Stage 2).
/// Mirrors `WorldMesh`'s XZ grid in spirit; sized so a typical brush footprint
/// overlaps only a handful of buckets.
pub const PAINT_TRI_GRID_BUCKET: f32 = 4.0;

/// One prepared terrain submesh's brush-acceleration cache (Stage 2): the decoded
/// triangle index list (built ONCE), the static world transform used to place the
/// triangles, and a world-XZ bucket grid mapping `(kx, kz)` → the triangle indices
/// (triangle = a triple at `3*tri` in `indices`) whose world-XZ AABB overlaps it.
/// Built lazily on the first dab, AFTER child `GlobalTransform`s are propagated.
#[derive(Default)]
pub struct SubmeshCull {
    /// Decoded indices (U32/U16/non-indexed → U32), length a multiple of 3.
    pub indices: Vec<u32>,
    /// The submesh entity's world transform at build time (terrain is static).
    pub model:   Mat4,
    /// World-XZ bucket → triangle indices (triangle `t` spans `indices[3t..3t+3]`).
    pub grid:    HashMap<(i32, i32), Vec<u32>>,
}

/// The terrain meshes prepared for painting. After unwrap these are the REMAPPED
/// (atlas-UV) mesh handles, paired with their owning entity. The brush stamp and
/// "Color All" read these. `cull` is the lazily-built (Stage 2) per-submesh
/// world-XZ acceleration cache, keyed by the same `(Handle<Mesh>, Entity)` order as
/// `meshes`; empty until the first dab builds it.
#[derive(Resource, Default)]
pub struct TerrainPaintTargets {
    pub meshes:   Vec<(Handle<Mesh>, Entity)>,
    pub prepared: bool,
    /// Per-submesh culling cache (same index/order as `meshes`); `Vec::is_empty()`
    /// ⇒ not yet built (build on first dab once transforms are propagated).
    pub cull:     Vec<SubmeshCull>,
}


// ── Shared display material ────────────────────────────────────────────────────

/// The terrain's painting display material: white `base_color` modulated by the
/// shared paint texture. Factored so `prepare_terrain_paint` and the `.aeonsw`
/// loader (`world_geometry::load_world_file`) build an identical material and
/// cannot drift.
pub fn make_display_material(paint_image: Handle<Image>) -> StandardMaterial {
    StandardMaterial {
        base_color:           Color::WHITE,
        base_color_texture:   Some(paint_image),
        unlit:                false,
        perceptual_roughness: 1.0,
        ..default()
    }
}


// ── One-time terrain preparation ──────────────────────────────────────────────

/// On first MapEditor entry, unwrap the terrain submeshes into one shared atlas,
/// create the paint texture + material, and swap each terrain mesh/material.
/// Retries each frame until the scene is instantiated.
#[allow(clippy::type_complexity)]
pub fn prepare_terrain_paint(
    mode:           Res<WindowMode>,
    map_size:       Res<MapSize>,
    mut commands:   Commands,
    mut targets:    ResMut<TerrainPaintTargets>,
    mut paint:      ResMut<PaintState>,
    mut materials:  ResMut<Assets<StandardMaterial>>,
    mut meshes:     ResMut<Assets<Mesh>>,
    mut images:     ResMut<Assets<Image>>,
    roots:          Query<Entity, With<TerrainSceneRoot>>,
    children_q:     Query<&Children>,
    mesh_q:         Query<&Mesh3d>,
) {
    if *mode != WindowMode::MapEditor { return; }
    if targets.prepared { return; }

    let Ok(root) = roots.single() else { return };

    // BFS the scene hierarchy for `Mesh3d` descendants, collecting (entity, mesh).
    let mut stack = vec![root];
    let mut submeshes: Vec<(Entity, Handle<Mesh>)> = Vec::new();
    while let Some(e) = stack.pop() {
        if let Ok(m) = mesh_q.get(e) { submeshes.push((e, m.0.clone())); }
        if let Ok(ch) = children_q.get(e) { stack.extend(ch.iter()); }
    }
    if submeshes.is_empty() { return; } // scene not instantiated yet — retry next frame.

    // 1. Unwrap into one shared atlas. `None` ⇒ retry next frame (no half-swap).
    let Some(result) = uv_unwrap::unwrap_terrain(&submeshes, &mut meshes, *map_size) else {
        return;
    };

    // 2. Shared paint texture + white-base_color display material.
    let paint_image = uv_unwrap::create_paint_image(result.atlas_edge, &mut images);
    let display_mat = materials.add(make_display_material(paint_image.clone()));

    // Seed the authoritative CPU mirror (Stage 1) from the freshly-created Image
    // bytes (the grey base fill). All brush/Color-All writes go to this mirror and
    // are mirrored into the main-world Image via `get_mut_untracked`.
    let mirror = images
        .get(&paint_image)
        .and_then(|img| img.data.clone())
        .unwrap_or_default();

    // 2b. Precompute the seam-fill (gutter dilation) map over the shared atlas, so
    //     the brush can bleed painted charts into their gutter texels and avoid the
    //     unpainted "No Man's Land" lines between UV charts. One-time (terrain is
    //     static); read the rebuilt atlas UVs + indices straight back from Assets.
    let mut sm_uv:  Vec<Vec<[f32; 2]>> = Vec::new();
    let mut sm_idx: Vec<Vec<u32>>      = Vec::new();
    for r in &result.remapped {
        let Some(mesh) = meshes.get(&r.new_handle) else { continue };
        let Some(VertexAttributeValues::Float32x2(uv)) = mesh.attribute(Mesh::ATTRIBUTE_UV_0)
        else { continue };
        let idx: Vec<u32> = match mesh.indices() {
            Some(Indices::U32(v)) => v.clone(),
            Some(Indices::U16(v)) => v.iter().map(|&i| i as u32).collect(),
            None                  => continue,
        };
        sm_uv.push(uv.clone());
        sm_idx.push(idx);
    }
    let submeshes: Vec<(&[[f32; 2]], &[u32])> =
        sm_uv.iter().zip(sm_idx.iter()).map(|(u, i)| (u.as_slice(), i.as_slice())).collect();
    let seam_fill =
        gpu_paint::compute_seam_fill(&submeshes, result.atlas_edge, gpu_paint::SEAM_FILL_MARGIN);

    // 3. Swap each terrain entity to the rebuilt mesh + paint material, and record
    //    the remapped (Handle<Mesh>, Entity) for the brush / Color All.
    targets.meshes.clear();
    targets.cull.clear(); // rebuilt lazily on the next dab (Stage 2)
    for r in &result.remapped {
        commands.entity(r.entity).try_insert((
            Mesh3d(r.new_handle.clone()),
            MeshMaterial3d(display_mat.clone()),
        ));
        targets.meshes.push((r.new_handle.clone(), r.entity));
    }

    paint.paint_image = Some(paint_image);
    paint.display_mat = Some(display_mat);
    paint.atlas_edge  = result.atlas_edge;
    paint.last_dab_vp = None;
    paint.mirror      = mirror;
    paint.seam_fill   = seam_fill;

    targets.prepared = true;
    info!(
        "Map editor: unwrapped {} terrain submeshes into a {}² paint atlas; \
         seam-fill map has {} source texels (gutter dilation).",
        result.remapped.len(),
        result.atlas_edge,
        paint.seam_fill.len(),
    );
}
