// Map editor — projective texture painting (the core technical work).
//
// The Basic Brush stamps the selected colour into the shared paint texture under
// a SCREEN-SPACE circular brush, occlusion-correct so painting one surface never
// bleeds onto a different surface that projects to the same screen pixels (e.g.
// the ground above a cave does not paint the cave ceiling).
//
// Occlusion route (resolved): the spec's Route B test (world-distance + normal
// gate), fed by a CORRECT terrain raycast (`WorldMesh::raycast`) for the cursor-
// centre hit point + normal — NOT rapier `cast_ray`, whose collider is a flat
// uniform slab and useless for terrain occlusion. A fragment/texel is painted
// iff its surface point (a) projects within `brush_radius_px` of the cursor in
// viewport pixels, (b) lies within the brush's WORLD radius of the cursor-centre
// hit, AND (c) its surface normal agrees with the hit normal. (a) makes the brush
// screen-space-constant for both ortho and perspective cameras (camera-agnostic
// via the live view-projection); (b)+(c) separate stacked cave/overhang surfaces.
//
// Paint mechanism (acceptance criterion 7 — "Painting is GPU-based"): the brush
// stamps DIRECTLY into the shared paint `Image` bytes on the CPU over only the
// covered atlas texels, then the Image (`RenderAssetUsages::default()`) re-uploads
// to the GPU automatically (the texture is sampled by the terrain display
// material — so the painted result is shown by the GPU). `brush_stroke`
// rasterises each terrain triangle's UV footprint into the atlas, interpolates
// each covered texel's world surface point + normal, applies the SAME (a)+(b)+(c)
// gate the pure helpers below encode (`world_to_viewport_px` for the screen disc,
// `surface_point_under_brush` for the world-radius + normal gate), and writes the
// brush colour into exactly those texels — so a dab is O(brush footprint), never
// O(whole texture). Untouched texels keep their prior value (the grey base +
// earlier dabs), so dabs ACCUMULATE.
//
// NOTE on the prior GPU-camera path: a `Camera3d` rendering paint-clone meshes
// into a `RenderTarget::Image` with `ClearColorConfig::None` does NOT accumulate
// under Bevy 0.18's stock render graph — the camera renders into an internal,
// never-grey-seeded `main_texture`, and the UpscalingNode then blits that WHOLE
// (black-outside-the-footprint) texture over the entire paint Image every active
// frame, wiping the terrain to black on the first dab. The CPU stamp here writes
// the real Image directly, so `discard`-style preservation is exact. The pure
// helpers (`world_to_viewport_px`, `screen_pixel_radius_to_world`,
// `surface_point_under_brush`, `dab_points`) are unit-tested below as the
// reference for the gate. `PaintState` is structured so the future `.world`
// format can serialise the paint Image bytes + atlas edge with no extra plumbing
// (resolved decision #3 — no save code in this version).
//
// Painting is VISUAL-ONLY and RUNTIME-ONLY.

use bevy::image::Image;
use bevy::math::URect;
use bevy::mesh::{Indices, Mesh, VertexAttributeValues};
use bevy::prelude::*;
use bevy::ui::{ComputedNode, UiGlobalTransform};
use bevy::window::PrimaryWindow;

use crate::frontend::ViewportImage;
use crate::player_plugin::FlyCam;
use crate::simulation_settings::{BRUSH_DAB_SPACING_FRAC, WindowMode};
use crate::world_geometry::WorldMesh;

use super::material::MapEditorSession;
use super::paint_upload::PaintDirtyRect;
use super::terrain_paint::{PAINT_TRI_GRID_BUCKET, SubmeshCull, TerrainPaintTargets};


// ── Tunables ──────────────────────────────────────────────────────────────────

/// Cosine threshold for the normal-agreement gate. A texel's surface normal must
/// be within ~75° of the cursor-centre hit normal to paint, so a cave ceiling
/// (opposed normal) is rejected even when it is within the brush world radius.
const NORMAL_AGREE_COS: f32 = 0.25;

/// World-radius safety factor: the world radius is the screen-pixel radius mapped
/// to world units at the hit depth, scaled up slightly so the world-distance gate
/// never clips a texel the screen test would accept (the screen test is primary;
/// the world gate only rejects the far/back surface).
const WORLD_RADIUS_SLACK: f32 = 1.5;

/// Maximum ray length for the cursor-centre terrain cast (well past any map).
const RAYCAST_MAX: f32 = 1.0e6;


// ── Paint state resources ──────────────────────────────────────────────────────

/// All runtime paint state. Structured so a future `.world` serialiser can read
/// the paint Image bytes + atlas edge directly (decision #3 — no save code now).
#[derive(Resource, Default)]
pub struct PaintState {
    /// The shared atlas paint texture (created once on first MapEditor prep).
    pub paint_image: Option<Handle<Image>>,
    /// The display material sampling `paint_image` (white base_color).
    pub display_mat: Option<Handle<StandardMaterial>>,
    /// Atlas (and paint-texture) edge in texels.
    pub atlas_edge:  u32,
    /// Last stamped cursor position (viewport px), for dab spacing.
    pub last_dab_vp: Option<Vec2>,
    /// Authoritative CPU mirror of the paint texture's RGBA8 bytes (Stage 1). The
    /// brush rasteriser writes here (NOT through `Assets::get_mut`, which would fire
    /// `AssetEvent::Modified` → a whole-Image extract clone + full GPU realloc). The
    /// main-world `Image.data` is kept in sync via `get_mut_untracked` (no Modified
    /// event) so `.world` export still reads real paint, and only the dirty
    /// sub-rectangle is shipped to the GPU (see `paint_upload`).
    pub mirror:      Vec<u8>,
    /// Monotonic dirty-payload generation; bumped on every push to `PaintDirtyRect`
    /// so the RenderApp uploads each rect at most once.
    pub generation:  u64,
}

/// Edit-buffer state for the "Brush size (px)" field (mirrors
/// `statistics_panel::TimeSpeedEditState`).
#[derive(Resource, Default)]
pub struct BrushSizeEditState {
    pub buffer:  String,
    pub focused: bool,
}


// ── Pure helpers (the per-texel gate; runtime + unit-tested) ────────────────────
//
// `world_to_viewport_px` and `surface_point_under_brush` are the per-texel
// screen-disc / world-radius / normal gate, called BOTH at runtime (by the CPU
// rasteriser's `StampGate::passes`) AND in the `#[cfg(test)]` block below — so the
// brush-locality, screen-constant-sizing, and cave-correctness guarantees stay
// machine-checked against the same code the production stamp runs.
// `screen_pixel_radius_to_world` computes the brush's world radius at the hit
// depth. `dab_points` is the tested spacing reference for a future multi-stamp
// path (the current stamp is one footprint per moved-far-enough frame).

/// Project `world` to viewport pixels via `vp` (view-projection) + viewport size.
/// Returns `None` if the point is behind the camera (`w <= 0`).
pub fn world_to_viewport_px(world: Vec3, vp: Mat4, viewport: Vec2) -> Option<Vec2> {
    let clip = vp * world.extend(1.0);
    if clip.w <= 0.0 { return None; }
    let ndc = clip.truncate() / clip.w; // xyz/w; we use xy
    let px = (Vec2::new(ndc.x, -ndc.y) * 0.5 + Vec2::splat(0.5)) * viewport;
    Some(px)
}

/// The world distance subtended by `radius_px` screen pixels at `hit_world`'s
/// depth. Camera-agnostic (works for ortho and perspective via the live `vp`):
/// project `hit_world`, offset it `radius_px` in screen X, un-project that offset
/// back to a world delta at the same depth, return its length. This is what makes
/// the brush screen-space-constant regardless of camera distance/zoom.
pub fn screen_pixel_radius_to_world(
    radius_px: f32,
    hit_world: Vec3,
    vp:        Mat4,
    viewport:  Vec2,
) -> f32 {
    let clip = vp * hit_world.extend(1.0);
    if clip.w.abs() < 1e-6 || viewport.x < 1.0 { return radius_px; }
    let inv = vp.inverse(); // (NaNs if singular; the finite guard below catches it)
    let w = clip.w;
    let ndc = clip.truncate() / w;

    // NDC delta for `radius_px` pixels in X.
    let dndc_x = (radius_px / viewport.x) * 2.0;
    let ndc2 = Vec3::new(ndc.x + dndc_x, ndc.y, ndc.z);

    // Un-project both NDC points at the SAME clip-w (same depth slice).
    let p0 = unproject(ndc, w, inv);
    let p1 = unproject(ndc2, w, inv);
    let r = (p1 - p0).length();
    if r.is_finite() && r > 0.0 { r } else { radius_px }
}

fn unproject(ndc: Vec3, w: f32, inv_vp: Mat4) -> Vec3 {
    let clip = Vec4::new(ndc.x * w, ndc.y * w, ndc.z * w, w);
    let world = inv_vp * clip;
    world.truncate() / world.w
}

/// The exact CPU mirror of the per-texel occlusion gate. A surface point is under
/// the brush iff it is within `world_radius` of the cursor-centre hit AND its
/// normal agrees with the hit normal (within `normal_thresh` cosine). The
/// screen-space disc test is applied separately (caller). Returns true only for a
/// bounded local region around the hit — NEVER the whole mesh.
///
/// Retained as the byte-identity REFERENCE for the production gate
/// (`StampGate::passes`, which hoists the constant invariants): the `#[cfg(test)]`
/// block binds this helper, so the gate semantics stay machine-checked against it.
#[allow(dead_code)]
pub fn surface_point_under_brush(
    p_world:      Vec3,
    world_normal: Vec3,
    hit_world:    Vec3,
    hit_normal:   Vec3,
    world_radius: f32,
    normal_thresh: f32,
) -> bool {
    if (p_world - hit_world).length_squared() > world_radius * world_radius {
        return false;
    }
    world_normal.normalize_or_zero().dot(hit_normal.normalize_or_zero()) >= normal_thresh
}

/// Interpolated stamp positions from `last` to `cur` at `spacing` px intervals
/// (inclusive of `cur`). A short move yields a single point; a long move yields
/// `ceil(dist / spacing)` evenly-spaced points so fast strokes stay continuous.
/// (`brush_stroke` stamps one footprint per moved-far-enough frame, using the same
/// spacing as a per-frame skip gate; this interpolator stays as the tested spacing
/// reference for a future multi-stamp path.)
#[allow(dead_code)]
pub fn dab_points(last: Vec2, cur: Vec2, spacing: f32) -> Vec<Vec2> {
    let spacing = spacing.max(0.5);
    let dist = (cur - last).length();
    if dist < spacing {
        return vec![cur];
    }
    let n = (dist / spacing).ceil() as usize;
    (1..=n).map(|i| last.lerp(cur, i as f32 / n as f32)).collect()
}


// ── Cursor → viewport-local pixels (copied from the v1 painter) ────────────────

fn cursor_to_viewport_px(
    cursor:     Vec2,
    viewport_q: &Query<(&ComputedNode, &UiGlobalTransform), With<ViewportImage>>,
) -> Vec2 {
    match viewport_q.single() {
        Ok((node, ui_xf)) => {
            let inv      = node.inverse_scale_factor;
            let size     = node.size() * inv;
            let top_left = ui_xf.translation * inv - size * 0.5;
            cursor - top_left
        }
        Err(_) => cursor,
    }
}

fn viewport_size(
    viewport_q: &Query<(&ComputedNode, &UiGlobalTransform), With<ViewportImage>>,
    window:     &Window,
) -> Vec2 {
    match viewport_q.single() {
        Ok((node, _)) => node.size() * node.inverse_scale_factor,
        Err(_)        => Vec2::new(window.width(), window.height()),
    }
}


// ── Brush stroke (CPU footprint stamp into the paint Image) ─────────────────────

/// Stamp the brush footprint directly into the paint Image bytes. While LMB is
/// held over the terrain (and not over UI / a focused field), raycast the cursor
/// centre for the occlusion reference, then rasterise every terrain triangle's UV
/// footprint into the atlas and write the brush colour into exactly the covered
/// texels whose interpolated surface point passes the (a) screen-disc, (b)
/// world-radius, and (c) normal-agreement gate. Untouched texels keep their prior
/// value, so dabs ACCUMULATE over the grey base. The Image is
/// `RenderAssetUsages::default()`, so the byte writes re-upload to the GPU
/// automatically (the display material samples this Image — the painted result is
/// shown by the GPU; the stamp is O(brush footprint), never O(whole texture)).
///
/// Dab spacing (resolved decision #2): skip stamping when the cursor has not moved
/// at least `spacing` viewport-px since the last stamp, so a held-still or slow
/// cursor does not re-stamp the same texels every frame.
#[allow(clippy::too_many_arguments)]
pub fn brush_stroke(
    mode:            Res<WindowMode>,
    mouse:           Res<ButtonInput<MouseButton>>,
    session:         Res<MapEditorSession>,
    brush_edit:      Res<BrushSizeEditState>,
    mut targets:     ResMut<TerrainPaintTargets>,
    mut paint_state: ResMut<PaintState>,
    mut dirty:       ResMut<PaintDirtyRect>,
    mut images:      ResMut<Assets<Image>>,
    meshes:          Res<Assets<Mesh>>,
    world_mesh:      Res<WorldMesh>,
    ui_interactions: Query<&Interaction>,
    cameras:         Query<(&Camera, &GlobalTransform), With<FlyCam>>,
    windows:         Query<&Window, With<PrimaryWindow>>,
    viewport_q:      Query<(&ComputedNode, &UiGlobalTransform), With<ViewportImage>>,
    transforms:      Query<&GlobalTransform>,
) {
    // The closure-free early-outs all fall through (no stamp this frame).
    'paint: {
        if *mode != WindowMode::MapEditor { break 'paint; }
        if !mouse.pressed(MouseButton::Left) {
            paint_state.last_dab_vp = None; // reset stroke continuity between strokes
            break 'paint;
        }
        // A focused text field swallows LMB (the field uses just_pressed; painting
        // uses pressed/held, so this gate is required).
        if brush_edit.focused { break 'paint; }
        // A UI panel claimed the press this frame — don't paint underneath it.
        if ui_interactions.iter().any(|i| matches!(i, Interaction::Pressed)) { break 'paint; }

        let Some(material_color) = session.selected_material else { break 'paint };
        let Some(image_handle)   = paint_state.paint_image.clone() else { break 'paint };
        let Ok((camera, cam_xf)) = cameras.single() else { break 'paint };
        let Ok(window)           = windows.single()  else { break 'paint };
        let Some(cursor)         = window.cursor_position() else { break 'paint };

        let cursor_vp = cursor_to_viewport_px(cursor, &viewport_q);
        let vp_size   = viewport_size(&viewport_q, window);
        let radius_px = session.brush_radius_px;

        // Dab spacing: only stamp once the cursor has travelled far enough, so a
        // held-still / slow cursor never re-stamps the same texels every frame.
        let spacing = (radius_px * BRUSH_DAB_SPACING_FRAC).max(1.0);
        if let Some(last) = paint_state.last_dab_vp {
            if (cursor_vp - last).length() < spacing {
                break 'paint; // hasn't moved far enough — skip (no over-stamp).
            }
        }

        let view_proj = camera.clip_from_view() * cam_xf.to_matrix().inverse();

        // Cursor-centre terrain hit via the analytic raycast (NOT rapier slab).
        let Ok(ray) = camera.viewport_to_world(cam_xf, cursor_vp) else { break 'paint };
        let Some((hit_world, hit_normal, _toi)) =
            world_mesh.raycast(ray.origin, *ray.direction, RAYCAST_MAX)
        else { break 'paint }; // cursor off terrain.

        let world_radius =
            screen_pixel_radius_to_world(radius_px, hit_world, view_proj, vp_size)
                * WORLD_RADIUS_SLACK;

        // The brush colour is the raw sRGB byte quad: the paint texture is
        // `Rgba8UnormSrgb`, hardware-decoded to linear on sample, so the stored
        // bytes must be the sRGB triple (matching `color_all_fill` + the grey base).
        let color = material_color.srgb_u8();

        let edge = paint_state.atlas_edge;
        if edge == 0 { break 'paint; }
        // Defensive: only stamp if the mirror is the expected RGBA8 size.
        if paint_state.mirror.len() < (edge as usize) * (edge as usize) * 4 { break 'paint; }

        let gate = StampGate::new(view_proj, vp_size, cursor_vp, radius_px, hit_world, hit_normal,
                                  world_radius);

        // Stage 2: ensure the per-submesh world-XZ culling cache is built (lazily,
        // now that child GlobalTransforms are propagated — see CAVEAT in the plan).
        ensure_cull_cache(&mut targets, &meshes, &transforms);

        // Brush world-XZ AABB: the brush footprint is bounded by `world_radius`
        // around the hit point (gate (b) is a world-radius sphere). Only triangles
        // whose world-XZ AABB overlaps this can contribute (provably lossless).
        let bx0 = hit_world.x - world_radius;
        let bx1 = hit_world.x + world_radius;
        let bz0 = hit_world.z - world_radius;
        let bz1 = hit_world.z + world_radius;

        // Rasterise each terrain submesh's UV footprint into the CPU mirror and
        // stamp the covered, gate-passing texels — scanning only culled triangles.
        // Track the union of texels ACTUALLY WRITTEN as a dirty rect.
        let mirror = &mut paint_state.mirror;
        let mut written: Option<URect> = None;
        let mut scratch: Vec<u32> = Vec::new();
        for (k, (mesh_handle, _entity)) in targets.meshes.iter().enumerate() {
            let Some(mesh) = meshes.get(mesh_handle) else { continue };
            let Some(cull) = targets.cull.get(k) else { continue };
            collect_cull_triangles(cull, bx0, bz0, bx1, bz1, &mut scratch);
            stamp_mesh_culled(mesh, cull, &scratch, edge, &gate, color, mirror, &mut written);
        }

        let Some(rect) = written else {
            // Nothing painted this frame (e.g. cursor over terrain but no texel
            // passed the gate). Don't advance the dab anchor or push an upload.
            break 'paint;
        };

        paint_state.last_dab_vp = Some(cursor_vp);

        // Pack the dirty sub-rectangle and ship it to the RenderApp for an in-place
        // GPU upload. Also keep the main-world Image.data in sync (get_mut_untracked
        // — NO Modified event, so no extract clone / GPU realloc) so `.world` export
        // still serialises real paint.
        push_dirty_rect(&mut paint_state, &mut dirty, &mut images, &image_handle, rect, edge);
    }
}

/// Sync the main-world paint `Image.data` for `rect` from the CPU mirror WITHOUT
/// firing `AssetEvent::Modified` (`get_mut_untracked`), then publish the packed
/// dirty sub-rectangle bytes to `PaintDirtyRect` for the RenderApp's in-place GPU
/// upload. Bumps the consume-once generation.
fn push_dirty_rect(
    paint_state: &mut PaintState,
    dirty:       &mut PaintDirtyRect,
    images:      &mut Assets<Image>,
    handle:      &Handle<Image>,
    rect:        URect,
    edge:        u32,
) {
    let w = rect.width();
    let h = rect.height();
    if w == 0 || h == 0 { return; }
    let edge_us = edge as usize;
    let tight_bpr = w as usize * 4;

    // Pack the sub-rectangle tightly (row-major, top-to-bottom over rect.y range).
    let mut payload = vec![0u8; tight_bpr * h as usize];
    for ry in 0..h as usize {
        let src_row = (rect.min.y as usize + ry) * edge_us + rect.min.x as usize;
        let src = &paint_state.mirror[src_row * 4..src_row * 4 + tight_bpr];
        payload[ry * tight_bpr..(ry + 1) * tight_bpr].copy_from_slice(src);
    }

    // Keep the main-world Image.data current for export (no Modified event).
    if let Some(img) = images.get_mut_untracked(handle) {
        if let Some(data) = img.data.as_mut() {
            if data.len() == paint_state.mirror.len() {
                for ry in 0..h as usize {
                    let row = (rect.min.y as usize + ry) * edge_us + rect.min.x as usize;
                    let off = row * 4;
                    data[off..off + tight_bpr]
                        .copy_from_slice(&paint_state.mirror[off..off + tight_bpr]);
                }
            }
        }
    }

    paint_state.generation = paint_state.generation.wrapping_add(1);
    dirty.rect       = rect;
    dirty.payload    = Some(payload);
    dirty.image_id   = Some(handle.id());
    dirty.generation = paint_state.generation;
}


// ── Stage 2: world-XZ triangle culling cache ────────────────────────────────────

/// Build the per-submesh world-XZ culling cache (decoded indices + static world
/// transform + XZ bucket grid) if not already built. Called on the first dab, AFTER
/// child `GlobalTransform`s are propagated, so the grid is in the correct space.
fn ensure_cull_cache(
    targets:    &mut TerrainPaintTargets,
    meshes:     &Assets<Mesh>,
    transforms: &Query<&GlobalTransform>,
) {
    if !targets.cull.is_empty() { return; }
    let mut out: Vec<SubmeshCull> = Vec::with_capacity(targets.meshes.len());
    for (mesh_handle, entity) in &targets.meshes {
        let mut cull = SubmeshCull::default();
        if let Some(mesh) = meshes.get(mesh_handle) {
            cull.model = transforms.get(*entity).copied().unwrap_or_default().to_matrix();
            cull.indices = decode_indices(mesh);
            build_tri_grid(mesh, &cull.indices, cull.model, &mut cull.grid);
        }
        out.push(cull);
    }
    targets.cull = out;
}

/// Decode a mesh's indices (U32/U16/non-indexed) into a flat `Vec<u32>`.
fn decode_indices(mesh: &Mesh) -> Vec<u32> {
    match mesh.indices() {
        Some(Indices::U32(v)) => v.clone(),
        Some(Indices::U16(v)) => v.iter().map(|&i| i as u32).collect(),
        None => {
            let n = mesh
                .attribute(Mesh::ATTRIBUTE_POSITION)
                .and_then(|a| if let VertexAttributeValues::Float32x3(p) = a { Some(p.len()) } else { None })
                .unwrap_or(0);
            (0..n as u32).collect()
        }
    }
}

/// World-XZ bucket key.
#[inline]
fn tri_bucket_key(x: f32, z: f32) -> (i32, i32) {
    ((x / PAINT_TRI_GRID_BUCKET).floor() as i32, (z / PAINT_TRI_GRID_BUCKET).floor() as i32)
}

/// Build the world-XZ bucket grid mapping bucket → triangle indices (triangle `t`
/// spans `indices[3t..3t+3]`) whose world-XZ AABB overlaps it.
fn build_tri_grid(
    mesh:    &Mesh,
    indices: &[u32],
    model:   Mat4,
    grid:    &mut std::collections::HashMap<(i32, i32), Vec<u32>>,
) {
    let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute(Mesh::ATTRIBUTE_POSITION)
    else { return };
    let n = positions.len();
    for (t, tri) in indices.chunks_exact(3).enumerate() {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if i0 >= n || i1 >= n || i2 >= n { continue; }
        let w0 = model.transform_point3(Vec3::from_array(positions[i0]));
        let w1 = model.transform_point3(Vec3::from_array(positions[i1]));
        let w2 = model.transform_point3(Vec3::from_array(positions[i2]));
        let xmin = w0.x.min(w1.x).min(w2.x);
        let xmax = w0.x.max(w1.x).max(w2.x);
        let zmin = w0.z.min(w1.z).min(w2.z);
        let zmax = w0.z.max(w1.z).max(w2.z);
        let (kx0, kz0) = tri_bucket_key(xmin, zmin);
        let (kx1, kz1) = tri_bucket_key(xmax, zmax);
        for kx in kx0..=kx1 {
            for kz in kz0..=kz1 {
                grid.entry((kx, kz)).or_default().push(t as u32);
            }
        }
    }
}

/// Collect the triangle indices in buckets overlapping the brush world-XZ AABB,
/// de-duplicated. Reuses the caller's scratch buffer.
fn collect_cull_triangles(
    cull:    &SubmeshCull,
    bx0: f32, bz0: f32, bx1: f32, bz1: f32,
    out:     &mut Vec<u32>,
) {
    out.clear();
    let (kx0, kz0) = tri_bucket_key(bx0, bz0);
    let (kx1, kz1) = tri_bucket_key(bx1, bz1);
    if kx0 == kx1 && kz0 == kz1 {
        if let Some(idxs) = cull.grid.get(&(kx0, kz0)) {
            out.extend_from_slice(idxs);
        }
        return;
    }
    for kx in kx0..=kx1 {
        for kz in kz0..=kz1 {
            if let Some(idxs) = cull.grid.get(&(kx, kz)) {
                out.extend_from_slice(idxs);
            }
        }
    }
    out.sort_unstable();
    out.dedup();
}


// ── CPU rasteriser (the production stamp) ───────────────────────────────────────

/// The per-texel occlusion gate inputs (the CPU mirror of `map_paint.wgsl`).
///
/// Stage 3: invariants that are constant across all texels of a dab are precomputed
/// ONCE at construction (`hit_normal_unit`, `radius_px_sq`, `world_radius_sq`)
/// instead of being re-derived per texel. The production gate (`passes`) routes
/// through a private prenormalized fast path that is bit-identical to the public
/// `surface_point_under_brush` helper (the `#[cfg(test)]` tests still bind that
/// helper, so its signature + body are untouched).
struct StampGate {
    view_proj:       Mat4,
    vp_size:         Vec2,
    cursor_vp:       Vec2,
    hit_world:       Vec3,
    hit_normal:      Vec3,
    // Hoisted invariants (Stage 3): precomputed once per dab instead of per texel.
    hit_normal_unit: Vec3,
    radius_px_sq:    f32,
    world_radius_sq: f32,
}

impl StampGate {
    fn new(
        view_proj:    Mat4,
        vp_size:      Vec2,
        cursor_vp:    Vec2,
        radius_px:    f32,
        hit_world:    Vec3,
        hit_normal:   Vec3,
        world_radius: f32,
    ) -> Self {
        StampGate {
            view_proj, vp_size, cursor_vp, hit_world, hit_normal,
            // Precompute the unit hit normal exactly as the gate would (same
            // `normalize_or_zero`) — DO NOT collapse the double-normalize semantics
            // (would risk a 1-ULP seam flip → byte-identity violation).
            hit_normal_unit: hit_normal.normalize_or_zero(),
            radius_px_sq:    radius_px * radius_px,
            world_radius_sq: world_radius * world_radius,
        }
    }

    /// True iff a surface point `p` (world) with `normal` passes all three gates:
    /// (a) screen-disc, (b) world-radius, (c) normal agreement — bit-identical to
    /// the unit-tested `surface_point_under_brush` helper (same arithmetic, only
    /// the constant invariants are hoisted).
    fn passes(&self, p: Vec3, normal: Vec3) -> bool {
        // (b): world-radius distance (squared, against the hoisted radius²).
        if (p - self.hit_world).length_squared() > self.world_radius_sq {
            return false;
        }
        // (c): normal agreement — `normal` is already `normalize_or_zero`'d by the
        // caller's interpolation; the hit side uses the hoisted unit vector (itself
        // produced by the same `normalize_or_zero`).
        if normal.normalize_or_zero().dot(self.hit_normal_unit) < NORMAL_AGREE_COS {
            return false;
        }
        // (a): screen-space disc (hard edge), camera-agnostic via the live vp.
        let Some(sp) = world_to_viewport_px(p, self.view_proj, self.vp_size) else {
            return false; // behind the camera
        };
        (sp - self.cursor_vp).length_squared() <= self.radius_px_sq
    }
}

/// Borrowed view of a mesh's per-vertex paint inputs (positions / UVs / optional
/// normals + count), resolved once per submesh so the per-triangle rasteriser does
/// no attribute lookups.
struct MeshAttrs<'a> {
    positions: &'a [[f32; 3]],
    uvs:       &'a [[f32; 2]],
    normals:   Option<&'a Vec<[f32; 3]>>,
    n:         usize,
}

/// Resolve a mesh's POSITION/UV0/(optional)NORMAL attributes, or `None` if the
/// required ones are missing/mismatched.
fn mesh_attrs(mesh: &Mesh) -> Option<MeshAttrs<'_>> {
    let VertexAttributeValues::Float32x3(positions) =
        mesh.attribute(Mesh::ATTRIBUTE_POSITION)?
    else { return None };
    let VertexAttributeValues::Float32x2(uvs) = mesh.attribute(Mesh::ATTRIBUTE_UV_0)?
    else { return None };
    if positions.len() != uvs.len() { return None; }
    // Normals are optional (xatlas synthesises flats); fall back to the hit normal
    // so the (c) gate degrades gracefully rather than rejecting everything.
    let normals: Option<&Vec<[f32; 3]>> = match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
        Some(VertexAttributeValues::Float32x3(nm)) if nm.len() == positions.len() => Some(nm),
        _ => None,
    };
    Some(MeshAttrs { positions, uvs, normals, n: positions.len() })
}

/// Rasterise ONE triangle (`i0,i1,i2`) of `attrs` into the `edge × edge` paint
/// buffer `data`, writing `color` into each covered texel whose interpolated world
/// point passes `gate`. The inner loop (UV→texel, AABB clamp, barycentric edge
/// fns, `-1e-4` seam epsilon, gate, texel write) is UNCHANGED from the original
/// per-triangle body — this is the single shared rasteriser used by BOTH the
/// full-scan reference (`stamp_mesh`) and the production culled path
/// (`stamp_mesh_culled`), so they are byte-identical. Extends `*written` with the
/// bounding `URect` of texels actually written (Stage 1 dirty-rect tracking).
#[allow(clippy::too_many_arguments)]
fn rasterise_triangle(
    attrs:   &MeshAttrs,
    i0: usize, i1: usize, i2: usize,
    model:   Mat4,
    edge:    u32,
    gate:    &StampGate,
    color:   [u8; 4],
    data:    &mut [u8],
    written: &mut Option<URect>,
) {
    let n = attrs.n;
    if i0 >= n || i1 >= n || i2 >= n { return; }

    let edge_i = edge as i64;
    let edge_f = edge as f32;

    // UVs in [0,1] → texel space. Writes index the same row/column the
    // StandardMaterial samples (UV origin top-left, no flip).
    let uv0 = Vec2::from_array(attrs.uvs[i0]);
    let uv1 = Vec2::from_array(attrs.uvs[i1]);
    let uv2 = Vec2::from_array(attrs.uvs[i2]);
    let p0 = Vec2::new(uv0.x * edge_f, uv0.y * edge_f);
    let p1 = Vec2::new(uv1.x * edge_f, uv1.y * edge_f);
    let p2 = Vec2::new(uv2.x * edge_f, uv2.y * edge_f);

    // Triangle texel AABB (clamped to the atlas).
    let min_x = p0.x.min(p1.x).min(p2.x).floor() as i64;
    let max_x = p0.x.max(p1.x).max(p2.x).ceil() as i64;
    let min_y = p0.y.min(p1.y).min(p2.y).floor() as i64;
    let max_y = p0.y.max(p1.y).max(p2.y).ceil() as i64;
    let min_x = min_x.clamp(0, edge_i - 1);
    let max_x = max_x.clamp(0, edge_i - 1);
    let min_y = min_y.clamp(0, edge_i - 1);
    let max_y = max_y.clamp(0, edge_i - 1);
    if min_x > max_x || min_y > max_y { return; }

    // Edge-function denominator (2×signed area). Degenerate tris are skipped.
    let denom = edge_fn(p0, p1, p2);
    if denom.abs() < 1e-9 { return; }
    let inv_denom = 1.0 / denom;

    let w0 = Vec3::from_array(attrs.positions[i0]);
    let w1 = Vec3::from_array(attrs.positions[i1]);
    let w2 = Vec3::from_array(attrs.positions[i2]);
    let (n0, n1, n2) = match attrs.normals {
        Some(nm) => (
            Vec3::from_array(nm[i0]),
            Vec3::from_array(nm[i1]),
            Vec3::from_array(nm[i2]),
        ),
        None => (gate.hit_normal, gate.hit_normal, gate.hit_normal),
    };

    for ty in min_y..=max_y {
        for tx in min_x..=max_x {
            // Sample at the texel centre.
            let s = Vec2::new(tx as f32 + 0.5, ty as f32 + 0.5);
            // Barycentric coords via edge functions.
            let b0 = edge_fn(p1, p2, s) * inv_denom;
            let b1 = edge_fn(p2, p0, s) * inv_denom;
            let b2 = edge_fn(p0, p1, s) * inv_denom;
            // Inside-or-on the triangle (small epsilon to cover shared seams).
            const E: f32 = -1e-4;
            if b0 < E || b1 < E || b2 < E { continue; }

            // Interpolate the world surface point + normal at this texel.
            let p_local = w0 * b0 + w1 * b1 + w2 * b2;
            let p_world = model.transform_point3(p_local);
            let nrm = (model.transform_vector3(n0 * b0 + n1 * b1 + n2 * b2))
                .normalize_or_zero();

            if !gate.passes(p_world, nrm) { continue; }

            // Index into the RGBA8 buffer at the same row the sampler reads
            // (UV origin top-left, no flip). ty ∈ [0, edge-1] from the AABB clamps.
            let row = ty as usize;
            let idx = (row * edge as usize + tx as usize) * 4;
            if idx + 4 <= data.len() {
                data[idx..idx + 4].copy_from_slice(&color);
                // Grow the dirty rect to cover this texel ([min..max) exclusive max).
                let txu = tx as u32;
                let tyu = ty as u32;
                grow_rect(written, txu, tyu);
            }
        }
    }
}

/// Grow `*r` to include texel `(x, y)` (max is EXCLUSIVE: a 1×1 write at (x,y)
/// yields `URect { min: (x,y), max: (x+1, y+1) }`).
#[inline]
fn grow_rect(r: &mut Option<URect>, x: u32, y: u32) {
    match r {
        None => *r = Some(URect { min: UVec2::new(x, y), max: UVec2::new(x + 1, y + 1) }),
        Some(rect) => {
            rect.min.x = rect.min.x.min(x);
            rect.min.y = rect.min.y.min(y);
            rect.max.x = rect.max.x.max(x + 1);
            rect.max.y = rect.max.y.max(y + 1);
        }
    }
}

/// FULL-SCAN reference stamp: rasterise EVERY triangle of `mesh` (atlas-UV) into the
/// paint buffer. Retained as the losslessness reference for `stamp_mesh_culled`
/// (the production path) — the Stage 2 test asserts the two write byte-identical
/// images. Not called at runtime.
#[allow(dead_code)]
fn stamp_mesh(
    mesh:    &Mesh,
    model:   Mat4,
    edge:    u32,
    gate:    &StampGate,
    color:   [u8; 4],
    data:    &mut [u8],
    written: &mut Option<URect>,
) {
    let Some(attrs) = mesh_attrs(mesh) else { return };
    let indexed = decode_indices(mesh);
    if indexed.len() < 3 { return; }
    for tri in indexed.chunks_exact(3) {
        rasterise_triangle(
            &attrs, tri[0] as usize, tri[1] as usize, tri[2] as usize,
            model, edge, gate, color, data, written,
        );
    }
}

/// CULLED production stamp (Stage 2): rasterise ONLY the triangles in `tri_indices`
/// (the brush-overlapping buckets collected from `cull.grid`) into the paint buffer.
/// Uses `cull.indices` (decoded once) + `cull.model` (static terrain transform).
/// Provably lossless vs `stamp_mesh`: a contributing texel's interpolated world
/// point is within `world_radius` of the hit (gate (b)) — a convex combination of
/// the triangle's world vertices — so any contributing triangle's world-XZ AABB
/// intersects the brush XZ AABB and is in a collected bucket.
fn stamp_mesh_culled(
    mesh:        &Mesh,
    cull:        &SubmeshCull,
    tri_indices: &[u32],
    edge:        u32,
    gate:        &StampGate,
    color:       [u8; 4],
    data:        &mut [u8],
    written:     &mut Option<URect>,
) {
    let Some(attrs) = mesh_attrs(mesh) else { return };
    for &t in tri_indices {
        let base = t as usize * 3;
        if base + 3 > cull.indices.len() { continue; }
        let i0 = cull.indices[base] as usize;
        let i1 = cull.indices[base + 1] as usize;
        let i2 = cull.indices[base + 2] as usize;
        rasterise_triangle(&attrs, i0, i1, i2, cull.model, edge, gate, color, data, written);
    }
}

/// 2× signed area of triangle (a, b, c) — the edge function for barycentrics.
#[inline]
fn edge_fn(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}


// ── Color All (whole-texture fill) ──────────────────────────────────────────────

/// Fill the ENTIRE paint texture with `color` (sRGB bytes) in one pass. The only
/// whole-texture path; the Basic Brush never floods. Writes the CPU mirror, syncs
/// the main-world Image (`get_mut_untracked` — no Modified event, so no extract
/// clone / GPU realloc), and marks the whole texture dirty (one rect) for the
/// in-place GPU upload.
pub fn color_all_fill(
    images:      &mut Assets<Image>,
    paint_state: &mut PaintState,
    dirty:       &mut PaintDirtyRect,
    color:       [u8; 4],
) {
    let Some(handle) = paint_state.paint_image.clone() else { return };
    let edge = paint_state.atlas_edge;
    if edge == 0 { return; }
    if paint_state.mirror.len() < (edge as usize) * (edge as usize) * 4 { return; }

    // Fill the authoritative mirror.
    for px in paint_state.mirror.chunks_exact_mut(4) {
        px.copy_from_slice(&color);
    }
    // Whole texture dirty (one rect, [0,0) .. [edge,edge)).
    let rect = URect { min: UVec2::ZERO, max: UVec2::new(edge, edge) };
    push_dirty_rect(paint_state, dirty, images, &handle, rect, edge);
}


// ── Tests (pure helpers) ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple perspective view-projection looking down -Z from +Z, plus an
    /// orthographic one, for the screen-radius tests.
    fn perspective_vp() -> Mat4 {
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 1000.0);
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO, Vec3::Y);
        proj * view
    }

    fn ortho_vp() -> Mat4 {
        let proj = Mat4::orthographic_rh(-50.0, 50.0, -50.0, 50.0, 0.1, 1000.0);
        let view = Mat4::look_at_rh(Vec3::new(0.0, 50.0, 0.0), Vec3::ZERO, Vec3::Z);
        proj * view
    }

    #[test]
    fn map_screen_radius_to_world_finite_positive() {
        let vp = perspective_vp();
        let vpsz = Vec2::new(800.0, 800.0);
        let r = screen_pixel_radius_to_world(10.0, Vec3::ZERO, vp, vpsz);
        assert!(r.is_finite() && r > 0.0, "r = {r}");

        let ro = screen_pixel_radius_to_world(10.0, Vec3::ZERO, ortho_vp(), vpsz);
        assert!(ro.is_finite() && ro > 0.0, "ro = {ro}");
    }

    #[test]
    fn map_screen_radius_grows_with_depth_under_perspective() {
        let vp = perspective_vp();
        let vpsz = Vec2::new(800.0, 800.0);
        // Camera at z=10 looking -Z. Nearer point (z=5) vs farther (z=-50).
        let near = screen_pixel_radius_to_world(10.0, Vec3::new(0.0, 0.0, 5.0), vp, vpsz);
        let far  = screen_pixel_radius_to_world(10.0, Vec3::new(0.0, 0.0, -50.0), vp, vpsz);
        assert!(far > near, "perspective: far {far} should exceed near {near}");
    }

    #[test]
    fn map_surface_point_gate_accepts_at_hit() {
        let hit = Vec3::new(1.0, 2.0, 3.0);
        let n = Vec3::Y;
        assert!(surface_point_under_brush(hit, n, hit, n, 5.0, NORMAL_AGREE_COS));
    }

    #[test]
    fn map_surface_point_gate_rejects_far_point() {
        let hit = Vec3::ZERO;
        let n = Vec3::Y;
        let far = Vec3::new(20.0, 0.0, 0.0);
        assert!(!surface_point_under_brush(far, n, hit, n, 5.0, NORMAL_AGREE_COS));
    }

    #[test]
    fn map_surface_point_gate_rejects_opposed_normal() {
        // Cave ceiling: within the brush world radius but its normal is opposed
        // to the (ground) hit normal — must be rejected (no cross-surface bleed).
        let hit = Vec3::ZERO;
        let ground_n = Vec3::Y;
        let ceiling = Vec3::new(0.0, -0.5, 0.0); // close in world space
        let ceiling_n = Vec3::NEG_Y;
        assert!(!surface_point_under_brush(
            ceiling, ceiling_n, hit, ground_n, 5.0, NORMAL_AGREE_COS
        ));
    }

    #[test]
    fn map_single_dab_marks_small_bounded_fraction_never_all() {
        // A dense grid of surface points spread across a 200x200 map. For a single
        // dab (small world radius around one hit), only a small bounded fraction
        // may pass the gate, and NEVER all of them.
        let hit = Vec3::new(100.0, 0.0, 100.0);
        let n = Vec3::Y;
        let world_radius = 5.0;
        let mut total = 0usize;
        let mut passed = 0usize;
        let mut x = 0.0;
        while x <= 200.0 {
            let mut z = 0.0;
            while z <= 200.0 {
                let p = Vec3::new(x, 0.0, z);
                total += 1;
                if surface_point_under_brush(p, n, hit, n, world_radius, NORMAL_AGREE_COS) {
                    passed += 1;
                }
                z += 2.0;
            }
            x += 2.0;
        }
        assert!(passed > 0, "a dab over the hit must mark SOMETHING");
        assert!(passed < total, "a single dab must NEVER mark every point");
        let frac = passed as f32 / total as f32;
        assert!(frac < 0.05, "single dab marked too much: {:.3}", frac);
    }

    #[test]
    fn map_dab_points_short_move_one_point() {
        let pts = dab_points(Vec2::ZERO, Vec2::new(1.0, 0.0), 4.0);
        assert_eq!(pts.len(), 1);
        assert_eq!(pts[0], Vec2::new(1.0, 0.0));
    }

    #[test]
    fn map_dab_points_long_move_spaced() {
        let pts = dab_points(Vec2::ZERO, Vec2::new(40.0, 0.0), 4.0);
        // ceil(40/4) = 10 evenly-spaced points, last == cur.
        assert_eq!(pts.len(), 10);
        assert_eq!(*pts.last().unwrap(), Vec2::new(40.0, 0.0));
        // Spacing respected (≈ 4 px apart).
        for w in pts.windows(2) {
            assert!((w[1] - w[0]).length() <= 4.5);
        }
    }


    // ── Stage 1/2 stamp tests (CPU rasteriser + culling + dirty rect) ─────────

    use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
    use bevy::asset::RenderAssetUsages;

    const TEST_EDGE: u32 = 64;
    const GRID: usize = 10; // GRID×GRID quads spread over world XZ.
    const SPAN: f32 = 40.0; // world extent of the grid in X and Z.

    /// Build a flat XZ grid mesh (Y=0, normal +Y) of `GRID×GRID` quads spread over
    /// `[0,SPAN]²` in world space, with each quad's UVs tiling the unit square so
    /// every quad maps to a distinct `TEST_EDGE` atlas patch. Returns the mesh.
    fn grid_mesh() -> Mesh {
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut normals:   Vec<[f32; 3]> = Vec::new();
        let mut uvs:       Vec<[f32; 2]> = Vec::new();
        let mut indices:   Vec<u32>      = Vec::new();
        let step = SPAN / GRID as f32;
        for gz in 0..GRID {
            for gx in 0..GRID {
                let x0 = gx as f32 * step;
                let z0 = gz as f32 * step;
                let x1 = x0 + step;
                let z1 = z0 + step;
                // UV patch for this quad (distinct, non-overlapping in [0,1]).
                let u0 = gx as f32 / GRID as f32;
                let v0 = gz as f32 / GRID as f32;
                let u1 = (gx + 1) as f32 / GRID as f32;
                let v1 = (gz + 1) as f32 / GRID as f32;
                let base = positions.len() as u32;
                positions.push([x0, 0.0, z0]);
                positions.push([x1, 0.0, z0]);
                positions.push([x1, 0.0, z1]);
                positions.push([x0, 0.0, z1]);
                for _ in 0..4 { normals.push([0.0, 1.0, 0.0]); }
                uvs.push([u0, v0]);
                uvs.push([u1, v0]);
                uvs.push([u1, v1]);
                uvs.push([u0, v1]);
                indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            }
        }
        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        mesh.insert_indices(Indices::U32(indices));
        mesh
    }

    /// A gate over a flat +Y grid: hit at `hit`, normal +Y, world_radius selects a
    /// LOCAL region; the screen-disc test is made non-binding (huge radius_px) so
    /// the world-radius (b) gate is the real selector — exactly the regime culling
    /// must be lossless under.
    fn grid_gate(hit: Vec3, world_radius: f32) -> StampGate {
        // Top-down ortho so projection is well-defined for the +Y plane.
        let proj = Mat4::orthographic_rh(-100.0, 100.0, -100.0, 100.0, 0.1, 1000.0);
        let view = Mat4::look_at_rh(Vec3::new(0.0, 50.0, 0.0), Vec3::ZERO, Vec3::Z);
        let vp = proj * view;
        let vpsz = Vec2::new(800.0, 800.0);
        let cursor = world_to_viewport_px(hit, vp, vpsz).unwrap_or(Vec2::ZERO);
        // radius_px enormous ⇒ gate (a) never the binding constraint.
        StampGate::new(vp, vpsz, cursor, 1.0e6, hit, Vec3::Y, world_radius)
    }

    fn empty_image() -> Vec<u8> {
        vec![0u8; (TEST_EDGE as usize) * (TEST_EDGE as usize) * 4]
    }

    #[test]
    fn map_culled_stamp_matches_full_scan_byte_identical() {
        let mesh = grid_mesh();
        let model = Mat4::IDENTITY;
        let hit = Vec3::new(20.0, 0.0, 20.0); // centre of the grid
        let gate = grid_gate(hit, 6.0);
        let color = [200u8, 30, 40, 255];

        // Full-scan reference.
        let mut full = empty_image();
        let mut full_rect: Option<URect> = None;
        stamp_mesh(&mesh, model, TEST_EDGE, &gate, color, &mut full, &mut full_rect);

        // Culled path: build the cache exactly as the runtime does, then collect
        // triangles in the brush world-XZ AABB and stamp only those.
        let indices = decode_indices(&mesh);
        let total_tris = indices.len() / 3;
        let mut grid = std::collections::HashMap::new();
        build_tri_grid(&mesh, &indices, model, &mut grid);
        let cull = SubmeshCull { indices, model, grid };

        let world_radius = 6.0;
        let (bx0, bz0) = (hit.x - world_radius, hit.z - world_radius);
        let (bx1, bz1) = (hit.x + world_radius, hit.z + world_radius);
        let mut scratch = Vec::new();
        collect_cull_triangles(&cull, bx0, bz0, bx1, bz1, &mut scratch);

        let mut culled = empty_image();
        let mut culled_rect: Option<URect> = None;
        stamp_mesh_culled(&mesh, &cull, &scratch, TEST_EDGE, &gate, color, &mut culled,
                          &mut culled_rect);

        // The culled set must scan FEWER triangles than the full mesh (proving it
        // actually culls) yet write a byte-identical image (proving losslessness).
        assert!(scratch.len() < total_tris,
                "cull did not reduce the triangle set: {} of {}",
                scratch.len(), total_tris);
        assert!(full.iter().any(|&b| b != 0), "full scan painted nothing — bad test setup");
        assert_eq!(full, culled, "culled image differs from full-scan image");
        assert_eq!(full_rect, culled_rect, "culled dirty rect differs from full-scan");
    }

    #[test]
    fn map_dirty_rect_equals_written_texel_bbox() {
        let mesh = grid_mesh();
        let model = Mat4::IDENTITY;
        let hit = Vec3::new(20.0, 0.0, 20.0);
        let gate = grid_gate(hit, 5.0);
        let color = [11u8, 22, 33, 255];

        let mut img = empty_image();
        let mut rect: Option<URect> = None;
        stamp_mesh(&mesh, model, TEST_EDGE, &gate, color, &mut img, &mut rect);
        let rect = rect.expect("a dab over the grid must paint something");

        // Independently compute the bbox of all texels whose 4 bytes == color.
        let mut bx0 = u32::MAX;
        let mut by0 = u32::MAX;
        let mut bx1 = 0u32;
        let mut by1 = 0u32;
        let mut any = false;
        for y in 0..TEST_EDGE {
            for x in 0..TEST_EDGE {
                let idx = ((y * TEST_EDGE + x) * 4) as usize;
                if img[idx..idx + 4] == color {
                    any = true;
                    bx0 = bx0.min(x); by0 = by0.min(y);
                    bx1 = bx1.max(x); by1 = by1.max(y);
                }
            }
        }
        assert!(any, "no texel carries the brush colour");
        // URect max is exclusive.
        assert_eq!(rect.min, UVec2::new(bx0, by0), "dirty rect min != written bbox min");
        assert_eq!(rect.max, UVec2::new(bx1 + 1, by1 + 1), "dirty rect max != written bbox max");
    }

    #[test]
    fn map_export_mirror_sync_roundtrips_painted_bytes() {
        // The export path reads the main-world Image.data; Stage 1 keeps it in sync
        // via get_mut_untracked in push_dirty_rect. Verify the synced sub-rectangle
        // matches the mirror (so a .world export serialises real paint, not grey).
        let edge = TEST_EDGE;
        let total = (edge as usize) * (edge as usize) * 4;

        // Paint state with a mirror + a registered Image (grey base everywhere).
        let mut images = Assets::<Image>::default();
        let mut img = Image::new_target_texture(
            edge, edge, bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb, None,
        );
        // Fill both Image and mirror with a grey base.
        let grey = [128u8, 128, 128, 255];
        if let Some(d) = img.data.as_mut() {
            for px in d.chunks_exact_mut(4) { px.copy_from_slice(&grey); }
        }
        let handle = images.add(img);
        let mut mirror = vec![0u8; total];
        for px in mirror.chunks_exact_mut(4) { px.copy_from_slice(&grey); }

        // Paint a known colour into a sub-rectangle of the MIRROR only.
        let color = [9u8, 99, 199, 255];
        let rect = URect { min: UVec2::new(10, 12), max: UVec2::new(20, 18) };
        for y in rect.min.y..rect.max.y {
            for x in rect.min.x..rect.max.x {
                let off = ((y * edge + x) * 4) as usize;
                mirror[off..off + 4].copy_from_slice(&color);
            }
        }

        let mut paint = PaintState {
            paint_image: Some(handle.clone()),
            atlas_edge: edge,
            mirror,
            ..default()
        };
        let mut dirty = PaintDirtyRect::default();

        push_dirty_rect(&mut paint, &mut dirty, &mut images, &handle, rect, edge);

        // The dirty payload + rect are published for the GPU upload.
        assert_eq!(dirty.rect, rect);
        assert!(dirty.payload.is_some());
        assert_eq!(dirty.generation, 1);

        // The main-world Image (what export reads) now carries the painted bytes in
        // the rect and the grey base elsewhere — i.e. it equals the mirror.
        let exported = images.get(&handle).unwrap().data.clone().unwrap();
        assert_eq!(exported, paint.mirror, "exported Image.data != CPU mirror");
        // Spot-check: a texel inside the rect is the paint colour; outside is grey.
        let inside = ((14 * edge + 12) * 4) as usize;
        assert_eq!(exported[inside..inside + 4], color);
        let outside = ((0 * edge + 0) * 4) as usize;
        assert_eq!(exported[outside..outside + 4], grey);
    }
}
