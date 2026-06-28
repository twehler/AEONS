// World geometry: load the simulation world from a Blender-exported `.glb`.
//
// At load: spawn the scene as a `SceneRoot`, walk the glTF node hierarchy
// accumulating transforms to collect every mesh primitive's triangles in world
// space, then build two collision resources:
//   - `HeightmapSampler` — 2D per-XZ-cell height grid (fast common case).
//   - `WorldMesh`        — raw triangles + XZ spatial grid for tri-vs-AABB
//     wall-collision queries.
//
// Coordinates: glTF/Bevy are Y-up right-handed; Blender's exporter does the
// Z-up → Y-up conversion. `world_path` is relative to the asset root; a leading
// `assets/` segment is stripped.
//
// Normalisation: uniformly scale+translate so X extent ≥ map_size.x, Z extent ≥
// map_size.z, and the AABB min corner sits at the origin (no negative coords).
// The same Transform is applied to collision data and the visual SceneRoot so
// they stay in lockstep. Both axes are enforced because organisms spawn
// uniformly in `[0, x] x [0, z]`; an unmet axis would place them off-mesh.

use bevy::prelude::*;
use bevy::gltf::{Gltf, GltfMesh, GltfNode};
use bevy::mesh::{Mesh, VertexAttributeValues, Indices, PrimitiveTopology};
use bevy::asset::{AssetId, RenderAssetUsages};
use bevy::image::ImageSampler;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::scene::SceneRoot;
use std::collections::{HashMap, HashSet};

use crate::map_editor::gpu_paint::PaintState;
use crate::map_editor::terrain_paint::{make_display_material, TerrainPaintTargets};
use crate::world_format::WorldMeshEntry;


// ── Tunables ────────────────────────────────────────────────────────────────

/// XZ resolution of the heightmap grid (world units). Larger = less
/// terrain-following precision but smaller/cache-friendlier grid (at 4.0 a
/// 2048² world is ~1 MB vs 16 MB at 1.0). Set to 1.0 to match the organism cell
/// scale (`RD_HALF_SIZE = 0.5` ⇒ a 1.0-unit cell): at 4.0 each grid cell stored
/// the MAX terrain Y over an 8×-larger-than-an-organism patch, so sliders (esp.
/// ocean-floor/benthic ones) rested on the upslope peak of their cell and
/// visibly hovered, or aliased below the surface at cell boundaries. Combined
/// with the bilinear `height_at` below this makes the floor follow the mesh.
pub const HEIGHTMAP_CELL_SIZE: f32 = 1.0;

/// Bucket edge length (world units) of the XZ spatial grid used to accelerate
/// triangle queries. Smaller = fewer triangles per query but more memory.
const TRIANGLE_GRID_BUCKET: f32 = 4.0;

/// Per-axis minimum world extent enforced at load, and the spawn-area upper
/// bound used by `colony.rs`, `reproduction.rs`, the IL1 photo brain, and the
/// colony editor. Post-normalisation the world spans at least
/// `[0, x] x [0, z]`. Inserted by `main.rs` (launcher fields; default 500²).
#[derive(Resource, Clone, Copy, Debug)]
pub struct MapSize {
    pub x: f32,
    pub z: f32,
}

impl Default for MapSize {
    fn default() -> Self {
        Self {
            x: crate::simulation_settings::DEFAULT_MAP_X,
            z: crate::simulation_settings::DEFAULT_MAP_Z,
        }
    }
}


pub use crate::simulation_settings::WORLD_SAFETY_MARGIN;


// ── Plugin ──────────────────────────────────────────────────────────────────

pub struct WorldPlugin {
    /// Path to a `.glb` under the assets root. `assets/foo.glb` and
    /// `foo.glb` are both accepted.
    pub world_path: String,
}

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(WorldSettings { world_path: self.world_path.clone() });
        // Expose the source path so the Map Editor's Export derives its output.
        app.insert_resource(LoadedWorldPath(self.world_path.clone()));

        // Branch on the path extension (case-insensitive): a `.world` file is our
        // own baked format (synchronous load); anything else (`.glb`/`.gltf`) goes
        // through the existing async Gltf path, UNCHANGED.
        let ext = std::path::Path::new(&self.world_path)
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase());
        match ext.as_deref() {
            Some("world") => {
                app.init_resource::<WorldFileLoaded>();
                app.init_resource::<WorldPaintAdoption>();
                app.add_systems(Update, load_world_file);
                // Paint-state adoption only runs where the Map Editor exists
                // (`run_simulation`); under `--editor` those resources are absent.
                app.add_systems(
                    Update,
                    adopt_world_paint_state
                        .after(load_world_file)
                        .run_if(resource_exists::<PaintState>)
                        .run_if(resource_exists::<TerrainPaintTargets>),
                );
            }
            _ => {
                app.add_systems(Startup, begin_load_world);
                app.add_systems(Update,  finish_load_world);
            }
        }
    }
}

#[derive(Resource)]
struct WorldSettings { world_path: String }

/// The source world path this run loaded (the `.glb`/`.gltf`/`.world` argv).
/// Read by the Map Editor's Export to derive the `.world` output path.
#[derive(Resource, Clone)]
pub struct LoadedWorldPath(pub String);

/// Run-once guard for the synchronous `.world` loader.
#[derive(Resource, Default)]
struct WorldFileLoaded(bool);

/// Handoff from the `.world` loader to the paint-state adopter (so the loader can
/// stay free of the Map Editor's optional resources). `Some` for exactly one
/// frame after a successful load; the adopter consumes and clears it.
#[derive(Resource, Default)]
struct WorldPaintAdoption(
    Option<(
        Handle<Image>,
        Handle<StandardMaterial>,
        u32,
        Vec<(Handle<Mesh>, Entity)>,
    )>,
);

#[derive(Resource)]
struct PendingWorld {
    handle: Handle<Gltf>,
    done:   bool,
}


// ── HeightmapSampler ────────────────────────────────────────────────────────
//
// One Y sample per integer XZ cell. Each cell stores the max Y of any
// overlapping triangle, so overhangs collapse to the upper walkable surface.

#[derive(Resource)]
pub struct HeightmapSampler {
    pub heights:    Vec<f32>,
    pub width:      u32,
    pub depth:      u32,
    pub min_x:      i32,  // world-X of column 0
    pub min_z:      i32,  // world-Z of column 0
    pub max_height: f32,
}

impl HeightmapSampler {
    /// Surface Y at (x, z), bilinearly interpolated between the four nearest
    /// grid samples. Clamps to the world's XZ bounds.
    ///
    /// Each stored height represents its cell CENTER (cell `c` is centred at
    /// world `(min + c + 0.5) * cell`), so we work in cell-centre space: the
    /// continuous coordinate `f = x/cell - 0.5 - min` puts integer values on
    /// cell centres, and we lerp across the `[floor(f), floor(f)+1]` bracket.
    /// Interpolating (vs the old nearest-cell pick) removes the stair-stepping
    /// that made sliders alias above/below the rendered surface at cell edges.
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        if self.width == 0 || self.depth == 0 { return 0.0; }
        let cell = HEIGHTMAP_CELL_SIZE;
        // Cell-centre-space coords, clamped into [0, dim-1] so the +1 neighbour
        // and the interpolation weight both stay in range at the borders.
        let fx = (x / cell - 0.5 - self.min_x as f32).clamp(0.0, (self.width - 1) as f32);
        let fz = (z / cell - 0.5 - self.min_z as f32).clamp(0.0, (self.depth - 1) as f32);
        let xi0 = fx.floor() as u32;
        let zi0 = fz.floor() as u32;
        let xi1 = (xi0 + 1).min(self.width - 1);
        let zi1 = (zi0 + 1).min(self.depth - 1);
        let tx = fx - xi0 as f32;
        let tz = fz - zi0 as f32;
        let at = |xi: u32, zi: u32| self.heights[(zi * self.width + xi) as usize];
        let top = at(xi0, zi0) + (at(xi1, zi0) - at(xi0, zi0)) * tx;
        let bot = at(xi0, zi1) + (at(xi1, zi1) - at(xi0, zi1)) * tx;
        top + (bot - top) * tz
    }
}


// ── WorldMesh ───────────────────────────────────────────────────────────────
//
// Triangle soup + uniform XZ spatial grid (bucket → indices of triangles whose
// XZ bbox overlaps it). Answers wall/overhang queries the heightmap can't.

#[derive(Resource)]
pub struct WorldMesh {
    pub triangles: Vec<[Vec3; 3]>,
    grid:          HashMap<(i32, i32), Vec<u32>>,
    bucket:        f32,
}

impl WorldMesh {
    fn key(x: f32, z: f32, bucket: f32) -> (i32, i32) {
        ((x / bucket).floor() as i32, (z / bucket).floor() as i32)
    }

    fn collect_nearby(&self, min: Vec3, max: Vec3, out: &mut Vec<u32>) {
        out.clear();
        let (kx0, kz0) = Self::key(min.x, min.z, self.bucket);
        let (kx1, kz1) = Self::key(max.x, max.z, self.bucket);
        // Single-bucket fast path: AABB fits one cell, so no duplicate indices
        // and sort/dedup can be skipped. Common case (~1u cells, 4u buckets).
        if kx0 == kx1 && kz0 == kz1 {
            if let Some(idxs) = self.grid.get(&(kx0, kz0)) {
                out.extend_from_slice(idxs);
            }
            return;
        }
        for kx in kx0..=kx1 {
            for kz in kz0..=kz1 {
                if let Some(idxs) = self.grid.get(&(kx, kz)) {
                    out.extend_from_slice(idxs);
                }
            }
        }
        out.sort_unstable();
        out.dedup();
    }

    /// True if any world triangle intersects the AABB `[min, max]`. Spatial-grid
    /// broad phase + Akenine-Möller SAT test. Caller passes a scratch `Vec<u32>`
    /// so the hot wall-collision path doesn't allocate per call.
    pub fn aabb_intersects_with(&self, min: Vec3, max: Vec3, scratch: &mut Vec<u32>) -> bool {
        self.collect_nearby(min, max, scratch);
        for &ti in scratch.iter() {
            if tri_aabb_overlap(self.triangles[ti as usize], min, max) {
                return true;
            }
        }
        false
    }

    /// Allocating wrapper, kept for callers outside the per-frame hot path.
    pub fn aabb_intersects(&self, min: Vec3, max: Vec3) -> bool {
        let mut nearby = Vec::new();
        self.aabb_intersects_with(min, max, &mut nearby)
    }

    /// Same as `max_y_in_xz` but takes a scratch buffer to skip the per-call
    /// allocation. Used by `apply_movement` via a `Local<Vec<u32>>`.
    pub fn max_y_in_xz_with(&self, min: Vec3, max: Vec3, scratch: &mut Vec<u32>) -> Option<f32> {
        self.collect_nearby(min, max, scratch);
        let mut best: Option<f32> = None;
        for &ti in scratch.iter() {
            let tri = self.triangles[ti as usize];
            let txmin = tri[0].x.min(tri[1].x).min(tri[2].x);
            let txmax = tri[0].x.max(tri[1].x).max(tri[2].x);
            let tzmin = tri[0].z.min(tri[1].z).min(tri[2].z);
            let tzmax = tri[0].z.max(tri[1].z).max(tri[2].z);
            if txmax < min.x || txmin > max.x || tzmax < min.z || tzmin > max.z { continue; }
            let tymax = tri[0].y.max(tri[1].y).max(tri[2].y);
            best = Some(best.map_or(tymax, |b| b.max(tymax)));
        }
        best
    }

    /// Allocating wrapper.
    pub fn max_y_in_xz(&self, min: Vec3, max: Vec3) -> Option<f32> {
        let mut nearby = Vec::new();
        self.max_y_in_xz_with(min, max, &mut nearby)
    }

    /// Nearest point on the world mesh to `pos` within `search` units, plus the
    /// unit surface normal there (oriented toward `pos`). `None` if no triangle
    /// lies within `search`. Localized via the XZ spatial grid (and the
    /// grid is XZ-only, so triangles of vertical walls / overhangs at this XZ
    /// are included regardless of their Y), so the cost is ~the triangles in the
    /// surrounding buckets — cheap enough for per-organism, per-frame surface
    /// adhesion with no physics solver. `scratch` is a caller-owned candidate
    /// buffer reused across calls.
    pub fn closest_surface(&self, pos: Vec3, search: f32, scratch: &mut Vec<u32>) -> Option<(Vec3, Vec3)> {
        let half = Vec3::splat(search);
        self.collect_nearby(pos - half, pos + half, scratch);
        let mut best_d2 = search * search;
        let mut best: Option<(Vec3, Vec3)> = None;
        for &ti in scratch.iter() {
            let [v0, v1, v2] = self.triangles[ti as usize];
            let cp = closest_point_on_triangle(pos, v0, v1, v2);
            let d2 = (cp - pos).length_squared();
            if d2 < best_d2 {
                let mut n = (v1 - v0).cross(v2 - v0).normalize_or_zero();
                if n == Vec3::ZERO { continue; }           // degenerate triangle
                if n.dot(pos - cp) < 0.0 { n = -n; }        // face the organism
                best_d2 = d2;
                best    = Some((cp, n));
            }
        }
        best
    }

    /// Möller–Trumbore ray cast against terrain triangles, accelerated by the XZ
    /// spatial grid. Returns `(hit point, unit normal oriented toward the ray
    /// origin, toi)` of the NEAREST hit, or `None` if no triangle within `max`
    /// units along the ray is struck. The grid is XZ-only, so triangles of
    /// vertical walls / overhangs are included regardless of their Y — this makes
    /// the cast cave/overhang-correct (it finds the front-most surface the ray
    /// pierces). Used by the map editor's brush to obtain the cursor-center
    /// surface hit (the rapier collider is a flat slab and useless for this).
    pub fn raycast(&self, origin: Vec3, dir: Vec3, max: f32) -> Option<(Vec3, Vec3, f32)> {
        let dir = dir.normalize_or_zero();
        if dir == Vec3::ZERO { return None; }
        // Candidate buckets: the XZ AABB swept by the ray from origin to origin+dir*max.
        let end = origin + dir * max;
        let min = origin.min(end) - Vec3::splat(self.bucket);
        let maxb = origin.max(end) + Vec3::splat(self.bucket);
        let mut scratch = Vec::new();
        self.collect_nearby(min, maxb, &mut scratch);

        let mut best_toi = max;
        let mut best: Option<(Vec3, Vec3, f32)> = None;
        for &ti in scratch.iter() {
            let [v0, v1, v2] = self.triangles[ti as usize];
            if let Some(toi) = ray_triangle_toi(origin, dir, v0, v1, v2) {
                if toi >= 0.0 && toi < best_toi {
                    let mut n = (v1 - v0).cross(v2 - v0).normalize_or_zero();
                    if n == Vec3::ZERO { continue; }       // degenerate triangle
                    if n.dot(-dir) < 0.0 { n = -n; }        // face the ray origin
                    best_toi = toi;
                    best     = Some((origin + dir * toi, n, toi));
                }
            }
        }
        best
    }
}

/// Möller–Trumbore ray/triangle intersection. Returns the time-of-impact along
/// the (unit) ray direction, or `None` for a miss / parallel ray. Double-sided.
fn ray_triangle_toi(orig: Vec3, dir: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<f32> {
    const EPS: f32 = 1e-7;
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let p = dir.cross(e2);
    let det = e1.dot(p);
    if det.abs() < EPS { return None; }     // ray parallel to triangle
    let inv = 1.0 / det;
    let t = orig - v0;
    let u = t.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) { return None; }
    let q = t.cross(e1);
    let v = dir.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 { return None; }
    let toi = e2.dot(q) * inv;
    if toi > EPS { Some(toi) } else { None }
}

/// Closest point on triangle `(a, b, c)` to `p` (Ericson, *Real-Time Collision
/// Detection*). Handles the vertex / edge / face Voronoi regions; no allocation,
/// ~30 FLOPs.
fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 { return a; }

    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 { return b; }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + ab * v;
    }

    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 { return c; }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a + ab * v + ac * w
}


// ── Loading ────────────────────────────────────────────────────────────────

fn begin_load_world(
    settings:     Res<WorldSettings>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
) {
    let asset_path = strip_assets_prefix(&settings.world_path);
    let handle: Handle<Gltf> = asset_server.load(asset_path);
    commands.insert_resource(PendingWorld { handle, done: false });
}

fn finish_load_world(
    mut commands:     Commands,
    pending:          Option<ResMut<PendingWorld>>,
    asset_server:     Res<AssetServer>,
    assets_gltf:      Res<Assets<Gltf>>,
    assets_node:      Res<Assets<GltfNode>>,
    assets_gltf_mesh: Res<Assets<GltfMesh>>,
    assets_mesh:      Res<Assets<Mesh>>,
    map_size:         Res<MapSize>,
) {
    let Some(mut pending) = pending else { return };
    if pending.done { return; }

    if !asset_server.load_state(pending.handle.id()).is_loaded() {
        return;
    }
    let Some(gltf) = assets_gltf.get(&pending.handle) else { return };

    // Gate on every referenced GltfNode/GltfMesh/Mesh being in asset storage;
    // cheap insurance against a partial frame, retry next tick. (Async world load.)
    if !nodes_ready(gltf, &assets_node, &assets_gltf_mesh, &assets_mesh) {
        return;
    }

    // Commit. Set `done` first so we never spawn the scene twice on any
    // future early-return below.
    pending.done = true;

    // 1. Walk the node hierarchy, collecting raw triangles (glb's frame).
    let mut triangles: Vec<[Vec3; 3]> = Vec::new();
    let roots = identify_roots(gltf, &assets_node);
    for root in &roots {
        walk_node(
            root,
            GlobalTransform::IDENTITY,
            &assets_node,
            &assets_gltf_mesh,
            &assets_mesh,
            &mut triangles,
        );
    }

    if triangles.is_empty() {
        warn!(
            "World glb '{}' produced zero triangles — collision will be a flat plane at Y=0.",
            asset_server.get_path(pending.handle.id()).map(|p| p.to_string()).unwrap_or_default()
        );
    }

    // 2. Normalisation scale+translation (into `[0, *)^3`, both XZ extents at
    //    minimum). Applied in-place to triangles and forwarded to the SceneRoot.
    let (scale, translation) = compute_normalisation(&triangles, *map_size);
    if scale != 1.0 || translation != Vec3::ZERO {
        for tri in triangles.iter_mut() {
            for v in tri.iter_mut() {
                *v = *v * scale + translation;
            }
        }
    }

    // 3. Spawn the visual scene with the normalisation Transform.
    //    Prefer the file's default scene, else the first.
    if let Some(scene_handle) = gltf.default_scene.as_ref().or_else(|| gltf.scenes.first()) {
        commands.spawn((
            SceneRoot(scene_handle.clone()),
            Transform::from_translation(translation).with_scale(Vec3::splat(scale)),
            // Tagged so the map editor can find the terrain hierarchy + read its
            // world transform for vertex-colour painting.
            crate::map_editor::TerrainSceneRoot,
        ));
    } else {
        warn!("World glb has no scenes — spawning nothing visible.");
    }

    // 4. Build runtime resources from the post-normalisation triangles.
    let heightmap = build_heightmap(&triangles);
    let world_mesh = build_world_mesh(triangles);

    info!(
        "World loaded: {} triangles, scale x{:.4}, translation {:?}, heightmap {}x{} (origin x={}, z={}), max height {:.2}",
        world_mesh.triangles.len(),
        scale, translation,
        heightmap.width, heightmap.depth,
        heightmap.min_x, heightmap.min_z,
        heightmap.max_height
    );

    commands.insert_resource(heightmap);
    commands.insert_resource(world_mesh);
}


// ── `.world` synchronous loader ──────────────────────────────────────────────
//
// A `.world` file bakes the terrain (every submesh's local geometry + its world
// `Transform`), the painted atlas texture, and `MapSize`. Unlike the glb path it
// needs no async asset wait, no node walk, and no re-normalisation — geometry is
// already in final world space via the stored transforms. We rebuild the visible
// terrain hierarchy + display material, derive `WorldMesh`/`HeightmapSampler` from
// the world-space triangles (the SAME builders the glb path uses), insert the
// file's `MapSize`, and hand the paint texture/material/targets to the Map Editor
// (via `WorldPaintAdoption`) so re-entering the editor does NOT destructively
// re-unwrap the loaded atlas.

#[allow(clippy::too_many_arguments)]
fn load_world_file(
    settings:      Res<WorldSettings>,
    mut done:      ResMut<WorldFileLoaded>,
    mut adoption:  ResMut<WorldPaintAdoption>,
    mut commands:  Commands,
    mut meshes:    ResMut<Assets<Mesh>>,
    mut images:    ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if done.0 { return; }
    done.0 = true; // run-once (synchronous; no async gate)

    // Resolve the real file location. Unlike the glb path (where the AssetServer
    // re-prepends the assets root, so we strip it), a `.world` is read directly
    // with `std::fs` relative to the cwd (repo root) — so we must NOT strip
    // `assets/`. Try the path as given (handles `assets/foo.world` and absolute
    // paths), then fall back to under `assets/` (handles a bare `foo.world`).
    let Some(path) = resolve_world_file_path(&settings.world_path) else {
        error!(
            "Could not find .world file '{}' (tried as-given and under assets/) — booting a flat world.",
            settings.world_path
        );
        commands.insert_resource(build_heightmap(&[]));
        commands.insert_resource(build_world_mesh(Vec::new()));
        return;
    };
    let data = match crate::world_format::read_world(&path) {
        Ok(d) => d,
        Err(e) => {
            error!("Failed to read .world '{path}': {e} — booting a flat world.");
            // Same zero-triangle fallback the glb path produces (a flat plane at
            // Y=0), so the app still boots and the sim has finite resources.
            commands.insert_resource(build_heightmap(&[]));
            commands.insert_resource(build_world_mesh(Vec::new()));
            return;
        }
    };

    // 1. Reconstruct the paint Image from the stored bytes (same recipe as
    //    `uv_unwrap::create_paint_image`: Rgba8UnormSrgb, nearest sampler,
    //    COPY_DST for CPU re-uploads, MAIN+RENDER usages via the default).
    let tex = make_paint_image_from_bytes(
        data.texture.width, data.texture.height, &data.texture.bytes, &mut images,
    );

    // 2. Shared display material (factored helper — cannot drift from the editor).
    let display_mat = materials.add(make_display_material(tex.clone()));

    // 3. Spawn the terrain hierarchy: an IDENTITY root + one child per submesh
    //    (the submesh carries its own world Transform), and accumulate world-space
    //    triangles for the runtime collision resources.
    let mut tri: Vec<[Vec3; 3]> = Vec::new();
    let mut paint_targets: Vec<(Handle<Mesh>, Entity)> = Vec::new();
    let root = commands
        .spawn((
            crate::map_editor::TerrainSceneRoot,
            Transform::IDENTITY,
            Visibility::default(),
        ))
        .id();
    for entry in &data.meshes {
        // 3a. World-space triangles for the runtime resources.
        let m = entry.transform.to_matrix();
        for chunk in entry.indices.chunks_exact(3) {
            let p = |i: u32| {
                let a = entry.positions[i as usize];
                m.transform_point3(Vec3::from_array(a))
            };
            let (a, b, c) = (chunk[0], chunk[1], chunk[2]);
            // Guard against an out-of-range index (defensive — loader validated
            // counts, not index values).
            let n = entry.positions.len() as u32;
            if a >= n || b >= n || c >= n { continue; }
            tri.push([p(a), p(b), p(c)]);
        }
        // 3b. Bevy render mesh.
        let mesh_handle = meshes.add(build_render_mesh(entry));
        let child = commands
            .spawn((
                Mesh3d(mesh_handle.clone()),
                MeshMaterial3d(display_mat.clone()),
                entry.transform,
                Visibility::default(),
            ))
            .id();
        commands.entity(root).add_child(child);
        paint_targets.push((mesh_handle, child));
    }

    // 4. Runtime resources from the world-space triangles (verbatim builders).
    let heightmap  = build_heightmap(&tri); // borrow first
    let world_mesh = build_world_mesh(tri);  // move

    info!(
        "World (.world) loaded: {} submeshes, {} triangles, {}x{} texture, heightmap {}x{}, max height {:.2}",
        data.meshes.len(),
        world_mesh.triangles.len(),
        data.texture.width, data.texture.height,
        heightmap.width, heightmap.depth,
        heightmap.max_height,
    );

    commands.insert_resource(heightmap);
    commands.insert_resource(world_mesh);

    // 5. MapSize from the file overrides argv (the baked geometry is its source).
    commands.insert_resource(MapSize { x: data.map_x, z: data.map_z });

    // 6. Hand the paint state to `adopt_world_paint_state` (which only runs where
    //    the Map Editor's resources exist) so a round-trip edit keeps the texture.
    adoption.0 = Some((tex, display_mat, data.texture.width, paint_targets));
}

/// Populate the Map Editor's `PaintState` + `TerrainPaintTargets` from a loaded
/// `.world` so re-entering the editor adopts the loaded atlas+texture instead of
/// re-unwrapping (which would wipe the painted texture). Gated by
/// `resource_exists` on both resources, so it is inert under `--editor` (which has
/// no Map Editor) — the geometry/collision load above still happens there.
fn adopt_world_paint_state(
    mut adoption: ResMut<WorldPaintAdoption>,
    mut paint:    ResMut<PaintState>,
    mut targets:  ResMut<TerrainPaintTargets>,
) {
    let Some((tex, display_mat, atlas_edge, paint_targets)) = adoption.0.take() else { return };
    paint.paint_image = Some(tex);
    paint.display_mat = Some(display_mat);
    paint.atlas_edge  = atlas_edge;
    paint.last_dab_vp = None;
    targets.meshes    = paint_targets;
    targets.prepared  = true;
}

/// Build a Bevy render `Mesh` (TriangleList) from a `.world` submesh entry, with
/// POSITION/NORMAL/UV_0 and U32 indices.
fn build_render_mesh(entry: &WorldMeshEntry) -> Mesh {
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, entry.positions.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL,   entry.normals.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0,     entry.uv0.clone());
    mesh.insert_indices(Indices::U32(entry.indices.clone()));
    mesh
}

/// Reconstruct the paint `Image` from stored RGBA8 bytes — identical render-asset
/// configuration to `uv_unwrap::create_paint_image` (Rgba8UnormSrgb, nearest
/// sampler, COPY_DST, MAIN+RENDER usages via `RenderAssetUsages::default()`), so
/// CPU byte writes re-upload to the GPU automatically.
fn make_paint_image_from_bytes(
    width:  u32,
    height: u32,
    bytes:  &[u8],
    images: &mut Assets<Image>,
) -> Handle<Image> {
    let mut img = Image::new(
        Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
        TextureDimension::D2,
        bytes.to_vec(),
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    img.texture_descriptor.usage |= TextureUsages::COPY_DST;
    img.sampler = ImageSampler::nearest();
    images.add(img)
}


// ── Normalisation ──────────────────────────────────────────────────────────

/// Returns `(uniform_scale, translation)` applied as `v * scale + translation`
/// to every world-space vertex. Guarantees post-transform invariants:
///
///   - `x_extent >= map_size.x` and `z_extent >= map_size.z` (uniform up-scale
///     when needed; never down-scaled).
///   - AABB min corner is `(0, 0, 0)` — i.e. no negative coordinates anywhere.
///
/// On an empty triangle list returns `(1.0, ZERO)` so callers can no-op safely.
fn compute_normalisation(triangles: &[[Vec3; 3]], map_size: MapSize) -> (f32, Vec3) {
    if triangles.is_empty() { return (1.0, Vec3::ZERO); }

    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for tri in triangles {
        for v in tri {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
    }

    let extent = hi - lo;
    const EPS: f32 = 1e-6;

    // Compute the per-axis up-scale ratio only when the extent is non-zero
    // and the current size is below the per-axis target. A zero or tiny
    // extent on either axis can't be made larger by scaling (scaling a
    // point still gives a point), so we skip it rather than produce an
    // infinite ratio.
    let need_x = if extent.x > EPS { (map_size.x / extent.x).max(1.0) } else { 1.0 };
    let need_z = if extent.z > EPS { (map_size.z / extent.z).max(1.0) } else { 1.0 };
    let scale  = need_x.max(need_z);

    // Translation cancels the scaled min: post-transform_min = scale*lo + t = 0.
    let translation = -lo * scale;

    (scale, translation)
}


// ── glTF traversal helpers ──────────────────────────────────────────────────

/// Normalise a world path to one Bevy's `AssetServer` can resolve (relative to
/// `assets/`):
///   * `world.glb` / `./world.glb`              → unchanged
///   * `assets/world.glb` / `assets\world.glb`  → strip leading `assets`
///   * absolute path (launcher "Open…" dialog)  → suffix after LAST `assets/`
/// The third branch is required: dialog paths are absolute, and without it the
/// AssetServer silently fails and the world never spawns.
/// Resolve a `.world` path for a direct `std::fs` read. Returns the first of
/// `[as-given, assets/<stripped>]` that is an existing file, or `None`. (The glb
/// path can't reuse this — it hands the stripped path to the `AssetServer`, which
/// re-prepends the assets root itself.)
fn resolve_world_file_path(p: &str) -> Option<String> {
    if std::path::Path::new(p).is_file() {
        return Some(p.to_string());
    }
    let under = format!("assets/{}", strip_assets_prefix(p));
    if std::path::Path::new(&under).is_file() {
        return Some(under);
    }
    None
}

fn strip_assets_prefix(p: &str) -> String {
    let trimmed = p.trim_start_matches("./");
    if let Some(rest) = trimmed.strip_prefix("assets/")  { return rest.to_string(); }
    if let Some(rest) = trimmed.strip_prefix("assets\\") { return rest.to_string(); }
    // Otherwise: take everything after the LAST `assets` segment.
    if let Some(idx) = trimmed.rfind("/assets/") {
        return trimmed[idx + "/assets/".len()..].to_string();
    }
    if let Some(idx) = trimmed.rfind("\\assets\\") {
        return trimmed[idx + "\\assets\\".len()..].to_string();
    }
    trimmed.to_string()
}

fn nodes_ready(
    gltf:             &Gltf,
    assets_node:      &Assets<GltfNode>,
    assets_gltf_mesh: &Assets<GltfMesh>,
    assets_mesh:      &Assets<Mesh>,
) -> bool {
    for nh in &gltf.nodes {
        let Some(node) = assets_node.get(nh) else { return false };
        if let Some(mh) = &node.mesh {
            let Some(gm) = assets_gltf_mesh.get(mh) else { return false };
            for prim in &gm.primitives {
                if assets_mesh.get(&prim.mesh).is_none() { return false; }
            }
        }
    }
    true
}

fn identify_roots(gltf: &Gltf, assets_node: &Assets<GltfNode>) -> Vec<Handle<GltfNode>> {
    // A root is any node not referenced as a child; works without a default scene.
    let mut child_ids: HashSet<AssetId<GltfNode>> = HashSet::new();
    for nh in &gltf.nodes {
        if let Some(n) = assets_node.get(nh) {
            for c in &n.children { child_ids.insert(c.id()); }
        }
    }
    gltf.nodes
        .iter()
        .filter(|nh| !child_ids.contains(&nh.id()))
        .cloned()
        .collect()
}

fn walk_node(
    node_handle:      &Handle<GltfNode>,
    parent_global:    GlobalTransform,
    assets_node:      &Assets<GltfNode>,
    assets_gltf_mesh: &Assets<GltfMesh>,
    assets_mesh:      &Assets<Mesh>,
    out_triangles:    &mut Vec<[Vec3; 3]>,
) {
    let Some(node) = assets_node.get(node_handle) else { return };
    let global = parent_global.mul_transform(node.transform);

    if let Some(mesh_handle) = &node.mesh {
        if let Some(gltf_mesh) = assets_gltf_mesh.get(mesh_handle) {
            for prim in &gltf_mesh.primitives {
                if let Some(mesh) = assets_mesh.get(&prim.mesh) {
                    extract_triangles(mesh, &global, out_triangles);
                }
            }
        }
    }

    for child in &node.children {
        walk_node(child, global, assets_node, assets_gltf_mesh, assets_mesh, out_triangles);
    }
}

fn extract_triangles(mesh: &Mesh, global: &GlobalTransform, out: &mut Vec<[Vec3; 3]>) {
    if mesh.primitive_topology() != PrimitiveTopology::TriangleList {
        // Skip non-triangle primitives silently — exporters emit auxiliary
        // primitives (e.g. wireframe overlays) that aren't terrain.
        return;
    }

    let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(VertexAttributeValues::Float32x3(p)) => p,
        Some(_) => { warn!("Mesh position attribute is not Float32x3 — skipping primitive."); return; }
        None    => return,
    };

    let xform = |i: usize| -> Vec3 {
        let p = positions[i];
        global.transform_point(Vec3::new(p[0], p[1], p[2]))
    };

    match mesh.indices() {
        Some(Indices::U32(idx)) => {
            for tri in idx.chunks_exact(3) {
                let a = tri[0] as usize; let b = tri[1] as usize; let c = tri[2] as usize;
                if a >= positions.len() || b >= positions.len() || c >= positions.len() { continue; }
                out.push([xform(a), xform(b), xform(c)]);
            }
        }
        Some(Indices::U16(idx)) => {
            for tri in idx.chunks_exact(3) {
                let a = tri[0] as usize; let b = tri[1] as usize; let c = tri[2] as usize;
                if a >= positions.len() || b >= positions.len() || c >= positions.len() { continue; }
                out.push([xform(a), xform(b), xform(c)]);
            }
        }
        None => {
            // Non-indexed: positions are an implicit triangle list.
            let n = positions.len() - (positions.len() % 3);
            for i in (0..n).step_by(3) {
                out.push([xform(i), xform(i + 1), xform(i + 2)]);
            }
        }
    }
}


// ── Heightmap construction ──────────────────────────────────────────────────

fn build_heightmap(triangles: &[[Vec3; 3]]) -> HeightmapSampler {
    if triangles.is_empty() {
        return HeightmapSampler {
            heights: vec![0.0], width: 1, depth: 1,
            min_x: 0, min_z: 0, max_height: 0.0,
        };
    }

    // World XZ bounds in integer cells. Cell (xi, zi) covers world XZ
    // [min_x + xi, min_x + xi + 1) × [min_z + zi, min_z + zi + 1).
    let (mut x_lo, mut x_hi) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut z_lo, mut z_hi) = (f32::INFINITY, f32::NEG_INFINITY);
    let mut max_height = f32::NEG_INFINITY;
    for tri in triangles {
        for v in tri {
            x_lo = x_lo.min(v.x); x_hi = x_hi.max(v.x);
            z_lo = z_lo.min(v.z); z_hi = z_hi.max(v.z);
            max_height = max_height.max(v.y);
        }
    }

    let cell = HEIGHTMAP_CELL_SIZE;
    let min_x = (x_lo / cell).floor() as i32;
    let max_x = (x_hi / cell).ceil()  as i32;
    let min_z = (z_lo / cell).floor() as i32;
    let max_z = (z_hi / cell).ceil()  as i32;

    let width = ((max_x - min_x) as u32).max(1);
    let depth = ((max_z - min_z) as u32).max(1);

    // NEG_INFINITY so any sample wins on first write; uncovered cells → 0.0 at end.
    let mut heights = vec![f32::NEG_INFINITY; (width * depth) as usize];

    let stamp = |heights: &mut Vec<f32>, xi: i32, zi: i32, y: f32| {
        if xi < 0 || zi < 0 || (xi as u32) >= width || (zi as u32) >= depth { return; }
        let idx = (zi as u32 * width + xi as u32) as usize;
        if y > heights[idx] { heights[idx] = y; }
    };

    for tri in triangles {
        let v0 = tri[0]; let v1 = tri[1]; let v2 = tri[2];

        // 1. Stamp the three vertex cells — handles thin/oblique triangles
        //    whose interiors miss every cell center.
        for v in [v0, v1, v2] {
            stamp(&mut heights, (v.x / cell).floor() as i32 - min_x,
                                (v.z / cell).floor() as i32 - min_z, v.y);
        }

        // 2. Rasterise the triangle's interior at cell-center samples.
        let bx_lo = (v0.x.min(v1.x).min(v2.x) / cell).floor() as i32;
        let bx_hi = (v0.x.max(v1.x).max(v2.x) / cell).ceil()  as i32;
        let bz_lo = (v0.z.min(v1.z).min(v2.z) / cell).floor() as i32;
        let bz_hi = (v0.z.max(v1.z).max(v2.z) / cell).ceil()  as i32;

        for cz in bz_lo..bz_hi {
            for cx in bx_lo..bx_hi {
                let px = (cx as f32 + 0.5) * cell;
                let pz = (cz as f32 + 0.5) * cell;
                if let Some(y) = tri_xz_height_at(v0, v1, v2, px, pz) {
                    stamp(&mut heights, cx - min_x, cz - min_z, y);
                }
            }
        }
    }

    // Flatten unsampled cells to 0.0 so off-mesh queries return a finite Y.
    for h in heights.iter_mut() {
        if !h.is_finite() { *h = 0.0; }
    }

    if !max_height.is_finite() { max_height = 0.0; }

    HeightmapSampler {
        heights, width, depth, min_x, min_z, max_height,
    }
}

/// Y of triangle (v0, v1, v2) at XZ point (px, pz) using 2D barycentric
/// coordinates on the XZ plane. Returns `None` if the point is outside the
/// triangle or the XZ projection is degenerate.
fn tri_xz_height_at(v0: Vec3, v1: Vec3, v2: Vec3, px: f32, pz: f32) -> Option<f32> {
    let den = (v1.z - v2.z) * (v0.x - v2.x) + (v2.x - v1.x) * (v0.z - v2.z);
    if den.abs() < 1e-8 { return None; }
    let a = ((v1.z - v2.z) * (px - v2.x) + (v2.x - v1.x) * (pz - v2.z)) / den;
    let b = ((v2.z - v0.z) * (px - v2.x) + (v0.x - v2.x) * (pz - v2.z)) / den;
    let c = 1.0 - a - b;
    // Epsilon so shared edges hit consistently from both sides (else thin
    // slivers of straddling cells are dropped).
    let e = 1e-4;
    if a < -e || b < -e || c < -e { return None; }
    Some(a * v0.y + b * v1.y + c * v2.y)
}


// ── WorldMesh construction ──────────────────────────────────────────────────

pub(crate) fn build_world_mesh(triangles: Vec<[Vec3; 3]>) -> WorldMesh {
    let bucket = TRIANGLE_GRID_BUCKET;
    let mut grid: HashMap<(i32, i32), Vec<u32>> = HashMap::new();

    for (i, tri) in triangles.iter().enumerate() {
        let xmin = tri[0].x.min(tri[1].x).min(tri[2].x);
        let xmax = tri[0].x.max(tri[1].x).max(tri[2].x);
        let zmin = tri[0].z.min(tri[1].z).min(tri[2].z);
        let zmax = tri[0].z.max(tri[1].z).max(tri[2].z);

        let kx0 = (xmin / bucket).floor() as i32;
        let kx1 = (xmax / bucket).floor() as i32;
        let kz0 = (zmin / bucket).floor() as i32;
        let kz1 = (zmax / bucket).floor() as i32;

        for kx in kx0..=kx1 {
            for kz in kz0..=kz1 {
                grid.entry((kx, kz)).or_default().push(i as u32);
            }
        }
    }

    WorldMesh { triangles, grid, bucket }
}


// ── Triangle / AABB intersection (Akenine-Möller SAT) ───────────────────────

/// Standard separating-axis test: 3 AABB axes + 1 triangle normal +
/// 9 edge-cross-axis axes. Returns true iff the triangle and the box
/// `[aabb_min, aabb_max]` share at least one point.
fn tri_aabb_overlap(tri: [Vec3; 3], aabb_min: Vec3, aabb_max: Vec3) -> bool {
    let center = (aabb_min + aabb_max) * 0.5;
    let half   = (aabb_max - aabb_min) * 0.5;

    let v0 = tri[0] - center;
    let v1 = tri[1] - center;
    let v2 = tri[2] - center;

    // (a) Triangle's AABB vs query AABB on each world axis.
    let tmin = v0.min(v1).min(v2);
    let tmax = v0.max(v1).max(v2);
    if tmin.x > half.x || tmax.x < -half.x { return false; }
    if tmin.y > half.y || tmax.y < -half.y { return false; }
    if tmin.z > half.z || tmax.z < -half.z { return false; }

    // (b) Triangle plane vs AABB.
    let normal = (v1 - v0).cross(v2 - v0);
    let r = half.x * normal.x.abs() + half.y * normal.y.abs() + half.z * normal.z.abs();
    if normal.dot(v0).abs() > r { return false; }

    // (c) 9 edge × axis tests. For edge e = vb - va, the projections of
    //     the two endpoints onto e × axis are equal, so we only need the
    //     edge endpoint and the third "off-edge" vertex.
    let edges = [v1 - v0, v2 - v1, v0 - v2];
    let verts = [v0, v1, v2];
    for i in 0..3 {
        let e   = edges[i];
        let v_e = verts[i];               // on the edge
        let v_o = verts[(i + 2) % 3];     // off the edge

        // axis = e × X = (0, e.z, -e.y); v · axis = v.y * e.z - v.z * e.y
        {
            let p_e = v_e.y * e.z - v_e.z * e.y;
            let p_o = v_o.y * e.z - v_o.z * e.y;
            let r   = half.y * e.z.abs() + half.z * e.y.abs();
            if p_e.min(p_o) > r || p_e.max(p_o) < -r { return false; }
        }
        // axis = e × Y = (-e.z, 0, e.x); v · axis = -v.x * e.z + v.z * e.x
        {
            let p_e = -v_e.x * e.z + v_e.z * e.x;
            let p_o = -v_o.x * e.z + v_o.z * e.x;
            let r   = half.x * e.z.abs() + half.z * e.x.abs();
            if p_e.min(p_o) > r || p_e.max(p_o) < -r { return false; }
        }
        // axis = e × Z = (e.y, -e.x, 0); v · axis = v.x * e.y - v.y * e.x
        {
            let p_e = v_e.x * e.y - v_e.y * e.x;
            let p_o = v_o.x * e.y - v_o.y * e.x;
            let r   = half.x * e.y.abs() + half.y * e.x.abs();
            if p_e.min(p_o) > r || p_e.max(p_o) < -r { return false; }
        }
    }

    true
}


// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod world_mesh_raycast_tests {
    use super::*;

    /// A single horizontal triangle on the Y=5 plane spanning the XZ origin.
    fn one_flat_triangle() -> WorldMesh {
        build_world_mesh(vec![[
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(10.0, 5.0, 0.0),
            Vec3::new(0.0, 5.0, 10.0),
        ]])
    }

    #[test]
    fn map_raycast_hits_known_triangle() {
        let wm = one_flat_triangle();
        // Cast straight down from above (2, 50, 2) — inside the triangle's XZ.
        let hit = wm.raycast(Vec3::new(2.0, 50.0, 2.0), Vec3::NEG_Y, 1000.0);
        let (point, normal, toi) = hit.expect("ray should hit the triangle");
        assert!((point.y - 5.0).abs() < 1e-3, "hit Y = {}", point.y);
        assert!((toi - 45.0).abs() < 1e-3, "toi = {}", toi);
        // Normal points toward the ray origin (upward, +Y).
        assert!(normal.y > 0.9, "normal = {normal:?}");
    }

    #[test]
    fn map_raycast_misses_when_offset() {
        let wm = one_flat_triangle();
        // Cast down well outside the triangle's XZ footprint.
        assert!(wm.raycast(Vec3::new(-50.0, 50.0, -50.0), Vec3::NEG_Y, 1000.0).is_none());
        // Cast away from the triangle (upward) — never reaches it.
        assert!(wm.raycast(Vec3::new(2.0, 50.0, 2.0), Vec3::Y, 1000.0).is_none());
    }

    #[test]
    fn map_raycast_returns_nearest_of_stacked_surfaces() {
        // Two horizontal triangles stacked at the same XZ (a "cave": ceiling +
        // floor). A downward ray must report the upper one first.
        let wm = build_world_mesh(vec![
            [Vec3::new(0.0, 10.0, 0.0), Vec3::new(10.0, 10.0, 0.0), Vec3::new(0.0, 10.0, 10.0)],
            [Vec3::new(0.0, 0.0, 0.0),  Vec3::new(10.0, 0.0, 0.0),  Vec3::new(0.0, 0.0, 10.0)],
        ]);
        let (point, _n, _toi) = wm
            .raycast(Vec3::new(2.0, 50.0, 2.0), Vec3::NEG_Y, 1000.0)
            .expect("hit");
        assert!((point.y - 10.0).abs() < 1e-3, "nearest hit Y = {}", point.y);
    }
}
