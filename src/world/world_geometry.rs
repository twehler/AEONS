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
use bevy::asset::AssetId;
use bevy::scene::SceneRoot;
use std::collections::{HashMap, HashSet};


// ── Tunables ────────────────────────────────────────────────────────────────

/// XZ resolution of the heightmap grid (world units). Larger = less
/// terrain-following precision but smaller/cache-friendlier grid (at 4.0 a
/// 2048² world is ~1 MB vs 16 MB at 1.0).
pub const HEIGHTMAP_CELL_SIZE: f32 = 4.0;

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
        Self { x: 500.0, z: 500.0 }
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
        app.add_systems(Startup, begin_load_world);
        app.add_systems(Update,  finish_load_world);
    }
}

#[derive(Resource)]
struct WorldSettings { world_path: String }

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
    /// Surface Y at (x, z). Clamps to the world's XZ bounds.
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        if self.width == 0 || self.depth == 0 { return 0.0; }
        let xi = (((x / HEIGHTMAP_CELL_SIZE).floor() as i32 - self.min_x).max(0) as u32).min(self.width - 1);
        let zi = (((z / HEIGHTMAP_CELL_SIZE).floor() as i32 - self.min_z).max(0) as u32).min(self.depth - 1);
        self.heights[(zi * self.width + xi) as usize]
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

fn build_world_mesh(triangles: Vec<[Vec3; 3]>) -> WorldMesh {
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
