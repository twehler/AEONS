// World geometry: load the simulation world from a Blender-exported `.glb`.
//
// The world is a single (or multi-mesh) glTF asset. At load time we:
//   1. Spawn the scene as a `SceneRoot` so it renders.
//   2. Walk the glTF node hierarchy and accumulate per-node transforms,
//      collecting every mesh primitive's triangles in *world space*.
//   3. Build two collision resources from those triangles:
//        - `HeightmapSampler`  — a 2D per-XZ-cell height grid for the fast
//          common case (floor placement, spawning, flood ground checks).
//        - `WorldMesh`         — the raw triangle list plus a uniform XZ
//          spatial grid for triangle-vs-AABB queries used by wall collision.
//
// Coordinate convention: glTF / Bevy are Y-up, right-handed. Blender's GLTF
// exporter handles the Z-up → Y-up conversion automatically.
//
// Path resolution: `world_path` is relative to Bevy's asset root (`assets/`).
// A leading `assets/` segment is stripped so users can pass either form.
//
// Normalisation: every loaded world is uniformly scaled and translated so
// that (a) its X extent is at least `MAP_MAX_X` and its Z extent is at least
// `MAP_MAX_Z`, and (b) the resulting AABB sits in strictly non-negative
// coordinates with its min corner at the origin. Both the collision data
// and the visual `SceneRoot` receive the same Transform, so the rendered
// world and the collision triangles stay in lockstep.
//
// Why both axes: `colony.rs` and `reproduction.rs` spawn organisms uniformly
// in `[0, MAP_MAX_X] x [0, MAP_MAX_Z]`. If only one axis met the bound the
// other could place organisms off-mesh, where `apply_world_bounds` would
// clamp them to the world edge each frame.

use bevy::prelude::*;
use bevy::gltf::{Gltf, GltfMesh, GltfNode};
use bevy::mesh::{Mesh, VertexAttributeValues, Indices, PrimitiveTopology};
use bevy::asset::AssetId;
use bevy::scene::SceneRoot;
use std::collections::{HashMap, HashSet};


// ── Tunables ────────────────────────────────────────────────────────────────

/// XZ resolution of the heightmap grid in world units. 1.0 matches the
/// integer-cell convention the rest of the codebase already uses
/// (organism cell size, gravity, climbing, etc).
const HEIGHTMAP_CELL_SIZE: f32 = 1.0;

/// Bucket edge length (world units) of the XZ spatial grid used to accelerate
/// triangle queries. Smaller = fewer triangles per query but more memory.
const TRIANGLE_GRID_BUCKET: f32 = 4.0;

/// Per-axis minimum world extent enforced at load time, and the spawn area
/// upper bound used by `colony.rs` / `reproduction.rs`. After normalisation
/// the world is guaranteed to span at least `[0, MAP_MAX_X] x [0, MAP_MAX_Z]`.
pub const MAP_MAX_X: f32 = 208.0;
pub const MAP_MAX_Z: f32 = 208.0;


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
// One Y sample per integer XZ cell of the world's bounding box. Built by
// rasterising every triangle into the grid: each cell stores the maximum
// Y of any triangle that overlaps it (so overhangs collapse to the upper
// surface — the common-case behaviour for terrain organisms walk on).

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
        let xi = ((x.floor() as i32 - self.min_x).max(0) as u32).min(self.width - 1);
        let zi = ((z.floor() as i32 - self.min_z).max(0) as u32).min(self.depth - 1);
        self.heights[(zi * self.width + xi) as usize]
    }
}


// ── WorldMesh ───────────────────────────────────────────────────────────────
//
// Triangle soup of the loaded world plus a uniform XZ spatial grid that maps
// each bucket to the indices of triangles whose XZ bounding box overlaps it.
// Used for wall / overhang queries that the heightmap alone can't answer.

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

    /// True if any triangle of the world intersects the axis-aligned box
    /// `[min, max]`. Uses the spatial grid as a broad phase, then a full
    /// triangle-vs-AABB SAT test (Akenine-Möller).
    pub fn aabb_intersects(&self, min: Vec3, max: Vec3) -> bool {
        let mut nearby = Vec::new();
        self.collect_nearby(min, max, &mut nearby);
        for ti in nearby {
            if tri_aabb_overlap(self.triangles[ti as usize], min, max) {
                return true;
            }
        }
        false
    }

    /// Highest Y of any vertex of any triangle whose XZ AABB intersects the
    /// XZ region of `[min, max]`. Returns `None` when there is no candidate.
    /// Used to compute climb height when a movement step is wall-blocked.
    pub fn max_y_in_xz(&self, min: Vec3, max: Vec3) -> Option<f32> {
        let mut nearby = Vec::new();
        self.collect_nearby(min, max, &mut nearby);
        let mut best: Option<f32> = None;
        for ti in nearby {
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
) {
    let Some(mut pending) = pending else { return };
    if pending.done { return; }

    if !asset_server.load_state(pending.handle.id()).is_loaded() {
        return;
    }
    let Some(gltf) = assets_gltf.get(&pending.handle) else { return };

    // Wait until every transitively referenced GltfNode + GltfMesh + Mesh is
    // present in its asset storage. Bevy's loader normally populates these by
    // the time `is_loaded()` returns true, but the readiness check is cheap
    // insurance against a partial frame where a primitive's Mesh handle has
    // not been inserted yet — bail out and retry next tick.
    if !nodes_ready(gltf, &assets_node, &assets_gltf_mesh, &assets_mesh) {
        return;
    }

    // From here on we commit. Setting `done` first ensures we never spawn
    // the scene twice if something below early-returns or an exception path
    // is added in the future.
    pending.done = true;

    // 1. Walk the node hierarchy with accumulated transforms and collect
    //    raw world-space triangles (in the glb's own coordinate frame).
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

    // 2. Compute the uniform scale + translation that pushes the world into
    //    `[0, *) ^ 3` and forces both XZ extents to meet the map minimums.
    //    Applied in-place to every triangle and forwarded to the SceneRoot
    //    so visuals match the collision geometry exactly.
    let (scale, translation) = compute_normalisation(&triangles);
    if scale != 1.0 || translation != Vec3::ZERO {
        for tri in triangles.iter_mut() {
            for v in tri.iter_mut() {
                *v = *v * scale + translation;
            }
        }
    }

    // 3. Spawn the visual scene with the normalisation Transform attached.
    //    Prefer the file's default scene; otherwise fall back to the first.
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
///   - `x_extent >= MAP_MAX_X` and `z_extent >= MAP_MAX_Z` (uniform up-scale
///     when needed; never down-scaled).
///   - AABB min corner is `(0, 0, 0)` — i.e. no negative coordinates anywhere.
///
/// On an empty triangle list returns `(1.0, ZERO)` so callers can no-op safely.
fn compute_normalisation(triangles: &[[Vec3; 3]]) -> (f32, Vec3) {
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
    let need_x = if extent.x > EPS { (MAP_MAX_X / extent.x).max(1.0) } else { 1.0 };
    let need_z = if extent.z > EPS { (MAP_MAX_Z / extent.z).max(1.0) } else { 1.0 };
    let scale  = need_x.max(need_z);

    // Translation cancels the scaled min: post-transform_min = scale*lo + t = 0.
    let translation = -lo * scale;

    (scale, translation)
}


// ── glTF traversal helpers ──────────────────────────────────────────────────

fn strip_assets_prefix(p: &str) -> String {
    let trimmed = p.trim_start_matches("./");
    if let Some(rest) = trimmed.strip_prefix("assets/") { return rest.to_string(); }
    if let Some(rest) = trimmed.strip_prefix("assets\\") { return rest.to_string(); }
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
    // A "root" is any node not referenced as a child of another node. This
    // works regardless of whether the file declared a default scene.
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
        // Strips/fans/lines/points contribute no collision geometry. Skipping
        // them silently is preferable to a hard error — Blender's exporter
        // can still emit auxiliary primitives (e.g. wireframe overlays) that
        // we do not want to treat as terrain.
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

    // Initialise to NEG_INFINITY so any sample wins on first write; cells
    // that get no triangle coverage are flattened to 0.0 at the end.
    let mut heights = vec![f32::NEG_INFINITY; (width * depth) as usize];

    let stamp = |heights: &mut Vec<f32>, xi: i32, zi: i32, y: f32| {
        if xi < 0 || zi < 0 || (xi as u32) >= width || (zi as u32) >= depth { return; }
        let idx = (zi as u32 * width + xi as u32) as usize;
        if y > heights[idx] { heights[idx] = y; }
    };

    for tri in triangles {
        let v0 = tri[0]; let v1 = tri[1]; let v2 = tri[2];

        // 1. Always stamp the three vertex cells. Handles thin/oblique
        //    triangles whose interiors miss every cell center.
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
    // Allow a small epsilon so triangle edges hit consistently from both
    // sides — without this, the rasteriser drops thin slivers of cells that
    // straddle a shared edge.
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
