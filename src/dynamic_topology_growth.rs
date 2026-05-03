use bevy::asset::RenderAssetUsages;
use bevy::mesh::PrimitiveTopology;
use bevy::prelude::*;

pub mod glb_import;
mod grow_to_sky;
mod growth;
mod rebuild_bpa;
mod rebuild_starshaped;

// Pull the public entry point of each submodule into this scope so the rest
// of the file can call them unqualified. Tests reach them via the qualified
// `super::growth::…`, `super::rebuild_bpa::…`, `super::rebuild_starshaped::…`
// paths — this keeps the source module of each function obvious at the call
// site in tests.
use growth::{cap_face, find_growth_candidate};
use rebuild_bpa::rebuild_mesh;

pub(super) const EPS: f32 = 1.0e-4;
pub(super) const TARGET_EDGE_LEN: f32 = 1.0;
const GROWTH_INTERVAL_SECS: f32 = 0.2;
pub(super) const MAX_4_CANDIDATES: usize = 64;

/// Tip-vs-interior bias. Selection weight is
/// `exp(TIP_BIAS * (outward_gain - mean_outward_gain))`, where
/// `outward_gain = (apex - mesh_centroid) · face_normal` measures how much
/// the cap advances the surface front along its own normal.
/// - `0.0`: neutral; combined with `BRANCHING = 1.0` this is the original
///   uniform-random behavior.
/// - Positive (try `1.0` to `2.0`): prefer faces on the most exposed tips →
///   long single-line branches with fewer forks.
/// - Negative (try `-1.0` to `-2.0`): prefer faces deep in the cluster →
///   forks catch up before tips elongate, producing a fuller, bushier shape.
pub(super) const TIP_BIAS: f32 = 0.0;

/// Branching factor in `[0.0, 1.0]`. Layered on top of `TIP_BIAS`.
/// - `1.0`: unchanged — uniform-random face selection (current branching).
/// - `0.0`: zero branching. Inward-pointing pocket faces are excluded
///   entirely (they would cap toward the mesh interior); among the remaining
///   convex frontier faces, the *slowest-advancing* one is preferred. This is
///   the advancing-front / DynaMesh isotropic-inflation criterion and
///   produces a roughly spherical, hole-free expanding shell.
/// - In between: pocket faces are damped by `BRANCHING`, and convex faces
///   are weighted by `exp((TIP_BIAS - (1 - BRANCHING) * BRANCHING_SHARPNESS) ·
///   (g - mean_g))` where `g` is each candidate's outward radial gain.
pub(super) const BRANCHING: f32 = 0.9;
pub(super) const BRANCHING_SHARPNESS: f32 = 8.0;

/// Chunkiness control. Minimum allowed distance between a candidate apex and
/// any non-face vertex. The strict plant uses `1.0` (= TARGET_EDGE_LEN), which
/// forces every vertex to be ≥ 1.0 from every other vertex and produces thin
/// arms. Lowering toward `~0.5` lets the surface curve back closer to itself,
/// which fattens trunks. Below ~0.4 the surface starts visibly self-piercing.
pub(super) const ADMISSIBILITY_FLOOR: f32 = 0.3;

/// Master switch for the periodic rebuild pulse.
/// - `true`: the full pipeline runs — growth, then every
///   `REBUILD_MESH_VERTEX_COUNT` vertices the mesh is rebuilt
///   (BPA + Taubin, with star-shaped fallback). Visual result: a
///   smooth-blob-with-features that periodically reconnects itself.
/// - `false`: rebuild and smoothing are never invoked. The mesh just keeps
///   growing forever via raw `cap_face` operations on the original
///   unit-edge topology — useful for inspecting the underlying growth
///   pattern in isolation.
const BRANCH_BASED_GROWTH: bool = false;

/// Every time the topology grows by this many vertices, the entire face list
/// is rebuilt from the vertex cloud — a periodic smoothing pulse that yields
/// a closed, hole-free, mostly-spherical surface that still preserves
/// non-star features (loops, overhangs, dents). Growth resumes on the rebuilt
/// mesh until the next pulse fires.
const REBUILD_MESH_VERTEX_COUNT: usize = 50;

/// Ball-pivoting radius expressed as a multiple of the mean nearest-neighbor
/// distance. Smaller values capture finer features; larger values smooth them
/// out. ~1.2 is a good sweet spot for our growth's typical sample density.
pub(super) const BPA_RADIUS_FACTOR: f32 = 1.8;

/// Number of nearest neighbors used for per-vertex normal estimation (local
/// centroid). Higher k → smoother normals (better in noisy regions); lower k
/// → more responsive to local features.
pub(super) const NORMAL_ESTIMATION_KNN: usize = 14;

/// Taubin smoothing knobs applied as a post-process to the BPA mesh. Two
/// alternating passes per iteration: one with `+λ` (shrinking Laplacian),
/// one with `−μ` (anti-shrink), so the mesh smooths *without* losing volume.
/// Standard recipe is `λ ≈ 0.5`, `μ ≈ −0.53` (slightly larger in magnitude
/// than λ to perfectly cancel the shrink).
pub(super) const TAUBIN_LAMBDA: f32 = 0.5;
pub(super) const TAUBIN_MU: f32 = -0.53;
pub(super) const TAUBIN_PASSES: usize = 5;

/// Multiplier applied to the Taubin step for vertices below
/// `frozen_vertex_count`. A *partial* freeze: 0.0 = strict freeze (the old
/// surface never moves, but seam kinks form between old and new layers),
/// 1.0 = no freeze (cumulative drift across rebuilds). Around 0.10–0.20
/// preserves most of the drift-prevention benefit while letting the old
/// surface flex just enough to absorb the new growth layer without seams.
pub(super) const FROZEN_VERTEX_TAUBIN_RATE: f32 = 0.15;

/// Source of the seed mesh used by the plugin.
pub enum TopologySeed {
    /// Default unit cube centered at the origin.
    Cube,
    /// Path to a `.glb` triangle mesh whose vertex/face list will be
    /// imported verbatim and used as the growth seed.
    Glb(std::path::PathBuf),
}

impl Default for TopologySeed {
    fn default() -> Self {
        TopologySeed::Cube
    }
}

#[derive(Default)]
pub struct DynamicTopologyGrowthPlugin {
    pub seed: TopologySeed,
}

impl Plugin for DynamicTopologyGrowthPlugin {
    fn build(&self, app: &mut App) {
        let topology = match &self.seed {
            TopologySeed::Cube => Topology::initial_cube(),
            TopologySeed::Glb(path) => match glb_import::topology_from_glb(path) {
                Ok(t) => t,
                Err(e) => {
                    error!(
                        "GLB import failed for {:?}: {}. Falling back to cube seed.",
                        path, e
                    );
                    Topology::initial_cube()
                }
            },
        };
        app.insert_resource(GrowthTimer(Timer::from_seconds(
            GROWTH_INTERVAL_SECS,
            TimerMode::Repeating,
        )))
        .insert_resource(topology)
        .add_systems(Startup, spawn_dynamic_mesh)
        .add_systems(Update, grow_topology_system);
    }
}

#[derive(Resource)]
struct GrowthTimer(Timer);

#[derive(Component)]
struct DynamicMesh;

/// A polygon face on the mesh boundary. `vertices` are stored in CCW order
/// when viewed from outside; `normal` is the cached outward normal.
#[derive(Clone)]
pub(super) struct Face {
    pub(super) vertices: Vec<u32>,
    pub(super) normal: Vec3,
}

/// Origin used by the cube seed: the position of vertex 0 of the starting
/// cube. Stored per-`Topology` as `relative_origin`, but for the cube case
/// this constant gives the canonical value.
pub(super) const CUBE_RELATIVE_ORIGIN: Vec3 = Vec3::new(-0.5, -0.5, -0.5);

/// One entry in the OCG ("order of cell growth") vector. `index` matches the
/// position of the corresponding entry in `Topology::vertices`; `position` is
/// the vertex's offset from `Topology::relative_origin` (local coordinates
/// anchored at the seed mesh's first vertex).
#[derive(Clone, Debug)]
pub(super) struct OcgEntry {
    /// Permanent growth-order ID — equal to this entry's position in the
    /// `ocg` vec. Stored explicitly so consumers iterating the ledger
    /// don't have to track indices alongside.
    #[allow(dead_code)]
    pub(super) index: u32,
    pub(super) position: Vec3,
}

#[derive(Resource)]
pub(super) struct Topology {
    pub(super) vertices: Vec<Vec3>,
    pub(super) faces: Vec<Face>,
    rng_state: u64,
    /// Vertex count at which the next convex-hull rebuild fires. Bumped by
    /// `REBUILD_MESH_VERTEX_COUNT` each time the rebuild runs, so pulses
    /// repeat at every multiple of the threshold.
    next_rebuild_at: usize,
    /// Vertices with index `< frozen_vertex_count` are immutable during
    /// Taubin smoothing. Set to the current vertex count at the end of every
    /// rebuild, so subsequent rebuilds re-smooth only the *fresh* vertices
    /// (those added since the previous rebuild) instead of drifting the
    /// already-smoothed surface — which avoids the spiky chaos caused by
    /// cumulative re-smoothing.
    pub(super) frozen_vertex_count: usize,
    /// "Order of cell growth" — append-only historical ledger of every vertex
    /// ever appended. Each entry's `index` is its permanent growth-order ID.
    /// Orphan vertices that get dropped during a rebuild keep their OCG entry
    /// with the last-known position; only the *live* `vertices` array shrinks.
    pub(super) ocg: Vec<OcgEntry>,
    /// Parallel to `vertices`: `ocg_id_for_vertex[i]` is the OCG index of the
    /// vertex at `vertices[i]`. Compaction shrinks this in lockstep with
    /// `vertices`; growth appends to it; Taubin doesn't touch it.
    pub(super) ocg_id_for_vertex: Vec<u32>,
    /// Anchor for OCG positions: every entry's `position` is stored relative
    /// to this point. Set once at seed construction (cube or imported mesh)
    /// and never changes thereafter.
    pub(super) relative_origin: Vec3,
}

impl Topology {
    pub(super) fn initial_cube() -> Self {
        let vertices = vec![
            Vec3::new(-0.5, -0.5, -0.5), // 0
            Vec3::new(0.5, -0.5, -0.5),  // 1
            Vec3::new(0.5, -0.5, 0.5),   // 2
            Vec3::new(-0.5, -0.5, 0.5),  // 3
            Vec3::new(-0.5, 0.5, -0.5),  // 4
            Vec3::new(0.5, 0.5, -0.5),   // 5
            Vec3::new(0.5, 0.5, 0.5),    // 6
            Vec3::new(-0.5, 0.5, 0.5),   // 7
        ];
        // Six quad faces, vertices in CCW order viewed from outside.
        let faces = vec![
            Face { vertices: vec![0, 1, 2, 3], normal: Vec3::NEG_Y }, // bottom
            Face { vertices: vec![4, 7, 6, 5], normal: Vec3::Y },     // top
            Face { vertices: vec![3, 2, 6, 7], normal: Vec3::Z },     // front
            Face { vertices: vec![1, 0, 4, 5], normal: Vec3::NEG_Z }, // back
            Face { vertices: vec![2, 1, 5, 6], normal: Vec3::X },     // right
            Face { vertices: vec![0, 3, 7, 4], normal: Vec3::NEG_X }, // left
        ];
        Self::from_seed_mesh(vertices, faces, CUBE_RELATIVE_ORIGIN)
    }

    /// Build a Topology from an externally-supplied vertex/face list and an
    /// origin used for OCG-relative coordinates. Shared between the cube
    /// seed and `.glb` imports.
    pub(super) fn from_seed_mesh(
        vertices: Vec<Vec3>,
        faces: Vec<Face>,
        relative_origin: Vec3,
    ) -> Self {
        let ocg: Vec<OcgEntry> = vertices
            .iter()
            .enumerate()
            .map(|(i, v)| OcgEntry {
                index: i as u32,
                position: *v - relative_origin,
            })
            .collect();
        let ocg_id_for_vertex: Vec<u32> = (0..vertices.len() as u32).collect();
        Self {
            vertices,
            faces,
            rng_state: 0xDEAD_BEEF_CAFE_F00D,
            next_rebuild_at: REBUILD_MESH_VERTEX_COUNT,
            frozen_vertex_count: 0,
            ocg,
            ocg_id_for_vertex,
            relative_origin,
        }
    }

    /// Update OCG entries for live vertices to reflect their current
    /// positions. Orphan entries (vertices that were dropped by compaction)
    /// retain their last-known position.
    pub(super) fn sync_ocg_positions(&mut self) {
        let origin = self.relative_origin;
        for (i, &ocg_idx) in self.ocg_id_for_vertex.iter().enumerate() {
            self.ocg[ocg_idx as usize].position = self.vertices[i] - origin;
        }
    }

    pub(super) fn next_rand(&mut self) -> u64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.rng_state
    }

    pub(super) fn rand_range(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_rand() as usize) % n
        }
    }

    /// Uniform [0.0, 1.0). Used for weighted face selection (TIP_BIAS).
    pub(super) fn next_rand_unit(&mut self) -> f32 {
        let x = (self.next_rand() >> 32) as u32;
        x as f32 / (u32::MAX as f32 + 1.0)
    }
}

pub(super) fn mesh_centroid(verts: &[Vec3]) -> Vec3 {
    verts.iter().copied().sum::<Vec3>() / verts.len() as f32
}

fn spawn_dynamic_mesh(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    topology: Res<Topology>,
) {
    let mesh_handle = meshes.add(build_mesh(&topology));
    let material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.03, 0.92, 0.214),
        cull_mode: None,
        double_sided: true,
        perceptual_roughness: 0.6,
        ..default()
    });
    commands.spawn((
        Mesh3d(mesh_handle),
        MeshMaterial3d(material),
        Transform::from_xyz(0.0, 0.0, 0.0),
        DynamicMesh,
    ));
}

fn build_mesh(topo: &Topology) -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    for face in &topo.faces {
        if face.vertices.len() < 3 {
            continue;
        }
        let n = face.normal.to_array();
        // Triangulate as a fan from face.vertices[0].
        let v0 = topo.vertices[face.vertices[0] as usize];
        for i in 1..face.vertices.len() - 1 {
            let v1 = topo.vertices[face.vertices[i] as usize];
            let v2 = topo.vertices[face.vertices[i + 1] as usize];
            positions.push(v0.to_array());
            positions.push(v1.to_array());
            positions.push(v2.to_array());
            normals.push(n);
            normals.push(n);
            normals.push(n);
        }
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh
}

fn grow_topology_system(
    time: Res<Time<Virtual>>,
    mut timer: ResMut<GrowthTimer>,
    mut topology: ResMut<Topology>,
    mut meshes: ResMut<Assets<Mesh>>,
    query: Query<&Mesh3d, With<DynamicMesh>>,
) {
    timer.0.tick(time.delta());
    if !timer.0.just_finished() {
        return;
    }

    let Some(candidate) = find_growth_candidate(&mut topology) else {
        info!("dynamic_topology_growth: no valid face to cap this tick");
        return;
    };

    let connectivity = topology.faces[candidate.face_idx].vertices.len();
    cap_face(&mut topology, candidate);

    let just_rebuilt =
        BRANCH_BASED_GROWTH && topology.vertices.len() >= topology.next_rebuild_at;
    if just_rebuilt {
        rebuild_mesh(&mut topology);
        topology.next_rebuild_at = topology.vertices.len() + REBUILD_MESH_VERTEX_COUNT;
    }

    // (Pure-growth ticks need no OCG sync — `cap_face` appended its entry.
    // Rebuilds handle their own compaction + position resync internally.)

    if let Ok(mesh3d) = query.single() {
        if let Some(mesh) = meshes.get_mut(&mesh3d.0) {
            *mesh = build_mesh(&topology);
        }
    }

    if just_rebuilt {
        info!(
            "dynamic_topology_growth: mesh rebuild at {} verts; faces={}; growth resumes",
            topology.vertices.len(),
            topology.faces.len()
        );
    } else {
        info!(
            "dynamic_topology_growth: capped {}-gon face; verts={}, faces={}",
            connectivity,
            topology.vertices.len(),
            topology.faces.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn is_closed_manifold(faces: &[Face]) -> bool {
        let mut edge_count: HashMap<(u32, u32), u32> = HashMap::new();
        for face in faces {
            let k = face.vertices.len();
            for i in 0..k {
                let a = face.vertices[i];
                let b = face.vertices[(i + 1) % k];
                let edge = if a < b { (a, b) } else { (b, a) };
                *edge_count.entry(edge).or_insert(0) += 1;
            }
        }
        edge_count.values().all(|&c| c == 2)
    }

    #[test]
    fn unit_sphere_intersection_on_cube_face() {
        let a = Vec3::new(-0.5, -0.5, -0.5);
        let b = Vec3::new(0.5, -0.5, -0.5);
        let c = Vec3::new(0.5, -0.5, 0.5);
        let pts = growth::unit_sphere_intersection(a, b, c);
        assert_eq!(pts.len(), 2);
        for p in &pts {
            assert!(((*p - a).length() - 1.0).abs() < 1.0e-5);
            assert!(((*p - b).length() - 1.0).abs() < 1.0e-5);
            assert!(((*p - c).length() - 1.0).abs() < 1.0e-5);
        }
    }

    #[test]
    fn initial_cube_is_closed_manifold() {
        let topo = Topology::initial_cube();
        assert!(is_closed_manifold(&topo.faces));
        assert_eq!(topo.faces.len(), 6);
        assert!(topo.faces.iter().all(|f| f.vertices.len() == 4));
    }

    #[test]
    fn initial_cube_yields_4_connect_cap() {
        let mut topo = Topology::initial_cube();
        let cand = growth::find_growth_candidate(&mut topo)
            .expect("initial cube must yield a candidate");
        assert_eq!(
            topo.faces[cand.face_idx].vertices.len(),
            4,
            "initial cube cap must be a 4-connect (quad face)"
        );
        for &idx in &topo.faces[cand.face_idx].vertices {
            let d = (topo.vertices[idx as usize] - cand.apex).length();
            assert!((d - 1.0).abs() < 1.0e-4);
        }
    }

    #[test]
    fn manifold_preserved_after_many_caps() {
        let mut topo = Topology::initial_cube();
        let mut caps_done = 0;
        for _ in 0..30 {
            let Some(cand) = growth::find_growth_candidate(&mut topo) else {
                break;
            };
            growth::cap_face(&mut topo, cand);
            caps_done += 1;
            assert!(
                is_closed_manifold(&topo.faces),
                "non-manifold after {} caps (faces={})",
                caps_done,
                topo.faces.len()
            );
        }
        assert!(caps_done >= 6, "expected at least the 6 cube-face caps; got {}", caps_done);
    }

    #[test]
    fn euler_characteristic_stays_2() {
        // V - E + F = 2 for any closed orientable manifold of genus 0.
        let mut topo = Topology::initial_cube();
        for _ in 0..20 {
            let Some(cand) = growth::find_growth_candidate(&mut topo) else { break; };
            growth::cap_face(&mut topo, cand);
            let v = topo.vertices.len() as i32;
            let f = topo.faces.len() as i32;
            let mut edges: std::collections::HashSet<(u32, u32)> =
                std::collections::HashSet::new();
            for face in &topo.faces {
                let k = face.vertices.len();
                for i in 0..k {
                    let a = face.vertices[i];
                    let b = face.vertices[(i + 1) % k];
                    edges.insert(if a < b { (a, b) } else { (b, a) });
                }
            }
            let e = edges.len() as i32;
            assert_eq!(v - e + f, 2, "Euler characteristic broken: V={} E={} F={}", v, e, f);
        }
    }

    #[test]
    fn all_new_edges_are_unit_length() {
        let mut topo = Topology::initial_cube();
        for _ in 0..15 {
            let Some(cand) = growth::find_growth_candidate(&mut topo) else { break; };
            growth::cap_face(&mut topo, cand);
        }
        for face in &topo.faces {
            let k = face.vertices.len();
            for i in 0..k {
                let a = topo.vertices[face.vertices[i] as usize];
                let b = topo.vertices[face.vertices[(i + 1) % k] as usize];
                let len = (b - a).length();
                assert!((len - 1.0).abs() < 1.0e-3, "non-unit edge: {}", len);
            }
        }
    }

    #[test]
    fn star_shaped_rebuild_weaves_in_interior_vertices() {
        let mut topo = Topology::initial_cube();
        // Inject an interior vertex slightly off-centroid. With star-shaped
        // reconstruction it must remain in `topo.vertices` AND become
        // referenced by some face (a real inward dimple).
        let interior = Vec3::new(0.15, 0.05, -0.1);
        topo.vertices.push(interior);
        let interior_idx = (topo.vertices.len() - 1) as u32;
        let pre_count = topo.vertices.len();
        rebuild_starshaped::rebuild_as_convex_hull(&mut topo);
        assert_eq!(
            topo.vertices.len(),
            pre_count,
            "star-shaped rebuild must preserve every input vertex"
        );
        let referenced = topo
            .faces
            .iter()
            .any(|f| f.vertices.iter().any(|&v| v == interior_idx));
        assert!(referenced, "interior vertex was not woven into the surface");
    }

    #[test]
    fn convex_hull_rebuild_is_closed_manifold() {
        let mut topo = Topology::initial_cube();
        for _ in 0..40 {
            let Some(cand) = growth::find_growth_candidate(&mut topo) else { break; };
            growth::cap_face(&mut topo, cand);
        }
        rebuild_starshaped::rebuild_as_convex_hull(&mut topo);
        assert!(
            is_closed_manifold(&topo.faces),
            "rebuild produced a non-manifold surface"
        );
        // Consistent outward orientation: every directed edge (a, b) appears
        // in at most one face, and its reverse (b, a) in another. This holds
        // for any closed orientable manifold, convex or not.
        let mut directed: std::collections::HashSet<(u32, u32)> =
            std::collections::HashSet::new();
        for f in &topo.faces {
            let k = f.vertices.len();
            for i in 0..k {
                let a = f.vertices[i];
                let b = f.vertices[(i + 1) % k];
                assert!(
                    directed.insert((a, b)),
                    "directed edge ({}, {}) appears in two faces — orientation inconsistent",
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn rebuild_mesh_produces_closed_manifold_after_growth() {
        let mut topo = Topology::initial_cube();
        for _ in 0..40 {
            let Some(cand) = growth::find_growth_candidate(&mut topo) else { break; };
            growth::cap_face(&mut topo, cand);
        }
        rebuild_mesh(&mut topo);
        assert!(
            is_closed_manifold(&topo.faces),
            "BPA+Taubin rebuild produced a non-manifold surface"
        );
        // Consistent outward orientation: every directed edge appears at most once.
        let mut directed: std::collections::HashSet<(u32, u32)> =
            std::collections::HashSet::new();
        for f in &topo.faces {
            let k = f.vertices.len();
            for i in 0..k {
                let a = f.vertices[i];
                let b = f.vertices[(i + 1) % k];
                assert!(
                    directed.insert((a, b)),
                    "directed edge ({}, {}) appears in two faces — orientation inconsistent",
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn ocg_ledger_is_complete_after_rebuild_and_compaction() {
        let mut topo = Topology::initial_cube();
        assert_eq!(topo.ocg.len(), topo.vertices.len());
        assert_eq!(topo.ocg_id_for_vertex.len(), topo.vertices.len());

        let appended_count = 30;
        for _ in 0..appended_count {
            let Some(cand) = growth::find_growth_candidate(&mut topo) else { break; };
            growth::cap_face(&mut topo, cand);
        }
        let total_appended = topo.ocg.len();
        rebuild_mesh(&mut topo);

        // OCG ledger never shrinks: every vertex ever appended is still recorded.
        assert_eq!(topo.ocg.len(), total_appended);
        // Live vertices and their parallel OCG-id mapping must match in length.
        assert_eq!(topo.vertices.len(), topo.ocg_id_for_vertex.len());
        // For each live vertex, its OCG entry must reflect the current position.
        for (i, &ocg_idx) in topo.ocg_id_for_vertex.iter().enumerate() {
            let abs = topo.vertices[i];
            let rel = topo.ocg[ocg_idx as usize].position;
            assert!(
                (rel - (abs - topo.relative_origin)).length() < EPS,
                "live vertex {} (OCG idx {}) out of sync",
                i,
                ocg_idx
            );
        }
        // Every live OCG-id is unique and in range.
        let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for &ocg_idx in &topo.ocg_id_for_vertex {
            assert!((ocg_idx as usize) < topo.ocg.len());
            assert!(seen.insert(ocg_idx), "OCG id {} maps to two live vertices", ocg_idx);
        }
    }

    #[test]
    fn second_rebuild_keeps_frozen_drift_bounded() {
        let mut topo = Topology::initial_cube();
        for _ in 0..40 {
            let Some(cand) = growth::find_growth_candidate(&mut topo) else { break; };
            growth::cap_face(&mut topo, cand);
        }
        rebuild_mesh(&mut topo);
        let frozen_count = topo.frozen_vertex_count;
        // Snapshot frozen vertices by *OCG id* (their permanent growth-order
        // ID) — live indices may shift if a later rebuild compacts orphans.
        let frozen_snapshot: Vec<(u32, Vec3)> = topo
            .ocg_id_for_vertex
            .iter()
            .take(frozen_count)
            .map(|&id| (id, topo.ocg[id as usize].position))
            .collect();

        for _ in 0..6 {
            let Some(cand) = growth::find_growth_candidate(&mut topo) else { break; };
            growth::cap_face(&mut topo, cand);
        }
        rebuild_mesh(&mut topo);

        let origin = topo.relative_origin;
        let mut matched = 0usize;
        for &(ocg_id, snap_rel) in &frozen_snapshot {
            if let Some(live_idx) = topo
                .ocg_id_for_vertex
                .iter()
                .position(|&id| id == ocg_id)
            {
                let live_rel = topo.vertices[live_idx] - origin;
                let drift = (live_rel - snap_rel).length();
                assert!(
                    drift < 0.3,
                    "OCG id {} drifted by {} on second rebuild — partial freeze ineffective",
                    ocg_id,
                    drift
                );
                matched += 1;
            }
            // else: vertex was compacted out — no drift to check.
        }
        assert!(
            matched > 0,
            "no frozen vertices survived second rebuild — test is vacuous"
        );
    }
}
