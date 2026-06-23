// Volumetric growth — translation-only RD/tetra cell attachment.
//
// Mesh pipeline ("remove doubles → drop interior faces"):
//   1. Each placed cell centre lives in `centers`.
//   2. Every cell emits its 14 verts + 24 triangles in canonical (axis-aligned,
//      untranslated) form, summed with the cell centre.
//   3. Coincident vertices are welded by HashMap on quantised positions
//      (eps 1e-4, far below RD's 1.0 min vertex spacing).
//   4. A triangle shared by two adjacent cells gets identical welded indices in
//      both; sorted-key duplicates are interior faces and dropped on both sides.
//   5. Surviving triangles keep outward winding from their unique source cell.
// Hole-free by construction on the RD lattice with shared orientation.
// Whole mesh recomputed each tick (≤30 cells × 14 verts ≈ sub-ms).

use bevy::prelude::*;
use std::collections::{HashMap, HashSet};
use rand::RngExt;
use crate::cell::CellType;
use crate::colony::Organism;

type LatticeKey = (i32, i32, i32);

/// RD lattice quantization: resolution 1/2048 world units.
#[inline]
fn lattice_key(v: Vec3) -> LatticeKey {
    const S: f32 = 2048.0;
    ((v.x * S).round() as i32, (v.y * S).round() as i32, (v.z * S).round() as i32)
}

/// Tetrahedron mode: spatial hash with cell size = MIN_CENTER_DIST for O(1) proximity.
#[inline]
fn tetra_spatial_key(v: Vec3) -> LatticeKey {
    const S: f32 = 1.0 / MIN_CENTER_DIST;
    ((v.x * S).floor() as i32, (v.y * S).floor() as i32, (v.z * S).floor() as i32)
}

/// Returns true if any placed center in `occupied` is within MIN_CENTER_DIST of `candidate`.
fn tetra_is_blocked(candidate: Vec3, occupied: &HashMap<LatticeKey, Vec3>) -> bool {
    let (bx, by, bz) = tetra_spatial_key(candidate);
    let r_sq = MIN_CENTER_DIST * MIN_CENTER_DIST;
    for dx in -1i32..=1 {
        for dy in -1i32..=1 {
            for dz in -1i32..=1 {
                if let Some(&p) = occupied.get(&(bx + dx, by + dy, bz + dz)) {
                    if (p - candidate).length_squared() <= r_sq {
                        return true;
                    }
                }
            }
        }
    }
    false
}

pub mod dodecahedron;
pub mod geometry;
pub mod growth_controls;
pub mod tetrahedron;
//pub mod fill_holes;
pub mod smooth_vertices;

use geometry::build_flat_mesh;
use growth_controls::{CandidateInfo, GrowthController};

// ── Mode ──────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum GrowthMode {
    Tetrahedron,
    Dodecahedron,
}

pub const GROWTH_MODE: GrowthMode = GrowthMode::Dodecahedron;
// EDGE_LEN drives BOTH the lattice spacing (`center_scale(edge) = 4·edge/√3`,
// linear) and the per-cell cube (`seed_vertices(edge)`, linear), so it IS the
// organism-geometry length unit. Both derive from the master
// `simulation_settings::GEOMETRY_SCALE` (one knob for organism size; their
// values at scale 1.0 were 1.0 and 0.5). Paired with `cell::CELL_SPACING` and
// `body_part::MIN_X_BILATERAL`. Pre-scale `.species`/`.colony` files are
// migrated ×GEOMETRY_SCALE on load (version-gated).
const EDGE_LEN: f32 = crate::simulation_settings::GEOMETRY_SCALE;
const MIN_CENTER_DIST: f32 = 0.5 * crate::simulation_settings::GEOMETRY_SCALE;
pub use crate::simulation_settings::MAX_CELLS;

/// Vertex-weld tolerance. RD's smallest inter-vertex distance is 1.0; expected
/// float drift across ≤60 translations is ~60·ε_f32 ≈ 8e-6, so 1e-4 is safe.
const WELD_EPS: f32 = 1e-4;

#[inline]
fn weld_key(p: Vec3) -> (i64, i64, i64) {
    (
        (p.x / WELD_EPS).round() as i64,
        (p.y / WELD_EPS).round() as i64,
        (p.z / WELD_EPS).round() as i64,
    )
}

// ── Mesh assembly ────────────────────────────────────────────────────────────

/// Assemble a closed manifold mesh from a list of cell centres.
///
/// Each cell contributes its 14 vertices and 24 triangles in canonical
/// orientation (axis-aligned, translation-only). Coincident vertices are
/// welded by a HashMap on quantised positions. Sorted-key triangle pairs
/// (shared rhombic faces between adjacent cells) cancel and are dropped.
/// Surviving triangles already have outward winding because each is the
/// outward copy from its unique source cell.
fn rebuild_mesh(centers: &[Vec3]) -> (Vec<Vec3>, Vec<[u32; 3]>, Vec<u32>) {
    if centers.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let local_verts: Vec<Vec3> = match GROWTH_MODE {
        GrowthMode::Dodecahedron => dodecahedron::seed_vertices(EDGE_LEN),
        GrowthMode::Tetrahedron => {
            let h = EDGE_LEN * (2.0_f32 / 3.0).sqrt();
            tetrahedron::cell_vertices(-(h / 4.0) * Vec3::Y, Vec3::Y, EDGE_LEN).to_vec()
        }
    };
    let local_tris: Vec<[u32; 3]> = match GROWTH_MODE {
        GrowthMode::Dodecahedron => dodecahedron::cell_all_tris(),
        GrowthMode::Tetrahedron => {
            let mut t = vec![tetrahedron::BASE_TRI];
            t.extend_from_slice(&tetrahedron::non_base_tris());
            t
        }
    };

    let mut verts: Vec<Vec3> = Vec::new();
    let mut vmap: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut tris: Vec<[u32; 3]> = Vec::new();
    // Parallel to `tris`: source cell index per triangle; survives dedup so
    // callers can colour kept triangles by their cell's `CellType`.
    let mut tri_src: Vec<u32> = Vec::new();

    for (cell_idx, &c) in centers.iter().enumerate() {
        let mut local_to_global: Vec<u32> = Vec::with_capacity(local_verts.len());
        for &v in &local_verts {
            let p = c + v;
            let key = weld_key(p);
            let idx = *vmap.entry(key).or_insert_with(|| {
                let i = verts.len() as u32;
                verts.push(p);
                i
            });
            local_to_global.push(idx);
        }
        for &[a, b, c2] in &local_tris {
            tris.push([
                local_to_global[a as usize],
                local_to_global[b as usize],
                local_to_global[c2 as usize],
            ]);
            tri_src.push(cell_idx as u32);
        }
    }

    // Drop interior triangles: boundary faces have sorted-key multiplicity 1,
    // shared faces 2 (cancel). RD geometry guarantees no face is shared by 3+
    // cells, so multiplicity is always 1 or 2.
    let mut bucket: HashMap<[u32; 3], u32> = HashMap::new();
    for &t in &tris {
        let mut k = t;
        k.sort_unstable();
        *bucket.entry(k).or_insert(0) += 1;
    }
    let mut kept_tris: Vec<[u32; 3]> = Vec::with_capacity(tris.len());
    let mut kept_src:  Vec<u32>      = Vec::with_capacity(tris.len());
    for (i, &t) in tris.iter().enumerate() {
        if t[0] == t[1] || t[1] == t[2] || t[0] == t[2] { continue; }
        let mut k = t;
        k.sort_unstable();
        if bucket[&k] == 1 {
            kept_tris.push(t);
            kept_src.push(tri_src[i]);
        }
    }

    (verts, kept_tris, kept_src)
}

// ── OCG pathway ───────────────────────────────────────────────────────────────

/// Smoothing parameters used when an organism becomes `adult`. `lambda` is the
/// per-iteration blend toward the one-ring centroid (0 = identity, 1 = collapse);
/// `iterations` is the Jacobi pass count. Tuned to round the silhouette without
/// erasing branch-attachment lobes.
pub const ADULT_SMOOTH_LAMBDA:     f32   = 0.3;
pub const ADULT_SMOOTH_ITERATIONS: usize = 3;

/// Split a part's OCG into its `(opaque, translucent)` meshes so the two can
/// render with different alpha modes. Translucent cells (e.g. Gelly) MUST live
/// in a separate alpha-blended mesh — putting the whole part in one Blend
/// material drops every cell into the transparent pass (no depth write), which
/// makes the solid cells render glassy too. Each side is `None` when it has no
/// cells (the common all-opaque part → `(Some, None)`, identical to before).
/// `smoothed` selects the adult Laplacian-smoothed builder.
pub fn build_part_meshes(
    ocg:      &[(usize, Vec3, CellType)],
    smoothed: bool,
) -> (Option<Mesh>, Option<Mesh>) {
    let build = |sub: Vec<(usize, Vec3, CellType)>| -> Option<Mesh> {
        if sub.is_empty() { return None; }
        Some(if smoothed { build_smoothed_mesh_from_ocg(&sub) } else { build_mesh_from_ocg(&sub) })
    };
    let opaque = ocg.iter().filter(|(_, _, ct)| !ct.is_translucent()).cloned().collect();
    let translucent = ocg.iter().filter(|(_, _, ct)| ct.is_translucent()).cloned().collect();
    (build(opaque), build(translucent))
}

/// Un-smoothed flat mesh: each OCG entry's position is a cell centre run
/// through translate → weld → dedup. Used during growth (faceted lobes).
pub fn build_mesh_from_ocg(ocg: &[(usize, Vec3, CellType)]) -> Mesh {
    if ocg.is_empty() {
        return build_flat_mesh(&[], &[]);
    }
    let centers: Vec<Vec3> = ocg.iter().map(|(_, p, _)| *p).collect();
    let (verts, tris, tri_src) = rebuild_mesh(&centers);
    let tri_colors = ocg_tri_colors(ocg, &tri_src);
    geometry::build_flat_mesh_colored(&verts, &tris, Some(&tri_colors))
}

/// Like `build_mesh_from_ocg` but WITHOUT per-vertex colours, so it renders by
/// `StandardMaterial::base_color`. Used for the species-editor preview cell,
/// which must not inherit the snapped cell type's colour.
pub fn build_uncolored_mesh_from_ocg(ocg: &[(usize, Vec3, CellType)]) -> Mesh {
    if ocg.is_empty() {
        return build_flat_mesh(&[], &[]);
    }
    let centers: Vec<Vec3> = ocg.iter().map(|(_, p, _)| *p).collect();
    let (verts, tris, _tri_src) = rebuild_mesh(&centers);
    build_flat_mesh(&verts, &tris)
}

/// Like `build_mesh_from_ocg` but runs the Jacobi vertex smoother first.
/// Called once per organism (at spawn for non-variable-form, or on the
/// growth tick crossing `MAX_CELLS` for variable-form); the mesh then stays
/// smoothed with no per-frame cost.
pub fn build_smoothed_mesh_from_ocg(ocg: &[(usize, Vec3, CellType)]) -> Mesh {
    if ocg.is_empty() {
        return build_flat_mesh(&[], &[]);
    }
    let centers: Vec<Vec3> = ocg.iter().map(|(_, p, _)| *p).collect();
    let (mut verts, tris, tri_src) = rebuild_mesh(&centers);

    // BILATERAL guard: an OCG spanning both sides of the YZ mirror plane
    // (x<0 AND x>0) is a combined body (right half + X-mirror across a dense
    // midline). Plain Laplacian smoothing would collapse the sparse side cells
    // into that heavy midline (halves visually vanish, cells/collider intact).
    // Freeze X while smoothing (Y/Z only) to preserve width. Single-sided
    // parts (limbs, NoSymmetry) smooth on all axes.
    let min_x = ocg.iter().map(|(_, p, _)| p.x).fold(f32::INFINITY, f32::min);
    let max_x = ocg.iter().map(|(_, p, _)| p.x).fold(f32::NEG_INFINITY, f32::max);
    const MIDLINE_EPS: f32 = 0.1;
    let bilateral = min_x < -MIDLINE_EPS && max_x > MIDLINE_EPS;

    let orig_x: Vec<f32> = if bilateral { verts.iter().map(|v| v.x).collect() } else { Vec::new() };
    smooth_vertices::smooth_vertices(
        &mut verts, &tris,
        ADULT_SMOOTH_LAMBDA, ADULT_SMOOTH_ITERATIONS,
    );
    if bilateral {
        // Restore original X; Y/Z stay smoothed.
        for (v, x) in verts.iter_mut().zip(orig_x) { v.x = x; }
    }
    // Smoothing moves positions only; tri→source mapping (colours) unaffected.
    let tri_colors = ocg_tri_colors(ocg, &tri_src);
    geometry::build_flat_mesh_colored(&verts, &tris, Some(&tri_colors))
}

/// Map each surviving triangle to its source cell's linear-RGBA colour
/// (alpha from the cell type — `< 1.0` for translucent types like Gelly).
/// `tri_src[i]` is the cell-index that emitted the i-th surviving triangle.
fn ocg_tri_colors(
    ocg:     &[(usize, Vec3, CellType)],
    tri_src: &[u32],
) -> Vec<[f32; 4]> {
    tri_src.iter().map(|&i| ocg[i as usize].2.color_rgba()).collect()
}

/// Grow one additional cell from the frontier of the OCG's final state.
/// Returns the extended sequence; if the surface is fully enclosed, returns
/// the input unchanged.
pub fn grow_ocg_one_step(
    ocg: &[(usize, Vec3, CellType)],
    rng: &mut impl rand::Rng,
) -> Vec<(usize, Vec3, CellType)> {
    if ocg.is_empty() {
        return Vec::new();
    }
    let mut state = VolumetricState::empty();
    for (_, center, _) in ocg.iter() {
        state.centers.push(*center);
        update_lattice_bookkeeping(&mut state, *center);
    }
    // Dodecahedron candidates read only `frontier_pos`/`occupied`, so the mesh
    // rebuild is wasted work; only Tetrahedron mode needs the cached triangles.
    if matches!(GROWTH_MODE, GrowthMode::Tetrahedron) {
        let (v, t, _) = rebuild_mesh(&state.centers);
        state.vertices = v;
        state.triangles = t;
    }

    let candidates = collect_candidates(&state, true);
    if candidates.is_empty() {
        return ocg.to_vec();
    }
    let pick = rng.random_range(0..candidates.len());
    let center = candidates[pick].center;

    let cell_type = ocg[0].2;
    let new_idx = ocg.len();
    let mut new_ocg = ocg.to_vec();
    new_ocg.push((new_idx, center, cell_type));
    new_ocg
}

/// All valid next-cell centres for the current OCG. `min_x = Some(v)` filters
/// to `centre.x >= v − 1e-3` (bilateral right-half-only path); `None` = no
/// constraint. Exposed for the Species Editor to enumerate the lattice frontier
/// (same machinery as `grow_ocg_one_step`, without the random pick).
pub fn candidate_centers_for_ocg(
    ocg:   &[(usize, Vec3, CellType)],
    min_x: Option<f32>,
) -> Vec<Vec3> {
    if ocg.is_empty() { return Vec::new(); }
    let mut state = VolumetricState::empty();
    for (_, center, _) in ocg.iter() {
        state.centers.push(*center);
        update_lattice_bookkeeping(&mut state, *center);
    }
    if matches!(GROWTH_MODE, GrowthMode::Tetrahedron) {
        let (v, t, _) = rebuild_mesh(&state.centers);
        state.vertices = v;
        state.triangles = t;
    }
    // Editor authoring: `respect_upward = false` for free 3D placement (incl. below root).
    let raw = collect_candidates(&state, false);
    match min_x {
        Some(v) => {
            let threshold = v - 1e-3;
            raw.into_iter().filter(|c| c.center.x >= threshold).map(|c| c.center).collect()
        }
        None => raw.into_iter().map(|c| c.center).collect(),
    }
}

/// Like [`candidate_centers_for_ocg`] but restricted to the **12 face-adjacent**
/// (nearest-FCC) lattice slots, excluding the 6 axis-aligned next-nearest ones.
///
/// The full RD coordination shell has two distance classes (face-adjacent at
/// `≈1.633·edge`, axis-adjacent a `√2` farther at `≈2.309·edge`). Adding ALL of
/// them at once — as the "Wrap" tool does — lays a clean face-shell *plus* the
/// farther axis cells, which read as a spurious second layer. Restricting to the
/// face slots gives a single, uniform one-cell coating. (Dodecahedron mode only;
/// falls back to the full set otherwise.)
pub fn face_adjacent_centers_for_ocg(
    ocg:   &[(usize, Vec3, CellType)],
    min_x: Option<f32>,
) -> Vec<Vec3> {
    if ocg.is_empty() { return Vec::new(); }
    if !matches!(GROWTH_MODE, GrowthMode::Dodecahedron) {
        return candidate_centers_for_ocg(ocg, min_x);
    }
    let scale     = dodecahedron::center_scale(EDGE_LEN);
    let occupied: HashSet<LatticeKey> = ocg.iter().map(|(_, c, _)| lattice_key(*c)).collect();
    let threshold = min_x.map(|v| v - 1e-3);
    let mut seen: HashSet<LatticeKey> = HashSet::new();
    let mut out  = Vec::new();
    // SLOT_DIRS[6..] are the 12 face-diagonal (nearest-FCC) directions.
    for (_, center, _) in ocg.iter() {
        for &dir in &dodecahedron::SLOT_DIRS[6..] {
            let nb = *center + dir * scale;
            if let Some(t) = threshold { if nb.x < t { continue; } }
            let key = lattice_key(nb);
            if occupied.contains(&key) { continue; }
            if seen.insert(key) { out.push(nb); }
        }
    }
    out
}

/// `grow_ocg_one_step` restricted to candidates with `x >= min_x` (slack -1e-3
/// for float drift; see body_part::MIN_X_BILATERAL). Keeps the bilateral right
/// half on the +X side of the YZ mirror plane; returns parent OCG if none fit.
pub fn grow_ocg_one_step_constrained(
    ocg:   &[(usize, Vec3, CellType)],
    rng:   &mut impl rand::Rng,
    min_x: f32,
) -> Vec<(usize, Vec3, CellType)> {
    if ocg.is_empty() {
        return Vec::new();
    }
    let mut state = VolumetricState::empty();
    for (_, center, _) in ocg.iter() {
        state.centers.push(*center);
        update_lattice_bookkeeping(&mut state, *center);
    }
    // Dodec-mode rebuild_mesh skip (see grow_ocg_one_step).
    if matches!(GROWTH_MODE, GrowthMode::Tetrahedron) {
        let (v, t, _) = rebuild_mesh(&state.centers);
        state.vertices = v;
        state.triangles = t;
    }

    let threshold = min_x - 1e-3;
    let candidates: Vec<_> = collect_candidates(&state, true)
        .into_iter()
        .filter(|c| c.center.x >= threshold)
        .collect();
    if candidates.is_empty() {
        return ocg.to_vec();
    }
    let pick = rng.random_range(0..candidates.len());
    let center = candidates[pick].center;

    let cell_type = ocg[0].2;
    let new_idx = ocg.len();
    let mut new_ocg = ocg.to_vec();
    new_ocg.push((new_idx, center, cell_type));
    new_ocg
}

// ── Plugin (sandbox) ──────────────────────────────────────────────────────────

pub struct VolumetricGrowthPlugin;

impl Plugin for VolumetricGrowthPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(VolumetricState::initial())
            .insert_resource(GrowthController::default())
            .insert_resource(GrowthTimer(Timer::from_seconds(1.0, TimerMode::Repeating)))
            .add_systems(Startup, spawn_volumetric_mesh)
            .add_systems(Update, grow_one_step);
    }
}

#[derive(Resource)]
struct GrowthTimer(Timer);

#[derive(Component)]
struct VolumetricMesh;

#[derive(Resource)]
struct VolumetricState {
    /// Cache: rebuilt from `centers` whenever `mesh_dirty` is set.
    vertices: Vec<Vec3>,
    triangles: Vec<[u32; 3]>,
    /// Source of truth: every placed cell centre, in placement order.
    /// `centers[0]` is the seed (Vec3::ZERO in the sandbox).
    centers: Vec<Vec3>,

    // Lattice bookkeeping for Dodecahedron candidate generation.
    occupied: HashSet<LatticeKey>,
    frontier: HashSet<LatticeKey>,
    frontier_pos: HashMap<LatticeKey, Vec3>,

    // Tetrahedron mode: spatial hash of placed centres for proximity checks.
    tetra_occupied: HashMap<LatticeKey, Vec3>,

    mesh_dirty: bool,
    rng: u64,
    done: bool,
}

impl VolumetricState {
    fn empty() -> Self {
        Self {
            vertices: Vec::new(),
            triangles: Vec::new(),
            centers: Vec::new(),
            occupied: HashSet::new(),
            frontier: HashSet::new(),
            frontier_pos: HashMap::new(),
            tetra_occupied: HashMap::new(),
            mesh_dirty: true,
            rng: 0xDEAD_BEEF_CAFE_F00D,
            done: false,
        }
    }

    fn initial() -> Self {
        let mut s = Self::empty();
        s.centers.push(Vec3::ZERO);
        update_lattice_bookkeeping(&mut s, Vec3::ZERO);
        let (v, t, _) = rebuild_mesh(&s.centers);
        s.vertices = v;
        s.triangles = t;
        s.mesh_dirty = false;
        s
    }
}

/// Update lattice / spatial bookkeeping after `center` joins `centers`.
fn update_lattice_bookkeeping(state: &mut VolumetricState, center: Vec3) {
    match GROWTH_MODE {
        GrowthMode::Dodecahedron => {
            let scale = dodecahedron::center_scale(EDGE_LEN);
            let placed_key = lattice_key(center);
            state.frontier.remove(&placed_key);
            state.frontier_pos.remove(&placed_key);
            state.occupied.insert(placed_key);
            for &dir in &dodecahedron::SLOT_DIRS {
                let neighbor = center + dir * scale;
                let key = lattice_key(neighbor);
                if !state.occupied.contains(&key) && state.frontier.insert(key) {
                    state.frontier_pos.insert(key, neighbor);
                }
            }
        }
        GrowthMode::Tetrahedron => {
            state.tetra_occupied.insert(tetra_spatial_key(center), center);
        }
    }
}

// ── Face geometry helpers ─────────────────────────────────────────────────────

#[inline]
fn face_centroid(tri: &[u32; 3], verts: &[Vec3]) -> Vec3 {
    let [a, b, c] = *tri;
    (verts[a as usize] + verts[b as usize] + verts[c as usize]) / 3.0
}

#[inline]
fn face_normal(tri: &[u32; 3], verts: &[Vec3]) -> Vec3 {
    let [a, b, c] = *tri;
    let av = verts[a as usize];
    let bv = verts[b as usize];
    let cv = verts[c as usize];
    (bv - av).cross(cv - av).normalize_or_zero()
}

// ── Candidate generation ──────────────────────────────────────────────────────

/// Legal next-cell positions for `state`. `respect_upward` honours
/// `GROW_ONLY_UPWARDS`: `true` for procedural growth (keeps plants skyward,
/// not burrowing); `false` for the Species Editor's free 3D placement.
fn collect_candidates(state: &VolumetricState, respect_upward: bool) -> Vec<CandidateInfo> {
    match GROWTH_MODE {
        GrowthMode::Dodecahedron => collect_candidates_dodec(state, respect_upward),
        GrowthMode::Tetrahedron  => collect_candidates_tetra(state, respect_upward),
    }
}

fn collect_candidates_dodec(state: &VolumetricState, respect_upward: bool) -> Vec<CandidateInfo> {
    let scale = dodecahedron::center_scale(EDGE_LEN);
    state
        .frontier_pos
        .values()
        .filter_map(|&center| {
            // Any occupied lattice neighbour gives the attach direction.
            let parent_center = dodecahedron::SLOT_DIRS
                .iter()
                .map(|&dir| center - dir * scale)
                .find(|p| state.occupied.contains(&lattice_key(*p)))?;
            if respect_upward
                && growth_controls::GROW_ONLY_UPWARDS
                && center.y < parent_center.y
            {
                return None;
            }
            Some(CandidateInfo {
                face_idx: 0,
                center,
                face_normal: (center - parent_center).normalize_or_zero(),
                face_centroid: (center + parent_center) * 0.5,
            })
        })
        .collect()
}

fn collect_candidates_tetra(state: &VolumetricState, respect_upward: bool) -> Vec<CandidateInfo> {
    if state.vertices.is_empty() || state.triangles.is_empty() {
        return Vec::new();
    }
    state
        .triangles
        .iter()
        .enumerate()
        .filter_map(|(face_idx, tri)| {
            let centroid = face_centroid(tri, &state.vertices);
            let normal = face_normal(tri, &state.vertices);
            let center = centroid + normal;
            if tetra_is_blocked(center, &state.tetra_occupied) {
                return None;
            }
            if respect_upward
                && growth_controls::GROW_ONLY_UPWARDS
                && center.y < centroid.y
            {
                return None;
            }
            Some(CandidateInfo {
                face_idx,
                center,
                face_normal: normal,
                face_centroid: centroid,
            })
        })
        .collect()
}

// ── Systems ───────────────────────────────────────────────────────────────────

fn spawn_volumetric_mesh(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    state: Res<VolumetricState>,
) {
    commands.spawn((
        Mesh3d(meshes.add(build_flat_mesh(&state.vertices, &state.triangles))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.2, 0.5, 0.9),
            cull_mode: None,
            double_sided: true,
            perceptual_roughness: 0.5,
            ..default()
        })),
        VolumetricMesh,
        Organism {
            body_parts: vec![crate::colony::root_body_part_from_ocg(
                &[(0, Vec3::ZERO, CellType::Photo)],
            )],
            symmetry:             crate::organism::Symmetry::NoSymmetry,
            intelligence_level:   crate::organism::IntelligenceLevel::Level1,
            is_sessile:           false,
            has_variable_form:    false,
            movement_mode:        crate::organism::MovementMode::Sliding,
            ground_based:         true,
        limb_targets:         [0.0; 10],
            adult:                false,
            photo_cell_count:     1,
            non_photo_cell_count: 0,
            energy: 0.0,
            in_sunlight: false,
            reproduced: false,
            reproductions: 0,
            predations: 0,
            hunger: 0.0,
            dopamine: 0.0,
            target_distance: crate::sensory::SENSORY_RADIUS,
            movement_speed: 0.0,
            movement_direction: Vec3::ZERO,
            velocity: Vec3::ZERO,
            is_climbing: false,
            climb_energy_debt: 0.0,
            cached_bounding_radius: 0.0,
            // Dev sandbox spawn — structurally correct DNA, though this entity
            // is never seen by speciation (sandbox plugin isn't wired into the sim).
            dna: crate::lineages::dna::structural_dna(
                crate::organism::OrganismKind::Photoautotroph,
                crate::organism::Symmetry::NoSymmetry,
                false, false,
                crate::organism::IntelligenceLevel::Level1,
            ),
            species_id: None,
        },
    ));
}

fn grow_one_step(
    time: Res<Time<Virtual>>,
    mut timer: ResMut<GrowthTimer>,
    mut state: ResMut<VolumetricState>,
    mut controller: ResMut<GrowthController>,
    mut meshes: ResMut<Assets<Mesh>>,
    mesh_query: Query<&Mesh3d, With<VolumetricMesh>>,
    mut organism_query: Query<&mut Organism, With<VolumetricMesh>>,
) {
    timer.0.tick(time.delta());
    if !timer.0.just_finished() || state.done {
        return;
    }

    let candidates = collect_candidates(&state, true);
    if candidates.is_empty() {
        state.done = true;
        info!(
            "volumetric_growth: no space left — {} cells placed",
            state.centers.len() - 1
        );
        return;
    }

    let Some(pick) = controller.strategy.select(&candidates, &mut state.rng) else {
        return;
    };
    let pick = pick.min(candidates.len() - 1);
    let center = candidates[pick].center;

    state.centers.push(center);
    update_lattice_bookkeeping(&mut state, center);
    state.mesh_dirty = true;

    if let Ok(mut organism) = organism_query.single_mut() {
        let relative_pos = center - state.centers[0];
        if let Some(root) = organism.body_parts.first_mut() {
            let ocg_idx = root.ocg.len();
            root.ocg.push((ocg_idx, relative_pos, CellType::Photo));
            info!(
                "organism.body_parts[0].ocg[{}] = ({:.3}, {:.3}, {:.3})",
                ocg_idx, relative_pos.x, relative_pos.y, relative_pos.z
            );
        }
    }

    let cells_placed = state.centers.len() - 1;
    if cells_placed >= MAX_CELLS {
        state.done = true;
    }

    if state.mesh_dirty {
        let (v, t, _) = rebuild_mesh(&state.centers);
        state.vertices = v;
        state.triangles = t;
        state.mesh_dirty = false;
        if let Ok(mesh3d) = mesh_query.single() {
            if let Some(m) = meshes.get_mut(&mesh3d.0) {
                *m = build_flat_mesh(&state.vertices, &state.triangles);
            }
        }
    }

    info!(
        "volumetric_growth: cell {}  verts={}  tris={}",
        state.centers.len() - 1,
        state.vertices.len(),
        state.triangles.len()
    );
}
