// Volumetric growth — translation-only RD/tetra cell attachment.
//
// Algorithm (Blender-inspired "remove doubles → drop interior faces" pipeline):
//
//   1. Each placed cell's centre lives in `centers`.
//   2. To get geometry, every cell emits its full 14 verts + 24 triangles in
//      canonical (axis-aligned, untranslated) form, summed with the cell centre.
//   3. Coincident vertices across cells are welded by a HashMap on quantised
//      positions (eps = 1e-4 — six orders of magnitude below RD's 1.0 minimum
//      vertex spacing).
//   4. After welding, a triangle shared by two adjacent cells appears with
//      identical (welded) vertex indices in both cells. Sorted-key duplicates
//      are interior faces and are dropped on both sides.
//   5. Surviving triangles inherit outward winding from their unique source
//      cell — no global recalc is required.
//
// The whole mesh is recomputed from `centers` each tick; with ≤ 30 cells × 14
// verts = 420 verts this is sub-millisecond. There is no surgical merge, no
// boundary-loop extraction, no zipper stitch, no fill-holes post-pass.

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
const EDGE_LEN: f32 = 1.0;
const MIN_CENTER_DIST: f32 = 0.5;
/// Sandbox and OCG growth stops after this many cells are appended to the seed.
pub const MAX_CELLS: usize = 60;

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
fn rebuild_mesh(centers: &[Vec3]) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    if centers.is_empty() {
        return (Vec::new(), Vec::new());
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

    for &c in centers {
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
        for &[a, b, c] in &local_tris {
            tris.push([
                local_to_global[a as usize],
                local_to_global[b as usize],
                local_to_global[c as usize],
            ]);
        }
    }

    // Drop interior triangles. Boundary triangles have sorted-key multiplicity
    // 1; shared faces (multiplicity 2) cancel. RD lattice geometry guarantees
    // no triangle is shared by 3+ cells, so multiplicity is always 1 or 2.
    let mut bucket: HashMap<[u32; 3], u32> = HashMap::new();
    for &t in &tris {
        let mut k = t;
        k.sort_unstable();
        *bucket.entry(k).or_insert(0) += 1;
    }
    tris.retain(|&t| {
        if t[0] == t[1] || t[1] == t[2] || t[0] == t[2] {
            return false;
        }
        let mut k = t;
        k.sort_unstable();
        bucket[&k] == 1
    });

    (verts, tris)
}

// ── OCG pathway ───────────────────────────────────────────────────────────────

/// Smoothing parameters used when an organism becomes `adult`. `lambda`
/// is the per-iteration blend toward the one-ring centroid (0 = identity,
/// 1 = collapse to centroid); `iterations` controls how many passes the
/// Jacobi smoother runs. The values here give a soft "rounded" silhouette
/// over a 30-cell rhombic-dodecahedron blob without erasing the lobes
/// produced by branch attachments.
pub const ADULT_SMOOTH_LAMBDA:     f32   = 0.3;
pub const ADULT_SMOOTH_ITERATIONS: usize = 3;

/// Produce the (un-smoothed) flat mesh by treating each OCG entry's
/// position as a cell centre and running the translate → weld → dedup
/// pipeline. Used during growth, when each new cell adds a faceted
/// rhombic-dodecahedron lobe.
pub fn build_mesh_from_ocg(ocg: &[(usize, Vec3, CellType)]) -> Mesh {
    if ocg.is_empty() {
        return build_flat_mesh(&[], &[]);
    }
    let centers: Vec<Vec3> = ocg.iter().map(|(_, p, _)| *p).collect();
    let (verts, tris) = rebuild_mesh(&centers);
    build_flat_mesh(&verts, &tris)
}

/// Same as `build_mesh_from_ocg` but additionally runs the Jacobi
/// vertex smoother (`smooth_vertices`) before constructing the Bevy
/// mesh. Called once per organism — at spawn for non-variable-form
/// organisms (they don't grow during their lifetime) and on the
/// continuous-growth tick that crosses `MAX_CELLS` for variable-form
/// organisms. After that the mesh stays smoothed for the rest of the
/// organism's life; no per-frame smoothing cost.
pub fn build_smoothed_mesh_from_ocg(ocg: &[(usize, Vec3, CellType)]) -> Mesh {
    if ocg.is_empty() {
        return build_flat_mesh(&[], &[]);
    }
    let centers: Vec<Vec3> = ocg.iter().map(|(_, p, _)| *p).collect();
    let (mut verts, tris) = rebuild_mesh(&centers);
    smooth_vertices::smooth_vertices(
        &mut verts, &tris,
        ADULT_SMOOTH_LAMBDA, ADULT_SMOOTH_ITERATIONS,
    );
    build_flat_mesh(&verts, &tris)
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
    // Dodecahedron candidate generation reads only `frontier_pos` and
    // `occupied` (both populated by `update_lattice_bookkeeping`), so the
    // full mesh rebuild is wasted work. Tetrahedron mode (currently
    // unused) still needs the cached `triangles` slab — keep the rebuild
    // there.
    if matches!(GROWTH_MODE, GrowthMode::Tetrahedron) {
        let (v, t) = rebuild_mesh(&state.centers);
        state.vertices = v;
        state.triangles = t;
    }

    let candidates = collect_candidates(&state);
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

/// Return ALL valid next-cell centre positions given the current OCG.
/// `min_x = Some(v)` filters to candidates with `centre.x >= v − 1e-3`
/// (used by the bilateral right-half-only path). `min_x = None` means
/// no constraint.
///
/// Exposed for interactive tools (the Species Editor) that need to
/// enumerate the lattice frontier instead of picking randomly.
/// Internally this is the same machinery `grow_ocg_one_step` uses;
/// just without the final random `pick`.
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
        let (v, t) = rebuild_mesh(&state.centers);
        state.vertices = v;
        state.triangles = t;
    }
    let raw = collect_candidates(&state);
    match min_x {
        Some(v) => {
            let threshold = v - 1e-3;
            raw.into_iter().filter(|c| c.center.x >= threshold).map(|c| c.center).collect()
        }
        None => raw.into_iter().map(|c| c.center).collect(),
    }
}

/// As `grow_ocg_one_step`, but only candidate cells with `x >= min_x` are
/// considered. Returns the parent OCG unchanged when no candidate satisfies
/// the constraint. Used by the bilateral pipeline to keep the right half
/// strictly on the +X side of the YZ mirror plane.
///
/// `min_x` is checked against the candidate's centre with a tiny float-drift
/// slack (`-1e-3`) — see body_part::MIN_X_BILATERAL for rationale.
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
    // Same Dodec-mode rebuild_mesh skip as in grow_ocg_one_step.
    if matches!(GROWTH_MODE, GrowthMode::Tetrahedron) {
        let (v, t) = rebuild_mesh(&state.centers);
        state.vertices = v;
        state.triangles = t;
    }

    let threshold = min_x - 1e-3;
    let candidates: Vec<_> = collect_candidates(&state)
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
        let (v, t) = rebuild_mesh(&s.centers);
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

fn collect_candidates(state: &VolumetricState) -> Vec<CandidateInfo> {
    match GROWTH_MODE {
        GrowthMode::Dodecahedron => collect_candidates_dodec(state),
        GrowthMode::Tetrahedron => collect_candidates_tetra(state),
    }
}

fn collect_candidates_dodec(state: &VolumetricState) -> Vec<CandidateInfo> {
    let scale = dodecahedron::center_scale(EDGE_LEN);
    state
        .frontier_pos
        .values()
        .filter_map(|&center| {
            // Pick any occupied lattice neighbour to derive the attach direction.
            let parent_center = dodecahedron::SLOT_DIRS
                .iter()
                .map(|&dir| center - dir * scale)
                .find(|p| state.occupied.contains(&lattice_key(*p)))?;
            if growth_controls::GROW_ONLY_UPWARDS && center.y < parent_center.y {
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

fn collect_candidates_tetra(state: &VolumetricState) -> Vec<CandidateInfo> {
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
            if growth_controls::GROW_ONLY_UPWARDS && center.y < centroid.y {
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
            // Dev sandbox spawn — give it a structurally correct DNA
            // vector even though this entity is never seen by the
            // speciation system in practice (the sandbox plugin is
            // not wired into the regular simulation app).
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

    let candidates = collect_candidates(&state);
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
        let (v, t) = rebuild_mesh(&state.centers);
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
