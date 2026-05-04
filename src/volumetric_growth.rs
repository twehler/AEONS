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
/// Checks the 27-cell neighbourhood of `candidate` in the spatial hash — O(1).
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
mod smooth_vertices;

use geometry::{build_flat_mesh, outward_tri, quads_to_tris};
use growth_controls::{CandidateInfo, GrowthController};
use smooth_vertices::smooth_vertices;

// ── Mode ──────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum GrowthMode {
    Tetrahedron,
    Dodecahedron,
}

// Cells based on Tetrahedron or Dodecahedron
pub const GROWTH_MODE: GrowthMode = GrowthMode::Dodecahedron;
const EDGE_LEN: f32 = 1.0;
const MIN_CENTER_DIST: f32 = 0.5;
const SMOOTHEN_AFTER: usize = 20;
const SMOOTH_LAMBDA: f32 = 0.3;
const SMOOTH_ITERATIONS: usize = 3;

// ── OCG replay ───────────────────────────────────────────────────────────────

/// Replay the OCG growth sequence into a `VolumetricState`. The seed cell
/// (ocg[0]) maps to the initial dodecahedron state; subsequent entries are
/// attached to the closest open face in order. The returned state is ready
/// for candidate collection (next growth step) or mesh extraction.
fn state_from_ocg(ocg: &[(usize, Vec3, CellType)]) -> VolumetricState {
    let mut state = VolumetricState::initial();
    for (_, target_center, _) in ocg.iter().skip(1) {
        let best = state.open_faces.iter().enumerate()
            .min_by(|(_, a), (_, b)| {
                let ca = face_centroid(&a.tri, &state.vertices);
                let cb = face_centroid(&b.tri, &state.vertices);
                (ca - *target_center).length_squared()
                    .partial_cmp(&(cb - *target_center).length_squared())
                    .unwrap()
            })
            .map(|(i, _)| i);
        if let Some(face_idx) = best {
            attach_cell(&mut state, face_idx, *target_center);
        }
    }
    state
}

/// Produce the adult mesh by replaying the OCG and extracting the surface.
pub fn build_mesh_from_ocg(ocg: &[(usize, Vec3, CellType)]) -> Mesh {
    if ocg.is_empty() {
        return geometry::build_flat_mesh(&[], &[]);
    }
    let state = state_from_ocg(ocg);
    geometry::build_flat_mesh(&state.vertices, &state.triangles)
}

/// Grow one additional cell from the frontier of the OCG's final state,
/// returning the extended sequence. The new cell is chosen randomly among
/// all valid candidates; it inherits the trophic cell type of the seed
/// (ocg[0].2). If no candidates exist (fully enclosed mesh), the input OCG
/// is returned unchanged — callers must handle the no-growth case.
pub fn grow_ocg_one_step(
    ocg: &[(usize, Vec3, CellType)],
    rng: &mut impl rand::Rng,
) -> Vec<(usize, Vec3, CellType)> {
    if ocg.is_empty() {
        return Vec::new();
    }
    let mut state = state_from_ocg(ocg);
    let candidates = collect_candidates(&state);
    if candidates.is_empty() {
        return ocg.to_vec();
    }
    let pick     = rng.random_range(0..candidates.len());
    let face_idx = candidates[pick].face_idx;
    let center   = candidates[pick].center;
    attach_cell(&mut state, face_idx, center);

    let cell_type = ocg[0].2; // inherit trophic type from seed
    let new_idx   = ocg.len();
    let mut new_ocg = ocg.to_vec();
    new_ocg.push((new_idx, center, cell_type));
    new_ocg
}


// ── Plugin ────────────────────────────────────────────────────────────────────

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

// ── Resources / components ────────────────────────────────────────────────────

#[derive(Resource)]
struct GrowthTimer(Timer);

#[derive(Component)]
struct VolumetricMesh;

#[derive(Clone)]
struct OpenFace {
    tri: [u32; 3],
    cell_key: LatticeKey,
}

#[derive(Resource)]
struct VolumetricState {
    vertices: Vec<Vec3>,
    triangles: Vec<[u32; 3]>,
    open_faces: Vec<OpenFace>,
    /// Cell LatticeKey → indices of that cell's open faces.
    cell_faces: HashMap<LatticeKey, Vec<usize>>,
    /// Open face triangle → its current index in open_faces (O(1) removal).
    tri_to_face_idx: HashMap<[u32; 3], usize>,
    centers: Vec<Vec3>,

    // ── Dodecahedron frontier ─────────────────────────────────────────────
    occupied: HashSet<LatticeKey>,
    frontier: HashSet<LatticeKey>,
    frontier_pos: HashMap<LatticeKey, Vec3>,

    // ── Tetrahedron frontier ──────────────────────────────────────────────
    /// Tetrahedron mode only: face_tri → pre-computed candidate center for
    /// every open face that is not blocked by an existing placed center.
    /// Avoids the O(|centers|) scan per face in collect_candidates_tetra.
    tetra_frontier: HashMap<[u32; 3], Vec3>,
    /// Tetrahedron mode only: spatial hash (cell size = MIN_CENTER_DIST) of
    /// all placed cell centers for O(1) proximity checks.
    tetra_occupied: HashMap<LatticeKey, Vec3>,

    // ── Shared ────────────────────────────────────────────────────────────
    centroid_sum: Vec3,
    mesh_dirty: bool,
    rng: u64,
    done: bool,
}

impl VolumetricState {
    fn initial() -> Self {
        let verts = dodecahedron::seed_vertices(EDGE_LEN);
        let triangles = quads_to_tris(&dodecahedron::seed_quads());
        let seed_key = lattice_key(Vec3::ZERO);

        let mut open_faces: Vec<OpenFace> = Vec::with_capacity(triangles.len());
        let mut cell_faces: HashMap<LatticeKey, Vec<usize>> = HashMap::new();
        let mut tri_to_face_idx: HashMap<[u32; 3], usize> = HashMap::new();

        for &tri in &triangles {
            let idx = open_faces.len();
            open_faces.push(OpenFace { tri, cell_key: seed_key });
            cell_faces.entry(seed_key).or_default().push(idx);
            tri_to_face_idx.insert(tri, idx);
        }

        // ── Dodecahedron frontier ─────────────────────────────────────────
        let scale = dodecahedron::center_scale(EDGE_LEN);
        let mut occupied = HashSet::new();
        occupied.insert(seed_key);
        let mut frontier = HashSet::new();
        let mut frontier_pos = HashMap::new();
        for &dir in &dodecahedron::SLOT_DIRS {
            let pos = dir * scale;
            let key = lattice_key(pos);
            if frontier.insert(key) {
                frontier_pos.insert(key, pos);
            }
        }

        // ── Tetrahedron frontier ──────────────────────────────────────────
        let mut tetra_occupied: HashMap<LatticeKey, Vec3> = HashMap::new();
        let mut tetra_frontier: HashMap<[u32; 3], Vec3> = HashMap::new();
        if GROWTH_MODE == GrowthMode::Tetrahedron {
            tetra_occupied.insert(tetra_spatial_key(Vec3::ZERO), Vec3::ZERO);
            for &tri in &triangles {
                let candidate =
                    face_centroid(&tri, &verts) + face_normal_outward(&tri, &verts);
                if !tetra_is_blocked(candidate, &tetra_occupied) {
                    tetra_frontier.insert(tri, candidate);
                }
            }
        }

        let centroid_sum: Vec3 = verts.iter().copied().sum();

        Self {
            vertices: verts,
            triangles,
            open_faces,
            cell_faces,
            tri_to_face_idx,
            centers: vec![Vec3::ZERO],
            occupied,
            frontier,
            frontier_pos,
            tetra_frontier,
            tetra_occupied,
            centroid_sum,
            mesh_dirty: false,
            rng: 0xDEAD_BEEF_CAFE_F00D,
            done: false,
        }
    }

    #[allow(dead_code)]
    fn next_rand(&mut self) -> u64 {
        self.rng = self
            .rng
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.rng
    }

    fn add_open_face(&mut self, tri: [u32; 3], cell_key: LatticeKey) {
        let idx = self.open_faces.len();
        self.open_faces.push(OpenFace { tri, cell_key });
        self.cell_faces.entry(cell_key).or_default().push(idx);
        self.tri_to_face_idx.insert(tri, idx);
        if GROWTH_MODE == GrowthMode::Tetrahedron {
            let candidate =
                face_centroid(&tri, &self.vertices) + face_normal_outward(&tri, &self.vertices);
            if !tetra_is_blocked(candidate, &self.tetra_occupied) {
                self.tetra_frontier.insert(tri, candidate);
            }
        }
    }

    fn remove_open_face(&mut self, idx: usize) {
        let last_idx = self.open_faces.len() - 1;
        let removed = self.open_faces[idx].clone();

        self.tri_to_face_idx.remove(&removed.tri);
        if let Some(v) = self.cell_faces.get_mut(&removed.cell_key) {
            v.retain(|&i| i != idx);
        }
        if GROWTH_MODE == GrowthMode::Tetrahedron {
            self.tetra_frontier.remove(&removed.tri);
        }

        if idx != last_idx {
            let moved = self.open_faces[last_idx].clone();
            if let Some(slot) = self.tri_to_face_idx.get_mut(&moved.tri) {
                *slot = idx;
            }
            if let Some(v) = self.cell_faces.get_mut(&moved.cell_key) {
                for slot in v.iter_mut() {
                    if *slot == last_idx {
                        *slot = idx;
                        break;
                    }
                }
            }
            // tetra_frontier is keyed by triangle, not index — no update needed for the moved face.
        }

        self.open_faces.swap_remove(idx);
    }
}

// ── On-demand face geometry ───────────────────────────────────────────────────

#[inline]
fn face_centroid(tri: &[u32; 3], verts: &[Vec3]) -> Vec3 {
    let [a, b, c] = *tri;
    (verts[a as usize] + verts[b as usize] + verts[c as usize]) / 3.0
}

#[inline]
fn face_normal_outward(tri: &[u32; 3], verts: &[Vec3]) -> Vec3 {
    let [a, b, c] = *tri;
    let av = verts[a as usize];
    let bv = verts[b as usize];
    let cv = verts[c as usize];
    let raw = (bv - av).cross(cv - av);
    let centroid = (av + bv + cv) / 3.0;
    if raw.dot(centroid) >= 0.0 { raw } else { -raw }.normalize_or_zero()
}

// ── Candidate collection ──────────────────────────────────────────────────────

fn collect_candidates(state: &VolumetricState) -> Vec<CandidateInfo> {
    match GROWTH_MODE {
        GrowthMode::Tetrahedron => collect_candidates_tetra(state),
        GrowthMode::Dodecahedron => collect_candidates_dodec(state),
    }
}

fn collect_candidates_tetra(state: &VolumetricState) -> Vec<CandidateInfo> {
    // tetra_frontier already excludes blocked faces — no O(|centers|) check needed.
    state
        .tetra_frontier
        .iter()
        .filter_map(|(&tri, &center)| {
            let &face_idx = state.tri_to_face_idx.get(&tri)?;
            if growth_controls::GROW_ONLY_UPWARDS {
                let parent_y = state.open_faces[face_idx].cell_key.1 as f32 / 2048.0;
                if center.y < parent_y {
                    return None;
                }
            }
            Some(CandidateInfo {
                face_idx,
                center,
                face_normal: face_normal_outward(&tri, &state.vertices),
                face_centroid: face_centroid(&tri, &state.vertices),
            })
        })
        .collect()
}

fn collect_candidates_dodec(state: &VolumetricState) -> Vec<CandidateInfo> {
    let scale = dodecahedron::center_scale(EDGE_LEN);
    state
        .frontier_pos
        .values()
        .filter_map(|&center| {
            // Only check open faces of the (at most 18) occupied lattice neighbours.
            let best_idx = dodecahedron::SLOT_DIRS
                .iter()
                .filter_map(|&dir| state.cell_faces.get(&lattice_key(center - dir * scale)))
                .flatten()
                .copied()
                .min_by(|&a, &b| {
                    let ca = face_centroid(&state.open_faces[a].tri, &state.vertices);
                    let cb = face_centroid(&state.open_faces[b].tri, &state.vertices);
                    (ca - center).length_squared()
                        .partial_cmp(&(cb - center).length_squared())
                        .unwrap()
                });
            best_idx.and_then(|face_idx| {
                if growth_controls::GROW_ONLY_UPWARDS {
                    let parent_y = state.open_faces[face_idx].cell_key.1 as f32 / 2048.0;
                    if center.y < parent_y {
                        return None;
                    }
                }
                let tri = &state.open_faces[face_idx].tri;
                Some(CandidateInfo {
                    face_idx,
                    center,
                    face_normal: face_normal_outward(tri, &state.vertices),
                    face_centroid: face_centroid(tri, &state.vertices),
                })
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
            body_parts: vec![],
            ocg: vec![(0, Vec3::ZERO, CellType::Photo)],
            energy: 0.0,
            in_sunlight: false,
            reproduced: false,
            reproductions: 0,
            movement_speed: 0.0,
            movement_direction: Vec3::ZERO,
            velocity: Vec3::ZERO,
            is_climbing: false,
            climb_energy_debt: 0.0,
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
            "volumetric_growth: no space left — {} cells placed ({} mode)",
            state.centers.len() - 1,
            match GROWTH_MODE {
                GrowthMode::Tetrahedron => "Tetrahedron",
                GrowthMode::Dodecahedron => "Dodecahedron",
            }
        );
        return;
    }

    let Some(pick) = controller.strategy.select(&candidates, &mut state.rng) else {
        return;
    };
    let pick = pick.min(candidates.len() - 1);
    let face_idx = candidates[pick].face_idx;
    let center = candidates[pick].center;
    attach_cell(&mut state, face_idx, center);

    if let Ok(mut organism) = organism_query.single_mut() {
        let relative_pos = center - state.centers[0];
        let ocg_idx = organism.ocg.len();
        organism.ocg.push((ocg_idx, relative_pos, CellType::Photo));
        info!("organism.ocg[{}] = ({:.3}, {:.3}, {:.3})", ocg_idx, relative_pos.x, relative_pos.y, relative_pos.z);
    }

    let cells_placed = state.centers.len() - 1;
    if cells_placed % SMOOTHEN_AFTER == 0 {
        // Split borrow: no triangles.clone() needed.
        let s = &mut *state;
        smooth_vertices(&mut s.vertices, &s.triangles, SMOOTH_LAMBDA, SMOOTH_ITERATIONS);
        s.centroid_sum = s.vertices.iter().copied().sum();
        // Tetrahedron candidate centers are derived from vertex positions, so
        // tetra_frontier must be rebuilt from scratch after smoothing.
        if GROWTH_MODE == GrowthMode::Tetrahedron {
            s.tetra_frontier.clear();
            for of in s.open_faces.iter() {
                let c = face_centroid(&of.tri, &s.vertices)
                    + face_normal_outward(&of.tri, &s.vertices);
                if !tetra_is_blocked(c, &s.tetra_occupied) {
                    s.tetra_frontier.insert(of.tri, c);
                }
            }
        }
    }

    if state.mesh_dirty {
        if let Ok(mesh3d) = mesh_query.single() {
            if let Some(m) = meshes.get_mut(&mesh3d.0) {
                *m = build_flat_mesh(&state.vertices, &state.triangles);
                state.mesh_dirty = false;
            }
        }
    }

    info!(
        "volumetric_growth: cell {}  open_faces={}  tris={}",
        state.centers.len() - 1,
        state.open_faces.len(),
        state.triangles.len()
    );
}

// ── Core attachment logic ─────────────────────────────────────────────────────

fn attach_cell(state: &mut VolumetricState, face_idx: usize, center: Vec3) {
    let of_tri = state.open_faces[face_idx].tri;
    let of_centroid = face_centroid(&of_tri, &state.vertices);
    let attach_normal = (center - of_centroid).normalize_or_zero();

    let new_verts: Vec<Vec3> = match GROWTH_MODE {
        GrowthMode::Tetrahedron => {
            let h = EDGE_LEN * (2.0_f32 / 3.0).sqrt();
            let base_center = center - (h / 4.0) * attach_normal;
            tetrahedron::cell_vertices(base_center, attach_normal, EDGE_LEN).to_vec()
        }
        GrowthMode::Dodecahedron => dodecahedron::cell_vertices(center, attach_normal, EDGE_LEN),
    };

    let global_centroid = state.centroid_sum / state.vertices.len() as f32;
    let t_base = state.vertices.len() as u32;

    let (ni, mi) = (0..new_verts.len())
        .flat_map(|ni| (0..state.vertices.len()).map(move |mi| (ni, mi)))
        .min_by(|&(na, ma), &(nb, mb)| {
            (new_verts[na] - state.vertices[ma]).length_squared()
                .partial_cmp(&(new_verts[nb] - state.vertices[mb]).length_squared())
                .unwrap()
        })
        .unwrap();

    let sphere_center = (new_verts[ni] + state.vertices[mi]) * 0.5;
    let r = growth_controls::CONNECTION_RADIUS;
    let r_sq = r * r;

    let inner_cell: HashSet<u32> = (0..new_verts.len() as u32)
        .filter(|&i| (new_verts[i as usize] - sphere_center).length_squared() < r_sq)
        .collect();

    let inner_mesh: HashSet<u32> = (0..t_base)
        .filter(|&i| (state.vertices[i as usize] - sphere_center).length_squared() < r_sq)
        .collect();

    state.vertices.extend(new_verts.iter().copied());
    state.centroid_sum += new_verts.iter().copied().sum::<Vec3>();

    let all_cell_tris_global: Vec<[u32; 3]> = {
        let rel: Vec<[u32; 3]> = match GROWTH_MODE {
            GrowthMode::Tetrahedron => {
                let mut t: Vec<[u32; 3]> = vec![tetrahedron::BASE_TRI];
                t.extend_from_slice(&tetrahedron::non_base_tris());
                t
            }
            GrowthMode::Dodecahedron => dodecahedron::cell_all_tris(),
        };
        rel.into_iter().map(|[a, b, c]| [a + t_base, b + t_base, c + t_base]).collect()
    };

    let (cell_inner_tris, cell_outer_tris_raw): (Vec<_>, Vec<_>) =
        all_cell_tris_global.into_iter().partition(|&[a, b, c]| {
            inner_cell.contains(&(a - t_base))
                && inner_cell.contains(&(b - t_base))
                && inner_cell.contains(&(c - t_base))
        });

    let mut mesh_inner_tri_set: HashSet<[u32; 3]> = state
        .triangles
        .iter()
        .filter(|&&[a, b, c]| {
            inner_mesh.contains(&a) && inner_mesh.contains(&b) && inner_mesh.contains(&c)
        })
        .copied()
        .collect();
    // Always include the attachment face so its edges are guaranteed to be bridged,
    // even when its vertices lie just outside the connection sphere.
    mesh_inner_tri_set.insert(of_tri);

    let cell_loops = boundary_loops_all(&cell_inner_tris);
    let mesh_loops = boundary_loops_all(&mesh_inner_tri_set.iter().copied().collect::<Vec<_>>());

    // Stitch every mesh boundary loop to the nearest cell boundary loop.
    let mut bridge_tris: Vec<[u32; 3]> = Vec::new();
    for mesh_loop in &mesh_loops {
        if let Some(cell_loop) = nearest_loop(mesh_loop, &cell_loops, &state.vertices) {
            bridge_tris.extend(zipper_stitch(
                cell_loop, mesh_loop, &state.vertices, attach_normal, global_centroid,
            ));
        }
    }

    let cell_outer_tris: Vec<[u32; 3]> = cell_outer_tris_raw
        .iter()
        .map(|&[a, b, c]| outward_tri(a, b, c, &state.vertices, global_centroid))
        .collect();

    // of_tri is inside mesh_inner_tri_set, so a single retain covers everything.
    state.triangles.retain(|&t| !mesh_inner_tri_set.contains(&t));
    state.triangles.extend(cell_outer_tris.iter().copied());
    state.triangles.extend(bridge_tris.iter().copied());

    // ── Remove old open faces ─────────────────────────────────────────────────
    // of_tri is in mesh_inner_tri_set — one loop handles it and all inner faces.
    let tris_to_remove: Vec<[u32; 3]> = mesh_inner_tri_set.iter().copied().collect();
    for tri in tris_to_remove {
        if let Some(&idx) = state.tri_to_face_idx.get(&tri) {
            state.remove_open_face(idx);
        }
    }

    // ── Tetrahedron frontier update ───────────────────────────────────────────
    // Must happen before add_open_face so newly added faces are checked against
    // the updated spatial hash.
    if GROWTH_MODE == GrowthMode::Tetrahedron {
        state.tetra_occupied.insert(tetra_spatial_key(center), center);
        let mc_sq = MIN_CENTER_DIST * MIN_CENTER_DIST;
        state.tetra_frontier.retain(|_, &mut cand| (cand - center).length_squared() > mc_sq);
    }

    // ── Add new open faces ────────────────────────────────────────────────────
    let placed_key = lattice_key(center);
    for &tri in &cell_outer_tris {
        state.add_open_face(tri, placed_key);
    }

    // ── Dodecahedron frontier update ──────────────────────────────────────────
    if GROWTH_MODE == GrowthMode::Dodecahedron {
        let scale = dodecahedron::center_scale(EDGE_LEN);
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

    state.centers.push(center);
    state.mesh_dirty = true;
}

// ── Boundary loop extraction ──────────────────────────────────────────────────

/// Returns every connected boundary loop of `inner_faces` as a separate `Vec<u32>`.
/// A boundary edge is one that appears exactly once across all faces.
/// Each returned loop is ordered (walking the ring) and has ≥ 3 vertices.
fn boundary_loops_all(inner_faces: &[[u32; 3]]) -> Vec<Vec<u32>> {
    if inner_faces.is_empty() {
        return vec![];
    }

    let mut edge_count: HashMap<(u32, u32), usize> = HashMap::new();
    for &[a, b, c] in inner_faces {
        for (p, q) in [(a, b), (b, c), (c, a)] {
            let key = if p < q { (p, q) } else { (q, p) };
            *edge_count.entry(key).or_insert(0) += 1;
        }
    }

    let mut adj: HashMap<u32, Vec<u32>> = HashMap::new();
    for ((a, b), count) in edge_count {
        if count == 1 {
            adj.entry(a).or_default().push(b);
            adj.entry(b).or_default().push(a);
        }
    }

    if adj.is_empty() {
        return vec![];
    }

    let all_starts: Vec<u32> = adj.keys().copied().collect();
    let mut globally_visited: HashSet<u32> = HashSet::new();
    let mut all_loops: Vec<Vec<u32>> = Vec::new();

    for start in all_starts {
        if globally_visited.contains(&start) {
            continue;
        }

        let mut loop_verts = vec![start];
        globally_visited.insert(start);
        let mut prev = u32::MAX;
        let mut cur = start;

        loop {
            // Accept the next vertex if it isn't where we came from and either
            // closes the ring (== start) or hasn't been visited yet.
            let next = adj[&cur]
                .iter()
                .copied()
                .find(|&v| v != prev && (v == start || !globally_visited.contains(&v)));
            match next {
                Some(v) if v == start => break, // closed loop
                Some(v) => {
                    globally_visited.insert(v);
                    loop_verts.push(v);
                    prev = cur;
                    cur = v;
                }
                None => break, // dead end (non-manifold mesh — emit partial loop)
            }
        }

        if loop_verts.len() >= 3 {
            all_loops.push(loop_verts);
        }
    }

    all_loops
}

/// Returns the loop from `loops` whose centroid is closest to the centroid of `target`.
fn nearest_loop<'a>(target: &[u32], loops: &'a [Vec<u32>], verts: &[Vec3]) -> Option<&'a Vec<u32>> {
    let tc: Vec3 = target.iter().map(|&i| verts[i as usize]).sum::<Vec3>() / target.len() as f32;
    loops.iter().min_by(|a, b| {
        let ca: Vec3 = a.iter().map(|&i| verts[i as usize]).sum::<Vec3>() / a.len() as f32;
        let cb: Vec3 = b.iter().map(|&i| verts[i as usize]).sum::<Vec3>() / b.len() as f32;
        (ca - tc)
            .length_squared()
            .partial_cmp(&(cb - tc).length_squared())
            .unwrap()
    })
}

// ── Greedy zipper stitch ──────────────────────────────────────────────────────

fn zipper_stitch(
    cell_loop: &[u32],
    mesh_loop: &[u32],
    verts: &[Vec3],
    axis: Vec3,
    interior: Vec3,
) -> Vec<[u32; 3]> {
    if cell_loop.is_empty() || mesh_loop.is_empty() {
        return vec![];
    }

    let up = if axis.dot(Vec3::Z).abs() < 0.9 { Vec3::Z } else { Vec3::X };
    let u_dir = axis.cross(up).normalize();
    let v_dir = axis.cross(u_dir);

    let ring_centroid = |loop_: &[u32]| -> Vec3 {
        loop_.iter().map(|&i| verts[i as usize]).sum::<Vec3>() / loop_.len() as f32
    };

    let origin = (ring_centroid(cell_loop) + ring_centroid(mesh_loop)) * 0.5;

    let angle_of = |idx: u32| -> f32 {
        let d = verts[idx as usize] - origin;
        d.dot(v_dir).atan2(d.dot(u_dir))
    };

    let sort_loop = |loop_: &[u32]| -> Vec<u32> {
        let mut v = loop_.to_vec();
        v.sort_by(|&a, &b| angle_of(a).partial_cmp(&angle_of(b)).unwrap());
        v
    };

    let sorted_cell = sort_loop(cell_loop);
    let sorted_mesh = sort_loop(mesh_loop);

    let nc = sorted_cell.len();
    let nm = sorted_mesh.len();
    let mut tris: Vec<[u32; 3]> = Vec::with_capacity(nc + nm);
    let mut ci = 0usize;
    let mut mi = 0usize;

    while ci < nc || mi < nm {
        let c0 = sorted_cell[ci % nc];
        let c1 = sorted_cell[(ci + 1) % nc];
        let m0 = sorted_mesh[mi % nm];
        let m1 = sorted_mesh[(mi + 1) % nm];

        let advance_c = if ci >= nc {
            false
        } else if mi >= nm {
            true
        } else {
            angle_of(c1) <= angle_of(m1)
        };

        if advance_c {
            tris.push(outward_tri(c0, c1, m0, verts, interior));
            ci += 1;
        } else {
            tris.push(outward_tri(c0, m0, m1, verts, interior));
            mi += 1;
        }
    }

    tris
}
