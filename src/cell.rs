// Cell, BodyPart, and procedural mesh generation.
//
// New architecture (May 2026 rewrite):
//
//   - A `Cell` is a *vertex-cell*: a logical 3D point of a given `CellType`
//     (Photo or NonPhoto). Cells are not rendered directly — they parametrise
//     the mesh that wraps them.
//
//   - A `BodyPart` is a list of cells with a local offset relative to the
//     organism root. Each body part renders as one Bevy `Mesh` (one entity).
//
//   - An `Organism` is a list of body parts. The organism root entity owns
//     the world transform; each body part is a child entity with its own
//     `Transform::from_translation(local_offset)` and its own `Mesh3d`.
//
// Mesh generation rules (`generate_body_part_mesh`):
//   - 0 cells     → empty mesh
//   - 1 cell      → rhombic dodecahedron centred on the cell. This is the
//                   canonical starter shape for every freshly-spawned
//                   organism (one body part, one cell of the appropriate
//                   trophic type — Photo for photoautotrophs, NonPhoto for
//                   heterotrophs).
//   - 2+ cells    → subdivided icosphere wrapping the cell cloud, with each
//                   surface vertex displaced toward the nearest cell to give
//                   an organic bumpy silhouette and per-vertex colours
//                   blended from cell influence. Surface regions weakly
//                   influenced by any cell fade to plain white (the safety
//                   fallback the design requires).
//
// Vertex colouring: for each surface vertex we compute an inverse-square
// falloff weighted average of nearby cells' colours (Photo green / NonPhoto
// red), then lerp toward white as the total influence weakens. This ensures
// every body part is fully covered in green/red spots near cells with white
// only appearing where cells genuinely don't reach — exactly the spec.

use bevy::prelude::*;
use bevy::mesh::Indices;
pub use bevy::render::render_resource::PrimitiveTopology;
use bevy::asset::RenderAssetUsages;
use std::collections::HashMap;


// ── Tunables ─────────────────────────────────────────────────────────────────

/// Uniform spacing between neighbouring cells inside a body part. Growth
/// always places new cells at exactly this distance from their nearest
/// existing neighbour, which guarantees consistent topology across all
/// organisms and keeps mesh generation well-conditioned.
pub const CELL_SPACING: f32 = 1.0;

/// Half-extent of the rhombic-dodecahedron starter shape used for the
/// 1-cell body part. The full polyhedron spans `4*RD_HALF_SIZE` along each
/// of its principal axes.
pub const RD_HALF_SIZE: f32 = 0.5;

/// Cell colour falloff distance — beyond this, a cell contributes 0 colour.
/// Picked at `CELL_SPACING * 1.2` so neighbouring cells' colour spots
/// always overlap and there are no zero-colour zones between cells.
pub const CELL_INFLUENCE_RADIUS: f32 = CELL_SPACING * 1.2;

/// Subdivision level of the body-part icosphere. 2 → 162 vertices / 320
/// triangles per body part. Big enough that vertex-colour interpolation
/// looks smooth, small enough to keep ~1100 organisms cheap to redraw.
pub const ICOSPHERE_SUBDIVISIONS: u32 = 2;

/// Outward padding the wrapping mesh takes beyond the cell cloud so cells
/// sit comfortably under the surface (otherwise extreme cells coincide with
/// the icosphere and look pinched).
pub const MESH_PADDING: f32 = CELL_SPACING * 0.55;

/// Effective collision radius of a single cell — sphere-vs-sphere narrow
/// phase uses this. Sized just over half a cell-spacing so two cells from
/// different organisms touch when within `CELL_SPACING`.
pub const CELL_COLLISION_RADIUS: f32 = CELL_SPACING * 0.55;


// ── Cell type ────────────────────────────────────────────────────────────────

/// The two cell types defined by the new architecture. Photo cells drive
/// photosynthesis (green); NonPhoto cells are everything else (mild red).
/// Additional types can be added later — the colouring code is data-driven
/// off `color()` so adding a variant requires no other edits here.
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub enum CellType {
    Photo,
    NonPhoto,
}

impl CellType {
    /// Linear-RGB colour painted onto the surface around a cell of this type.
    #[inline]
    pub fn color(&self) -> [f32; 3] {
        match self {
            CellType::Photo    => [0.16, 0.85, 0.18], // bright green
            CellType::NonPhoto => [0.85, 0.20, 0.16], // mild red
        }
    }
}


// ── Cell ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Cell {
    /// Position relative to the body part's origin.
    pub local_pos: Vec3,
    pub cell_type: CellType,
}


// ── Body part ────────────────────────────────────────────────────────────────

/// Anatomical role of a body part. Currently a hint; not consumed by
/// anything load-bearing yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyPartKind {
    Body,
    Limb,
    Organ,
}

#[derive(Clone, Debug)]
pub struct BodyPart {
    pub kind: BodyPartKind,
    /// Offset of this body part's origin relative to the organism root.
    pub local_offset: Vec3,
    /// Cells composing this body part. Mesh, collision, energy and
    /// photosynthesis counts all read from here.
    pub cells: Vec<Cell>,
    /// True after `predation.rs` has eaten this body part. Marks it as
    /// soft-deleted: cells are cleared and the child mesh entity is
    /// despawned, but the slot stays in `Organism::body_parts` so existing
    /// `BodyPartIndex` references on sibling children remain stable. All
    /// iteration sites filter on `is_alive()`.
    pub consumed: bool,
}

impl BodyPart {
    /// True if the body part still has at least one cell and has not been
    /// eaten by a predator. Iterators that compute aggregate properties
    /// (mass, bounding radius, photo cell count) skip non-alive body parts.
    #[inline]
    pub fn is_alive(&self) -> bool {
        !self.consumed && !self.cells.is_empty()
    }

    /// Local-space AABB enclosing all grown cells, padded so the wrapping
    /// mesh fits inside it. Returns `None` for body parts with no grown
    /// cells (the only legitimate case is a body part that has just been
    /// scheduled but not yet had its first cell grown). Currently unused —
    /// kept as a building block for future spatial-index optimisations of
    /// world-mesh collision queries.
    #[allow(dead_code)]
    pub fn local_aabb(&self) -> Option<(Vec3, Vec3)> {
        if self.cells.is_empty() { return None; }
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        // Pad enough to cover both the icosphere mesh wrapping and the
        // single-cell rhombic-dodecahedron starter shape.
        let pad = Vec3::splat(MESH_PADDING.max(2.0 * RD_HALF_SIZE));
        for c in &self.cells {
            lo = lo.min(c.local_pos - pad);
            hi = hi.max(c.local_pos + pad);
        }
        Some((lo, hi))
    }

    /// Local-space bounding-sphere radius (centred at the cell centroid)
    /// used by the broad phase of organism-vs-organism collision.
    pub fn local_bounding_radius(&self) -> f32 {
        if self.cells.is_empty() { return 0.0; }
        let centroid = self.cells.iter().fold(Vec3::ZERO, |a, c| a + c.local_pos)
                        / self.cells.len() as f32;
        self.cells.iter()
            .map(|c| (c.local_pos - centroid).length())
            .fold(0.0_f32, f32::max)
            + MESH_PADDING.max(2.0 * RD_HALF_SIZE)
    }

    /// (photo, non_photo) tally over grown cells.
    pub fn cell_counts(&self) -> (u32, u32) {
        let mut p = 0; let mut np = 0;
        for c in &self.cells {
            match c.cell_type {
                CellType::Photo    => p  += 1,
                CellType::NonPhoto => np += 1,
            }
        }
        (p, np)
    }
}


// ── Marker components ────────────────────────────────────────────────────────

/// Marker on the child entity that owns the body part's `Mesh3d`. Used by
/// the gizmo and mesh-rebuild systems to find body-part meshes.
#[derive(Component)]
pub struct OrganismMesh;

/// Index back into `Organism::body_parts` for a body-part child entity.
/// Lets systems that walk children (mesh rebuild, collision visualisation)
/// recover which body part each child represents in O(1).
#[derive(Component, Clone, Copy)]
pub struct BodyPartIndex(pub usize);


// ── Mesh generation ──────────────────────────────────────────────────────────

/// Build the renderable mesh for one body part.
///
/// Topology is a function of cell count:
///   - 1 cell  → rhombic dodecahedron (the canonical organism starter shape)
///   - 2+ cells → subdivided icosphere wrapping the cell cloud, surface
///                displaced toward the dominant cell direction at each
///                vertex, per-vertex colour blended from cell influence
///                with white as the safety fallback.
pub fn generate_body_part_mesh(part: &BodyPart) -> Mesh {
    let cells = part.cells.as_slice();
    if cells.is_empty()      { return empty_mesh(); }
    if cells.len() == 1      { return generate_rhombic_dodecahedron_mesh(&cells[0]); }
    generate_wrapping_mesh(cells)
}


fn empty_mesh() -> Mesh {
    let mut m = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    m.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new());
    m.insert_attribute(Mesh::ATTRIBUTE_NORMAL,   Vec::<[f32; 3]>::new());
    m.insert_attribute(Mesh::ATTRIBUTE_COLOR,    Vec::<[f32; 4]>::new());
    m.insert_indices(Indices::U32(Vec::new()));
    m
}


// Rhombic dodecahedron — 14 vertices, 12 quad faces (24 triangles). Centred
// on the cell's local position. All vertices share the cell's colour, so
// the polyhedron renders as a solid green or red shape — the deliberate,
// recognisable starter look for new organisms before any growth has
// happened.
fn generate_rhombic_dodecahedron_mesh(cell: &Cell) -> Mesh {
    let s = RD_HALF_SIZE;
    let centre = cell.local_pos;

    // 8 cube corners + 6 axis tips. The polyhedron's 12 face quads are listed
    // below; ordering matches the historical face-emission order so face
    // normals come out consistently outward.
    let v: [Vec3; 14] = [
        Vec3::new( s,  s,  s), Vec3::new( s,  s, -s),
        Vec3::new( s, -s,  s), Vec3::new( s, -s, -s),
        Vec3::new(-s,  s,  s), Vec3::new(-s,  s, -s),
        Vec3::new(-s, -s,  s), Vec3::new(-s, -s, -s),
        Vec3::new( 2.0*s,  0.0,    0.0  ),
        Vec3::new(-2.0*s,  0.0,    0.0  ),
        Vec3::new( 0.0,    2.0*s,  0.0  ),
        Vec3::new( 0.0,   -2.0*s,  0.0  ),
        Vec3::new( 0.0,    0.0,    2.0*s),
        Vec3::new( 0.0,    0.0,   -2.0*s),
    ];

    // 12 quads (top cap, bottom cap, middle ring).
    const QUADS: [[usize; 4]; 12] = [
        [12,  0, 10,  4],
        [12,  4,  9,  6],
        [12,  6, 11,  2],
        [12,  2,  8,  0],
        [ 5, 10,  1, 13],
        [ 7,  9,  5, 13],
        [ 3, 11,  7, 13],
        [ 1,  8,  3, 13],
        [ 1, 10,  0,  8],
        [ 5,  9,  4, 10],
        [ 7, 11,  6,  9],
        [ 3,  8,  2, 11],
    ];

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(48);
    let mut normals:   Vec<[f32; 3]> = Vec::with_capacity(48);
    let mut colors:    Vec<[f32; 4]> = Vec::with_capacity(48);
    let mut indices:   Vec<u32>      = Vec::with_capacity(72);

    let cc    = cell.cell_type.color();
    let color = [cc[0], cc[1], cc[2], 1.0];

    for q in QUADS {
        let p0 = v[q[0]]; let p1 = v[q[1]]; let p2 = v[q[2]]; let p3 = v[q[3]];
        let n  = (p1 - p0).cross(p2 - p0).normalize_or_zero();
        let n_arr = [n.x, n.y, n.z];
        let base = positions.len() as u32;
        for p in [p0, p1, p2, p3] {
            let world = p + centre;
            positions.push([world.x, world.y, world.z]);
            normals.push(n_arr);
            colors.push(color);
        }
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL,   normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR,    colors);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}


// 2+ cells: subdivided icosphere centred on the cell centroid, sized to
// wrap all cells with `MESH_PADDING` of slack. Each surface vertex is
// nudged toward the dominant cell direction (organic bumpy silhouette)
// and coloured by inverse-square cell influence with a white fallback.
fn generate_wrapping_mesh(cells: &[Cell]) -> Mesh {
    // Cell centroid + furthest-cell radius → wrapping sphere.
    let centroid = cells.iter().fold(Vec3::ZERO, |a, c| a + c.local_pos)
                   / cells.len() as f32;
    let cloud_r  = cells.iter()
        .map(|c| (c.local_pos - centroid).length())
        .fold(0.0_f32, f32::max);
    let radius   = cloud_r + MESH_PADDING;

    let (mut positions, indices) = make_icosphere(centroid, radius, ICOSPHERE_SUBDIVISIONS);

    // ── Surface displacement ───────────────────────────────────────────────
    // For each vertex, find the cell whose direction-from-centroid most
    // aligns with the vertex's outward direction, then bias the vertex
    // toward that cell's direction. This produces gentle bumps over each
    // cell and matches the cell-driven silhouette the spec asks for.
    let cell_dirs: Vec<Vec3> = cells.iter()
        .map(|c| (c.local_pos - centroid).normalize_or_zero())
        .collect();

    for v in positions.iter_mut() {
        let dir_v = (*v - centroid).normalize_or_zero();
        // Find the most-aligned cell direction.
        let mut best_dot = f32::NEG_INFINITY;
        let mut best_dir = dir_v;
        for &cd in &cell_dirs {
            let d = cd.dot(dir_v);
            if d > best_dot { best_dot = d; best_dir = cd; }
        }
        // Smooth bias toward the dominant direction; bumps remain subtle
        // because best_dot already saturates near 1 only for the
        // closest-aligned cell.
        let t = best_dot.max(0.0).powi(2);
        let bumped = (dir_v * (1.0 - t) + best_dir * t).normalize_or_zero();
        // Slight inward modulation between cells keeps the silhouette
        // reading as cell clusters rather than a uniform sphere.
        let r = radius * (0.92 + 0.08 * t);
        *v = centroid + bumped * r;
    }

    // ── Per-vertex colouring ───────────────────────────────────────────────
    // Inverse-square falloff over cells in range. Total weight is clamped
    // to [0,1] and used to lerp toward white where cells don't reach the
    // surface — that's the safety fallback the spec requires.
    let r2 = CELL_INFLUENCE_RADIUS * CELL_INFLUENCE_RADIUS;
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(positions.len());
    for v in &positions {
        let mut weighted = Vec3::ZERO;
        let mut weight   = 0.0_f32;
        for cell in cells {
            let d2 = (*v - cell.local_pos).length_squared();
            if d2 < r2 {
                let w  = (1.0 - d2 / r2).max(0.0).powi(2);
                let cc = cell.cell_type.color();
                weighted += Vec3::new(cc[0], cc[1], cc[2]) * w;
                weight   += w;
            }
        }
        let color = if weight > 1e-4 {
            let avg       = weighted / weight;
            let influence = weight.min(1.0);
            // Lerp toward white as influence weakens — gives a smooth
            // transition into white pads where cells are out of range.
            let mixed = avg * influence + Vec3::ONE * (1.0 - influence);
            [mixed.x, mixed.y, mixed.z, 1.0]
        } else {
            [1.0, 1.0, 1.0, 1.0] // white safety fallback
        };
        colors.push(color);
    }

    let normals = compute_normals(&positions, &indices);
    let positions_arr: Vec<[f32; 3]> = positions.iter().map(|v| [v.x, v.y, v.z]).collect();

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions_arr);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL,   normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR,    colors);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}


// ── Icosphere ────────────────────────────────────────────────────────────────

/// Generates a subdivided icosphere centred at `center` with given `radius`.
/// `subdivisions = 0` is the base 12-vertex icosahedron; each level multiplies
/// the triangle count by 4.
pub fn make_icosphere(center: Vec3, radius: f32, subdivisions: u32) -> (Vec<Vec3>, Vec<u32>) {
    let phi      = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let inv_norm = 1.0 / (1.0 + phi*phi).sqrt();

    let mut positions: Vec<Vec3> = vec![
        Vec3::new(-1.0,  phi,  0.0),
        Vec3::new( 1.0,  phi,  0.0),
        Vec3::new(-1.0, -phi,  0.0),
        Vec3::new( 1.0, -phi,  0.0),
        Vec3::new( 0.0, -1.0,  phi),
        Vec3::new( 0.0,  1.0,  phi),
        Vec3::new( 0.0, -1.0, -phi),
        Vec3::new( 0.0,  1.0, -phi),
        Vec3::new( phi,  0.0, -1.0),
        Vec3::new( phi,  0.0,  1.0),
        Vec3::new(-phi,  0.0, -1.0),
        Vec3::new(-phi,  0.0,  1.0),
    ].into_iter().map(|v| v * inv_norm).collect();

    let mut indices: Vec<u32> = vec![
        0, 11, 5,    0, 5, 1,     0, 1, 7,     0, 7, 10,    0, 10, 11,
        1, 5, 9,     5, 11, 4,    11, 10, 2,   10, 7, 6,    7, 1, 8,
        3, 9, 4,     3, 4, 2,     3, 2, 6,     3, 6, 8,     3, 8, 9,
        4, 9, 5,     2, 4, 11,    6, 2, 10,    8, 6, 7,     9, 8, 1,
    ];

    for _ in 0..subdivisions {
        let mut new_indices = Vec::with_capacity(indices.len() * 4);
        let mut cache: HashMap<(u32, u32), u32> = HashMap::new();

        for tri in indices.chunks_exact(3) {
            let a = tri[0]; let b = tri[1]; let c = tri[2];
            let ab = midpoint(a, b, &mut positions, &mut cache);
            let bc = midpoint(b, c, &mut positions, &mut cache);
            let ca = midpoint(c, a, &mut positions, &mut cache);

            new_indices.extend_from_slice(&[a, ab, ca]);
            new_indices.extend_from_slice(&[b, bc, ab]);
            new_indices.extend_from_slice(&[c, ca, bc]);
            new_indices.extend_from_slice(&[ab, bc, ca]);
        }
        indices = new_indices;
    }

    for p in positions.iter_mut() {
        *p = *p * radius + center;
    }

    (positions, indices)
}

fn midpoint(
    a: u32, b: u32,
    positions: &mut Vec<Vec3>,
    cache: &mut HashMap<(u32, u32), u32>,
) -> u32 {
    let key = if a < b { (a, b) } else { (b, a) };
    if let Some(&i) = cache.get(&key) { return i; }
    let mid = ((positions[a as usize] + positions[b as usize]) * 0.5).normalize();
    let i   = positions.len() as u32;
    positions.push(mid);
    cache.insert(key, i);
    i
}

fn compute_normals(positions: &[Vec3], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![Vec3::ZERO; positions.len()];
    for tri in indices.chunks_exact(3) {
        let a = positions[tri[0] as usize];
        let b = positions[tri[1] as usize];
        let c = positions[tri[2] as usize];
        let n = (b - a).cross(c - a);
        normals[tri[0] as usize] += n;
        normals[tri[1] as usize] += n;
        normals[tri[2] as usize] += n;
    }
    normals.iter().map(|n| {
        let unit = n.normalize_or_zero();
        [unit.x, unit.y, unit.z]
    }).collect()
}
