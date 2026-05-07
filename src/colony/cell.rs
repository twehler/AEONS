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

/// Energy a freshly grown cell starts with. Owned by `physiology.rs` —
/// every constructor here uses it as the default so cells exist in a
/// well-defined initial state regardless of who builds them.
pub const DEFAULT_CELL_ENERGY: f32 = 1.0;

/// Maximum number of rhombic-dodecahedron neighbours any single cell can
/// have. The RD lattice exposes exactly 18 adjacency slots (6 axis-aligned
/// + 12 FCC face-diagonal — see `volumetric_growth/dodecahedron::SLOT_DIRS`).
/// All physiology formulas that scale with neighbour density should
/// normalise against this constant rather than a hardcoded literal.
pub const MAX_RD_NEIGHBOURS: u8 = 18;


/// Photosynthesis-specific data stored on Photo cells.
///
/// `neighbour_count` mirrors `Cell::neighbour_count` so the value is
/// available without an additional indirection when the physiology tick
/// only iterates the photo subset. `energy_production` is the cached
/// per-tick output, derived from the formula
///
///   E = BASE_PHOTO_PRODUCTION
///       + (PHOTO_PRODUCTION_PER_CELL / MAX_RD_NEIGHBOURS) * neighbour_count
///
/// The constant baseline (0.2) ensures isolated single-cell photoautotrophs
/// still gain a small amount of energy per tick — without it, a gen-0
/// 1-cell organism (zero neighbours) would never reach the reproduction
/// threshold and the species would die out at gen 0.
///
/// Both fields change only when the cell's neighbour set changes (currently
/// only at birth — new cells appended during reproduction trigger a
/// recompute on the affected body part). Per-tick reads of
/// `energy_production` are O(1).
#[derive(Clone, Debug)]
pub struct PhotosyntheticCell {
    pub neighbour_count:   u8,
    pub energy_production: f32,
}

/// Floor on per-cell photosynthetic production, applied before the
/// neighbour-count term. Guarantees isolated photo cells still produce
/// non-zero energy.
pub const BASE_PHOTO_PRODUCTION: f32 = 0.2;

impl PhotosyntheticCell {
    /// Build a `PhotosyntheticCell` for a cell whose current neighbour
    /// count is `n`, using `photo_production_per_cell` as the baseline
    /// "fully-surrounded" production rate (i.e. the value for n = 18).
    /// Caller passes the constant rather than us importing it from
    /// `energy.rs` to keep `cell.rs` free of cross-module deps.
    #[inline]
    pub fn new(n: u8, photo_production_per_cell: f32) -> Self {
        let n_clamped = n.min(MAX_RD_NEIGHBOURS);
        let energy_production = BASE_PHOTO_PRODUCTION
            + (photo_production_per_cell / MAX_RD_NEIGHBOURS as f32) * n_clamped as f32;
        Self { neighbour_count: n_clamped, energy_production }
    }
}


#[derive(Clone, Debug)]
pub struct Cell {
    /// Position relative to the body part's origin.
    pub local_pos: Vec3,
    pub cell_type: CellType,
    /// Per-cell energy reservoir. Read and updated by `PhysiologyPlugin`
    /// (in `src/physiology/physiology.rs`); independent of the
    /// organism-level `Organism::energy`. Used as the basis for cell-level
    /// physiology rules (decay, photosynthesis credit, growth gating, …).
    pub cell_energy: f32,
    /// Count of RD-adjacent neighbour cells (within the same body part).
    /// Range [0, MAX_RD_NEIGHBOURS = 18]. Updated only when cells are added
    /// or removed (currently birth + predation), not per tick.
    pub neighbour_count: u8,
    /// Some(...) only for Photo cells, carrying their cached photosynthesis
    /// data. None for NonPhoto cells. `physiology::recompute_body_part`
    /// keeps this in lockstep with `cell_type` and `neighbour_count`.
    pub photo: Option<PhotosyntheticCell>,
}

impl Cell {
    /// Construct a cell with the default starting energy, no neighbours
    /// counted yet, and no photosynthetic data populated. Body-part
    /// constructors call `physiology::recompute_body_part` after assembling
    /// their cell list, which fills in `neighbour_count` and `photo`.
    #[inline]
    pub fn new(local_pos: Vec3, cell_type: CellType) -> Self {
        Self {
            local_pos,
            cell_type,
            cell_energy: DEFAULT_CELL_ENERGY,
            neighbour_count: 0,
            photo: None,
        }
    }
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
    /// Per-part genome — order-of-cell-growth ledger for THIS body part.
    /// Each entry is `(index, position relative to body-part origin, type)`.
    /// `volumetric_growth::build_mesh_from_ocg` welds vertices across all
    /// positions it sees, so every body part owns its own OCG; sharing
    /// across parts would fuse mesh topology and break independent rotation.
    pub ocg: Vec<(usize, Vec3, CellType)>,
    /// Some(...) when this body part hangs off another. `None` for the root
    /// part (index 0). The attachment carries the pivot point in the parent's
    /// local frame plus a rotation Quat for future limb animation.
    pub attachment: Option<crate::body_part::Attachment>,
    /// True after `predation.rs` has eaten this body part. Marks it as
    /// soft-deleted: cells are cleared and the child mesh entity is
    /// despawned, but the slot stays in `Organism::body_parts` so existing
    /// `BodyPartIndex` references on sibling children remain stable. All
    /// iteration sites filter on `is_alive()`.
    pub consumed: bool,
    /// Debug flag — render this body part with a blue material instead of
    /// the trophic colour. Set on freshly-spawned branches so the user can
    /// see appendages clearly while the system is being developed.
    pub debug_blue: bool,
    /// Can this body part still grow? When `false`, mesh-building and
    /// mutation skip this part (used by Krishi's hand-built body, where
    /// the cell layout is fixed).
    pub regrowable: bool,
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


