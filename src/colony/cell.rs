// Cell, BodyPart, and procedural mesh generation.
//
//   - A `Cell` is a vertex-cell: a logical 3D point of a `CellType`. Cells
//     are not rendered directly — they parametrise the wrapping mesh.
//   - A `BodyPart` is a list of cells with a local offset; renders as one
//     Bevy `Mesh` (one entity).
//   - An `Organism` is a list of body parts. The root owns the world
//     transform; each body part is a child with its own offset + `Mesh3d`.

use bevy::prelude::*;


// ── Tunables ─────────────────────────────────────────────────────────────────

/// Uniform spacing between neighbouring cells inside a body part. Growth
/// places new cells at exactly this distance from their nearest neighbour,
/// guaranteeing consistent topology and well-conditioned mesh generation.
pub const CELL_SPACING: f32 = 1.0;

/// Half-extent of the rhombic-dodecahedron 1-cell starter shape. The full
/// polyhedron spans `4*RD_HALF_SIZE` along each principal axis.
pub const RD_HALF_SIZE: f32 = 0.5;

/// Outward padding the wrapping mesh takes beyond the cell cloud so extreme
/// cells don't coincide with the icosphere and look pinched.
pub const MESH_PADDING: f32 = CELL_SPACING * 0.55;

/// Single-cell collision radius (sphere-vs-sphere narrow phase). Sized so the
/// contact threshold (2× = 1.4) gives an along-axis window of
/// `√(1.4² − 1.155²) ≈ 0.79` units against the bilateral perpendicular offset
/// `MIN_X_BILATERAL ≈ 1.155` — enough that the V-shape triggers predation
/// even on an off-axis approach.
pub const CELL_COLLISION_RADIUS: f32 = CELL_SPACING * 0.7;


// ── Cell type ────────────────────────────────────────────────────────────────

/// Cell types. Photo drives photosynthesis (green); NonPhoto is everything
/// else (red). Colouring is data-driven off `color()`.
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub enum CellType {
    Photo,
    NonPhoto,
    /// Inert debug/placeholder cell. Renders blue. Never photosynthesises;
    /// counts as body mass like `NonPhoto` for upkeep. Used by the species
    /// editor to sketch out new (main-body) limbs.
    Placeholder,
    /// Inert PURPLE cell — identical to `Placeholder` but a distinct colour
    /// so sub-limbs (limbs attached to a parent limb) read differently from
    /// their parent limbs. Auto-selected by the species editor for sub-limbs.
    SubLimb,
    /// Behaves exactly like `NonPhoto` (non-photosynthetic body mass); only the
    /// render colour differs. Placeholder for future differentiated cell roles.
    YellowCell,
    /// Behaves exactly like `NonPhoto`; renders orange. See `YellowCell`.
    OrangeCell,
    /// Behaves exactly like `NonPhoto`; renders brown. See `YellowCell`.
    BrownCell,
}

impl CellType {
    /// Linear-RGB colour for `Mesh::ATTRIBUTE_COLOR` (Bevy treats vertex
    /// colours as linear). Values are the linear equivalents of the sRGB
    /// display colours noted per arm — keep the comments in sync.
    #[inline]
    pub fn color(&self) -> [f32; 3] {
        match self {
            // sRGB (0.2, 0.8, 0.2)  — bright green
            CellType::Photo       => [0.0331, 0.6038, 0.0331],
            // sRGB (0.8, 0.2, 0.2) — mild red
            CellType::NonPhoto    => [0.6038, 0.0331, 0.0331],
            // sRGB (0.2, 0.4, 0.95) — debug blue
            CellType::Placeholder => [0.0331, 0.1329, 0.8902],
            // sRGB (0.6, 0.2, 0.9) — purple (sub-limb marker)
            CellType::SubLimb     => [0.3186, 0.0331, 0.7874],
            // sRGB (1.0, 0.95, 0.1) — yellow
            CellType::YellowCell  => [1.0, 0.8879, 0.0099],
            // sRGB (1.0, 0.55, 0.0) — orange
            CellType::OrangeCell  => [1.0, 0.2623, 0.0],
            // sRGB (0.45, 0.27, 0.12) — brown
            CellType::BrownCell   => [0.1651, 0.0578, 0.0144],
        }
    }

    /// True only for `Photo`; all others are non-photosynthetic.
    #[inline]
    pub fn is_photo(&self) -> bool {
        matches!(self, CellType::Photo)
    }
}


// ── Cell ─────────────────────────────────────────────────────────────────────

/// Energy a freshly grown cell starts with.
pub const DEFAULT_CELL_ENERGY: f32 = 1.0;

/// Max RD neighbours any cell can have: the lattice exposes exactly 18
/// adjacency slots (6 axis-aligned + 12 FCC face-diagonal). Neighbour-density
/// formulas should normalise against this, not a literal.
pub const MAX_RD_NEIGHBOURS: u8 = 18;


/// Photosynthesis data cached on Photo cells.
///
///   E = BASE_PHOTO_PRODUCTION
///       + (PHOTO_PRODUCTION_PER_CELL / MAX_RD_NEIGHBOURS) * neighbour_count
///
/// The baseline keeps isolated 1-cell photoautotrophs gaining energy, so a
/// gen-0 organism can still reach the reproduction threshold. Both fields
/// change only when the neighbour set changes; per-tick reads are O(1).
#[derive(Clone, Debug)]
pub struct PhotosyntheticCell {
    pub neighbour_count:   u8,
    pub energy_production: f32,
}

/// Floor on per-cell photosynthetic production (before the neighbour term);
/// keeps isolated photo cells producing non-zero energy.
pub const BASE_PHOTO_PRODUCTION: f32 = 0.2;

impl PhotosyntheticCell {
    /// Build for neighbour count `n`. `photo_production_per_cell` is the
    /// fully-surrounded rate (n = 18); passed in to keep `cell.rs` free of
    /// cross-module deps.
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
    /// Per-cell energy reservoir, owned by `PhysiologyPlugin`; independent of
    /// `Organism::energy`. Basis for cell-level physiology rules.
    pub cell_energy: f32,
    /// RD-adjacent neighbour count within the same body part, range
    /// [0, MAX_RD_NEIGHBOURS]. Updated at composition changes, not per tick.
    pub neighbour_count: u8,
    /// `Some` only for Photo cells (cached photosynthesis data); kept in sync
    /// with `cell_type`/`neighbour_count` by `physiology::recompute_body_part`.
    pub photo: Option<PhotosyntheticCell>,
}

impl Cell {
    /// Cell with default energy, no neighbours counted, no photo data;
    /// `physiology::recompute_body_part` fills `neighbour_count`/`photo` later.
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

/// Anatomical role of a body part — and, for appendages, how it behaves at
/// spawn (mirroring) and runtime (joint + brain wiring):
///   * `Body`    — the root part (index 0); never an appendage.
///   * `Limb`    — paired moving appendage: in a Bilateral organism it expands
///                 into a right+left PAIR of separately-jointed parts; motorized
///                 joint driven by the brain.
///   * `Organ`   — legacy non-limb appendage (colony-editor / pre-v11 `.species`):
///                 behaves like `Limb` at runtime (mirrors + motorized) but tagged
///                 distinctly (cosmetic sliding-animation gate).
///   * `Segment` — midline MOVING structure: a Bilateral organism fuses its two
///                 halves into ONE part (like the base body) instead of splitting;
///                 motorized joint driven by the brain. For a segmented spine.
///   * `Static`  — midline FIXED structure: fuses to one part like `Segment`, but
///                 attaches with a rigid fixed joint — no articulation, and the
///                 brain has no movement connection to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyPartKind {
    Body,
    Limb,
    Organ,
    Segment,
    Static,
}

#[derive(Clone, Debug)]
pub struct BodyPart {
    pub kind: BodyPartKind,
    /// Offset of this body part's origin relative to the organism root.
    pub local_offset: Vec3,
    /// Cells composing this body part. Mesh/collision/energy/photo read here.
    pub cells: Vec<Cell>,
    /// Per-part genome: order-of-cell-growth ledger `(index, pos relative to
    /// part origin, type)`. Each body part owns its own OCG — `build_mesh_from_ocg`
    /// welds across all positions, so sharing would fuse topology and break
    /// independent rotation.
    pub ocg: Vec<(usize, Vec3, CellType)>,
    /// `Some` when this part hangs off another; `None` for the root (index 0).
    /// Carries the pivot point in the parent's local frame plus a rotation Quat.
    pub attachment: Option<crate::body_part::Attachment>,
    /// True after predation ate this part — soft-deleted: cells cleared and
    /// mesh despawned, but the slot stays so sibling `BodyPartIndex` references
    /// stay stable. All iteration sites filter on `is_alive()`.
    pub consumed: bool,
    /// Debug: render blue instead of the trophic colour (set on fresh branches).
    pub debug_blue: bool,
    /// When `false`, mesh-building and mutation skip this part (Krishi's
    /// hand-built fixed body).
    pub regrowable: bool,
}

impl BodyPart {
    /// True if the part still has a cell and hasn't been eaten. Aggregate-
    /// property iterators skip non-alive parts.
    #[inline]
    pub fn is_alive(&self) -> bool {
        !self.consumed && !self.cells.is_empty()
    }

    /// Local-space AABB enclosing all grown cells, padded for the wrapping
    /// mesh. `None` when there are no grown cells. Currently unused.
    #[allow(dead_code)]
    pub fn local_aabb(&self) -> Option<(Vec3, Vec3)> {
        if self.cells.is_empty() { return None; }
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        // Pad to cover both the icosphere mesh and the RD starter shape.
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
            // Placeholder/SubLimb count as non-photo body mass for upkeep.
            match c.cell_type {
                CellType::Photo                          => p  += 1,
                CellType::NonPhoto | CellType::Placeholder | CellType::SubLimb
                | CellType::YellowCell | CellType::OrangeCell | CellType::BrownCell => np += 1,
            }
        }
        (p, np)
    }
}


// ── Marker components ────────────────────────────────────────────────────────

/// Marker on the child entity owning the body part's `Mesh3d`.
#[derive(Component)]
pub struct OrganismMesh;

/// Index back into `Organism::body_parts` for a body-part child entity (O(1)
/// recovery for systems that walk children).
#[derive(Component, Clone, Copy)]
pub struct BodyPartIndex(pub usize);


