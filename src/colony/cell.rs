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
// Derives from the master `simulation_settings::GEOMETRY_SCALE` (one knob for
// organism size). Drives CELL_COLLISION_RADIUS + MESH_PADDING + the editor wrap
// COINCIDE_DIST, which all follow automatically.
pub const CELL_SPACING: f32 = crate::simulation_settings::GEOMETRY_SCALE;

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
//
// All per-type data — render colour & opacity, display label, serialisation
// tag, behavioural role, and upkeep cost — lives in ONE place: `CellType::def()`.
// To add a new cell type:
//   1. add a variant to `CellType`,
//   2. add it to `CellType::ALL`,
//   3. add one arm to the `def()` match (the compiler forces this).
// Nothing else enumerates the variants — everything reads them through `def()`
// / the thin accessors below, and the species-editor palette is driven straight
// off `CellType::ALL`, so a new type shows up there automatically.

/// Cell types. The variant is just an identity tag — its colour, name, save
/// tag, role, and upkeep all come from [`CellType::def`].
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub enum CellType {
    Photo,
    /// Red absorption cell — the default heterotroph body tissue (nutrient
    /// uptake across the membrane).
    AbsorptionCell,
    /// Inert debug/placeholder cell (blue). Used by the species editor to
    /// sketch out new (main-body) limbs.
    Placeholder,
    /// PURPLE gill cell — gas/ion exchange tissue. (Historically the species
    /// editor's sub-limb marker; same colour.)
    GillCell,
    /// Coloured body-mass placeholders for future differentiated cell roles —
    /// today they behave exactly like `AbsorptionCell`; only the colour differs.
    YellowCell,
    OrangeCell,
    BrownCell,
    /// Blueish, translucent jelly tissue. Living body mass like `AbsorptionCell`,
    /// but half the upkeep.
    Gelly,
    /// Inert dead material (hard, dark-brown chitin — e.g. a beetle's shell).
    HardChitin,
    /// Inert dead material (softer, lighter-brown chitin).
    SoftChitin,
    /// Inert dead material (very light-brown keratin — horns, claws, hair).
    Keratin,
    /// Inert dead material (white hydroxylapatite — bone/tooth mineral).
    HydroxylApatite,
    /// Dark, deep-red digestion cell — breaks down absorbed material.
    DigestionCell,
    /// Red blood-vessel cell — transports nutrients; tone between
    /// `AbsorptionCell` and `DigestionCell`.
    BloodVesselCell,
    /// Red muscle cell — contractile tissue (the original AbsorptionCell red).
    MuscleCell,
}

/// Behavioural role of a cell — what the *simulation* does with it, decoupled
/// from its appearance. Add a new role only when there is genuinely new
/// behaviour; a cell that merely looks different is a new [`CellType`] reusing
/// an existing role.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CellRole {
    /// Photosynthesises: produces energy and is counted as a photo cell.
    Photosynthetic,
    /// Living, non-photosynthetic body mass ("heterotrophic" tissue): costs
    /// standard upkeep, never produces energy.
    Structural,
    /// Hard, dead material (horns, armour plates, shell): costs only a small
    /// fraction of standard upkeep and never produces energy.
    Inert,
}

/// Everything the rest of the codebase needs to know about a [`CellType`], in
/// one row. Single source of truth — see the module note above.
#[derive(Clone, Copy, Debug)]
pub struct CellTypeDef {
    /// Stable byte tag for `.colony` / `.species` serialisation. NEVER renumber
    /// or reuse an existing tag — old save files depend on it.
    pub tag: u8,
    /// Display name (species-editor palette, debug output).
    pub label: &'static str,
    /// Display colour in **sRGB**. The single colour definition: the linear
    /// vertex colour ([`CellType::color`]) and UI swatches both derive from it.
    pub srgb: [f32; 3],
    /// Opacity in `[0, 1]`. `1.0` = fully opaque; `< 1.0` renders translucent
    /// (the body part's material switches to alpha-blending).
    pub alpha: f32,
    /// Behavioural classification.
    pub role: CellRole,
    /// Per-cell upkeep as a multiple of `NON_PHOTO_CONSUMPTION_PER_CELL`. `0.0`
    /// for photo cells (no upkeep), `1.0` for standard living tissue, less for
    /// cheaper tissue (jelly) or dead material (inert).
    pub upkeep_mult: f32,
}

impl CellType {
    /// Every variant, in serialisation-tag order. Drives the species-editor
    /// palette and [`CellType::from_tag`].
    pub const ALL: [CellType; 15] = [
        CellType::Photo,
        CellType::AbsorptionCell,
        CellType::Placeholder,
        CellType::GillCell,
        CellType::YellowCell,
        CellType::OrangeCell,
        CellType::BrownCell,
        CellType::Gelly,
        CellType::HardChitin,
        CellType::SoftChitin,
        CellType::Keratin,
        CellType::HydroxylApatite,
        CellType::DigestionCell,
        CellType::BloodVesselCell,
        CellType::MuscleCell,
    ];

    /// The one-row descriptor for this type — the single source of truth for
    /// all per-type data.
    pub const fn def(self) -> CellTypeDef {
        match self {
            CellType::Photo => CellTypeDef {
                tag: 0, label: "Photo",       srgb: [0.2,  0.8,  0.2 ], alpha: 1.0,
                role: CellRole::Photosynthetic, upkeep_mult: 0.0,
            },
            CellType::AbsorptionCell => CellTypeDef {
                tag: 1, label: "Absorption",  srgb: [0.95, 0.72, 0.66], alpha: 1.0,
                role: CellRole::Structural,     upkeep_mult: 1.0,
            },
            CellType::Placeholder => CellTypeDef {
                tag: 2, label: "Placeholder", srgb: [0.2,  0.45, 0.95], alpha: 1.0,
                role: CellRole::Structural,     upkeep_mult: 1.0,
            },
            CellType::GillCell => CellTypeDef {
                tag: 3, label: "Gill",        srgb: [0.6,  0.2,  0.9 ], alpha: 1.0,
                role: CellRole::Structural,     upkeep_mult: 1.0,
            },
            CellType::YellowCell => CellTypeDef {
                tag: 4, label: "Yellow",      srgb: [1.0,  0.95, 0.1 ], alpha: 1.0,
                role: CellRole::Structural,     upkeep_mult: 1.0,
            },
            CellType::OrangeCell => CellTypeDef {
                tag: 5, label: "Orange",      srgb: [1.0,  0.55, 0.0 ], alpha: 1.0,
                role: CellRole::Structural,     upkeep_mult: 1.0,
            },
            CellType::BrownCell => CellTypeDef {
                tag: 6, label: "Brown",       srgb: [0.45, 0.27, 0.12], alpha: 1.0,
                role: CellRole::Structural,     upkeep_mult: 1.0,
            },
            // Blueish jelly: translucent, living tissue at half the upkeep.
            CellType::Gelly => CellTypeDef {
                tag: 7, label: "Gelly",       srgb: [0.3,  0.55, 0.95], alpha: 0.3,
                role: CellRole::Structural,     upkeep_mult: 0.5,
            },
            // ── Inert dead material: 10% of standard upkeep ──────────────────
            CellType::HardChitin => CellTypeDef {
                tag: 8, label: "Hard Chitin", srgb: [0.25, 0.13, 0.05], alpha: 1.0,
                role: CellRole::Inert,          upkeep_mult: 0.1,
            },
            CellType::SoftChitin => CellTypeDef {
                tag: 9, label: "Soft Chitin", srgb: [0.55, 0.36, 0.18], alpha: 1.0,
                role: CellRole::Inert,          upkeep_mult: 0.1,
            },
            CellType::Keratin => CellTypeDef {
                tag: 10, label: "Keratin",    srgb: [0.80, 0.68, 0.50], alpha: 1.0,
                role: CellRole::Inert,          upkeep_mult: 0.1,
            },
            CellType::HydroxylApatite => CellTypeDef {
                tag: 11, label: "Hydroxylapatite", srgb: [0.95, 0.95, 0.95], alpha: 1.0,
                role: CellRole::Inert,          upkeep_mult: 0.1,
            },
            // ── Red digestive-system tissue (living, standard upkeep) ────────
            // Tones run light → dark: Absorption [0.8,0.2,0.2] → BloodVessel →
            // Digestion [0.4,0.05,0.05].
            CellType::DigestionCell => CellTypeDef {
                tag: 12, label: "Digestion",  srgb: [0.22, 0.01, 0.01], alpha: 1.0,
                role: CellRole::Structural,     upkeep_mult: 1.0,
            },
            CellType::BloodVesselCell => CellTypeDef {
                tag: 13, label: "Blood Vessel", srgb: [0.6, 0.13, 0.13], alpha: 1.0,
                role: CellRole::Structural,     upkeep_mult: 1.0,
            },
            CellType::MuscleCell => CellTypeDef {
                tag: 14, label: "Muscle",     srgb: [0.8,  0.2,  0.2 ], alpha: 1.0,
                role: CellRole::Structural,     upkeep_mult: 1.0,
            },
        }
    }

    /// Behavioural role. See [`CellRole`].
    #[inline]
    pub const fn role(self) -> CellRole { self.def().role }

    /// Display label.
    #[inline]
    pub const fn label(self) -> &'static str { self.def().label }

    /// Display colour in sRGB — the single colour source for UI swatches.
    #[inline]
    pub const fn srgb(self) -> [f32; 3] { self.def().srgb }

    /// Opacity in `[0, 1]`; `< 1.0` means translucent. See [`CellType::is_translucent`].
    #[inline]
    pub const fn alpha(self) -> f32 { self.def().alpha }

    /// True when this type renders translucent (opacity `< 1.0`), so the body
    /// part's material must alpha-blend.
    #[inline]
    pub fn is_translucent(self) -> bool { self.alpha() < 1.0 }

    /// Per-cell upkeep as a multiple of `NON_PHOTO_CONSUMPTION_PER_CELL`.
    #[inline]
    pub const fn upkeep_mult(self) -> f32 { self.def().upkeep_mult }

    /// Stable serialisation tag. See [`CellTypeDef::tag`].
    #[inline]
    pub const fn tag(self) -> u8 { self.def().tag }

    /// Inverse of [`CellType::tag`]; `None` on an unknown byte (corrupt or
    /// newer-format file).
    pub fn from_tag(tag: u8) -> Option<CellType> {
        CellType::ALL.into_iter().find(|ct| ct.tag() == tag)
    }

    /// Linear-RGB colour for `Mesh::ATTRIBUTE_COLOR` (Bevy treats vertex colours
    /// as linear), derived from the sRGB display colour so there's one source.
    #[inline]
    pub fn color(self) -> [f32; 3] {
        let [r, g, b] = self.srgb();
        let lin = bevy::color::LinearRgba::from(bevy::color::Srgba::rgb(r, g, b));
        [lin.red, lin.green, lin.blue]
    }

    /// Linear-RGBA vertex colour — `color()` plus this type's `alpha()`.
    #[inline]
    pub fn color_rgba(self) -> [f32; 4] {
        let [r, g, b] = self.color();
        [r, g, b, self.alpha()]
    }

    /// True only for photosynthetic cells.
    #[inline]
    pub const fn is_photo(self) -> bool {
        matches!(self.role(), CellRole::Photosynthetic)
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

impl BodyPartKind {
    /// Stable `.colony`/`.species` serialisation tag
    /// (Body=0, Limb=1, Organ=2, Segment=3, Static=4).
    #[inline]
    pub fn to_tag(self) -> u8 {
        match self {
            BodyPartKind::Body    => 0,
            BodyPartKind::Limb    => 1,
            BodyPartKind::Organ   => 2,
            BodyPartKind::Segment => 3,
            BodyPartKind::Static  => 4,
        }
    }
    /// Inverse of [`BodyPartKind::to_tag`]; `None` on an unknown byte (corrupt
    /// or newer-format file). Mirrors the original reader's error-on-unknown.
    #[inline]
    pub fn from_tag(tag: u8) -> Option<BodyPartKind> {
        match tag {
            0 => Some(BodyPartKind::Body),
            1 => Some(BodyPartKind::Limb),
            2 => Some(BodyPartKind::Organ),
            3 => Some(BodyPartKind::Segment),
            4 => Some(BodyPartKind::Static),
            _ => None,
        }
    }
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
    /// When `false`, mesh-building and mutation skip this part (a
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
            // Everything that isn't a photo cell is non-photo body mass for upkeep.
            if c.cell_type.is_photo() { p += 1; } else { np += 1; }
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


