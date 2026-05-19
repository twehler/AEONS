// DNA encoding + distance metric.
//
// Every organism carries a fixed-length `Organism::dna: Vec<f32>` whose
// components are normalised to `[0, 1]`. The L¹/19 distance metric
// (mean absolute difference) thus returns a value in `[0, 1]` directly
// comparable to `simulation_settings::SPECIES_SEPARATION_THRESHOLD`.
//
// The encoding captures **the whole phenotype** in numeric form so the
// LineagesPlugin can mathematically compare organisms without needing
// to know about Bevy components, brain pools, or body-part trees:
//
//   Brain hyperparameter genes (6 dims) — populated for L2/L3 organisms
//     from each per-slot gene array; left at 0 for L0/L1/herbivore/
//     photoautotroph slots that don't have those genes.
//
//   Classification (3 dims) — intelligence level (normalised 0..1),
//     carnivore flag, photoautotroph flag. Inherited verbatim from
//     parent to offspring; can only change via species-editor placement.
//
//   Body plan (5 dims) — symmetry, has-variable-form, is-sessile,
//     normalised cell count, photo/total cell ratio. Reproduction's
//     mutation pipeline shifts cell count by 1 per birth and may swap
//     a cell's type, so these dims drift over generations.
//
//   Body geometry (5 dims) — per-axis extent of the cell cloud (XYZ),
//     bounding sphere radius, and a compactness measure (mean cell-
//     to-centroid distance divided by body radius). Drift in cell
//     placement (each growth step picks one of the lattice frontier
//     positions) reshapes these.
//
// Reproductive inheritance: the offspring's `dna` is initialised as a
// copy of the parent's (the `Organism::clone`-based reproduction
// already preserves it). The `sync_dna_from_phenotype` system overwrites
// the body-geometry / body-plan / classification slots from the
// organism's current state every tick, so any structural mutation
// shows up in the DNA within the next speciation interval.

use crate::organism::Organism;
use crate::simulation_settings::{
    L1_K_CURIOSITY_RANGE, L1_K_EAT_RANGE, L1_K_PROGRESS_RANGE,
    L1_LAMBDA_ENERGY_RANGE, L1_K_REPRO_RANGE, L1_SIGMA_RANGE,
};


/// Total dimension of the DNA vector. Six brain-gene slots + three
/// classification slots + five body-plan slots + five body-geometry
/// slots.
pub const DNA_DIM: usize = 19;

// ── Slot indices ────────────────────────────────────────────────────────────

// Brain hyperparameter genes (only populated for L2 / L3 organisms).
pub const D_SIGMA:          usize = 0;
pub const D_K_EAT:          usize = 1;
pub const D_K_REPRO:        usize = 2;
pub const D_LAMBDA_ENERGY:  usize = 3;
pub const D_K_CURIOSITY:    usize = 4;
pub const D_K_PROGRESS:     usize = 5;

// Classification.
const D_INTEL_LEVEL:        usize = 6;
const D_IS_CARNIVORE:       usize = 7;
const D_IS_PHOTO:           usize = 8;

// Body plan.
const D_SYMMETRY:           usize = 9;
const D_HAS_VARIABLE_FORM:  usize = 10;
const D_IS_SESSILE:         usize = 11;
const D_CELL_COUNT_NORM:    usize = 12;
const D_PHOTO_CELL_RATIO:   usize = 13;

// Body geometry.
const D_BODY_EXTENT_X:      usize = 14;
const D_BODY_EXTENT_Y:      usize = 15;
const D_BODY_EXTENT_Z:      usize = 16;
const D_BODY_RADIUS:        usize = 17;
const D_BODY_COMPACTNESS:   usize = 18;

/// Human-readable names for every DNA slot, indexed by the
/// `D_*` constants above. Used by the dataset export to emit
/// per-slot column headers (`dna_sigma`, `dna_k_eat`, …) so the
/// CSV stays self-describing without exposing the raw `dna: Vec<f32>`
/// as one opaque column.
pub const DNA_FIELD_NAMES: [&str; DNA_DIM] = [
    "sigma",
    "k_eat",
    "k_repro",
    "lambda_energy",
    "k_curiosity",
    "k_progress",
    "intel_level",
    "is_carnivore",
    "is_photo",
    "symmetry",
    "has_variable_form",
    "is_sessile",
    "cell_count_norm",
    "photo_cell_ratio",
    "body_extent_x",
    "body_extent_y",
    "body_extent_z",
    "body_radius",
    "body_compactness",
];

/// Maximum cells an organism can have — used to normalise
/// `D_CELL_COUNT_NORM`. Must stay in sync with
/// `volumetric_growth::MAX_CELLS`. Hard-coded here rather than
/// importing the constant so this module stays leaf-level (no
/// dependency on growth internals).
const MAX_CELLS_DNA_NORM: f32 = 60.0;

/// Scale used to normalise per-axis body extent and bounding radius
/// into `[0, 1]`. Corresponds to the longest plausible cell chain
/// (`MAX_CELLS × EDGE_LEN ≈ 60` units). Larger bodies clamp to 1.0.
const BODY_NORM_SCALE: f32 = 60.0;


/// Normalise a single brain-gene value into its `[0, 1]` slot. Returns
/// 0 (the "no gene yet" sentinel) when the range is degenerate.
#[inline]
pub fn normalise(value: f32, range: (f32, f32)) -> f32 {
    let (lo, hi) = range;
    let w = hi - lo;
    if w <= 0.0 { return 0.0; }
    ((value - lo) / w).clamp(0.0, 1.0)
}


/// Default DNA used at construction time before the first
/// `sync_dna_from_phenotype` tick. Length is correct so in-place
/// writes don't reallocate.
#[inline]
pub fn empty_dna() -> Vec<f32> {
    vec![0.0; DNA_DIM]
}


/// Compute the body-plan + body-geometry slots from an organism's
/// current state. Pure function: reads the Organism component only,
/// no ECS access. Called from `sync_dna_from_phenotype` and at
/// `spawn_organism` (so newborns have meaningful DNA from frame 1
/// without waiting for the first sync tick).
pub fn write_phenotype_dims(
    dna:          &mut [f32],
    organism:     &Organism,
    is_photo:     bool,
    is_carnivore: bool,
) {
    debug_assert_eq!(dna.len(), DNA_DIM);

    // ── Classification ───────────────────────────────────────────
    dna[D_INTEL_LEVEL] = match organism.intelligence_level {
        crate::organism::IntelligenceLevel::Level0 => 0.0,
        crate::organism::IntelligenceLevel::Level1 => 1.0 / 3.0,
        crate::organism::IntelligenceLevel::Level2 => 2.0 / 3.0,
        crate::organism::IntelligenceLevel::Level3 => 1.0,
    };
    dna[D_IS_CARNIVORE] = if is_carnivore { 1.0 } else { 0.0 };
    dna[D_IS_PHOTO]     = if is_photo     { 1.0 } else { 0.0 };

    // ── Body plan ────────────────────────────────────────────────
    dna[D_SYMMETRY] = match organism.symmetry {
        crate::organism::Symmetry::NoSymmetry => 0.0,
        crate::organism::Symmetry::Bilateral  => 1.0,
    };
    dna[D_HAS_VARIABLE_FORM] = if organism.has_variable_form { 1.0 } else { 0.0 };
    dna[D_IS_SESSILE]        = if organism.is_sessile        { 1.0 } else { 0.0 };

    let cell_count   = organism.grown_cell_count() as f32;
    let photo_cells  = organism.photo_cell_count.max(0) as f32;
    let total_cells  = cell_count.max(1.0); // avoid div-by-zero
    dna[D_CELL_COUNT_NORM]  = (cell_count / MAX_CELLS_DNA_NORM).clamp(0.0, 1.0);
    dna[D_PHOTO_CELL_RATIO] = (photo_cells / total_cells).clamp(0.0, 1.0);

    // ── Body geometry ────────────────────────────────────────────
    //
    // Walk every alive body part's cells once, compute per-axis
    // extents, centroid, and mean cell-to-centroid distance. Bounded
    // by MAX_CELLS = 30 per part × small body-parts count, so the
    // cost is well under a microsecond per organism.
    let mut max_x = 0.0_f32;
    let mut max_y = 0.0_f32;
    let mut max_z = 0.0_f32;
    let mut sum_pos = bevy::math::Vec3::ZERO;
    let mut n_cells = 0_u32;
    for bp in organism.body_parts.iter().filter(|b| b.is_alive()) {
        let bp_origin = bp.attachment.as_ref()
            .map(|a| a.origin_local)
            .unwrap_or(bevy::math::Vec3::ZERO);
        for cell in &bp.cells {
            let p = bp_origin + cell.local_pos;
            max_x = max_x.max(p.x.abs());
            max_y = max_y.max(p.y.abs());
            max_z = max_z.max(p.z.abs());
            sum_pos += p;
            n_cells += 1;
        }
    }

    dna[D_BODY_EXTENT_X] = (max_x / BODY_NORM_SCALE).clamp(0.0, 1.0);
    dna[D_BODY_EXTENT_Y] = (max_y / BODY_NORM_SCALE).clamp(0.0, 1.0);
    dna[D_BODY_EXTENT_Z] = (max_z / BODY_NORM_SCALE).clamp(0.0, 1.0);

    let body_radius = organism.bounding_radius();
    dna[D_BODY_RADIUS] = (body_radius / BODY_NORM_SCALE).clamp(0.0, 1.0);

    // Compactness: mean Euclidean distance of each cell from the
    // body's centroid, normalised by the bounding radius. 0 means
    // every cell sits at the centroid (impossible — at least one
    // cell exists), values near 1 mean cells are evenly spread out
    // toward the bounding sphere. Defensive against a single-cell
    // body where `body_radius` could be small enough to amplify
    // float noise: clamp to `[0, 1]`.
    if n_cells > 0 && body_radius > 1e-3 {
        let centroid = sum_pos / n_cells as f32;
        let mut sum_d = 0.0_f32;
        for bp in organism.body_parts.iter().filter(|b| b.is_alive()) {
            let bp_origin = bp.attachment.as_ref()
                .map(|a| a.origin_local)
                .unwrap_or(bevy::math::Vec3::ZERO);
            for cell in &bp.cells {
                let p = bp_origin + cell.local_pos;
                sum_d += (p - centroid).length();
            }
        }
        let mean_d = sum_d / n_cells as f32;
        dna[D_BODY_COMPACTNESS] = (mean_d / body_radius).clamp(0.0, 1.0);
    } else {
        dna[D_BODY_COMPACTNESS] = 0.0;
    }
}


/// Spawn-time helper. Returns an `empty_dna()` with only the
/// classification + body-plan-flag slots populated (the ones that
/// are knowable from the constructor arguments before the
/// `Organism` exists). The body-geometry slots (extents, radius,
/// compactness) and cell-count / photo-ratio slots are filled by
/// `sync_dna_from_phenotype` on the first frame after spawn — the
/// 1-frame lag is below the 1 Hz speciation tick so it doesn't
/// affect classification timing.
pub fn structural_dna(
    kind:               crate::organism::OrganismKind,
    symmetry:           crate::organism::Symmetry,
    has_variable_form:  bool,
    is_sessile:         bool,
    intelligence_level: crate::organism::IntelligenceLevel,
) -> Vec<f32> {
    let mut dna = empty_dna();
    dna[D_INTEL_LEVEL] = match intelligence_level {
        crate::organism::IntelligenceLevel::Level0 => 0.0,
        crate::organism::IntelligenceLevel::Level1 => 1.0 / 3.0,
        crate::organism::IntelligenceLevel::Level2 => 2.0 / 3.0,
        crate::organism::IntelligenceLevel::Level3 => 1.0,
    };
    dna[D_IS_PHOTO]     = match kind {
        crate::organism::OrganismKind::Photoautotroph => 1.0,
        crate::organism::OrganismKind::Heterotroph    => 0.0,
    };
    // is_carnivore unknown at construction time (Carnivore marker is
    // inserted by the editor's spawn pipeline AFTER spawn_organism
    // returns) — leave at 0 and let the sync tick correct it.
    dna[D_SYMMETRY] = match symmetry {
        crate::organism::Symmetry::NoSymmetry => 0.0,
        crate::organism::Symmetry::Bilateral  => 1.0,
    };
    dna[D_HAS_VARIABLE_FORM] = if has_variable_form { 1.0 } else { 0.0 };
    dna[D_IS_SESSILE]        = if is_sessile        { 1.0 } else { 0.0 };
    dna
}


/// Convenience: derive the trophic `OrganismKind` from an organism's
/// cached cell counts. Retained because some legacy callers still
/// rely on it.
#[allow(dead_code)]
pub fn kind_from_organism(organism: &Organism) -> crate::organism::OrganismKind {
    if organism.photo_cell_count > 0 && organism.non_photo_cell_count == 0 {
        crate::organism::OrganismKind::Photoautotroph
    } else {
        crate::organism::OrganismKind::Heterotroph
    }
}


/// Pack the six brain-gene values (relevant for L2 / L3 organisms)
/// into their DNA slots. Use the simulation-wide `L1_*_RANGE`
/// constants so the same range that bounds the gene at spawn is the
/// one that normalises it here. Idempotent.
pub fn write_l1_hetero_genes(
    dna:           &mut [f32],
    sigma:         f32,
    k_eat:         f32,
    k_repro:       f32,
    lambda_energy: f32,
    k_curiosity:   f32,
    k_progress:    f32,
) {
    debug_assert_eq!(dna.len(), DNA_DIM);
    dna[D_SIGMA]         = normalise(sigma,         L1_SIGMA_RANGE);
    dna[D_K_EAT]         = normalise(k_eat,         L1_K_EAT_RANGE);
    dna[D_K_REPRO]       = normalise(k_repro,       L1_K_REPRO_RANGE);
    dna[D_LAMBDA_ENERGY] = normalise(lambda_energy, L1_LAMBDA_ENERGY_RANGE);
    dna[D_K_CURIOSITY]   = normalise(k_curiosity,   L1_K_CURIOSITY_RANGE);
    dna[D_K_PROGRESS]    = normalise(k_progress,    L1_K_PROGRESS_RANGE);
}


/// Symmetric mean absolute difference between two DNA vectors. Both
/// must have length `DNA_DIM`; mismatched lengths return `f32::INFINITY`
/// so the caller treats them as completely different.
pub fn distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != DNA_DIM || b.len() != DNA_DIM {
        return f32::INFINITY;
    }
    let mut acc = 0.0_f32;
    for i in 0..DNA_DIM {
        acc += (a[i] - b[i]).abs();
    }
    acc / DNA_DIM as f32
}
