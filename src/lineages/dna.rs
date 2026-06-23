// DNA encoding + distance metric.
//
// Each organism carries `Organism::dna: Vec<f32>` of components normalised
// to `[0, 1]`. The mean-absolute-difference metric returns `[0, 1]`,
// directly comparable to `SPECIES_SEPARATION_THRESHOLD`. The encoding
// captures the whole phenotype numerically so LineagesPlugin can compare
// organisms without touching Bevy components / brain pools / body-part trees:
//
//   Brain hyperparameter genes (6) — populated for L2/L3 only; 0 otherwise.
//   Classification (3) — intelligence level, carnivore, photo flags.
//     Inherited verbatim; changes only via species-editor placement.
//   Body plan (5) — symmetry, has-variable-form, is-sessile, normalised
//     cell count, photo/total ratio. Drift via mutation each birth.
//   Body geometry (5) — per-axis cell-cloud extent (XYZ), bounding radius,
//     compactness (mean cell-to-centroid dist / body radius).
//
// Inheritance: offspring `dna` is a copy of the parent's;
// `sync_dna_from_phenotype` overwrites the structural slots from current
// state each tick, so mutations show up within the next speciation interval.

use crate::organism::Organism;
use crate::simulation_settings::{
    L1_K_CURIOSITY_RANGE, L1_K_EAT_RANGE, L1_K_PROGRESS_RANGE,
    L1_LAMBDA_ENERGY_RANGE, L1_K_REPRO_RANGE, L1_SIGMA_RANGE,
};


/// Total DNA dimension: 6 brain-gene + 3 classification + 5 body-plan +
/// 5 body-geometry slots.
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

/// Human-readable names per DNA slot (indexed by the `D_*` constants).
/// Used by dataset export for per-slot CSV column headers.
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

/// Normaliser for `D_CELL_COUNT_NORM`. Keep in sync with
/// `volumetric_growth::MAX_CELLS`; hard-coded so this module stays leaf-level.
const MAX_CELLS_DNA_NORM: f32 = 60.0;

/// Scale normalising per-axis extent + bounding radius into `[0, 1]`
/// (≈ longest plausible cell chain). Larger bodies clamp to 1.0.
const BODY_NORM_SCALE: f32 = 60.0;


/// Normalise a brain-gene value into `[0, 1]`. Returns 0 (the "no gene"
/// sentinel) when the range is degenerate.
#[inline]
pub fn normalise(value: f32, range: (f32, f32)) -> f32 {
    let (lo, hi) = range;
    let w = hi - lo;
    if w <= 0.0 { return 0.0; }
    ((value - lo) / w).clamp(0.0, 1.0)
}


/// Default DNA before the first `sync_dna_from_phenotype` tick. Correct
/// length so in-place writes don't reallocate.
#[inline]
pub fn empty_dna() -> Vec<f32> {
    vec![0.0; DNA_DIM]
}


/// Compute the body-plan + body-geometry slots from an organism's current
/// state. Pure (reads Organism only). Called from `sync_dna_from_phenotype`
/// and `spawn_organism` (so newborns have meaningful DNA from frame 1).
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
    // Walk alive cells once for per-axis extents, centroid, and mean
    // cell-to-centroid distance. Bounded (~MAX_CELLS per part), sub-µs.
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

    // Compactness: mean cell-to-centroid distance / bounding radius.
    // Near 0 = cells clustered at centroid, near 1 = spread to the
    // sphere. Clamp to `[0, 1]` (guards single-cell float noise).
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


/// Spawn-time helper: `empty_dna()` with only the classification + body-
/// plan-flag slots populated (knowable before the `Organism` exists). The
/// geometry / cell-count slots are filled by `sync_dna_from_phenotype` on
/// the first frame — the 1-frame lag is below the 1 Hz speciation tick.
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
    // is_carnivore unknown here (Carnivore marker inserted after
    // spawn_organism returns) — leave 0; the sync tick corrects it.
    dna[D_SYMMETRY] = match symmetry {
        crate::organism::Symmetry::NoSymmetry => 0.0,
        crate::organism::Symmetry::Bilateral  => 1.0,
    };
    dna[D_HAS_VARIABLE_FORM] = if has_variable_form { 1.0 } else { 0.0 };
    dna[D_IS_SESSILE]        = if is_sessile        { 1.0 } else { 0.0 };
    dna
}


/// Derive the trophic `OrganismKind` from cached cell counts.
#[allow(dead_code)]
pub fn kind_from_organism(organism: &Organism) -> crate::organism::OrganismKind {
    if organism.photo_cell_count > 0 && organism.non_photo_cell_count == 0 {
        crate::organism::OrganismKind::Photoautotroph
    } else {
        crate::organism::OrganismKind::Heterotroph
    }
}


/// Pack the six brain-gene values (L2/L3) into their DNA slots, normalised
/// by the same `L1_*_RANGE` constants that bound them at spawn. Idempotent.
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


/// Mean absolute difference between two DNA vectors. Mismatched lengths
/// return `f32::INFINITY` (treated as completely different).
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
