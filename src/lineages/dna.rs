// DNA encoding + distance metric.
//
// Every organism carries an `Organism::dna: Vec<f32>` of fixed length
// `DNA_DIM`. Each component is normalised to `[0, 1]` so the distance
// metric (mean absolute difference) returns a value in `[0, 1]` that
// can be compared against the speciation threshold (
// `simulation_settings::SPECIES_SEPARATION_THRESHOLD = 0.05`, i.e. "5%
// factor difference") directly.
//
// The vector is split in two halves:
//   * Structural / metabolic dims — set once at `spawn_organism` time
//     from the Organism's static fields. Never change after birth.
//   * L1 hetero brain genes — set lazily by `sync_dna_from_brain_pool`
//     (in `speciation.rs`) once the brain-pool's `assign_brains_l1_hetero`
//     has populated the per-slot gene arrays. For non-heterotroph
//     organisms (Photoautotrophs, Krishi) and L0 sessiles the gene
//     slots stay at 0.0, so they contribute nothing to inter-organism
//     distance.
//
// The encoding is intentionally lossless w.r.t. the simulation-relevant
// genes — every field of `Organism` that can differ between individuals
// AND can mutate between parent/offspring has a slot here. Things that
// can't ever differ (e.g. `OrganismRoot` markers) are excluded.

use crate::organism::{IntelligenceLevel, Organism, OrganismKind, Symmetry};
use crate::simulation_settings::{
    L1_K_CURIOSITY_RANGE, L1_K_EAT_RANGE, L1_K_PROGRESS_RANGE,
    L1_LAMBDA_ENERGY_RANGE, L1_K_REPRO_RANGE, L1_SIGMA_RANGE,
};


/// Total dimension of the DNA vector. 5 structural slots + 6 L1 hetero
/// brain-gene slots = 11.
pub const DNA_DIM: usize = 11;

// ── Slot indices ────────────────────────────────────────────────────────────
// Structural / metabolic — set at spawn, never mutated.
const D_SYMMETRY:           usize = 0;
const D_KIND:               usize = 1;
const D_HAS_VARIABLE_FORM:  usize = 2;
const D_IS_SESSILE:         usize = 3;
const D_INTELLIGENCE:       usize = 4;
// Per-organism brain genes (L1 hetero only) — written by the sync system.
pub const D_SIGMA:           usize = 5;
pub const D_K_EAT:           usize = 6;
pub const D_K_REPRO:         usize = 7;
pub const D_LAMBDA_ENERGY:   usize = 8;
pub const D_K_CURIOSITY:     usize = 9;
pub const D_K_PROGRESS:      usize = 10;


/// Encode the static (non-brain) half of the DNA vector. Brain-gene
/// slots stay at 0 — they get filled in by `sync_dna_from_brain_pool`
/// once the brain slot has been assigned.
pub fn structural_dna(
    kind:               OrganismKind,
    symmetry:           Symmetry,
    has_variable_form:  bool,
    is_sessile:         bool,
    intelligence_level: IntelligenceLevel,
) -> Vec<f32> {
    let mut dna = vec![0.0; DNA_DIM];
    dna[D_SYMMETRY] = match symmetry {
        Symmetry::NoSymmetry => 0.0,
        Symmetry::Bilateral  => 1.0,
    };
    dna[D_KIND] = match kind {
        OrganismKind::Photoautotroph => 0.0,
        OrganismKind::Heterotroph    => 1.0,
    };
    dna[D_HAS_VARIABLE_FORM] = if has_variable_form { 1.0 } else { 0.0 };
    dna[D_IS_SESSILE]        = if is_sessile        { 1.0 } else { 0.0 };
    dna[D_INTELLIGENCE]      = match intelligence_level {
        IntelligenceLevel::Level0 => 0.0,
        IntelligenceLevel::Level1 => 1.0 / 3.0,
        IntelligenceLevel::Level2 => 2.0 / 3.0,
        IntelligenceLevel::Level3 => 1.0,
    };
    dna
}


/// Normalise a single brain-gene value into its `[0, 1]` slot. Returns
/// 0 (the "no gene yet" sentinel) when the range is degenerate to
/// keep callers from dividing by zero.
#[inline]
pub fn normalise(value: f32, range: (f32, f32)) -> f32 {
    let (lo, hi) = range;
    let w = hi - lo;
    if w <= 0.0 { return 0.0; }
    ((value - lo) / w).clamp(0.0, 1.0)
}


/// Pack the six L1 hetero brain-gene values into the slots of an
/// existing DNA vector. Called by `sync_dna_from_brain_pool` after
/// the brain pool's `assign_brains_l1_hetero` has produced the per-
/// slot values. Idempotent — safe to call repeatedly.
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


/// Default DNA used as a placeholder when the speciation system has
/// no organism to read from yet (e.g. when `spawn_loaded_organism`
/// runs before the structural fields are known). Length is correct
/// so subsequent in-place writes don't have to reallocate.
#[inline]
pub fn empty_dna() -> Vec<f32> {
    vec![0.0; DNA_DIM]
}


/// Symmetric mean absolute difference between two DNA vectors. Both
/// must have length `DNA_DIM`; mismatched lengths return `f32::INFINITY`
/// so the caller treats them as completely different (defensive — this
/// shouldn't happen at runtime).
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


/// Convenience: derive the trophic `OrganismKind` from an organism's
/// cached cell counts. Used by the DNA encoder when it doesn't have
/// direct access to the trophic marker components. Photoautotrophs
/// have only Photo cells; heterotrophs have only NonPhoto cells.
/// Mixed-population edge cases (none expected in this codebase) fall
/// through to Heterotroph.
pub fn kind_from_organism(organism: &Organism) -> OrganismKind {
    if organism.photo_cell_count > 0 && organism.non_photo_cell_count == 0 {
        OrganismKind::Photoautotroph
    } else {
        OrganismKind::Heterotroph
    }
}
