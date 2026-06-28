// Lineages — speciation + ancestry (data-domain half). The tree-of-life UI
// that consumes this registry lives in `frontend/tree_view.rs`.
//
// Submodules:
//   * dna.rs        — fixed-dim normalised genome encoding + distance metric.
//   * species.rs    — `SpeciesRegistry` resource + `Species`. Every species
//                     ever produced is kept (extinct ones flagged, for the tree).
//   * speciation.rs — 1 Hz tick: sync DNA, recompute trimmed-mean centroids,
//                     reclassify drifted organisms (forks set `parent_id` to
//                     the previous species, building ancestry edges).

pub mod dna;
pub mod speciation;
pub mod species;

use bevy::prelude::*;

use crate::lineages::speciation::{
    classify_organisms, sync_dna_from_phenotype, update_species_averages,
    SpeciationTimer,
};
use crate::lineages::species::SpeciesRegistry;


pub struct LineagesPlugin;

impl Plugin for LineagesPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<SpeciesRegistry>()
            .init_resource::<SpeciationTimer>()
            .add_systems(Update, (
                // Ordering is load-bearing: sync writes DNA → averages
                // consume updated DNA → classify consumes fresh centroids.
                // `chain()` ties them to one 1 Hz cadence: `sync_dna_from_phenotype`
                // (first) OWNS the `SpeciationTimer` tick; the other two read
                // `just_finished()`. All three early-return on off-tick frames, so
                // the per-organism DNA walk no longer runs every frame.
                sync_dna_from_phenotype,
                update_species_averages,
                classify_organisms,
            ).chain());
    }
}
