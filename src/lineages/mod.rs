// Lineages — speciation + ancestry, the data-domain half. The
// tree-of-life UI that consumes this registry lives under
// `frontend/tree_view.rs` and is registered by `FrontendPlugin`,
// keeping the UI / domain split clean.
//
// Submodule layout:
//   * dna.rs        — fixed-dim normalised genome encoding +
//                     distance metric. Computed on the `Organism`
//                     struct itself.
//   * species.rs    — `SpeciesRegistry` resource, `Species` struct.
//                     Every species the simulation has ever produced
//                     stays in the registry (extinct ones are
//                     flagged but kept for the tree view).
//   * speciation.rs — 1 Hz tick that syncs DNA from the L1 hetero
//                     brain pool, recomputes species centroids
//                     (trimmed mean — outliers excluded), and
//                     reclassifies organisms whose DNA has drifted
//                     past the threshold. New species are forked
//                     with their `parent_id` set to the diverging
//                     organism's previous species, building the
//                     ancestry tree edges.

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
                // Ordering is load-bearing: the DNA sync writes
                // brain genes into `Organism::dna`, then
                // `update_species_averages` consumes the updated
                // DNAs, then `classify_organisms` consumes the
                // freshly-recomputed centroids. `chain()` ties them
                // to a single 1 Hz cadence (the timer is owned by
                // `update_species_averages` and queried by
                // `classify_organisms` via the same `Res`).
                sync_dna_from_phenotype,
                update_species_averages,
                classify_organisms,
            ).chain());
    }
}
