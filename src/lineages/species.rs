// Species registry — the canonical list of every species the
// simulation has ever produced, plus their ancestry edges.
//
// IDs are monotonically increasing (`u32`); a species is never
// deleted, even after its last member dies, because the
// "Lineages" window mode renders the full ancestry tree across
// the entire run.
//
// The registry owns each species' running average DNA so the
// speciation system in `speciation.rs` can compare a fresh
// organism's DNA against a species' centroid in O(1) — the
// distance metric is mean-absolute-difference, see
// `dna::distance`.

use bevy::prelude::*;

use crate::lineages::dna::{empty_dna, DNA_DIM};


/// One species — a stable identity that organisms can be classified
/// into. Created on the first organism that diverges from every
/// existing species (or on the very first speciation tick when the
/// registry is empty). Never removed.
#[derive(Debug, Clone)]
pub struct Species {
    pub id:         u32,
    /// User-visible label, "Species 1", "Species 2", ... — matches
    /// `id`. Cached as a `String` so the floating-label updater
    /// doesn't allocate every frame.
    pub name:       String,
    /// Species this one split off from, or `None` if it was the
    /// initial founder of a lineage (no prior species existed when
    /// it was created). The Lineages tree view edges this field.
    pub parent_id:  Option<u32>,
    /// Trimmed mean of all current members' DNA vectors. Recomputed
    /// at every speciation tick by `update_species_averages` —
    /// outliers (members > SPECIES_SEPARATION_THRESHOLD from the previous
    /// tick's mean) are excluded from the mean so a single drifting
    /// individual doesn't pull the whole centroid with it. Length
    /// is always `DNA_DIM`.
    pub avg_dna:    Vec<f32>,
    /// Number of currently-alive organisms classified into this
    /// species. Recomputed every speciation tick.
    pub member_count: u32,
    /// `true` once `member_count` hit 0 at any point — the species
    /// went extinct. We keep the entry for the tree-of-life view
    /// (extinct branches are still part of the history) but skip it
    /// when classifying new organisms.
    pub extinct:    bool,
}

impl Species {
    fn founder(id: u32, parent_id: Option<u32>, dna: Vec<f32>) -> Self {
        Self {
            id,
            name:         format!("Species {id}"),
            parent_id,
            avg_dna:      dna,
            member_count: 1,
            extinct:      false,
        }
    }

    fn founder_named(id: u32, name: String, parent_id: Option<u32>, dna: Vec<f32>) -> Self {
        Self {
            id,
            name,
            parent_id,
            avg_dna:      dna,
            member_count: 1,
            extinct:      false,
        }
    }
}


/// Marker on organisms that were spawned from a `.species` import in
/// merged-mode (Edit Colony). The string is the filename stem (no
/// extension), e.g. `"herbivore1"` for `herbivore1.species`. Read once
/// by `lineages::speciation::classify_organisms` to bypass the normal
/// nearest-centroid join — imported organisms always seed their own
/// founder species (no parent edge), so the tree-of-life view shows
/// the imported lineage as a separate tree beside the simulation's own
/// evolutionary history.
#[derive(Component, Clone, Debug)]
pub struct ImportedSpeciesOrigin {
    pub name: String,
}


/// Global registry. `Resource` — every system that needs to look up
/// a species name or walk the ancestry tree reads this. The
/// speciation system is the sole writer; everyone else holds
/// `Res<SpeciesRegistry>`.
#[derive(Resource, Default)]
pub struct SpeciesRegistry {
    pub species: Vec<Species>,
    /// Last-issued id. The next species created uses
    /// `next_id + 1` so user-facing names start at "Species 1".
    next_id: u32,
}

impl SpeciesRegistry {
    /// Append a brand-new species seeded with `founder_dna`. Returns
    /// the new id. `parent_id` is the id of the species this one
    /// split off from (or `None` if this is the first species in
    /// its lineage — e.g. the very first organism the speciation
    /// system has ever seen).
    pub fn create(&mut self, founder_dna: Vec<f32>, parent_id: Option<u32>) -> u32 {
        self.next_id += 1;
        let id = self.next_id;
        let dna = if founder_dna.len() == DNA_DIM { founder_dna } else { empty_dna() };
        self.species.push(Species::founder(id, parent_id, dna));
        id
    }

    /// Create a brand-new species with an explicit display name (used
    /// for `.species` imports — the imported lineage is named after
    /// the source filename, not auto-numbered). `parent_id` should
    /// normally be `None` for an import; a `Some` value would tie the
    /// imported lineage to an existing species, which defeats the
    /// "fresh tree beside the main one" semantics.
    pub fn create_with_name(
        &mut self,
        name:        String,
        founder_dna: Vec<f32>,
        parent_id:   Option<u32>,
    ) -> u32 {
        self.next_id += 1;
        let id = self.next_id;
        let dna = if founder_dna.len() == DNA_DIM { founder_dna } else { empty_dna() };
        self.species.push(Species::founder_named(id, name, parent_id, dna));
        id
    }

    /// Find the first non-extinct species with the given display name.
    /// Used by the `.species`-import classification path so multiple
    /// instances loaded from the same file collapse into one species
    /// rather than spawning a separate founder for each click.
    pub fn find_alive_by_name(&self, name: &str) -> Option<&Species> {
        self.species.iter().find(|s| !s.extinct && s.name == name)
    }

    /// Lookup by id. Returns `None` for unknown ids (shouldn't
    /// happen at runtime — every `Organism::species_id` was issued
    /// by `create`).
    pub fn get(&self, id: u32) -> Option<&Species> {
        self.species.iter().find(|s| s.id == id)
    }

    pub fn get_mut(&mut self, id: u32) -> Option<&mut Species> {
        self.species.iter_mut().find(|s| s.id == id)
    }

    /// Iterator over every species that's still classifying live
    /// members. Used by `speciation::classify_organisms` to find
    /// the nearest non-extinct species for an unassigned organism.
    pub fn alive_iter(&self) -> impl Iterator<Item = &Species> {
        self.species.iter().filter(|s| !s.extinct)
    }
}
