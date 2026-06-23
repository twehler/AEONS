// Species registry — every species ever produced, plus ancestry edges.
//
// IDs are monotonic `u32`; species are never deleted (the Lineages view
// renders the full run history). The registry owns each species' running
// average DNA (centroid) for O(1) distance comparison in `speciation.rs`.

use bevy::prelude::*;

use crate::lineages::dna::{empty_dna, DNA_DIM};


/// One species — a stable identity organisms classify into. Created when an
/// organism diverges from every existing species (or seeds an empty registry).
/// Never removed.
#[derive(Debug, Clone)]
pub struct Species {
    pub id:         u32,
    /// User-visible label "Species N" (matches `id`). Cached as a `String`
    /// so the floating-label updater doesn't allocate every frame.
    pub name:       String,
    /// Species this split off from, or `None` for a lineage founder. The
    /// Lineages tree view edges this field.
    pub parent_id:  Option<u32>,
    /// Trimmed mean of current members' DNA (centroid). Recomputed each tick
    /// by `update_species_averages`, excluding outliers (> threshold from the
    /// previous mean) so one drifter can't drag it. Always `DNA_DIM` long.
    pub avg_dna:    Vec<f32>,
    /// Alive members classified here; recomputed each speciation tick.
    pub member_count: u32,
    /// `true` once `member_count` hit 0. Kept for the tree-of-life view but
    /// skipped when classifying new organisms.
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


/// Marker on organisms spawned from a `.species` import (Edit Colony). The
/// string is the filename stem. Read by `classify_organisms` to bypass the
/// nearest-centroid join — imports seed their own founder species (no parent
/// edge), so the tree shows the imported lineage as a separate tree.
#[derive(Component, Clone, Debug)]
pub struct ImportedSpeciesOrigin {
    pub name: String,
}


/// Global registry. Speciation is the sole writer; everyone else reads
/// `Res<SpeciesRegistry>`.
#[derive(Resource, Default)]
pub struct SpeciesRegistry {
    pub species: Vec<Species>,
    /// Last-issued id; next uses `next_id + 1` so names start at "Species 1".
    next_id: u32,
}

impl SpeciesRegistry {
    /// Append a new species seeded with `founder_dna`; returns its id.
    /// `parent_id` is the species this split from (`None` for a lineage founder).
    pub fn create(&mut self, founder_dna: Vec<f32>, parent_id: Option<u32>) -> u32 {
        self.next_id += 1;
        let id = self.next_id;
        let dna = if founder_dna.len() == DNA_DIM { founder_dna } else { empty_dna() };
        self.species.push(Species::founder(id, parent_id, dna));
        id
    }

    /// Create a species with an explicit display name (for `.species`
    /// imports, named after the source file). `parent_id` should be `None`
    /// for an import — a `Some` would tie it into an existing lineage.
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

    /// First non-extinct species with the given name. Lets multiple imports
    /// from one file collapse into a single species.
    pub fn find_alive_by_name(&self, name: &str) -> Option<&Species> {
        self.species.iter().find(|s| !s.extinct && s.name == name)
    }

    /// Lookup by id. `None` for unknown ids (shouldn't happen at runtime).
    pub fn get(&self, id: u32) -> Option<&Species> {
        self.species.iter().find(|s| s.id == id)
    }

    pub fn get_mut(&mut self, id: u32) -> Option<&mut Species> {
        self.species.iter_mut().find(|s| s.id == id)
    }

    /// Iterator over non-extinct species (nearest-centroid classification).
    pub fn alive_iter(&self) -> impl Iterator<Item = &Species> {
        self.species.iter().filter(|s| !s.extinct)
    }
}
