// Speciation tick — assigns / reassigns species ids based on DNA
// distance to species centroids.
//
// Runs at ~1 Hz (`SPECIATION_TICK_SECS`) — speciation is a slow
// process, the tick rate just has to be small compared with the
// reproduction cadence so newly-born organisms get reclassified
// before they themselves reproduce.
//
// Three sub-steps in order:
//   1. `sync_dna_from_phenotype` — fills in the L1 hetero brain-gene
//      slots of each Organism's DNA vector. The structural slots are
//      already populated by `spawn_organism`.
//   2. `update_species_averages` — recomputes every alive species'
//      `avg_dna` as a trimmed mean over its members (drops any
//      member further than `SPECIES_SEPARATION_THRESHOLD` from the previous
//      tick's mean).
//   3. `classify_organisms` — for each organism:
//        * No species_id ⇒ assign nearest species within threshold,
//          or create a new species if no fit exists.
//        * Has species_id but DNA has drifted > threshold from its
//          species' centroid ⇒ same flow as above; the new species'
//          parent_id is recorded as the diverged organism's previous
//          species_id (= the lineage we forked from).
//
// Extinction: any species with zero alive members at the end of a
// tick is marked `extinct = true`. Extinct species are skipped by
// classification but persisted for the tree-of-life view.

use bevy::prelude::*;

use crate::intelligence_level_2_sliding::{BrainPoolL2, BrainSlotL2};
use crate::intelligence_level_3_sliding::{BrainPoolL3, BrainSlotL3};
use crate::lineages::dna::{
    distance, empty_dna, write_l1_hetero_genes, write_phenotype_dims, DNA_DIM,
};
use crate::lineages::species::{ImportedSpeciesOrigin, SpeciesRegistry};
use crate::organism::Organism;
use crate::simulation_settings::SPECIES_SEPARATION_THRESHOLD;


/// Cadence in real seconds. Speciation isn't latency-sensitive and
/// the tick walks every organism in the world, so 1 Hz is plenty.
const SPECIATION_TICK_SECS: f32 = 1.0;


#[derive(Resource)]
pub struct SpeciationTimer(pub Timer);

impl Default for SpeciationTimer {
    fn default() -> Self {
        Self(Timer::from_seconds(SPECIATION_TICK_SECS, TimerMode::Repeating))
    }
}


// ── DNA sync (full phenotype) ───────────────────────────────────────────────
//
// Runs on every organism every frame. Reads the organism's current
// state (body parts, classification flags, intelligence level) AND
// any per-slot brain genes from the L2 / L3 pools when applicable.
// This is the canonical pipeline that keeps `Organism::dna` in sync
// with everything the lineage system compares.
//
// The brain-gene slots only populate for organisms that actually
// carry brain hyperparameters — currently that's L2 / L3 organisms.
// L0 sessiles, L1 photoautotrophs, and L1 herbivores have no gene
// vector and leave those slots at 0, so they cluster on body-plan +
// classification features alone.

pub fn sync_dna_from_phenotype(
    pool_l2:   Option<NonSend<BrainPoolL2>>,
    pool_l3:   Option<NonSend<BrainPoolL3>>,
    mut q:     Query<(
        &mut Organism,
        Has<crate::colony::Photoautotroph>,
        Has<crate::colony::Carnivore>,
        Option<&BrainSlotL2>,
        Option<&BrainSlotL3>,
    )>,
) {
    for (mut organism, is_photo, is_carnivore, slot_l2, slot_l3) in &mut q {
        // Resize on demand — newborns spawn with `empty_dna()` of the
        // right length, but defensive against any caller that
        // constructed an Organism with an empty Vec.
        if organism.dna.len() != DNA_DIM {
            organism.dna = empty_dna();
        }

        // ── Body plan + classification + body geometry (always written) ──
        //
        // Split the borrow: take the mutable reference to `dna` out
        // of the Organism via a temporary swap with an empty Vec,
        // then pass it alongside an immutable `&Organism` to
        // `write_phenotype_dims`. Bevy's `Mut<Organism>` exposes
        // `into_inner()` but we want change-detection unchanged, so
        // a swap-and-restore is the cleanest split. Allocation-free:
        // the empty Vec lives for two statements.
        let mut dna_buf = std::mem::take(&mut organism.dna);
        write_phenotype_dims(&mut dna_buf, &organism, is_photo, is_carnivore);
        organism.dna = dna_buf;

        // ── Brain genes: only for organisms enrolled in L2 / L3 ──
        // (L1 herbivore is supervised and has no gene vector; the
        // L1 photo brain's `slot_yaw` isn't a hyperparameter we
        // currently track in the DNA.)
        if let (Some(slot), Some(pool)) = (slot_l2, pool_l2.as_deref()) {
            let s = slot.0 as usize;
            if s < pool.sigma.len() {
                write_l1_hetero_genes(
                    &mut organism.dna,
                    pool.sigma[s],
                    pool.k_eat[s],
                    pool.k_repro[s],
                    pool.lambda_energy[s],
                    pool.k_curiosity[s],
                    pool.k_progress[s],
                );
            }
        } else if let (Some(slot), Some(pool)) = (slot_l3, pool_l3.as_deref()) {
            let s = slot.0 as usize;
            if s < pool.sigma.len() {
                write_l1_hetero_genes(
                    &mut organism.dna,
                    pool.sigma[s],
                    pool.k_eat[s],
                    pool.k_repro[s],
                    pool.lambda_energy[s],
                    pool.k_curiosity[s],
                    pool.k_progress[s],
                );
            }
        }
    }
}


// ── Species-average recompute (trimmed mean) ────────────────────────────────

pub fn update_species_averages(
    mut registry: ResMut<SpeciesRegistry>,
    organisms:    Query<&Organism>,
    time:         Res<Time<Real>>,
    mut timer:    ResMut<SpeciationTimer>,
) {
    if !timer.0.tick(time.delta()).just_finished() { return; }
    if registry.species.is_empty() { return; }

    // Two-pass trimmed mean: pass 1 finds the simple mean of every
    // assigned member, pass 2 averages only members within
    // SPECIES_SEPARATION_THRESHOLD of that simple mean. Excludes outliers
    // so a single drifting individual can't drag the whole centroid
    // with it.
    //
    // We bucket by species_id once, then process each species.
    let mut buckets: Vec<Vec<Vec<f32>>> = vec![Vec::new(); registry.species.len()];
    let id_to_idx: std::collections::HashMap<u32, usize> =
        registry.species.iter().enumerate().map(|(i, s)| (s.id, i)).collect();

    for org in &organisms {
        if let Some(id) = org.species_id {
            if let Some(&idx) = id_to_idx.get(&id) {
                if org.dna.len() == DNA_DIM {
                    buckets[idx].push(org.dna.clone());
                }
            }
        }
    }

    for (idx, dnas) in buckets.iter().enumerate() {
        let species = &mut registry.species[idx];
        if dnas.is_empty() {
            // Last member died — preserve the historical centroid
            // (for the tree view), flag the species extinct, and
            // move on.
            species.member_count = 0;
            species.extinct      = true;
            continue;
        }
        species.member_count = dnas.len() as u32;

        // ── Pass 1: simple mean ─────────────────────────────────
        let simple = mean_of(dnas);

        // ── Pass 2: trimmed mean (only members within threshold of
        //   simple mean count). If everyone is outside the threshold
        //   (degenerate — shouldn't happen normally), fall back to
        //   the simple mean so we never lose the centroid.
        let mut kept: Vec<&Vec<f32>> = dnas.iter()
            .filter(|d| distance(d, &simple) <= SPECIES_SEPARATION_THRESHOLD)
            .collect();
        if kept.is_empty() {
            species.avg_dna = simple;
            continue;
        }
        // Borrow-by-reference shuffle to avoid cloning.
        let mut acc = vec![0.0f32; DNA_DIM];
        for d in &kept {
            for i in 0..DNA_DIM { acc[i] += d[i]; }
        }
        let n = kept.len() as f32;
        for i in 0..DNA_DIM { acc[i] /= n; }
        species.avg_dna = acc;
        // (`kept` only existed for the borrow lifetime; drop is
        // implicit — explicit `.clear()` would be a no-op.)
        kept.clear();
    }
}


fn mean_of(dnas: &[Vec<f32>]) -> Vec<f32> {
    let n = dnas.len() as f32;
    let mut acc = vec![0.0f32; DNA_DIM];
    for d in dnas {
        for i in 0..DNA_DIM { acc[i] += d[i]; }
    }
    for i in 0..DNA_DIM { acc[i] /= n; }
    acc
}


// ── Per-organism classification ─────────────────────────────────────────────

pub fn classify_organisms(
    mut registry: ResMut<SpeciesRegistry>,
    mut commands: Commands,
    mut q:        Query<(Entity, &mut Organism, Option<&ImportedSpeciesOrigin>)>,
    timer:        Res<SpeciationTimer>,
) {
    // Piggy-back on `update_species_averages`'s timer — both fire
    // on the same tick, but classification needs the freshly-
    // updated averages, so it runs immediately after via system
    // ordering (`.chain()`). Use the same `just_finished` query so
    // off-tick frames are cheap.
    if !timer.0.just_finished() { return; }

    for (entity, mut org, imported) in &mut q {
        if org.dna.len() != DNA_DIM { continue; }

        // ── Imported-species short-circuit ────────────────────────
        // If the organism was spawned from a `.species` import and
        // hasn't been classified yet, route it into a name-keyed
        // founder species (parent = None, so it sits in its own
        // tree besides the main lineages). Multiple imports from the
        // same file collapse into one species. After assignment we
        // strip the marker so subsequent ticks run the normal drift
        // logic on the imported organism — its descendants can fork
        // off sub-species through the standard pipeline.
        if let (Some(origin), None) = (imported, org.species_id) {
            let species_id = match registry.find_alive_by_name(&origin.name) {
                Some(s) => s.id,
                None    => registry.create_with_name(
                    origin.name.clone(),
                    org.dna.clone(),
                    None,
                ),
            };
            org.species_id = Some(species_id);
            commands.entity(entity).try_remove::<ImportedSpeciesOrigin>();
            continue;
        }

        // Find the nearest alive species + its distance.
        let (nearest_id, nearest_dist) = nearest_species(&registry, &org.dna);

        match org.species_id {
            None => {
                // Unassigned — first-time classification.
                //
                // The user's 5% rule applies to OFFSPRING drifting
                // away from their parent species' centroid, not to
                // initial classification of organisms that have
                // never had a species. Without this asymmetry the
                // initial cohort — whose L1 brain genes are sampled
                // uniformly from the full `L1_*_RANGE`s, so
                // per-individual distances are already well above
                // 5% — explodes into N singleton species, none
                // connected to any other, because each becomes its
                // own unparented founder.
                //
                // So: when nothing is classified yet, the FIRST
                // unclassified organism founds Species 1 (no
                // existing species to join). Every subsequent
                // unclassified organism joins its closest existing
                // species regardless of distance. The trimmed-mean
                // centroid recompute next tick excludes the
                // outliers from the centroid, and the
                // `Some(current)` branch below then forks them off
                // into child species with a proper `parent_id`
                // edge.
                match nearest_id {
                    Some(id) => org.species_id = Some(id),
                    None     => {
                        let new_id = registry.create(org.dna.clone(), None);
                        org.species_id = Some(new_id);
                    }
                }
            }
            Some(current) => {
                // Currently classified — check drift against own
                // species' centroid.
                let own_dist = registry
                    .get(current)
                    .map(|s| distance(&org.dna, &s.avg_dna))
                    .unwrap_or(f32::INFINITY);
                if own_dist <= SPECIES_SEPARATION_THRESHOLD {
                    // Still a good fit — leave it alone.
                    continue;
                }
                // Drifted out — does another existing species fit
                // better? If yes, hop over (no new lineage edge —
                // the organism is just re-classified into an
                // existing species). If no, fork a new species,
                // recording the current species as the parent of
                // the new branch.
                if nearest_dist <= SPECIES_SEPARATION_THRESHOLD && nearest_id != Some(current) {
                    org.species_id = nearest_id;
                } else {
                    let new_id = registry.create(org.dna.clone(), Some(current));
                    org.species_id = Some(new_id);
                }
            }
        }
    }
}


/// Find the closest alive species to `dna`. Returns `(None,
/// f32::INFINITY)` when the registry is empty (or only contains
/// extinct entries) — callers interpret this as "create a brand-new
/// founder species".
fn nearest_species(registry: &SpeciesRegistry, dna: &[f32]) -> (Option<u32>, f32) {
    let mut best_id:   Option<u32> = None;
    let mut best_dist: f32         = f32::INFINITY;
    for s in registry.alive_iter() {
        let d = distance(dna, &s.avg_dna);
        if d < best_dist {
            best_dist = d;
            best_id   = Some(s.id);
        }
    }
    (best_id, best_dist)
}
