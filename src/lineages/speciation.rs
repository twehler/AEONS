// Speciation tick — assigns / reassigns species ids based on DNA
// distance to species centroids.
//
// Runs at ~1 Hz (`SPECIATION_TICK_SECS`) — speciation is a slow
// process, the tick rate just has to be small compared with the
// reproduction cadence so newly-born organisms get reclassified
// before they themselves reproduce.
//
// Three sub-steps in order:
//   1. `sync_dna_from_brain_pool` — fills in the L1 hetero brain-gene
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

use crate::intelligence_level_1_hetero::{BrainPoolL1Hetero, BrainSlotL1Hetero};
use crate::lineages::dna::{
    distance, write_l1_hetero_genes, DNA_DIM,
};
use crate::lineages::species::SpeciesRegistry;
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


// ── DNA sync (from brain pool) ──────────────────────────────────────────────
//
// The Organism's DNA structural slots are populated at spawn time;
// the six brain-gene slots are populated here once the brain pool
// has assigned a slot to the organism. For organisms without a
// BrainSlotL1Hetero (photoautotrophs, sessiles), the gene slots stay
// at 0.0 — they contribute nothing to inter-organism distance.

pub fn sync_dna_from_brain_pool(
    pool:      Option<NonSend<BrainPoolL1Hetero>>,
    mut q:     Query<(&mut Organism, &BrainSlotL1Hetero)>,
) {
    let Some(pool) = pool else { return };
    for (mut organism, slot) in &mut q {
        let s = slot.0 as usize;
        // Defensive bounds check — the brain pool's per-slot vecs
        // are length `OrganismPoolSize`, slot indices are stamped
        // by `assign_brains_l1_hetero` which clamps to that. A
        // length mismatch would mean a torn invariant.
        if s >= pool.sigma.len() { continue; }
        // Resize on demand — fresh organisms created at spawn time
        // have the right length, but defensive against any caller
        // that constructed an Organism with an empty Vec.
        if organism.dna.len() != DNA_DIM {
            organism.dna = crate::lineages::dna::empty_dna();
        }
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
    mut q:        Query<&mut Organism>,
    timer:        Res<SpeciationTimer>,
) {
    // Piggy-back on `update_species_averages`'s timer — both fire
    // on the same tick, but classification needs the freshly-
    // updated averages, so it runs immediately after via system
    // ordering (`.chain()`). Use the same `just_finished` query so
    // off-tick frames are cheap.
    if !timer.0.just_finished() { return; }

    for mut org in &mut q {
        if org.dna.len() != DNA_DIM { continue; }

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
