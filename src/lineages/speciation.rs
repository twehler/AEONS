// Speciation tick — assigns/reassigns species ids by DNA distance to
// centroids. Runs at ~1 Hz (must just be fast vs. reproduction cadence so
// newborns are reclassified before they reproduce).
//
// Three ordered sub-steps:
//   1. `sync_dna_from_phenotype` — fill brain-gene slots (structural slots
//      already populated by `spawn_organism`).
//   2. `update_species_averages` — recompute each alive species' centroid as
//      a trimmed mean (drops members > threshold from the previous mean).
//   3. `classify_organisms` — unassigned ⇒ nearest species within threshold
//      else new; drifted past threshold ⇒ same flow, new species' parent_id =
//      the previous species (the forked lineage).
//
// Extinction: zero-member species are flagged extinct (skipped by
// classification, persisted for the tree view).

use bevy::prelude::*;

use crate::intelligence_level_2_sliding::{BrainPoolL2, BrainSlotL2};
use crate::intelligence_level_3_sliding::{BrainPoolL3, BrainSlotL3};
use crate::lineages::dna::{
    distance, empty_dna, write_l1_hetero_genes, write_phenotype_dims, DNA_DIM,
};
use crate::lineages::species::{ImportedSpeciesOrigin, SpeciesRegistry};
use crate::organism::Organism;
use crate::simulation_settings::SPECIES_SEPARATION_THRESHOLD;


/// Cadence in real seconds. 1 Hz — not latency-sensitive, walks all organisms.
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
// Syncs `Organism::dna` from current state + L2/L3 brain genes. Brain-gene slots
// populate only for L2/L3; L0 sessiles, L1 photos, and L1 herbivores leave them 0
// and cluster on body-plan + classification features alone.
//
// Gated to the `SpeciationTimer` (~1 Hz) like the rest of the pipeline — its only
// consumers are the 1 Hz `update_species_averages`/`classify_organisms`, so the
// per-organism two-pass cell walk in `write_phenotype_dims` ran ~60× more often
// than it could matter. This is the FIRST chained system, so it OWNS the tick;
// the other two read `just_finished()` without re-ticking.

pub fn sync_dna_from_phenotype(
    time:      Res<Time<Real>>,
    mut timer: ResMut<SpeciationTimer>,
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
    if !timer.0.tick(time.delta()).just_finished() { return; }

    // ── Pass 1 (parallel): body plan + classification + body geometry ──
    // `write_phenotype_dims` is pure (reads `&Organism`, writes only that
    // organism's own `dna`) — entity-disjoint and free of NonSend access, so the
    // heavy per-cell geometry walk fans out over `ComputeTaskPool`. This is the
    // expensive part of the system.
    q.par_iter_mut().for_each(|(mut organism, is_photo, is_carnivore, _slot_l2, _slot_l3)| {
        // Resize on demand — defensive against an Organism built with an empty Vec.
        if organism.dna.len() != DNA_DIM {
            organism.dna = empty_dna();
        }
        // Swap `dna` out to split the borrow (pass &mut dna alongside
        // &Organism to `write_phenotype_dims`), then restore — keeps change-
        // detection intact and is allocation-free.
        let mut dna_buf = std::mem::take(&mut organism.dna);
        write_phenotype_dims(&mut dna_buf, &organism, is_photo, is_carnivore);
        organism.dna = dna_buf;
    });

    // ── Pass 2 (serial): L2/L3 brain-gene overlay ──
    // Reads the NonSend `BrainPoolL2/L3` (CUDA-pinned), which CANNOT enter a
    // parallel closure — so this stays serial. Only L2/L3 organisms carry a slot;
    // every other row is a cheap Option check that never touches `Organism`.
    for (mut organism, _is_photo, _is_carnivore, slot_l2, slot_l3) in &mut q {
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
    timer:        Res<SpeciationTimer>,
) {
    // The timer is ticked by `sync_dna_from_phenotype` (first in the chain); we
    // only read whether it fired this frame.
    if !timer.0.just_finished() { return; }
    if registry.species.is_empty() { return; }

    // Two-pass trimmed mean: pass 1 = simple mean of all members, pass 2 =
    // mean of only those within threshold of it (excludes outliers so a
    // drifter can't drag the centroid). Bucket organisms' DNA *by reference*
    // (no per-organism clone of the 19-float genome) into per-species lists.
    let id_to_idx: std::collections::HashMap<u32, usize> =
        registry.species.iter().enumerate().map(|(i, s)| (s.id, i)).collect();
    let mut buckets: Vec<Vec<&[f32]>> = vec![Vec::new(); registry.species.len()];

    for org in &organisms {
        if let Some(id) = org.species_id {
            if let Some(&idx) = id_to_idx.get(&id) {
                if org.dna.len() == DNA_DIM {
                    buckets[idx].push(org.dna.as_slice());
                }
            }
        }
    }

    for (idx, dnas) in buckets.iter().enumerate() {
        let species = &mut registry.species[idx];
        if dnas.is_empty() {
            // Last member died — keep the historical centroid, flag extinct.
            species.member_count = 0;
            species.extinct      = true;
            continue;
        }
        species.member_count = dnas.len() as u32;
        // A species with live members is not extinct. Without this, a species
        // flagged extinct on a transient empty-bucket tick stays extinct even
        // after members reappear — `classify_organisms` then skips it via
        // `alive_iter`, stranding those members (and, now, their per-species
        // brain) on a ghost species forever.
        species.extinct = false;

        // ── Pass 1: simple mean ─────────────────────────────────
        let simple = mean_of(dnas);

        // ── Pass 2: trimmed mean (members within threshold of simple mean).
        //   If none qualify (degenerate), fall back to the simple mean.
        let kept: Vec<&[f32]> = dnas.iter()
            .copied()
            .filter(|d| distance(d, &simple) <= SPECIES_SEPARATION_THRESHOLD)
            .collect();
        if kept.is_empty() {
            species.avg_dna = simple;
            continue;
        }
        let mut acc = vec![0.0f32; DNA_DIM];
        for d in &kept {
            for i in 0..DNA_DIM { acc[i] += d[i]; }
        }
        let n = kept.len() as f32;
        for i in 0..DNA_DIM { acc[i] /= n; }
        species.avg_dna = acc;
    }
}


fn mean_of(dnas: &[&[f32]]) -> Vec<f32> {
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
    ai_training:  Res<crate::simulation_settings::AiTrainingMode>,
) {
    // Shares `update_species_averages`'s timer (runs right after via
    // `.chain()`, needing the fresh averages). Same `just_finished` check
    // keeps off-tick frames cheap.
    if !timer.0.just_finished() { return; }

    // Per-tick lookup tables, built once instead of re-scanning the species
    // Vec per organism: `id_to_idx` for the own-species centroid lookup, and
    // `alive_idx` (indices of non-extinct species) for the nearest-centroid
    // search. Both are kept fresh as `create`/`create_with_name` append new
    // species mid-loop — appends land at `registry.species.len()-1`, so a
    // species founded earlier in this loop is a candidate for later organisms
    // exactly as in the old per-call `alive_iter()` filter.
    let mut id_to_idx: std::collections::HashMap<u32, usize> =
        registry.species.iter().enumerate().map(|(i, s)| (s.id, i)).collect();
    let mut alive_idx: Vec<usize> = registry.species.iter().enumerate()
        .filter(|(_, s)| !s.extinct)
        .map(|(i, _)| i)
        .collect();

    for (entity, mut org, imported) in &mut q {
        if org.dna.len() != DNA_DIM { continue; }

        // ── Imported-species short-circuit ────────────────────────
        // Unclassified `.species` import ⇒ route into a name-keyed founder
        // species (parent = None, own tree). Multiple imports from one file
        // collapse into one. Strip the marker after so later ticks run normal
        // drift logic and descendants can fork sub-species.
        if let (Some(origin), None) = (imported, org.species_id) {
            let species_id = match registry.find_alive_by_name(&origin.name) {
                Some(s) => s.id,
                None    => {
                    let id = registry.create_with_name(
                        origin.name.clone(),
                        org.dna.clone(),
                        None,
                    );
                    let new_idx = registry.species.len() - 1;
                    id_to_idx.insert(id, new_idx);
                    alive_idx.push(new_idx);
                    id
                }
            };
            org.species_id = Some(species_id);
            commands.entity(entity).try_remove::<ImportedSpeciesOrigin>();
            continue;
        }

        // Find the nearest alive species + its distance.
        let (nearest_id, nearest_dist) =
            nearest_species(&registry, &alive_idx, &org.dna);

        match org.species_id {
            None => {
                // Unassigned — first-time classification. The drift
                // threshold applies to OFFSPRING diverging from a parent
                // centroid, NOT to initial classification: applying it here
                // would explode the initial cohort (whose genes are sampled
                // across the full ranges) into N unconnected singletons.
                // So: first unclassified organism founds Species 1; every
                // later one joins its nearest species regardless of distance.
                // The trimmed-mean recompute then sheds outliers, and the
                // `Some(current)` branch forks them off with proper edges.
                match nearest_id {
                    Some(id) => org.species_id = Some(id),
                    None     => {
                        let new_id = registry.create(org.dna.clone(), None);
                        let new_idx = registry.species.len() - 1;
                        id_to_idx.insert(new_id, new_idx);
                        alive_idx.push(new_idx);
                        org.species_id = Some(new_id);
                    }
                }
            }
            Some(current) => {
                // Classified — check drift against own species' centroid.
                let own_dist = id_to_idx.get(&current)
                    .map(|&i| distance(&org.dna, &registry.species[i].avg_dna))
                    .unwrap_or(f32::INFINITY);
                if own_dist <= SPECIES_SEPARATION_THRESHOLD {
                    continue; // still a good fit
                }
                // Drifted out — hop to a better-fitting existing species (no
                // edge) if one is within threshold, else fork a new species
                // with `current` as parent.
                if nearest_dist <= SPECIES_SEPARATION_THRESHOLD && nearest_id != Some(current) {
                    org.species_id = nearest_id;
                } else if !ai_training.0 {
                    let new_id = registry.create(org.dna.clone(), Some(current));
                    let new_idx = registry.species.len() - 1;
                    id_to_idx.insert(new_id, new_idx);
                    alive_idx.push(new_idx);
                    org.species_id = Some(new_id);
                }
                // AI-training mode: no fork — a misfit drifter stays put,
                // keeping the species set fixed for the run.
            }
        }
    }
}


/// Closest alive species to `dna`, searching only the precomputed `alive_idx`
/// indices into `registry.species`. `(None, INFINITY)` when no alive species
/// exist — callers treat that as "create a new founder species".
fn nearest_species(
    registry:  &SpeciesRegistry,
    alive_idx: &[usize],
    dna:       &[f32],
) -> (Option<u32>, f32) {
    let mut best_id:   Option<u32> = None;
    let mut best_dist: f32         = f32::INFINITY;
    for &i in alive_idx {
        let s = &registry.species[i];
        let d = distance(dna, &s.avg_dna);
        if d < best_dist {
            best_dist = d;
            best_id   = Some(s.id);
        }
    }
    (best_id, best_dist)
}
