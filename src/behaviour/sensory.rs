// Herbivore sensory layer.
//
// `update_target_distance` (each brain tick) writes the nearest
// photoautotroph distance within `SENSORY_RADIUS` into
// `Organism::target_distance`. Consumed as (1) a "near food?" brain
// input and (2) a per-tick progress reward (Δdistance). Reuses the
// already-rebuilt `WorldModelGrid`; its bucket = `WORLD_MODEL_RADIUS`
// ≥ the query radius, so a 3×3 bucket probe covers it.

use bevy::prelude::*;

use crate::colony::{Carnivore, Heterotroph, Organism};
use crate::world_model::{for_each_in_ring, OrganismType, WorldModelGrid};


pub use crate::simulation_settings::SENSORY_RADIUS;


/// For every herbivore (Heterotroph, non-Carnivore), find the
/// nearest photoautotroph within `SENSORY_RADIUS` and write its
/// distance into `target_distance`. If none is in range, the field
/// is set to `SENSORY_RADIUS` (the "out of range" sentinel).
///
/// Uses XZ-plane distance so it's directly comparable across the
/// terrain — height differences between the herbivore and the
/// photo don't inflate the perceived distance the way 3D would.
pub fn update_target_distance(
    world_grid: Res<WorldModelGrid>,
    mut heteros: Query<
        (&Transform, &mut Organism),
        (With<Heterotroph>, Without<Carnivore>),
    >,
) {
    let radius_sq = SENSORY_RADIUS * SENSORY_RADIUS;

    // Entity-disjoint per-herbivore writes; reads only `&world_grid` (HashMap read,
    // Sync). `best_d2`/`found` are per-closure locals (not a shared `Local`), so this
    // fans out over ComputeTaskPool with no shared mutable state.
    let world_grid = &*world_grid;
    heteros.par_iter_mut().for_each(|(transform, mut organism)| {
        let pos = transform.translation;

        let mut best_d2 = radius_sq;
        let mut found   = false;
        // 3×3 bucket ring (span = 1): the grid's bucket = WORLD_MODEL_RADIUS
        // ≥ SENSORY_RADIUS, so the ring covers the full query radius.
        for_each_in_ring(world_grid, pos, 1, |entry| {
            if !matches!(entry.ty, OrganismType::Photo) { return; }
            let rel = entry.pos - pos;
            // XZ-plane distance — Y doesn't matter for the
            // herbivore's "can I reach this?" estimate.
            let d2  = rel.x * rel.x + rel.z * rel.z;
            if d2 > radius_sq { return; }
            if d2 < best_d2 {
                best_d2 = d2;
                found   = true;
            }
        });

        let new_td = if found { best_d2.sqrt() } else { SENSORY_RADIUS };
        // Only write when changed so Change<Organism>-keyed
        // systems don't see spurious dirty markers every tick.
        if organism.target_distance.to_bits() != new_td.to_bits() {
            organism.target_distance = new_td;
        }
    });
}
