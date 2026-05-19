// Herbivore sensory layer.
//
// Every brain tick this module runs `update_target_distance`, which
// scans for the nearest photoautotroph within `SENSORY_RADIUS` of
// each herbivore and writes the result into `Organism::target_distance`.
// Two consumers downstream:
//
//   1. The RL brain reads `target_distance / SENSORY_RADIUS` as one
//      of its input observations (a tight-radius "am I near food?"
//      channel that complements the broader 60-unit world model).
//
//   2. The same brain uses Δ`target_distance` between successive
//      ticks as a *progress reward*: when the distance shrinks the
//      agent earns a small reward, when it grows the agent pays a
//      small penalty. This is added to the primary Δ`dopamine`
//      reward — see `intelligence_level_herbivore_1::REINFORCE`.
//
// Implementation note: the scan reuses the heterotroph world-model
// grid (`WorldModelGrid`) which is already rebuilt once per brain
// tick by `world_model::rebuild_world_model_grid`. That grid has
// bucket size = `WORLD_MODEL_RADIUS` (60 units), so probing 3×3
// buckets covers every photo inside our 50-unit query radius
// without any extra spatial-hash machinery. Cost is roughly
// `Σ(heteros) × neighbours_in_3×3_buckets` per tick — under a
// millisecond at 4k organisms.

use bevy::prelude::*;

use crate::colony::{Carnivore, Heterotroph, Organism};
use crate::world_model::{OrganismType, WorldModelGrid};


/// Radius (world units) within which the sensory algorithm looks
/// for a target photo. Anything beyond this is treated as "no
/// target", and `Organism::target_distance` saturates at this
/// value so the input observation stays bounded.
pub const SENSORY_RADIUS: f32 = 50.0;


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
    let bucket    = crate::world_model::WORLD_MODEL_RADIUS;
    let radius_sq = SENSORY_RADIUS * SENSORY_RADIUS;

    for (transform, mut organism) in &mut heteros {
        let pos = transform.translation;
        let kx  = (pos.x / bucket).floor() as i32;
        let kz  = (pos.z / bucket).floor() as i32;

        let mut best_d2 = radius_sq;
        let mut found   = false;
        for dx in -1..=1 {
            for dz in -1..=1 {
                let Some(entries) = world_grid.grid.get(&(kx + dx, kz + dz)) else { continue };
                for &entry in entries {
                    if !matches!(entry.ty, OrganismType::Photo) { continue; }
                    let rel = entry.pos - pos;
                    // XZ-plane distance — Y doesn't matter for the
                    // herbivore's "can I reach this?" estimate.
                    let d2  = rel.x * rel.x + rel.z * rel.z;
                    if d2 > radius_sq { continue; }
                    if d2 < best_d2 {
                        best_d2 = d2;
                        found   = true;
                    }
                }
            }
        }

        let new_td = if found { best_d2.sqrt() } else { SENSORY_RADIUS };
        // Only write when changed so Change<Organism>-keyed
        // systems don't see spurious dirty markers every tick.
        if organism.target_distance.to_bits() != new_td.to_bits() {
            organism.target_distance = new_td;
        }
    }
}
