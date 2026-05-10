// Heterotroph world model.
//
// Every brain tick rebuilds an XZ spatial hash of every photoautotroph
// AND heterotroph (positions + type). Each Level 1/2/3 hetero pool
// reads from this hash to fill its per-organism input vector with the
// top-K nearest neighbours within `WORLD_MODEL_RADIUS`. Bucket size =
// the radius, so each query touches at most 9 buckets.
//
// One shared resource means the grid is built exactly once per brain
// tick instead of three times (one per pool).

use bevy::prelude::*;
use std::collections::HashMap;

use crate::colony::{Heterotroph, Photoautotroph};


/// Radius (world units) within which neighbour organisms are considered
/// part of the heterotroph's world model.
pub const WORLD_MODEL_RADIUS: f32 = 60.0;

/// Number of nearest neighbours encoded into the world model. Padded
/// with zeros when fewer than `K` organisms are within radius.
pub const WORLD_MODEL_K: usize = 4;

/// Per-neighbour input dims: (rel_x_norm, rel_z_norm, photo_one_hot,
/// hetero_one_hot). Total world-model contribution to the input vector
/// is `WORLD_MODEL_K * WORLD_MODEL_NEIGHBOUR_DIMS`.
pub const WORLD_MODEL_NEIGHBOUR_DIMS: usize = 4;
pub const WORLD_MODEL_DIMS:           usize = WORLD_MODEL_K * WORLD_MODEL_NEIGHBOUR_DIMS;


#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OrganismType { Photo, Hetero }


/// XZ spatial hash of all live organisms, rebuilt every brain tick.
/// Bucket size = `WORLD_MODEL_RADIUS` so probing 3×3 buckets around any
/// query position covers the full neighbourhood.
///
/// Each entry is `(world_position, type, entity)`. The `entity` is
/// what enables identity-aware reward shaping: the IL1-hetero pool
/// uses it to detect when the "nearest prey" changes between ticks
/// and zero its progress signal across that boundary, avoiding the
/// spurious-reinforcement-on-identity-flip pathology.
#[derive(Resource, Default)]
pub struct WorldModelGrid {
    pub grid: HashMap<(i32, i32), Vec<(Vec3, OrganismType, Entity)>>,
}


/// Rebuild the grid from the current Photoautotroph + Heterotroph
/// transforms. Run once per brain tick (gated on the 30 Hz brain
/// timer in `behaviour.rs`).
pub fn rebuild_world_model_grid(
    mut grid: ResMut<WorldModelGrid>,
    photos:  Query<(Entity, &Transform), With<Photoautotroph>>,
    heteros: Query<(Entity, &Transform), With<Heterotroph>>,
) {
    grid.grid.clear();
    let bucket = WORLD_MODEL_RADIUS;
    for (e, tf) in photos.iter() {
        let p = tf.translation;
        let key = ((p.x / bucket).floor() as i32, (p.z / bucket).floor() as i32);
        grid.grid.entry(key).or_default().push((p, OrganismType::Photo, e));
    }
    for (e, tf) in heteros.iter() {
        let p = tf.translation;
        let key = ((p.x / bucket).floor() as i32, (p.z / bucket).floor() as i32);
        grid.grid.entry(key).or_default().push((p, OrganismType::Hetero, e));
    }
}


/// Fill `out` (must be exactly `WORLD_MODEL_DIMS` long) with encoded
/// neighbour data for the organism at `self_pos`. Skips entries at
/// `self_pos` itself (self-exclusion via exact-match position). Pads
/// remaining slots with zeros if fewer than K neighbours are within
/// radius.
pub fn fill_world_model(
    grid:       &WorldModelGrid,
    self_pos:   Vec3,
    out:        &mut [f32],
) {
    debug_assert_eq!(out.len(), WORLD_MODEL_DIMS);
    out.fill(0.0);

    let bucket   = WORLD_MODEL_RADIUS;
    let radius_sq = WORLD_MODEL_RADIUS * WORLD_MODEL_RADIUS;
    let kx = (self_pos.x / bucket).floor() as i32;
    let kz = (self_pos.z / bucket).floor() as i32;

    // Collect (distance², rel_pos, type) of every neighbour within radius.
    let mut candidates: Vec<(f32, Vec3, OrganismType)> = Vec::new();
    for dx in -1..=1 {
        for dz in -1..=1 {
            if let Some(bucket_entries) = grid.grid.get(&(kx + dx, kz + dz)) {
                // The grid carries an `Entity` per entry now (used by
                // `nearest_prey` for identity tracking); the world-model
                // input vector itself doesn't need it, hence the `_`.
                for &(p, ty, _) in bucket_entries {
                    let rel = p - self_pos;
                    let d2  = rel.length_squared();
                    if d2 < 1e-6 { continue; }       // self-exclusion
                    if d2 > radius_sq { continue; }
                    candidates.push((d2, rel, ty));
                }
            }
        }
    }

    // Take the K closest.
    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    candidates.truncate(WORLD_MODEL_K);

    // Encode into `out`. Each slot is 4 dims:
    // [rel_x_norm, rel_z_norm, is_photo, is_hetero].
    for (i, (_, rel, ty)) in candidates.iter().enumerate() {
        let off = i * WORLD_MODEL_NEIGHBOUR_DIMS;
        out[off    ] = rel.x / WORLD_MODEL_RADIUS;
        out[off + 1] = rel.z / WORLD_MODEL_RADIUS;
        out[off + 2] = if matches!(ty, OrganismType::Photo)  { 1.0 } else { 0.0 };
        out[off + 3] = if matches!(ty, OrganismType::Hetero) { 1.0 } else { 0.0 };
    }
}


/// Find the nearest **photoautotroph** within `WORLD_MODEL_RADIUS` of
/// `self_pos`. Returns `(rel_pos, distance, entity)` or `None` when
/// no prey is in range.
///
/// Used by the heterotroph reward shaper to compute the per-tick
/// "did I get closer to prey?" (progress) and "am I facing prey?"
/// (alignment) signals. The returned `Entity` lets the caller detect
/// when the nearest prey CHANGES between ticks: in that case the
/// progress reward is meaningless (the delta would compare distances
/// against two different organisms) and the caller zeroes it.
///
/// Same 3×3 bucket probe as `fill_world_model`; the tiny duplication
/// is preferable to threading nearest-prey state through the latter's
/// signature because the call sites have different mutability needs.
pub fn nearest_prey(grid: &WorldModelGrid, self_pos: Vec3) -> Option<(Vec3, f32, Entity)> {
    let bucket    = WORLD_MODEL_RADIUS;
    let radius_sq = WORLD_MODEL_RADIUS * WORLD_MODEL_RADIUS;
    let kx = (self_pos.x / bucket).floor() as i32;
    let kz = (self_pos.z / bucket).floor() as i32;

    let mut best: Option<(f32, Vec3, Entity)> = None; // (distance², rel, entity)
    for dx in -1..=1 {
        for dz in -1..=1 {
            if let Some(entries) = grid.grid.get(&(kx + dx, kz + dz)) {
                for &(p, ty, ent) in entries {
                    if !matches!(ty, OrganismType::Photo) { continue; }
                    let rel = p - self_pos;
                    let d2  = rel.length_squared();
                    if d2 < 1e-6 { continue; }       // self / coincident
                    if d2 > radius_sq { continue; }
                    if best.map_or(true, |(b, _, _)| d2 < b) {
                        best = Some((d2, rel, ent));
                    }
                }
            }
        }
    }
    best.map(|(d2, rel, ent)| (rel, d2.sqrt(), ent))
}
