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
#[derive(Resource, Default)]
pub struct WorldModelGrid {
    pub grid: HashMap<(i32, i32), Vec<(Vec3, OrganismType)>>,
}


/// Rebuild the grid from the current Photoautotroph + Heterotroph
/// transforms. Run once per brain tick (gated on the 30 Hz brain
/// timer in `behaviour.rs`).
pub fn rebuild_world_model_grid(
    mut grid: ResMut<WorldModelGrid>,
    photos:  Query<&Transform, With<Photoautotroph>>,
    heteros: Query<&Transform, With<Heterotroph>>,
) {
    grid.grid.clear();
    let bucket = WORLD_MODEL_RADIUS;
    for tf in photos.iter() {
        let p = tf.translation;
        let key = ((p.x / bucket).floor() as i32, (p.z / bucket).floor() as i32);
        grid.grid.entry(key).or_default().push((p, OrganismType::Photo));
    }
    for tf in heteros.iter() {
        let p = tf.translation;
        let key = ((p.x / bucket).floor() as i32, (p.z / bucket).floor() as i32);
        grid.grid.entry(key).or_default().push((p, OrganismType::Hetero));
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
                for &(p, ty) in bucket_entries {
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
