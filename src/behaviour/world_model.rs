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
//
// Mirrors `world_model_old.rs` 1:1 for the grid + filter logic
// (Photoautotroph + Heterotroph markers, two-query rebuild,
// relative-position return from `nearest_prey`). The only addition
// is `prey_in_radius`, used by the new L1-hetero DQN's body-frame
// state encoder.

use bevy::prelude::*;
use std::collections::HashMap;

use crate::colony::{Heterotroph, Organism, Photoautotroph};


/// Radius (world units) within which neighbour organisms are considered
/// part of the heterotroph's world model.
pub const WORLD_MODEL_RADIUS: f32 = 60.0;

/// Number of nearest neighbours encoded into the world model. Padded
/// with zeros when fewer than `K` organisms are within radius.
pub const WORLD_MODEL_K: usize = 4;

/// Per-neighbour input dims: (rel_x_norm, rel_z_norm, vel_x_norm,
/// vel_z_norm, photo_one_hot, hetero_one_hot). Velocity is normalised
/// by `VELOCITY_NORM_SCALE` so the brain sees values typically in
/// `[-1, +1]` even when prey is moving at top speed.
pub const WORLD_MODEL_NEIGHBOUR_DIMS: usize = 6;
pub const WORLD_MODEL_DIMS:           usize = WORLD_MODEL_K * WORLD_MODEL_NEIGHBOUR_DIMS;

/// Velocity normalisation factor. Roughly the expected top speed in
/// world-units / second — see `intelligence_level_1_hetero::MAX_SPEED`.
/// A neighbour cruising at MAX_SPEED registers as `±1` on the
/// corresponding velocity dim.
pub const VELOCITY_NORM_SCALE: f32 = 20.0;


#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OrganismType { Photo, Hetero }


/// One spatial-hash entry — bundles position, velocity, trophic type,
/// and identity so callers can encode any of these without re-querying
/// the ECS.
#[derive(Clone, Copy)]
pub struct GridEntry {
    pub pos:      Vec3,
    pub velocity: Vec3,
    pub ty:       OrganismType,
    pub entity:   Entity,
}


/// XZ spatial hash of all live organisms, rebuilt every brain tick.
/// Bucket size = `WORLD_MODEL_RADIUS` so probing 3×3 buckets around any
/// query position covers the full neighbourhood.
#[derive(Resource, Default)]
pub struct WorldModelGrid {
    pub grid: HashMap<(i32, i32), Vec<GridEntry>>,
}


/// Rebuild the grid from the current Photoautotroph + Heterotroph
/// transforms + velocities. Run once per brain tick (gated on the
/// hetero brain timer in `behaviour.rs`). Krishi entities also carry
/// the `Heterotroph` marker (Krishi is a fixed-form heterotroph variant)
/// so they land in the grid tagged as `Hetero`.
pub fn rebuild_world_model_grid(
    mut grid: ResMut<WorldModelGrid>,
    photos:  Query<(Entity, &Transform, &Organism), With<Photoautotroph>>,
    heteros: Query<(Entity, &Transform, &Organism), With<Heterotroph>>,
) {
    grid.grid.clear();
    let bucket = WORLD_MODEL_RADIUS;
    for (e, tf, org) in photos.iter() {
        let p = tf.translation;
        let key = ((p.x / bucket).floor() as i32, (p.z / bucket).floor() as i32);
        grid.grid.entry(key).or_default().push(GridEntry {
            pos: p, velocity: org.velocity, ty: OrganismType::Photo, entity: e,
        });
    }
    for (e, tf, org) in heteros.iter() {
        let p = tf.translation;
        let key = ((p.x / bucket).floor() as i32, (p.z / bucket).floor() as i32);
        grid.grid.entry(key).or_default().push(GridEntry {
            pos: p, velocity: org.velocity, ty: OrganismType::Hetero, entity: e,
        });
    }
}


/// Resolved per-neighbour snapshot returned by `collect_neighbours`.
/// Pre-shifted into the heterotroph's frame (`rel = pos - self_pos`)
/// so callers don't repeat the subtraction.
#[derive(Clone, Copy)]
pub struct Neighbour {
    pub entity:   Entity,
    pub rel:      Vec3,
    pub velocity: Vec3,
    pub ty:       OrganismType,
}


/// Resolve the K-nearest neighbours of `self_pos` from the grid,
/// sorted by ascending distance. Returns an array of length
/// `WORLD_MODEL_K`; trailing slots are `None` when fewer than K
/// neighbours are in range.
pub fn collect_neighbours(
    grid:     &WorldModelGrid,
    self_pos: Vec3,
) -> [Option<Neighbour>; WORLD_MODEL_K] {
    let bucket    = WORLD_MODEL_RADIUS;
    let radius_sq = WORLD_MODEL_RADIUS * WORLD_MODEL_RADIUS;
    let kx = (self_pos.x / bucket).floor() as i32;
    let kz = (self_pos.z / bucket).floor() as i32;

    let mut candidates: Vec<(f32, Neighbour)> = Vec::new();
    for dx in -1..=1 {
        for dz in -1..=1 {
            if let Some(entries) = grid.grid.get(&(kx + dx, kz + dz)) {
                for &e in entries {
                    let rel = e.pos - self_pos;
                    let d2  = rel.length_squared();
                    if d2 < 1e-6 { continue; }      // self-exclusion
                    if d2 > radius_sq { continue; }
                    candidates.push((d2, Neighbour {
                        entity: e.entity, rel, velocity: e.velocity, ty: e.ty,
                    }));
                }
            }
        }
    }
    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut out: [Option<Neighbour>; WORLD_MODEL_K] = [None; WORLD_MODEL_K];
    for (i, (_, n)) in candidates.into_iter().take(WORLD_MODEL_K).enumerate() {
        out[i] = Some(n);
    }
    out
}


/// Count Photoautotroph grid entries within `radius` (XZ distance) of
/// `self_pos`. Used by the herbivore reward channel to detect "how
/// rich is this patch of the world" so the W_EAT channel can be
/// normalised against random-walk baseline encounter rates in dense
/// vs sparse conditions. Implementation mirrors `collect_neighbours`
/// — scans the same 3×3 bucket window — but stops at a count rather
/// than building a sorted list.
pub fn count_local_prey(
    grid:     &WorldModelGrid,
    self_pos: Vec3,
    radius:   f32,
) -> u32 {
    let bucket    = WORLD_MODEL_RADIUS;
    let radius_sq = radius * radius;
    let kx = (self_pos.x / bucket).floor() as i32;
    let kz = (self_pos.z / bucket).floor() as i32;
    let mut n: u32 = 0;
    for dx in -1..=1 {
        for dz in -1..=1 {
            if let Some(entries) = grid.grid.get(&(kx + dx, kz + dz)) {
                for &e in entries {
                    if !matches!(e.ty, OrganismType::Photo) { continue; }
                    let rel = e.pos - self_pos;
                    let d2  = rel.x * rel.x + rel.z * rel.z;
                    if d2 < 1e-6 { continue; }
                    if d2 <= radius_sq { n += 1; }
                }
            }
        }
    }
    n
}


/// Encode pre-resolved neighbours into the network input slice
/// (`out` must be exactly `WORLD_MODEL_DIMS` long). 6 dims per slot:
/// `(rel_x_norm, rel_z_norm, vel_x_norm, vel_z_norm, is_photo, is_hetero)`.
/// Empty neighbour slots fill with zeros.
pub fn encode_neighbours(
    neighbours: &[Option<Neighbour>; WORLD_MODEL_K],
    out:        &mut [f32],
) {
    debug_assert_eq!(out.len(), WORLD_MODEL_DIMS);
    out.fill(0.0);
    for (i, slot) in neighbours.iter().enumerate() {
        let Some(n) = slot else { continue };
        let off = i * WORLD_MODEL_NEIGHBOUR_DIMS;
        out[off    ] = n.rel.x      / WORLD_MODEL_RADIUS;
        out[off + 1] = n.rel.z      / WORLD_MODEL_RADIUS;
        out[off + 2] = n.velocity.x / VELOCITY_NORM_SCALE;
        out[off + 3] = n.velocity.z / VELOCITY_NORM_SCALE;
        out[off + 4] = if matches!(n.ty, OrganismType::Photo)  { 1.0 } else { 0.0 };
        out[off + 5] = if matches!(n.ty, OrganismType::Hetero) { 1.0 } else { 0.0 };
    }
}


/// Convenience: fill `out` with encoded neighbours in one call.
/// Equivalent to `encode_neighbours(&collect_neighbours(grid, self_pos), out)`
/// but exposed so legacy single-pool callers don't need to thread the
/// intermediate array.
pub fn fill_world_model(
    grid:       &WorldModelGrid,
    self_pos:   Vec3,
    out:        &mut [f32],
) {
    let neighbours = collect_neighbours(grid, self_pos);
    encode_neighbours(&neighbours, out);
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

    let mut best: Option<(f32, Vec3, Entity)> = None;
    for dx in -1..=1 {
        for dz in -1..=1 {
            if let Some(entries) = grid.grid.get(&(kx + dx, kz + dz)) {
                for &e in entries {
                    if !matches!(e.ty, OrganismType::Photo) { continue; }
                    let rel = e.pos - self_pos;
                    let d2  = rel.length_squared();
                    // No self-exclusion needed — the type filter above
                    // restricts to Photoautotrophs, and the caller is
                    // always a Heterotroph. The previous `d2 < 1e-6`
                    // skip was a hangover from a generic version of
                    // this function. It produced the "dance on one
                    // spot" bug: a herbivore parked exactly on a photo
                    // (d_root < 0.001) lost its target, fell through
                    // to the wander branch, drifted away, re-targeted,
                    // approached, parked, wandered — forever.
                    if d2 > radius_sq { continue; }
                    if best.map_or(true, |(b, _, _)| d2 < b) {
                        best = Some((d2, rel, e.entity));
                    }
                }
            }
        }
    }
    best.map(|(d2, rel, ent)| (rel, d2.sqrt(), ent))
}


/// Every photoautotroph within `WORLD_MODEL_RADIUS` of `self_pos`,
/// as `(rel_xz, distance)`. Used by the L1-hetero DQN's body-frame
/// state encoder to fill its 8 angular bins; the caller does the
/// bin-assignment and yaw rotation itself.
///
/// Self-exclusion via exact-position match, same as the other two
/// helpers in this file.
pub fn prey_in_radius(grid: &WorldModelGrid, self_pos: Vec3) -> Vec<(Vec2, f32)> {
    let bucket    = WORLD_MODEL_RADIUS;
    let radius_sq = WORLD_MODEL_RADIUS * WORLD_MODEL_RADIUS;
    let kx = (self_pos.x / bucket).floor() as i32;
    let kz = (self_pos.z / bucket).floor() as i32;

    let mut out: Vec<(Vec2, f32)> = Vec::new();
    for dx in -1..=1 {
        for dz in -1..=1 {
            if let Some(entries) = grid.grid.get(&(kx + dx, kz + dz)) {
                for &e in entries {
                    if !matches!(e.ty, OrganismType::Photo) { continue; }
                    let rel = e.pos - self_pos;
                    let d2  = rel.length_squared();
                    if d2 < 1e-6 { continue; }
                    if d2 > radius_sq { continue; }
                    out.push((Vec2::new(rel.x, rel.z), d2.sqrt()));
                }
            }
        }
    }
    out
}
