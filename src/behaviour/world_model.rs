// Heterotroph world model.
//
// Every brain tick rebuilds an XZ spatial hash of every photoautotroph
// AND heterotroph (positions + type). Each hetero pool reads top-K
// nearest neighbours within `WORLD_MODEL_RADIUS`. Bucket size = the
// radius, so each query touches at most 9 buckets. Shared resource so
// the grid is built once per tick, not once per pool.

use bevy::prelude::*;
use std::collections::HashMap;

use crate::colony::{Heterotroph, Organism, Photoautotroph};


pub use crate::simulation_settings::WORLD_MODEL_RADIUS;

/// Number of nearest neighbours encoded into the world model. Padded
/// with zeros when fewer than `K` organisms are within radius.
pub const WORLD_MODEL_K: usize = 4;

/// Per-neighbour input dims: (rel_x_norm, rel_z_norm, vel_x_norm,
/// vel_z_norm, photo_one_hot, hetero_one_hot). Velocity is normalised
/// by `VELOCITY_NORM_SCALE` so the brain sees values typically in
/// `[-1, +1]` even when prey is moving at top speed.
pub const WORLD_MODEL_NEIGHBOUR_DIMS: usize = 6;
pub const WORLD_MODEL_DIMS:           usize = WORLD_MODEL_K * WORLD_MODEL_NEIGHBOUR_DIMS;

pub use crate::simulation_settings::VELOCITY_NORM_SCALE;


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


/// Rebuild the grid from current Photoautotroph + Heterotroph transforms
/// + velocities. Run once per brain tick.
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
/// `self_pos`. Used to normalise the herbivore W_EAT reward channel
/// against local prey density. Same 3×3 bucket scan as
/// `collect_neighbours`, but counts instead of sorting.
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


/// Body-frame variant of `encode_neighbours`: rotates each neighbour's
/// rel position + velocity by the inverse observer yaw (from `base_rot`)
/// so the brain sees body-local "front/left" and need not learn the
/// world↔body transform. Used only by the limb pools (sliding pools
/// chase geometrically and gain nothing from body-frame inputs).
pub fn encode_neighbours_body_local(
    neighbours: &[Option<Neighbour>; WORLD_MODEL_K],
    base_rot:   Quat,
    out:        &mut [f32],
) {
    debug_assert_eq!(out.len(), WORLD_MODEL_DIMS);
    out.fill(0.0);

    // Body's forward in world frame is `base_rot * +Z`. Project to XZ
    // and normalise to read off the yaw's sin/cos directly:
    //   fwd_world = (sin θ, *, cos θ)
    let fwd = base_rot * Vec3::Z;
    let len = (fwd.x * fwd.x + fwd.z * fwd.z).sqrt();
    let (sin_h, cos_h) = if len > 1e-6 {
        (fwd.x / len, fwd.z / len)
    } else {
        (0.0, 1.0)
    };

    // Inverse yaw: rotate world (vx, vz) into body-local (bx, bz) via
    //   bx =  vx·cos θ − vz·sin θ
    //   bz =  vx·sin θ + vz·cos θ
    // (Right-handed +Y-up: a CCW yaw turns body +Z toward world +X.)
    let rotate = |x: f32, z: f32| -> (f32, f32) {
        (x * cos_h - z * sin_h, x * sin_h + z * cos_h)
    };

    for (i, slot) in neighbours.iter().enumerate() {
        let Some(n) = slot else { continue };
        let off = i * WORLD_MODEL_NEIGHBOUR_DIMS;
        let (rx, rz) = rotate(n.rel.x,      n.rel.z);
        let (vx, vz) = rotate(n.velocity.x, n.velocity.z);
        out[off    ] = rx / WORLD_MODEL_RADIUS;
        out[off + 1] = rz / WORLD_MODEL_RADIUS;
        out[off + 2] = vx / VELOCITY_NORM_SCALE;
        out[off + 3] = vz / VELOCITY_NORM_SCALE;
        out[off + 4] = if matches!(n.ty, OrganismType::Photo)  { 1.0 } else { 0.0 };
        out[off + 5] = if matches!(n.ty, OrganismType::Hetero) { 1.0 } else { 0.0 };
    }
}


/// Convenience: `encode_neighbours(&collect_neighbours(grid, self_pos), out)`
/// in one call, so callers needn't thread the intermediate array.
pub fn fill_world_model(
    grid:       &WorldModelGrid,
    self_pos:   Vec3,
    out:        &mut [f32],
) {
    let neighbours = collect_neighbours(grid, self_pos);
    encode_neighbours(&neighbours, out);
}


/// Nearest **photoautotroph** within `WORLD_MODEL_RADIUS` of `self_pos`,
/// as `(rel_pos, distance, entity)` or `None`. The returned `Entity`
/// lets the reward shaper detect when nearest prey CHANGES between ticks
/// (progress delta would then compare two different organisms — caller
/// zeroes it). Same 3×3 bucket probe as `fill_world_model`.
pub fn nearest_prey(grid: &WorldModelGrid, self_pos: Vec3) -> Option<(Vec3, f32, Entity)> {
    nearest_prey_within(grid, self_pos, crate::simulation_settings::PREY_SCAN_RADIUS)
}

/// `nearest_prey` with an explicit scan radius — lets a caller use a
/// movement-mode-specific sensory range (e.g. swimmers at
/// `SWIM_SENSORY_RADIUS`) without changing the default `PREY_SCAN_RADIUS`
/// the limb/sliding pools rely on.
pub fn nearest_prey_within(grid: &WorldModelGrid, self_pos: Vec3, radius: f32) -> Option<(Vec3, f32, Entity)> {
    // Prey perception is DECOUPLED from the neighbour-encoding radius: scan out to
    // `radius` (≫ WORLD_MODEL_RADIUS) so the brain can steer toward food that is
    // tens-of-units away, while the gait observation stays at the small radius. The
    // grid is still bucketed at WORLD_MODEL_RADIUS, so probe enough bucket rings to
    // cover the larger scan radius.
    let bucket    = WORLD_MODEL_RADIUS;
    let radius_sq = radius * radius;
    let span      = (radius / bucket).ceil() as i32;
    let kx = (self_pos.x / bucket).floor() as i32;
    let kz = (self_pos.z / bucket).floor() as i32;

    let mut best: Option<(f32, Vec3, Entity)> = None;
    for dx in -span..=span {
        for dz in -span..=span {
            if let Some(entries) = grid.grid.get(&(kx + dx, kz + dz)) {
                for &e in entries {
                    if !matches!(e.ty, OrganismType::Photo) { continue; }
                    let rel = e.pos - self_pos;
                    let d2  = rel.length_squared();
                    // No self-exclusion: the Photo filter + Heterotroph
                    // caller make it unnecessary, and a `d2 < 1e-6` skip
                    // would drop a photo a herbivore is parked exactly on.
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


/// Every photoautotroph within `WORLD_MODEL_RADIUS` of `self_pos`, as
/// `(rel_xz, distance)`. Self-excluded via exact-position match.
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
