// Organism-vs-organism contact detection.
//
// Three-phase pipeline (mirrors the old design but indexed by body parts):
//   1. Broad phase  — root-position spatial hash, cheap distance test.
//   2. Mid phase    — body-part-pair bounding-sphere overlap test.
//   3. Narrow phase — cell-vs-cell sphere intersection between the two
//                     candidate body parts.
//
// On contact, this system emits one `OrganismContactEvent` per body-part
// pair that touched. The event carries the body-part indices on each side
// so `predation.rs` can scope its consumption to the body part that was
// physically reached, rather than nuking the whole prey on first contact.

use bevy::prelude::*;
use std::collections::{HashMap, HashSet};

use crate::cell::*;
use crate::colony::*;


// ── Events ───────────────────────────────────────────────────────────────────

/// Fired whenever a body part of one organism physically touches a body
/// part of another. Carries both organisms and the body-part indices that
/// made contact. Consumed by `predation.rs` (and any future interaction
/// logic — e.g. mating, parasitism).
#[derive(Message, Clone, Copy)]
pub struct OrganismContactEvent {
    pub a:           Entity,
    pub b:           Entity,
    pub body_part_a: usize,
    pub body_part_b: usize,
}


// ── Constants ────────────────────────────────────────────────────────────────

/// Broad phase: organism root positions must be closer than this for any
/// further checks to run. Set generously — it's only a cheap distance test.
const ORGANISM_BROAD_RADIUS: f32 = 10.0;

/// Tick interval — running the full pipeline every frame is wasteful at
/// 1100-organism scale, and contacts emerge cleanly at 10 Hz.
const COLLISION_TICK: f32 = 0.1;

/// Fraction of (averaged) movement speed converted into a deflection push
/// when two organisms touch. Push is along the cell-pair contact axis.
const PUSH_STRENGTH: f32 = 1.0;


// ── Timer resource ───────────────────────────────────────────────────────────

#[derive(Resource)]
pub struct OrganismCollisionTimer {
    pub timer: Timer,
}

impl Default for OrganismCollisionTimer {
    fn default() -> Self {
        Self { timer: Timer::from_seconds(COLLISION_TICK, TimerMode::Repeating) }
    }
}


// ── Spatial hashing ──────────────────────────────────────────────────────────

#[inline]
fn grid_key(pos: Vec3, bucket: f32) -> [i32; 3] {
    [
        (pos.x / bucket).floor() as i32,
        (pos.y / bucket).floor() as i32,
        (pos.z / bucket).floor() as i32,
    ]
}

fn neighbour_keys(key: [i32; 3]) -> [[i32; 3]; 27] {
    let mut keys = [[0i32; 3]; 27];
    let mut i = 0;
    for dx in -1i32..=1 {
        for dy in -1i32..=1 {
            for dz in -1i32..=1 {
                keys[i] = [key[0] + dx, key[1] + dy, key[2] + dz];
                i += 1;
            }
        }
    }
    keys
}


// ── Snapshot types ───────────────────────────────────────────────────────────

/// Captured per body part once per tick; the world-space cell positions are
/// pre-computed so the cell-pair narrow phase doesn't have to re-transform.
struct BodyPartSnapshot {
    index:           usize,
    world_centroid:  Vec3,
    bound_radius:    f32, // mid-phase sphere radius
    cells_world:     Vec<Vec3>,
}

struct OrganismSnapshot {
    entity:         Entity,
    root_world_pos: Vec3,
    movement_speed: f32,
    body_parts:     Vec<BodyPartSnapshot>,
}

fn snapshot(
    entity:    Entity,
    organism:  &Organism,
    transform: &Transform,
) -> OrganismSnapshot {
    let body_parts: Vec<BodyPartSnapshot> = organism.body_parts.iter()
        .enumerate()
        .filter(|(_, bp)| !bp.cells.is_empty())
        .map(|(i, bp)| {
            // Cell positions in world space, computed once per tick.
            let cells_world: Vec<Vec3> = bp.cells.iter()
                .map(|c| transform.transform_point(bp.local_offset + c.local_pos))
                .collect();

            // Centroid + radius for the mid-phase sphere test.
            let centroid = cells_world.iter().fold(Vec3::ZERO, |a, b| a + *b)
                            / cells_world.len() as f32;
            let bound = bp.local_bounding_radius();

            BodyPartSnapshot {
                index: i,
                world_centroid: centroid,
                bound_radius: bound,
                cells_world,
            }
        })
        .collect();

    OrganismSnapshot {
        entity,
        root_world_pos: transform.translation,
        movement_speed: organism.movement_speed,
        body_parts,
    }
}


// ── Main system ──────────────────────────────────────────────────────────────

pub fn apply_organism_collision(
    time:               Res<Time>,
    mut timer:          ResMut<OrganismCollisionTimer>,
    mut contact_events: MessageWriter<OrganismContactEvent>,
    mut params: ParamSet<(
        Query<(&Organism, &Transform, Entity), With<OrganismRoot>>,
        Query<&mut Organism, With<OrganismRoot>>,
    )>,
) {
    timer.timer.tick(time.delta());
    if !timer.timer.just_finished() { return; }

    // ── 1. Snapshot every organism (immutable read of (Organism, Transform))
    let snapshots: Vec<OrganismSnapshot> = params.p0().iter()
        .map(|(o, t, e)| snapshot(e, o, t))
        .collect();

    if snapshots.is_empty() { return; }

    // ── 2. Build the broad-phase spatial hash on root positions
    let mut organism_grid: HashMap<[i32; 3], Vec<usize>> = HashMap::new();
    for (i, snap) in snapshots.iter().enumerate() {
        organism_grid
            .entry(grid_key(snap.root_world_pos, ORGANISM_BROAD_RADIUS))
            .or_default()
            .push(i);
    }

    // Push accumulators + dedup set for emitted events.
    let mut pushes: HashMap<Entity, Vec3> = HashMap::new();
    let mut emitted: HashSet<(Entity, Entity, usize, usize)> = HashSet::new();

    let cell_contact_d  = CELL_COLLISION_RADIUS * 2.0;
    let cell_contact_d2 = cell_contact_d * cell_contact_d;

    // ── 3. Pair loop — broad → mid → narrow ──────────────────────────────────
    for (idx_a, snap_a) in snapshots.iter().enumerate() {
        let key   = grid_key(snap_a.root_world_pos, ORGANISM_BROAD_RADIUS);
        let nkeys = neighbour_keys(key);

        for nkey in nkeys {
            let Some(bucket) = organism_grid.get(&nkey) else { continue };

            for &idx_b in bucket {
                if idx_b <= idx_a { continue; }
                let snap_b = &snapshots[idx_b];

                // Broad phase: skip pairs whose roots are far apart.
                let dx = snap_a.root_world_pos - snap_b.root_world_pos;
                if dx.length_squared() >= ORGANISM_BROAD_RADIUS * ORGANISM_BROAD_RADIUS {
                    continue;
                }

                // Mid phase: per-body-part bounding-sphere overlap.
                for bp_a in &snap_a.body_parts {
                    for bp_b in &snap_b.body_parts {
                        let bp_d  = bp_a.world_centroid - bp_b.world_centroid;
                        let r_sum = bp_a.bound_radius + bp_b.bound_radius;
                        if bp_d.length_squared() >= r_sum * r_sum { continue; }

                        // Narrow phase: cell-pair sphere intersection.
                        // Quadratic in cells per body part — fine because
                        // typical body parts carry < 50 cells in adult form
                        // and the broad/mid phases keep most pairs out.
                        for &ca in &bp_a.cells_world {
                            for &cb in &bp_b.cells_world {
                                let d2 = (cb - ca).length_squared();
                                if d2 >= cell_contact_d2 || d2 < 1e-6 { continue; }

                                // Canonical (entity, body-part) key for dedup
                                // — sort by entity so (A,B,bp_a,bp_b) and
                                // (B,A,bp_b,bp_a) collapse to one event.
                                let key = if snap_a.entity < snap_b.entity {
                                    (snap_a.entity, snap_b.entity, bp_a.index, bp_b.index)
                                } else {
                                    (snap_b.entity, snap_a.entity, bp_b.index, bp_a.index)
                                };

                                if emitted.insert(key) {
                                    contact_events.write(OrganismContactEvent {
                                        a: snap_a.entity,
                                        b: snap_b.entity,
                                        body_part_a: bp_a.index,
                                        body_part_b: bp_b.index,
                                    });
                                }

                                // Deflection push proportional to penetration
                                // and average speed. Splits 50/50 between the
                                // pair so the system is symmetric.
                                let push_dir = (cb - ca) / d2.sqrt();
                                let pen      = 1.0 - d2.sqrt() / cell_contact_d;
                                let avg_v    = (snap_a.movement_speed + snap_b.movement_speed) * 0.5;
                                let mag      = PUSH_STRENGTH * pen * avg_v;

                                *pushes.entry(snap_a.entity).or_insert(Vec3::ZERO) -= push_dir * mag;
                                *pushes.entry(snap_b.entity).or_insert(Vec3::ZERO) += push_dir * mag;
                            }
                        }
                    }
                }
            }
        }
    }

    // ── 4. Apply accumulated pushes
    let mut write_query = params.p1();
    for (entity, push) in &pushes {
        let Ok(mut org) = write_query.get_mut(*entity) else { continue };
        let new_dir = org.movement_direction + *push;
        if new_dir.length_squared() > 1e-6 {
            org.movement_direction = new_dir.normalize();
        }
    }
}
