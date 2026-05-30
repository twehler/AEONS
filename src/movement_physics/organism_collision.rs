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

/// Maximum positional separation applied to any single organism per
/// collision tick (XZ plane, world units). Caps the integrated push
/// so deeply-overlapping organisms — e.g. a 30-cell vs 30-cell pair
/// can generate up to 900 narrow-phase contacts in one tick — don't
/// snap apart by an absurd amount. At 10 Hz this allows a maximum
/// separation speed of 5 world units / second, fast enough to
/// resolve any plausible penetration within ~1 s yet slow enough
/// that the eye reads it as a firm push rather than a teleport.
const MAX_SEPARATION_PER_TICK: f32 = 0.5;


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
    /// Type flags captured so the push accumulator can detect
    /// predator-prey pairs and skip them. Predation handles
    /// "separation" by consuming the prey one body part at a time —
    /// pushing the predator away from the prey only fights that
    /// resolution and produces the visible "approach → contact →
    /// pushed back → re-approach" dance.
    is_photo:       bool,
    is_carnivore:   bool,
    /// `false` for limb-based organisms. This custom collision system
    /// computes cell world positions as `root_transform × local`,
    /// which is only valid for sliding organisms (single Kinematic
    /// root). Limb organisms have independent Dynamic body-part bodies
    /// whose true world positions come from Avian, so any pair
    /// involving a limb organism is skipped here and handled by
    /// `avian_setup::emit_limb_contact_events` instead.
    sliding_movement: bool,
    body_parts:     Vec<BodyPartSnapshot>,
}

fn snapshot(
    entity:       Entity,
    organism:     &Organism,
    transform:    &Transform,
    is_photo:     bool,
    is_carnivore: bool,
) -> OrganismSnapshot {
    // One snapshot per alive body part. Each body part's OCG lives in its
    // own local frame; for branches that frame is offset from the root by
    // the attachment origin (parent_idx = 0 in the current single-level
    // hierarchy). We approximate the world position by adding the attachment
    // origin in the parent's frame, then transforming by the root.
    //
    // This ignores the branch's own `attachment.rotation` for now — once
    // body-part rotation is animated, branch cell positions should be
    // computed via `parent_world * Translate(origin) * Rotate * cell_local`.
    // Until then, attachment.rotation is Quat::IDENTITY and the simpler
    // composition is exact.
    let mut body_parts: Vec<BodyPartSnapshot> = Vec::new();
    for (idx, bp) in organism.body_parts.iter().enumerate() {
        if !bp.is_alive() { continue; }
        let part_origin_local = bp.attachment.as_ref()
            .map(|a| a.origin_local)
            .unwrap_or(Vec3::ZERO);
        let cells_world: Vec<Vec3> = bp.ocg.iter()
            .map(|(_, p, _)| transform.transform_point(part_origin_local + *p))
            .collect();
        if cells_world.is_empty() { continue; }
        let centroid = cells_world.iter().copied().fold(Vec3::ZERO, |a, b| a + b)
                       / cells_world.len() as f32;
        let bound_radius = cells_world.iter()
            .map(|&p| (p - centroid).length())
            .fold(0.0_f32, f32::max)
            + CELL_COLLISION_RADIUS;
        body_parts.push(BodyPartSnapshot {
            index: idx,
            world_centroid: centroid,
            bound_radius,
            cells_world,
        });
    }

    OrganismSnapshot {
        entity,
        root_world_pos: transform.translation,
        is_photo,
        is_carnivore,
        sliding_movement: organism.sliding_movement,
        body_parts,
    }
}


// ── Main system ──────────────────────────────────────────────────────────────

pub fn apply_organism_collision(
    time:               Res<Time>,
    mut timer:          ResMut<OrganismCollisionTimer>,
    mut contact_events: MessageWriter<OrganismContactEvent>,
    mut params: ParamSet<(
        Query<(
            &Organism, &Transform, Entity,
            Has<crate::colony::Photoautotroph>,
            Has<crate::colony::Carnivore>,
        ), With<OrganismRoot>>,
        Query<&mut Transform, With<OrganismRoot>>,
    )>,
) {
    timer.timer.tick(time.delta());
    if !timer.timer.just_finished() { return; }

    // ── 1. Snapshot every organism (immutable read of (Organism, Transform))
    let snapshots: Vec<OrganismSnapshot> = params.p0().iter()
        .map(|(o, t, e, is_photo, is_carn)| snapshot(e, o, t, is_photo, is_carn))
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

    // Dedup set for emitted events. Positional separation accumulator
    // removed — see the comment in the narrow phase below.
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

                // Skip any pair involving a limb-based organism. Their
                // true body-part world positions live in Avian, not in
                // `root_transform × local`, so this system would test
                // collisions against stale geometry. `avian_setup::
                // emit_limb_contact_events` handles every limb-involved
                // contact via Avian's narrow phase instead — routing it
                // through the same `OrganismContactEvent` that predation
                // already consumes. (Sliding↔sliding pairs stay here.)
                if !snap_a.sliding_movement || !snap_b.sliding_movement {
                    continue;
                }

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

                                // Positional separation removed. Every
                                // push variant we tried (speed-scaled
                                // deflection, half-overlap shove,
                                // sessile-aware routing, predator-prey
                                // skip) introduced a different failure
                                // mode. Most recently: two herbivores
                                // converging on the same photo got
                                // shoved off-axis perpendicular to
                                // their shared photo-axis on every
                                // collision tick, while the brain
                                // pulled them back; the result was an
                                // axis-locked slide-on-one-spot that
                                // never released until the organism
                                // starved.
                                //
                                // Conclusion: the collision system now
                                // ONLY emits events (so predation
                                // continues to fire). Geometric overlap
                                // between two herbivores is tolerated
                                // — they stack visually for the brief
                                // window before predation finishes the
                                // shared photo, then naturally
                                // separate as they re-target.
                            }
                        }
                    }
                }
            }
        }
    }

    // No push-application step — see the narrow-phase comment for
    // why positional separation was removed. `params.p1()` (the
    // `&mut Transform` query) is now unused; we keep it on the
    // signature so the previous pushed-write code can be reinstated
    // by uncommenting if a future failure mode requires it.
    let _ = params;
}
