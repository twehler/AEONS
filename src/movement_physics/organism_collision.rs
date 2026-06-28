// Organism-vs-organism contact detection (sliding↔sliding only).
//
// Three-phase pipeline:
//   1. Broad  — root-position spatial hash, distance test.
//   2. Mid    — body-part bounding-sphere overlap.
//   3. Narrow — cell-vs-cell sphere intersection.
//
// Emits one `OrganismContactEvent` per touching body-part pair (carrying the
// indices) so `predation.rs` scopes consumption to the part physically reached.

use bevy::prelude::*;
use std::collections::{HashMap, HashSet};
use rayon::prelude::*;

use crate::cell::*;
use crate::colony::*;


// ── Events ───────────────────────────────────────────────────────────────────

/// Fired when a body part of one organism touches a body part of another.
/// Consumed by `predation.rs`.
#[derive(Message, Clone, Copy)]
pub struct OrganismContactEvent {
    pub a:           Entity,
    pub b:           Entity,
    pub body_part_a: usize,
    pub body_part_b: usize,
}


// ── Constants ────────────────────────────────────────────────────────────────

use crate::simulation_settings::{ORGANISM_BROAD_RADIUS, COLLISION_TICK};


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

/// Per-body-part snapshot; cell world positions pre-computed so the narrow
/// phase doesn't re-transform.
struct BodyPartSnapshot {
    index:           usize,
    world_centroid:  Vec3,
    bound_radius:    f32, // mid-phase sphere radius
    cells_world:     Vec<Vec3>,
}

struct OrganismSnapshot {
    entity:         Entity,
    root_world_pos: Vec3,
    /// `is_photo` prunes plant↔plant pairs (never predatory); `is_carnivore`
    /// is retained for the event-only path.
    is_photo:       bool,
    is_carnivore:   bool,
    /// `false` for limb organisms. This system computes cell positions as
    /// `root_transform × local`, valid only for sliding organisms; limb
    /// pairs are skipped here and handled by `emit_limb_contact_events`.
    sliding:        bool,
    body_parts:     Vec<BodyPartSnapshot>,
}

fn snapshot(
    entity:       Entity,
    organism:     &Organism,
    transform:    &Transform,
    is_photo:     bool,
    is_carnivore: bool,
) -> OrganismSnapshot {
    // One snapshot per alive body part. World position = root transform
    // applied to (attachment origin + cell local). Ignores
    // `attachment.rotation` — exact while it stays Quat::IDENTITY.
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
        sliding: organism.movement_mode.is_sliding(),
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

    // ── 1. Snapshot every organism (PARALLEL). `snapshot` transforms every cell
    //    to world space (O(cells)) independently per organism. The query rows are
    //    materialised into a Vec of references first — `&Organism`/`&Transform`
    //    are Send+Sync (no interior mutability) — then mapped over rayon. The block
    //    scope keeps the borrowing query alive only for the snapshot build.
    let snapshots: Vec<OrganismSnapshot> = {
        let q = params.p0();
        let rows: Vec<(&Organism, &Transform, Entity, bool, bool)> = q.iter().collect();
        rows.par_iter()
            .map(|&(o, t, e, is_photo, is_carn)| snapshot(e, o, t, is_photo, is_carn))
            .collect()
    };

    if snapshots.is_empty() { return; }

    // ── 2. Build the broad-phase spatial hash on root positions
    let mut organism_grid: HashMap<[i32; 3], Vec<usize>> = HashMap::new();
    for (i, snap) in snapshots.iter().enumerate() {
        organism_grid
            .entry(grid_key(snap.root_world_pos, ORGANISM_BROAD_RADIUS))
            .or_default()
            .push(i);
    }

    let cell_contact_d  = CELL_COLLISION_RADIUS * 2.0;
    let cell_contact_d2 = cell_contact_d * cell_contact_d;
    let broad_r2        = ORGANISM_BROAD_RADIUS * ORGANISM_BROAD_RADIUS;

    // ── 3. Pair loop — broad → mid → narrow, PARALLEL over the outer index. ──
    // Each unordered pair (idx_a, idx_b<idx_a excluded) is processed under exactly
    // ONE idx_a (the smaller), so the dedup HashSet — needed because the inner
    // cell loops can re-emit the same body-part-pair key — is per-task and needs no
    // cross-thread coordination. rayon preserves sequential order in `collect`, so
    // the emitted-event order (and thus predation's first-claim-wins behaviour) is
    // identical to the serial version. `snapshots`/`organism_grid` are read-only here.
    let events: Vec<OrganismContactEvent> = (0..snapshots.len()).into_par_iter()
        .flat_map_iter(|idx_a| {
            let snap_a = &snapshots[idx_a];
            let mut local: Vec<OrganismContactEvent> = Vec::new();
            let mut emitted: HashSet<(Entity, Entity, usize, usize)> = HashSet::new();
            let nkeys = neighbour_keys(grid_key(snap_a.root_world_pos, ORGANISM_BROAD_RADIUS));

            for nkey in nkeys {
                let Some(bucket) = organism_grid.get(&nkey) else { continue };

                for &idx_b in bucket {
                    if idx_b <= idx_a { continue; }
                    let snap_b = &snapshots[idx_b];

                    // Skip pairs involving a limb organism (their true positions
                    // live in Avian, not `root × local`); `handle_limb_collisions`
                    // handles those. Sliding↔sliding pairs stay here.
                    if !snap_a.sliding || !snap_b.sliding { continue; }

                    // Two photoautotrophs can never predate each other, so the only
                    // outcome of testing a plant↔plant pair is an event `predation.rs`
                    // immediately discards. Skip them — this drops the bulk of the
                    // cell-vs-cell work in a dense seated plant field.
                    if snap_a.is_photo && snap_b.is_photo { continue; }

                    // Broad phase: skip pairs whose roots are far apart.
                    let dx = snap_a.root_world_pos - snap_b.root_world_pos;
                    if dx.length_squared() >= broad_r2 { continue; }

                    // Mid phase: per-body-part bounding-sphere overlap.
                    for bp_a in &snap_a.body_parts {
                        for bp_b in &snap_b.body_parts {
                            let bp_d  = bp_a.world_centroid - bp_b.world_centroid;
                            let r_sum = bp_a.bound_radius + bp_b.bound_radius;
                            if bp_d.length_squared() >= r_sum * r_sum { continue; }

                            // Narrow phase: cell-pair sphere intersection.
                            // Quadratic but cheap (<50 cells/part, broad/mid prune).
                            for &ca in &bp_a.cells_world {
                                for &cb in &bp_b.cells_world {
                                    let d2 = (cb - ca).length_squared();
                                    if d2 >= cell_contact_d2 || d2 < 1e-6 { continue; }

                                    // Canonical key (sort by entity) so the pair
                                    // collapses to one event regardless of order.
                                    let key = if snap_a.entity < snap_b.entity {
                                        (snap_a.entity, snap_b.entity, bp_a.index, bp_b.index)
                                    } else {
                                        (snap_b.entity, snap_a.entity, bp_b.index, bp_a.index)
                                    };

                                    if emitted.insert(key) {
                                        local.push(OrganismContactEvent {
                                            a: snap_a.entity,
                                            b: snap_b.entity,
                                            body_part_a: bp_a.index,
                                            body_part_b: bp_b.index,
                                        });
                                    }
                                    // Events only — no positional push (every push
                                    // variant introduced a worse failure mode).
                                }
                            }
                        }
                    }
                }
            }
            local
        })
        .collect();

    for ev in events {
        contact_events.write(ev);
    }

    // No push step (events only). `params.p1()` (`&mut Transform`) is unused
    // but kept on the signature for easy reinstatement.
    let _ = params;
}
