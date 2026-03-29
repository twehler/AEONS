use bevy::prelude::*;
use crate::colony::*;
use crate::cell::GLOBAL_CELL_SIZE;
use std::collections::HashMap;

// ── Constants ─────────────────────────────────────────────────────────────────

// Broad phase: organism root positions must be closer than this for any
// further checks to run. Set generously — it is only a cheap distance test.
const ORGANISM_BROAD_RADIUS: f32 = 10.0;

// Mid phase: two collections must be closer than this (starter-to-starter)
// for their cells to enter the narrow phase.
const COLLECTION_MID_RADIUS: f32 = 5.0;

// Narrow phase: two cells are in contact when their centres are closer than this.
const CELL_CONTACT_RADIUS: f32 = GLOBAL_CELL_SIZE;

// Fraction of movement speed converted into a deflection push on contact.
const PUSH_STRENGTH: f32 = 0.5;

// ── Timer resource ────────────────────────────────────────────────────────────

// Running the full collision pipeline every frame is expensive at scale.
// The timer throttles it — reduce the interval for more responsive collision,
// increase it to save CPU when organism counts are high.
#[derive(Resource)]
pub struct OrganismCollisionTimer {
    pub timer: Timer,
}

impl Default for OrganismCollisionTimer {
    fn default() -> Self {
        Self {
            timer: Timer::from_seconds(0.05, TimerMode::Repeating),
        }
    }
}

// ── Spatial hash helpers ──────────────────────────────────────────────────────

fn grid_key(pos: Vec3, bucket_size: f32) -> [i32; 3] {
    [
        (pos.x / bucket_size).floor() as i32,
        (pos.y / bucket_size).floor() as i32,
        (pos.z / bucket_size).floor() as i32,
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

// ── Snapshot types ────────────────────────────────────────────────────────────

// Immutable snapshot of one organism captured before any mutation happens.
// Avoids re-borrowing the query during the push-accumulation phase.
struct OrganismSnapshot {
    entity:             Entity,
    root_world_pos:     Vec3,
    movement_direction: Vec3,
    movement_speed:     f32,
    // Per-collection: starter world position + per-cell world positions.
    // Indexed as: collections[i] = (CollectionId, starter_world_pos, [cell_world_pos...])
    collections: Vec<(CollectionId, Vec3, Vec<Vec3>)>,
}

// Builds a snapshot for one organism by computing all cell world positions
// from grown_cells (the OCG slice representing currently present cells).
// Using grown_cells instead of active_cells means juvenile organisms with
// partially grown bodies are handled correctly — only existing cells collide.
fn snapshot(
    entity:    Entity,
    organism:  &Organism,
    transform: &Transform,
) -> OrganismSnapshot {
    // Build a map from CollectionId → list of cell world positions
    let mut coll_cells: HashMap<CollectionId, (Vec3, Vec<Vec3>)> = HashMap::new();

    for entry in &organism.grown_cells {
        if let Some(collection) = organism.collections.get(&entry.collection_id) {
            // mesh_space_pos is the cell's local position relative to the organism root
            let mesh_space_pos = collection.starter_cell_position + Vec3::from(entry.offset);
            let world_pos      = transform.transform_point(mesh_space_pos);
            let starter_world  = transform.transform_point(collection.starter_cell_position);

            let entry_ref = coll_cells
                .entry(entry.collection_id)
                .or_insert_with(|| (starter_world, Vec::new()));
            entry_ref.1.push(world_pos);
        }
    }

    let collections = coll_cells
        .into_iter()
        .map(|(id, (starter, cells))| (id, starter, cells))
        .collect();

    OrganismSnapshot {
        entity,
        root_world_pos: transform.translation,
        movement_direction: organism.movement_direction,
        movement_speed: organism.movement_speed,
        collections,
    }
}

// ── Main system ───────────────────────────────────────────────────────────────

// Three-phase hierarchical collision:
//
//   Broad phase  — organism root vs organism root (O(n log n) with spatial hash)
//                  Skips any pair whose roots are farther than ORGANISM_BROAD_RADIUS.
//
//   Mid phase    — collection starter vs collection starter (only for broad-passing pairs)
//                  Skips any collection pair farther than COLLECTION_MID_RADIUS.
//                  This is the key performance gate: organisms with many collections
//                  only pay for per-cell work on the collections that are actually close.
//
//   Narrow phase — cell vs cell (only for mid-passing collection pairs)
//                  Uses a spatial hash of one side's cells, probes with the other side.
//                  Produces deflection pushes accumulated into movement_direction.
//
// Pushes are applied after all detection is complete to avoid order-dependent results.
pub fn apply_organism_collision(
    time:       Res<Time>,
    mut timer:  ResMut<OrganismCollisionTimer>,
    mut params: ParamSet<(
        Query<(&Organism, &Transform, Entity), With<OrganismRoot>>,
        Query<&mut Organism, With<OrganismRoot>>,
    )>,
) {
    timer.timer.tick(time.delta());
    if !timer.timer.just_finished() {
        return;
    }

    // ── Snapshot all organisms ────────────────────────────────────────────────
    // Capture everything we need immutably before entering the collision loop.
    // This avoids mid-loop re-borrows of the query and makes the borrow checker
    // happy when we later switch to the mutable query for push application.
    let snapshots: Vec<OrganismSnapshot> = params.p0()
        .iter()
        .map(|(organism, transform, entity)| snapshot(entity, organism, transform))
        .collect();

    // ── Broad phase spatial hash (organism roots) ─────────────────────────────
    let mut organism_grid: HashMap<[i32; 3], Vec<usize>> = HashMap::new();
    for (idx, snap) in snapshots.iter().enumerate() {
        organism_grid
            .entry(grid_key(snap.root_world_pos, ORGANISM_BROAD_RADIUS))
            .or_default()
            .push(idx);
    }

    // Accumulate deflection pushes: entity → total push vector
    let mut pushes: HashMap<Entity, Vec3> = HashMap::new();

    // ── Pair loop ─────────────────────────────────────────────────────────────
    for (idx_a, snap_a) in snapshots.iter().enumerate() {
        let key_a        = grid_key(snap_a.root_world_pos, ORGANISM_BROAD_RADIUS);
        let nkeys        = neighbour_keys(key_a);

        for nkey in nkeys {
            let Some(bucket) = organism_grid.get(&nkey) else { continue };

            for &idx_b in bucket {
                // Process each pair only once (lower index is always "a")
                if idx_b <= idx_a {
                    continue;
                }

                let snap_b = &snapshots[idx_b];

                // ── Broad phase check ─────────────────────────────────────────
                let organism_dist = snap_a.root_world_pos.distance(snap_b.root_world_pos);
                if organism_dist >= ORGANISM_BROAD_RADIUS {
                    continue;
                }

                // ── Mid phase: collection-pair loop ───────────────────────────
                for (id_a, starter_a, cells_a) in &snap_a.collections {
                    for (id_b, starter_b, cells_b) in &snap_b.collections {

                        let collection_dist = starter_a.distance(*starter_b);
                        if collection_dist >= COLLECTION_MID_RADIUS {
                            continue;
                        }

                        // ── Narrow phase: cell-level spatial hash ─────────────
                        // Hash side-A cells, probe with side-B cells.
                        // Bucket size = CELL_CONTACT_RADIUS so a single
                        // neighbour_keys probe covers all possible contacts.
                        let mut cell_grid: HashMap<[i32; 3], Vec<Vec3>> = HashMap::new();
                        for &cell_pos_a in cells_a {
                            cell_grid
                                .entry(grid_key(cell_pos_a, CELL_CONTACT_RADIUS))
                                .or_default()
                                .push(cell_pos_a);
                        }

                        for &cell_pos_b in cells_b {
                            let bkey   = grid_key(cell_pos_b, CELL_CONTACT_RADIUS);
                            let nkeys2 = neighbour_keys(bkey);

                            for nk in nkeys2 {
                                let Some(a_cells) = cell_grid.get(&nk) else { continue };

                                for &cell_pos_a in a_cells {
                                    let dist = cell_pos_b.distance(cell_pos_a);
                                    if dist >= CELL_CONTACT_RADIUS || dist < 1e-6 {
                                        continue;
                                    }

                                    // Push direction: B away from A
                                    let push_dir = (cell_pos_b - cell_pos_a).normalize();

                                    // Strength scales linearly with penetration depth
                                    let penetration = 1.0 - dist / CELL_CONTACT_RADIUS;
                                    let avg_speed   = (snap_a.movement_speed
                                                      + snap_b.movement_speed) / 2.0;
                                    let magnitude   = PUSH_STRENGTH * penetration * avg_speed;

                                    *pushes.entry(snap_a.entity).or_insert(Vec3::ZERO)
                                        -= push_dir * magnitude;
                                    *pushes.entry(snap_b.entity).or_insert(Vec3::ZERO)
                                        += push_dir * magnitude;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ── Apply pushes ──────────────────────────────────────────────────────────
    // p0 is fully released; p1 (mutable) is now safe to access.


    let mut write_query = params.p1();
    for (entity, push_vec) in &pushes {
    let Ok(mut organism) = write_query.get_mut(*entity) else { continue };
    let new_dir = organism.movement_direction + *push_vec;
    if new_dir.length_squared() > 1e-6 {
        organism.movement_direction = new_dir.normalize();
    }
}


}
