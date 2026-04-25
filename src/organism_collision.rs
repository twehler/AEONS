use bevy::prelude::*;
use crate::colony::*;
use crate::cell::GLOBAL_CELL_SIZE;
use std::collections::{HashMap, HashSet};

// ── Events ────────────────────────────────────────────────────────────────────

// Fired whenever two organisms physically touch. 
// Used by predation.rs and any future interaction logic.
#[derive(Message, Clone, Copy)]
pub struct OrganismContactEvent {
    pub a: Entity,
    pub b: Entity,
}

// ── Constants ─────────────────────────────────────────────────────────────────

// Broad phase: organism root positions must be closer than this for any
// further checks to run. Set generously — it is only a cheap distance test.
const ORGANISM_BROAD_RADIUS: f32 = 10.0;

// Mid phase: two collections must be closer than this (starter-to-starter)
// for their cells to enter the narrow phase.
const COLLECTION_MID_RADIUS: f32 = 5.0;

// Narrow phase: two cells are in contact when their centres are closer than this.
const CELL_CONTACT_RADIUS: f32 = GLOBAL_CELL_SIZE * 1.1;

// Fraction of movement speed converted into a deflection push on contact.
const PUSH_STRENGTH: f32 = 1.0;

const ORGANISM_COLLISION_TIMER: f32 = 0.1; // in seconds

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
            timer: Timer::from_seconds(ORGANISM_COLLISION_TIMER, TimerMode::Repeating),
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
// from the grown OCG slice (ocg[..grown_cell_count]).
// Using the grown subset instead of active_cells means juvenile organisms with
// partially grown bodies are handled correctly — only existing cells collide.
fn snapshot(
    entity:    Entity,
    organism:  &Organism,
    transform: &Transform,
) -> OrganismSnapshot {
    // Build a map from CollectionId → list of cell world positions
    let mut coll_cells: HashMap<CollectionId, (Vec3, Vec<Vec3>)> = HashMap::new();

    for entry in &organism.ocg[..organism.grown_cell_count] {
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

pub fn apply_organism_collision(
    time:       Res<Time>,
    mut timer:  ResMut<OrganismCollisionTimer>,
    mut contact_events: MessageWriter<OrganismContactEvent>, // The event writer
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
    
    // Track which pairs have already triggered an event this frame 
    // to prevent spamming if multiple cells touch.
    let mut emitted_events: HashSet<(Entity, Entity)> = HashSet::new();

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
                for (_, starter_a, cells_a) in &snap_a.collections {
                    for (_, starter_b, cells_b) in &snap_b.collections {

                        let collection_dist = starter_a.distance(*starter_b);
                        if collection_dist >= COLLECTION_MID_RADIUS {
                            continue;
                        }

                        // ── Narrow phase: cell-level spatial hash ─────────────
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

                                    // ── FIRE COLLISION EVENT ──────────────────────
                                    // Order entities deterministically so (A,B) and (B,A) hash identically
                                    let pair = if snap_a.entity < snap_b.entity {
                                        (snap_a.entity, snap_b.entity)
                                    } else {
                                        (snap_b.entity, snap_a.entity)
                                    };

                                    if !emitted_events.contains(&pair) {
                                        emitted_events.insert(pair);
                                        contact_events.write(OrganismContactEvent {
                                            a: snap_a.entity,
                                            b: snap_b.entity,
                                        });
                                    }

                                    // ── PHYSICS PUSH ──────────────────────────────
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
    let mut write_query = params.p1();
    for (entity, push_vec) in &pushes {
        let Ok(mut organism) = write_query.get_mut(*entity) else { continue };
        let new_dir = organism.movement_direction + *push_vec;
        if new_dir.length_squared() > 1e-6 {
            organism.movement_direction = new_dir.normalize();
        }
    }
}
