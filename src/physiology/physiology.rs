// Physiology — cell-level rules, vs. the organism-level bookkeeping in `energy.rs`.
//
//   * `energy.rs`: whole-organism reservoirs, metabolism, movement/climb costs,
//     starvation despawn.
//   * `physiology.rs`: per-cell `cell_energy` and (for photos) cached
//     `PhotosyntheticCell`; the per-tick system credits each organism with the
//     sum of its photo cells' cached `energy_production`.
//
// Composition-dependent cell data (neighbour counts, photo caches) is recomputed
// only on add/remove, never per tick. `recompute_body_parts` is the single
// entry point spawn/reproduction callers use to resync those caches.

use bevy::prelude::*;

use crate::cell::*;
use crate::colony::*;
use crate::energy::PHOTO_PRODUCTION_PER_CELL;


// ── Tunables ─────────────────────────────────────────────────────────────────

use crate::simulation_settings::PHYSIOLOGY_TICK_INTERVAL;

/// Distance² window (same body part's local frame) for RD lattice neighbours.
/// Axis-aligned RD adjacency d² ≈ 5.33 (dist 4/√3); FCC face-diagonal d² ≈ 2.67.
/// Lower bound 0.1 excludes self; upper bound 6.0 sits just above the axis-aligned
/// d² with float-drift slack.
const NEIGHBOUR_DSQ_MIN: f32 = 0.1;
const NEIGHBOUR_DSQ_MAX: f32 = 6.0;


// ── Resources ────────────────────────────────────────────────────────────────

#[derive(Resource)]
pub struct PhysiologyTimer {
    pub timer: Timer,
}

impl Default for PhysiologyTimer {
    fn default() -> Self {
        Self {
            timer: Timer::from_seconds(PHYSIOLOGY_TICK_INTERVAL, TimerMode::Repeating),
        }
    }
}


// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct PhysiologyPlugin;

impl Plugin for PhysiologyPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(PhysiologyTimer::default());
        app.add_systems(Update, photosynthesise);
    }
}


// ── Public construction-time helpers ─────────────────────────────────────────

/// Recompute neighbour counts and `PhotosyntheticCell` caches for every cell in
/// every body part. Call once after assembling a cell list (spawn, mutation,
/// branch) so the per-tick system stays O(photo cells), not O(cells²).
pub fn recompute_body_parts(body_parts: &mut [BodyPart]) {
    for bp in body_parts.iter_mut() {
        recompute_body_part(bp);
    }
}

/// Single-body-part variant — useful when only one part has changed.
pub fn recompute_body_part(bp: &mut BodyPart) {
    // Snapshot positions so the inner loop reads a stable slice while mutating cells.
    let positions: Vec<Vec3> = bp.cells.iter().map(|c| c.local_pos).collect();

    for (i, cell) in bp.cells.iter_mut().enumerate() {
        let n = count_rd_neighbours(&positions, i);
        cell.neighbour_count = n;
        cell.photo = if cell.cell_type.is_photo() {
            Some(PhotosyntheticCell::new(n, PHOTO_PRODUCTION_PER_CELL))
        } else {
            None
        };
    }
}

/// Count cells in `positions` that are RD-adjacent to `positions[me]` —
/// neighbour distance falls inside `[NEIGHBOUR_DSQ_MIN, NEIGHBOUR_DSQ_MAX]`.
/// Capped at `MAX_RD_NEIGHBOURS` so a malformed input can't overflow the
/// downstream `u8` bookkeeping.
fn count_rd_neighbours(positions: &[Vec3], me: usize) -> u8 {
    let p = positions[me];
    let mut count: u32 = 0;
    for (j, &q) in positions.iter().enumerate() {
        if j == me { continue; }
        let d2 = (q - p).length_squared();
        if d2 > NEIGHBOUR_DSQ_MIN && d2 <= NEIGHBOUR_DSQ_MAX {
            count += 1;
        }
    }
    count.min(MAX_RD_NEIGHBOURS as u32) as u8
}


// ── Per-tick system ──────────────────────────────────────────────────────────

/// Credit each photoautotroph with the sum of its alive photo cells' cached
/// `energy_production` per tick. Organisms with `photo_cell_count == 0` are
/// skipped before the inner loop. `energy.rs` clamps to max storage next tick.
fn photosynthesise(
    time:        Res<Time>,
    mut timer:   ResMut<PhysiologyTimer>,
    mut query:   Query<&mut Organism, (With<OrganismRoot>, With<Photoautotroph>)>,
) {
    timer.timer.tick(time.delta());
    if !timer.timer.just_finished() { return; }

    for mut organism in &mut query {
        if organism.photo_cell_count <= 0 { continue; }

        let mut produced = 0.0_f32;
        for bp in organism.body_parts.iter().filter(|bp| bp.is_alive()) {
            for cell in &bp.cells {
                if let Some(ph) = &cell.photo {
                    produced += ph.energy_production;
                }
            }
        }

        // Shadowed cells produce half. `in_sunlight` is set by
        // photosynthesis::update_sunlight in PreUpdate.
        if !organism.in_sunlight {
            produced *= 0.5;
        }

        organism.energy += produced;
    }
}
