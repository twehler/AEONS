// Physiology — global cell-level rules.
//
// `PhysiologyPlugin` is the home for rules that act on individual cells
// (`Cell::cell_energy`, `Cell::neighbour_count`, `PhotosyntheticCell`), as
// opposed to the organism-level bookkeeping in `energy.rs`. The split is
// deliberate:
//
//   * `energy.rs` runs at the whole-organism level: organism reservoirs,
//     metabolism, movement / climbing costs, starvation despawn.
//
//   * `physiology.rs` runs at the cell level: each `Cell` carries its own
//     `cell_energy` and (for photos) a cached `PhotosyntheticCell`. The
//     plugin's per-tick system credits the organism with the sum of every
//     photo cell's cached `energy_production`.
//
// Cell-level data that depends on cell composition (neighbour counts,
// photosynthesis caches) is recomputed only when cells are added or
// removed — never per tick. `recompute_body_parts` is the single entry
// point spawn / reproduction / krishi callers use to bring those caches
// in sync with the cell list they just built.

use bevy::prelude::*;

use crate::cell::*;
use crate::colony::*;
use crate::energy::PHOTO_PRODUCTION_PER_CELL;


// ── Tunables ─────────────────────────────────────────────────────────────────

/// How often the physiology tick runs. 0.5 s matches the energy tick so
/// per-cell and per-organism updates stay roughly in phase, simplifying
/// reasoning about which lags which.
const PHYSIOLOGY_TICK_INTERVAL: f32 = 0.5;

/// Hard ceiling on per-cell energy. Cells never exceed this regardless of
/// how much they would otherwise gain in one tick — keeps the value
/// comparable across cell types and prevents overflow into pathological
/// regimes that future rules would have to special-case.
pub const MAX_CELL_ENERGY: f32 = 1.0;

/// Distance² window for two cells (in the same body part's local frame) to
/// count as RD lattice neighbours. Axis-aligned RD adjacency lands at
/// distance 4/√3 ≈ 2.309 (d² ≈ 5.33); FCC face-diagonal adjacency lands at
/// (4/√3) · √0.5 ≈ 1.633 (d² ≈ 2.67). The lower bound (0.1) excludes the
/// cell from being its own neighbour; the upper bound (6.0) sits safely
/// above the axis-aligned d² with a small slack for float drift.
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

/// Recompute neighbour counts and `PhotosyntheticCell` caches for every cell
/// in every body part of `body_parts`. Call this once after assembling a
/// cell list — at spawn, after appending a mutated cell, after building a
/// branch — and the per-tick physiology system is then O(photo cells)
/// instead of O(all cells × all cells).
pub fn recompute_body_parts(body_parts: &mut [BodyPart]) {
    for bp in body_parts.iter_mut() {
        recompute_body_part(bp);
    }
}

/// Single-body-part variant — useful when only one part has changed.
pub fn recompute_body_part(bp: &mut BodyPart) {
    // Snapshot positions so the inner loop reads from a stable slice while
    // we mutate cells one by one.
    let positions: Vec<Vec3> = bp.cells.iter().map(|c| c.local_pos).collect();

    for (i, cell) in bp.cells.iter_mut().enumerate() {
        let n = count_rd_neighbours(&positions, i);
        cell.neighbour_count = n;
        cell.photo = match cell.cell_type {
            CellType::Photo                            => Some(PhotosyntheticCell::new(n, PHOTO_PRODUCTION_PER_CELL)),
            CellType::NonPhoto | CellType::Placeholder => None,
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

/// Credit each photoautotroph with the sum of every alive photo cell's
/// cached `PhotosyntheticCell::energy_production` per tick. Heterotrophs
/// (and any organism with `photo_cell_count == 0`) are skipped at the
/// cached-counter check — no inner-loop cost paid for them.
///
/// Energy is clamped to the organism's max storage (cell-count × per-cell
/// capacity) by `energy.rs` on its next tick, so we don't bother clamping
/// here.
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

        // Shadowed photo cells produce half their nominal energy.
        // `Organism::in_sunlight` is maintained by
        // `photosynthesis::update_sunlight` in PreUpdate.
        if !organism.in_sunlight {
            produced *= 0.5;
        }

        organism.energy += produced;
    }
}
