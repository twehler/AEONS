// Mutation — one-step OCG growth applied to every offspring at birth.
//
// A child inherits the parent's OCG plus one new cell from the volumetric
// growth algorithm (position random on the parent's frontier; cell type
// inherited from ocg[0].2, preserving trophic identity). If the frontier is
// fully enclosed, the child is born with the parent's OCG unchanged.

use rand::prelude::*;
use bevy::math::Vec3;
use crate::cell::CellType;
use crate::volumetric_growth::{grow_ocg_one_step, grow_ocg_one_step_constrained};


/// Child OCG: parent's genome extended by one volumetrically-grown cell.
pub fn mutate_ocg(
    parent_ocg: &[(usize, Vec3, CellType)],
    rng: &mut impl Rng,
) -> Vec<(usize, Vec3, CellType)> {
    grow_ocg_one_step(parent_ocg, rng)
}

/// Bilateral mutation: grow one cell on the right half with `x >= 0` (may land
/// on the midline x=0, never crosses to the −X half). Returns the extended right
/// OCG, or `None` if no candidate fits this tick. Caller mirrors the left half
/// via `body_part::mirror_right_to_left(&right)` on Some.
pub fn mutate_bilateral(
    parent_right_ocg: &[(usize, Vec3, CellType)],
    rng:              &mut impl Rng,
) -> Option<Vec<(usize, Vec3, CellType)>> {
    // min_x = 0 (with built-in −1e-3 slack) admits midline candidates but
    // rejects any −X (left-half) slot.
    let grown = grow_ocg_one_step_constrained(parent_right_ocg, rng, 0.0);
    // Unchanged length means "no growth this tick" → None.
    if grown.len() == parent_right_ocg.len() { None } else { Some(grown) }
}
