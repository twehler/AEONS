// Mutation — one-step OCG growth applied to every offspring at birth.
//
// Each time an organism reproduces, its child inherits the parent's OCG and
// gains exactly one new cell appended by the volumetric growth algorithm.
// Over successive generations, organisms accumulate cells: generation N has
// (seed_cells + N) cells.
//
// The new cell position is drawn randomly from the set of valid growth
// candidates on the parent's mesh frontier. Cell type is inherited from the
// seed entry (ocg[0].2), preserving trophic identity across generations.
//
// If the growth frontier is fully enclosed (no valid candidates), the child
// is born with the parent's OCG unchanged — no panic, no infinite loop.

use rand::prelude::*;
use bevy::math::Vec3;
use crate::body_part::MIN_X_BILATERAL;
use crate::cell::CellType;
use crate::volumetric_growth::{grow_ocg_one_step, grow_ocg_one_step_constrained};


/// Return a child OCG: parent's genome extended by one cell grown via the
/// volumetric growth algorithm. Called once per reproduction event so every
/// child is born one cell larger than its parent.
pub fn mutate_ocg(
    parent_ocg: &[(usize, Vec3, CellType)],
    rng: &mut impl Rng,
) -> Vec<(usize, Vec3, CellType)> {
    grow_ocg_one_step(parent_ocg, rng)
}

/// Bilateral mutation step. Grows one cell on the right half subject to
/// `x >= MIN_X_BILATERAL`. Returns the extended right OCG, or `None` if no
/// valid candidate exists this tick (caller should reuse the parent's
/// right OCG verbatim). The left half is the caller's responsibility:
/// `body_part::mirror_ocg_x(&right)` after this returns Some.
pub fn mutate_bilateral(
    parent_right_ocg: &[(usize, Vec3, CellType)],
    rng:              &mut impl Rng,
) -> Option<Vec<(usize, Vec3, CellType)>> {
    let grown = grow_ocg_one_step_constrained(parent_right_ocg, rng, MIN_X_BILATERAL);
    // grow_ocg_one_step_constrained returns the input unchanged if no
    // candidate fits; convert that "no growth this tick" signal to `None`
    // so the caller can decide whether to reuse the parent verbatim or to
    // skip the birth.
    if grown.len() == parent_right_ocg.len() { None } else { Some(grown) }
}
