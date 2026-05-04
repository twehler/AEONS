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
use crate::cell::CellType;
use crate::volumetric_growth::grow_ocg_one_step;


/// Return a child OCG: parent's genome extended by one cell grown via the
/// volumetric growth algorithm. Called once per reproduction event so every
/// child is born one cell larger than its parent.
pub fn mutate_ocg(
    parent_ocg: &[(usize, Vec3, CellType)],
    rng: &mut impl Rng,
) -> Vec<(usize, Vec3, CellType)> {
    grow_ocg_one_step(parent_ocg, rng)
}
