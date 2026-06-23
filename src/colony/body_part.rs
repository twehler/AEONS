// Body-part branching — appendage spawn logic for reproduction.
//
// `should_branch` decides (20%) whether to grow a new body part instead of
// extending the existing one. The new part attaches at a chosen origin on the
// parent with a rotation Quat for the pivot.
//
// Each body part owns its own OCG: `build_mesh_from_ocg` welds vertices across
// all positions, so a shared OCG would fuse topology and break rotation.
//
// `pick_attachment`'s "outward direction" is the lattice slot most aligned
// with the outward normal at the origin cell; offsetting the seed by
// `EDGE_LEN` along it lands the new part as a clean FCC-neighbour.

use bevy::prelude::*;
use rand::prelude::*;

use crate::cell::*;


/// Probability that a reproduction event spawns a new body part on the child
/// instead of extending the existing one. 0.2 = 20%.
pub const NEW_BODY_PART_PROBABILITY: f32 = 0.2;

/// Max body parts INCLUDING the base body (`body_parts[0]`). 3 = base + one
/// mirrored appendage pair.
pub const MAX_BODY_PARTS: usize = 3;

/// Min +X any cell in a Bilateral right half may occupy. `2/√3 ≈ 1.1547`
/// (half-distance between axis-aligned RD neighbours / RD long-radius), so no
/// cell touches or crosses the YZ mirror plane and the halves meet flush.
// = 0.5·center_scale(EDGE_LEN) = the x of the first off-midline lattice column.
// Derives from the master `simulation_settings::GEOMETRY_SCALE`: 1.154_700_5 is
// its value at scale 1.0 (= 2/√3), so it tracks the knob ∝ scale.
pub const MIN_X_BILATERAL: f32 = 1.154_700_5 * crate::simulation_settings::GEOMETRY_SCALE;

/// RD edge length. Mirrors `volumetric_growth::EDGE_LEN` (private there) —
/// keep in sync or the seed-cell offset stops matching the lattice.
const EDGE_LEN: f32 = 1.0;


/// How a body part hangs off its parent. The child entity is a Bevy child of
/// the parent body-part entity with `Transform { translation: origin_local,
/// rotation, .. }`, so Bevy's transform propagation swings it around the pivot.
#[derive(Clone, Debug)]
pub struct Attachment {
    /// Index into `Organism::body_parts` of the parent part.
    pub parent_idx:   usize,
    /// Pivot point in the parent part's local frame.
    pub origin_local: Vec3,
    /// Orientation relative to the parent (initially `Quat::IDENTITY`).
    pub rotation:     Quat,
}


/// 20% coin flip: should this offspring grow a new body part?
#[inline]
pub fn should_branch(rng: &mut impl Rng) -> bool {
    rng.random::<f32>() < NEW_BODY_PART_PROBABILITY
}


/// Pick an attachment origin on the parent for a new branch.
/// Returns `(origin_in_parent_local, outward_dir)` where `outward_dir` is the
/// unit lattice slot along which to offset the seed for a flush FCC-neighbour.
/// Falls back to `(ZERO, Y)` for a single-cell parent.
pub fn pick_attachment(
    parent_ocg: &[(usize, Vec3, CellType)],
    rng:        &mut impl Rng,
) -> (Vec3, Vec3) {
    if parent_ocg.is_empty() {
        return (Vec3::ZERO, Vec3::Y);
    }

    let centroid: Vec3 = parent_ocg.iter().map(|(_, p, _)| *p).sum::<Vec3>()
        / parent_ocg.len() as f32;

    // Score cells by distance from centroid; pick randomly among the top tier
    // so sibling branches don't all land on the same cell.
    let mut scored: Vec<(usize, f32)> = parent_ocg.iter()
        .enumerate()
        .map(|(i, (_, p, _))| (i, (*p - centroid).length_squared()))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_n = scored.len().min(3).max(1);
    let pick = rng.random_range(0..top_n);
    let chosen_idx = scored[pick].0;
    let origin = parent_ocg[chosen_idx].1;

    // Outward dir: centroid→cell, snapped to the nearest lattice slot so the
    // seed lands on a real RD adjacency. (SLOT_DIRS isn't pub, so the 18
    // directions are duplicated in `snap_to_lattice_slot`.)
    let outward_raw = (origin - centroid).normalize_or_zero();
    let outward = if outward_raw == Vec3::ZERO {
        Vec3::Y
    } else {
        snap_to_lattice_slot(outward_raw)
    };

    (origin, outward)
}


/// Build a fresh branch BodyPart. The seed sits at `outward_dir * EDGE_LEN`
/// in the new part's frame; since the part's origin coincides with
/// `attachment.origin_local`, this lands the seed flush against the parent as
/// an FCC-neighbour. `regrowable: true` so future ticks can grow it.
pub fn create_branch_body_part(
    seed_cell_type: CellType,
    attachment:     Attachment,
    outward_dir:    Vec3,
) -> BodyPart {
    let seed_local = outward_dir * EDGE_LEN;
    let cells = vec![Cell::new(seed_local, seed_cell_type)];
    let ocg = vec![(0usize, seed_local, seed_cell_type)];

    BodyPart {
        kind:         BodyPartKind::Limb,
        local_offset: Vec3::ZERO,
        cells,
        ocg,
        attachment:   Some(attachment),
        consumed:     false,
        debug_blue:   true,
        regrowable:   true,
    }
}


// ── Bilateral symmetry helpers ───────────────────────────────────────────────

/// Mirror an OCG across the YZ-plane (negate X). Types and indices preserved.
pub fn mirror_ocg_x(
    ocg: &[(usize, Vec3, CellType)],
) -> Vec<(usize, Vec3, CellType)> {
    ocg.iter()
        .map(|(idx, pos, ct)| (*idx, Vec3::new(-pos.x, pos.y, pos.z), *ct))
        .collect()
}

/// |x| below which a cell is ON the bilateral mirror plane. Lattice x are
/// multiples of `MIN_X_BILATERAL` (≈1.1547), so any small epsilon separates a
/// midline cell (x = 0) from the innermost paramedian column.
pub const BILATERAL_MIDLINE_EPS: f32 = 1e-3;

/// Build the LEFT half from a right-half OCG (`x ≥ 0`), mirroring across YZ.
/// Midline cells (`|x| < BILATERAL_MIDLINE_EPS`) are their own mirror and are
/// SKIPPED — duplicating them would make the mesh dedup erase them. Indices
/// and types preserved; callers renumber after concatenation.
pub fn mirror_right_to_left(
    right_ocg: &[(usize, Vec3, CellType)],
) -> Vec<(usize, Vec3, CellType)> {
    right_ocg.iter()
        .filter(|(_, p, _)| p.x > BILATERAL_MIDLINE_EPS)
        .map(|(idx, pos, ct)| (*idx, Vec3::new(-pos.x, pos.y, pos.z), *ct))
        .collect()
}

/// Build the single bilateral body part from a right-half OCG (`x ≥ 0`).
/// Returns ONE `BodyPart` whose OCG holds both halves (right cells, then their
/// mirror); `build_mesh_from_ocg`'s weld + drop-interior-faces pipeline runs
/// over the combined list.
///
/// The halves connect only through **midline cells** (`x = 0`): a +X cell and
/// its −X mirror differ by a pure-X displacement, which is never RD-face-
/// adjacent (only the 12 FCC slots share a face), but a midline cell like
/// `(0,t,0)` is face-adjacent to both `(t,0,0)` and `(−t,0,0)` and so welds
/// them. Without midline cells the halves touch only at points.
pub fn bilateral_body_part_from_right_ocg(
    right_ocg: &[(usize, Vec3, CellType)],
) -> BodyPart {
    let left_ocg = mirror_right_to_left(right_ocg);

    // Right cells then left mirrors, renumbered to a contiguous [0..N) ledger.
    let combined_ocg: Vec<(usize, Vec3, CellType)> = right_ocg.iter()
        .chain(left_ocg.iter())
        .enumerate()
        .map(|(i, (_, p, ct))| (i, *p, *ct))
        .collect();

    let cells: Vec<Cell> = combined_ocg.iter()
        .map(|(_, p, ct)| Cell::new(*p, *ct))
        .collect();

    BodyPart {
        kind:         BodyPartKind::Body,
        local_offset: Vec3::ZERO,
        cells,
        ocg:          combined_ocg,
        attachment:   None,
        consumed:     false,
        debug_blue:   false,
        regrowable:   true,
    }
}


// ── Internal helpers ─────────────────────────────────────────────────────────

/// Snap `dir` to the nearest of the 18 RD lattice slot directions
/// (6 axis-aligned + 12 FCC face-diagonal). Returned vector is unit-length.
fn snap_to_lattice_slot(dir: Vec3) -> Vec3 {
    const SLOTS: [Vec3; 18] = [
        // 6 axis-aligned
        Vec3::new( 1.0,  0.0,  0.0), Vec3::new(-1.0,  0.0,  0.0),
        Vec3::new( 0.0,  1.0,  0.0), Vec3::new( 0.0, -1.0,  0.0),
        Vec3::new( 0.0,  0.0,  1.0), Vec3::new( 0.0,  0.0, -1.0),
        // 12 FCC face-diagonal (normalised below)
        Vec3::new( 0.0,  0.5,  0.5), Vec3::new( 0.5,  0.5,  0.0),
        Vec3::new( 0.5,  0.0,  0.5), Vec3::new( 0.0, -0.5,  0.5),
        Vec3::new( 0.0,  0.5, -0.5), Vec3::new( 0.5,  0.0, -0.5),
        Vec3::new(-0.5,  0.0,  0.5), Vec3::new( 0.5, -0.5,  0.0),
        Vec3::new(-0.5,  0.5,  0.0), Vec3::new(-0.5, -0.5,  0.0),
        Vec3::new(-0.5,  0.0, -0.5), Vec3::new( 0.0, -0.5, -0.5),
    ];
    let mut best = SLOTS[0];
    let mut best_dot = f32::NEG_INFINITY;
    for &slot in &SLOTS {
        let n = slot.normalize();
        let d = n.dot(dir);
        if d > best_dot { best_dot = d; best = n; }
    }
    best
}
