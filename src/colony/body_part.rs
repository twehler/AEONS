// Body-part branching — appendage spawn logic for reproduction.
//
// When an offspring is born, `should_branch` decides (20% probability) whether
// to grow a new body part instead of extending the existing one. The new part
// attaches to a chosen point on the parent body part (its "origin"), with a
// rotation Quat that animates around that pivot in future iterations.
//
// Each body part owns its own OCG (cell catalogue, in the part's local frame).
// Per-part OCG matters because `volumetric_growth::build_mesh_from_ocg` welds
// vertices across all positions it sees — sharing an OCG between two parts
// would fuse their mesh topology and break independent rotation.
//
// The "outward direction" returned by `pick_attachment` is the lattice slot
// (one of `dodecahedron::SLOT_DIRS`) most aligned with the outward face
// normal at the chosen origin cell. Offsetting the seed by `EDGE_LEN` along
// it places the new part as an FCC-neighbour of the parent surface cell —
// clean lattice alignment, no interior overlap.

use bevy::prelude::*;
use rand::prelude::*;

use crate::cell::*;


/// Probability that a reproduction event spawns a new body part on the child
/// instead of extending the existing one. 0.2 = 20%.
pub const NEW_BODY_PART_PROBABILITY: f32 = 0.2;

/// Maximum number of body parts an organism may have, INCLUDING the base
/// body (`body_parts[0]`). Bilateral appendages are added as mirrored
/// left/right PAIRS, so reproduction only branches while two more parts
/// still fit. 3 = base body + one symmetric appendage pair.
pub const MAX_BODY_PARTS: usize = 3;

/// Minimum +X coordinate any cell in the RIGHT body part of a Bilateral
/// organism may occupy. Equals `2/√3 ≈ 1.1547` — the half-distance between
/// two axis-aligned RD neighbours, also the long-radius (B-vertex offset)
/// of an RD cell. Guarantees no cell touches or crosses the YZ mirror
/// plane, so right- and left-half meshes meet flush without overlap.
pub const MIN_X_BILATERAL: f32 = 1.154_700_5;

/// Edge length of the rhombic dodecahedron used by volumetric growth. Mirrors
/// `growth::volumetric_growth::EDGE_LEN` (private to that module). If they
/// drift apart, the seed-cell offset will no longer match the lattice — keep
/// them in sync.
const EDGE_LEN: f32 = 1.0;


/// How a body part hangs off its parent.
///
/// `origin_local` is a 3D point in the PARENT body part's local frame — the
/// pivot around which `rotation` is applied. The body part's child entity is
/// expected to live as a Bevy child of the parent body-part entity, with
/// `Transform { translation: origin_local, rotation, .. }`. Bevy's
/// `TransformSystems::Propagate` then composes the world transform chain
/// automatically, so rotating an attachment swings the body part around its
/// origin without any manual matrix work.
#[derive(Clone, Debug)]
pub struct Attachment {
    /// Index into `Organism::body_parts` of the parent body part.
    pub parent_idx:   usize,
    /// Pivot point in the parent body part's local frame.
    pub origin_local: Vec3,
    /// Orientation of this body part relative to its parent. Initially
    /// `Quat::IDENTITY`; animated later.
    pub rotation:     Quat,
}


/// 20% coin flip — should this offspring grow a new body part on top of
/// extending its parent's body plan?
#[inline]
pub fn should_branch(rng: &mut impl Rng) -> bool {
    rng.random::<f32>() < NEW_BODY_PART_PROBABILITY
}


/// Pick an attachment origin on the parent body part for a new branch.
///
/// Returns `(origin_in_parent_local, outward_dir)`:
///   * `origin_in_parent_local` — chosen cell position in parent's local frame
///   * `outward_dir` — unit-length lattice slot direction along which the
///     new part's seed cell should be offset to sit flush against the parent
///     as an FCC-neighbour
///
/// Strategy: pick the parent OCG entry with the largest distance from the
/// part's centroid (most "outward" cell), then snap to the nearest of the 18
/// lattice slot directions. Falls back to `(ZERO, Y)` for a single-cell parent.
pub fn pick_attachment(
    parent_ocg: &[(usize, Vec3, CellType)],
    rng:        &mut impl Rng,
) -> (Vec3, Vec3) {
    if parent_ocg.is_empty() {
        return (Vec3::ZERO, Vec3::Y);
    }

    // Centroid of the parent body part in its local frame.
    let centroid: Vec3 = parent_ocg.iter().map(|(_, p, _)| *p).sum::<Vec3>()
        / parent_ocg.len() as f32;

    // Score every cell by how far it sticks out from the centroid, then pick
    // randomly among the top tier so two siblings born in the same tick don't
    // end up with attachment points at literally the same cell.
    let mut scored: Vec<(usize, f32)> = parent_ocg.iter()
        .enumerate()
        .map(|(i, (_, p, _))| (i, (*p - centroid).length_squared()))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_n = scored.len().min(3).max(1);
    let pick = rng.random_range(0..top_n);
    let chosen_idx = scored[pick].0;
    let origin = parent_ocg[chosen_idx].1;

    // Outward direction: unit vector from centroid to chosen cell, snapped
    // to the nearest lattice slot direction so the seed lands on a real RD
    // adjacency. SLOT_DIRS isn't pub from the volumetric_growth module, so we
    // hardcode the same 18 directions here (axis-aligned + FCC face-diagonal).
    let outward_raw = (origin - centroid).normalize_or_zero();
    let outward = if outward_raw == Vec3::ZERO {
        Vec3::Y
    } else {
        snap_to_lattice_slot(outward_raw)
    };

    (origin, outward)
}


/// Build a fresh BodyPart for a branch attached to its parent.
///
/// The seed cell sits at `outward_dir * EDGE_LEN` in the new part's local
/// frame — the part's local origin coincides with `attachment.origin_local`
/// (in parent's frame), so this places the seed flush against the parent
/// surface cell as an FCC-neighbour (one shared rhombic face). `regrowable`
/// is `true` so future growth ticks know this part can accept new cells.
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

/// Mirror an OCG across the YZ-plane (negate X on every entry). Cell types
/// and ledger indices are preserved; only positions change.
pub fn mirror_ocg_x(
    ocg: &[(usize, Vec3, CellType)],
) -> Vec<(usize, Vec3, CellType)> {
    ocg.iter()
        .map(|(idx, pos, ct)| (*idx, Vec3::new(-pos.x, pos.y, pos.z), *ct))
        .collect()
}

/// |x| below which a cell is considered to lie ON the bilateral mirror
/// plane. Lattice x-coordinates are multiples of `MIN_X_BILATERAL`
/// (≈1.1547), so any sane epsilon well under that cleanly separates a
/// midline cell (x = 0) from the innermost paramedian column.
pub const BILATERAL_MIDLINE_EPS: f32 = 1e-3;

/// Build the LEFT half of a bilateral body from a right-half OCG
/// (cells with `x ≥ 0`), mirroring across the YZ-plane. Cells ON the
/// mirror plane (`|x| < BILATERAL_MIDLINE_EPS`) are their own mirror
/// and are SKIPPED here — they live once in the right-half OCG and
/// must not be duplicated, or the mesh dedup (which drops faces shared
/// by two coincident cells) would erase them. Indices and cell types
/// are preserved; callers renumber after concatenating the halves.
pub fn mirror_right_to_left(
    right_ocg: &[(usize, Vec3, CellType)],
) -> Vec<(usize, Vec3, CellType)> {
    right_ocg.iter()
        .filter(|(_, p, _)| p.x > BILATERAL_MIDLINE_EPS)
        .map(|(idx, pos, ct)| (*idx, Vec3::new(-pos.x, pos.y, pos.z), *ct))
        .collect()
}

/// Build the single bilateral body part from a right-half OCG
/// (cells with `x ≥ 0`).
///
/// Returns ONE `BodyPart` whose OCG contains both halves: the input
/// right-side cells followed by the mirror of the strictly-positive-x
/// ones (`mirror_right_to_left`). `volumetric_growth::build_mesh_from_ocg`
/// then runs its translate → weld → drop-interior-faces pipeline over
/// the combined cell list.
///
/// **How the halves connect.** A right cell and its pure-X mirror are
/// separated by `(2x, 0, 0)`. In the RD lattice only the 12 FCC slots
/// (two ±½ components) share a rhombic FACE; the 6 axis slots share
/// just a vertex. A pure-X displacement has a single non-zero
/// component, so a cell and its own mirror are NEVER face-adjacent —
/// they meet at most at a point. The two halves are therefore bridged
/// by **midline cells** (`x = 0`): such a cell is FCC-face-adjacent to
/// both a +X cell and that cell's −X mirror (e.g. `(0, t, 0)` neighbours
/// both `(t, 0, 0)` and `(−t, 0, 0)`), so it welds the halves with real
/// shared faces. Midline cells are stored once (see
/// `mirror_right_to_left`); without any midline cells the body still
/// renders, just with the two halves touching only at points.
///
/// The weld step (HashMap on quantised positions, `WELD_EPS = 1e-4`)
/// merges coincident vertices and the dedup step drops faces shared by
/// adjacent cells (multiplicity 2); surviving triangles point outward
/// by construction. No manual zipper stitching, no `fill_holes`.
pub fn bilateral_body_part_from_right_ocg(
    right_ocg: &[(usize, Vec3, CellType)],
) -> BodyPart {
    let left_ocg = mirror_right_to_left(right_ocg);

    // Combined OCG: right cells first, then left mirrors. Indices are
    // re-numbered sequentially so the body part's OCG is a contiguous
    // [0..N) ledger as every other body part is.
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
        // 6 axis-aligned (already unit length)
        Vec3::new( 1.0,  0.0,  0.0), Vec3::new(-1.0,  0.0,  0.0),
        Vec3::new( 0.0,  1.0,  0.0), Vec3::new( 0.0, -1.0,  0.0),
        Vec3::new( 0.0,  0.0,  1.0), Vec3::new( 0.0,  0.0, -1.0),
        // 12 FCC face-diagonal (magnitude √0.5; each is unit-length when normalised)
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
