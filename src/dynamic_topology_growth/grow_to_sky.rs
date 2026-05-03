//! Upward-steering filter for the growth algorithm.
//!
//! When `STEER_GROWTH_UPWARDS` is `true`, the growth pass rejects any cap
//! whose apex y-coordinate is below the maximum y of the face it would cap.
//! Combined with a non-zero `BRANCHING` (set in
//! `dynamic_topology_growth.rs`), this biases the topology toward tree-like
//! upward growth: every new vertex sits at least as high as the parent face
//! that birthed it, so branches can only ascend or extend laterally — never
//! double back down.

use bevy::prelude::*;

use super::Face;

/// Master switch for upward-steering growth. `true` enables the
/// max-parent-y filter described above; `false` reverts to fully
/// directional-agnostic growth.
pub(super) const STEER_GROWTH_UPWARDS: bool = true;

/// Returns `true` if the candidate apex passes the upward-steering filter,
/// or if steering is disabled. The check: `apex.y >= max(face_vertex.y)`.
pub(super) fn passes_upward_filter(face: &Face, verts: &[Vec3], apex: Vec3) -> bool {
    if !STEER_GROWTH_UPWARDS {
        return true;
    }
    let max_parent_y = face
        .vertices
        .iter()
        .map(|&i| verts[i as usize].y)
        .fold(f32::NEG_INFINITY, f32::max);
    apex.y >= max_parent_y
}
