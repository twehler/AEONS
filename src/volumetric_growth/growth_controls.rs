use bevy::prelude::{Resource, Vec3};

// ── Connection geometry ───────────────────────────────────────────────────────

/// Radius of the virtual sphere (in world units) used to determine which
/// vertices of the new cell and the existing mesh are connected at each
/// attachment. Larger values create a wider "neck" between cells.
///
/// Rule of thumb for dodecahedra (EDGE_LEN = 1):
///   0.6  – roughly equivalent to the old single-triangle prism bridge
///   1.2  – captures most junction-side vertices of both cells
///   2.0+ – merges large regions; open-face count drops quickly
pub const CONNECTION_RADIUS: f32 = 1.2;

/// When `true`, a new cell is only accepted if its centre's Y coordinate is
/// ≥ the parent cell's Y coordinate, producing upward / horizontal growth
/// (useful for plant-like structures). Horizontal branching (same Y) is
/// allowed. Set to `false` to restore omnidirectional growth.
pub const GROW_ONLY_UPWARDS: bool = true;

// ── Public view of a single growth candidate ─────────────────────────────────

#[allow(dead_code)]
/// All geometric information available to a growth strategy when choosing
/// which candidate to grow next.
///
/// Extend this struct whenever a new strategy needs additional context
/// (e.g. surface curvature, depth from seed, chemical gradient, …).
#[derive(Clone)]
pub struct CandidateInfo {
    /// Index into the internal `open_faces` list — passed back to the engine.
    pub face_idx: usize,
    /// World-space position where the new cell centre would be placed.
    pub center: Vec3,
    /// Outward normal of the attachment face (unit length).
    pub face_normal: Vec3,
    /// World-space centroid of the attachment face.
    pub face_centroid: Vec3,
}

// ── Strategy trait ────────────────────────────────────────────────────────────

/// Decides which candidate to grow next each tick.
///
/// Implement this trait to plug in custom growth logic — procedural rules,
/// gradient descent, a neural-network policy, or anything else.
///
/// `select` returns `Some(index into candidates)` or `None` to skip this tick.
pub trait GrowthStrategy: Send + Sync {
    fn select(&mut self, candidates: &[CandidateInfo], rng: &mut u64) -> Option<usize>;
}

// ── Built-in strategies ───────────────────────────────────────────────────────

/// Picks a candidate uniformly at random (the default behaviour).
pub struct RandomStrategy;

impl GrowthStrategy for RandomStrategy {
    fn select(&mut self, candidates: &[CandidateInfo], rng: &mut u64) -> Option<usize> {
        if candidates.is_empty() {
            return None;
        }
        Some(lcg_next(rng) as usize % candidates.len())
    }
}

// ── Controller resource ───────────────────────────────────────────────────────

/// Bevy resource that owns the active growth strategy.
///
/// To swap strategies at runtime, replace `controller.strategy` with a
/// `Box<dyn GrowthStrategy>` of your choice.
#[derive(Resource)]
pub struct GrowthController {
    pub strategy: Box<dyn GrowthStrategy>,
}

impl Default for GrowthController {
    fn default() -> Self {
        Self { strategy: Box::new(RandomStrategy) }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// LCG step — mirrors `VolumetricState::next_rand` so strategies that need
/// random numbers share the same quality and can be reproduced.
pub fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}
