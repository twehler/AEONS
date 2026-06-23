use bevy::prelude::{Resource, Vec3};

// ── Connection geometry ───────────────────────────────────────────────────────

/// Radius (world units) of the sphere deciding which new-cell and existing-mesh
/// vertices connect at an attachment; larger = wider "neck" between cells.
pub const CONNECTION_RADIUS: f32 = 1.2;

/// When `true`, a new cell needs centre.y ≥ parent.y, giving upward/horizontal
/// (plant-like) growth. `false` restores omnidirectional growth.
pub const GROW_ONLY_UPWARDS: bool = true;

// ── Public view of a single growth candidate ─────────────────────────────────

#[allow(dead_code)]
/// Geometric context a growth strategy uses to choose the next candidate.
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

/// Decides which candidate to grow next each tick. `select` returns
/// `Some(index into candidates)` or `None` to skip this tick.
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

/// Owns the active growth strategy; swap at runtime via `controller.strategy`.
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

/// LCG step — mirrors `VolumetricState::next_rand` so strategy RNG is reproducible.
pub fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}
