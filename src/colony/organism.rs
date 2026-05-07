// Organism — top-level component data for one creature in the simulation.
//
// Moved out of `colony.rs` so the struct definition has a dedicated home as
// the simulation grows. `colony.rs` re-exports everything in this module via
// `pub use crate::organism::*;` so the rest of the codebase keeps importing
// `crate::colony::Organism` etc. unchanged.
//
// The cell-tally fields (`photo_cell_count`, `non_photo_cell_count`) are
// caches updated only at construction time (i.e. at birth — currently the
// only event that mutates an organism's cell composition). Predation, which
// removes cells from a body part, also adjusts these counts so the cache
// stays consistent. They let downstream systems (energy, physiology) read
// the cell mix in O(1) instead of iterating every body part's cells each
// tick.

use bevy::prelude::*;

use crate::cell::*;


// ── Organism ─────────────────────────────────────────────────────────────────

/// Body-plan symmetry strategy. Read by reproduction to pick which growth
/// pipeline to run; by spawn helpers to decide whether to build one or two
/// body parts. Inherited verbatim by offspring (mutation never changes
/// symmetry — that would require restructuring the child's body, currently
/// out of scope).
///
/// * `NoSymmetry` — the legacy single-root path: body_parts[0] is the root,
///   subsequent entries are branches (the 20% branch-spawn path runs).
/// * `Bilateral`  — body_parts[0] is the right half (cells with x ≥
///   `body_part::MIN_X_BILATERAL`); body_parts[1] is the left half (mirror
///   across the YZ-plane). Reproduction grows the right half by one cell
///   and re-mirrors the left. Branches are skipped for bilateral organisms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Symmetry {
    NoSymmetry,
    Bilateral,
}

#[derive(Component, Clone)]
pub struct Organism {
    /// Every body part the organism has, alive or consumed. Layout depends
    /// on `symmetry`:
    ///   * `NoSymmetry`: index 0 is the root; subsequent entries are
    ///     branches that reference their parent through
    ///     `BodyPart::attachment`.
    ///   * `Bilateral`: index 0 is the right half (cells with x > 0);
    ///     index 1 is the left half (mirror).
    /// Each body part owns its own OCG; the global organism cell catalogue
    /// is the union of all `body_parts[*].ocg` over alive parts.
    pub body_parts: Vec<BodyPart>,
    /// Body-plan strategy — see `Symmetry`.
    pub symmetry: Symmetry,
    /// When `true`, the organism never moves: `apply_movement` skips it
    /// entirely, regardless of whatever `movement_speed` /
    /// `movement_direction` the brain systems write. Inherited verbatim
    /// by offspring. Auto-set whenever `has_variable_form` is true.
    pub is_sessile: bool,
    /// When `true`, the organism is forced to `Symmetry::NoSymmetry` and
    /// `is_sessile` (a plant-like body that grows asymmetrically and
    /// stays rooted). Inherited verbatim by offspring. The roll happens
    /// at initial colony spawn (80% chance for photoautotrophs); after
    /// that the trait is hereditary.
    pub has_variable_form: bool,
    /// Cached count of `CellType::Photo` cells across all alive body parts.
    /// Updated only when cells are added or removed (currently birth +
    /// predation). Read by `energy.rs` and `physiology.rs` for per-tick
    /// energy bookkeeping without re-iterating every cell.
    pub photo_cell_count: i32,
    /// Cached count of `CellType::NonPhoto` cells across all alive body
    /// parts. Same lifecycle as `photo_cell_count`.
    pub non_photo_cell_count: i32,
    pub energy: f32,
    /// True when the organism's position has an unobstructed line to the sun.
    /// Maintained by `photosynthesis.rs` and consumed by Level 1 brains.
    pub in_sunlight: bool,
    /// Hard gate consulted by `reproduction.rs`: once `true`, this organism
    /// will never spawn another offspring for the rest of its life.
    pub reproduced: bool,
    /// Running count of successful reproductions, used by `reproduction.rs`
    /// to decide when to flip `reproduced` (heterotrophs flip after the
    /// first, photoautotrophs after the second).
    pub reproductions: u8,
    pub movement_speed: f32,
    pub movement_direction: Vec3,
    pub velocity: Vec3,
    pub is_climbing: bool,
    /// Vertical metres climbed since the last energy tick, awaiting payment
    /// at `ELEVATION_ENERGY_PER_UNIT` per unit. Reset to 0 each tick. Krishi
    /// is excluded from the energy system entirely so its debt never drains.
    pub climb_energy_debt: f32,
}

impl Organism {
    /// Total currently-grown cells across alive body parts. Predation-
    /// consumed body parts are skipped — they no longer contribute to
    /// energy, weight or photosynthesis bookkeeping.
    #[inline]
    pub fn grown_cell_count(&self) -> usize {
        self.body_parts.iter()
            .filter(|bp| bp.is_alive())
            .map(|bp| bp.ocg.len())
            .sum()
    }

    /// (photo_count, non_photo_count) — read straight from the cached
    /// fields. Maintainers of `body_parts` MUST keep these counts in sync
    /// (see `physiology::recompute_body_parts` and `predation_system`).
    #[inline]
    pub fn cell_counts(&self) -> (u32, u32) {
        (self.photo_cell_count.max(0) as u32, self.non_photo_cell_count.max(0) as u32)
    }

    /// Effective biological mass — proportional to grown cell count of
    /// alive body parts. Floored at 1.0 so single-cell juveniles don't
    /// divide by zero in energy / drag calculations.
    #[inline]
    pub fn weight(&self) -> f32 {
        (self.grown_cell_count() as f32).max(1.0)
    }

    /// Maximum distance from the organism root that any grown cell on any
    /// alive body part can reach. For branches, the distance is computed in
    /// the parent body part's frame (origin + cell offset) since branches'
    /// world position chains through their parent. We approximate this by
    /// summing each body part's local cell extent with its attachment origin
    /// — sufficient for broad-phase collision and water buoyancy, which only
    /// need a conservative upper bound.
    pub fn bounding_radius(&self) -> f32 {
        let mut max_r = 2.0 * RD_HALF_SIZE; // single-cell starter floor
        let pad = MESH_PADDING.max(2.0 * RD_HALF_SIZE);

        for bp in self.body_parts.iter().filter(|bp| bp.is_alive()) {
            // World-relative offset where this body part's local origin sits.
            // For root (no attachment) this is zero; for a branch it's the
            // attachment origin in the parent's frame, which is itself
            // located at the parent's local_offset (zero in the current
            // single-level hierarchy). Once nested branches arrive, this
            // should walk the attachment chain.
            let part_origin = bp.attachment.as_ref()
                .map(|a| a.origin_local)
                .unwrap_or(Vec3::ZERO);
            for (_, pos, _) in &bp.ocg {
                let r = (part_origin + *pos).length() + pad;
                if r > max_r { max_r = r; }
            }
        }
        max_r
    }

    /// Number of body parts that still have cells and haven't been eaten.
    #[inline]
    pub fn alive_body_part_count(&self) -> usize {
        self.body_parts.iter().filter(|b| b.is_alive()).count()
    }

    /// Recompute `photo_cell_count` and `non_photo_cell_count` from the
    /// current `body_parts` content. Called by the construction helpers in
    /// `colony.rs` / `reproduction.rs` / `krishi.rs` after any mutation to
    /// the body-part list, so the caches reflect the truth at that moment.
    pub fn recompute_cell_counts(&mut self) {
        let mut p = 0i32;
        let mut np = 0i32;
        for bp in self.body_parts.iter().filter(|bp| bp.is_alive()) {
            for cell in &bp.cells {
                match cell.cell_type {
                    CellType::Photo    => p  += 1,
                    CellType::NonPhoto => np += 1,
                }
            }
        }
        self.photo_cell_count     = p;
        self.non_photo_cell_count = np;
    }
}


// ── Marker components ────────────────────────────────────────────────────────

/// Marks an organism as a photoautotroph (energy from photosynthesis).
#[derive(Component, Clone, Copy)]
pub struct Photoautotroph;

/// Marks an organism as a heterotroph (energy from consuming other organisms).
#[derive(Component, Clone, Copy)]
pub struct Heterotroph;

#[derive(Component)]
pub struct OrganismRoot;


/// Trophic strategy chosen at spawn time. Decides which marker component is
/// inserted on the root entity and which colour the starter cell takes.
#[derive(Clone, Copy, Debug)]
pub enum OrganismKind {
    Photoautotroph,
    Heterotroph,
}

impl OrganismKind {
    #[inline]
    pub fn starter_cell_type(self) -> CellType {
        match self {
            OrganismKind::Photoautotroph => CellType::Photo,
            OrganismKind::Heterotroph    => CellType::NonPhoto,
        }
    }
}
