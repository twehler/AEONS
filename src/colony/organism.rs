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
use rand::prelude::*;

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


/// Intelligence level — selects WHICH brain pool processes the organism
/// AND the size of that pool's MLP. All non-Level0 pools share the same
/// REINFORCE-style RL training loop (reward = energy delta) and the
/// same I/O dimensions; they differ only in hidden-layer width:
///
///   * `Level0` — no brain, no GPU work. Sessile organisms always get
///     this; movement systems skip them anyway. Implemented in
///     `intelligence_level_0.rs` (a marker only — the file has no
///     systems and no resources).
///   * `Level1` — small RL brain. Hidden width tuned to be cheap.
///   * `Level2` — medium RL brain.
///   * `Level3` — large RL brain.
///
/// **Assignment policy**:
///   * Initial colony spawn (in `spawn_colony`): rolled by
///     `IntelligenceLevel::for_initial_spawn` per the documented
///     distribution (photoautotrophs 80% Level0 / 20% Level1 — the
///     80% naturally falls out of the existing `has_variable_form`
///     roll since sessile organisms always get Level0; heterotrophs
///     50% Level1 / 40% Level2 / 10% Level3).
///   * Reproduction: offspring inherits the parent's
///     `intelligence_level` verbatim (no re-roll, no mutation).
///   * Loaded colony: deserialised from the save file as-is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntelligenceLevel {
    Level0,
    Level1,
    Level2,
    Level3,
}

impl IntelligenceLevel {
    /// Roll an intelligence level for an organism in the *initial*
    /// colony cohort (i.e. `spawn_colony`, not reproduction or load).
    /// Sessile organisms always get Level 0 regardless of the roll.
    pub fn for_initial_spawn(
        kind:        OrganismKind,
        is_sessile:  bool,
        rng:         &mut impl rand::Rng,
    ) -> Self {
        if is_sessile { return IntelligenceLevel::Level0; }
        match kind {
            OrganismKind::Photoautotroph => {
                // Mobile photoautotrophs (the bilateral 20% of the
                // initial photo cohort) all get Level 1. The "80%
                // Level 0" target falls out of the sessile branch
                // above, since 80% of photoautotrophs are
                // has_variable_form (and therefore is_sessile).
                IntelligenceLevel::Level1
            }
            OrganismKind::Heterotroph => {
                let r = rng.random::<f32>();
                if      r < 0.5 { IntelligenceLevel::Level1 }
                else if r < 0.9 { IntelligenceLevel::Level2 }
                else            { IntelligenceLevel::Level3 }
            }
        }
    }
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
    /// Brain tier — see `IntelligenceLevel`. Set at spawn from
    /// `is_sessile` and trophic kind via
    /// `IntelligenceLevel::for_spawn`. Read by `spawn_organism` /
    /// `spawn_loaded_organism` to decide which marker components to
    /// insert (e.g. `BrainLevel0` for sessile organisms). Inherited
    /// verbatim by offspring through reproduction / continuous
    /// growth — but since the inputs (`is_sessile`, `kind`) inherit
    /// verbatim too, the inherited value is always self-consistent
    /// with `IntelligenceLevel::for_spawn`.
    pub intelligence_level: IntelligenceLevel,
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
    /// `true` once the organism has finished growing — i.e. its meshes
    /// have been smoothed via `volumetric_growth::smooth_vertices`. For
    /// non-variable-form organisms this is `true` from spawn (they don't
    /// grow during their own lifetime — only their offspring inherit
    /// extra cells). For variable-form organisms this flips to `true`
    /// inside `continuous_growth` on the tick where their grown cell
    /// count first reaches `MAX_CELLS`. Once `true`, no further
    /// smoothing work runs on the organism.
    pub adult: bool,
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
    /// Cached output of `compute_bounding_radius` — the maximum distance
    /// from the organism root that any grown cell can reach. Updated only
    /// at composition-change events (birth, growth, predation) by
    /// `recompute_caches`. Read every frame by movement / floor /
    /// buoyancy queries — caching this avoids an O(cells) walk per
    /// access × thousands of accesses per frame.
    pub cached_bounding_radius: f32,
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
    /// alive body part can reach. Reads the cached value populated by
    /// `recompute_caches`. For broad-phase collision and water buoyancy
    /// — needs a conservative upper bound, not exact geometry.
    #[inline]
    pub fn bounding_radius(&self) -> f32 {
        self.cached_bounding_radius
    }

    /// Re-derive the bounding radius from the current `body_parts`. For
    /// branches the distance is taken in the parent body part's frame
    /// (origin + cell offset). Called from `recompute_caches`.
    fn compute_bounding_radius(&self) -> f32 {
        let mut max_r = 2.0 * RD_HALF_SIZE; // single-cell starter floor
        let pad = MESH_PADDING.max(2.0 * RD_HALF_SIZE);

        for bp in self.body_parts.iter().filter(|bp| bp.is_alive()) {
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
        self.cached_bounding_radius = self.compute_bounding_radius();
    }

    /// Update only the cached bounding radius — call this from sites
    /// that maintain `photo_cell_count` / `non_photo_cell_count`
    /// incrementally and don't want to re-iterate every cell.
    /// Currently used by `continuous_growth` (after appending a cell)
    /// and `predation` (after consuming a body part).
    #[inline]
    pub fn recompute_bounding_radius(&mut self) {
        self.cached_bounding_radius = self.compute_bounding_radius();
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
