// Organism — top-level component data for one creature.
//
// `colony.rs` re-exports this module (`pub use crate::organism::*;`) so the
// rest of the codebase keeps importing `crate::colony::Organism` etc.
//
// The cell-tally fields (`photo_cell_count`, `non_photo_cell_count`) are
// caches maintained at composition changes (birth + predation) so downstream
// systems read the cell mix in O(1) instead of iterating every cell per tick.

use bevy::prelude::*;
use rand::prelude::*;

use crate::cell::*;


// ── Organism ─────────────────────────────────────────────────────────────────

/// Body-plan symmetry strategy. Selects the growth pipeline and how many
/// body parts spawn builds. Inherited verbatim; mutation never changes it.
///
/// * `NoSymmetry` — single-root: body_parts[0] is the root, later entries
///   are branches (the 20% branch-spawn path runs).
/// * `Bilateral`  — body_parts[0] holds both halves combined (right cells
///   with x ≥ `body_part::MIN_X_BILATERAL` plus their YZ-plane mirror).
///   Reproduction grows the right half by one cell and re-mirrors. No branches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Symmetry {
    NoSymmetry,
    Bilateral,
}

impl Symmetry {
    /// Stable `.colony`/`.species` serialisation tag (NoSymmetry=0, Bilateral=1).
    #[inline]
    pub fn to_tag(self) -> u8 {
        match self {
            Symmetry::NoSymmetry => 0,
            Symmetry::Bilateral  => 1,
        }
    }
    /// Inverse of [`Symmetry::to_tag`]; `None` on an unknown byte (corrupt or
    /// newer-format file). Mirrors the original reader, which errored on unknown.
    #[inline]
    pub fn from_tag(tag: u8) -> Option<Symmetry> {
        match tag {
            0 => Some(Symmetry::NoSymmetry),
            1 => Some(Symmetry::Bilateral),
            _ => None,
        }
    }
}


/// Locomotion paradigm. Selects the physics representation, brain pool, and
/// movement-driving system. Inherited verbatim; defaults to `Sliding`.
///
/// `is_sliding()` is the kinematic family (`Sliding` only): kinematic
/// root, REINFORCE brain pool, `apply_movement`-driven, manual collision
/// separation, contact predation. Everything else (including Swimming, now a
/// DYNAMIC limb-physics mode) takes the dynamic/limb path; `is_swimming()`
/// gates Swimming's fluid specialisations on top of that.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MovementMode {
    /// Kinematic root; REINFORCE brain writes movement_speed/direction; apply_movement translates the root. 2D (XZ + gravity).
    #[default]
    Sliding,
    /// Dynamic per-part rigid bodies + revolute joints; PPO brain outputs joint target angles (terrestrial).
    LimbBasedWalking,
    /// Dynamic per-part rigid bodies + BALL (spherical) joints, neutral buoyancy, confined below the water plane; the swimming PPO brain (swimming_movement/) outputs 3-axis joint targets and propulsion emerges from blade-element fluid drag.
    Swimming,
    /// Placeholder for a future fluid-flight mode; currently routed to the dynamic path, inert.
    Flying,
}
impl MovementMode {
    /// Kinematic-root family: REINFORCE brain pool, apply_movement-driven, manual collision separation, contact predation. True for Sliding only (Swimming is now a dynamic limb-physics mode).
    pub fn is_sliding(self) -> bool { matches!(self, MovementMode::Sliding) }
    /// 3D-free, gravity-exempt, water-confined. True only for Swimming.
    pub fn is_swimming(self) -> bool { matches!(self, MovementMode::Swimming) }
    /// Canonical `Organism::ground_based` value for this movement paradigm:
    /// sliders/walkers live on the ground (`true`); swimmers (and future
    /// flyers) live in the fluid volume (`false`). Phototrophs may OVERRIDE
    /// this to `false` via the species editor (floating algae) — every other
    /// organism derives it from here.
    pub fn default_ground_based(self) -> bool {
        !matches!(self, MovementMode::Swimming | MovementMode::Flying)
    }
    /// Display label for the species-editor cycler / UI.
    pub fn label(self) -> &'static str {
        match self {
            MovementMode::Sliding          => "Sliding",
            MovementMode::LimbBasedWalking  => "Limb-Movement",
            MovementMode::Swimming          => "Swimming",
            MovementMode::Flying            => "Flying",
        }
    }
    /// Stable `.colony`/`.species` serialisation tag (v009+ 4-way movement_mode:
    /// Sliding=0, LimbBasedWalking=1, Swimming=2, Flying=3). NOT the pre-v009
    /// sliding/limb bool — that legacy decode stays inline in the loader.
    #[inline]
    pub fn to_tag(self) -> u8 {
        match self {
            MovementMode::Sliding          => 0,
            MovementMode::LimbBasedWalking => 1,
            MovementMode::Swimming         => 2,
            MovementMode::Flying           => 3,
        }
    }
    /// Inverse of [`MovementMode::to_tag`]; `None` on an unknown byte (corrupt
    /// or newer-format file). Mirrors the original reader, which errored on unknown.
    #[inline]
    pub fn from_tag(tag: u8) -> Option<MovementMode> {
        match tag {
            0 => Some(MovementMode::Sliding),
            1 => Some(MovementMode::LimbBasedWalking),
            2 => Some(MovementMode::Swimming),
            3 => Some(MovementMode::Flying),
            _ => None,
        }
    }
}


/// Intelligence level — selects which brain pool processes the organism
/// and that pool's MLP width.
///   * `Level0` — no brain, no GPU work; sessile organisms always get this
///     (movement systems skip them). Marker only (`intelligence_level_0.rs`).
///   * `Level1`/`Level2`/`Level3` — small/medium/large RL brains.
///
/// Assignment: initial spawn rolls via `for_initial_spawn`; reproduction
/// inherits the parent's level verbatim; load deserialises as-is.
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
                // Photoautotrophs are always Level 0 (the L1 photo brain
                // is silenced). Mobile photos sit on Level 0 too.
                IntelligenceLevel::Level0
            }
            OrganismKind::Heterotroph => {
                // L2/L3 are placeholder-only at initial spawn; all mobile
                // heterotroph rolls collapse to Level 1. (Save files with
                // explicit L2/L3 still load with the level preserved.)
                let _ = rng.random::<f32>();
                IntelligenceLevel::Level1
            }
        }
    }

    /// Stable `.colony`/`.species` serialisation tag (Level0=0 … Level3=3).
    #[inline]
    pub fn to_tag(self) -> u8 {
        match self {
            IntelligenceLevel::Level0 => 0,
            IntelligenceLevel::Level1 => 1,
            IntelligenceLevel::Level2 => 2,
            IntelligenceLevel::Level3 => 3,
        }
    }
    /// Inverse of [`IntelligenceLevel::to_tag`]; `None` on an unknown byte
    /// (corrupt or newer-format file). Mirrors the original reader's error-on-unknown.
    #[inline]
    pub fn from_tag(tag: u8) -> Option<IntelligenceLevel> {
        match tag {
            0 => Some(IntelligenceLevel::Level0),
            1 => Some(IntelligenceLevel::Level1),
            2 => Some(IntelligenceLevel::Level2),
            3 => Some(IntelligenceLevel::Level3),
            _ => None,
        }
    }
}

#[derive(Component, Clone)]
pub struct Organism {
    /// Every body part, alive or consumed. Layout depends on `symmetry`
    /// (see `Symmetry`). Each part owns its own OCG; the organism cell
    /// catalogue is the union of `body_parts[*].ocg` over alive parts.
    pub body_parts: Vec<BodyPart>,
    /// Body-plan strategy — see `Symmetry`.
    pub symmetry: Symmetry,
    /// Brain tier — see `IntelligenceLevel`. Inherited verbatim by offspring;
    /// since its inputs (`is_sessile`, kind) also inherit, it stays consistent.
    pub intelligence_level: IntelligenceLevel,
    /// When `true` the organism never moves (`apply_movement` skips it,
    /// ignoring brain writes). Inherited; auto-set when `has_variable_form`.
    pub is_sessile: bool,
    /// When `true`, forced to `NoSymmetry` + `is_sessile` (plant-like body
    /// that grows asymmetrically and stays rooted). Rolled at initial spawn
    /// (80% for photoautotrophs); inherited thereafter.
    pub has_variable_form: bool,
    /// Movement paradigm. Inherited verbatim; defaults to `Sliding`.
    /// See `MovementMode`: four variants —
    ///   * `Sliding` — kinematic root; REINFORCE brain writes
    ///     `movement_speed`/`movement_direction`, `apply_movement` translates
    ///     the root each tick. 2D (XZ + gravity).
    ///   * `LimbBasedWalking` — dynamic per-part rigid-body chain in Avian;
    ///     PPO brain outputs joint target angles, locomotion emerges.
    ///   * `Swimming` — dynamic per-part bodies on BALL (spherical) joints,
    ///     neutral buoyancy, water-confined; the swimming PPO brain
    ///     (`swimming_movement/`) writes 3-axis joint targets into the root's
    ///     `SwimJointTargets` and locomotion emerges from fluid drag.
    ///   * `Flying` — future fluid-flight placeholder, routed to the dynamic
    ///     path, currently inert.
    /// `is_sliding()` selects the kinematic family (`Sliding` only);
    /// everything else takes the dynamic/limb path.
    pub movement_mode: MovementMode,
    /// `true` = lives on the ground (sliders/walkers): gravity always applies.
    /// `false` = lives in the water volume (swimmers, future flyers, and
    /// WATER-BASED phototrophs — floating algae): gravity applies ONLY while
    /// the organism is above the water surface, so it neither sinks to the
    /// bottom nor rises out of the water. Derived from `movement_mode`
    /// (`default_ground_based()`) for every organism except phototrophs,
    /// which may opt into water-based via the species editor. Inherited
    /// verbatim by offspring.
    pub ground_based: bool,
    /// Latest limb commands from the brain. SWING `limb_targets[0..8]`: body-part
    /// `k+1`'s hinge target (fraction of `LIMB_SWING_LIMIT`), read by
    /// `drive_limb_motors`. GROUPED TWIST `limb_targets[8..10]`: per-group twist
    /// effort about the limbs' long axes (fraction of `MAX_LIMB_TWIST_TORQUE`),
    /// read by `drive_limb_twist` — the off-swing-axis DOF so limbs move in 3D.
    /// The brain moves each limb directly (no CPG). Ignored for sliding organisms.
    /// Array length MUST equal `limb_ppo::OUT` (= MAX_LIMB_JOINTS + N_LIMB_TWIST_GROUPS, 10).
    pub limb_targets: [f32; 10],
    /// `true` once meshes have been smoothed (`volumetric_growth::smooth_vertices`).
    /// `true` from spawn for non-variable-form organisms; for variable-form it
    /// flips in `continuous_growth` when grown cells first reach `MAX_CELLS`.
    pub adult: bool,
    /// Cached `CellType::Photo` count over alive body parts. Maintained at
    /// composition changes (birth + predation); read by energy/physiology.
    pub photo_cell_count: i32,
    /// Cached `CellType::AbsorptionCell` count; same lifecycle as `photo_cell_count`.
    pub non_photo_cell_count: i32,
    /// Cached Σ `CellType::upkeep_mult` over alive cells — the per-cell energy
    /// upkeep weight. Same lifecycle as the cell counts (maintained at every
    /// composition change: spawn/load recompute, growth +=, predation -=); read
    /// by `energy::manage_energy` to avoid an O(cells) walk per energy tick.
    pub upkeep_weight: f32,
    pub energy: f32,
    /// True when the organism has an unobstructed line to the sun. Maintained
    /// by `photosynthesis.rs`.
    pub in_sunlight: bool,
    /// Once `true`, `reproduction.rs` never spawns another offspring.
    pub reproduced: bool,
    /// Successful-reproduction count; `reproduction.rs` flips `reproduced`
    /// after the first (heterotrophs) or second (photoautotrophs).
    pub reproductions: u8,
    /// Successful-predation count (body parts eaten by this organism). The
    /// Level 1 hetero brain reads the per-tick delta as its eat-reward signal.
    /// Saturating-wraps at `u8::MAX`; not saved to `.colony` (resets on load).
    pub predations: u8,
    /// Hunger signal in `[0, 1]` (0 = no urge, 1 = max), recomputed each
    /// energy tick by `energy::update_hunger_levels`; formula is
    /// classification-specific (see `energy.rs`). Read by the herbivore brain
    /// to scale pursuit aggression.
    pub hunger: f32,
    /// Reward signal in `[0, 1]` for the herbivore RL brain. +0.6 (clamp 1.0)
    /// per predation, 1.0 per reproduction; depletes by `hunger/3` per virtual
    /// second. The brain reads the per-tick delta as its reward.
    pub dopamine: f32,
    /// Distance (world units) to the nearest photoautotroph within
    /// `sensory::SENSORY_RADIUS` (=50); saturates at the radius when none in
    /// range. Written each brain tick. The herbivore brain uses it as both an
    /// observation and a secondary reward channel (Δ<0 reward, Δ>0 penalty).
    pub target_distance: f32,
    pub movement_speed: f32,
    pub movement_direction: Vec3,
    pub velocity: Vec3,
    pub is_climbing: bool,
    /// Vertical metres climbed since the last energy tick, paid at
    /// `ELEVATION_ENERGY_PER_UNIT`; reset to 0 each tick.
    pub climb_energy_debt: f32,
    /// Cached max distance from the root any grown cell can reach. Updated at
    /// composition changes by `recompute_caches`; read every frame by
    /// movement/floor/buoyancy queries to avoid an O(cells) walk per access.
    pub cached_bounding_radius: f32,

    /// Fixed-dimension genome vector — see `lineages::dna`. Structural slots
    /// are filled at spawn and frozen; brain-gene slots start 0.0 and are
    /// populated by speciation once the brain slot is assigned. Used to
    /// classify organisms into species.
    pub dna: Vec<f32>,

    /// Species this organism belongs to. `None` until the first speciation
    /// tick, which keeps/reassigns/forks it. Offspring inherit the parent's,
    /// re-evaluated on the next tick.
    pub species_id: Option<u32>,
}

impl Organism {
    /// Total grown cells across alive body parts (consumed parts skipped).
    /// O(1): every alive cell is classified into exactly one of the two cached
    /// counts and (for alive parts) `cells.len() == ocg.len()`, so their sum IS
    /// the grown cell count — no per-cell walk needed.
    #[inline]
    pub fn grown_cell_count(&self) -> usize {
        (self.photo_cell_count.max(0) + self.non_photo_cell_count.max(0)) as usize
    }

    /// (photo_count, non_photo_count) from the cached fields. Maintainers of
    /// `body_parts` MUST keep these in sync (recompute_body_parts, predation).
    #[inline]
    pub fn cell_counts(&self) -> (u32, u32) {
        (self.photo_cell_count.max(0) as u32, self.non_photo_cell_count.max(0) as u32)
    }

    /// Effective biological mass ∝ grown cell count. Floored at 1.0 so
    /// single-cell juveniles don't divide by zero in energy/drag math.
    #[inline]
    pub fn weight(&self) -> f32 {
        (self.grown_cell_count() as f32).max(1.0)
    }

    /// Total per-cell upkeep weight: the sum of every alive cell's
    /// `CellType::upkeep_mult`. Multiply by `NON_PHOTO_CONSUMPTION_PER_CELL` for
    /// the energy cost. Photo cells contribute 0; standard tissue 1.0; jelly and
    /// inert material less — so a creature of only standard cells gives exactly
    /// its non-photo cell count (unchanged from the old flat-count upkeep).
    /// O(1): returns the cached `upkeep_weight` (see the field doc for the
    /// maintenance contract). `recompute_upkeep_weight` re-derives it from cells.
    #[inline]
    pub fn upkeep_cell_weight(&self) -> f32 {
        self.upkeep_weight
    }

    /// Re-derive `upkeep_weight` from the current alive cells. Used by the load
    /// path (which sets the cell counts from file, not via `recompute_cell_counts`).
    pub fn recompute_upkeep_weight(&mut self) {
        self.upkeep_weight = self.body_parts.iter()
            .filter(|bp| bp.is_alive())
            .flat_map(|bp| bp.cells.iter())
            .map(|cell| cell.cell_type.upkeep_mult())
            .sum();
    }

    /// Cached max distance from the root any grown cell can reach. A
    /// conservative upper bound for broad-phase collision and buoyancy.
    #[inline]
    pub fn bounding_radius(&self) -> f32 {
        self.cached_bounding_radius
    }

    /// Re-derive the bounding radius from `body_parts`. Branch distances are
    /// taken in the parent's frame (origin + cell offset).
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

    /// Recompute the cached cell counts (+ bounding radius) from `body_parts`.
    /// Construction helpers call this after mutating the body-part list.
    pub fn recompute_cell_counts(&mut self) {
        let mut p = 0i32;
        let mut np = 0i32;
        let mut upkeep = 0.0f32;
        for bp in self.body_parts.iter().filter(|bp| bp.is_alive()) {
            for cell in &bp.cells {
                if cell.cell_type.is_photo() { p += 1; } else { np += 1; }
                upkeep += cell.cell_type.upkeep_mult();
            }
        }
        self.photo_cell_count     = p;
        self.non_photo_cell_count = np;
        self.upkeep_weight        = upkeep;
        self.cached_bounding_radius = self.compute_bounding_radius();
    }

    /// Update only the cached bounding radius — for sites that maintain the
    /// cell counts incrementally (continuous_growth, predation).
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

/// Heterotroph sub-class: `Carnivore` chases/eats other heterotrophs; its
/// absence means herbivore (default), which chases/eats photoautotrophs.
/// Brain pools read this to pick which neighbour type to chase. Initial-spawn
/// heterotrophs default to herbivore.
#[derive(Component, Clone, Copy)]
pub struct Carnivore;


/// Per-organism ancestry + age tracking for the dataset-export lineage
/// columns. Attached to every OrganismRoot at spawn. NOT persisted to
/// `.colony` — loaded organisms re-init as initial-cohort.
#[derive(Component, Debug, Clone)]
pub struct LineageRecord {
    /// `Some(parent)` for reproduction offspring; `None` for initial-cohort,
    /// auto-spawns, editor placements, and loaded organisms.
    pub parent_id:            Option<Entity>,
    /// Virtual-time seconds at spawn.
    pub spawn_time_secs:      f32,
    /// Successful reproductions by this organism (bumped by reproduction_system).
    pub times_reproduced_self: u32,
}

impl LineageRecord {
    /// Fresh initial-cohort / auto-spawn / editor / load record.
    pub fn new_initial(spawn_time_secs: f32) -> Self {
        Self {
            parent_id: None,
            spawn_time_secs,
            times_reproduced_self: 0,
        }
    }
    /// Offspring of a reproduction event.
    pub fn new_offspring(parent: Entity, spawn_time_secs: f32) -> Self {
        Self {
            parent_id: Some(parent),
            spawn_time_secs,
            times_reproduced_self: 0,
        }
    }
}

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
            OrganismKind::Heterotroph    => CellType::AbsorptionCell,
        }
    }
}
