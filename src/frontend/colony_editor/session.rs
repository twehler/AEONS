// Editor-wide working state. `EditorSession` is the single source of
// truth shared by every UI / placement / save system: created templates,
// the active selection, draft fields, and loaded species.

use bevy::prelude::*;

use crate::organism::{IntelligenceLevel, MovementMode, Symmetry};
use crate::colony_editor::template::{Form, Metabolism, OrganismTemplate};
use crate::cell::{BodyPartKind, CellType};


// ── Loaded species ──────────────────────────────────────────────────────────
//
// One entry per loaded `.species` file. Body plan + OCG come from the
// file; per-organism hyperparameter genes are NOT stored and are
// re-sampled from `L1_*_RANGE` at brain-slot assignment per spawn — so
// "load a species" is a pure body-blueprint operation.

#[derive(Clone, Debug)]
pub struct LoadedSpecies {
    pub id:                u32,
    /// Display name — derived from the `.species` filename stem.
    pub name:              String,
    pub metabolism:        Metabolism,
    pub symmetry:          Symmetry,
    pub intelligence:      IntelligenceLevel,
    pub form:              Form,
    pub is_sessile:        bool,
    /// `true` ⇒ Carnivore, `false` ⇒ Herbivore. Drives the `Carnivore`
    /// marker that IL2/IL3 brains read to filter prey type.
    pub is_carnivore:      bool,
    /// Maps to `Organism::movement_mode` (loaded from the `.species` file).
    pub movement_mode:  MovementMode,
    /// Maps to `Organism::ground_based` (loaded from the `.species` file;
    /// only ever an override for phototrophs — floating, water-based algae).
    pub ground_based:   bool,
    /// Base-body OCG (pre-expanded to right + mirrored-left for bilateral,
    /// so downstream placement/save need not special-case it).
    pub ocg:               Vec<(usize, Vec3, CellType)>,
    /// Appendage parts as `(OCG, kind, parent)` — raw stored OCG, `BodyPartKind`
    /// (Limb / Segment / Static / Organ), and EDITOR parent index. Expanded to
    /// runtime parts at spawn: a Bilateral `Limb`/`Organ` → mirrored pair;
    /// `Segment`/`Static` → one fused midline part. Empty for single-part species.
    pub appendages:        Vec<(Vec<(usize, Vec3, CellType)>, BodyPartKind, usize)>,
    /// `Some` when the `.species` file carried trained brain weights; each spawn
    /// gets a copy of the matching restore component, consumed by the relevant
    /// pool's `assign_brains_*`. `None` for fresh/legacy files.
    pub brain: Option<crate::species_editor::save::LoadedBrain>,
}


/// Pending fields for the bottom-panel cyclers.
#[derive(Clone, Copy)]
pub struct DraftOrganism {
    pub metabolism:   Metabolism,
    pub intelligence: IntelligenceLevel,
    pub symmetry:     Symmetry,
    pub form:         Form,
}

impl Default for DraftOrganism {
    fn default() -> Self {
        // Bilateral + Fixed: invariant-compatible with every metabolism
        // and intelligence level (Variable would force NoSymmetry).
        Self {
            metabolism:   Metabolism::Heterotroph,
            intelligence: IntelligenceLevel::Level1,
            symmetry:     Symmetry::Bilateral,
            form:         Form::Fixed,
        }
    }
}


/// All editor state worth referencing from more than one place.
#[derive(Resource, Default)]
pub struct EditorSession {
    /// Monotonic id issued to every created template.
    pub next_id:    u32,
    /// All created organism templates.
    pub templates:  Vec<OrganismTemplate>,
    /// Currently-selected template (right-click placement target), or `None`.
    pub active_id:  Option<u32>,
    /// Fields bound to the bottom-panel cyclers.
    pub draft:      DraftOrganism,
    /// One-shot Save flag; consumed by `dispatch_save_requests`.
    pub save_requested: bool,

    /// Unsaved changes since the last save. Drives the unsaved-work modal.
    pub dirty:          bool,

    /// "Return to Menu" flag; the `mod.rs` dispatcher exits if clean,
    /// or raises `show_exit_modal` if dirty.
    pub exit_requested: bool,

    /// Whether the unsaved-work confirmation modal is visible.
    pub show_exit_modal: bool,

    /// Whether the "Clear All" confirmation modal is visible. Yes drains
    /// every template AND despawns every live `OrganismRoot`.
    pub show_clear_modal: bool,

    // ── Species navigator state ───────────────────────────────────
    /// All `.species` files loaded this session.
    pub loaded_species:        Vec<LoadedSpecies>,
    /// Monotonic id per loaded species.
    pub next_species_id:       u32,
    /// Selected species; drives single-click placement and bulk-spawn.
    pub selected_species_id:   Option<u32>,
    /// One-shot rfd-chosen path; consumed by `dispatch_load_species_requests`.
    pub load_species_path:     Option<std::path::PathBuf>,
}
