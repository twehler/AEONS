// Editor-wide working state.
//
// `EditorSession` owns the list of templates the user has created,
// the currently-selected one (right-click placement target), and the
// draft fields the bottom "Create" panel binds to. Centralising it
// here keeps every UI / placement / save system reading from a single
// source of truth.

use bevy::prelude::*;

use crate::organism::{IntelligenceLevel, Symmetry};
use crate::colony_editor::template::{Form, Metabolism, OrganismTemplate};
use crate::cell::CellType;


// ── Loaded species ──────────────────────────────────────────────────────────
//
// One entry per `.species` file the user has loaded in this editor
// session. The body plan + OCG come from the file; per-organism
// hyperparameter genes (curiosity, K_EAT, σ, etc.) are NOT stored on
// the species and are re-sampled from `simulation_settings.rs`'s
// `L1_*_RANGE` ranges at brain-slot assignment time, every time a
// species-derived organism is spawned. This matches the existing
// reproduction / initial-spawn paths and means "load a species" is a
// pure body-blueprint operation — gene diversity comes from the
// pool, not from the file.

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
    /// `true` ⇒ Carnivore (target other heterotrophs);
    /// `false` ⇒ Herbivore (target photoautotrophs, default).
    /// Drives the `Carnivore` marker component at spawn time, which
    /// IL2 / IL3 brains read to filter their prey type.
    pub is_carnivore:      bool,
    /// Full OCG ready for `root_body_part_from_ocg`. For bilateral
    /// species this has been pre-expanded to right + mirrored-left
    /// at load time so downstream code (placement, save) doesn't
    /// need to know the difference.
    pub ocg:               Vec<(usize, Vec3, CellType)>,
}


/// Pending fields for the bottom-panel cyclers. Each left-click on
/// the map reads this snapshot and turns it into a fresh
/// `OrganismTemplate`.
#[derive(Clone, Copy)]
pub struct DraftOrganism {
    pub metabolism:   Metabolism,
    pub intelligence: IntelligenceLevel,
    pub symmetry:     Symmetry,
    pub form:         Form,
}

impl Default for DraftOrganism {
    fn default() -> Self {
        // Bilateral + Fixed is the only invariant-compatible default
        // we can pick that doesn't force the user into a particular
        // shape: it works with every metabolism and intelligence
        // level. (Variable form would force NoSymmetry on the
        // first render.)
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
    /// Monotonically-increasing identifier issued to every created
    /// template. Never decremented.
    pub next_id:    u32,
    /// All organism templates the user has created.
    pub templates:  Vec<OrganismTemplate>,
    /// Identifier of the currently-selected template, or `None` if
    /// nothing is selected. Right-click placement targets this one.
    pub active_id:  Option<u32>,
    /// Fields bound to the bottom-panel cyclers.
    pub draft:      DraftOrganism,
    /// One-shot flag set by the inventory panel's Save button.
    /// `colony_editor::save_dispatch` consumes it on the next tick
    /// to open a file dialog and write the .colony file.
    pub save_requested: bool,

    /// True iff the editor state has unsaved changes. Set by every
    /// system that mutates `templates` (create / move). Reset to
    /// `false` after a successful save. The Return-to-Menu flow
    /// reads this to decide whether to show the unsaved-work modal.
    pub dirty:          bool,

    /// Set by the inventory panel's "Return to Menu" button. The
    /// dispatcher in `mod.rs` consumes it on the next tick:
    ///   * `dirty == false` ⇒ exit immediately
    ///   * `dirty == true`  ⇒ raise `show_exit_modal` so the modal
    ///                         appears, blocking input until the
    ///                         user resolves it.
    pub exit_requested: bool,

    /// True while the unsaved-work confirmation modal should be
    /// visible. Toggled by the Return-to-Menu button (on) and the
    /// modal's own Yes/No buttons (off; Yes also triggers the exit).
    pub show_exit_modal: bool,

    /// True while the "Clear All" confirmation modal should be
    /// visible. Set by the Clear-All button (`inventory_panel`) and
    /// reset by the modal's own Yes/No buttons. Yes additionally
    /// drains every template AND despawns every live `OrganismRoot`
    /// in the simulation — destructive enough to warrant an explicit
    /// confirmation step.
    pub show_clear_modal: bool,

    // ── Species navigator state ───────────────────────────────────
    /// All `.species` files the user has loaded this editor session.
    pub loaded_species:        Vec<LoadedSpecies>,
    /// Monotonic id issued to each loaded species.
    pub next_species_id:       u32,
    /// Currently-selected species (highlighted in the navigator list).
    /// Drives both single-click placement and bulk-spawn.
    pub selected_species_id:   Option<u32>,
    /// One-shot: when set, `dispatch_load_species_requests` reads the
    /// file on the next Update tick, decodes it, mirrors the OCG if
    /// Bilateral, and appends a new `LoadedSpecies`. The Load Species
    /// button stuffs the rfd-chosen path here.
    pub load_species_path:     Option<std::path::PathBuf>,
}
