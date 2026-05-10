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
}
