// Species editor — Undo (Ctrl+Z).
//
// Snapshot-based: a species under construction is tiny (a few body
// parts, ≤ tens of cells each), so cloning the whole `body_parts` list
// on each structural change is cheap and far more robust than tracking
// per-action inverse operations. Every change to `body_parts` (cell
// placement, "Begin New Body-Part", rename commit, Clear) pushes the
// PRIOR state onto a stack; Ctrl+Z pops and restores it.
//
// `track_species_undo` watches `SpeciesSession::body_parts` for a
// structural change and records the previous state. The `suppress`
// flag lets `handle_species_undo_shortcut` apply a restore without it
// being re-captured as a new edit.

use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::prelude::*;

use crate::simulation_settings::WindowMode;

use super::session::{EditorBodyPart, SpeciesSession};


/// Maximum number of undo steps retained (mirrors the colony editor).
const MAX_UNDO_DEPTH: usize = 100;


#[derive(Resource, Default)]
pub struct SpeciesUndo {
    /// History of prior `body_parts` states, oldest first.
    stack: Vec<Vec<EditorBodyPart>>,
    /// Last-seen `body_parts`, the baseline the tracker diffs against.
    last:  Vec<EditorBodyPart>,
    /// Set when a restore is applied so the resulting change is synced
    /// into `last` instead of being pushed as a fresh edit.
    suppress: bool,
}


/// Record the previous `body_parts` whenever it changes structurally.
pub fn track_species_undo(
    mode:    Res<WindowMode>,
    session: Res<SpeciesSession>,
    mut undo: ResMut<SpeciesUndo>,
) {
    if *mode != WindowMode::SpeciesEditor { return; }

    if undo.suppress {
        // The change came from an undo restore — adopt it as the new
        // baseline without recording it.
        undo.last = session.body_parts.clone();
        undo.suppress = false;
        return;
    }

    if session.body_parts != undo.last {
        let prior = std::mem::replace(&mut undo.last, session.body_parts.clone());
        undo.stack.push(prior);
        if undo.stack.len() > MAX_UNDO_DEPTH {
            undo.stack.remove(0);
        }
    }
}


/// Ctrl+Z → restore the most recent prior `body_parts` state.
///
/// Matches the LOGICAL "z" key (like the colony editor) so it works on
/// non-QWERTY layouts. Suppressed while a body-part rename is in
/// progress so the keystroke edits the name instead.
pub fn handle_species_undo_shortcut(
    mode:        Res<WindowMode>,
    keys:        Res<ButtonInput<KeyCode>>,
    mut events:  MessageReader<KeyboardInput>,
    mut undo:    ResMut<SpeciesUndo>,
    mut session: ResMut<SpeciesSession>,
) {
    if *mode != WindowMode::SpeciesEditor {
        for _ in events.read() {}   // drain
        return;
    }

    let ctrl = keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight);
    let mut trigger = false;
    for ev in events.read() {
        if !ev.state.is_pressed() { continue; }
        if !ctrl { continue; }
        if let Key::Character(s) = &ev.logical_key {
            if s.eq_ignore_ascii_case("z") { trigger = true; }
        }
    }

    // Don't steal Ctrl+Z while the user is typing a rename.
    if session.renaming_body_part.is_some() || !trigger { return; }

    let Some(prev) = undo.stack.pop() else { return; };
    session.body_parts = prev;
    if session.active_body_part >= session.body_parts.len() {
        session.active_body_part = session.body_parts.len().saturating_sub(1);
    }
    session.first_cell_spawned = !session.body_parts.is_empty();
    session.renaming_body_part = None;
    session.dirty = true;
    undo.suppress = true;
}
