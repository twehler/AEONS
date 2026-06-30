// Species editor — Undo (Ctrl+Z). Snapshot-based: a species is tiny, so cloning
// the whole `body_parts` on each structural change is cheap and more robust than
// per-action inverses. `track_species_undo` pushes the prior state on change;
// Ctrl+Z pops and restores. The `suppress` flag stops a restore being recaptured
// as a fresh edit.

use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::prelude::*;

use crate::simulation_settings::WindowMode;

use super::sculpt::SculptStrokeActive;
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
    stroke:  Res<SculptStrokeActive>,
    mut undo: ResMut<SpeciesUndo>,
) {
    if *mode != WindowMode::SpeciesEditor { return; }

    // Coalesce a whole sculpt drag into ONE undo entry: while LMB is held the
    // tracker neither pushes nor updates the baseline, so the pre-stroke `last`
    // is preserved. On release, the accumulated change is recorded as a single
    // entry against that pre-stroke baseline.
    if stroke.0 && !undo.suppress { return; }

    if undo.suppress {
        // Change came from a restore — adopt as the new baseline, don't record.
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


/// Ctrl+Z → restore the most recent prior `body_parts` state. Matches the
/// LOGICAL "z" key so it works on non-QWERTY layouts. Suppressed during a rename
/// so the keystroke edits the name instead.
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
