// Editor undo system + global keyboard shortcuts.
//
// Two shortcuts handled here:
//   * Ctrl+Z — pop the most recent `EditorAction` off the undo
//     stack and reverse its effect on `EditorSession::templates`.
//   * Ctrl+S — set `session.save_requested = true`, which
//     `dispatch_save_requests` in `mod.rs` consumes to open the
//     file dialog and write a `.colony` file.
//
// What can be undone:
//   * Created(ids)            — single-click placement OR bulk-add.
//                               Undo despawns each entity and
//                               removes it from `session.templates`.
//   * Deleted(template)       — right-click delete.
//                               Undo re-spawns the visual + pushes
//                               the template back into the list
//                               with a fresh `Entity` reference.
//   * Cleared(templates)      — inventory panel's "Clear All".
//                               Undo re-spawns every snapshotted
//                               template at its original position.
//
// The stack is capped (`MAX_UNDO_DEPTH`) so a very long editing
// session doesn't grow unbounded. Older actions silently fall off
// the bottom when the cap is exceeded.

use bevy::prelude::*;
use bevy::input::keyboard::{Key, KeyboardInput};

use crate::colony_editor::placement::respawn_template;
use crate::colony_editor::session::EditorSession;
use crate::colony_editor::template::OrganismTemplate;


/// How many actions stay on the undo stack. After this many edits
/// the oldest action is dropped on each new push. 100 is generous
/// for a single editing session and bounded in memory.
const MAX_UNDO_DEPTH: usize = 100;


/// Single reversible edit. Each variant carries enough information
/// to UNDO itself — the redo direction isn't currently supported.
pub enum EditorAction {
    /// A batch of templates that were just created. Single-click
    /// placement produces `Created(vec![id])`; bulk-add produces
    /// `Created(vec![ids…])`. Undo despawns each.
    Created(Vec<u32>),
    /// A template that was deleted. Undo re-spawns it.
    Deleted(OrganismTemplate),
    /// "Clear All" — many templates wiped at once. Undo re-spawns
    /// every one of them with fresh visual entities.
    Cleared(Vec<OrganismTemplate>),
}


/// LIFO history of edits since the editor was opened (capped at
/// `MAX_UNDO_DEPTH`). Mutated by every system that creates / deletes
/// templates.
#[derive(Resource, Default)]
pub struct UndoStack {
    actions: Vec<EditorAction>,
}

impl UndoStack {
    pub fn push(&mut self, action: EditorAction) {
        self.actions.push(action);
        if self.actions.len() > MAX_UNDO_DEPTH {
            self.actions.remove(0);
        }
    }

    fn pop(&mut self) -> Option<EditorAction> {
        self.actions.pop()
    }

    /// Wipe the entire history. Currently unused; reserved for a
    /// future "load a different colony into the editor" flow that
    /// would otherwise allow undo to resurrect templates from the
    /// previous session.
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.actions.clear();
    }
}


// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct UndoPlugin;

impl Plugin for UndoPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<UndoStack>()
            .add_systems(Update, (
                handle_undo_shortcut,
                handle_save_shortcut,
            )
                // Merged mode: only consume Ctrl+Z / Ctrl+S while
                // EditColony is active. Standalone editor (no
                // WindowMode resource) always fires.
                .run_if(in_edit_mode_or_standalone));
    }
}

fn in_edit_mode_or_standalone(mode: Option<Res<crate::simulation_settings::WindowMode>>) -> bool {
    match mode {
        Some(m) => *m == crate::simulation_settings::WindowMode::EditColony,
        None    => true,
    }
}


// ── Shortcut handlers ───────────────────────────────────────────────────────
//
// Both shortcuts match on the LOGICAL key (`Key::Character`) rather
// than the physical key code. `KeyCode::KeyZ` corresponds to the
// USB-HID position of "Z" on a US QWERTY layout — on QWERTZ
// (German) the key the user reads as "Z" sits at the physical
// KeyY position, on AZERTY at KeyW, on Dvorak elsewhere again. The
// logical-key path means Ctrl + the-key-labelled-Z works on every
// layout.

fn handle_undo_shortcut(
    keys:           Res<ButtonInput<KeyCode>>,
    mut events:     MessageReader<KeyboardInput>,
    mut undo_stack: ResMut<UndoStack>,
    mut session:    ResMut<EditorSession>,
    mut commands:   Commands,
    mut meshes:     ResMut<Assets<Mesh>>,
    mut materials:  ResMut<Assets<StandardMaterial>>,
    org_materials:  Option<Res<crate::colony::OrganismMaterials>>,
    smoothing:      Option<Res<crate::simulation_settings::Smoothing>>,
) {
    let smoothing_on = smoothing.as_deref().map(|s| s.0).unwrap_or(true);
    let org_mats_ref = org_materials.as_deref();
    let ctrl = ctrl_held(&keys);
    let mut trigger = false;
    for ev in events.read() {
        if !ev.state.is_pressed() { continue; }
        if !ctrl { continue; }
        if let Key::Character(s) = &ev.logical_key {
            if s.eq_ignore_ascii_case("z") {
                trigger = true;
            }
        }
    }
    if !trigger { return; }

    let Some(action) = undo_stack.pop() else { return; };
    match action {
        EditorAction::Created(ids) => {
            for id in ids {
                if let Some(idx) = session.templates.iter().position(|t| t.id == id) {
                    let removed = session.templates.swap_remove(idx);
                    commands.entity(removed.entity).despawn();
                    if session.active_id == Some(id) {
                        session.active_id = None;
                    }
                }
            }
            // After an undo of a creation, the user is back to the
            // state before the edit — still treated as dirty unless
            // the stack is empty AND the template list is empty.
            session.dirty = !session.templates.is_empty()
                          || !undo_stack.actions.is_empty();
        }

        EditorAction::Deleted(template) => {
            let new_entity = respawn_template(
                &template, &mut commands, &mut meshes, &mut materials,
                org_mats_ref, smoothing_on,
            );
            session.templates.push(OrganismTemplate {
                entity: new_entity,
                ..template
            });
            session.dirty = true;
        }

        EditorAction::Cleared(templates) => {
            for t in templates {
                let new_entity = respawn_template(
                    &t, &mut commands, &mut meshes, &mut materials,
                    org_mats_ref, smoothing_on,
                );
                session.templates.push(OrganismTemplate {
                    entity: new_entity,
                    ..t
                });
            }
            session.dirty = true;
        }
    }
}

fn handle_save_shortcut(
    keys:        Res<ButtonInput<KeyCode>>,
    mut events:  MessageReader<KeyboardInput>,
    mut session: ResMut<EditorSession>,
) {
    let ctrl = ctrl_held(&keys);
    let mut trigger = false;
    for ev in events.read() {
        if !ev.state.is_pressed() { continue; }
        if !ctrl { continue; }
        if let Key::Character(s) = &ev.logical_key {
            if s.eq_ignore_ascii_case("s") {
                trigger = true;
            }
        }
    }
    if !trigger { return; }
    // Mirrors the Save Colony… button — `dispatch_save_requests`
    // consumes this flag on the next tick to open the file dialog.
    session.save_requested = true;
}


// ── Helpers ─────────────────────────────────────────────────────────────────

fn ctrl_held(keys: &ButtonInput<KeyCode>) -> bool {
    keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight)
}
