// Editor undo system + global keyboard shortcuts.
//
//   * Ctrl+Z — pop the most recent `EditorAction` and reverse it.
//   * Ctrl+S — set `save_requested` (consumed by `mod.rs`).
//
// Undoable actions: Created (despawn each), Deleted (re-spawn), Cleared
// (re-spawn all). The stack is capped at `MAX_UNDO_DEPTH`.

use bevy::prelude::*;
use bevy::input::keyboard::{Key, KeyboardInput};

use crate::colony_editor::placement::respawn_template;
use crate::colony_editor::session::EditorSession;
use crate::colony_editor::template::OrganismTemplate;


/// Undo-stack depth; oldest action drops once exceeded.
const MAX_UNDO_DEPTH: usize = 100;


/// Single reversible edit (undo-only; no redo).
pub enum EditorAction {
    /// Templates just created (single-click or bulk). Undo despawns each.
    Created(Vec<u32>),
    /// A deleted template. Undo re-spawns it.
    Deleted(OrganismTemplate),
    /// "Clear All". Undo re-spawns every snapshotted template.
    Cleared(Vec<OrganismTemplate>),
}


/// LIFO edit history, capped at `MAX_UNDO_DEPTH`.
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

    /// Wipe the history. Unused; reserved for a future "load a
    /// different colony" flow.
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
                // Merged: only while EditColony active. Standalone: always.
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
// Both shortcuts match the LOGICAL key (`Key::Character`), not the
// physical `KeyCode`, so Ctrl+the-key-labelled-Z/S works regardless of
// keyboard layout (QWERTZ/AZERTY/Dvorak).

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
            // Stays dirty unless both the template list and the stack are empty.
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
    // Mirrors the Save Colony… button; consumed by `dispatch_save_requests`.
    session.save_requested = true;
}


// ── Helpers ─────────────────────────────────────────────────────────────────

fn ctrl_held(keys: &ButtonInput<KeyCode>) -> bool {
    keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight)
}
