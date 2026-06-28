// "Clear All" confirmation modal.
//
// "Clear All" sets `show_clear_modal`; visibility tracks the flag edge.
// Yes → drain every template AND despawn every live `OrganismRoot`; No
// → cancel. "No" is the highlighted default (default = cancel for a
// destructive prompt), so an accidental Enter takes the safe option.

use bevy::prelude::*;

use crate::colony_editor::session::EditorSession;
use crate::colony_editor::template::OrganismTemplate;
use crate::organism::OrganismRoot;
use crate::ui_modal::{
    self, ConfirmModalSpec, NoButtonStyle,
    YES_BTN_COLOR, YES_BTN_HOVER,
};


// ── Tunables ────────────────────────────────────────────────────────────────

const MODAL_CARD_WIDTH: f32 = 520.0;


// ── Marker components ───────────────────────────────────────────────────────

#[derive(Component)]
struct ClearModalRoot;

#[derive(Component)]
struct ClearModalYesButton;

#[derive(Component)]
struct ClearModalNoButton;


// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct ClearModalPlugin;

impl Plugin for ClearModalPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (
            manage_clear_modal_visibility,
            handle_clear_modal_buttons,
        ));
    }
}


// ── Modal lifecycle: spawn / despawn based on `show_clear_modal` ───────────

fn manage_clear_modal_visibility(
    mut commands: Commands,
    session:      Res<EditorSession>,
    existing:     Query<Entity, With<ClearModalRoot>>,
) {
    ui_modal::sync_modal_visibility(
        &mut commands, session.show_clear_modal, &existing, spawn_modal);
}


fn spawn_modal(commands: &mut Commands) {
    ui_modal::spawn_confirm_modal(
        commands,
        &ConfirmModalSpec {
            title:      Some("Clear colony".to_string()),
            body:       "Are you sure to delete ALL organisms in the current colony?".to_string(),
            body_font_size: ui_modal::BODY_FONT_SIZE,
            body_color:     ui_modal::BODY_COLOR,
            card_width: MODAL_CARD_WIDTH,
            // 110 > exit_modal's 100, so this wins if both flags are set.
            z_index:    110,
            // No is the highlighted "safe default" for this destructive prompt.
            no_style:   NoButtonStyle::Safe,
        },
        ClearModalRoot,
        ClearModalNoButton,
        ClearModalYesButton,
    );
}


// ── Modal button handlers ───────────────────────────────────────────────────

fn handle_clear_modal_buttons(
    mut yes_q:       Query<(&Interaction, &mut BackgroundColor),
                           (Changed<Interaction>, With<ClearModalYesButton>, Without<ClearModalNoButton>)>,
    mut no_q:        Query<(&Interaction, &mut BackgroundColor),
                           (Changed<Interaction>, With<ClearModalNoButton>, Without<ClearModalYesButton>)>,
    mut session:     ResMut<EditorSession>,
    mut commands:    Commands,
    organisms_q:     Query<Entity, With<OrganismRoot>>,
) {
    for (interaction, mut bg) in &mut yes_q {
        if ui_modal::modal_button_pressed(interaction, &mut bg, YES_BTN_COLOR, YES_BTN_HOVER) {
            // Despawn visual entities before draining templates so a
            // half-cleared frame can't re-spawn rows for dead entities.
            let removed: Vec<OrganismTemplate> = session.templates.drain(..).collect();
            for t in &removed {
                commands.entity(t.entity).despawn();
            }
            session.active_id = None;
            session.dirty     = true;

            // Recursive despawn of every live organism; RemovedComponents
            // observers reclaim brain slots / update counters.
            for e in &organisms_q {
                commands.entity(e).despawn();
            }

            // No undo entry — wiping wild organisms is irreversible
            // (mixing it with the reversible template path would confuse).
            session.show_clear_modal = false;
        }
    }
    let no_style = NoButtonStyle::Safe;
    for (interaction, mut bg) in &mut no_q {
        if ui_modal::modal_button_pressed(interaction, &mut bg, no_style.base(), no_style.hover()) {
            session.show_clear_modal = false;
        }
    }
}
