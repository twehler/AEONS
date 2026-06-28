// Unsaved-work modal + "exit to launcher" dispatcher.
//
// "Return to Menu" sets `exit_requested`; `dispatch_exit_request`
// exits immediately if clean, or raises `show_exit_modal` if dirty.
// The modal's Yes → `exit_to_launcher`, No → cancel. `exit_to_launcher`
// re-spawns the binary as a launcher (no argv) and `AppExit::Success`.

use std::process::Command;

use bevy::prelude::*;
use bevy::app::AppExit;

use crate::colony_editor::session::EditorSession;
use crate::ui_modal::{
    self, ConfirmModalSpec, NoButtonStyle,
    YES_BTN_COLOR, YES_BTN_HOVER,
};


// ── Tunables ────────────────────────────────────────────────────────────────

const MODAL_CARD_WIDTH: f32 = 460.0;


// ── Marker components ───────────────────────────────────────────────────────

#[derive(Component)]
struct ExitModalRoot;

#[derive(Component)]
struct ModalYesButton;

#[derive(Component)]
struct ModalNoButton;


// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct ExitModalPlugin;

impl Plugin for ExitModalPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (
            dispatch_exit_request,
            manage_modal_visibility,
            handle_modal_buttons,
        ));
    }
}


// ── Dispatcher: button click → modal-or-exit ────────────────────────────────

fn dispatch_exit_request(
    mut session: ResMut<EditorSession>,
    mut exit:    MessageWriter<AppExit>,
) {
    if !session.exit_requested { return; }
    session.exit_requested = false;

    if session.dirty {
        // Defer exit until the user resolves the modal.
        session.show_exit_modal = true;
    } else {
        exit_to_launcher(&mut exit);
    }
}


// ── Modal lifecycle: spawn / despawn based on `show_exit_modal` ─────────────

fn manage_modal_visibility(
    mut commands:  Commands,
    session:       Res<EditorSession>,
    existing:      Query<Entity, With<ExitModalRoot>>,
) {
    ui_modal::sync_modal_visibility(
        &mut commands, session.show_exit_modal, &existing, spawn_modal);
}


fn spawn_modal(commands: &mut Commands) {
    ui_modal::spawn_confirm_modal(
        commands,
        &ConfirmModalSpec {
            title:      Some("Your work is not saved.".to_string()),
            body:       "Are you sure you want to return to the main menu?".to_string(),
            body_font_size: ui_modal::BODY_FONT_SIZE,
            body_color:     ui_modal::BODY_COLOR,
            card_width: MODAL_CARD_WIDTH,
            z_index:    100, // above the panels (default zindex 0)
            no_style:   NoButtonStyle::Plain,
        },
        ExitModalRoot,
        ModalNoButton,
        ModalYesButton,
    );
}


// ── Modal button handlers ───────────────────────────────────────────────────

fn handle_modal_buttons(
    mut yes_q: Query<(&Interaction, &mut BackgroundColor),
                     (Changed<Interaction>, With<ModalYesButton>, Without<ModalNoButton>)>,
    mut no_q:  Query<(&Interaction, &mut BackgroundColor),
                     (Changed<Interaction>, With<ModalNoButton>, Without<ModalYesButton>)>,
    mut session: ResMut<EditorSession>,
    mut exit:    MessageWriter<AppExit>,
) {
    for (interaction, mut bg) in &mut yes_q {
        if ui_modal::modal_button_pressed(interaction, &mut bg, YES_BTN_COLOR, YES_BTN_HOVER) {
            session.show_exit_modal = false;
            exit_to_launcher(&mut exit);
        }
    }
    let no_style = NoButtonStyle::Plain;
    for (interaction, mut bg) in &mut no_q {
        if ui_modal::modal_button_pressed(interaction, &mut bg, no_style.base(), no_style.hover()) {
            session.show_exit_modal = false;
        }
    }
}


// ── Exit transition ─────────────────────────────────────────────────────────

/// Re-spawn the binary as a fresh launcher (no argv) and shut down the
/// current App. The new launcher process is independent.
fn exit_to_launcher(exit: &mut MessageWriter<AppExit>) {
    match std::env::current_exe() {
        Ok(exe) => {
            // Fire-and-forget (no .wait()); on spawn failure still exit
            // so the user isn't left in a dead editor.
            match Command::new(exe).spawn() {
                Ok(_)  => info!("re-spawned launcher"),
                Err(e) => error!("failed to re-spawn launcher: {e}"),
            }
        }
        Err(e) => error!("failed to locate own executable: {e}"),
    }
    exit.write(AppExit::Success);
}
