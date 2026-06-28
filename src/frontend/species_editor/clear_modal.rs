// Species editor — Clear/New button + unsaved-changes confirmation modal.
//
// Clicking Clear/New resets immediately when not dirty, else raises
// `show_clear_modal`. The modal lifecycle system spawns/despawns the full-screen
// modal on that flag; "Yes" resets + despawns visuals, "No" just drops the flag.
// "No" is the highlighted default (safe option), matching
// `colony_editor::clear_modal`.

use bevy::prelude::*;

use crate::frontend::PANEL_BG_COLOR;
use crate::simulation_settings::WindowMode;
use crate::ui_modal::{
    self, ConfirmModalSpec, NoButtonStyle,
    YES_BTN_COLOR, YES_BTN_HOVER,
};

use super::placement::{SpeciesBilateralAxis, SpeciesEditorMesh, SpeciesPreviewCell};
use super::session::SpeciesSession;
use super::{BOTTOM_PANEL_HEIGHT_PX, SpeciesEditorPanel};


// ── Tunables ────────────────────────────────────────────────────────────────

const BTN_WIDTH:   f32 = 160.0;
const BTN_HEIGHT:  f32 = 36.0;
const BTN_MARGIN:  f32 = 12.0;

const CLEAR_BTN_COLOR: Color = Color::srgb(0.55, 0.18, 0.18);
const CLEAR_BTN_HOVER: Color = Color::srgb(0.68, 0.22, 0.22);

const MODAL_CARD_WIDTH: f32 = 520.0;


// ── Markers ─────────────────────────────────────────────────────────────────

#[derive(Component)]
pub struct ClearNewButton;

#[derive(Component)]
struct ClearModalRoot;

#[derive(Component)]
struct ClearModalYesButton;

#[derive(Component)]
struct ClearModalNoButton;

// Mute the unused `PANEL_BG_COLOR` import (button sits on the viewport, not a
// panel background).
const _: Color = PANEL_BG_COLOR;


// ── Spawn the floating Clear/New button ─────────────────────────────────────

/// Anchored bottom-right, above the bottom panel so it doesn't overlap the
/// cell-type tiles. Called from `mod.rs::spawn_overlay_panels`.
pub fn spawn_clear_new_button(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((
            ClearNewButton,
            SpeciesEditorPanel,
            Button,
            Node {
                position_type:   PositionType::Absolute,
                bottom:          Val::Px(BOTTOM_PANEL_HEIGHT_PX + BTN_MARGIN),
                right:           Val::Px(BTN_MARGIN),
                width:           Val::Px(BTN_WIDTH),
                height:          Val::Px(BTN_HEIGHT),
                align_items:     AlignItems::Center,
                justify_content: JustifyContent::Center,
                display:         Display::None,  // shown only in SpeciesEditor mode
                ..default()
            },
            BackgroundColor(CLEAR_BTN_COLOR),
        ))
        .with_children(|btn| {
            btn.spawn((
                Text::new("Clear/New".to_string()),
                TextFont { font_size: 14.0, ..default() },
                TextColor(Color::WHITE),
                Pickable::IGNORE,
            ));
        });
}


// ── Plugin ──────────────────────────────────────────────────────────────────

pub struct ClearModalPlugin;

impl Plugin for ClearModalPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (
            handle_clear_new_button,
            manage_clear_modal_visibility,
            handle_clear_modal_buttons,
        ));
    }
}


// ── Top-level Clear/New click ───────────────────────────────────────────────

fn handle_clear_new_button(
    mode:             Res<WindowMode>,
    mut interactions: Query<(&Interaction, &mut BackgroundColor),
                            (Changed<Interaction>, With<ClearNewButton>)>,
    mut session:      ResMut<SpeciesSession>,
    mut commands:     Commands,
    mesh_q:           Query<Entity, With<SpeciesEditorMesh>>,
    preview_q:        Query<Entity, With<SpeciesPreviewCell>>,
    axis_q:           Query<Entity, With<SpeciesBilateralAxis>>,
) {
    if *mode != WindowMode::SpeciesEditor { return; }

    for (interaction, mut bg) in &mut interactions {
        match *interaction {
            Interaction::Pressed => {
                if session.dirty {
                    session.show_clear_modal = true;
                } else {
                    perform_clear(&mut session, &mut commands, &mesh_q, &preview_q, &axis_q);
                }
                *bg = BackgroundColor(CLEAR_BTN_HOVER);
            }
            Interaction::Hovered => *bg = BackgroundColor(CLEAR_BTN_HOVER),
            Interaction::None    => *bg = BackgroundColor(CLEAR_BTN_COLOR),
        }
    }
}


// ── Modal lifecycle ─────────────────────────────────────────────────────────

fn manage_clear_modal_visibility(
    mut commands: Commands,
    session:      Res<SpeciesSession>,
    existing:     Query<Entity, With<ClearModalRoot>>,
) {
    ui_modal::sync_modal_visibility(
        &mut commands, session.show_clear_modal, &existing, spawn_modal);
}


fn spawn_modal(commands: &mut Commands) {
    ui_modal::spawn_confirm_modal(
        commands,
        &ConfirmModalSpec {
            title:      Some("Clear species".to_string()),
            body:       "Are you sure? Your progress hasn't been saved.".to_string(),
            body_font_size: ui_modal::BODY_FONT_SIZE,
            body_color:     ui_modal::BODY_COLOR,
            card_width: MODAL_CARD_WIDTH,
            z_index:    110,
            no_style:   NoButtonStyle::Safe,
        },
        ClearModalRoot,
        ClearModalNoButton,
        ClearModalYesButton,
    );
}


// ── Modal button handlers ───────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn handle_clear_modal_buttons(
    mut yes_q:       Query<(&Interaction, &mut BackgroundColor),
                           (Changed<Interaction>, With<ClearModalYesButton>, Without<ClearModalNoButton>)>,
    mut no_q:        Query<(&Interaction, &mut BackgroundColor),
                           (Changed<Interaction>, With<ClearModalNoButton>, Without<ClearModalYesButton>)>,
    mut session:     ResMut<SpeciesSession>,
    mut commands:    Commands,
    mesh_q:          Query<Entity, With<SpeciesEditorMesh>>,
    preview_q:       Query<Entity, With<SpeciesPreviewCell>>,
    axis_q:          Query<Entity, With<SpeciesBilateralAxis>>,
) {
    for (interaction, mut bg) in &mut yes_q {
        if ui_modal::modal_button_pressed(interaction, &mut bg, YES_BTN_COLOR, YES_BTN_HOVER) {
            perform_clear(&mut session, &mut commands, &mesh_q, &preview_q, &axis_q);
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


// ── Shared reset path ───────────────────────────────────────────────────────

/// Reset the session and despawn the 3D entities the placement systems own.
fn perform_clear(
    session:   &mut SpeciesSession,
    commands:  &mut Commands,
    mesh_q:    &Query<Entity, With<SpeciesEditorMesh>>,
    preview_q: &Query<Entity, With<SpeciesPreviewCell>>,
    axis_q:    &Query<Entity, With<SpeciesBilateralAxis>>,
) {
    session.reset();
    for e in mesh_q.iter()    { commands.entity(e).despawn(); }
    for e in preview_q.iter() { commands.entity(e).despawn(); }
    for e in axis_q.iter()    { commands.entity(e).despawn(); }
}
