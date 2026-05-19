// Species editor — Clear/New button + unsaved-changes confirmation modal.
//
// Layout:
//   * A floating "Clear/New" button anchored to the bottom-right of
//     the editor's viewport area, just above the bottom panel. Same
//     visibility lifecycle as the other species-editor panels (toggled
//     via the `SpeciesEditorPanel` marker in
//     `frontend::apply_mode_transition`).
//   * A full-screen modal that only spawns when
//     `SpeciesSession::show_clear_modal == true`. Two buttons, "No"
//     highlighted as the default (brighter green + 2-px outline) and
//     "Yes" muted-red as the destructive option. Same convention as
//     `colony_editor::clear_modal`.
//
// Flow:
//   1. User clicks "Clear/New".
//      * If `session.dirty == false` → reset session + despawn editor
//        visuals immediately. No modal.
//      * If `session.dirty == true`  → set `show_clear_modal = true`.
//   2. Modal lifecycle system spawns the modal entity on the rising
//      edge.
//   3. Button handler:
//      * Yes → reset the session, despawn editor visuals, drop the
//        flag.
//      * No  → just drop the flag.

use bevy::prelude::*;

use crate::frontend::PANEL_BG_COLOR;
use crate::simulation_settings::WindowMode;

use super::placement::{SpeciesBilateralAxis, SpeciesEditorMesh, SpeciesPreviewCell};
use super::session::SpeciesSession;
use super::{BOTTOM_PANEL_HEIGHT_PX, SpeciesEditorPanel};


// ── Tunables ────────────────────────────────────────────────────────────────

const BTN_WIDTH:   f32 = 160.0;
const BTN_HEIGHT:  f32 = 36.0;
const BTN_MARGIN:  f32 = 12.0;

const CLEAR_BTN_COLOR: Color = Color::srgb(0.55, 0.18, 0.18);
const CLEAR_BTN_HOVER: Color = Color::srgb(0.68, 0.22, 0.22);

const MODAL_BACKDROP_COLOR: Color = Color::srgba(0.0, 0.0, 0.0, 0.55);
const MODAL_CARD_COLOR:     Color = Color::srgb(0.15, 0.15, 0.18);
const MODAL_CARD_BORDER:    Color = Color::srgb(0.40, 0.40, 0.45);

const MODAL_CARD_WIDTH:     f32   = 520.0;
const MODAL_CARD_PADDING:   f32   = 22.0;

const MODAL_BTN_WIDTH:      f32   = 110.0;
const MODAL_BTN_HEIGHT:     f32   = 36.0;
const MODAL_BTN_GAP:        f32   = 16.0;

const YES_BTN_COLOR:        Color = Color::srgb(0.55, 0.18, 0.18);
const YES_BTN_HOVER:        Color = Color::srgb(0.68, 0.22, 0.22);

const NO_BTN_COLOR:         Color = Color::srgb(0.24, 0.56, 0.36);
const NO_BTN_HOVER:         Color = Color::srgb(0.32, 0.66, 0.42);
const NO_BTN_BORDER:        Color = Color::srgb(0.95, 0.95, 0.95);
const NO_BTN_BORDER_WIDTH:  f32   = 2.0;


// ── Markers ─────────────────────────────────────────────────────────────────

#[derive(Component)]
pub struct ClearNewButton;

#[derive(Component)]
struct ClearModalRoot;

#[derive(Component)]
struct ClearModalYesButton;

#[derive(Component)]
struct ClearModalNoButton;

// Mute the `PANEL_BG_COLOR` import — we want the bottom-right button
// to sit on the viewport, not on a panel background. Imported only so
// future styling can pick it up without re-plumbing.
const _: Color = PANEL_BG_COLOR;


// ── Spawn the floating Clear/New button ─────────────────────────────────────

/// Called from `mod.rs::spawn_overlay_panels`. Anchored to the bottom-
/// right of the screen, sitting just above the bottom panel so it
/// doesn't overlap the cell-type tiles.
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
                    // Defer to the modal.
                    session.show_clear_modal = true;
                } else {
                    // Nothing to lose — clear immediately.
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
    let want_visible = session.show_clear_modal;
    let is_visible   = !existing.is_empty();

    if want_visible && !is_visible {
        spawn_modal(&mut commands);
    } else if !want_visible && is_visible {
        for e in &existing { commands.entity(e).despawn(); }
    }
}


fn spawn_modal(commands: &mut Commands) {
    commands
        .spawn((
            ClearModalRoot,
            Node {
                position_type: PositionType::Absolute,
                top:    Val::Px(0.0),
                left:   Val::Px(0.0),
                width:  Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                align_items:     AlignItems::Center,
                ..default()
            },
            BackgroundColor(MODAL_BACKDROP_COLOR),
            GlobalZIndex(110),
        ))
        .with_children(|root| {
            root
                .spawn((
                    Node {
                        width:  Val::Px(MODAL_CARD_WIDTH),
                        flex_direction: FlexDirection::Column,
                        align_items:    AlignItems::Center,
                        padding: UiRect::all(Val::Px(MODAL_CARD_PADDING)),
                        border:  UiRect::all(Val::Px(1.0)),
                        ..default()
                    },
                    BackgroundColor(MODAL_CARD_COLOR),
                    BorderColor::all(MODAL_CARD_BORDER),
                ))
                .with_children(|card| {
                    card.spawn((
                        Text::new("Clear species"),
                        TextFont { font_size: 18.0, ..default() },
                        TextColor(Color::WHITE),
                        Node { margin: UiRect::bottom(Val::Px(6.0)), ..default() },
                        Pickable::IGNORE,
                    ));
                    card.spawn((
                        Text::new("Are you sure? Your progress hasn't been saved."),
                        TextFont { font_size: 14.0, ..default() },
                        TextColor(Color::srgb(0.85, 0.85, 0.85)),
                        Node { margin: UiRect::bottom(Val::Px(20.0)), ..default() },
                        Pickable::IGNORE,
                    ));

                    // Button row. "No" first (left, highlighted) so the
                    // safe option is the visual default.
                    card.spawn(Node {
                        flex_direction:  FlexDirection::Row,
                        justify_content: JustifyContent::Center,
                        align_items:     AlignItems::Center,
                        ..default()
                    })
                    .with_children(|row| {
                        // ── No (highlighted) ──────────────────────
                        row.spawn((
                            ClearModalNoButton,
                            Button,
                            Node {
                                width:  Val::Px(MODAL_BTN_WIDTH),
                                height: Val::Px(MODAL_BTN_HEIGHT),
                                margin: UiRect::right(Val::Px(MODAL_BTN_GAP)),
                                border: UiRect::all(Val::Px(NO_BTN_BORDER_WIDTH)),
                                align_items:     AlignItems::Center,
                                justify_content: JustifyContent::Center,
                                ..default()
                            },
                            BackgroundColor(NO_BTN_COLOR),
                            BorderColor::all(NO_BTN_BORDER),
                        ))
                        .with_children(|btn| {
                            btn.spawn((
                                Text::new("No"),
                                TextFont { font_size: 16.0, ..default() },
                                TextColor(Color::WHITE),
                                Pickable::IGNORE,
                            ));
                        });

                        // ── Yes (destructive) ─────────────────────
                        row.spawn((
                            ClearModalYesButton,
                            Button,
                            Node {
                                width:  Val::Px(MODAL_BTN_WIDTH),
                                height: Val::Px(MODAL_BTN_HEIGHT),
                                align_items:     AlignItems::Center,
                                justify_content: JustifyContent::Center,
                                ..default()
                            },
                            BackgroundColor(YES_BTN_COLOR),
                        ))
                        .with_children(|btn| {
                            btn.spawn((
                                Text::new("Yes"),
                                TextFont { font_size: 16.0, ..default() },
                                TextColor(Color::WHITE),
                                Pickable::IGNORE,
                            ));
                        });
                    });
                });
        });
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
        match *interaction {
            Interaction::Pressed => {
                perform_clear(&mut session, &mut commands, &mesh_q, &preview_q, &axis_q);
                session.show_clear_modal = false;
                *bg = BackgroundColor(YES_BTN_HOVER);
            }
            Interaction::Hovered => *bg = BackgroundColor(YES_BTN_HOVER),
            Interaction::None    => *bg = BackgroundColor(YES_BTN_COLOR),
        }
    }
    for (interaction, mut bg) in &mut no_q {
        match *interaction {
            Interaction::Pressed => {
                session.show_clear_modal = false;
                *bg = BackgroundColor(NO_BTN_HOVER);
            }
            Interaction::Hovered => *bg = BackgroundColor(NO_BTN_HOVER),
            Interaction::None    => *bg = BackgroundColor(NO_BTN_COLOR),
        }
    }
}


// ── Shared reset path ───────────────────────────────────────────────────────

/// Reset the session and despawn the 3D entities the placement systems
/// own. Once `session.ocg` is empty, `refresh_species_mesh` and friends
/// won't re-spawn anything until the user spawns a new first cell.
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
