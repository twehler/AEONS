// Unsaved-work modal + "exit to launcher" dispatcher.
//
// Flow:
//   1. The inventory panel's "Return to Menu" button sets
//      `EditorSession::exit_requested = true`.
//   2. `dispatch_exit_request` consumes that flag:
//        * clean state → call `exit_to_launcher()` immediately
//        * dirty state → set `show_exit_modal = true`, which causes
//          `manage_modal_visibility` to spawn the modal entity.
//   3. The modal's Yes / No buttons resolve the choice. Yes →
//      `exit_to_launcher()`. No → drop both flags, despawn modal.
//
// `exit_to_launcher()` re-spawns the binary as a launcher (no argv)
// and terminates this process via `AppExit::Success`.

use std::process::Command;

use bevy::prelude::*;
use bevy::app::AppExit;

use crate::colony_editor::session::EditorSession;


// ── Tunables ────────────────────────────────────────────────────────────────

const MODAL_BACKDROP_COLOR: Color = Color::srgba(0.0, 0.0, 0.0, 0.55);
const MODAL_CARD_COLOR:     Color = Color::srgb(0.15, 0.15, 0.18);
const MODAL_CARD_BORDER:    Color = Color::srgb(0.40, 0.40, 0.45);

const MODAL_CARD_WIDTH:     f32   = 460.0;
const MODAL_CARD_PADDING:   f32   = 22.0;

const MODAL_BTN_WIDTH:      f32   = 110.0;
const MODAL_BTN_HEIGHT:     f32   = 36.0;
const MODAL_BTN_GAP:        f32   = 16.0;

const YES_BTN_COLOR:        Color = Color::srgb(0.55, 0.18, 0.18);
const YES_BTN_HOVER:        Color = Color::srgb(0.68, 0.22, 0.22);
const NO_BTN_COLOR:         Color = Color::srgb(0.20, 0.45, 0.30);
const NO_BTN_HOVER:         Color = Color::srgb(0.26, 0.55, 0.36);


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
        // Defer the actual exit until the user resolves the modal.
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
    let want_visible = session.show_exit_modal;
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
            ExitModalRoot,
            // Full-screen backdrop. Picks up clicks so they don't
            // "fall through" to the editor below; both modal buttons
            // sit on top of it and consume their own clicks.
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
            // Render on top of every other UI node. 100 is well
            // above the panels (which use the default zindex of 0).
            GlobalZIndex(100),
        ))
        .with_children(|root| {
            // Card.
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
                        Text::new("Your work is not saved."),
                        TextFont { font_size: 18.0, ..default() },
                        TextColor(Color::WHITE),
                        Node { margin: UiRect::bottom(Val::Px(6.0)), ..default() },
                        Pickable::IGNORE,
                    ));
                    card.spawn((
                        Text::new("Are you sure you want to return to the main menu?"),
                        TextFont { font_size: 14.0, ..default() },
                        TextColor(Color::srgb(0.85, 0.85, 0.85)),
                        Node { margin: UiRect::bottom(Val::Px(20.0)), ..default() },
                        Pickable::IGNORE,
                    ));

                    // Button row.
                    card.spawn(Node {
                        flex_direction:  FlexDirection::Row,
                        justify_content: JustifyContent::Center,
                        ..default()
                    })
                    .with_children(|row| {
                        // No (cancel) on the left.
                        row.spawn((
                            ModalNoButton,
                            Button,
                            Node {
                                width:  Val::Px(MODAL_BTN_WIDTH),
                                height: Val::Px(MODAL_BTN_HEIGHT),
                                margin: UiRect::right(Val::Px(MODAL_BTN_GAP)),
                                align_items:     AlignItems::Center,
                                justify_content: JustifyContent::Center,
                                ..default()
                            },
                            BackgroundColor(NO_BTN_COLOR),
                        ))
                        .with_children(|btn| {
                            btn.spawn((
                                Text::new("No"),
                                TextFont { font_size: 16.0, ..default() },
                                TextColor(Color::WHITE),
                                Pickable::IGNORE,
                            ));
                        });

                        // Yes (confirm exit) on the right.
                        row.spawn((
                            ModalYesButton,
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

fn handle_modal_buttons(
    mut yes_q: Query<(&Interaction, &mut BackgroundColor),
                     (Changed<Interaction>, With<ModalYesButton>, Without<ModalNoButton>)>,
    mut no_q:  Query<(&Interaction, &mut BackgroundColor),
                     (Changed<Interaction>, With<ModalNoButton>, Without<ModalYesButton>)>,
    mut session: ResMut<EditorSession>,
    mut exit:    MessageWriter<AppExit>,
) {
    for (interaction, mut bg) in &mut yes_q {
        match *interaction {
            Interaction::Pressed => {
                session.show_exit_modal = false;
                exit_to_launcher(&mut exit);
                *bg = BackgroundColor(YES_BTN_HOVER);
            }
            Interaction::Hovered => *bg = BackgroundColor(YES_BTN_HOVER),
            Interaction::None    => *bg = BackgroundColor(YES_BTN_COLOR),
        }
    }
    for (interaction, mut bg) in &mut no_q {
        match *interaction {
            Interaction::Pressed => {
                session.show_exit_modal = false;
                *bg = BackgroundColor(NO_BTN_HOVER);
            }
            Interaction::Hovered => *bg = BackgroundColor(NO_BTN_HOVER),
            Interaction::None    => *bg = BackgroundColor(NO_BTN_COLOR),
        }
    }
}


// ── Exit transition ─────────────────────────────────────────────────────────

/// Re-spawn the binary as a fresh launcher (no argv → launcher mode)
/// and ask Bevy to shut down the current App. The new launcher
/// process is independent; the original `cargo run` shell sees its
/// child exit and itself returns, but by then the user already has
/// the new launcher window in front of them.
fn exit_to_launcher(exit: &mut MessageWriter<AppExit>) {
    match std::env::current_exe() {
        Ok(exe) => {
            // We deliberately don't .wait() — fire and forget. If the
            // spawn fails we log it but still exit so the user isn't
            // stuck staring at a frozen editor.
            match Command::new(exe).spawn() {
                Ok(_)  => info!("re-spawned launcher"),
                Err(e) => error!("failed to re-spawn launcher: {e}"),
            }
        }
        Err(e) => error!("failed to locate own executable: {e}"),
    }
    exit.write(AppExit::Success);
}
