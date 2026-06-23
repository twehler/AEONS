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


// ── Tunables ────────────────────────────────────────────────────────────────

const MODAL_BACKDROP_COLOR: Color = Color::srgba(0.0, 0.0, 0.0, 0.55);
const MODAL_CARD_COLOR:     Color = Color::srgb(0.15, 0.15, 0.18);
const MODAL_CARD_BORDER:    Color = Color::srgb(0.40, 0.40, 0.45);

const MODAL_CARD_WIDTH:     f32   = 520.0;
const MODAL_CARD_PADDING:   f32   = 22.0;

const MODAL_BTN_WIDTH:      f32   = 110.0;
const MODAL_BTN_HEIGHT:     f32   = 36.0;
const MODAL_BTN_GAP:        f32   = 16.0;

// Yes (destructive): muted red, no special border.
const YES_BTN_COLOR:        Color = Color::srgb(0.55, 0.18, 0.18);
const YES_BTN_HOVER:        Color = Color::srgb(0.68, 0.22, 0.22);

// No (highlighted default): brighter green + 2px white-ish outline so
// it visually pops as the recommended choice.
const NO_BTN_COLOR:         Color = Color::srgb(0.24, 0.56, 0.36);
const NO_BTN_HOVER:         Color = Color::srgb(0.32, 0.66, 0.42);
const NO_BTN_BORDER:        Color = Color::srgb(0.95, 0.95, 0.95);
const NO_BTN_BORDER_WIDTH:  f32   = 2.0;


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
            // Full-screen backdrop blocks clicks reaching the editor.
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
            // 110 > exit_modal's 100, so this wins if both flags are set.
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
                        Text::new("Clear colony"),
                        TextFont { font_size: 18.0, ..default() },
                        TextColor(Color::WHITE),
                        Node { margin: UiRect::bottom(Val::Px(6.0)), ..default() },
                        Pickable::IGNORE,
                    ));
                    card.spawn((
                        Text::new("Are you sure to delete ALL organisms in the current colony?"),
                        TextFont { font_size: 14.0, ..default() },
                        TextColor(Color::srgb(0.85, 0.85, 0.85)),
                        Node { margin: UiRect::bottom(Val::Px(20.0)), ..default() },
                        Pickable::IGNORE,
                    ));

                    // Button row; No first (highlighted, the safe default).
                    card.spawn(Node {
                        flex_direction:  FlexDirection::Row,
                        justify_content: JustifyContent::Center,
                        align_items:     AlignItems::Center,
                        ..default()
                    })
                    .with_children(|row| {
                        // ── No (highlighted) ────────────────────
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

                        // ── Yes (destructive) ───────────────────
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
        match *interaction {
            Interaction::Pressed => {
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
