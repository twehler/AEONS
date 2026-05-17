// "Clear All" confirmation modal.
//
// Flow:
//   1. The inventory panel's "Clear All" button sets
//      `EditorSession::show_clear_modal = true`.
//   2. `manage_clear_modal_visibility` spawns the modal entity on
//      the rising edge of that flag and despawns it on the falling
//      edge.
//   3. `handle_clear_modal_buttons` resolves the Yes / No choice:
//        * Yes → drain every editor template AND despawn every live
//          `OrganismRoot` in the simulation, then drop the flag.
//        * No  → drop the flag, despawn the modal.
//
// "No" is rendered in the highlighted style (brighter colour +
// 2-pixel border) so that an accidental Enter / glance-and-click
// goes to the safe option. This mirrors the "default = cancel"
// convention for destructive prompts on every modern desktop OS —
// the user has to actively reach past the highlighted choice to
// confirm the destruction.

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
            // Full-screen backdrop — same pattern as `exit_modal`.
            // Both Yes/No sit on top with their own picking, so the
            // backdrop just blocks accidental clicks reaching the
            // editor panels behind it.
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
            // ZIndex 110 puts this above `exit_modal` (100) just in
            // case both flags somehow get set in the same frame —
            // the most-recent destructive prompt wins focus.
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

                    // Button row. No first (left, highlighted) so the
                    // pointer's natural resting position after the
                    // user's reflex glance lands on the safe option.
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
                // Drain every template — visual entities first so a
                // half-cleared frame can't reach `sync_rows` while
                // `session.templates` is empty and re-spawn rows for
                // already-despawned entities.
                let removed: Vec<OrganismTemplate> = session.templates.drain(..).collect();
                for t in &removed {
                    commands.entity(t.entity).despawn();
                }
                session.active_id = None;
                session.dirty     = true;

                // Despawn every live organism — same path as
                // right-click delete. Recursive despawn drops body
                // parts; RemovedComponents observers handle brain
                // slot reclaim, statistics counters, etc.
                for e in &organisms_q {
                    commands.entity(e).despawn();
                }

                // Close the modal. Note: NO undo entry here — wiping
                // wild organisms is not part of the editor's
                // reversible action model (the templates path is,
                // but mixing reversible + irreversible into one
                // entry would be confusing), so we keep the whole
                // op irreversible for clarity.
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
