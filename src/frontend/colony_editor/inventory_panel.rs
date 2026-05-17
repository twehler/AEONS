// Right-side inventory panel.
//
// Lists every `OrganismTemplate` the user has created. Click a row to
// make that template the active placement target — right-click in the
// viewport will move it. The active row is rendered in a brighter
// colour. A "Save Colony" button at the top opens an rfd save dialog
// and writes a v003 .colony file.

use bevy::prelude::*;

use crate::colony_editor::session::EditorSession;
use crate::colony_editor::layout::PANEL_BG_COLOR;
use crate::colony_editor::template::{form_label, intel_label, sym_label, OrganismTemplate};
use crate::colony_editor::creation_panel::BOTTOM_PANEL_HEIGHT_PX;


// ── Tunables ─────────────────────────────────────────────────────────────────

/// Logical-pixel width of the right-side inventory panel. `pub` so
/// `camera.rs` can rect-test the cursor against it when deciding
/// whether an LMB press should be a viewport click.
pub const PANEL_WIDTH_PX: f32 = 260.0;
const PADDING_PX:      f32 = 10.0;
const ROW_HEIGHT_PX:   f32 = 72.0;
const ROW_GAP_PX:      f32 = 4.0;

const ROW_BG_INACTIVE:  Color = Color::srgb(0.20, 0.20, 0.22);
const ROW_BG_HOVER:     Color = Color::srgb(0.28, 0.28, 0.30);
const ROW_BG_ACTIVE:    Color = Color::srgb(0.30, 0.45, 0.65);

const SAVE_BTN_HEIGHT:  f32 = 36.0;
const SAVE_BTN_COLOR:   Color = Color::srgb(0.20, 0.40, 0.65);
const SAVE_BTN_HOVER:   Color = Color::srgb(0.28, 0.50, 0.75);

const BACK_BTN_HEIGHT:  f32 = 30.0;
const BACK_BTN_COLOR:   Color = Color::srgb(0.30, 0.30, 0.32);
const BACK_BTN_HOVER:   Color = Color::srgb(0.40, 0.40, 0.42);

// Clear All — destructive action, dim red.
const CLEAR_BTN_HEIGHT: f32   = 32.0;
const CLEAR_BTN_COLOR:  Color = Color::srgb(0.55, 0.20, 0.20);
const CLEAR_BTN_HOVER:  Color = Color::srgb(0.70, 0.26, 0.26);


// ── Marker components ───────────────────────────────────────────────────────

#[derive(Component)]
pub struct InventoryPanel;

#[derive(Component)]
pub struct InventoryList;

#[derive(Component, Clone, Copy)]
struct InventoryRow { id: u32 }

#[derive(Component)]
struct SaveButton;

#[derive(Component)]
struct ReturnButton;

#[derive(Component)]
struct ClearAllButton;


// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct InventoryPanelPlugin;

impl Plugin for InventoryPanelPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (
            sync_rows,
            handle_row_clicks,
            handle_save_button,
            handle_return_button,
            handle_clear_all_button,
        ));
    }
}


// ── Spawning ────────────────────────────────────────────────────────────────

pub fn spawn(parent: &mut ChildSpawnerCommands) {
    spawn_with_offset(parent, 0.0);
}

/// Variant that reserves `top_offset_px` for a mode-switcher bar at
/// the top of the screen — used by the merged-mode editor overlay so
/// the panel doesn't extend up under the bar. Standalone editor uses
/// `spawn` (passes 0).
pub fn spawn_with_offset(parent: &mut ChildSpawnerCommands, top_offset_px: f32) {
    parent
        .spawn((
            InventoryPanel,
            Node {
                position_type: PositionType::Absolute,
                top:    Val::Px(top_offset_px),
                right:  Val::Px(0.0),
                bottom: Val::Px(BOTTOM_PANEL_HEIGHT_PX),
                width:  Val::Px(PANEL_WIDTH_PX),
                flex_direction: FlexDirection::Column,
                padding:        UiRect::all(Val::Px(PADDING_PX)),
                ..default()
            },
            BackgroundColor(PANEL_BG_COLOR),
        ))
        .with_children(|panel| {
            // Title.
            panel.spawn((
                Text::new("Created Organisms"),
                TextFont { font_size: 16.0, ..default() },
                TextColor(Color::srgb(0.92, 0.92, 0.92)),
                Node { margin: UiRect::bottom(Val::Px(8.0)), ..default() },
                Pickable::IGNORE,
            ));

            // Save button.
            panel
                .spawn((
                    SaveButton,
                    Button,
                    Node {
                        width:  Val::Percent(100.0),
                        height: Val::Px(SAVE_BTN_HEIGHT),
                        margin: UiRect::bottom(Val::Px(6.0)),
                        align_items:     AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        ..default()
                    },
                    BackgroundColor(SAVE_BTN_COLOR),
                ))
                .with_children(|b| {
                    b.spawn((
                        Text::new("Save Colony…"),
                        TextFont { font_size: 14.0, ..default() },
                        TextColor(Color::WHITE),
                        Pickable::IGNORE,
                    ));
                });

            // Return-to-Menu button — sits directly under Save so
            // the visual flow reads "save your work, then go back".
            panel
                .spawn((
                    ReturnButton,
                    Button,
                    Node {
                        width:  Val::Percent(100.0),
                        height: Val::Px(BACK_BTN_HEIGHT),
                        margin: UiRect::bottom(Val::Px(10.0)),
                        align_items:     AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        ..default()
                    },
                    BackgroundColor(BACK_BTN_COLOR),
                ))
                .with_children(|b| {
                    b.spawn((
                        Text::new("Return to Menu"),
                        TextFont { font_size: 13.0, ..default() },
                        TextColor(Color::srgb(0.92, 0.92, 0.92)),
                        Pickable::IGNORE,
                    ));
                });

            // Scrollable list. Same `Overflow::scroll_y()` pattern as
            // the simulation's individuum navigator.
            panel.spawn((
                InventoryList,
                Node {
                    flex_grow: 1.0,
                    flex_basis: Val::Px(0.0),
                    min_height: Val::Px(0.0),
                    flex_direction: FlexDirection::Column,
                    overflow: Overflow::scroll_y(),
                    ..default()
                },
                ScrollPosition::default(),
            ));

            // ── Clear All — sits at the very bottom of the panel
            //    below the scroll list. Destructive (dim red) so it
            //    reads as different from the calm-blue Save above.
            //    Wipes every template; reversible with Ctrl+Z.
            panel
                .spawn((
                    ClearAllButton,
                    Button,
                    Node {
                        width:           Val::Percent(100.0),
                        height:          Val::Px(CLEAR_BTN_HEIGHT),
                        margin:          UiRect::top(Val::Px(6.0)),
                        align_items:     AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        ..default()
                    },
                    BackgroundColor(CLEAR_BTN_COLOR),
                ))
                .with_children(|b| {
                    b.spawn((
                        Text::new("Clear All"),
                        TextFont { font_size: 13.0, ..default() },
                        TextColor(Color::WHITE),
                        Pickable::IGNORE,
                    ));
                });
        });
}


// ── Systems ──────────────────────────────────────────────────────────────────

fn sync_rows(
    mut commands: Commands,
    list_q:       Query<Entity, With<InventoryList>>,
    mut rows_q:   Query<(Entity, &InventoryRow, &mut BackgroundColor)>,
    session:      Res<EditorSession>,
) {
    let Ok(list) = list_q.single() else { return };

    let known_ids:    Vec<u32> = rows_q.iter().map(|(_, row, _)| row.id).collect();
    let template_ids: Vec<u32> = session.templates.iter().map(|t| t.id).collect();

    // Despawn rows whose template is gone.
    for (entity, row, _) in rows_q.iter() {
        if !template_ids.contains(&row.id) {
            commands.entity(entity).despawn();
        }
    }

    // Spawn rows for new templates (preserving session order).
    for tpl in session.templates.iter() {
        if !known_ids.contains(&tpl.id) {
            spawn_row(&mut commands, list, tpl);
        }
    }

    // Recolour rows according to which is active. Only writes when
    // the result actually differs from the current colour, so this
    // is cheap to run every frame.
    let active = session.active_id;
    for (_, row, mut bg) in &mut rows_q {
        let target = if Some(row.id) == active { ROW_BG_ACTIVE } else { ROW_BG_INACTIVE };
        if bg.0 != target { *bg = BackgroundColor(target); }
    }
}

fn spawn_row(commands: &mut Commands, list: Entity, tpl: &OrganismTemplate) {
    let row = commands
        .spawn((
            InventoryRow { id: tpl.id },
            Button,
            Node {
                width:  Val::Percent(100.0),
                height: Val::Px(ROW_HEIGHT_PX),
                margin: UiRect::bottom(Val::Px(ROW_GAP_PX)),
                padding: UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                flex_direction: FlexDirection::Column,
                justify_content: JustifyContent::Center,
                align_items:    AlignItems::FlexStart,
                flex_shrink:    0.0,
                ..default()
            },
            BackgroundColor(ROW_BG_INACTIVE),
        ))
        .with_children(|btn| {
            btn.spawn((
                Text::new(tpl.display_name()),
                TextFont { font_size: 14.0, ..default() },
                TextColor(Color::WHITE),
                Pickable::IGNORE,
            ));
            btn.spawn((
                Text::new(format!(
                    "{} · {}",
                    intel_label(tpl.intelligence),
                    sym_label(tpl.symmetry),
                )),
                TextFont { font_size: 11.0, ..default() },
                TextColor(Color::srgb(0.75, 0.75, 0.75)),
                Pickable::IGNORE,
            ));
            btn.spawn((
                Text::new(form_label(tpl.form).to_string()),
                TextFont { font_size: 10.0, ..default() },
                TextColor(Color::srgb(0.62, 0.62, 0.62)),
                Pickable::IGNORE,
            ));
            btn.spawn((
                Text::new(format!(
                    "@ ({:.0}, {:.0}, {:.0})",
                    tpl.position.x, tpl.position.y, tpl.position.z,
                )),
                TextFont { font_size: 10.0, ..default() },
                TextColor(Color::srgb(0.55, 0.55, 0.55)),
                Pickable::IGNORE,
            ));
        })
        .id();
    commands.entity(list).add_child(row);
}

fn handle_row_clicks(
    mut interactions: Query<(&Interaction, &InventoryRow, &mut BackgroundColor), Changed<Interaction>>,
    mut session:      ResMut<EditorSession>,
) {
    for (interaction, row, mut bg) in &mut interactions {
        let is_active = Some(row.id) == session.active_id;
        match *interaction {
            Interaction::Pressed => {
                session.active_id = Some(row.id);
                *bg = BackgroundColor(ROW_BG_ACTIVE);
            }
            Interaction::Hovered => {
                if !is_active { *bg = BackgroundColor(ROW_BG_HOVER); }
            }
            Interaction::None => {
                *bg = BackgroundColor(if is_active { ROW_BG_ACTIVE } else { ROW_BG_INACTIVE });
            }
        }
    }
}

fn handle_save_button(
    mut interactions: Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<SaveButton>)>,
    mut session:      ResMut<EditorSession>,
) {
    for (interaction, mut bg) in &mut interactions {
        match *interaction {
            Interaction::Pressed => {
                session.save_requested = true;
                *bg = BackgroundColor(SAVE_BTN_HOVER);
            }
            Interaction::Hovered => *bg = BackgroundColor(SAVE_BTN_HOVER),
            Interaction::None    => *bg = BackgroundColor(SAVE_BTN_COLOR),
        }
    }
}

fn handle_return_button(
    mut interactions: Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<ReturnButton>)>,
    mut session:      ResMut<EditorSession>,
) {
    for (interaction, mut bg) in &mut interactions {
        match *interaction {
            Interaction::Pressed => {
                // Hand off to the exit-modal dispatcher; it decides
                // whether to show the unsaved-work warning or exit
                // immediately based on `session.dirty`.
                session.exit_requested = true;
                *bg = BackgroundColor(BACK_BTN_HOVER);
            }
            Interaction::Hovered => *bg = BackgroundColor(BACK_BTN_HOVER),
            Interaction::None    => *bg = BackgroundColor(BACK_BTN_COLOR),
        }
    }
}

fn handle_clear_all_button(
    mut interactions: Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<ClearAllButton>)>,
    mut session:      ResMut<EditorSession>,
) {
    for (interaction, mut bg) in &mut interactions {
        match *interaction {
            Interaction::Pressed => {
                *bg = BackgroundColor(CLEAR_BTN_HOVER);
                // Hand off to the confirmation modal — the actual
                // template + organism wipe lives in
                // `clear_modal.rs::handle_clear_modal_buttons`.
                // Idempotent: if the modal is already up, this is a
                // no-op.
                session.show_clear_modal = true;
            }
            Interaction::Hovered => *bg = BackgroundColor(CLEAR_BTN_HOVER),
            Interaction::None    => *bg = BackgroundColor(CLEAR_BTN_COLOR),
        }
    }
}
