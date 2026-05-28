// Species editor — Body-part index panel (right side).
//
// A vertical list of the species' body parts. Index 0 is the base body;
// later entries are appendages added with "Begin New Body-Part". The
// user picks which part new cells are placed on by clicking its row;
// clicking the already-active row a second time enters rename mode
// (type, Enter commits, Escape cancels).
//
// "Begin New Body-Part" appends a fresh appendage, makes it active, and
// auto-selects the blue Placeholder cell type so the new part is built
// out of debug-blue cells (its first cell attaches to the base body's
// growth frontier — see `placement::update_preview_cell`).

use bevy::input::keyboard::KeyboardInput;
use bevy::prelude::*;

use crate::cell::CellType;
use crate::frontend::PANEL_BG_COLOR;
use crate::simulation_settings::WindowMode;

use super::session::{EditorBodyPart, SpeciesSession};
use super::TOP_PANEL_HEIGHT_PX;


// ── Tunables ─────────────────────────────────────────────────────────────────

pub const BODY_PART_PANEL_WIDTH_PX: f32 = 240.0;

const ROW_HEIGHT_PX:   f32 = 30.0;
const RENAME_MAX_LEN:  usize = 28;

const ROW_BG_IDLE:     Color = Color::srgb(0.18, 0.18, 0.18);
const ROW_BG_ACTIVE:   Color = Color::srgb(0.20, 0.40, 0.62);
const ROW_BG_HOVER:    Color = Color::srgb(0.30, 0.30, 0.30);
const ROW_BG_RENAME:   Color = Color::srgb(0.45, 0.40, 0.12);

const BEGIN_BG:        Color = Color::srgb(0.20, 0.45, 0.70);
const BEGIN_BG_HOVER:  Color = Color::srgb(0.30, 0.55, 0.80);
const BEGIN_BG_DISABLED: Color = Color::srgb(0.18, 0.18, 0.18);


// ── Markers ──────────────────────────────────────────────────────────────────

#[derive(Component)]
pub struct BodyPartPanel;

/// Column container the per-part rows are parented under.
#[derive(Component)]
pub struct BodyPartList;

#[derive(Component)]
pub struct BeginNewBodyPartButton;

/// One row in the index, carrying the body-part index it represents.
#[derive(Component, Clone, Copy)]
pub struct BodyPartRow(pub usize);

/// Text child of a row, so the label-sync system finds it directly.
#[derive(Component, Clone, Copy)]
pub struct BodyPartRowLabel(pub usize);


// ── Spawn ────────────────────────────────────────────────────────────────────

pub fn spawn_body_part_panel(parent: &mut ChildSpawnerCommands, top_offset_px: f32) {
    parent
        .spawn((
            BodyPartPanel,
            super::SpeciesEditorPanel,
            Node {
                position_type: PositionType::Absolute,
                top:    Val::Px(top_offset_px + TOP_PANEL_HEIGHT_PX),
                right:  Val::Px(0.0),
                bottom: Val::Px(super::BOTTOM_PANEL_HEIGHT_PX),
                width:  Val::Px(BODY_PART_PANEL_WIDTH_PX),
                padding: UiRect::all(Val::Px(8.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(6.0),
                display: Display::None,   // shown only in SpeciesEditor mode
                ..default()
            },
            BackgroundColor(PANEL_BG_COLOR),
        ))
        .with_children(|panel| {
            // Title.
            panel.spawn((
                Text::new("Body part index"),
                TextFont { font_size: 15.0, ..default() },
                TextColor(Color::WHITE),
                Pickable::IGNORE,
            ));

            // "Begin New Body-Part" button.
            panel
                .spawn((
                    BeginNewBodyPartButton,
                    Button,
                    Node {
                        width:  Val::Percent(100.0),
                        height: Val::Px(32.0),
                        align_items:     AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        ..default()
                    },
                    BackgroundColor(BEGIN_BG),
                ))
                .with_children(|b| {
                    b.spawn((
                        Text::new("Begin New Body-Part"),
                        TextFont { font_size: 13.0, ..default() },
                        TextColor(Color::WHITE),
                        Pickable::IGNORE,
                    ));
                });

            // Scrollable list of rows.
            panel.spawn((
                BodyPartList,
                Node {
                    width: Val::Percent(100.0),
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(4.0),
                    overflow: Overflow::scroll_y(),
                    ..default()
                },
            ));
        });
}


// ── "Begin New Body-Part" ─────────────────────────────────────────────────────

pub fn handle_begin_new_body_part(
    mode:             Res<WindowMode>,
    mut interactions: Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<BeginNewBodyPartButton>)>,
    mut session:      ResMut<SpeciesSession>,
) {
    if *mode != WindowMode::SpeciesEditor { return; }

    for (interaction, mut bg) in &mut interactions {
        // Disabled until the base body exists.
        let enabled = session.first_cell_spawned;
        match *interaction {
            Interaction::Pressed if enabled => {
                let n = session.body_parts.len();
                session.body_parts.push(EditorBodyPart {
                    name: format!("Body Part {n}"),
                    ocg:  Vec::new(),
                });
                session.active_body_part = n;
                session.renaming_body_part = None;
                // New parts are sketched in the blue Placeholder type.
                session.selected_cell_type = Some(CellType::Placeholder);
                session.dirty = true;
                *bg = BackgroundColor(BEGIN_BG_HOVER);
            }
            Interaction::Pressed => *bg = BackgroundColor(BEGIN_BG_DISABLED),
            Interaction::Hovered if enabled => *bg = BackgroundColor(BEGIN_BG_HOVER),
            _ => *bg = BackgroundColor(if enabled { BEGIN_BG } else { BEGIN_BG_DISABLED }),
        }
    }
}


// ── Row list lifecycle ────────────────────────────────────────────────────────

/// Rebuild the row entities whenever the number of body parts changes.
/// Per-frame label text + highlight are handled by `sync_body_part_rows`,
/// so this only fires on add/remove.
pub fn manage_body_part_list(
    session:    Res<SpeciesSession>,
    mut commands: Commands,
    list_q:     Query<Entity, With<BodyPartList>>,
    rows_q:     Query<(Entity, &BodyPartRow)>,
) {
    let Ok(list) = list_q.single() else { return };
    let want = session.body_parts.len();
    let have = rows_q.iter().count();
    if want == have { return; }

    for (e, _) in &rows_q { commands.entity(e).despawn(); }

    commands.entity(list).with_children(|list| {
        for (i, part) in session.body_parts.iter().enumerate() {
            list.spawn((
                BodyPartRow(i),
                Button,
                Node {
                    width:  Val::Percent(100.0),
                    height: Val::Px(ROW_HEIGHT_PX),
                    align_items:     AlignItems::Center,
                    justify_content: JustifyContent::FlexStart,
                    padding: UiRect::horizontal(Val::Px(8.0)),
                    ..default()
                },
                BackgroundColor(ROW_BG_IDLE),
            ))
            .with_children(|row| {
                row.spawn((
                    BodyPartRowLabel(i),
                    Text::new(part.name.clone()),
                    TextFont { font_size: 13.0, ..default() },
                    TextColor(Color::WHITE),
                    Pickable::IGNORE,
                ));
            });
        }
    });
}


// ── Row clicks (select / begin rename) ────────────────────────────────────────

pub fn handle_body_part_row_clicks(
    mode:             Res<WindowMode>,
    mut interactions: Query<(&Interaction, &BodyPartRow), Changed<Interaction>>,
    mut session:      ResMut<SpeciesSession>,
) {
    if *mode != WindowMode::SpeciesEditor { return; }

    for (interaction, row) in &mut interactions {
        if !matches!(*interaction, Interaction::Pressed) { continue; }
        let i = row.0;
        if i >= session.body_parts.len() { continue; }

        if session.active_body_part == i && session.renaming_body_part != Some(i) {
            // Second click on the already-active row → start renaming.
            session.renaming_body_part = Some(i);
            session.rename_buffer = session.body_parts[i].name.clone();
        } else {
            // Select this part as the active placement target.
            session.active_body_part = i;
            session.renaming_body_part = None;
        }
    }
}


// ── Rename keyboard input ──────────────────────────────────────────────────────

pub fn handle_rename_input(
    mode:        Res<WindowMode>,
    mut keyboard: MessageReader<KeyboardInput>,
    mut session: ResMut<SpeciesSession>,
) {
    let renaming = if *mode == WindowMode::SpeciesEditor { session.renaming_body_part } else { None };
    let Some(i) = renaming else {
        for _ in keyboard.read() {}   // drain
        return;
    };
    if i >= session.body_parts.len() {
        session.renaming_body_part = None;
        for _ in keyboard.read() {}
        return;
    }

    for ev in keyboard.read() {
        if !ev.state.is_pressed() { continue; }
        match ev.key_code {
            KeyCode::Enter | KeyCode::NumpadEnter => {
                let buf = session.rename_buffer.trim().to_string();
                if !buf.is_empty() {
                    session.body_parts[i].name = buf;
                    session.dirty = true;
                }
                session.renaming_body_part = None;
                session.rename_buffer.clear();
            }
            KeyCode::Escape => {
                session.renaming_body_part = None;
                session.rename_buffer.clear();
            }
            KeyCode::Backspace => { session.rename_buffer.pop(); }
            _ => {
                if let Some(text) = ev.text.as_ref() {
                    for c in text.chars() {
                        if session.rename_buffer.len() >= RENAME_MAX_LEN { break; }
                        // Printable, non-control characters only.
                        if !c.is_control() {
                            session.rename_buffer.push(c);
                        }
                    }
                }
            }
        }
    }
}


// ── Row label + highlight sync ─────────────────────────────────────────────────

pub fn sync_body_part_rows(
    session:    Res<SpeciesSession>,
    mut rows:   Query<(&BodyPartRow, &Interaction, &mut BackgroundColor)>,
    mut labels: Query<(&BodyPartRowLabel, &mut Text)>,
) {
    for (label, mut text) in &mut labels {
        let i = label.0;
        let Some(part) = session.body_parts.get(i) else { continue };
        let new = if session.renaming_body_part == Some(i) {
            format!("{}_", session.rename_buffer)
        } else {
            part.name.clone()
        };
        if text.0 != new { text.0 = new; }
    }

    for (row, interaction, mut bg) in &mut rows {
        let i = row.0;
        let target = if session.renaming_body_part == Some(i) {
            ROW_BG_RENAME
        } else if session.active_body_part == i {
            ROW_BG_ACTIVE
        } else if matches!(*interaction, Interaction::Hovered | Interaction::Pressed) {
            ROW_BG_HOVER
        } else {
            ROW_BG_IDLE
        };
        *bg = BackgroundColor(target);
    }
}
