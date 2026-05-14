// Left-side "Tool Panel".
//
// Currently hosts one feature — **Bulk-Add organism at random
// locations** — but is laid out to grow new tools in the future
// (each tool gets its own labelled section in the column).
//
// Bulk-add reads the current draft (Metabolism / Intelligence /
// Symmetry / Form from the bottom creation panel) and spawns N
// `OrganismTemplate`s at uniformly-random surface points within the
// map's XZ bounds. The spawn itself reuses the same helper as
// left-click placement, so each newly-added organism is rendered
// with the proper rhombic-dodecahedron mesh and dropped into the
// session's template list (and thus into the inventory panel).

use bevy::prelude::*;
use bevy::input::keyboard::KeyboardInput;
use rand::prelude::*;

use crate::world_geometry::{HeightmapSampler, MapSize};
use crate::colony_editor::session::EditorSession;
use crate::colony_editor::layout::PANEL_BG_COLOR;
use crate::colony_editor::creation_panel::BOTTOM_PANEL_HEIGHT_PX;
use crate::colony_editor::placement::spawn_template_at;
use crate::colony_editor::undo::{EditorAction, UndoStack};


// ── Tunables ─────────────────────────────────────────────────────────────────

/// Logical-pixel width of the left-side tool panel. `pub` so
/// `camera.rs` can rect-test the cursor against it when deciding
/// whether an LMB press should be a viewport click.
pub const TOOL_PANEL_WIDTH_PX: f32 = 220.0;

const PADDING_PX:           f32   = 10.0;
const TITLE_FONT_PX:        f32   = 16.0;
const SECTION_HEADING_PX:   f32   = 13.0;
const LABEL_FONT_PX:        f32   = 11.0;

const COUNT_FIELD_HEIGHT:   f32   = 30.0;
/// Bound on the editable buffer length. 4 digits ⇒ up to 9999.
const COUNT_BUFFER_MAX_LEN: usize = 4;
const COUNT_MIN:            u32   = 1;
/// Bulk-add cap. Bounded conservatively so a mistyped 4-digit count
/// can't spawn tens of thousands of organisms and freeze the editor.
const COUNT_MAX:            u32   = 500;

const COUNT_BG_IDLE:        Color = Color::srgb(0.20, 0.20, 0.22);
const COUNT_BG_FOCUSED:     Color = Color::srgb(0.32, 0.30, 0.18);

const BULK_BTN_HEIGHT:      f32   = 38.0;
const BULK_BTN_COLOR:       Color = Color::srgb(0.25, 0.45, 0.25);
const BULK_BTN_HOVER:       Color = Color::srgb(0.35, 0.55, 0.35);


// ── Components / Resources ───────────────────────────────────────────────────

#[derive(Component)]
pub struct ToolPanel;

#[derive(Component)]
struct BulkAddCountField;

#[derive(Component)]
struct BulkAddCountText;

#[derive(Component)]
struct BulkAddButton;

/// Edit + commit state for the count field. Mirrors the
/// `TimeSpeedEditState` pattern in `statistics_panel.rs`: a focused
/// flag, a live edit buffer (only meaningful while focused), and a
/// committed integer value the Bulk-Add button consumes.
#[derive(Resource)]
pub struct ToolPanelState {
    pub count_buffer:    String,
    pub count_focused:   bool,
    pub count_committed: u32,
}

impl Default for ToolPanelState {
    fn default() -> Self {
        Self {
            count_buffer:    String::new(),
            count_focused:   false,
            count_committed: 10,
        }
    }
}


// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct ToolPanelPlugin;

impl Plugin for ToolPanelPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<ToolPanelState>()
            .add_systems(Update, (
                handle_count_input,
                update_count_text,
                handle_bulk_add_button,
            ));
    }
}


// ── Spawning ────────────────────────────────────────────────────────────────

/// Append the tool panel as a child of the layout root. Called from
/// `layout.rs::setup_ui`.
pub fn spawn(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((
            ToolPanel,
            Node {
                position_type: PositionType::Absolute,
                top:    Val::Px(0.0),
                left:   Val::Px(0.0),
                bottom: Val::Px(BOTTOM_PANEL_HEIGHT_PX),
                width:  Val::Px(TOOL_PANEL_WIDTH_PX),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(PADDING_PX)),
                ..default()
            },
            BackgroundColor(PANEL_BG_COLOR),
        ))
        .with_children(|panel| {
            // Title.
            panel.spawn((
                Text::new("Tool Panel"),
                TextFont { font_size: TITLE_FONT_PX, ..default() },
                TextColor(Color::srgb(0.92, 0.92, 0.92)),
                Node { margin: UiRect::bottom(Val::Px(12.0)), ..default() },
                Pickable::IGNORE,
            ));

            // Top spacer — pushes the Bulk-Add section toward the
            // vertical centre of the panel.
            panel.spawn(Node { flex_grow: 1.0, ..default() });

            // ── Bulk-Add section ────────────────────────────────────
            panel.spawn((
                Text::new("Bulk-Add organism at\nrandom locations"),
                TextFont { font_size: SECTION_HEADING_PX, ..default() },
                TextColor(Color::srgb(0.85, 0.85, 0.85)),
                Node { margin: UiRect::bottom(Val::Px(8.0)), ..default() },
                Pickable::IGNORE,
            ));

            // Count label + editable field.
            panel.spawn((
                Text::new("Count"),
                TextFont { font_size: LABEL_FONT_PX, ..default() },
                TextColor(Color::srgb(0.70, 0.70, 0.70)),
                Pickable::IGNORE,
            ));
            panel
                .spawn((
                    BulkAddCountField,
                    Button,
                    Node {
                        width:           Val::Percent(100.0),
                        height:          Val::Px(COUNT_FIELD_HEIGHT),
                        padding:         UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                        align_items:     AlignItems::Center,
                        justify_content: JustifyContent::FlexStart,
                        margin:          UiRect {
                            top:    Val::Px(4.0),
                            bottom: Val::Px(10.0),
                            ..default()
                        },
                        ..default()
                    },
                    BackgroundColor(COUNT_BG_IDLE),
                ))
                .with_children(|btn| {
                    btn.spawn((
                        BulkAddCountText,
                        Text::new("10"),
                        TextFont { font_size: 14.0, ..default() },
                        TextColor(Color::WHITE),
                        Pickable::IGNORE,
                    ));
                });

            // Bulk-Add button.
            panel
                .spawn((
                    BulkAddButton,
                    Button,
                    Node {
                        width:           Val::Percent(100.0),
                        height:          Val::Px(BULK_BTN_HEIGHT),
                        align_items:     AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        ..default()
                    },
                    BackgroundColor(BULK_BTN_COLOR),
                ))
                .with_children(|btn| {
                    btn.spawn((
                        Text::new("Add organisms"),
                        TextFont { font_size: 14.0, ..default() },
                        TextColor(Color::WHITE),
                        Pickable::IGNORE,
                    ));
                });

            // Bottom spacer — keeps the section vertically centred
            // even as new tools land in the panel later.
            panel.spawn(Node { flex_grow: 1.0, ..default() });
        });
}


// ── Systems ──────────────────────────────────────────────────────────────────

/// Click / keyboard router for the count-edit field. Same shape as
/// `statistics_panel::handle_time_speed_input` — click-to-focus,
/// click-elsewhere-or-Enter-to-commit, Escape-to-cancel, backspace,
/// digit-only typing.
fn handle_count_input(
    mouse:           Res<ButtonInput<MouseButton>>,
    mut keyboard:    MessageReader<KeyboardInput>,
    interaction_q:   Query<&Interaction, With<BulkAddCountField>>,
    mut state:       ResMut<ToolPanelState>,
) {
    let click_on_field = mouse.just_pressed(MouseButton::Left)
        && interaction_q.iter().any(|i| matches!(i, Interaction::Pressed));
    let click_outside  = mouse.just_pressed(MouseButton::Left) && !click_on_field;

    if click_on_field && !state.count_focused {
        state.count_focused = true;
        // `to_string` then assign avoids the simultaneous borrow that
        // `write!(state.count_buffer, "{}", state.count_committed)`
        // would produce on the single `state` `Resource`.
        state.count_buffer = state.count_committed.to_string();
    }

    if click_outside && state.count_focused {
        commit_count(&mut state);
    }

    if !state.count_focused {
        // Drain so events don't accumulate.
        for _ in keyboard.read() {}
        return;
    }

    for ev in keyboard.read() {
        if !ev.state.is_pressed() { continue; }
        match ev.key_code {
            KeyCode::Enter | KeyCode::NumpadEnter => {
                commit_count(&mut state);
            }
            KeyCode::Escape => {
                state.count_focused = false;
                state.count_buffer.clear();
            }
            KeyCode::Backspace => {
                state.count_buffer.pop();
            }
            _ => {
                if let Some(text) = ev.text.as_ref() {
                    for c in text.chars() {
                        if state.count_buffer.len() >= COUNT_BUFFER_MAX_LEN { break; }
                        if c.is_ascii_digit() {
                            state.count_buffer.push(c);
                        }
                    }
                }
            }
        }
    }
}

fn commit_count(state: &mut ToolPanelState) {
    if let Ok(v) = state.count_buffer.parse::<u32>() {
        state.count_committed = v.clamp(COUNT_MIN, COUNT_MAX);
    }
    state.count_focused = false;
    state.count_buffer.clear();
}

/// Sync the count field's text + background colour with the current
/// state. Early-returns when the resource hasn't changed so this is
/// effectively free in steady state.
fn update_count_text(
    state:      Res<ToolPanelState>,
    mut text_q: Query<&mut Text, With<BulkAddCountText>>,
    mut bg_q:   Query<&mut BackgroundColor, With<BulkAddCountField>>,
) {
    if !state.is_changed() { return; }

    let display = if state.count_focused {
        format!("{}_", state.count_buffer)
    } else {
        format!("{}", state.count_committed)
    };
    for mut text in &mut text_q {
        text.0 = display.clone();
    }

    let bg = if state.count_focused { COUNT_BG_FOCUSED } else { COUNT_BG_IDLE };
    for mut b in &mut bg_q {
        if b.0 != bg { *b = BackgroundColor(bg); }
    }
}

/// On click of the Bulk-Add button, spawn `count_committed` template
/// organisms at uniformly-random surface points within the map's
/// XZ rect. Each spawn uses the editor's current draft, so the
/// bulk-added cohort shares whatever metabolism / intelligence /
/// symmetry / form is currently selected in the bottom creation
/// panel.
///
/// Silently no-ops if the heightmap resource isn't loaded yet
/// (world still streaming). All spawned organisms count as
/// unsaved work, so `session.dirty` is set by `spawn_template_at`.
fn handle_bulk_add_button(
    mut interactions: Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<BulkAddButton>)>,
    state:            Res<ToolPanelState>,
    mut session:      ResMut<EditorSession>,
    heightmap:        Option<Res<HeightmapSampler>>,
    map_size:         Res<MapSize>,
    mut commands:     Commands,
    mut meshes:       ResMut<Assets<Mesh>>,
    mut materials:    ResMut<Assets<StandardMaterial>>,
    mut undo_stack:   ResMut<UndoStack>,
) {
    for (interaction, mut bg) in &mut interactions {
        match *interaction {
            Interaction::Pressed => {
                *bg = BackgroundColor(BULK_BTN_HOVER);
                let Some(heightmap) = heightmap.as_deref() else {
                    warn!("bulk-add: heightmap not loaded yet — try again in a moment");
                    continue;
                };
                let mut rng = rand::rng();
                let mut new_ids = Vec::with_capacity(state.count_committed as usize);
                for _ in 0..state.count_committed {
                    let x = rng.random_range(0.0_f32..map_size.x);
                    let z = rng.random_range(0.0_f32..map_size.z);
                    let y = heightmap.height_at(x, z) + 0.5;
                    let id = spawn_template_at(
                        Vec3::new(x, y, z),
                        &mut session,
                        &mut commands,
                        &mut meshes,
                        &mut materials,
                    );
                    new_ids.push(id);
                }
                if !new_ids.is_empty() {
                    // One undo step rolls back the entire bulk batch.
                    undo_stack.push(EditorAction::Created(new_ids));
                }
                info!("bulk-added {} organism(s)", state.count_committed);
            }
            Interaction::Hovered => *bg = BackgroundColor(BULK_BTN_HOVER),
            Interaction::None    => *bg = BackgroundColor(BULK_BTN_COLOR),
        }
    }
}
