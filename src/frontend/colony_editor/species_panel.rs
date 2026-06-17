// Left-side "Species Navigator" panel. Two sections:
//   * Species list — "Load Species" stashes an rfd-chosen `.species`
//     path on the session (consumed by `mod.rs` next tick); the
//     scrollable list selects `selected_species_id`.
//   * Bulk-Spawn — count field + button; spawns N of the selected
//     species at random heightmap points. Disabled with no selection.
//
// Per-organism RL hyperparameter genes are sampled fresh by the brain
// pool from `L1_*_RANGE` on every spawn, not specified here.

use bevy::prelude::*;
use bevy::input::keyboard::KeyboardInput;
use rand::prelude::*;

use crate::world_geometry::{HeightmapSampler, MapSize};
use crate::colony_editor::session::EditorSession;
use crate::colony_editor::layout::PANEL_BG_COLOR;
use crate::colony_editor::creation_panel::BOTTOM_PANEL_HEIGHT_PX;
use crate::colony_editor::placement::spawn_species_template_at;
use crate::colony_editor::undo::{EditorAction, UndoStack};


// ── Tunables ─────────────────────────────────────────────────────────────────

/// Logical-pixel width of the left-side panel.
pub const SPECIES_PANEL_WIDTH_PX: f32 = 240.0;
/// Backwards-compat alias for the camera click-rect check.
pub const TOOL_PANEL_WIDTH_PX: f32 = SPECIES_PANEL_WIDTH_PX;

const PADDING_PX:         f32 = 10.0;
const TITLE_FONT_PX:      f32 = 16.0;
const SECTION_HEADING_PX: f32 = 13.0;
const LABEL_FONT_PX:      f32 = 11.0;


const SPECIES_ROW_HEIGHT: f32 = 30.0;
const SPECIES_ROW_GAP:    f32 = 3.0;
const SPECIES_ROW_IDLE:        Color = Color::srgb(0.20, 0.20, 0.22);
const SPECIES_ROW_HOVER:       Color = Color::srgb(0.30, 0.30, 0.34);
/// Highlight for the currently-selected species row.
const SPECIES_ROW_SELECTED:    Color = Color::srgb(0.20, 0.45, 0.85);

const COUNT_FIELD_HEIGHT:   f32   = 30.0;
const COUNT_BUFFER_MAX_LEN: usize = 4;
const COUNT_MIN:            u32   = 1;
const COUNT_MAX:            u32   = 500;

const COUNT_BG_IDLE:    Color = Color::srgb(0.20, 0.20, 0.22);
const COUNT_BG_FOCUSED: Color = Color::srgb(0.32, 0.30, 0.18);

const BULK_BTN_HEIGHT:    f32 = 38.0;
const BULK_BTN_COLOR:     Color = Color::srgb(0.25, 0.45, 0.25);
const BULK_BTN_HOVER:     Color = Color::srgb(0.35, 0.55, 0.35);
const BULK_BTN_DISABLED:  Color = Color::srgb(0.18, 0.18, 0.18);


// ── Components / Resources ───────────────────────────────────────────────────

#[derive(Component)]
pub struct SpeciesPanel;

#[derive(Component)]
struct SpeciesListContainer;

#[derive(Component, Clone, Copy)]
struct SpeciesListRow(pub u32);

#[derive(Component)]
struct BulkSpawnCountField;

#[derive(Component)]
struct BulkSpawnCountText;

#[derive(Component)]
struct BulkSpawnButton;

#[derive(Resource)]
pub struct SpeciesPanelState {
    pub count_buffer:    String,
    pub count_focused:   bool,
    pub count_committed: u32,
}

impl Default for SpeciesPanelState {
    fn default() -> Self {
        Self { count_buffer: String::new(), count_focused: false, count_committed: 10 }
    }
}


// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct SpeciesPanelPlugin;

impl Plugin for SpeciesPanelPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<SpeciesPanelState>()
            .add_systems(Update, (
                rebuild_species_list,
                handle_species_row_clicks,
                sync_species_row_visuals,
                handle_count_input,
                update_count_text,
                handle_bulk_spawn_button,
                sync_bulk_spawn_button_state,
            ));
    }
}


// ── Spawning ────────────────────────────────────────────────────────────────

pub fn spawn(parent: &mut ChildSpawnerCommands) {
    spawn_with_offset(parent, 0.0);
}

pub fn spawn_with_offset(parent: &mut ChildSpawnerCommands, top_offset_px: f32) {
    parent
        .spawn((
            SpeciesPanel,
            Node {
                position_type:  PositionType::Absolute,
                top:    Val::Px(top_offset_px),
                left:   Val::Px(0.0),
                bottom: Val::Px(BOTTOM_PANEL_HEIGHT_PX),
                width:  Val::Px(SPECIES_PANEL_WIDTH_PX),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(PADDING_PX)),
                ..default()
            },
            BackgroundColor(PANEL_BG_COLOR),
        ))
        .with_children(|panel| {
            // Title.
            panel.spawn((
                Text::new("Species Navigator"),
                TextFont { font_size: TITLE_FONT_PX, ..default() },
                TextColor(Color::srgb(0.92, 0.92, 0.92)),
                Node { margin: UiRect::bottom(Val::Px(10.0)), ..default() },
                Pickable::IGNORE,
            ));

            // Species are auto-loaded from `species/` on every editor entry
            // (`reload_species_*` in mod.rs) — no manual Load button.

            // Scrollable species list (`Overflow::scroll_y()`). No
            // explicit wheel handling — the list is bounded at ~tens of
            // entries and the panel is tall.
            panel.spawn((
                SpeciesListContainer,
                Node {
                    flex_grow:      1.0,
                    flex_basis:     Val::Px(0.0),
                    min_height:     Val::Px(0.0),
                    flex_direction: FlexDirection::Column,
                    overflow:       Overflow::scroll_y(),
                    margin:         UiRect::bottom(Val::Px(10.0)),
                    ..default()
                },
                ScrollPosition::default(),
            ));

            // ── Bulk-Spawn section ────────────────────────────────
            panel.spawn((
                Text::new("Bulk-Spawn Species"),
                TextFont { font_size: SECTION_HEADING_PX, ..default() },
                TextColor(Color::srgb(0.85, 0.85, 0.85)),
                Node { margin: UiRect::bottom(Val::Px(6.0)), ..default() },
                Pickable::IGNORE,
            ));

            panel.spawn((
                Text::new("Count"),
                TextFont { font_size: LABEL_FONT_PX, ..default() },
                TextColor(Color::srgb(0.70, 0.70, 0.70)),
                Pickable::IGNORE,
            ));
            panel.spawn((
                BulkSpawnCountField,
                Button,
                Node {
                    width:           Val::Percent(100.0),
                    height:          Val::Px(COUNT_FIELD_HEIGHT),
                    padding:         UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                    align_items:     AlignItems::Center,
                    justify_content: JustifyContent::FlexStart,
                    margin:          UiRect { top: Val::Px(4.0), bottom: Val::Px(8.0), ..default() },
                    ..default()
                },
                BackgroundColor(COUNT_BG_IDLE),
            ))
            .with_children(|btn| {
                btn.spawn((
                    BulkSpawnCountText,
                    Text::new("10"),
                    TextFont { font_size: 14.0, ..default() },
                    TextColor(Color::WHITE),
                    Pickable::IGNORE,
                ));
            });

            panel.spawn((
                BulkSpawnButton,
                Button,
                Node {
                    width:           Val::Percent(100.0),
                    height:          Val::Px(BULK_BTN_HEIGHT),
                    align_items:     AlignItems::Center,
                    justify_content: JustifyContent::Center,
                    ..default()
                },
                BackgroundColor(BULK_BTN_DISABLED),
            ))
            .with_children(|btn| {
                btn.spawn((
                    Text::new("Bulk-Spawn Species"),
                    TextFont { font_size: 14.0, ..default() },
                    TextColor(Color::WHITE),
                    Pickable::IGNORE,
                ));
            });
        });
}


// ── Systems ──────────────────────────────────────────────────────────────────

/// Rebuild the list rows when the loaded-species set actually changes. Gated on
/// `session.is_changed()` AND a content diff: comparing the existing rows' ids to
/// `loaded_species` means a spurious session change-mark (any unrelated field
/// touch) can NEVER despawn+respawn the rows → no flicker, regardless of upstream.
fn rebuild_species_list(
    session:       Res<EditorSession>,
    mut commands:  Commands,
    container_q:   Query<Entity, With<SpeciesListContainer>>,
    existing_rows: Query<(Entity, &SpeciesListRow)>,
) {
    if !session.is_changed() { return; }
    let Ok(container) = container_q.single() else { return };

    // Only rebuild if the species-id set differs from what's already shown.
    let mut current: Vec<u32> = existing_rows.iter().map(|(_, r)| r.0).collect();
    let mut wanted:  Vec<u32> = session.loaded_species.iter().map(|s| s.id).collect();
    current.sort_unstable();
    wanted.sort_unstable();
    if current == wanted { return; }

    // Despawn-and-rebuild — cheap given few rows and rare rebuilds.
    for (e, _) in &existing_rows { commands.entity(e).despawn(); }
    if session.loaded_species.is_empty() { return; }

    for species in &session.loaded_species {
        let row_entity = commands.spawn((
            SpeciesListRow(species.id),
            Button,
            Node {
                width:           Val::Percent(100.0),
                height:          Val::Px(SPECIES_ROW_HEIGHT),
                margin:          UiRect::bottom(Val::Px(SPECIES_ROW_GAP)),
                padding:         UiRect::axes(Val::Px(8.0), Val::Px(2.0)),
                align_items:     AlignItems::Center,
                justify_content: JustifyContent::FlexStart,
                flex_shrink:     0.0,
                ..default()
            },
            BackgroundColor(SPECIES_ROW_IDLE),
        ))
        .with_children(|r| {
            r.spawn((
                Text::new(species.name.clone()),
                TextFont { font_size: 12.0, ..default() },
                TextColor(Color::WHITE),
                Pickable::IGNORE,
            ));
        })
        .id();
        commands.entity(container).add_child(row_entity);
    }
}

/// Click handler: clicking a row sets it as the active species.
fn handle_species_row_clicks(
    mut interactions: Query<(&Interaction, &SpeciesListRow), Changed<Interaction>>,
    mut session:      ResMut<EditorSession>,
) {
    for (interaction, row) in &mut interactions {
        if matches!(*interaction, Interaction::Pressed) {
            session.selected_species_id = Some(row.0);
        }
    }
}

/// Per-row background colour: idle / hover / selected.
fn sync_species_row_visuals(
    session:  Res<EditorSession>,
    mut rows: Query<(&SpeciesListRow, &Interaction, &mut BackgroundColor)>,
) {
    let selected = session.selected_species_id;
    for (row, interaction, mut bg) in &mut rows {
        let is_selected = selected == Some(row.0);
        let target = match (is_selected, *interaction) {
            (true, _)                       => SPECIES_ROW_SELECTED,
            (false, Interaction::Hovered)
                | (false, Interaction::Pressed)
                                            => SPECIES_ROW_HOVER,
            _                               => SPECIES_ROW_IDLE,
        };
        if bg.0 != target { *bg = BackgroundColor(target); }
    }
}


// ── Bulk-spawn count input ──────────────────────────────────────────────────

fn handle_count_input(
    mouse:           Res<ButtonInput<MouseButton>>,
    mut keyboard:    MessageReader<KeyboardInput>,
    interaction_q:   Query<&Interaction, With<BulkSpawnCountField>>,
    mut state:       ResMut<SpeciesPanelState>,
) {
    let click_on_field = mouse.just_pressed(MouseButton::Left)
        && interaction_q.iter().any(|i| matches!(i, Interaction::Pressed));
    let click_outside  = mouse.just_pressed(MouseButton::Left) && !click_on_field;

    if click_on_field && !state.count_focused {
        state.count_focused = true;
        state.count_buffer = state.count_committed.to_string();
    }
    if click_outside && state.count_focused {
        commit_count(&mut state);
    }
    if !state.count_focused {
        for _ in keyboard.read() {}
        return;
    }
    for ev in keyboard.read() {
        if !ev.state.is_pressed() { continue; }
        match ev.key_code {
            KeyCode::Enter | KeyCode::NumpadEnter => commit_count(&mut state),
            KeyCode::Escape    => { state.count_focused = false; state.count_buffer.clear(); }
            KeyCode::Backspace => { state.count_buffer.pop(); }
            _ => if let Some(text) = ev.text.as_ref() {
                for c in text.chars() {
                    if state.count_buffer.len() >= COUNT_BUFFER_MAX_LEN { break; }
                    if c.is_ascii_digit() { state.count_buffer.push(c); }
                }
            }
        }
    }
}

fn commit_count(state: &mut SpeciesPanelState) {
    if let Ok(v) = state.count_buffer.parse::<u32>() {
        state.count_committed = v.clamp(COUNT_MIN, COUNT_MAX);
    }
    state.count_focused = false;
    state.count_buffer.clear();
}

fn update_count_text(
    state:      Res<SpeciesPanelState>,
    mut text_q: Query<&mut Text, With<BulkSpawnCountText>>,
    mut bg_q:   Query<&mut BackgroundColor, With<BulkSpawnCountField>>,
) {
    if !state.is_changed() { return; }
    let display = if state.count_focused {
        format!("{}_", state.count_buffer)
    } else {
        format!("{}", state.count_committed)
    };
    for mut t in &mut text_q { t.0 = display.clone(); }
    let bg = if state.count_focused { COUNT_BG_FOCUSED } else { COUNT_BG_IDLE };
    for mut b in &mut bg_q { if b.0 != bg { *b = BackgroundColor(bg); } }
}


// ── Bulk-Spawn Species button ───────────────────────────────────────────────

fn handle_bulk_spawn_button(
    mut interactions: Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<BulkSpawnButton>)>,
    state:            Res<SpeciesPanelState>,
    mut session:      ResMut<EditorSession>,
    heightmap:        Option<Res<HeightmapSampler>>,
    map_size:         Res<MapSize>,
    mut commands:     Commands,
    mut meshes:       ResMut<Assets<Mesh>>,
    mut materials:    ResMut<Assets<StandardMaterial>>,
    mut undo_stack:   ResMut<UndoStack>,
    org_materials:    Option<Res<crate::colony::OrganismMaterials>>,
    smoothing:        Option<Res<crate::simulation_settings::Smoothing>>,
) {
    let selected = session.selected_species_id;
    for (interaction, mut bg) in &mut interactions {
        let enabled = selected.is_some();
        match *interaction {
            Interaction::Pressed => {
                if !enabled {
                    *bg = BackgroundColor(BULK_BTN_DISABLED);
                    continue;
                }
                *bg = BackgroundColor(BULK_BTN_HOVER);
                let Some(heightmap) = heightmap.as_deref() else {
                    warn!("bulk-spawn: heightmap not loaded yet — try again in a moment");
                    continue;
                };
                let Some(species) = selected
                    .and_then(|id| session.loaded_species.iter().find(|s| s.id == id).cloned())
                else { continue };

                let smoothing_on = smoothing.as_deref().map(|s| s.0).unwrap_or(true);
                let mut rng = rand::rng();
                let mut new_ids = Vec::with_capacity(state.count_committed as usize);
                for _ in 0..state.count_committed {
                    let x = rng.random_range(0.0_f32..map_size.x);
                    let z = rng.random_range(0.0_f32..map_size.z);
                    let y = heightmap.height_at(x, z) + 0.5;
                    let id = spawn_species_template_at(
                        Vec3::new(x, y, z),
                        &species,
                        &mut session,
                        &mut commands,
                        &mut meshes,
                        &mut materials,
                        org_materials.as_deref(),
                        smoothing_on,
                    );
                    new_ids.push(id);
                }
                if !new_ids.is_empty() {
                    undo_stack.push(EditorAction::Created(new_ids));
                }
                info!(
                    "bulk-spawned {} \"{}\" organism(s)",
                    state.count_committed, species.name,
                );
            }
            Interaction::Hovered if enabled => *bg = BackgroundColor(BULK_BTN_HOVER),
            _ => *bg = BackgroundColor(if enabled { BULK_BTN_COLOR } else { BULK_BTN_DISABLED }),
        }
    }
}

/// Edge-update the bulk-spawn button's background when the species
/// selection changes — so an enabled-vs-disabled flip is reflected
/// even on a frame where no Interaction event fires.
fn sync_bulk_spawn_button_state(
    session: Res<EditorSession>,
    mut btn: Query<&mut BackgroundColor, With<BulkSpawnButton>>,
) {
    if !session.is_changed() { return; }
    let target = if session.selected_species_id.is_some() { BULK_BTN_COLOR } else { BULK_BTN_DISABLED };
    for mut b in &mut btn {
        if b.0 != target { *b = BackgroundColor(target); }
    }
}
