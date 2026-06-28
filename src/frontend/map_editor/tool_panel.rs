// Map editor — left tool panel: a brush selector (dropdown) + a one-shot
// "Color All" button. Mirrors `species_editor/editor_mode.rs` (the dropdown) and
// `species_editor/wrap.rs` (the one-shot action button).

use bevy::input::keyboard::KeyboardInput;
use bevy::prelude::*;
use std::fmt::Write as _;

use crate::frontend::PANEL_BG_COLOR;
use crate::simulation_settings::{BRUSH_RADIUS_PX_MAX, BRUSH_RADIUS_PX_MIN, WindowMode};

use super::gpu_paint::{self, BrushSizeEditState, PaintState};
use super::material::{MapBrush, MapEditorSession};
use super::paint_upload::PaintDirtyRect;
use super::{BOTTOM_PANEL_HEIGHT_PX, MapEditorPanel, TOP_PANEL_HEIGHT_PX};


// ── Tunables ──────────────────────────────────────────────────────────────────

const TOOL_PANEL_WIDTH_PX: f32 = 220.0;

const HEADER_HEIGHT: f32 = 32.0;
const OPTION_HEIGHT: f32 = 28.0;

const HEADER_BG:    Color = Color::srgb(0.22, 0.30, 0.40);
const HEADER_HOVER: Color = Color::srgb(0.30, 0.40, 0.52);
const OPT_BG:       Color = Color::srgb(0.18, 0.20, 0.24);
const OPT_HOVER:    Color = Color::srgb(0.28, 0.32, 0.38);
const OPT_ACTIVE:   Color = Color::srgb(0.20, 0.45, 0.55);

const COLOR_ALL_BG:    Color = Color::srgb(0.20, 0.45, 0.40);
const COLOR_ALL_HOVER: Color = Color::srgb(0.28, 0.58, 0.50);

// Brush-size field (mirrors statistics_panel's TimeSpeed field).
const BRUSH_SIZE_BG_IDLE:    Color = Color::srgb(0.20, 0.20, 0.22);
const BRUSH_SIZE_BG_FOCUSED: Color = Color::srgb(0.32, 0.30, 0.18);
const BRUSH_SIZE_BUF_MAX:    usize = 8;
const BRUSH_SIZE_HEIGHT_PX:  f32   = 26.0;


// ── Markers ──────────────────────────────────────────────────────────────────

#[derive(Component)]
pub struct MapToolPanel;
#[derive(Component)]
pub struct MapBrushHeaderButton;
#[derive(Component)]
pub struct MapBrushHeaderLabel;
#[derive(Component)]
pub struct MapBrushOption(pub MapBrush);
#[derive(Component)]
pub struct ColorAllButton;
#[derive(Component)]
pub struct BrushSizeInput;
#[derive(Component)]
pub struct BrushSizeText;


// ── Spawn ────────────────────────────────────────────────────────────────────

pub fn spawn_tool_panel(parent: &mut ChildSpawnerCommands, top_offset_px: f32) {
    parent
        .spawn((
            MapToolPanel,
            MapEditorPanel,
            Node {
                position_type:  PositionType::Absolute,
                top:    Val::Px(top_offset_px + TOP_PANEL_HEIGHT_PX),
                left:   Val::Px(0.0),
                bottom: Val::Px(BOTTOM_PANEL_HEIGHT_PX),
                width:  Val::Px(TOOL_PANEL_WIDTH_PX),
                padding: UiRect::all(Val::Px(8.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(6.0),
                display: Display::None, // shown only in MapEditor mode
                ..default()
            },
            BackgroundColor(PANEL_BG_COLOR),
        ))
        .with_children(|panel| {
            // Panel title.
            panel.spawn((
                Text::new("Brush"),
                TextFont { font_size: 15.0, ..default() },
                TextColor(Color::WHITE),
                Pickable::IGNORE,
            ));

            // Brush dropdown header (always visible; click toggles open).
            panel
                .spawn((
                    MapBrushHeaderButton,
                    Button,
                    Node {
                        width:           Val::Percent(100.0),
                        height:          Val::Px(HEADER_HEIGHT),
                        align_items:     AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        ..default()
                    },
                    BackgroundColor(HEADER_BG),
                ))
                .with_children(|h| {
                    h.spawn((
                        MapBrushHeaderLabel,
                        Text::new(header_text(MapBrush::default())),
                        TextFont { font_size: 13.0, ..default() },
                        TextColor(Color::WHITE),
                        Pickable::IGNORE,
                    ));
                });

            // Option rows (revealed below the header on open).
            for b in MapBrush::ALL {
                panel
                    .spawn((
                        MapBrushOption(b),
                        Button,
                        Node {
                            width:           Val::Percent(100.0),
                            height:          Val::Px(OPTION_HEIGHT),
                            align_items:     AlignItems::Center,
                            justify_content: JustifyContent::Center,
                            display:         Display::None, // shown only while open
                            ..default()
                        },
                        BackgroundColor(OPT_BG),
                    ))
                    .with_children(|o| {
                        o.spawn((
                            Text::new(b.label()),
                            TextFont { font_size: 12.0, ..default() },
                            TextColor(Color::WHITE),
                            Pickable::IGNORE,
                        ));
                    });
            }

            // ── "Brush size (px)" editable field (under the brush dropdown). ──
            panel.spawn((
                Text::new("Brush size (px)"),
                TextFont { font_size: 12.0, ..default() },
                TextColor(Color::srgb(0.70, 0.70, 0.70)),
                Node { margin: UiRect::top(Val::Px(6.0)), ..default() },
                Pickable::IGNORE,
            ));
            panel
                .spawn((
                    BrushSizeInput,
                    Button,
                    Node {
                        width:           Val::Percent(100.0),
                        height:          Val::Px(BRUSH_SIZE_HEIGHT_PX),
                        padding:         UiRect::axes(Val::Px(6.0), Val::Px(2.0)),
                        align_items:     AlignItems::Center,
                        justify_content: JustifyContent::FlexStart,
                        ..default()
                    },
                    BackgroundColor(BRUSH_SIZE_BG_IDLE),
                ))
                .with_children(|btn| {
                    btn.spawn((
                        BrushSizeText,
                        Text::new("10"),
                        TextFont { font_size: 14.0, ..default() },
                        TextColor(Color::WHITE),
                        Pickable::IGNORE,
                    ));
                });

            // Flexible spacer, then the "Color All" one-shot button at the bottom.
            panel.spawn(Node { flex_grow: 1.0, ..default() });
            panel
                .spawn((
                    ColorAllButton,
                    Button,
                    Node {
                        width:           Val::Percent(100.0),
                        height:          Val::Px(30.0),
                        align_items:     AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        ..default()
                    },
                    BackgroundColor(COLOR_ALL_BG),
                ))
                .with_children(|b| {
                    b.spawn((
                        Text::new("Color All"),
                        TextFont { font_size: 13.0, ..default() },
                        TextColor(Color::WHITE),
                        Pickable::IGNORE,
                    ));
                });
        });
}

fn header_text(brush: MapBrush) -> String {
    format!("Brush: {}", brush.label())
}


// ── Dropdown click handling ────────────────────────────────────────────────────

pub fn handle_brush_dropdown_clicks(
    mode:        Res<WindowMode>,
    mut session: ResMut<MapEditorSession>,
    headers:     Query<&Interaction, (Changed<Interaction>, With<MapBrushHeaderButton>)>,
    options:     Query<(&Interaction, &MapBrushOption), Changed<Interaction>>,
) {
    if *mode != WindowMode::MapEditor { return; }

    for interaction in &headers {
        if matches!(interaction, Interaction::Pressed) {
            session.brush_dropdown_open = !session.brush_dropdown_open;
        }
    }
    for (interaction, opt) in &options {
        if matches!(interaction, Interaction::Pressed) {
            session.selected_brush = opt.0;
            session.brush_dropdown_open = false;
        }
    }
}


// ── Per-frame sync (label, colours, option visibility) ──────────────────────────

#[allow(clippy::type_complexity)]
pub fn sync_brush_widget(
    mode:          Res<WindowMode>,
    mut session:   ResMut<MapEditorSession>,
    mut header_tx: Query<&mut Text, With<MapBrushHeaderLabel>>,
    mut header_bg: Query<(&Interaction, &mut BackgroundColor),
                         (With<MapBrushHeaderButton>, Without<MapBrushOption>)>,
    mut options:   Query<(&Interaction, &MapBrushOption, &mut BackgroundColor, &mut Node),
                         Without<MapBrushHeaderButton>>,
) {
    if *mode != WindowMode::MapEditor {
        // Force-close so the dropdown is collapsed on the next entry; the panel
        // itself is hidden by the visibility toggle.
        if session.brush_dropdown_open { session.brush_dropdown_open = false; }
        return;
    }

    let cur  = session.selected_brush;
    let open = session.brush_dropdown_open;

    // Header label tracks the current brush.
    let label = header_text(cur);
    for mut text in &mut header_tx {
        if text.0 != label { text.0 = label.clone(); }
    }
    // Header colour from hover.
    for (interaction, mut bg) in &mut header_bg {
        let c = match interaction {
            Interaction::Hovered | Interaction::Pressed => HEADER_HOVER,
            Interaction::None                           => HEADER_BG,
        };
        if bg.0 != c { bg.0 = c; }
    }
    // Options: visibility (open) + colour (hover / active).
    for (interaction, opt, mut bg, mut node) in &mut options {
        let want = if open { Display::Flex } else { Display::None };
        if node.display != want { node.display = want; }
        let c = match interaction {
            Interaction::Hovered | Interaction::Pressed => OPT_HOVER,
            Interaction::None if opt.0 == cur           => OPT_ACTIVE,
            Interaction::None                           => OPT_BG,
        };
        if bg.0 != c { bg.0 = c; }
    }
}


// ── "Color All" button handler ──────────────────────────────────────────────────

pub fn handle_color_all_click(
    mode:             Res<WindowMode>,
    session:          Res<MapEditorSession>,
    mut paint_state:  ResMut<PaintState>,
    mut dirty:        ResMut<PaintDirtyRect>,
    mut images:       ResMut<Assets<Image>>,
    mut interactions: Query<(&Interaction, &mut BackgroundColor),
                            (Changed<Interaction>, With<ColorAllButton>)>,
) {
    if *mode != WindowMode::MapEditor { return; }

    for (interaction, mut bg) in &mut interactions {
        match *interaction {
            Interaction::Pressed => {
                if let Some(material) = session.selected_material {
                    // Whole-texture fill — the only flood path (the brush never floods).
                    // Updates the CPU mirror, syncs the main-world Image untracked, and
                    // marks the whole texture dirty for the in-place GPU upload (no
                    // material refresh needed — the texture is updated in place).
                    gpu_paint::color_all_fill(
                        &mut images, &mut paint_state, &mut dirty, material.srgb_u8(),
                    );
                }
                *bg = BackgroundColor(COLOR_ALL_HOVER);
            }
            Interaction::Hovered => *bg = BackgroundColor(COLOR_ALL_HOVER),
            Interaction::None    => *bg = BackgroundColor(COLOR_ALL_BG),
        }
    }
}


// ── Brush-size field handlers (mirror statistics_panel's TimeSpeed field) ────────

/// Click + keyboard router for the "Brush size (px)" field.
///   * LMB on box → focus, prefill buffer with the committed value.
///   * LMB outside while focused → commit and unfocus.
///   * Focused keys: Enter → commit; Escape → cancel; Backspace → del;
///     digit/'.' → append (bounded by BRUSH_SIZE_BUF_MAX).
/// Commit parses the buffer, requires finite, clamps to the px range, and writes
/// `MapEditorSession::brush_radius_px`.
pub fn handle_brush_size_input(
    mode:          Res<WindowMode>,
    mouse:         Res<ButtonInput<MouseButton>>,
    mut keyboard:  MessageReader<KeyboardInput>,
    interaction_q: Query<&Interaction, With<BrushSizeInput>>,
    mut state:     ResMut<BrushSizeEditState>,
    mut session:   ResMut<MapEditorSession>,
) {
    if *mode != WindowMode::MapEditor {
        if state.focused { state.focused = false; state.buffer.clear(); }
        for _ in keyboard.read() {}
        return;
    }

    let click_on_input = mouse.just_pressed(MouseButton::Left)
        && interaction_q.iter().any(|i| matches!(i, Interaction::Pressed));
    let click_outside  = mouse.just_pressed(MouseButton::Left) && !click_on_input;

    if click_on_input && !state.focused {
        state.focused = true;
        state.buffer.clear();
        let _ = write!(state.buffer, "{:.0}", session.brush_radius_px);
    }

    if click_outside && state.focused {
        commit_brush_size(&mut state, &mut session);
    }

    if !state.focused {
        for _ in keyboard.read() {}   // drain to avoid event-buffer growth
        return;
    }

    for ev in keyboard.read() {
        if !ev.state.is_pressed() { continue; }
        match ev.key_code {
            KeyCode::Enter | KeyCode::NumpadEnter => commit_brush_size(&mut state, &mut session),
            KeyCode::Escape => { state.focused = false; state.buffer.clear(); }
            KeyCode::Backspace => { state.buffer.pop(); }
            _ => {
                if let Some(text) = ev.text.as_ref() {
                    for c in text.chars() {
                        if state.buffer.len() >= BRUSH_SIZE_BUF_MAX { break; }
                        if c.is_ascii_digit() {
                            state.buffer.push(c);
                        } else if c == '.' && !state.buffer.contains('.') {
                            state.buffer.push(c);
                        }
                    }
                }
            }
        }
    }
}

/// Parse + clamp the buffer into `brush_radius_px`. Always unfocus + clear.
fn commit_brush_size(state: &mut BrushSizeEditState, session: &mut MapEditorSession) {
    if let Ok(v) = state.buffer.parse::<f32>() {
        if v.is_finite() {
            session.brush_radius_px = v.clamp(BRUSH_RADIUS_PX_MIN, BRUSH_RADIUS_PX_MAX);
        }
    }
    state.focused = false;
    state.buffer.clear();
}

/// Sync the field's text + background colour with state. Early-returns unchanged.
pub fn update_brush_size_text(
    mode:       Res<WindowMode>,
    state:      Res<BrushSizeEditState>,
    session:    Res<MapEditorSession>,
    mut text_q: Query<&mut Text, With<BrushSizeText>>,
    mut bg_q:   Query<&mut BackgroundColor, With<BrushSizeInput>>,
) {
    if *mode != WindowMode::MapEditor { return; }
    if !state.is_changed() && !session.is_changed() { return; }

    let display = if state.focused {
        format!("{}_", state.buffer)
    } else {
        format!("{:.0}", session.brush_radius_px)
    };
    for mut text in &mut text_q {
        if text.0 != display { text.0 = display.clone(); }
    }

    let bg = if state.focused { BRUSH_SIZE_BG_FOCUSED } else { BRUSH_SIZE_BG_IDLE };
    for mut b in &mut bg_q {
        if b.0 != bg { *b = BackgroundColor(bg); }
    }
}
