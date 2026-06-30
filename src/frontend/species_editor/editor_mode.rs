// Species editor — left tool panel: editor-mode dropdown + Diagnostics mode.
//
// `species_editor_tool_panel` is a left-side vertical panel mirroring the
// Body-part-index panel's form (anchored below the top bar, above the bottom
// panel, fixed width). It holds the Editor-Mode dropdown — a header button
// reading "Editor Mode: <mode>" plus option rows revealed below it on click —
// and, beneath that, the Diagnostics text field. The dropdown selects the
// active `EditorMode`:
//   * Addition    — left-click places the selected cell (the default).
//   * Deletion    — hovered cell is highlighted red, left-click removes it
//                   (the pick/delete logic lives in `deletion.rs`).
//   * Diagnostics — hovered cell is highlighted cyan and its `CellType` is
//                   reported in the text field; never mutates.
//
// Placement is live ONLY in Addition (see `placement.rs`).

use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::cell::CellType;
use crate::frontend::{PANEL_BG_COLOR, ViewportImage};
use crate::player_plugin::FlyCam;
use crate::simulation_settings::WindowMode;
use crate::volumetric_growth::build_uncolored_mesh_from_ocg;

use super::session::{EditorMode, SpeciesSession};
use super::{BOTTOM_PANEL_HEIGHT_PX, SpeciesEditorPanel, TOP_PANEL_HEIGHT_PX,
            SPECIES_EDITOR_LAYER, SPECIES_EDITOR_ORIGIN};


// ── Tunables ──────────────────────────────────────────────────────────────────

/// Width of the left tool panel (a touch narrower than the right Body-part panel).
pub const TOOL_PANEL_WIDTH_PX: f32 = 220.0;

const HEADER_HEIGHT: f32 = 32.0;
const OPTION_HEIGHT: f32 = 28.0;
const FIELD_HEIGHT:  f32 = 28.0;

const HEADER_BG:    Color = Color::srgb(0.22, 0.30, 0.40);
const HEADER_HOVER: Color = Color::srgb(0.30, 0.40, 0.52);
const OPT_BG:       Color = Color::srgb(0.18, 0.20, 0.24);
const OPT_HOVER:    Color = Color::srgb(0.28, 0.32, 0.38);
const OPT_ACTIVE:   Color = Color::srgb(0.20, 0.45, 0.55);
const FIELD_BG:     Color = Color::srgb(0.12, 0.13, 0.16);

/// Sculpt numeric-field backgrounds (mirror the Map-Editor brush-size field).
pub const SCULPT_FIELD_BG_IDLE:    Color = Color::srgb(0.20, 0.20, 0.22);
pub const SCULPT_FIELD_BG_FOCUSED: Color = Color::srgb(0.32, 0.30, 0.18);
/// Add/Erase toggle backgrounds (green-ish for Add, red-ish for Erase).
pub const SCULPT_OP_ADD_BG:   Color = Color::srgb(0.20, 0.45, 0.40);
pub const SCULPT_OP_ERASE_BG: Color = Color::srgb(0.50, 0.22, 0.22);

/// Cyan overlay painted over the hovered cell in Diagnostics mode (distinct from
/// deletion's red).
const DIAG_HILITE: Color = Color::srgb(0.10, 0.85, 0.95);

/// Default diagnostics field text when nothing is hovered.
const FIELD_IDLE_TEXT: &str = "Hover a cell";

/// Max pixel distance from cursor to a cell's screen projection to count as
/// hovered. Mirrors `deletion::PICK_RADIUS_PX`.
const PICK_RADIUS_PX: f32 = 45.0;


// ── Markers ─────────────────────────────────────────────────────────────────

/// The left tool panel (`species_editor_tool_panel`). Carries `SpeciesEditorPanel`
/// so the mode-transition shows it only in the species editor.
#[derive(Component)]
pub struct SpeciesEditorToolPanel;
/// The always-visible header button that opens/closes the dropdown.
#[derive(Component)]
pub struct EditorModeHeaderButton;
/// The text inside the header (shows the current mode's label).
#[derive(Component)]
pub struct EditorModeHeaderLabel;
/// One selectable option row.
#[derive(Component)]
pub struct EditorModeOption(pub EditorMode);
/// The diagnostics text-field container (display controlled by `sync_*`).
#[derive(Component)]
pub struct DiagnosticsField;
/// The text inside the diagnostics field.
#[derive(Component)]
pub struct DiagnosticsLabel;
/// Sculpt-mode "Brush radius (cell lengths)" caption (shown only in Sculpt mode).
#[derive(Component)]
pub struct SculptRadiusCaption;
/// Sculpt-mode "Brush radius" numeric input box (shown only in Sculpt mode).
#[derive(Component)]
pub struct SculptRadiusInput;
/// The text inside the sculpt-radius input box.
#[derive(Component)]
pub struct SculptRadiusText;
/// Sculpt-mode Add/Erase toggle button (shown only in Sculpt mode).
#[derive(Component)]
pub struct SculptOpToggle;
/// The text inside the Add/Erase toggle ("Add" / "Erase").
#[derive(Component)]
pub struct SculptOpText;
/// The 3D overlay highlighting the hovered cell in Diagnostics mode.
#[derive(Component)]
pub struct DiagnosticsHighlight;


// ── Shared cell-pick helper ───────────────────────────────────────────────────

/// Nearest RENDERED cell to `cursor` (logical px), within `pick_radius_px`.
/// Returns `(body_part_index, ocg_index, world_pos, cell_type)`; `ocg_index` is
/// the right-half index (`combined_index % part.ocg.len()`) so Bilateral mirror
/// cells map back to the OCG entry they came from. Shared by Deletion (which
/// removes that entry) and Diagnostics (which reports its type).
pub fn nearest_rendered_cell(
    session:        &SpeciesSession,
    camera:         &Camera,
    cam_xf:         &GlobalTransform,
    cursor:         Vec2,
    inv_scale:      f32,
    pick_radius_px: f32,
) -> Option<(usize, usize, Vec3, CellType)> {
    let mut best: Option<(f32, usize, usize, Vec3, CellType)> = None; // (d², part, ocg_index, world, type)
    for (pi, part) in session.body_parts.iter().enumerate() {
        if part.ocg.is_empty() { continue; }
        let n = part.ocg.len();
        for (ci, entry) in session.combined_ocg(&part.ocg).iter().enumerate() {
            let world = SPECIES_EDITOR_ORIGIN + entry.1;
            let Ok(vp_phys) = camera.world_to_viewport(cam_xf, world) else { continue };
            let vp = vp_phys * inv_scale;
            let d2 = (vp - cursor).length_squared();
            if best.map_or(true, |(b, ..)| d2 < b) {
                best = Some((d2, pi, ci % n, world, entry.2));
            }
        }
    }
    match best {
        Some((d2, pi, oi, world, ct)) if d2.sqrt() <= pick_radius_px => Some((pi, oi, world, ct)),
        _ => None,
    }
}


// ── Spawn ─────────────────────────────────────────────────────────────────────

/// Spawn the left tool panel (`species_editor_tool_panel`) — same form as the
/// right-side Body-part-index panel but anchored to the left edge. It holds the
/// Editor-Mode dropdown (header reading "Editor Mode: <mode>" plus the option
/// rows revealed below it when open) and the Diagnostics text field beneath.
pub fn spawn_tool_panel(parent: &mut ChildSpawnerCommands, top_offset_px: f32) {
    parent
        .spawn((
            SpeciesEditorToolPanel,
            SpeciesEditorPanel,
            Node {
                position_type:  PositionType::Absolute,
                top:    Val::Px(top_offset_px + TOP_PANEL_HEIGHT_PX),
                left:   Val::Px(0.0),
                bottom: Val::Px(BOTTOM_PANEL_HEIGHT_PX),
                width:  Val::Px(TOOL_PANEL_WIDTH_PX),
                padding: UiRect::all(Val::Px(8.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(6.0),
                display: Display::None,   // shown only in SpeciesEditor mode
                ..default()
            },
            BackgroundColor(PANEL_BG_COLOR),
        ))
        .with_children(|panel| {
            // Panel title.
            panel.spawn((
                Text::new("Editor tools"),
                TextFont { font_size: 15.0, ..default() },
                TextColor(Color::WHITE),
                Pickable::IGNORE,
            ));

            // Editor-Mode dropdown header (always visible; click toggles open).
            panel
                .spawn((
                    EditorModeHeaderButton,
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
                        EditorModeHeaderLabel,
                        Text::new(header_text(EditorMode::Addition)),
                        TextFont { font_size: 13.0, ..default() },
                        TextColor(Color::WHITE),
                        Pickable::IGNORE,
                    ));
                });

            // Option rows in normal column flow (revealed below the header on open).
            for mode in EditorMode::ALL {
                panel
                    .spawn((
                        EditorModeOption(mode),
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
                            Text::new(mode.label()),
                            TextFont { font_size: 12.0, ..default() },
                            TextColor(Color::WHITE),
                            Pickable::IGNORE,
                        ));
                    });
            }

            // "Wrap" tool: coat the active part in the selected cell type.
            super::wrap::spawn_wrap_button(panel);

            // Diagnostics text field (shown only in Diagnostics mode — see sync).
            panel
                .spawn((
                    DiagnosticsField,
                    Node {
                        width:           Val::Percent(100.0),
                        height:          Val::Px(FIELD_HEIGHT),
                        align_items:     AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        display:         Display::None,
                        ..default()
                    },
                    BackgroundColor(FIELD_BG),
                ))
                .with_children(|f| {
                    f.spawn((
                        DiagnosticsLabel,
                        Text::new(FIELD_IDLE_TEXT),
                        TextFont { font_size: 13.0, ..default() },
                        TextColor(Color::WHITE),
                        Pickable::IGNORE,
                    ));
                });

            // ── Sculpt-mode widgets (shown only in Sculpt mode by sync). ──
            // Brush-radius caption.
            panel.spawn((
                SculptRadiusCaption,
                Text::new("Brush radius (cell lengths)"),
                TextFont { font_size: 12.0, ..default() },
                TextColor(Color::srgb(0.70, 0.70, 0.70)),
                Node {
                    margin:  UiRect::top(Val::Px(6.0)),
                    display: Display::None,
                    ..default()
                },
                Pickable::IGNORE,
            ));
            // Brush-radius numeric input (copies the Map-Editor BrushSizeInput shape).
            panel
                .spawn((
                    SculptRadiusInput,
                    Button,
                    Node {
                        width:           Val::Percent(100.0),
                        height:          Val::Px(FIELD_HEIGHT),
                        padding:         UiRect::axes(Val::Px(6.0), Val::Px(2.0)),
                        align_items:     AlignItems::Center,
                        justify_content: JustifyContent::FlexStart,
                        display:         Display::None,
                        ..default()
                    },
                    BackgroundColor(SCULPT_FIELD_BG_IDLE),
                ))
                .with_children(|btn| {
                    btn.spawn((
                        SculptRadiusText,
                        Text::new("2.0"),
                        TextFont { font_size: 14.0, ..default() },
                        TextColor(Color::WHITE),
                        Pickable::IGNORE,
                    ));
                });
            // Add/Erase toggle button.
            panel
                .spawn((
                    SculptOpToggle,
                    Button,
                    Node {
                        width:           Val::Percent(100.0),
                        height:          Val::Px(FIELD_HEIGHT),
                        align_items:     AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        margin:          UiRect::top(Val::Px(6.0)),
                        display:         Display::None,
                        ..default()
                    },
                    BackgroundColor(SCULPT_OP_ADD_BG),
                ))
                .with_children(|b| {
                    b.spawn((
                        SculptOpText,
                        Text::new("Add"),
                        TextFont { font_size: 13.0, ..default() },
                        TextColor(Color::WHITE),
                        Pickable::IGNORE,
                    ));
                });

            // Flexible spacer, then the camera-movement toggle pinned to the
            // very bottom of the panel.
            panel.spawn(Node { flex_grow: 1.0, ..default() });
            super::camera::spawn_camera_mode_button(panel);
        });
}

/// Header button text for a mode: `"Editor Mode: <short>"`.
fn header_text(mode: EditorMode) -> String {
    format!("Editor Mode: {}", mode.short_label())
}


// ── Click handling ──────────────────────────────────────────────────────────

/// Header click toggles the dropdown; option click sets the mode and closes.
pub fn handle_editor_mode_clicks(
    mode:        Res<WindowMode>,
    mut session: ResMut<SpeciesSession>,
    headers:     Query<&Interaction, (Changed<Interaction>, With<EditorModeHeaderButton>)>,
    options:     Query<(&Interaction, &EditorModeOption), Changed<Interaction>>,
) {
    if *mode != WindowMode::SpeciesEditor { return; }

    for interaction in &headers {
        if matches!(interaction, Interaction::Pressed) {
            session.mode_dropdown_open = !session.mode_dropdown_open;
        }
    }
    for (interaction, opt) in &options {
        if matches!(interaction, Interaction::Pressed) {
            session.editor_mode = opt.0;
            session.mode_dropdown_open = false;
        }
    }
}


// ── Per-frame sync (labels, colours, visibility) ──────────────────────────────

#[allow(clippy::type_complexity)]
pub fn sync_editor_mode_widget(
    mode:          Res<WindowMode>,
    mut session:   ResMut<SpeciesSession>,
    mut header_tx: Query<&mut Text, With<EditorModeHeaderLabel>>,
    mut header_bg: Query<(&Interaction, &mut BackgroundColor),
                         (With<EditorModeHeaderButton>, Without<EditorModeOption>)>,
    mut options:   Query<(&Interaction, &EditorModeOption, &mut BackgroundColor, &mut Node),
                         Without<EditorModeHeaderButton>>,
    mut field:     Query<&mut Node, (With<DiagnosticsField>, Without<EditorModeOption>)>,
    mut sc_cap:    Query<&mut Node, (With<SculptRadiusCaption>, Without<EditorModeOption>,
                                     Without<DiagnosticsField>, Without<SculptRadiusInput>,
                                     Without<SculptOpToggle>)>,
    mut sc_inp:    Query<&mut Node, (With<SculptRadiusInput>, Without<EditorModeOption>,
                                     Without<DiagnosticsField>, Without<SculptRadiusCaption>,
                                     Without<SculptOpToggle>)>,
    mut sc_op:     Query<&mut Node, (With<SculptOpToggle>, Without<EditorModeOption>,
                                     Without<DiagnosticsField>, Without<SculptRadiusCaption>,
                                     Without<SculptRadiusInput>)>,
) {
    if *mode != WindowMode::SpeciesEditor {
        // Force-close so the dropdown is collapsed on the next entry; the
        // container itself is hidden by the panel mode-transition.
        if session.mode_dropdown_open { session.mode_dropdown_open = false; }
        for mut node in &mut field {
            if node.display != Display::None { node.display = Display::None; }
        }
        for mut node in &mut sc_cap { if node.display != Display::None { node.display = Display::None; } }
        for mut node in &mut sc_inp { if node.display != Display::None { node.display = Display::None; } }
        for mut node in &mut sc_op  { if node.display != Display::None { node.display = Display::None; } }
        return;
    }

    let cur  = session.editor_mode;
    let open = session.mode_dropdown_open;

    // Header label tracks the current mode ("Editor Mode: <mode>").
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
    // Diagnostics field: visible only in Diagnostics mode.
    let want = if session.is_diagnostics() { Display::Flex } else { Display::None };
    for mut node in &mut field {
        if node.display != want { node.display = want; }
    }
    // Sculpt widgets (radius caption + input + Add/Erase toggle): Sculpt mode only.
    let want_sculpt = if session.is_sculpt() { Display::Flex } else { Display::None };
    for mut node in &mut sc_cap { if node.display != want_sculpt { node.display = want_sculpt; } }
    for mut node in &mut sc_inp { if node.display != want_sculpt { node.display = want_sculpt; } }
    for mut node in &mut sc_op  { if node.display != want_sculpt { node.display = want_sculpt; } }
}


// ── Diagnostics hover ─────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub fn update_diagnostics_hover(
    mode:          Res<WindowMode>,
    session:       Res<SpeciesSession>,
    mut commands:  Commands,
    mut meshes:    ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    cameras:       Query<(&Camera, &GlobalTransform), With<FlyCam>>,
    windows:       Query<&Window, With<PrimaryWindow>>,
    viewport_q:    Query<&bevy::ui::ComputedNode, With<ViewportImage>>,
    mut hl_q:      Query<(Entity, &mut Transform, &mut Visibility), With<DiagnosticsHighlight>>,
    mut label_q:   Query<&mut Text, With<DiagnosticsLabel>>,
) {
    if *mode != WindowMode::SpeciesEditor || !session.is_diagnostics() {
        for (e, _, _) in &hl_q { commands.entity(e).despawn(); }
        return;
    }

    let Ok((camera, cam_xf)) = cameras.single()      else { return };
    let Ok(window)           = windows.single()       else { return };
    let Ok(viewport_node)    = viewport_q.single()    else { return };
    let Some(cursor)         = window.cursor_position() else {
        for (_, _, mut v) in &mut hl_q { *v = Visibility::Hidden; }
        set_label(&mut label_q, FIELD_IDLE_TEXT);
        return;
    };
    let inv_scale = viewport_node.inverse_scale_factor;

    match nearest_rendered_cell(&session, camera, cam_xf, cursor, inv_scale, PICK_RADIUS_PX) {
        Some((_pi, _oi, world, ct)) => {
            set_label(&mut label_q, ct.label());
            if let Ok((_, mut t, mut v)) = hl_q.single_mut() {
                t.translation = world;
                *v = Visibility::Inherited;
            } else {
                // Single-cell overlay, scaled up slightly to cover the cell
                // (avoids z-fighting); unlit cyan. Geometry is type-agnostic.
                let mesh = meshes.add(build_uncolored_mesh_from_ocg(&[(0, Vec3::ZERO, CellType::DigestionCell)]));
                let mat  = materials.add(StandardMaterial {
                    base_color: DIAG_HILITE,
                    unlit:      true,
                    ..default()
                });
                commands.spawn((
                    DiagnosticsHighlight,
                    Mesh3d(mesh),
                    MeshMaterial3d(mat),
                    Transform::from_translation(world).with_scale(Vec3::splat(1.12)),
                    RenderLayers::layer(SPECIES_EDITOR_LAYER),
                    bevy::light::NotShadowCaster,
                ));
            }
        }
        None => {
            for (_, _, mut v) in &mut hl_q { *v = Visibility::Hidden; }
            set_label(&mut label_q, FIELD_IDLE_TEXT);
        }
    }
}

fn set_label(label_q: &mut Query<&mut Text, With<DiagnosticsLabel>>, s: &str) {
    for mut t in label_q {
        if t.0 != s { t.0 = s.to_string(); }
    }
}
