// Species editor — bottom panel: a row of cell-type tiles (flat colour
// swatches). Clicking a tile sets `SpeciesSession::selected_cell_type`, driving
// the preview-cell flow in `placement.rs`.

use bevy::prelude::*;

use crate::cell::CellType;
use crate::frontend::PANEL_BG_COLOR;
use crate::simulation_settings::WindowMode;

use super::session::SpeciesSession;
use super::BOTTOM_PANEL_HEIGHT_PX;


// ── Tunables ─────────────────────────────────────────────────────────────────

const TILE_SIZE_PX:   f32 = 56.0;
const TILE_GAP_PX:    f32 = 12.0;
const TILE_BORDER_PX: f32 = 3.0;
const TILE_BORDER_IDLE:     Color = Color::srgb(0.25, 0.25, 0.25);
const TILE_BORDER_SELECTED: Color = Color::srgb(0.95, 0.95, 0.2);
const TILE_BORDER_HOVER:    Color = Color::srgb(0.65, 0.65, 0.65);

/// Width of each tile column (swatch + label). Slightly wider than the swatch so
/// single-word labels (e.g. "Placeholder") fit on one line.
const LABEL_WIDTH_PX: f32 = 64.0;
const LABEL_FONT_PX:  f32 = 10.0;
/// Vertical gap between the swatch and its label.
const LABEL_GAP_PX:   f32 = 4.0;

/// UI swatch colour — the species editor renders in sRGB, so read the cell
/// type's sRGB display colour (and opacity) straight from the registry.
fn ui_color_for(ct: CellType) -> Color {
    let [r, g, b] = ct.srgb();
    Color::srgba(r, g, b, ct.alpha())
}

/// Palette label for a cell type. Same as `CellType::label()`, except the one
/// long single-word name is broken across two lines so it stays within the
/// tile width.
fn tile_label(ct: CellType) -> String {
    match ct {
        CellType::HydroxylApatite => "Hydroxyl-\nApatite".to_string(),
        _ => ct.label().to_string(),
    }
}


// ── Markers ──────────────────────────────────────────────────────────────────

#[derive(Component)]
pub struct SpeciesBottomPanel;

#[derive(Component, Clone, Copy)]
pub struct CellTypeTile(pub CellType);


// ── Spawn ────────────────────────────────────────────────────────────────────

pub fn spawn_bottom_panel(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((
            SpeciesBottomPanel,
            super::SpeciesEditorPanel,
            Node {
                position_type:   PositionType::Absolute,
                bottom:          Val::Px(0.0),
                left:            Val::Px(0.0),
                right:           Val::Px(0.0),
                height:          Val::Px(BOTTOM_PANEL_HEIGHT_PX),
                padding:         UiRect {
                    left:   Val::Px(10.0),
                    right:  Val::Px(10.0),
                    top:    Val::Px(12.0),
                    bottom: Val::Px(8.0),
                },
                flex_direction:  FlexDirection::Row,
                // Top-align the columns so every swatch lines up regardless of
                // whether its label wraps to one or two lines.
                align_items:     AlignItems::FlexStart,
                justify_content: JustifyContent::Center,
                column_gap:      Val::Px(TILE_GAP_PX),
                display:         Display::None,
                ..default()
            },
            BackgroundColor(PANEL_BG_COLOR),
        ))
        .with_children(|panel| {
            // Every cell type is offered automatically — no per-type opt-in.
            for ct in CellType::ALL {
                tile(panel, ct);
            }
        });
}

fn tile(parent: &mut ChildSpawnerCommands, ct: CellType) {
    // Column: the clickable colour swatch on top, its label centred beneath.
    parent
        .spawn(Node {
            flex_direction: FlexDirection::Column,
            align_items:    AlignItems::Center,
            width:          Val::Px(LABEL_WIDTH_PX),
            row_gap:        Val::Px(LABEL_GAP_PX),
            ..default()
        })
        .with_children(|col| {
            col.spawn((
                CellTypeTile(ct),
                Button,
                Node {
                    width:  Val::Px(TILE_SIZE_PX),
                    height: Val::Px(TILE_SIZE_PX),
                    border: UiRect::all(Val::Px(TILE_BORDER_PX)),
                    ..default()
                },
                BorderColor::all(TILE_BORDER_IDLE),
                BackgroundColor(ui_color_for(ct)),
            ));
            col.spawn((
                Text::new(tile_label(ct)),
                TextFont { font_size: LABEL_FONT_PX, ..default() },
                TextColor(Color::WHITE),
                TextLayout::new_with_justify(Justify::Center),
                Node { width: Val::Px(LABEL_WIDTH_PX), ..default() },
                Pickable::IGNORE,
            ));
        });
}


// ── Click handler ────────────────────────────────────────────────────────────

pub fn handle_tile_clicks(
    mode:             Res<WindowMode>,
    mut interactions: Query<(&Interaction, &CellTypeTile), Changed<Interaction>>,
    mut session:      ResMut<SpeciesSession>,
) {
    if *mode != WindowMode::SpeciesEditor { return; }
    if !session.first_cell_spawned { return; }

    for (interaction, tile) in &mut interactions {
        if matches!(*interaction, Interaction::Pressed) {
            session.selected_cell_type = Some(tile.0);
        }
    }
}

/// Highlight the selected tile and update hover borders.
pub fn sync_tile_borders(
    session:     Res<SpeciesSession>,
    mut tiles:   Query<(&Interaction, &CellTypeTile, &mut BorderColor)>,
) {
    let selected = session.selected_cell_type;
    for (interaction, tile, mut border) in &mut tiles {
        let is_selected = selected == Some(tile.0);
        let target = match (is_selected, *interaction) {
            (true, _)                       => TILE_BORDER_SELECTED,
            (false, Interaction::Hovered)   => TILE_BORDER_HOVER,
            (false, Interaction::Pressed)   => TILE_BORDER_HOVER,
            _                               => TILE_BORDER_IDLE,
        };
        *border = BorderColor::all(target);
    }
}
