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

/// UI tile colours, manually kept in sync with `CellType::color()`. NOT a 1:1
/// copy: these are sRGB but `CellType::color()` returns linear RGB, so reusing
/// it directly would double-correct.
fn ui_color_for(ct: CellType) -> Color {
    match ct {
        CellType::Photo       => Color::srgb(0.2, 0.8, 0.2),
        CellType::NonPhoto    => Color::srgb(0.8, 0.2, 0.2),
        CellType::Placeholder => Color::srgb(0.2, 0.45, 0.95),
        CellType::SubLimb     => Color::srgb(0.6, 0.2, 0.9),
        CellType::YellowCell  => Color::srgb(1.0, 0.95, 0.1),
        CellType::OrangeCell  => Color::srgb(1.0, 0.55, 0.0),
        CellType::BrownCell   => Color::srgb(0.45, 0.27, 0.12),
    }
}

fn label_for(ct: CellType) -> &'static str {
    match ct {
        CellType::Photo       => "Photo",
        CellType::NonPhoto    => "NonPhoto",
        CellType::Placeholder => "Placeholder",
        CellType::SubLimb     => "Sub-Limb",
        CellType::YellowCell  => "Yellow",
        CellType::OrangeCell  => "Orange",
        CellType::BrownCell   => "Brown",
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
                padding:         UiRect::all(Val::Px(10.0)),
                flex_direction:  FlexDirection::Row,
                align_items:     AlignItems::Center,
                justify_content: JustifyContent::Center,
                column_gap:      Val::Px(TILE_GAP_PX),
                display:         Display::None,
                ..default()
            },
            BackgroundColor(PANEL_BG_COLOR),
        ))
        .with_children(|panel| {
            for ct in [CellType::Photo, CellType::NonPhoto, CellType::Placeholder, CellType::SubLimb] {
                tile(panel, ct);
            }
        });
}

fn tile(parent: &mut ChildSpawnerCommands, ct: CellType) {
    parent
        .spawn((
            CellTypeTile(ct),
            Button,
            Node {
                width:           Val::Px(TILE_SIZE_PX),
                height:          Val::Px(TILE_SIZE_PX),
                border:          UiRect::all(Val::Px(TILE_BORDER_PX)),
                align_items:     AlignItems::FlexEnd,
                justify_content: JustifyContent::Center,
                padding:         UiRect::bottom(Val::Px(2.0)),
                ..default()
            },
            BorderColor::all(TILE_BORDER_IDLE),
            BackgroundColor(ui_color_for(ct)),
        ))
        .with_children(|t| {
            t.spawn((
                Text::new(label_for(ct).to_string()),
                TextFont { font_size: 10.0, ..default() },
                TextColor(Color::WHITE),
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
