// Map editor — bottom panel: a row of paint-colour swatches. Clicking a tile
// sets `MapEditorSession::selected_material`, which the brush + "Color All"
// systems read. Mirrors `species_editor/bottom_panel.rs`.

use bevy::prelude::*;

use crate::frontend::PANEL_BG_COLOR;
use crate::simulation_settings::WindowMode;

use super::material::{MapEditorSession, MapMaterial};
use super::{BOTTOM_PANEL_HEIGHT_PX, MapEditorPanel};


// ── Tunables ──────────────────────────────────────────────────────────────────

const TILE_SIZE_PX:   f32 = 56.0;
const TILE_GAP_PX:    f32 = 12.0;
const TILE_BORDER_PX: f32 = 3.0;
const TILE_BORDER_IDLE:     Color = Color::srgb(0.25, 0.25, 0.25);
const TILE_BORDER_SELECTED: Color = Color::srgb(0.95, 0.95, 0.2);
const TILE_BORDER_HOVER:    Color = Color::srgb(0.65, 0.65, 0.65);

const LABEL_WIDTH_PX: f32 = 64.0;
const LABEL_FONT_PX:  f32 = 10.0;
const LABEL_GAP_PX:   f32 = 4.0;


// ── Markers ──────────────────────────────────────────────────────────────────

#[derive(Component)]
pub struct MapBottomPanel;

#[derive(Component, Clone, Copy)]
pub struct MapMaterialTile(pub MapMaterial);


// ── Spawn ────────────────────────────────────────────────────────────────────

pub fn spawn_bottom_panel(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((
            MapBottomPanel,
            MapEditorPanel,
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
                align_items:     AlignItems::FlexStart,
                justify_content: JustifyContent::Center,
                column_gap:      Val::Px(TILE_GAP_PX),
                display:         Display::None, // shown only in MapEditor mode
                ..default()
            },
            BackgroundColor(PANEL_BG_COLOR),
        ))
        .with_children(|panel| {
            for m in MapMaterial::ALL {
                tile(panel, m);
            }
        });
}

fn tile(parent: &mut ChildSpawnerCommands, m: MapMaterial) {
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
                MapMaterialTile(m),
                Button,
                Node {
                    width:  Val::Px(TILE_SIZE_PX),
                    height: Val::Px(TILE_SIZE_PX),
                    border: UiRect::all(Val::Px(TILE_BORDER_PX)),
                    ..default()
                },
                BorderColor::all(TILE_BORDER_IDLE),
                BackgroundColor(m.ui_color()),
            ));
            col.spawn((
                Text::new(m.label()),
                TextFont { font_size: LABEL_FONT_PX, ..default() },
                TextColor(Color::WHITE),
                TextLayout::new_with_justify(Justify::Center),
                Node { width: Val::Px(LABEL_WIDTH_PX), ..default() },
                Pickable::IGNORE,
            ));
        });
}


// ── Click handler + selection sync ─────────────────────────────────────────────

pub fn handle_tile_clicks(
    mode:             Res<WindowMode>,
    mut interactions: Query<(&Interaction, &MapMaterialTile), Changed<Interaction>>,
    mut session:      ResMut<MapEditorSession>,
) {
    if *mode != WindowMode::MapEditor { return; }

    for (interaction, tile) in &mut interactions {
        if matches!(*interaction, Interaction::Pressed) {
            session.selected_material = Some(tile.0);
        }
    }
}

/// Highlight the selected tile and update hover borders.
pub fn sync_tile_borders(
    mode:         Res<WindowMode>,
    session:      Res<MapEditorSession>,
    interactions: Query<&Interaction, (With<MapMaterialTile>, Changed<Interaction>)>,
    mut tiles:    Query<(&Interaction, &MapMaterialTile, &mut BorderColor)>,
) {
    if *mode != WindowMode::MapEditor { return; }
    // Re-sync only when the selection changed or any tile's hover/press state changed.
    if !session.is_changed() && interactions.is_empty() { return; }
    let selected = session.selected_material;
    for (interaction, tile, mut border) in &mut tiles {
        let is_selected = selected == Some(tile.0);
        let target = match (is_selected, *interaction) {
            (true, _)                                            => TILE_BORDER_SELECTED,
            (false, Interaction::Hovered | Interaction::Pressed) => TILE_BORDER_HOVER,
            _                                                    => TILE_BORDER_IDLE,
        };
        *border = BorderColor::all(target);
    }
}
