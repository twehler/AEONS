// Bottom UI strip — placement hint only (placement is species-driven).
// Kept at the original `BOTTOM_PANEL_HEIGHT_PX` so the camera UI-rect
// test and the side panels' bottom insets need no recalibration.

use bevy::prelude::*;

use crate::colony_editor::session::DraftOrganism;
use crate::colony_editor::layout::PANEL_BG_COLOR;


// ── Tunables ─────────────────────────────────────────────────────────────────

pub const BOTTOM_PANEL_HEIGHT_PX: f32 = 90.0;
const ROW_PADDING_PX:    f32 = 12.0;

const HINT_COLOR: Color = Color::srgb(0.65, 0.70, 0.55);


// ── Marker components ───────────────────────────────────────────────────────

#[derive(Component)]
pub struct CreationPanel;


// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct CreationPanelPlugin;

impl Plugin for CreationPanelPlugin {
    fn build(&self, _app: &mut App) {
        // No systems — purely informational. Plugin kept so the
        // registration site needn't change.
    }
}


// ── Spawning ────────────────────────────────────────────────────────────────

pub fn spawn(parent: &mut ChildSpawnerCommands, draft: DraftOrganism) {
    spawn_with_offset(parent, draft, 0.0);
}

/// Like `spawn`; `_draft` / `_top_offset_px` are inert here (this panel
/// anchors to the bottom) but kept for signature parity with the others.
pub fn spawn_with_offset(parent: &mut ChildSpawnerCommands, _draft: DraftOrganism, _top_offset_px: f32) {
    parent.spawn((
        CreationPanel,
        Node {
            position_type: PositionType::Absolute,
            left:   Val::Px(0.0),
            right:  Val::Px(0.0),
            bottom: Val::Px(0.0),
            height: Val::Px(BOTTOM_PANEL_HEIGHT_PX),
            flex_direction: FlexDirection::Row,
            align_items:    AlignItems::Center,
            justify_content: JustifyContent::FlexEnd,
            padding:        UiRect::all(Val::Px(ROW_PADDING_PX)),
            ..default()
        },
        BackgroundColor(PANEL_BG_COLOR),
    ))
    .with_children(|panel| {
        panel
            .spawn(Node {
                flex_direction: FlexDirection::Column,
                align_items:    AlignItems::FlexEnd,
                justify_content: JustifyContent::Center,
                ..default()
            })
            .with_children(|hint| {
                hint.spawn((
                    Text::new("Left-click on the map to place the selected species"),
                    TextFont { font_size: 14.0, ..default() },
                    TextColor(HINT_COLOR),
                    Pickable::IGNORE,
                ));
                hint.spawn((
                    Text::new("Right-click an organism to delete it"),
                    TextFont { font_size: 12.0, ..default() },
                    TextColor(Color::srgb(0.55, 0.55, 0.55)),
                    Pickable::IGNORE,
                ));
            });
    });
}
