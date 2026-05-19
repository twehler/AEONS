// Bottom UI strip — placement hint.
//
// The four trait cyclers (Metabolism / Intelligence / Symmetry /
// Form) that used to live here were retired when placement became
// species-driven: the user picks a `.species` file in the left-side
// Species Navigator and every spawn (left-click on the map or the
// Bulk-Spawn button) instantiates that species's full OCG + body
// plan. The cyclers no longer drove anything reachable from the UI,
// so they've been removed. This panel now only carries the placement
// hint text — kept at the original `BOTTOM_PANEL_HEIGHT_PX` so the
// camera's UI-rect test, the species panel's bottom inset, and the
// inventory panel's bottom inset don't need recalibration.

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
        // No systems — the panel is purely informational now. The
        // plugin still exists so the editor-mod registration site
        // doesn't need to change.
    }
}


// ── Spawning ────────────────────────────────────────────────────────────────

pub fn spawn(parent: &mut ChildSpawnerCommands, draft: DraftOrganism) {
    spawn_with_offset(parent, draft, 0.0);
}

/// Same as `spawn` but `top_offset_px` reserves vertical space at the
/// top of the screen — kept for parity with the other panel spawn
/// signatures even though this panel anchors to the bottom and the
/// parameter is structurally inert. `_draft` is also accepted (and
/// ignored) so reproduction-style callers don't need to drop the
/// argument.
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
