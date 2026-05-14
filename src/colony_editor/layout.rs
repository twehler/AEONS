// UI layout root for the colony editor.
//
// Plain absolute-positioned panels overlaying the 3D viewport
// (which renders directly to the window — no off-screen image,
// no flex resizing). Three side panels: bottom (creation), right
// (inventory), left (tools).
//
// A 2D camera with a higher render order than the 3D camera owns
// the UI compositing pass.

use bevy::prelude::*;

use crate::colony_editor::session::DraftOrganism;
use crate::colony_editor::creation_panel;
use crate::colony_editor::inventory_panel;
use crate::colony_editor::tool_panel;


/// Shared background colour for both panels — slightly darker than
/// the simulation's `frontend::PANEL_BG_COLOR` so the editor reads
/// as a separate environment at a glance.
pub const PANEL_BG_COLOR: Color = Color::srgb(0.12, 0.12, 0.14);


pub struct EditorLayoutPlugin;

impl Plugin for EditorLayoutPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_ui);
    }
}

fn setup_ui(mut commands: Commands, draft: Local<DraftOrganism>) {
    // 2D camera owning the UI pass. Higher order so it composites
    // over the 3D camera's output.
    commands.spawn((
        Camera2d,
        Camera { order: 1, ..default() },
        IsDefaultUiCamera,
    ));

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top:    Val::Px(0.0),
                left:   Val::Px(0.0),
                width:  Val::Percent(100.0),
                height: Val::Percent(100.0),
                ..default()
            },
            // Crucially, we DON'T attach `Pickable::IGNORE` to the
            // root — but its children that aren't actual buttons do
            // ignore picking, so right-click reaches the viewport.
            Pickable::IGNORE,
        ))
        .with_children(|root| {
            tool_panel::spawn(root);
            inventory_panel::spawn(root);
            creation_panel::spawn(root, *draft);
        });
}
