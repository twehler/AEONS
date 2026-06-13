// UI layout root for the colony editor: absolute-positioned panels
// (bottom creation, right inventory, left species) over the 3D viewport.
// A higher-order 2D camera owns the UI compositing pass.

use bevy::prelude::*;

use crate::colony_editor::session::DraftOrganism;
use crate::colony_editor::creation_panel;
use crate::colony_editor::inventory_panel;
use crate::colony_editor::species_panel;


/// Panel background — darker than `frontend::PANEL_BG_COLOR` so the
/// editor reads as a separate environment.
pub const PANEL_BG_COLOR: Color = Color::srgb(0.12, 0.12, 0.14);


pub struct EditorLayoutPlugin;

impl Plugin for EditorLayoutPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_ui);
    }
}

fn setup_ui(mut commands: Commands, draft: Local<DraftOrganism>) {
    // 2D camera owning the UI pass; higher order composites over the 3D camera.
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
            // Non-button children ignore picking so right-click reaches the viewport.
            Pickable::IGNORE,
        ))
        .with_children(|root| {
            species_panel::spawn(root);
            inventory_panel::spawn(root);
            creation_panel::spawn(root, *draft);
        });
}
