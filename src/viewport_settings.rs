use bevy::prelude::*;
use bevy::mesh::Mesh3d;

#[derive(Resource, Default)]
pub struct GizmoSettings {
    pub show_orientation: bool,
}


pub fn toggle_orientation_gizmos(
    keys: Res<ButtonInput<KeyCode>>,
    mut gizmo_settings: ResMut<GizmoSettings>,
) {
    if keys.just_pressed(KeyCode::F3) {
        gizmo_settings.show_orientation = !gizmo_settings.show_orientation;
    }
}


pub fn draw_orientation_gizmos(
    query: Query<&Transform, With<Mesh3d>>,
    mut gizmos: Gizmos,
    gizmo_settings: Res<GizmoSettings>,
) {

    if !gizmo_settings.show_orientation {  // <-- add this check
        return;
    }

    for transform in &query {
        let pos = transform.translation;
        let length = 100.0;

        // X axis - Red (right)
        gizmos.arrow(pos, pos + transform.right() * length, Color::srgb(1.0, 0.0, 0.0));
        // Y axis - Yellow (up)
        gizmos.arrow(pos, pos + transform.up() * length, Color::srgb(1.0, 1.0, 0.0));
        // Z axis - Blue (forward/back)
        gizmos.arrow(pos, pos + transform.forward() * length, Color::srgb(0.0, 0.0, 1.0));
    }
}
