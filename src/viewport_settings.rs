use bevy::prelude::*;
use bevy::mesh::Mesh3d;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};


pub struct ViewportSettingsPlugin;

impl Plugin for ViewportSettingsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ViewportSettings>()
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_systems(Update, toggle_advanced_viewport)
        .add_systems(Update, draw_orientation_gizmos)
        .add_systems(Startup, setup_fps_counter)
        .add_systems(Update, update_fps_text);

    }
}



#[derive(Resource, Default)]
pub struct ViewportSettings {
    pub show_advanced_viewport: bool,
}


pub fn toggle_advanced_viewport(
    keys: Res<ButtonInput<KeyCode>>,
    mut viewport_settings: ResMut<ViewportSettings>,
) {
    if keys.just_pressed(KeyCode::F3) {
        viewport_settings.show_advanced_viewport = !viewport_settings.show_advanced_viewport;
    }
}



#[derive(Component)]
pub struct FpsText {
    timer: Timer,
}

pub fn setup_fps_counter(mut commands: Commands, viewport_settings: Res<ViewportSettings>) {


    commands.spawn((
    Text::new("FPS: 0.0"),
    TextFont {
        font_size: 20.0,
        ..default()
    },
    TextColor(Color::WHITE),
    // the Layout/Node component (replaces the .with_style part)
    Node {
        position_type: PositionType::Absolute,
        top: Val::Px(5.0),
        left: Val::Px(5.0),
        ..default()
    },

    FpsText {
        // updating counter every 0.05 seconds
        timer: Timer::from_seconds(0.05, TimerMode::Repeating),
    },
    ));
}

pub fn update_fps_text(
    time: Res<Time>,
    diagnostics: Res<DiagnosticsStore>,
    viewport_settings: Res<ViewportSettings>,
    mut query: Query<(&mut Text, &mut FpsText, &mut Visibility)> // Query both for the timer
) {
    for (mut text, mut fps_marker, mut visibility) in &mut query {
            // Synchronize visibility with the resource
            if !viewport_settings.show_advanced_viewport {
                *visibility = Visibility::Hidden;
                return; // Skip logic to save performance
            } else {
                *visibility = Visibility::Visible;
            }

            // The rest of your timer and diagnostic logic...
            fps_marker.timer.tick(time.delta());
            if fps_marker.timer.just_finished() {
                if let Some(fps_diag) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) {
                    if let Some(fps_value) = fps_diag.smoothed() {
                        text.0 = format!("FPS: {:.1}", fps_value);
                    }
                }
            }
        }
}

pub fn draw_orientation_gizmos(
    query: Query<&Transform, With<Mesh3d>>,
    mut gizmos: Gizmos,
    viewport_settings: Res<ViewportSettings>,
) {

    if !viewport_settings.show_advanced_viewport {  
        return;
    }

    for transform in &query {
        let pos = transform.translation;
        let length = 100.0;

        // X axis - Red (right)
        gizmos.arrow(pos, pos + transform.right() * length, Color::srgb(1.0, 0.0, 0.0));
        // Y axis - Yellow (up)
        gizmos.arrow(pos, pos + transform.up() * length, Color::srgb(1.0, 1.0, 0.0));
        // Z axis - Blue (forward/back, where -Z goes forward)
        gizmos.arrow(pos, pos + transform.forward() * length, Color::srgb(0.0, 0.0, 1.0));
    }
}
