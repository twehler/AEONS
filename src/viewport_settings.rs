use bevy::prelude::*;
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

// Add this component to any entity you want orientation gizmos drawn for.
// Intentionally NOT added to terrain chunks — only to cells or organisms.
#[derive(Component)]
pub struct ShowGizmo;


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

pub fn setup_fps_counter(mut commands: Commands) {
    commands.spawn((
        Text::new("FPS: 0.0"),
        TextFont {
            font_size: 20.0,
            ..default()
        },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(5.0),
            left: Val::Px(5.0),
            ..default()
        },
        FpsText {
            timer: Timer::from_seconds(0.05, TimerMode::Repeating),
        },
    ));
}

pub fn update_fps_text(
    time: Res<Time>,
    diagnostics: Res<DiagnosticsStore>,
    viewport_settings: Res<ViewportSettings>,
    mut query: Query<(&mut Text, &mut FpsText, &mut Visibility)>,
) {
    for (mut text, mut fps_marker, mut visibility) in &mut query {
        if !viewport_settings.show_advanced_viewport {
            *visibility = Visibility::Hidden;
            return;
        } else {
            *visibility = Visibility::Visible;
        }

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
    // Only queries entities explicitly tagged with ShowGizmo —
    // terrain chunks, UI, lights etc. are never included.
    query: Query<&Transform, With<ShowGizmo>>,
    mut gizmos: Gizmos,
    viewport_settings: Res<ViewportSettings>,
) {
    if !viewport_settings.show_advanced_viewport {
        return;
    }

    for transform in &query {
        let pos = transform.translation;
        let length = 1.5; // Scaled to cell size (~1 unit), not terrain scale

        gizmos.arrow(pos, pos + transform.right()   * length, Color::srgb(1.0, 0.0, 0.0));
        gizmos.arrow(pos, pos + transform.up()      * length, Color::srgb(1.0, 1.0, 0.0));
        gizmos.arrow(pos, pos + transform.forward() * length, Color::srgb(0.0, 0.0, 1.0));
    }
}
