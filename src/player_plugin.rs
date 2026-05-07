use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::simulation_settings::PlayerControlsActive;

// Major credit goes to: https://github.com/sburris0/bevy_flycam
//
// Notes on the new control flow (May 2026):
//   * Cursor is NOT grabbed at startup. The frontend layer keeps it visible
//     so the player can interact with the UI.
//   * A left-click inside the viewport activates `PlayerControlsActive`
//     (handled in `frontend.rs`); from then on WASD / Space / Shift /
//     mouse-look run.
//   * Esc deactivates `PlayerControlsActive` (handled here). Simulation
//     keeps running; only the camera goes idle.
//   * Pausing/resuming the simulation is done via the Start/Stop button
//     in the statistics panel — no keyboard shortcut for it.
//
// Movement systems consume `Time<Real>` so the camera keeps responding
// when `Time<Virtual>` is paused (player should be able to look around
// and reposition while the world is frozen).


// Mouse sensitivity and movement speed
#[derive(Resource)]
pub struct MovementSettings {
    pub sensitivity: f32,
    pub speed: f32,
}

pub fn change_speed_on_scroll(
    mut mouse_wheel_events: MessageReader<MouseWheel>,
    mut settings: ResMut<MovementSettings>,
) {
    for event in mouse_wheel_events.read() {
        if event.y > 0.0 {
            settings.speed *= 1.5;
        } else if event.y < 0.0 {
            settings.speed /= 1.5;
        }
    }
}

impl Default for MovementSettings {
    fn default() -> Self {
        Self {
            sensitivity: 0.00005,
            speed: 8.0,
        }
    }
}

// Key configuration
#[derive(Resource)]
pub struct KeyBindings {
    pub move_forward:        KeyCode,
    pub move_backward:       KeyCode,
    pub move_left:           KeyCode,
    pub move_right:          KeyCode,
    pub move_ascend:         KeyCode,
    pub move_descend:        KeyCode,
    pub release_controls:    KeyCode,
}

impl Default for KeyBindings {
    fn default() -> Self {
        Self {
            move_forward:     KeyCode::KeyW,
            move_backward:    KeyCode::KeyS,
            move_left:        KeyCode::KeyA,
            move_right:       KeyCode::KeyD,
            move_ascend:      KeyCode::Space,
            move_descend:     KeyCode::ShiftLeft,
            release_controls: KeyCode::Escape,
        }
    }
}

// A marker component used in queries when you want flycams and not other cameras
#[derive(Component)]
pub struct FlyCam;


fn setup_player(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        // Tight far plane bounds both the visible mesh set and the
        // directional light's cascade-shadow ortho fit. Bevy 0.18's
        // default is 1000 units; at large maps that pulls a huge swath
        // of terrain into every shadow-pass draw and inflates extract-
        // phase work over many entities.
        Projection::Perspective(PerspectiveProjection {
            far: 300.0,
            near: 0.1,
            ..default()
        }),
        FlyCam,
        Transform::from_xyz(20.0, 100.0, 20.0).looking_at(Vec3::ZERO, Vec3::Y),
        AmbientLight {
            color: Color::srgb(0.8, 0.8, 1.0),
            brightness: 500.0,
            ..default()
        },
    ));
}


// Handles keyboard input and movement. Gated on `PlayerControlsActive`
// — when the player hasn't clicked into the viewport (or just pressed
// Esc), the camera stays still.
fn player_move(
    keys:          Res<ButtonInput<KeyCode>>,
    real_time:     Res<Time<Real>>,
    player_active: Res<PlayerControlsActive>,
    settings:      Res<MovementSettings>,
    key_bindings:  Res<KeyBindings>,
    mut query:     Query<(&FlyCam, &mut Transform)>,
) {
    if !player_active.0 { return; }

    for (_camera, mut transform) in query.iter_mut() {
        let mut velocity = Vec3::ZERO;
        let local_z = transform.local_z();
        let forward = -Vec3::new(local_z.x, 0., local_z.z);
        let right   =  Vec3::new(local_z.z, 0., -local_z.x);

        for key in keys.get_pressed() {
            let key = *key;
            if key == key_bindings.move_forward       { velocity += forward; }
            else if key == key_bindings.move_backward { velocity -= forward; }
            else if key == key_bindings.move_left     { velocity -= right;   }
            else if key == key_bindings.move_right    { velocity += right;   }
            else if key == key_bindings.move_ascend   { velocity += Vec3::Y; }
            else if key == key_bindings.move_descend  { velocity -= Vec3::Y; }
        }

        velocity = velocity.normalize_or_zero();
        transform.translation += velocity * real_time.delta_secs() * settings.speed;
    }
}


fn player_look(
    settings:        Res<MovementSettings>,
    primary_window:  Query<&Window, With<PrimaryWindow>>,
    player_active:   Res<PlayerControlsActive>,
    mut state:       MessageReader<MouseMotion>,
    mut query:       Query<&mut Transform, With<FlyCam>>,
) {
    if !player_active.0 {
        // Drain the queue anyway so accumulated motion doesn't snap the
        // camera the next time controls are activated.
        for _ in state.read() {}
        return;
    }
    let Ok(window) = primary_window.single() else {
        warn!("Primary window not found for `player_look`!");
        return;
    };
    for mut transform in query.iter_mut() {
        for ev in state.read() {
            let (mut yaw, mut pitch, _) = transform.rotation.to_euler(EulerRot::YXZ);
            let window_scale = window.height().min(window.width());
            pitch -= (settings.sensitivity * ev.delta.y * window_scale).to_radians();
            yaw   -= (settings.sensitivity * ev.delta.x * window_scale).to_radians();
            pitch = pitch.clamp(-1.54, 1.54);
            transform.rotation =
                Quat::from_axis_angle(Vec3::Y, yaw) * Quat::from_axis_angle(Vec3::X, pitch);
        }
    }
}


/// Esc releases player controls (cursor freed by `apply_player_controls_state`
/// in `frontend.rs`). The simulation continues running.
fn release_player_controls_on_esc(
    keys:              Res<ButtonInput<KeyCode>>,
    key_bindings:      Res<KeyBindings>,
    mut player_active: ResMut<PlayerControlsActive>,
) {
    if keys.just_pressed(key_bindings.release_controls) && player_active.0 {
        player_active.0 = false;
    }
}


pub struct PlayerPlugin;
impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MovementSettings>()
            .init_resource::<KeyBindings>()
            .add_systems(Startup, setup_player)
            .add_systems(Update, player_move)
            .add_systems(Update, player_look)
            .add_systems(Update, release_player_controls_on_esc)
            .add_systems(Update, change_speed_on_scroll);
    }
}
