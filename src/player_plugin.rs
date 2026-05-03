use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

// Major credit goes to: https://github.com/sburris0/bevy_flycam

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
        // event.y is positive for scrolling "up" (away from user)
        // and negative for scrolling "down"
        if event.y > 0.0 {
            settings.speed *= 1.5;
        } else if event.y < 0.0 {
            // Ensure speed doesn't go below a reasonable minimum
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
    pub move_forward: KeyCode,
    pub move_backward: KeyCode,
    pub move_left: KeyCode,
    pub move_right: KeyCode,
    pub move_ascend: KeyCode,
    pub move_descend: KeyCode,
    pub toggle_pause: KeyCode, // Renamed for clarity
}

impl Default for KeyBindings {
    fn default() -> Self {
        Self {
            move_forward: KeyCode::KeyW,
            move_backward: KeyCode::KeyS,
            move_left: KeyCode::KeyA,
            move_right: KeyCode::KeyD,
            move_ascend: KeyCode::Space,
            move_descend: KeyCode::ShiftLeft,
            toggle_pause: KeyCode::Escape,
        }
    }
}

// A marker component used in queries when you want flycams and not other cameras
#[derive(Component)]
pub struct FlyCam;

// Helper to strictly grab the cursor for initialization
fn grab_cursor(mut primary_cursor_options: Single<&mut CursorOptions, With<PrimaryWindow>>) {
    primary_cursor_options.grab_mode = CursorGrabMode::Confined;
    primary_cursor_options.visible = false;
}

// Grabs the cursor when game first starts
fn initial_grab_cursor(primary_cursor_options: Single<&mut CursorOptions, With<PrimaryWindow>>) {
    grab_cursor(primary_cursor_options);
}

// Spawns the `Camera3dBundle` to be controlled
fn setup_player(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        FlyCam,
        Transform::from_xyz(20.0, 100.0, 20.0).looking_at(Vec3::ZERO, Vec3::Y),
        AmbientLight {
            color: Color::srgb(0.8, 0.8, 1.0),
            brightness: 500.0,
            ..default()
        },
    ));
}

// Handles keyboard input and movement
fn player_move(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time<Virtual>>,
    primary_cursor_options: Single<&mut CursorOptions, With<PrimaryWindow>>,
    settings: Res<MovementSettings>,
    key_bindings: Res<KeyBindings>,
    mut query: Query<(&FlyCam, &mut Transform)>, 
) {
    // Completely skip camera movement if the game is paused
    if time.is_paused() {
        return;
    }

    for (_camera, mut transform) in query.iter_mut() {
        let mut velocity = Vec3::ZERO;
        let local_z = transform.local_z();
        let forward = -Vec3::new(local_z.x, 0., local_z.z);
        let right = Vec3::new(local_z.z, 0., -local_z.x);

        for key in keys.get_pressed() {
            match primary_cursor_options.grab_mode {
                CursorGrabMode::None => (),
                _ => {
                    let key = *key;
                    if key == key_bindings.move_forward {
                        velocity += forward;
                    } else if key == key_bindings.move_backward {
                        velocity -= forward;
                    } else if key == key_bindings.move_left {
                        velocity -= right;
                    } else if key == key_bindings.move_right {
                        velocity += right;
                    } else if key == key_bindings.move_ascend {
                        velocity += Vec3::Y;
                    } else if key == key_bindings.move_descend {
                        velocity -= Vec3::Y;
                    }
                }
            }
        }

        velocity = velocity.normalize_or_zero();
        transform.translation += velocity * time.delta_secs() * settings.speed
    }
}

// Handles looking around if cursor is locked
fn player_look(
    settings: Res<MovementSettings>,
    primary_window: Query<&mut Window, With<PrimaryWindow>>,
    primary_cursor_options: Single<&mut CursorOptions, With<PrimaryWindow>>,
    mut state: MessageReader<MouseMotion>,
    mut query: Query<&mut Transform, With<FlyCam>>,
) {
    if let Ok(window) = primary_window.single() {
        for mut transform in query.iter_mut() {
            for ev in state.read() {
                let (mut yaw, mut pitch, _) = transform.rotation.to_euler(EulerRot::YXZ);
                
                // If cursor is freed (game paused), ignore mouse look events
                match primary_cursor_options.grab_mode {
                    CursorGrabMode::None => (),
                    _ => {
                        let window_scale = window.height().min(window.width());
                        pitch -= (settings.sensitivity * ev.delta.y * window_scale).to_radians();
                        yaw -= (settings.sensitivity * ev.delta.x * window_scale).to_radians();
                    }
                }

                pitch = pitch.clamp(-1.54, 1.54);

                transform.rotation =
                    Quat::from_axis_angle(Vec3::Y, yaw) * Quat::from_axis_angle(Vec3::X, pitch);
            }
        }
    } else {
        warn!("Primary window not found for `player_look`!");
    }
}

// ── NEW: Toggles Simulation Pause & Cursor Grab ──────────────────────────────
fn toggle_pause(
    keys: Res<ButtonInput<KeyCode>>,
    key_bindings: Res<KeyBindings>,
    mut primary_cursor_options: Single<&mut CursorOptions, With<PrimaryWindow>>,
    mut time: ResMut<Time<Virtual>>, // Controls the engine's main virtual clock
) {
    if keys.just_pressed(key_bindings.toggle_pause) {
        if time.is_paused() {
            // Resume Simulation & Capture Cursor
            time.unpause();
            primary_cursor_options.grab_mode = CursorGrabMode::Confined;
            primary_cursor_options.visible = false;
        } else {
            // Pause Simulation & Free Cursor
            time.pause();
            primary_cursor_options.grab_mode = CursorGrabMode::None;
            primary_cursor_options.visible = true;
        }
    }
}

// Grab cursor when an entity with FlyCam is added
fn initial_grab_on_flycam_spawn(
    query_added: Query<Entity, Added<FlyCam>>,
    primary_cursor_options: Single<&mut CursorOptions, With<PrimaryWindow>>,
) {
    if query_added.is_empty() {
        return;
    }
    grab_cursor(primary_cursor_options);
}

pub struct PlayerPlugin;
impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MovementSettings>()
            .init_resource::<KeyBindings>()
            .add_systems(Startup, setup_player)
            .add_systems(Startup, initial_grab_cursor)
            .add_systems(Startup, initial_grab_on_flycam_spawn)
            .add_systems(Update, player_move)
            .add_systems(Update, player_look)
            .add_systems(Update, toggle_pause)
            .add_systems(Update, change_speed_on_scroll);
    }
}
