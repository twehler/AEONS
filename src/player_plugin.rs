use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::simulation_settings::{PlayerControlsActive, WindowMode};

// Credit: https://github.com/sburris0/bevy_flycam
//
// Control flow:
//   * Cursor is NOT grabbed at startup (frontend keeps it visible for UI).
//   * Esc TOGGLES `PlayerControlsActive` (Simulation mode only): engages
//     WASD/Space/Shift/mouse-look + cursor capture, then releases.
//   * Pause/resume is the statistics-panel Start/Stop button — no key.
//
// Movement systems use `Time<Real>` so the camera responds while
// `Time<Virtual>` is paused.


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


/// True while holding LMB after pressing over the viewport image in
/// `WindowMode::EditColony` — drives "hold-LMB-to-rotate" in `player_look`.
/// Set by the viewport's `Pointer<Pressed>` observer (frontend.rs), cleared
/// by `release_editor_look_on_lmb_up`. Unused in Simulation mode.
#[derive(Resource, Default)]
pub struct EditorLookActive(pub bool);


fn setup_player(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        // Tight far plane bounds both the visible mesh set and the cascade-
        // shadow ortho fit; Bevy's 1000-unit default inflates shadow/extract
        // cost on large maps.
        Projection::Perspective(PerspectiveProjection {
            far: 300.0,
            near: 0.1,
            ..default()
        }),
        FlyCam,
        // Spawn at altitude 100 looking down-away from the origin.
        Transform::from_xyz(20.0, 100.0, 20.0).looking_at(Vec3::new(40.0, 0.0, 40.0), Vec3::Y),
        AmbientLight {
            color: Color::srgb(0.8, 0.8, 1.0),
            brightness: 500.0,
            ..default()
        },
    ));
}


// Keyboard movement. Active in Simulation mode while `PlayerControlsActive`
// is on; in EditColony mode the camera is always WASD-controllable.
fn player_move(
    keys:          Res<ButtonInput<KeyCode>>,
    real_time:     Res<Time<Real>>,
    player_active: Res<PlayerControlsActive>,
    window_mode:   Res<WindowMode>,
    settings:      Res<MovementSettings>,
    key_bindings:  Res<KeyBindings>,
    mut query:     Query<(&FlyCam, &mut Transform)>,
) {
    let active = match *window_mode {
        WindowMode::Simulation    => player_active.0,
        WindowMode::EditColony    => true,
        // Lineages mode hides the viewport entirely (tree view
        // covers it) — no point letting WASD drive the camera.
        WindowMode::Lineages      => false,
        // Species editor uses its own orbit camera system that
        // controls the same FlyCam transform; the flycam itself
        // stays inactive.
        WindowMode::SpeciesEditor => false,
    };
    if !active { return; }

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
    window_mode:     Res<WindowMode>,
    editor_look:     Res<EditorLookActive>,
    mut state:       MessageReader<MouseMotion>,
    mut query:       Query<&mut Transform, With<FlyCam>>,
) {
    // Rotation gated on the captured-cursor flag (Simulation) or the
    // hold-LMB state (EditColony).
    let active = match *window_mode {
        WindowMode::Simulation    => player_active.0,
        WindowMode::EditColony    => editor_look.0,
        WindowMode::Lineages      => false,
        WindowMode::SpeciesEditor => false,
    };
    if !active {
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


/// Esc toggles player camera control (WASD/mouse-look + cursor capture).
/// Gated to `WindowMode::Simulation`: editor modes need the cursor free for
/// pointer placement, so Esc must not grab it there.
fn toggle_player_controls_on_esc(
    keys:              Res<ButtonInput<KeyCode>>,
    key_bindings:      Res<KeyBindings>,
    window_mode:       Res<WindowMode>,
    mut player_active: ResMut<PlayerControlsActive>,
) {
    if *window_mode != WindowMode::Simulation { return; }
    if keys.just_pressed(key_bindings.release_controls) {
        player_active.0 = !player_active.0;
    }
}

/// Space engages camera control from a standstill (Simulation mode, controls
/// OFF). While already flying, Space keeps its "ascend" role in `player_move`
/// — no conflict since this only fires when controls are off.
fn engage_flying_on_space(
    keys:              Res<ButtonInput<KeyCode>>,
    window_mode:       Res<WindowMode>,
    mut player_active: ResMut<PlayerControlsActive>,
) {
    if *window_mode != WindowMode::Simulation { return; }
    if !player_active.0 && keys.just_pressed(KeyCode::Space) {
        player_active.0 = true;
    }
}


/// Clears `EditorLookActive` on LMB release, regardless of cursor location
/// (so a drag that wanders over a panel still ends cleanly).
pub fn release_editor_look_on_lmb_up(
    mouse:           Res<ButtonInput<MouseButton>>,
    mut editor_look: ResMut<EditorLookActive>,
) {
    if mouse.just_released(MouseButton::Left) && editor_look.0 {
        editor_look.0 = false;
    }
}


pub struct PlayerPlugin;
impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MovementSettings>()
            .init_resource::<KeyBindings>()
            .init_resource::<EditorLookActive>()
            .add_systems(Startup, setup_player)
            .add_systems(Update, player_move)
            .add_systems(Update, player_look)
            .add_systems(Update, toggle_player_controls_on_esc)
            .add_systems(Update, engage_flying_on_space)
            .add_systems(Update, release_editor_look_on_lmb_up)
            .add_systems(Update, change_speed_on_scroll);
    }
}
