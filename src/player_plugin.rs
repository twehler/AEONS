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


/// Look-sensitivity multiplier applied in the editor modes (`EditColony` /
/// `SpeciesEditor`) on top of `MovementSettings::sensitivity`; the simulation
/// flycam uses the base value (×1).
pub const EDITOR_LOOK_SENSITIVITY_MULT: f32 = 3.0;

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


/// True while holding the MIDDLE mouse button after pressing over the viewport
/// image in an editor mode (`EditColony` / `SpeciesEditor`) — drives
/// "hold-middle-to-rotate" in `player_look`. Set by the viewport's
/// `Pointer<Press>` observer (frontend.rs), cleared by
/// `release_editor_look_on_mmb_up`. Unused in Simulation mode.
#[derive(Resource, Default)]
pub struct EditorLookActive(pub bool);

/// Gates the shared flycam in `WindowMode::SpeciesEditor`: `true` = the flycam
/// drives WASD/look (Free camera mode); `false` = the species editor's orbit
/// camera owns the transform instead. Set each frame by
/// `species_editor::camera::sync_species_camera`. Only consulted in the
/// SpeciesEditor arm, so its value is irrelevant in other modes.
#[derive(Resource)]
pub struct SpeciesEditorFlycam(pub bool);

impl Default for SpeciesEditorFlycam {
    fn default() -> Self { Self(true) }
}


fn setup_player(
    mut commands: Commands,
    water:        Res<crate::environment::WaterLevel>,
) {
    use std::f32::consts::{PI, FRAC_PI_4};
    // Spawn 50 units above the water surface, yawed 180° about Y and pitched
    // 45° downward. Built in the same (yaw·Y) * (pitch·X) convention
    // `player_look` reconstructs from, so the first mouse-look won't snap.
    // (`WaterLevel` is inserted at app-build time, so it's present here at
    // Startup; a `.colony` that overrides the level applies that override later,
    // during its async load, so this uses the launcher/arg/default level.)
    let cam_y = water.0 + 50.0;
    let rotation =
        Quat::from_axis_angle(Vec3::Y, PI) * Quat::from_axis_angle(Vec3::X, -FRAC_PI_4);
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
        Transform::from_xyz(20.0, cam_y, 20.0).with_rotation(rotation),
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
    species_flycam: Res<SpeciesEditorFlycam>,
    mut query:     Query<(&FlyCam, &mut Transform)>,
) {
    let active = match *window_mode {
        WindowMode::Simulation    => player_active.0,
        // Both editors fly the camera with WASD at all times (cursor stays free
        // for UI/placement); middle-hold adds rotation via `player_look`.
        WindowMode::EditColony    => true,
        // …except the species editor can switch to an orbit camera, which
        // suppresses the flycam (see `SpeciesEditorFlycam`).
        WindowMode::SpeciesEditor => species_flycam.0,
        // Lineages mode hides the viewport entirely (tree view
        // covers it) — no point letting WASD drive the camera.
        WindowMode::Lineages      => false,
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
    species_flycam:  Res<SpeciesEditorFlycam>,
    mut state:       MessageReader<MouseMotion>,
    mut query:       Query<&mut Transform, With<FlyCam>>,
) {
    // Rotation gated on the captured-cursor flag (Simulation) or the
    // hold-middle-mouse state (both editor modes); the species editor also
    // suppresses it while the orbit camera is active.
    let active = match *window_mode {
        WindowMode::Simulation    => player_active.0,
        WindowMode::EditColony    => editor_look.0,
        WindowMode::SpeciesEditor => editor_look.0 && species_flycam.0,
        WindowMode::Lineages      => false,
    };
    // Editors get a higher look sensitivity than the simulation flycam.
    let sens_mult = match *window_mode {
        WindowMode::EditColony | WindowMode::SpeciesEditor => EDITOR_LOOK_SENSITIVITY_MULT,
        _ => 1.0,
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
            let sens = settings.sensitivity * sens_mult;
            pitch -= (sens * ev.delta.y * window_scale).to_radians();
            yaw   -= (sens * ev.delta.x * window_scale).to_radians();
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


/// Clears `EditorLookActive` on MIDDLE-mouse release, regardless of cursor
/// location (so a drag that wanders over a panel still ends cleanly).
pub fn release_editor_look_on_mmb_up(
    mouse:           Res<ButtonInput<MouseButton>>,
    mut editor_look: ResMut<EditorLookActive>,
) {
    if mouse.just_released(MouseButton::Middle) && editor_look.0 {
        editor_look.0 = false;
    }
}


pub struct PlayerPlugin;
impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MovementSettings>()
            .init_resource::<KeyBindings>()
            .init_resource::<EditorLookActive>()
            .init_resource::<SpeciesEditorFlycam>()
            .add_systems(Startup, setup_player)
            .add_systems(Update, player_move)
            .add_systems(Update, player_look)
            .add_systems(Update, toggle_player_controls_on_esc)
            .add_systems(Update, engage_flying_on_space)
            .add_systems(Update, release_editor_look_on_mmb_up)
            .add_systems(Update, change_speed_on_scroll);
    }
}
