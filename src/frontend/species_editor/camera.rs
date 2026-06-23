// Species editor — camera (Free flycam + Orbit), selectable from the tool panel.
//
//   * Free  — the SHARED flycam (`player_plugin`): WASD travels exactly as in
//             the simulation; holding MIDDLE rotates. The species editor only
//             frames the organism on entry and swaps RenderLayers.
//   * Orbit — middle-mouse drag orbits around the editor origin, scroll zooms.
//             While orbiting, the shared flycam is suppressed via
//             `SpeciesEditorFlycam` so the two never fight over the transform.
//
// `sync_species_camera` flips the flycam gate from the active `CameraMode` and,
// on a switch into Orbit, seeds the orbit angles from the current pose so the
// view doesn't jump.

use bevy::camera::visibility::RenderLayers;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;

use crate::player_plugin::{FlyCam, SpeciesEditorFlycam};
use crate::simulation_settings::WindowMode;

use super::session::{CameraMode, SpeciesSession};
use super::{SPECIES_EDITOR_ORIGIN, SPECIES_EDITOR_LAYER};


// ── Tunables ─────────────────────────────────────────────────────────────────

/// Free-camera offset from `SPECIES_EDITOR_ORIGIN` on entry — back, up, and to
/// the side so the freshly-framed organism sits in view before the user flies.
const INITIAL_OFFSET: Vec3 = Vec3::new(8.0, 6.0, 12.0);

// Orbit camera.
const INITIAL_RADIUS: f32 = 12.0;
const MIN_RADIUS:     f32 = 2.0;
const MAX_RADIUS:     f32 = 200.0;
const ZOOM_STEP:      f32 = 0.9;   // scroll-wheel zoom factor per notch
const ROTATE_SENS:    f32 = 0.006; // radians per device pixel
/// Pitch clamp — keep the camera within `[+epsilon, π-epsilon]` so it never
/// aligns exactly with the +Y axis (which would gimbal-lock the yaw/pitch
/// parameterisation).
const PITCH_MIN:      f32 = 0.10;
const PITCH_MAX:      f32 = std::f32::consts::PI - 0.10;
const INITIAL_YAW:    f32 = 0.7;
const INITIAL_PITCH:  f32 = 1.1;

// Camera-movement toggle button.
const CAM_BTN_BG:    Color = Color::srgb(0.30, 0.30, 0.38);
const CAM_BTN_HOVER: Color = Color::srgb(0.40, 0.40, 0.50);


/// Per-camera orbit state. Yaw + pitch are spherical-coordinate angles around
/// `SPECIES_EDITOR_ORIGIN`.
#[derive(Component)]
pub struct OrbitCamera {
    pub yaw:    f32,
    pub pitch:  f32,
    pub radius: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self { yaw: INITIAL_YAW, pitch: INITIAL_PITCH, radius: INITIAL_RADIUS }
    }
}

/// Stash for the `FlyCam`'s simulation-mode transform while the species editor
/// is active. Without it, returning would leave the camera parked at
/// `SPECIES_EDITOR_ORIGIN` (~100k units away), outside the simulation world.
#[derive(Resource, Default)]
pub struct StashedSimCameraTransform(pub Option<Transform>);


// ── UI markers ─────────────────────────────────────────────────────────────

#[derive(Component)]
pub struct CameraModeButton;
#[derive(Component)]
pub struct CameraModeLabel;


// ── Mode-entry framing ─────────────────────────────────────────────────────

/// On any mode change, swap the camera's RenderLayers to the active mode's layer
/// (load-bearing for editor/simulation isolation) and, for SpeciesEditor, frame
/// the organism per the active `CameraMode`. Always ensures an `OrbitCamera`
/// component exists so a later Free→Orbit switch finds its state.
pub fn snap_camera_on_mode_entry(
    mode:         Res<WindowMode>,
    session:      Res<SpeciesSession>,
    mut commands: Commands,
    mut stash:    ResMut<StashedSimCameraTransform>,
    mut cam_q:    Query<(Entity, &mut Transform, Option<&OrbitCamera>), With<FlyCam>>,
) {
    if !mode.is_changed() { return; }

    let Ok((entity, mut transform, orbit_opt)) = cam_q.single_mut() else { return };

    match *mode {
        WindowMode::SpeciesEditor => {
            // Stash the pre-editor transform for restore on exit. Skip if the
            // stash is already populated (defensive against re-entry without
            // restore — don't overwrite it with our editor position).
            if stash.0.is_none() {
                stash.0 = Some(*transform);
            }
            commands.entity(entity).insert(RenderLayers::layer(SPECIES_EDITOR_LAYER));
            let orbit = match orbit_opt {
                Some(o) => OrbitCamera { yaw: o.yaw, pitch: o.pitch, radius: o.radius },
                None    => {
                    commands.entity(entity).insert(OrbitCamera::default());
                    OrbitCamera::default()
                }
            };
            match session.camera_mode {
                // Free: a fixed start pose; the flycam flies from here.
                CameraMode::Free => {
                    *transform = Transform::from_translation(SPECIES_EDITOR_ORIGIN + INITIAL_OFFSET)
                        .looking_at(SPECIES_EDITOR_ORIGIN, Vec3::Y);
                }
                CameraMode::Orbit => apply_orbit_to_transform(&orbit, &mut transform),
            }
        }
        // Any non-species mode: render the default world layer and restore the
        // stashed simulation-time transform.
        _ => {
            commands.entity(entity).insert(RenderLayers::layer(0));
            if let Some(saved) = stash.0.take() {
                *transform = saved;
            }
        }
    }
}


// ── Per-frame sync: flycam gate, smooth switch, button label ──────────────────

pub fn sync_species_camera(
    mode:        Res<WindowMode>,
    session:     Res<SpeciesSession>,
    mut flycam:  ResMut<SpeciesEditorFlycam>,
    mut last:    Local<Option<CameraMode>>,
    mut cam_q:   Query<(&mut Transform, &mut OrbitCamera), With<FlyCam>>,
    mut labels:  Query<&mut Text, With<CameraModeLabel>>,
) {
    let in_editor = *mode == WindowMode::SpeciesEditor;

    // The shared flycam owns the transform unless we're orbiting in the editor.
    let want_flycam = !(in_editor && session.camera_mode == CameraMode::Orbit);
    if flycam.0 != want_flycam { flycam.0 = want_flycam; }

    if !in_editor {
        *last = None;
        return;
    }

    let cur = session.camera_mode;

    // Keep the toggle button's label current.
    let label = camera_button_text(cur);
    for mut t in &mut labels {
        if t.0 != label { t.0 = label.clone(); }
    }

    // On a switch INTO orbit, seed the orbit angles from the current (flycam)
    // pose so the view continues smoothly instead of snapping to a default.
    if *last != Some(cur) {
        *last = Some(cur);
        if cur == CameraMode::Orbit {
            if let Ok((mut tf, mut orbit)) = cam_q.single_mut() {
                let offset = tf.translation - SPECIES_EDITOR_ORIGIN;
                let r = offset.length().max(MIN_RADIUS);
                orbit.radius = r.clamp(MIN_RADIUS, MAX_RADIUS);
                orbit.pitch  = (offset.y / r).clamp(-1.0, 1.0).acos().clamp(PITCH_MIN, PITCH_MAX);
                orbit.yaw    = offset.z.atan2(offset.x);
                apply_orbit_to_transform(&orbit, &mut tf);
            }
        }
    }
}


// ── Orbit input (active only in Orbit mode) ───────────────────────────────────

/// Orbit input: middle-mouse drag rotates, scroll wheel zooms. Uses real-time
/// motion so it responds while virtual time is paused (always, in this mode).
pub fn orbit_camera_input(
    mode:       Res<WindowMode>,
    session:    Res<SpeciesSession>,
    mouse:      Res<ButtonInput<MouseButton>>,
    mut motion: MessageReader<MouseMotion>,
    mut wheel:  MessageReader<MouseWheel>,
    mut cam_q:  Query<(&mut Transform, &mut OrbitCamera), With<FlyCam>>,
) {
    if *mode != WindowMode::SpeciesEditor || session.camera_mode != CameraMode::Orbit {
        // Drain the readers so a backlog doesn't fire on the next orbit entry.
        for _ in motion.read() {}
        for _ in wheel.read()  {}
        return;
    }

    let Ok((mut transform, mut orbit)) = cam_q.single_mut() else {
        for _ in motion.read() {}
        for _ in wheel.read()  {}
        return;
    };

    let mut changed = false;

    if mouse.pressed(MouseButton::Middle) {
        let mut dx = 0.0;
        let mut dy = 0.0;
        for ev in motion.read() {
            dx += ev.delta.x;
            dy += ev.delta.y;
        }
        if dx != 0.0 || dy != 0.0 {
            orbit.yaw   -= dx * ROTATE_SENS;
            orbit.pitch += dy * ROTATE_SENS;
            orbit.pitch = orbit.pitch.clamp(PITCH_MIN, PITCH_MAX);
            changed = true;
        }
    } else {
        // Middle button not held — drain motion so no backlog fires on next press.
        for _ in motion.read() {}
    }

    let mut wheel_delta = 0.0;
    for ev in wheel.read() {
        wheel_delta += ev.y;
    }
    if wheel_delta != 0.0 {
        // Positive wheel.y == scroll forward == zoom in == smaller radius.
        let factor = ZOOM_STEP.powf(wheel_delta);
        orbit.radius = (orbit.radius * factor).clamp(MIN_RADIUS, MAX_RADIUS);
        changed = true;
    }

    if changed {
        apply_orbit_to_transform(&orbit, &mut transform);
    }
}

/// Convert spherical orbit angles into a `Transform` looking at
/// `SPECIES_EDITOR_ORIGIN`.
fn apply_orbit_to_transform(orbit: &OrbitCamera, transform: &mut Transform) {
    let target = SPECIES_EDITOR_ORIGIN;
    let sp     = orbit.pitch.sin();
    let cp     = orbit.pitch.cos();
    let sy     = orbit.yaw.sin();
    let cy     = orbit.yaw.cos();
    // Standard spherical → cartesian with Y up:
    //   x = r·sin(pitch)·cos(yaw)
    //   y = r·cos(pitch)
    //   z = r·sin(pitch)·sin(yaw)
    let offset = Vec3::new(orbit.radius * sp * cy,
                           orbit.radius * cp,
                           orbit.radius * sp * sy);
    transform.translation = target + offset;
    transform.look_at(target, Vec3::Y);
}


// ── Camera-movement toggle button ─────────────────────────────────────────────

fn camera_button_text(mode: CameraMode) -> String {
    format!("Camera Movement: {}", mode.label())
}

/// Spawn the "Camera Movement: <mode>" toggle (called from the tool-panel build).
pub fn spawn_camera_mode_button(panel: &mut ChildSpawnerCommands) {
    panel
        .spawn((
            CameraModeButton,
            Button,
            Node {
                width:           Val::Percent(100.0),
                height:          Val::Px(30.0),
                align_items:     AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            BackgroundColor(CAM_BTN_BG),
        ))
        .with_children(|b| {
            b.spawn((
                CameraModeLabel,
                // Matches the default `CameraMode::Free`; `sync_species_camera`
                // keeps it current thereafter.
                Text::new(camera_button_text(CameraMode::Free)),
                TextFont { font_size: 13.0, ..default() },
                TextColor(Color::WHITE),
                Pickable::IGNORE,
            ));
        });
}

/// Toggle Free ↔ Orbit on click (the label is refreshed by `sync_species_camera`).
pub fn handle_camera_mode_button(
    mode:        Res<WindowMode>,
    mut session: ResMut<SpeciesSession>,
    mut buttons: Query<(&Interaction, &mut BackgroundColor),
                       (Changed<Interaction>, With<CameraModeButton>)>,
) {
    if *mode != WindowMode::SpeciesEditor { return; }
    for (interaction, mut bg) in &mut buttons {
        match *interaction {
            Interaction::Pressed => {
                session.camera_mode = session.camera_mode.toggle();
                *bg = BackgroundColor(CAM_BTN_HOVER);
            }
            Interaction::Hovered => *bg = BackgroundColor(CAM_BTN_HOVER),
            Interaction::None    => *bg = BackgroundColor(CAM_BTN_BG),
        }
    }
}
