// Species editor — orbit camera. Drives the shared camera while
// `WindowMode == SpeciesEditor`: middle-mouse drag orbits around the editor
// origin, scroll zooms. LMB is NOT consumed — it's used for cell placement in
// `placement.rs`.

use bevy::camera::visibility::RenderLayers;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::player_plugin::FlyCam;
use crate::simulation_settings::WindowMode;

use super::{SPECIES_EDITOR_ORIGIN, SPECIES_EDITOR_LAYER};


// ── Tunables ─────────────────────────────────────────────────────────────────

const INITIAL_RADIUS: f32 = 12.0;
const MIN_RADIUS:     f32 = 2.0;
const MAX_RADIUS:     f32 = 200.0;
const ZOOM_STEP:      f32 = 0.9;   // scroll-wheel zoom factor per notch
const ROTATE_SENS:    f32 = 0.006; // radians per device pixel
/// Pitch clamp — keep the camera within `[+epsilon, π-epsilon]` so it
/// never aligns exactly with the +Y axis (which would gimbal-lock the
/// yaw / pitch parameterisation).
const PITCH_MIN:      f32 = 0.10;
const PITCH_MAX:      f32 = std::f32::consts::PI - 0.10;
const INITIAL_YAW:    f32 = 0.7;
const INITIAL_PITCH:  f32 = 1.1;


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


// ── Systems ──────────────────────────────────────────────────────────────────

/// On any mode change, swap the camera's RenderLayers to the active mode's layer
/// (load-bearing for editor/simulation isolation) and, for SpeciesEditor, snap
/// to the orbit origin.
pub fn snap_camera_on_mode_entry(
    mode:         Res<WindowMode>,
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
            // restore — don't overwrite it with our orbit position).
            if stash.0.is_none() {
                stash.0 = Some(*transform);
            }
            commands.entity(entity).insert(RenderLayers::layer(SPECIES_EDITOR_LAYER));
            let orbit = match orbit_opt {
                Some(o) => OrbitCamera { yaw: o.yaw, pitch: o.pitch, radius: o.radius },
                None    => {
                    let o = OrbitCamera::default();
                    commands.entity(entity).insert(OrbitCamera::default());
                    o
                }
            };
            apply_orbit_to_transform(&orbit, &mut transform);
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

/// Orbit input: middle-mouse drag rotates, scroll wheel zooms. Uses real-time
/// delta so it responds while virtual time is paused (always, in this mode).
pub fn orbit_camera_input(
    mode:           Res<WindowMode>,
    mouse:          Res<ButtonInput<MouseButton>>,
    mut motion:     MessageReader<MouseMotion>,
    mut wheel:      MessageReader<MouseWheel>,
    mut cam_q:      Query<(&mut Transform, &mut OrbitCamera), With<FlyCam>>,
    window_q:       Query<&Window, With<PrimaryWindow>>,
) {
    if *mode != WindowMode::SpeciesEditor {
        // Drain the readers so a backlog doesn't fire on the next mode entry.
        for _ in motion.read() {}
        for _ in wheel.read()  {}
        return;
    }
    let _ = window_q; // window query reserved for future cursor-on-viewport gating

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
