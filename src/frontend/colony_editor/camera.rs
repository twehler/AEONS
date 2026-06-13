// Editor flycam + dual-purpose LMB.
//
//   * WASD — horizontal translate; Space/Shift — vertical.
//   * LMB tap (release without dragging past `LOOK_DRAG_THRESHOLD_PX`)
//     fires a `ViewportClick` for `placement.rs`; LMB drag rotates.
//   * Mouse wheel — adjust move speed (×1.5 / ÷1.5).
//
// Unlike the player camera, the editor never grabs the OS cursor, so
// the user keeps a visible pointer for UI panels.

use bevy::prelude::*;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::window::PrimaryWindow;

use crate::colony_editor::session::EditorSession;
use crate::colony_editor::creation_panel::BOTTOM_PANEL_HEIGHT_PX;
use crate::colony_editor::inventory_panel::PANEL_WIDTH_PX;
use crate::colony_editor::species_panel::TOOL_PANEL_WIDTH_PX;

/// Optional resource: `.0` logical px reserved as a top strip so clicks
/// on the merged-mode mode-bar aren't viewport clicks. Absent in
/// standalone (top strip = 0).
#[derive(Resource, Default)]
pub struct CursorTopReservedPx(pub f32);


// ── Tunables ─────────────────────────────────────────────────────────────────

const DEFAULT_MOVE_SPEED:  f32 = 25.0;
const MIN_MOVE_SPEED:      f32 = 1.0;
const MAX_MOVE_SPEED:      f32 = 400.0;
const MOUSE_SENSITIVITY:   f32 = 0.0015;
/// Clamped shy of straight up/down so the look matrix stays non-degenerate.
const PITCH_LIMIT:         f32 = std::f32::consts::FRAC_PI_2 - 0.05;
/// Cumulative LMB-hold motion (logical px) above which a hold is a
/// drag (look) rather than a tap (place). Forgives a shaky finger.
const LOOK_DRAG_THRESHOLD_PX: f32 = 5.0;


// ── Components / resources ───────────────────────────────────────────────────

#[derive(Component)]
pub struct EditorCamera {
    pub yaw:        f32,
    pub pitch:      f32,
    pub move_speed: f32,
}

impl Default for EditorCamera {
    fn default() -> Self {
        Self { yaw: 0.0, pitch: -0.4, move_speed: DEFAULT_MOVE_SPEED }
    }
}


/// Tracks the current LMB-press's tap-vs-drag status; `ViewportClick`
/// fires only on a clean tap. Reset on each press-down.
#[derive(Resource, Default)]
pub struct LmbPressState {
    /// True until the drag threshold is exceeded, then flips to drag mode.
    tap_pending:    bool,
    /// Cumulative motion during the current LMB hold.
    drag_distance:  f32,
    /// Cursor at LMB-down; the placement raycast uses the press pixel,
    /// not the (possibly shifted) release pixel.
    press_cursor:   Option<Vec2>,
}


/// Fired by `handle_mouse_look` on a clean LMB tap. `placement.rs`
/// listens for these and spawns a fresh organism template.
#[derive(Message, Clone, Copy, Debug)]
pub struct ViewportClick {
    /// Cursor position at LMB-down, in window-logical pixels.
    pub cursor: Vec2,
}


// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct EditorCameraPlugin;

impl Plugin for EditorCameraPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<LmbPressState>()
            .add_message::<ViewportClick>()
            .add_systems(Startup, spawn_camera)
            .add_systems(Update, (
                handle_mouse_look,
                handle_keyboard_move,
                handle_wheel_speed,
            ));
    }
}

fn spawn_camera(mut commands: Commands) {
    let cam = EditorCamera::default();
    commands.spawn((
        Camera3d::default(),
        // Far plane mirrors the player camera (tight cascades, cheap culling).
        Projection::Perspective(PerspectiveProjection {
            fov:    std::f32::consts::FRAC_PI_3,
            near:   0.1,
            far:    400.0,
            aspect_ratio: 1.0,
            ..default()
        }),
        Transform::from_xyz(50.0, 40.0, 50.0)
            .looking_at(Vec3::new(80.0, 0.0, 80.0), Vec3::Y),
        cam,
    ));
}

pub fn handle_mouse_look(
    mouse_buttons:    Res<ButtonInput<MouseButton>>,
    mut motion:       MessageReader<MouseMotion>,
    mut cam_q:        Query<(&mut Transform, &mut EditorCamera)>,
    mut press_state:  ResMut<LmbPressState>,
    mut click_writer: MessageWriter<ViewportClick>,
    windows:          Query<&Window, With<PrimaryWindow>>,
    session:          Res<EditorSession>,
    top_reserved:     Option<Res<CursorTopReservedPx>>,
) {
    let top_strip = top_reserved.map(|r| r.0).unwrap_or(0.0);
    // Drain motion events regardless of LMB so they never pile up.
    let total: Vec2 = motion.read().fold(Vec2::ZERO, |acc, ev| acc + ev.delta);

    // LMB-down: start a fresh press.
    if mouse_buttons.just_pressed(MouseButton::Left) {
        let window = windows.single().ok();
        let cursor = window.and_then(|w| w.cursor_position());

        // Suppress press tracking when a modal is open or the cursor is
        // over a UI panel (those clicks belong to the panel/modal).
        let suppress = session.show_exit_modal
            || match (cursor, window) {
                (Some(c), Some(w)) => cursor_over_ui_panel_with_top(c, w, top_strip),
                _ => true,
            };

        if suppress {
            press_state.tap_pending   = false;
            press_state.drag_distance = 0.0;
            press_state.press_cursor  = None;
        } else {
            press_state.tap_pending   = true;
            press_state.drag_distance = 0.0;
            press_state.press_cursor  = cursor;
        }
    }

    // While LMB held: accumulate distance, possibly look.
    if mouse_buttons.pressed(MouseButton::Left) {
        if total != Vec2::ZERO {
            press_state.drag_distance += total.length();
            if press_state.tap_pending && press_state.drag_distance > LOOK_DRAG_THRESHOLD_PX {
                press_state.tap_pending = false;
            }
        }

        if !press_state.tap_pending {
            // Look-drag mode: rotate by this frame's delta.
            if total != Vec2::ZERO {
                if let Ok((mut tf, mut cam)) = cam_q.single_mut() {
                    cam.yaw   -= total.x * MOUSE_SENSITIVITY;
                    cam.pitch -= total.y * MOUSE_SENSITIVITY;
                    cam.pitch = cam.pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT);
                    tf.rotation = Quat::from_axis_angle(Vec3::Y, cam.yaw)
                                * Quat::from_axis_angle(Vec3::X, cam.pitch);
                }
            }
        }
    }

    // LMB-up: emit a click event if it was a clean tap.
    if mouse_buttons.just_released(MouseButton::Left) {
        if press_state.tap_pending {
            if let Some(cursor) = press_state.press_cursor {
                click_writer.write(ViewportClick { cursor });
            }
        }
        press_state.tap_pending   = false;
        press_state.drag_distance = 0.0;
        press_state.press_cursor  = None;
    }
}

pub fn handle_keyboard_move(
    keys:       Res<ButtonInput<KeyCode>>,
    time:       Res<Time>,
    mut cam_q:  Query<(&mut Transform, &EditorCamera)>,
    windows:    Query<&Window, With<PrimaryWindow>>,
) {
    // No movement when unfocused (avoids ghost drift on alt-tab).
    if !windows.iter().any(|w| w.focused) { return; }

    // No movement while Ctrl is held, so Ctrl+S/Ctrl+Z don't also
    // drift the camera via the S/Z keys.
    if keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight) {
        return;
    }

    let Ok((mut tf, cam)) = cam_q.single_mut() else { return };

    let mut dir = Vec3::ZERO;
    let forward = tf.forward();
    let right   = tf.right();
    if keys.pressed(KeyCode::KeyW)        { dir += *forward; }
    if keys.pressed(KeyCode::KeyS)        { dir -= *forward; }
    if keys.pressed(KeyCode::KeyD)        { dir += *right; }
    if keys.pressed(KeyCode::KeyA)        { dir -= *right; }
    if keys.pressed(KeyCode::Space)       { dir += Vec3::Y; }
    if keys.pressed(KeyCode::ShiftLeft)
       || keys.pressed(KeyCode::ShiftRight) { dir -= Vec3::Y; }

    if dir == Vec3::ZERO { return; }
    let dt = time.delta_secs();
    tf.translation += dir.normalize() * cam.move_speed * dt;
}

pub fn handle_wheel_speed(
    mut wheel:  MessageReader<MouseWheel>,
    mut cam_q:  Query<&mut EditorCamera>,
) {
    let mut delta_y = 0.0;
    for ev in wheel.read() { delta_y += ev.y; }
    if delta_y == 0.0 { return; }

    let Ok(mut cam) = cam_q.single_mut() else { return };
    let factor = if delta_y > 0.0 { 1.5_f32 } else { 1.0 / 1.5 };
    cam.move_speed = (cam.move_speed * factor).clamp(MIN_MOVE_SPEED, MAX_MOVE_SPEED);
}


/// True when `cursor` (window-logical px) is over a UI panel. Rects are
/// derived from window size + panel constants (no UI query needed).
/// `pub` so `placement.rs`'s right-click delete shares the same test.
pub fn cursor_over_ui_panel_for_test(cursor: Vec2, window: &Window) -> bool {
    cursor_over_ui_panel_with_top(cursor, window, 0.0)
}

/// Shared rect test for LMB capture and right-click delete. `top_strip_px`
/// reserves the merged-mode mode-bar band (0 in standalone).
pub fn cursor_over_ui_panel_with_top(cursor: Vec2, window: &Window, top_strip_px: f32) -> bool {
    let w = window.width();
    let h = window.height();
    let in_top       = top_strip_px > 0.0 && cursor.y <= top_strip_px;
    let in_inventory = cursor.x >= w - PANEL_WIDTH_PX
                    && cursor.y <= h - BOTTOM_PANEL_HEIGHT_PX
                    && cursor.y >= top_strip_px;
    let in_tool      = cursor.x <= TOOL_PANEL_WIDTH_PX
                    && cursor.y <= h - BOTTOM_PANEL_HEIGHT_PX
                    && cursor.y >= top_strip_px;
    let in_bottom    = cursor.y >= h - BOTTOM_PANEL_HEIGHT_PX;
    in_top || in_inventory || in_tool || in_bottom
}
