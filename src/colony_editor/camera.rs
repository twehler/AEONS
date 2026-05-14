// Editor flycam + dual-purpose LMB.
//
// Controls (intentionally distinct from the simulation's player camera):
//   * WASD          — translate horizontally on the camera's local axes.
//   * Space / Shift — translate vertically (world up / down).
//   * LMB tap       — fires a `ViewportClick` message used by
//                     `placement.rs` to spawn a new organism at the
//                     ray-vs-heightmap hit point. A tap is "press
//                     and release without dragging more than
//                     `LOOK_DRAG_THRESHOLD_PX` cumulative pixels".
//   * LMB drag      — rotate (mouse motion drives yaw + pitch). The
//                     drag has to cross the threshold first; until
//                     it does we don't rotate, so a clean tap stays
//                     visually still.
//   * Mouse wheel   — adjust movement speed (×1.5 / ÷1.5).
//
// Compared with `player_plugin::PlayerPlugin`, the editor never grabs
// the OS cursor. The cursor stays visible and unconfined so the user
// can still drag UI sliders / click panel buttons without losing their
// pointer.

use bevy::prelude::*;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::window::PrimaryWindow;

use crate::colony_editor::session::EditorSession;
use crate::colony_editor::creation_panel::BOTTOM_PANEL_HEIGHT_PX;
use crate::colony_editor::inventory_panel::PANEL_WIDTH_PX;
use crate::colony_editor::tool_panel::TOOL_PANEL_WIDTH_PX;


// ── Tunables ─────────────────────────────────────────────────────────────────

const DEFAULT_MOVE_SPEED:  f32 = 25.0;
const MIN_MOVE_SPEED:      f32 = 1.0;
const MAX_MOVE_SPEED:      f32 = 400.0;
const MOUSE_SENSITIVITY:   f32 = 0.0015;
/// Pitch is clamped just shy of straight up/down so the look matrix
/// never goes degenerate.
const PITCH_LIMIT:         f32 = std::f32::consts::FRAC_PI_2 - 0.05;
/// Cumulative cursor motion (logical px) the user has to exceed
/// during a single LMB hold before we treat it as a "drag" (camera
/// look) instead of a "tap" (place organism). 5 px is loose enough
/// to forgive a shaky finger, tight enough that an intentional drag
/// crosses it within the first few mouse-deltas.
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


/// Tracks the current LMB-press's "tap or drag" status so we can
/// fire `ViewportClick` only on clean taps. Reset on every press
/// down; on release, queried by the click-emit logic.
#[derive(Resource, Default)]
struct LmbPressState {
    /// True while LMB is held AND the user hasn't yet exceeded the
    /// drag threshold. A press starts in this state; once enough
    /// motion accumulates it flips to `false` and we treat the
    /// remainder of the hold as a camera-look drag.
    tap_pending:    bool,
    /// Cumulative motion magnitude during the current LMB hold.
    drag_distance:  f32,
    /// Cursor position at the moment of LMB-down — used by the
    /// click handler so the placement raycast goes through the
    /// pixel where the click *started*, not the (possibly slightly
    /// shifted) pixel where the release happened.
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
        // 300-unit far plane mirrors the simulation's player camera —
        // keeps shadow cascades tight and culling cheap.
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

fn handle_mouse_look(
    mouse_buttons:    Res<ButtonInput<MouseButton>>,
    mut motion:       MessageReader<MouseMotion>,
    mut cam_q:        Query<(&mut Transform, &mut EditorCamera)>,
    mut press_state:  ResMut<LmbPressState>,
    mut click_writer: MessageWriter<ViewportClick>,
    windows:          Query<&Window, With<PrimaryWindow>>,
    session:          Res<EditorSession>,
) {
    // Drain motion events regardless of LMB so they never pile up.
    let total: Vec2 = motion.read().fold(Vec2::ZERO, |acc, ev| acc + ev.delta);

    // ── LMB-down: start a fresh press, captured tap-pending ─────
    if mouse_buttons.just_pressed(MouseButton::Left) {
        let window = windows.single().ok();
        let cursor = window.and_then(|w| w.cursor_position());

        // Suppress press tracking entirely when:
        //   * the unsaved-work modal is open (clicks must only land
        //     on Yes / No, never on the editor below), or
        //   * the cursor is over a UI panel (clicks there belong to
        //     the panel, not to world placement / camera-look).
        let suppress = session.show_exit_modal
            || match (cursor, window) {
                (Some(c), Some(w)) => cursor_over_ui_panel(c, w),
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

    // ── While LMB held, accumulate distance, possibly look ──────
    if mouse_buttons.pressed(MouseButton::Left) {
        if total != Vec2::ZERO {
            press_state.drag_distance += total.length();
            if press_state.tap_pending && press_state.drag_distance > LOOK_DRAG_THRESHOLD_PX {
                press_state.tap_pending = false;
            }
        }

        if !press_state.tap_pending {
            // We're in look-drag mode: rotate the camera by this
            // frame's delta.
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

    // ── LMB-up: if it was a clean tap, emit a click event ──────
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

fn handle_keyboard_move(
    keys:       Res<ButtonInput<KeyCode>>,
    time:       Res<Time>,
    mut cam_q:  Query<(&mut Transform, &EditorCamera)>,
    windows:    Query<&Window, With<PrimaryWindow>>,
) {
    // Suppress movement when the OS-level focus is gone — avoids
    // ghost movement when alt-tabbing.
    if !windows.iter().any(|w| w.focused) { return; }

    // Suppress movement while a Ctrl-modifier keyboard shortcut is
    // engaged (e.g. Ctrl+S to save, Ctrl+Z to undo). Otherwise the
    // user's S/Z key down would simultaneously drift the camera.
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

fn handle_wheel_speed(
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


/// True when `cursor` (window-logical px, top-left origin) falls
/// inside the right-side inventory panel or the bottom creation
/// panel. Both panels are absolutely positioned with fixed
/// dimensions, so the rects can be derived from the window size +
/// the two panel-width / -height constants — no UI query needed.
///
/// `pub` so `placement.rs`'s right-click delete handler can use the
/// same rect check (rather than re-deriving the rects).
pub fn cursor_over_ui_panel_for_test(cursor: Vec2, window: &Window) -> bool {
    cursor_over_ui_panel(cursor, window)
}

fn cursor_over_ui_panel(cursor: Vec2, window: &Window) -> bool {
    let w = window.width();
    let h = window.height();
    let in_inventory = cursor.x >= w - PANEL_WIDTH_PX
                    && cursor.y <= h - BOTTOM_PANEL_HEIGHT_PX;
    let in_tool      = cursor.x <= TOOL_PANEL_WIDTH_PX
                    && cursor.y <= h - BOTTOM_PANEL_HEIGHT_PX;
    let in_bottom    = cursor.y >= h - BOTTOM_PANEL_HEIGHT_PX;
    in_inventory || in_tool || in_bottom
}
