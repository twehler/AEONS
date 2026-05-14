// Frontend plugin — owns the entire UI: viewport render-target, layout
// composition, divider dragging, statistics panel (bottom), individuum
// navigator panel (right), in-world identifier labels, orientation
// gizmos. The actual panel contents live in their own files
// (`statistics_panel.rs`, `individuum_navigator.rs`) so the layout
// stays readable.
//
// Layout:
//
//   ┌─────────────────────────────────┬─┬───────────┐
//   │            3D viewport          │V│ individuum│
//   │       (flex_grow inside top-row)│ │ navigator │
//   │                                 │ │           │
//   ├─────────────────────────────────┴─┴───────────┤  ← horizontal divider
//   │                  statistics panel              │
//   └────────────────────────────────────────────────┘
//
// "V" is the vertical drag-handle between the viewport and the
// navigator panel. Dragging it resizes the navigator panel's width;
// dragging the horizontal divider resizes the statistics panel's
// height. Both panels share the top-row's height, so the navigator
// panel's height tracks the statistics panel's slider automatically.
// A transparent screen-overlay container (also added to the layout
// root) hosts the per-organism floating identifier labels.

use bevy::prelude::*;
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::camera::RenderTarget;
use bevy::render::render_resource::{Extent3d, TextureFormat};
use bevy::window::PrimaryWindow;

use crate::statistics_panel::{
    self, GraphState, OrganismCounts, StatisticsPanel,
};
use crate::individuum_navigator::{
    self, IndividuumNavigatorPlugin,
};
use crate::simulation_settings::{PlayerControlsActive, SimulationRunning, Smoothing, TimeSpeed};


// ── Tunables ─────────────────────────────────────────────────────────────────

/// Initial statistics-pane height as a fraction of the window's logical
/// height. Doubled from the previous 0.1 to make room for the lounge area
/// (10vh top space above the graph) plus the Start/Stop button.
const PANEL_INITIAL_FRACTION: f32 = 0.2;
/// Thickness of the draggable splitter between the top row and the
/// statistics panel (logical px).
const DIVIDER_HEIGHT: f32 = 6.0;
/// Lower bound on the statistics pane size while dragging (logical px).
const PANEL_MIN_PX:    f32 = 24.0;
/// Lower bound on the top row's size while dragging (logical px).
const VIEWPORT_MIN_PX: f32 = 50.0;

/// Shared background colour for both UI panels (statistics + simulation
/// settings). Re-exported so the panel modules don't duplicate the value.
pub const PANEL_BG_COLOR: Color = Color::srgb(0.15, 0.15, 0.15);


// ── Resources ────────────────────────────────────────────────────────────────

/// Toggleable visualisation flags. Currently only governs the orientation
/// gizmo overlay (F3).
#[derive(Resource, Default)]
pub struct ViewportSettings {
    pub show_advanced_viewport: bool,
}

/// Captured at DragStart so subsequent Drag events (which give cumulative
/// distance from the start point) can resolve to an absolute height.
#[derive(Resource, Default)]
struct DividerDragState {
    initial_panel_height: f32,
}

/// Owns the texture the 3D camera renders into and tracks its current
/// size so we only reallocate the GPU buffer when the viewport pane
/// actually changes pixel dimensions.
#[derive(Resource)]
struct ViewportRender {
    image:        Handle<Image>,
    bound:        bool,
    current_size: UVec2,
}


// ── Marker components ────────────────────────────────────────────────────────

/// Add this component to any entity you want orientation gizmos drawn for.
/// Intentionally NOT added to terrain chunks — only to cells or organisms.
#[derive(Component)]
pub struct ShowGizmo;

/// Marker on the off-screen image-node that displays the 3D camera's
/// render target. `pub` so the navigator's label-projection system can
/// query the node's screen-space rect to anchor floating labels.
#[derive(Component)]
pub struct ViewportImage;

#[derive(Component)]
struct PanelDivider;


// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct FrontendPlugin;

impl Plugin for FrontendPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<ViewportSettings>()
            .init_resource::<OrganismCounts>()
            .init_resource::<DividerDragState>()
            .init_resource::<SimulationRunning>()
            .init_resource::<PlayerControlsActive>()
            .init_resource::<Smoothing>()
            .init_resource::<TimeSpeed>()
            .init_resource::<crate::simulation_settings::MaxOrganisms>()
            .init_resource::<crate::simulation_settings::OrganismPoolSize>()
            .init_resource::<statistics_panel::TimeSpeedEditState>()
            .init_resource::<statistics_panel::MaxOrganismsEditState>()
            .init_resource::<statistics_panel::CullMessage>()
            .insert_resource(GraphState::new())
            .add_plugins(FrameTimeDiagnosticsPlugin::default())
            .add_plugins(IndividuumNavigatorPlugin)
            .add_systems(Startup, setup_panes)
            .add_systems(Update, (
                bind_main_camera_to_viewport,
                resize_render_target,
                toggle_advanced_viewport,
                statistics_panel::update_fps_text,
                statistics_panel::update_cell_count_text,
                statistics_panel::update_sim_timer_text,
                statistics_panel::track_organism_births,
                statistics_panel::track_organism_deaths,
                statistics_panel::update_counter_texts,
                statistics_panel::graph_tick,
                statistics_panel::graph_redraw,
                statistics_panel::handle_start_stop_button,
                statistics_panel::handle_save_button,
                statistics_panel::update_start_stop_label,
                statistics_panel::handle_time_speed_input,
                statistics_panel::update_time_speed_text,
                statistics_panel::apply_time_speed,
                apply_player_controls_state,
            ))
            // Split off the Max-Organisms + cull-notification systems into
            // a second `add_systems` call to stay under Bevy's tuple size
            // limit for variadic system configs.
            .add_systems(Update, (
                statistics_panel::handle_max_organisms_input,
                statistics_panel::update_max_organisms_text,
                statistics_panel::apply_max_organisms_cull,
                statistics_panel::update_cull_message,
            ));
        // Gizmos run only when the F3-toggled overlay is active. Putting
        // the check on the system registration (rather than inside the
        // body) means the scheduler skips the system entirely in the
        // common-off state, avoiding the per-frame query iteration over
        // thousands of `ShowGizmo` body-part children.
        app.add_systems(
            Update,
            draw_orientation_gizmos.run_if(
                |vs: Res<ViewportSettings>| vs.show_advanced_viewport
            ),
        );
    }
}


// ── Layout ───────────────────────────────────────────────────────────────────

fn setup_panes(
    mut commands: Commands,
    mut images:   ResMut<Assets<Image>>,
    mut graph:    ResMut<GraphState>,
    windows:      Query<&Window, With<PrimaryWindow>>,
) {
    let (logical_h, init_w, init_h) = match windows.single() {
        Ok(w) => (
            w.height(),
            w.physical_width().max(1),
            w.physical_height().max(1),
        ),
        Err(_) => (720.0, 1280, 720),
    };
    let initial_panel_logical = (logical_h * PANEL_INITIAL_FRACTION).round();

    // Render target the 3D camera will draw into. Initial dimensions
    // match the window; `resize_render_target` re-fits it once UI layout
    // produces a real ComputedNode size for the viewport pane.
    let image_handle = images.add(Image::new_target_texture(
        init_w, init_h,
        TextureFormat::Rgba8Unorm,
        Some(TextureFormat::Rgba8UnormSrgb),
    ));
    commands.insert_resource(ViewportRender {
        image:        image_handle.clone(),
        bound:        false,
        current_size: UVec2::new(0, 0),
    });

    // 2D camera owns the window output and is marked as the default UI
    // camera so UI propagation never picks the player's Camera3d (which
    // we redirect to the off-screen image below). Higher order so it
    // composites after the 3D pass that produces the viewport texture.
    commands.spawn((
        Camera2d,
        Camera { order: 1, ..default() },
        IsDefaultUiCamera,
    ));

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top:           Val::Px(0.0),
                left:          Val::Px(0.0),
                width:         Val::Percent(100.0),
                height:        Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            Pickable::IGNORE,
        ))
        .with_children(|root| {
            // Top row: viewport (flex-grow) + vertical divider +
            // navigator panel. Wrapping these in a single row container
            // means the horizontal divider below resizes the row as a
            // whole, so the navigator panel's height automatically
            // tracks the viewport's height.
            root.spawn(Node {
                width:          Val::Percent(100.0),
                flex_direction: FlexDirection::Row,
                flex_grow:      1.0,
                flex_basis:     Val::Px(0.0),
                min_height:     Val::Px(0.0),
                ..default()
            })
            .with_children(|top_row| {
                top_row.spawn((
                    ViewportImage,
                    ImageNode::new(image_handle),
                    Node {
                        height:    Val::Percent(100.0),
                        flex_grow: 1.0,
                        flex_basis: Val::Px(0.0),
                        min_width: Val::Px(0.0),
                        ..default()
                    },
                    // Explicit Pickable so the picking backend
                    // registers hits on this entity. Without it,
                    // bevy_picking's UI backend skips the bare
                    // ImageNode (it auto-registers nodes with
                    // `BackgroundColor` like the divider, but not
                    // plain image nodes). `should_block_lower: false`
                    // because nothing meaningful sits behind the
                    // viewport — the click just needs to reach this
                    // entity's observer.
                    Pickable {
                        should_block_lower: false,
                        is_hoverable:       true,
                    },
                ))
                .observe(viewport_click);

                individuum_navigator::spawn_vertical_divider(top_row);
                individuum_navigator::spawn_navigator_panel(top_row);
            });

            // Horizontal divider — drag handle. Drives the statistics
            // panel's height; the top row above auto-fills the rest.
            root.spawn((
                PanelDivider,
                Node {
                    width:       Val::Percent(100.0),
                    height:      Val::Px(DIVIDER_HEIGHT),
                    flex_shrink: 0.0,
                    ..default()
                },
                BackgroundColor(Color::srgb(0.35, 0.35, 0.35)),
            ))
            .observe(divider_drag_start)
            .observe(divider_drag);

            // Bottom row: statistics panel.
            statistics_panel::spawn_panel(
                root,
                &mut images,
                &mut graph,
                initial_panel_logical,
            );
            // Per-organism floating identifier labels are spawned by
            // `individuum_navigator::manage_label_lifecycle` directly
            // as children of the `ViewportImage` — no separate
            // overlay container needed.
        });
}


// ── Divider drag (resizes statistics panel height) ───────────────────────────

fn divider_drag_start(
    _ev:        On<Pointer<DragStart>>,
    panel_q:    Query<&ComputedNode, With<StatisticsPanel>>,
    mut state:  ResMut<DividerDragState>,
) {
    if let Ok(panel) = panel_q.single() {
        // ComputedNode reports physical pixels. Convert to logical so it
        // composes with Drag::distance (which is in logical pixels) and
        // the Val::Px height we'll write back.
        state.initial_panel_height = panel.size().y * panel.inverse_scale_factor;
    }
}

fn divider_drag(
    ev:           On<Pointer<Drag>>,
    state:        Res<DividerDragState>,
    windows:      Query<&Window, With<PrimaryWindow>>,
    mut panel_q:  Query<&mut Node, With<StatisticsPanel>>,
) {
    let Ok(window) = windows.single() else { return };
    let Ok(mut panel_node) = panel_q.single_mut() else { return };
    let max_h = (window.height() - DIVIDER_HEIGHT - VIEWPORT_MIN_PX).max(PANEL_MIN_PX);
    let new_h = (state.initial_panel_height - ev.distance.y).clamp(PANEL_MIN_PX, max_h);
    panel_node.height = Val::Px(new_h);
}


// ── Viewport render-target plumbing ──────────────────────────────────────────

/// One-shot: redirect the existing Camera3d (spawned by `player_plugin`)
/// to render into our off-screen image. Runs every Update until it
/// succeeds, then the `bound` flag short-circuits.
fn bind_main_camera_to_viewport(
    mut commands: Commands,
    mut viewport: ResMut<ViewportRender>,
    cameras:      Query<Entity, With<Camera3d>>,
) {
    if viewport.bound { return; }
    let Some(cam) = cameras.iter().next() else { return };
    commands
        .entity(cam)
        .insert(RenderTarget::Image(viewport.image.clone().into()));
    viewport.bound = true;
}

/// Match the GPU texture size to the viewport pane's actual layout size
/// so the camera renders at native resolution without stretch.
fn resize_render_target(
    mut images:    ResMut<Assets<Image>>,
    mut viewport:  ResMut<ViewportRender>,
    viewport_node: Query<&ComputedNode, With<ViewportImage>>,
) {
    let Ok(node) = viewport_node.single() else { return };
    let size = node.size();
    let new_w = size.x.max(1.0).round() as u32;
    let new_h = size.y.max(1.0).round() as u32;
    let new_size = UVec2::new(new_w, new_h);
    if new_size == viewport.current_size { return; }
    if let Some(image) = images.get_mut(&viewport.image) {
        image.resize(Extent3d {
            width:  new_w,
            height: new_h,
            depth_or_array_layers: 1,
        });
        viewport.current_size = new_size;
    }
}


// ── Viewport click → activate player controls ────────────────────────────────

/// Left-click inside the viewport TOGGLES player camera capture: a first
/// click engages WASD/mouse-look (cursor captured), a second click
/// releases them again (cursor visible). Esc still works as a one-way
/// release. Works regardless of whether the simulation is running or
/// paused — the player can fly around a frozen world to inspect it.
fn viewport_click(
    ev:                 On<Pointer<Click>>,
    mut player_active:  ResMut<PlayerControlsActive>,
) {
    // Only respond to the primary (left) mouse button so right-clicks /
    // middle-clicks pass through cleanly.
    if !matches!(ev.button, PointerButton::Primary) { return; }
    player_active.0 = !player_active.0;
}

/// Mirror `PlayerControlsActive` onto the OS cursor's grab state. Runs
/// every frame but only writes when the desired state differs from the
/// current state, so this is effectively free. On the active → inactive
/// transition the cursor is also recentred so the user lands on the
/// Pause/Stop button in the statistics panel rather than wherever the
/// pointer happened to be when they pressed Esc / clicked.
fn apply_player_controls_state(
    player_active: Res<PlayerControlsActive>,
    window_q:      Single<(&mut Window, &mut bevy::window::CursorOptions), With<PrimaryWindow>>,
) {
    let (mut window, mut cursor) = window_q.into_inner();
    let want_grab = player_active.0;
    let currently_grabbed = matches!(cursor.grab_mode, bevy::window::CursorGrabMode::Confined);
    if want_grab == currently_grabbed { return; }
    if want_grab {
        cursor.grab_mode = bevy::window::CursorGrabMode::Confined;
        cursor.visible   = false;
    } else {
        cursor.grab_mode = bevy::window::CursorGrabMode::None;
        cursor.visible   = true;
        let center = Vec2::new(window.width() / 2.0, window.height() / 2.0);
        window.set_cursor_position(Some(center));
    }
}


// ── Toggles and gizmos ───────────────────────────────────────────────────────

pub fn toggle_advanced_viewport(
    keys:                 Res<ButtonInput<KeyCode>>,
    mut viewport_settings: ResMut<ViewportSettings>,
) {
    if keys.just_pressed(KeyCode::F3) {
        viewport_settings.show_advanced_viewport = !viewport_settings.show_advanced_viewport;
    }
}

pub fn draw_orientation_gizmos(
    // Only queries entities explicitly tagged with `ShowGizmo` — terrain
    // chunks, UI, lights etc. are never included. The
    // `viewport_settings.show_advanced_viewport` toggle is enforced by
    // a `.run_if` on the system registration so this body only runs
    // when gizmos are wanted.
    query:      Query<&Transform, With<ShowGizmo>>,
    mut gizmos: Gizmos,
) {
    for transform in &query {
        let pos    = transform.translation;
        let length = 1.5; // Scaled to cell size (~1 unit), not terrain scale

        gizmos.arrow(pos, pos + transform.right()   * length, Color::srgb(1.0, 0.0, 0.0));
        gizmos.arrow(pos, pos + transform.up()      * length, Color::srgb(1.0, 1.0, 0.0));
        gizmos.arrow(pos, pos + transform.forward() * length, Color::srgb(0.0, 0.0, 1.0));
    }
}
