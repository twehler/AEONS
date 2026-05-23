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
    self, IndividuumNavigatorPlugin, NavigatorPanel,
};
use crate::simulation_settings::{
    PlayerControlsActive, SimulationRunning, Smoothing, TimeSpeed, WindowMode,
};
use crate::colony_editor::{self, EditorOverlayPlugin, EditorOverlayPanel};


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

/// Thin top bar that hosts the WindowMode buttons (Simulation Mode /
/// Edit Colony). Sits above every other panel — never hidden — so the
/// user always has a way to switch back out of whichever mode they're
/// in. `pub` so the editor overlay can reserve a matching gap at the
/// top of the screen (used as `top_offset_px` when spawning editor
/// panels and as the `CursorTopReservedPx` resource value).
pub const TOP_BAR_HEIGHT_PX: f32 = 36.0;


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

/// Marker on the top mode-switcher bar — never hidden, but tagged so
/// other systems can size around it (e.g. for cursor-hit rect tests).
#[derive(Component)]
pub struct TopModeBar;

/// Marker on each mode-switch button. The variant tells us which
/// `WindowMode` to switch to on click.
#[derive(Component, Clone, Copy)]
pub struct ModeSwitchButton(pub WindowMode);

/// Marker on the wrapping row that hosts the (viewport + vertical
/// divider + navigator) trio. We toggle this row's `Display` along
/// with the navigator panel itself so the row's flex layout doesn't
/// leave a phantom rule when the navigator is hidden.
#[derive(Component)]
struct ViewportRow;


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
            .init_resource::<WindowMode>()
            .init_resource::<statistics_panel::TimeSpeedEditState>()
            .init_resource::<statistics_panel::MaxOrganismsEditState>()
            .init_resource::<statistics_panel::MaxHerbivoresEditState>()
            .init_resource::<statistics_panel::CullMessage>()
            .init_resource::<crate::simulation_settings::AiTrainingMode>()
            .init_resource::<crate::simulation_settings::MaxHerbivores>()
            .insert_resource(GraphState::new())
            .add_plugins(FrameTimeDiagnosticsPlugin::default())
            // Lineages tree-view pan/zoom state + layout cache.
            // Render-target image is created inside `setup_panes`
            // (needs `Assets<Image>`) and inserted as
            // `LineagesViewportRender` after the layout root is
            // built; `setup_lineages_camera` consumes it on the
            // following Startup tick.
            .add_systems(Startup, crate::tree_view::setup_lineages_camera.after(setup_panes));
        crate::tree_view::init_resources(app);
        app
            .add_plugins(IndividuumNavigatorPlugin)
            .add_plugins(crate::species_navigator::SpeciesNavigatorPlugin)
            // Editor lives behind the EditColony WindowMode. The
            // overlay plugin (re)uses the existing 3D camera +
            // lighting from the simulation; it only contributes
            // editor UI + the editor's input handlers (the latter
            // gated on EditColony at the system level).
            .add_plugins(EditorOverlayPlugin)
            .add_systems(Update, (
                handle_mode_switch_buttons,
                apply_mode_transition,
                // Tree-of-life renderer lives in `tree_view.rs`
                // (frontend folder); `LineagesPlugin` owns the
                // `SpeciesRegistry`, the renderer turns it into a
                // pan/zoom 2D scene.
                crate::tree_view::resize_lineages_target,
                crate::tree_view::rebuild_tree_layout,
                crate::tree_view::rebuild_tree_visuals,
                crate::tree_view::lineages_zoom_wheel,
                crate::tree_view::apply_lineages_camera_state,
                crate::tree_view::apply_label_lod,
            ))
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
                statistics_panel::handle_ai_training_checkbox,
                statistics_panel::update_ai_training_checkbox_mark,
                statistics_panel::handle_max_herbivores_input,
                statistics_panel::update_max_herbivores_text,
                statistics_panel::handle_export_dataset_button,
                // TEMPORARY: log the breakdown of "what's a
                // Photoautotroph" buckets every 5 s, to track down
                // why the displayed count diverges from the visual
                // population. Remove once the bug is found.
                statistics_panel::diag_photo_breakdown,
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

    // Lineages render-target image is created inside
    // `spawn_lineages_panel` below; we capture its handle here so we
    // can insert the matching `LineagesViewportRender` resource
    // after `with_children` returns (commands can't be enqueued
    // from inside the child-spawner closure).
    let mut spawned_lineages_image: Option<Handle<Image>> = None;

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
            // ── Top mode-switcher bar ─────────────────────────────────
            spawn_top_mode_bar(root);

            // Top row: viewport (flex-grow) + vertical divider +
            // navigator panel. Wrapping these in a single row container
            // means the horizontal divider below resizes the row as a
            // whole, so the navigator panel's height automatically
            // tracks the viewport's height.
            root.spawn((
                ViewportRow,
                Node {
                    width:          Val::Percent(100.0),
                    flex_direction: FlexDirection::Row,
                    flex_grow:      1.0,
                    flex_basis:     Val::Px(0.0),
                    min_height:     Val::Px(0.0),
                    ..default()
                },
            ))
            .with_children(|top_row| {
                // Left-side species navigator — flex sibling of the
                // viewport image so it shares the row's vertical
                // extent and ends exactly at the horizontal divider
                // above the statistics panel. Spawned BEFORE the
                // viewport so flex-row places it on the left.
                crate::species_navigator::spawn_species_navigator(top_row);

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
                .observe(viewport_click)
                .observe(viewport_press);

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

            // ── Editor overlay panels ────────────────────────────────
            // Children of the root so their absolute positioning (e.g.
            // `bottom: 0`, `right: 0`) resolves against the window. We
            // spawn them up-front and tag them with both their natural
            // panel marker (CreationPanel, ToolPanel, InventoryPanel)
            // and an `EditorOverlayPanel` marker so the mode-transition
            // system can flip every editor panel's `Display` in one
            // query. Initial state is `Display::None` because we boot
            // in `WindowMode::Simulation`.
            colony_editor::spawn_overlay_panels(root, TOP_BAR_HEIGHT_PX);

            // ── Species editor panels ────────────────────────────────
            // Top + bottom panels for the manual species-construction
            // mode. Both spawn with `Display::None`; the mode-transition
            // system flips them to `Flex` when WindowMode::SpeciesEditor.
            crate::species_editor::spawn_overlay_panels(root, TOP_BAR_HEIGHT_PX);

            // (The simulation-mode species navigator is now spawned
            // inside the ViewportRow above so it shares the row's
            // vertical extent — see the `top_row.with_children`
            // block.)

            // Lineages mode panel — full-content-area tree-of-life
            // view. Spawned hidden; the mode-transition system
            // toggles its `Display` like the editor panels.
            let lineages_image = crate::tree_view::spawn_lineages_panel(
                root, TOP_BAR_HEIGHT_PX, &mut images,
            );
            // Stash the render-target handle as a resource that
            // `setup_lineages_camera` consumes once Startup
            // finishes. Inserting via Commands inside `with_children`
            // is awkward, so we use the closure's captured `commands`
            // queue at the outer call site below (after this
            // `with_children` block ends).
            spawned_lineages_image = Some(lineages_image);
        });
    if let Some(handle) = spawned_lineages_image {
        commands.insert_resource(crate::tree_view::LineagesViewportRender {
            image:        handle,
            current_size: UVec2::ZERO,
        });
    }
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
    window_mode:        Res<WindowMode>,
    windows:            Query<&Window, With<PrimaryWindow>>,
    mut click_writer:   MessageWriter<crate::camera::ViewportClick>,
) {
    // Only respond to the primary (left) mouse button so right-clicks /
    // middle-clicks pass through cleanly.
    if !matches!(ev.button, PointerButton::Primary) { return; }
    // In EditColony mode, left-click is the editor's "place organism"
    // verb — emit a `ViewportClick` at the cursor and let
    // `colony_editor::placement::handle_left_click` ray-cast against
    // the heightmap. `Pointer<Click>` fires only on a clean tap
    // (press+release without intervening drag), so a hold-LMB camera
    // rotation never spawns an unintended template.
    if *window_mode == WindowMode::EditColony {
        if let Ok(window) = windows.single() {
            if let Some(cursor) = window.cursor_position() {
                click_writer.write(crate::camera::ViewportClick { cursor });
            }
        }
        return;
    }
    // Species-editor mode handles its own left-click semantics
    // (placing a cell at the cursor-snapped lattice position) in
    // `species_editor::placement::handle_left_click_place`. The player
    // controls must stay released so the cursor remains visible and
    // usable for that pointer-based placement workflow.
    if *window_mode == WindowMode::SpeciesEditor { return; }
    player_active.0 = !player_active.0;
}

/// Sets `EditorLookActive(true)` whenever the user presses LMB on the
/// viewport image while in EditColony mode. The flag persists until
/// LMB release (cleared by `release_editor_look_on_lmb_up` in
/// `player_plugin`). A press on any UI panel (mode bar, editor
/// panels, etc.) does NOT fire this observer because Bevy's picking
/// routes the event to the panel button instead — that's exactly the
/// "press over UI suppresses rotation" semantics we want.
fn viewport_press(
    ev:              On<Pointer<Press>>,
    window_mode:     Res<WindowMode>,
    mut editor_look: ResMut<crate::player_plugin::EditorLookActive>,
) {
    if !matches!(ev.button, PointerButton::Primary) { return; }
    if *window_mode != WindowMode::EditColony { return; }
    editor_look.0 = true;
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
    let currently_grabbed = matches!(cursor.grab_mode, bevy::window::CursorGrabMode::Locked);
    if want_grab == currently_grabbed { return; }
    if want_grab {
        // Warp the cursor to the window centre BEFORE locking. With
        // `CursorGrabMode::Locked` the cursor stays pinned at this
        // position for the duration of the grab — UI hover events
        // fire at the centre only (inside the viewport image, not on
        // any panel buttons), so panels no longer light up
        // "randomly" while the user moves the mouse to look around.
        // The mouse-look system reads relative motion from
        // `MouseMotion` events, which Bevy delivers correctly under
        // `Locked` regardless of the visible cursor position.
        let center = Vec2::new(window.width() / 2.0, window.height() / 2.0);
        window.set_cursor_position(Some(center));
        cursor.grab_mode = bevy::window::CursorGrabMode::Locked;
        cursor.visible   = false;
    } else {
        cursor.grab_mode = bevy::window::CursorGrabMode::None;
        cursor.visible   = true;
        // Release the cursor from the centre — the user can now move
        // it freely. Set its initial position to the centre as well
        // so the released pointer doesn't appear over a random UI
        // element it might accidentally trigger.
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


// ── Top mode-switch bar ──────────────────────────────────────────────────────
//
// Sits above the rest of the layout (first flex child of the root) and
// always stays visible. Two wide buttons — Simulation Mode and Edit
// Colony — each tagged with `ModeSwitchButton(WindowMode::...)`.
// `handle_mode_switch_buttons` reads `Changed<Interaction>` to flip
// `WindowMode`. The actual visibility / pause work is done in
// `apply_mode_transition`, which fires when `WindowMode` flips.

const MODE_BUTTON_ACTIVE:   Color = Color::srgb(0.30, 0.50, 0.30);
const MODE_BUTTON_INACTIVE: Color = Color::srgb(0.22, 0.22, 0.22);
const MODE_BUTTON_HOVER:    Color = Color::srgb(0.32, 0.32, 0.32);

fn spawn_top_mode_bar(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((
            TopModeBar,
            Node {
                width:           Val::Percent(100.0),
                height:          Val::Px(TOP_BAR_HEIGHT_PX),
                flex_shrink:     0.0,
                flex_direction:  FlexDirection::Row,
                align_items:     AlignItems::Stretch,
                ..default()
            },
            BackgroundColor(Color::srgb(0.10, 0.10, 0.10)),
        ))
        .with_children(|bar| {
            mode_button(bar, "Simulation Mode", WindowMode::Simulation,   true);
            mode_button(bar, "Edit Colony",     WindowMode::EditColony,   false);
            mode_button(bar, "Lineages",        WindowMode::Lineages,     false);
            mode_button(bar, "Species Editor",  WindowMode::SpeciesEditor, false);
        });
}

fn mode_button(
    parent:  &mut ChildSpawnerCommands,
    label:   &str,
    mode:    WindowMode,
    active:  bool,
) {
    let bg = if active { MODE_BUTTON_ACTIVE } else { MODE_BUTTON_INACTIVE };
    parent
        .spawn((
            ModeSwitchButton(mode),
            Button,
            Node {
                flex_grow:       1.0,
                flex_basis:      Val::Px(0.0),
                height:          Val::Percent(100.0),
                align_items:     AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            BackgroundColor(bg),
        ))
        .with_children(|b| {
            b.spawn((
                Text::new(label.to_string()),
                TextFont { font_size: 16.0, ..default() },
                TextColor(Color::WHITE),
                Pickable::IGNORE,
            ));
        });
}

/// Click router for the two mode buttons. On `Pressed`, writes the
/// chosen mode into the `WindowMode` resource — actual side-effects
/// (panel show/hide, pause, etc.) are applied by
/// `apply_mode_transition`, which keys off `Res::is_changed()`.
fn handle_mode_switch_buttons(
    mut interactions: Query<
        (&Interaction, &ModeSwitchButton, &mut BackgroundColor),
        Changed<Interaction>,
    >,
    mut window_mode:  ResMut<WindowMode>,
) {
    for (interaction, button, mut bg) in &mut interactions {
        let target_mode = button.0;
        let is_active   = *window_mode == target_mode;
        match *interaction {
            Interaction::Pressed => {
                if !is_active {
                    *window_mode = target_mode;
                }
                *bg = BackgroundColor(MODE_BUTTON_HOVER);
            }
            Interaction::Hovered => {
                *bg = BackgroundColor(MODE_BUTTON_HOVER);
            }
            Interaction::None => {
                *bg = BackgroundColor(if is_active { MODE_BUTTON_ACTIVE } else { MODE_BUTTON_INACTIVE });
            }
        }
    }
}

/// Drive the per-mode side-effects whenever `WindowMode` flips:
///   * Pause the simulation on entry into `EditColony` (Time<Virtual>
///     is paused via the resource flag; the sim systems read this).
///   * Release `PlayerControlsActive` on entry into `EditColony` so
///     the cursor isn't grabbed under the editor panels.
///   * Toggle `Display` on the simulation panels (statistics +
///     navigator + dividers + viewport-row container) vs the editor
///     overlay panels.
///   * Refresh the mode-bar button colours so the active mode is
///     visually distinct.
fn apply_mode_transition(
    window_mode:        Res<WindowMode>,
    mut sim_running:    ResMut<SimulationRunning>,
    mut player_active:  ResMut<PlayerControlsActive>,
    mut virtual_time:   ResMut<Time<Virtual>>,
    mut stats_q:        Query<&mut Node, (With<StatisticsPanel>,
                                          Without<NavigatorPanel>,
                                          Without<PanelDivider>,
                                          Without<ViewportRow>,
                                          Without<EditorOverlayPanel>,
                                          Without<crate::tree_view::LineagesPanel>,
                                          Without<crate::species_editor::SpeciesEditorPanel>,
                                          Without<crate::species_navigator::SpeciesNavigatorPanel>,
                                          Without<ModeSwitchButton>)>,
    mut nav_q:          Query<&mut Node, (With<NavigatorPanel>,
                                          Without<StatisticsPanel>,
                                          Without<PanelDivider>,
                                          Without<ViewportRow>,
                                          Without<EditorOverlayPanel>,
                                          Without<crate::tree_view::LineagesPanel>,
                                          Without<crate::species_editor::SpeciesEditorPanel>,
                                          Without<crate::species_navigator::SpeciesNavigatorPanel>,
                                          Without<ModeSwitchButton>)>,
    mut divider_q:      Query<&mut Node, (With<PanelDivider>,
                                          Without<StatisticsPanel>,
                                          Without<NavigatorPanel>,
                                          Without<ViewportRow>,
                                          Without<EditorOverlayPanel>,
                                          Without<crate::tree_view::LineagesPanel>,
                                          Without<crate::species_editor::SpeciesEditorPanel>,
                                          Without<crate::species_navigator::SpeciesNavigatorPanel>,
                                          Without<ModeSwitchButton>)>,
    mut editor_panels:  Query<&mut Node, (With<EditorOverlayPanel>,
                                          Without<StatisticsPanel>,
                                          Without<NavigatorPanel>,
                                          Without<PanelDivider>,
                                          Without<ViewportRow>,
                                          Without<crate::tree_view::LineagesPanel>,
                                          Without<crate::species_editor::SpeciesEditorPanel>,
                                          Without<crate::species_navigator::SpeciesNavigatorPanel>,
                                          Without<ModeSwitchButton>)>,
    mut lineages_panel: Query<&mut Node, (With<crate::tree_view::LineagesPanel>,
                                          Without<StatisticsPanel>,
                                          Without<NavigatorPanel>,
                                          Without<PanelDivider>,
                                          Without<ViewportRow>,
                                          Without<EditorOverlayPanel>,
                                          Without<crate::species_editor::SpeciesEditorPanel>,
                                          Without<ModeSwitchButton>)>,
    mut species_nav_p:  Query<&mut Node, (With<crate::species_navigator::SpeciesNavigatorPanel>,
                                          Without<StatisticsPanel>,
                                          Without<NavigatorPanel>,
                                          Without<PanelDivider>,
                                          Without<ViewportRow>,
                                          Without<EditorOverlayPanel>,
                                          Without<crate::tree_view::LineagesPanel>,
                                          Without<crate::species_editor::SpeciesEditorPanel>,
                                          Without<ModeSwitchButton>)>,
    mut species_panels: Query<&mut Node, (With<crate::species_editor::SpeciesEditorPanel>,
                                          Without<StatisticsPanel>,
                                          Without<NavigatorPanel>,
                                          Without<PanelDivider>,
                                          Without<ViewportRow>,
                                          Without<EditorOverlayPanel>,
                                          Without<crate::tree_view::LineagesPanel>,
                                          Without<crate::species_navigator::SpeciesNavigatorPanel>,
                                          Without<ModeSwitchButton>)>,
    mut buttons_q:      Query<(&ModeSwitchButton, &mut BackgroundColor)>,
) {
    if !window_mode.is_changed() { return; }

    // Per-mode visibility table. Simulation mode shows the stats /
    // navigator stack; EditColony shows the editor overlays;
    // Lineages shows only the tree-of-life panel (sim viewport
    // stays hidden because the panel covers the whole content
    // area).
    let mode = *window_mode;
    let sim_visible      = mode == WindowMode::Simulation;
    let editor_visible   = mode == WindowMode::EditColony;
    let lineages_visible = mode == WindowMode::Lineages;
    let species_visible  = mode == WindowMode::SpeciesEditor;
    let to_display = |v| if v { Display::Flex } else { Display::None };

    for mut n in &mut stats_q        { n.display = to_display(sim_visible);      }
    for mut n in &mut nav_q          { n.display = to_display(sim_visible);      }
    for mut n in &mut divider_q      { n.display = to_display(sim_visible);      }
    for mut n in &mut editor_panels  { n.display = to_display(editor_visible);   }
    for mut n in &mut lineages_panel { n.display = to_display(lineages_visible); }
    for mut n in &mut species_panels { n.display = to_display(species_visible);  }
    // Species Navigator (left side) — Simulation mode only.
    for mut n in &mut species_nav_p  { n.display = to_display(sim_visible);      }

    // Refresh mode-bar button colours so the active mode is highlighted
    // immediately on the transition tick.
    for (button, mut bg) in &mut buttons_q {
        let is_active = mode == button.0;
        *bg = BackgroundColor(if is_active { MODE_BUTTON_ACTIVE } else { MODE_BUTTON_INACTIVE });
    }

    // Modes other than Simulation pause the simulation on entry. The
    // user can resume manually via the statistics-panel button after
    // returning to Simulation mode.
    if editor_visible || lineages_visible || species_visible {
        sim_running.0 = false;
        virtual_time.pause();
        player_active.0 = false;
    }
}
