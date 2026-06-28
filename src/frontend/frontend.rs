// Frontend plugin — owns the UI: viewport render-target, layout composition,
// divider dragging, statistics panel (bottom), individuum navigator (right),
// in-world labels, orientation gizmos. Panel contents live in their own files.
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
// "V" is the vertical drag-handle. Wrapping viewport + navigator in one
// top-row means the navigator's height tracks the statistics-panel slider
// automatically.

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
    CinematicMode, PlayerControlsActive, SimulationRunning, Smoothing, TimeSpeed, WindowMode,
};
use crate::colony_editor::{self, EditorOverlayPlugin, EditorOverlayPanel};


// ── Tunables ─────────────────────────────────────────────────────────────────

/// Initial statistics-pane height as a fraction of the window's logical
/// height (leaves room for the space above the graph + Start/Stop button).
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

/// Thin top bar hosting the WindowMode buttons; never hidden, so the user can
/// always switch modes. `pub` so editor overlays reserve a matching top gap
/// (`top_offset_px` / `CursorTopReservedPx`).
pub const TOP_BAR_HEIGHT_PX: f32 = 36.0;


// ── Resources ────────────────────────────────────────────────────────────────

/// Toggleable visualisation flags (currently just the F3 gizmo overlay).
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
            .init_resource::<crate::simulation_settings::CinematicMode>()
            .init_resource::<OrganismCounts>()
            .init_resource::<DividerDragState>()
            .init_resource::<SimulationRunning>()
            .init_resource::<PlayerControlsActive>()
            .init_resource::<Smoothing>()
            .init_resource::<TimeSpeed>()
            .init_resource::<crate::simulation_settings::MaxPhotoautotrophs>()
            .init_resource::<crate::simulation_settings::OrganismPoolSize>()
            .init_resource::<WindowMode>()
            .init_resource::<statistics_panel::TimeSpeedEditState>()
            .init_resource::<statistics_panel::MaxPhotoautotrophsEditState>()
            .init_resource::<statistics_panel::MaxHerbivoresEditState>()
            .init_resource::<statistics_panel::CullMessage>()
            .init_resource::<crate::simulation_settings::AiTrainingMode>()
            .init_resource::<crate::simulation_settings::MaxHerbivores>()
            .insert_resource(GraphState::new())
            .add_plugins(FrameTimeDiagnosticsPlugin::default())
            // Lineages render-target image is created in `setup_panes`,
            // inserted as `LineagesViewportRender`, and consumed by
            // `setup_lineages_camera` on the following Startup tick (hence the
            // `.after`).
            .add_systems(Startup, crate::tree_view::setup_lineages_camera.after(setup_panes));
        crate::tree_view::init_resources(app);
        app
            .add_plugins(IndividuumNavigatorPlugin)
            .add_plugins(crate::species_navigator::SpeciesNavigatorPlugin)
            // Editor lives behind the EditColony WindowMode; reuses the sim's
            // 3D camera + lighting and only adds editor UI + input handlers
            // (gated on EditColony at the system level).
            .add_plugins(EditorOverlayPlugin)
            .add_systems(Update, (
                handle_mode_switch_buttons,
                apply_mode_transition,
                // Tree-of-life renderer; `LineagesPlugin` owns the
                // `SpeciesRegistry`, this turns it into a pan/zoom 2D scene.
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
                toggle_cinematic_mode,
                apply_cinematic_chrome,
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
            // Second `add_systems` call to stay under Bevy's tuple-size limit
            // for variadic system configs.
            .add_systems(Update, (
                dismiss_fly_hint,
                statistics_panel::update_camera_coords_text,
                statistics_panel::handle_max_phototrophs_input,
                statistics_panel::update_max_phototrophs_text,
                statistics_panel::apply_max_phototrophs_cull,
                statistics_panel::update_cull_message,
                statistics_panel::handle_ai_training_checkbox,
                statistics_panel::update_ai_training_checkbox_mark,
                statistics_panel::handle_max_herbivores_input,
                statistics_panel::update_max_herbivores_text,
                statistics_panel::handle_export_dataset_button,
                // TEMPORARY: log photoautotroph-visibility buckets every 5 s.
                statistics_panel::diag_photo_breakdown,
            ));
        // Gizmos gated by `run_if` (not an in-body check) so the scheduler
        // skips the system entirely when off, avoiding per-frame iteration
        // over thousands of `ShowGizmo` children.
        app.add_systems(
            Update,
            draw_orientation_gizmos.run_if(
                |vs: Res<ViewportSettings>, m: Res<WindowMode>|
                    vs.show_advanced_viewport && *m == WindowMode::Simulation
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

    // Render target the 3D camera draws into; `resize_render_target` re-fits
    // it once UI layout produces a real viewport-pane size.
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

    // Captured here so we can insert `LineagesViewportRender` after
    // `with_children` returns — commands can't be enqueued from inside the
    // child-spawner closure.
    let mut spawned_lineages_image: Option<Handle<Image>> = None;

    // 2D camera owns the window output and is the default UI camera so UI
    // never picks the player's Camera3d (redirected off-screen below). Higher
    // order so it composites after the 3D pass that produces the texture.
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

            // Top row: viewport (flex-grow) + vertical divider + navigator.
            // Wrapping them in one container lets the horizontal divider
            // resize the row as a whole, so the navigator's height tracks
            // the viewport's.
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
                // Left-side species navigator — flex sibling of the viewport
                // so it shares the row's vertical extent. Spawned BEFORE the
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
                    // Explicit Pickable: bevy_picking's UI backend skips bare
                    // ImageNodes (only auto-registers nodes with
                    // `BackgroundColor`). `should_block_lower: false` since
                    // nothing meaningful sits behind the viewport.
                    Pickable {
                        should_block_lower: false,
                        is_hoverable:       true,
                    },
                ))
                .observe(viewport_click)
                .observe(viewport_press)
                .with_children(|vp| {
                    // One-line startup hint; despawned by `dismiss_fly_hint`
                    // the first time the player flies.
                    vp.spawn((
                        FlyHint,
                        Node {
                            position_type:   PositionType::Absolute,
                            top:             Val::Px(16.0),
                            left:            Val::Px(0.0),
                            width:           Val::Percent(100.0),
                            justify_content: JustifyContent::Center,
                            ..default()
                        },
                        Pickable::IGNORE,
                    ))
                    .with_children(|h| {
                        h.spawn((
                            Text::new("Press Space to fly"),
                            TextFont { font_size: 18.0, ..default() },
                            TextColor(Color::WHITE),
                            Pickable::IGNORE,
                        ));
                    });
                });

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
            // Floating identifier labels are spawned by
            // `individuum_navigator::manage_label_lifecycle` as children of
            // the `ViewportImage` — no separate overlay container.

            // ── Editor overlay panels ────────────────────────────────
            // Children of the root so absolute positioning resolves against
            // the window. Tagged with `EditorOverlayPanel` so the
            // mode-transition system can flip every panel's `Display` in one
            // query. Start hidden (boot in `WindowMode::Simulation`).
            colony_editor::spawn_overlay_panels(root, TOP_BAR_HEIGHT_PX);

            // ── Species editor panels ────────────────────────────────
            // Spawn hidden; mode-transition flips them on WindowMode::SpeciesEditor.
            crate::species_editor::spawn_overlay_panels(root, TOP_BAR_HEIGHT_PX);

            // ── Map editor panels ─────────────────────────────────────
            // Spawn hidden; the map-editor visibility toggle flips them on
            // WindowMode::MapEditor.
            crate::map_editor::spawn_overlay_panels(root, TOP_BAR_HEIGHT_PX);

            // Lineages mode panel — full-content tree-of-life view. Spawned
            // hidden; mode-transition toggles its `Display`.
            let lineages_image = crate::tree_view::spawn_lineages_panel(
                root, TOP_BAR_HEIGHT_PX, &mut images,
            );
            // Stashed for the outer `commands.insert_resource` below;
            // commands can't be enqueued inside `with_children`.
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
        // ComputedNode is in physical px; convert to logical to match
        // Drag::distance and the Val::Px height we write back.
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

/// Redirect the existing Camera3d (from `player_plugin`) to render into our
/// off-screen image. Retries each Update until it succeeds (`bound` guard).
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


// ── Fly hint ─────────────────────────────────────────────────────────────────

/// Marker on the one-line "Press Space to fly" startup hint over the viewport.
#[derive(Component)]
struct FlyHint;

/// Despawn the fly hint the first time the player starts flying. Driven by
/// `PlayerControlsActive` becoming true (Space engages it — see
/// `player_plugin::engage_flying_on_space`). Once the hint entities are gone
/// the query is empty, so this costs nothing thereafter.
fn dismiss_fly_hint(
    mut commands:  Commands,
    player_active: Res<PlayerControlsActive>,
    hints:         Query<Entity, With<FlyHint>>,
) {
    if !player_active.0 { return; }
    for e in &hints {
        commands.entity(e).despawn();
    }
}


// ── Viewport click → activate player controls ────────────────────────────────

/// Viewport left-click handling. Camera capture is NOT toggled here — that's
/// driven solely by Esc (`player_plugin::toggle_player_controls_on_esc`).
/// Simulation mode → pick organism; EditColony → editor placement.
fn viewport_click(
    ev:                 On<Pointer<Click>>,
    window_mode:        Res<WindowMode>,
    windows:            Query<&Window, With<PrimaryWindow>>,
    mut click_writer:   MessageWriter<crate::camera::ViewportClick>,
    mut pick_writer:    MessageWriter<individuum_navigator::ViewportPick>,
) {
    // Left button only; right/middle pass through.
    if !matches!(ev.button, PointerButton::Primary) { return; }
    // Simulation mode: select the heterotroph under the cursor.
    if *window_mode == WindowMode::Simulation {
        if let Ok(window) = windows.single() {
            if let Some(cursor) = window.cursor_position() {
                pick_writer.write(individuum_navigator::ViewportPick { cursor });
            }
        }
        return;
    }
    // EditColony: emit a `ViewportClick` for `placement::handle_left_click`.
    // `Pointer<Click>` fires only on a clean tap (no intervening drag), so a
    // hold-LMB camera rotation never spawns an unintended template.
    if *window_mode == WindowMode::EditColony {
        if let Ok(window) = windows.single() {
            if let Some(cursor) = window.cursor_position() {
                click_writer.write(crate::camera::ViewportClick { cursor });
            }
        }
    }
    // Species-editor mode: placement is handled by that editor's own observer.
}

/// Sets `EditorLookActive(true)` on MIDDLE-mouse press over the viewport in the
/// editor modes (cleared on release by `player_plugin::release_editor_look_on_mmb_up`).
/// A press over any UI panel routes to the panel instead, so it doesn't fire
/// here — giving "press over UI suppresses rotation" semantics. Middle-hold
/// rotates the shared flycam in both the Colony and Species editors; LMB stays
/// free for placement/selection.
fn viewport_press(
    ev:              On<Pointer<Press>>,
    window_mode:     Res<WindowMode>,
    mut editor_look: ResMut<crate::player_plugin::EditorLookActive>,
) {
    if !matches!(ev.button, PointerButton::Middle) { return; }
    if !matches!(*window_mode, WindowMode::EditColony | WindowMode::SpeciesEditor) { return; }
    editor_look.0 = true;
}

/// Mirror `PlayerControlsActive` onto the OS cursor grab state; only writes
/// on a state change. Recentres the cursor on transitions so the released
/// pointer doesn't land on a random UI element.
fn apply_player_controls_state(
    player_active: Res<PlayerControlsActive>,
    window_q:      Single<(&mut Window, &mut bevy::window::CursorOptions), With<PrimaryWindow>>,
) {
    let (mut window, mut cursor) = window_q.into_inner();
    let want_grab = player_active.0;
    let currently_grabbed = matches!(cursor.grab_mode, bevy::window::CursorGrabMode::Locked);
    if want_grab == currently_grabbed { return; }
    if want_grab {
        // Warp to centre BEFORE locking: under `CursorGrabMode::Locked` the
        // cursor stays pinned here, so UI hover events fire only at the
        // centre (inside the viewport, not on panel buttons). Mouse-look
        // uses relative `MouseMotion`, delivered correctly while locked.
        let center = Vec2::new(window.width() / 2.0, window.height() / 2.0);
        window.set_cursor_position(Some(center));
        cursor.grab_mode = bevy::window::CursorGrabMode::Locked;
        cursor.visible   = false;
    } else {
        cursor.grab_mode = bevy::window::CursorGrabMode::None;
        cursor.visible   = true;
        // Recentre on release so the freed pointer doesn't land on a random
        // UI element it might trigger.
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

/// F1 toggles cinematic mode, only in `WindowMode::Simulation` (other modes
/// have their own full-screen panels); also clears it if no longer in
/// Simulation. Writing `cinematic.0` re-runs `apply_mode_transition` +
/// `apply_cinematic_chrome`.
pub fn toggle_cinematic_mode(
    keys:          Res<ButtonInput<KeyCode>>,
    window_mode:   Res<WindowMode>,
    mut cinematic: ResMut<CinematicMode>,
) {
    if *window_mode == WindowMode::Simulation {
        if keys.just_pressed(KeyCode::F1) {
            cinematic.0 = !cinematic.0;
        }
    } else if cinematic.0 {
        // Safety net: cinematic only makes sense in Simulation mode.
        cinematic.0 = false;
    }
}

/// Hide / show the chrome `apply_mode_transition` doesn't own (top mode bar +
/// vertical divider) in lock-step with cinematic mode. Separate so these node
/// types stay out of that system's disjoint-`&mut Node` query set.
pub fn apply_cinematic_chrome(
    cinematic:   Res<CinematicMode>,
    mut top_bar: Query<&mut Node, (With<TopModeBar>,
                                   Without<crate::individuum_navigator::VerticalDivider>)>,
    mut vdiv:    Query<&mut Node, (With<crate::individuum_navigator::VerticalDivider>,
                                   Without<TopModeBar>)>,
) {
    if !cinematic.is_changed() { return; }
    let display = if cinematic.0 { Display::None } else { Display::Flex };
    for mut n in &mut top_bar { n.display = display; }
    for mut n in &mut vdiv    { n.display = display; }
}

pub fn draw_orientation_gizmos(
    // Only `ShowGizmo`-tagged entities; the F3 toggle is enforced by a
    // `.run_if` on the registration.
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
// First flex child of the root, always visible. Buttons tagged with
// `ModeSwitchButton(WindowMode::...)`; `handle_mode_switch_buttons` flips
// `WindowMode`, and `apply_mode_transition` does the visibility / pause work.

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
            mode_button(bar, "Map Editor",      WindowMode::MapEditor,     false);
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

/// Click router for the mode buttons. On `Pressed`, writes `WindowMode`;
/// side-effects are applied by `apply_mode_transition` (keys off `is_changed`).
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
    cinematic:          Res<CinematicMode>,
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
    // Runs on a window-mode change OR a cinematic-mode toggle — both affect
    // which sim panels are visible.
    let mode_changed = window_mode.is_changed();
    if !mode_changed && !cinematic.is_changed() { return; }

    // Per-mode visibility. Cinematic hides the Simulation-mode panels so the
    // flex-grow viewport fills the window (`resize_render_target` follows).
    let mode = *window_mode;
    let sim_visible      = mode == WindowMode::Simulation && !cinematic.0;
    let editor_visible   = mode == WindowMode::EditColony;
    let lineages_visible = mode == WindowMode::Lineages;
    let species_visible  = mode == WindowMode::SpeciesEditor;
    let map_visible      = mode == WindowMode::MapEditor;
    let to_display = |v| if v { Display::Flex } else { Display::None };

    for mut n in &mut stats_q        { n.display = to_display(sim_visible);      }
    for mut n in &mut nav_q          { n.display = to_display(sim_visible);      }
    for mut n in &mut divider_q      { n.display = to_display(sim_visible);      }
    for mut n in &mut editor_panels  { n.display = to_display(editor_visible);   }
    for mut n in &mut lineages_panel { n.display = to_display(lineages_visible); }
    for mut n in &mut species_panels { n.display = to_display(species_visible);  }
    // Species Navigator (left side) — Simulation mode only.
    for mut n in &mut species_nav_p  { n.display = to_display(sim_visible);      }

    // Refresh mode-bar button colours to highlight the active mode.
    for (button, mut bg) in &mut buttons_q {
        let is_active = mode == button.0;
        *bg = BackgroundColor(if is_active { MODE_BUTTON_ACTIVE } else { MODE_BUTTON_INACTIVE });
    }

    // Non-Simulation modes pause on entry. Gated on an actual mode change so
    // a cinematic toggle (which also re-runs this) never pauses the sim.
    if mode_changed && (editor_visible || lineages_visible || species_visible || map_visible) {
        sim_running.0 = false;
        virtual_time.pause();
        player_active.0 = false;
    }
}
