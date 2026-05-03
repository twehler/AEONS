use std::collections::VecDeque;

use bevy::prelude::*;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::camera::RenderTarget;
use bevy::render::render_resource::{Extent3d, TextureFormat};
use bevy::window::PrimaryWindow;

use crate::colony::{Photoautotroph, Heterotroph};


// Initial statistics-pane height as a fraction of the window's logical height.
const PANEL_INITIAL_FRACTION: f32 = 0.1;
// Thickness of the draggable splitter between the panes (logical px).
const DIVIDER_HEIGHT: f32 = 6.0;
// Lower bound on the statistics pane size while dragging (logical px).
const PANEL_MIN_PX: f32 = 24.0;
// Lower bound on the viewport pane size while dragging (logical px).
const VIEWPORT_MIN_PX: f32 = 50.0;

// Live-graph window: 60 segments → 61 sample points.
const GRAPH_DURATION_SECS: usize = 60;
const GRAPH_SAMPLE_CAP: usize = GRAPH_DURATION_SECS + 1;
// Graph background (sRGB-encoded bytes — see image format note in setup_panes).
const GRAPH_BG: [u8; 4] = [20, 20, 20, 255];
// Phototrophic line colour (green) and heterotrophic line colour (red).
const GRAPH_PHOTO_COLOR: [u8; 4] = [40, 220, 60, 255];
const GRAPH_HETERO_COLOR: [u8; 4] = [220, 50, 50, 255];
// Floor for the auto-scaled y-axis so a 1-organism world doesn't slam to the
// top of the chart.
const GRAPH_Y_MIN_RANGE: f32 = 10.0;


pub struct ViewportSettingsPlugin;

impl Plugin for ViewportSettingsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ViewportSettings>()
            .init_resource::<OrganismCounts>()
            .init_resource::<DividerDragState>()
            .insert_resource(GraphState::new())
            .add_plugins(FrameTimeDiagnosticsPlugin::default())
            .add_systems(Startup, setup_panes)
            .add_systems(Update, (
                bind_main_camera_to_viewport,
                resize_render_target,
                toggle_advanced_viewport,
                update_fps_text,
                track_organism_births,
                track_organism_deaths,
                update_counter_texts,
                graph_tick,
                graph_redraw,
                draw_orientation_gizmos,
            ));
    }
}


#[derive(Resource, Default)]
pub struct ViewportSettings {
    pub show_advanced_viewport: bool,
}

// Live tally of organisms by trophic strategy.
#[derive(Resource, Default)]
pub struct OrganismCounts {
    pub photo: i32,
    pub hetero: i32,
}

// Captured at DragStart so Drag events (which give cumulative distance from
// the start point) can resolve to an absolute height.
#[derive(Resource, Default)]
struct DividerDragState {
    initial_panel_height: f32,
}

// Owns the texture the 3D camera renders into and tracks its current size
// so we only reallocate the GPU buffer when the viewport pane actually changes.
#[derive(Resource)]
struct ViewportRender {
    image: Handle<Image>,
    bound: bool,
    current_size: UVec2,
}

// Live-graph state. `samples[0]` is the most recent (rendered at x=0); the
// deque caps at GRAPH_SAMPLE_CAP and `pop_back` drops samples older than 60s.
// `dirty` is set both on tick and on resize; the redraw system clears it after
// rasterising. Keeping the dirty flag means we don't touch the GPU texture in
// frames where nothing changed.
#[derive(Resource)]
struct GraphState {
    samples: VecDeque<(i32, i32)>,
    timer: Timer,
    image: Handle<Image>,
    image_size: UVec2,
    dirty: bool,
}

impl GraphState {
    fn new() -> Self {
        Self {
            samples: VecDeque::with_capacity(GRAPH_SAMPLE_CAP),
            timer: Timer::from_seconds(1.0, TimerMode::Repeating),
            image: Handle::default(),
            image_size: UVec2::ZERO,
            dirty: false,
        }
    }
}


// Add this component to any entity you want orientation gizmos drawn for.
// Intentionally NOT added to terrain chunks — only to cells or organisms.
#[derive(Component)]
pub struct ShowGizmo;

#[derive(Component)]
struct StatisticsPanel;

#[derive(Component)]
struct ViewportImage;

#[derive(Component)]
struct PanelDivider;

#[derive(Component)]
struct FpsText {
    timer: Timer,
}

#[derive(Component)]
struct PhotoCountText;

#[derive(Component)]
struct HeteroCountText;

#[derive(Component)]
struct GraphImage;


pub fn toggle_advanced_viewport(
    keys: Res<ButtonInput<KeyCode>>,
    mut viewport_settings: ResMut<ViewportSettings>,
) {
    if keys.just_pressed(KeyCode::F3) {
        viewport_settings.show_advanced_viewport = !viewport_settings.show_advanced_viewport;
    }
}


fn setup_panes(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut graph: ResMut<GraphState>,
    windows: Query<&Window, With<PrimaryWindow>>,
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

    // Render target the 3D camera will draw into. Initial dimensions match
    // the window; resize_render_target re-fits it once UI layout produces a
    // real ComputedNode size for the viewport pane.
    let image_handle = images.add(Image::new_target_texture(
        init_w,
        init_h,
        TextureFormat::Rgba8Unorm,
        Some(TextureFormat::Rgba8UnormSrgb),
    ));

    commands.insert_resource(ViewportRender {
        image: image_handle.clone(),
        bound: false,
        current_size: UVec2::new(0, 0),
    });

    // Graph texture. Same sRGB-view trick as the viewport: storage is
    // Rgba8Unorm but the sampler treats bytes as already-sRGB, so the colour
    // bytes we write match what the user sees on a sRGB swapchain.
    // 1×1 placeholder; graph_redraw resizes to the node's real pixel size on
    // the first frame after layout.
    let graph_image = images.add(Image::new_target_texture(
        1,
        1,
        TextureFormat::Rgba8Unorm,
        Some(TextureFormat::Rgba8UnormSrgb),
    ));
    graph.image = graph_image.clone();

    // 2D camera owns the window output and is marked as the default UI camera
    // so UI propagation never picks the player's Camera3d (which we redirect
    // to the off-screen image below). Higher order so it composites after the
    // 3D pass that produces the viewport texture.
    commands.spawn((
        Camera2d,
        Camera { order: 1, ..default() },
        IsDefaultUiCamera,
    ));

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(0.0),
                left: Val::Px(0.0),
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            Pickable::IGNORE,
        ))
        .with_children(|root| {
            root.spawn((
                ViewportImage,
                ImageNode::new(image_handle),
                Node {
                    width: Val::Percent(100.0),
                    flex_grow: 1.0,
                    flex_basis: Val::Px(0.0),
                    min_height: Val::Px(0.0),
                    ..default()
                },
                Pickable::IGNORE,
            ));

            root.spawn((
                PanelDivider,
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(DIVIDER_HEIGHT),
                    flex_shrink: 0.0,
                    ..default()
                },
                BackgroundColor(Color::srgb(0.35, 0.35, 0.35)),
            ))
            .observe(divider_drag_start)
            .observe(divider_drag);

            root.spawn((
                StatisticsPanel,
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(initial_panel_logical),
                    flex_direction: FlexDirection::Row,
                    justify_content: JustifyContent::SpaceBetween,
                    align_items: AlignItems::Center,
                    padding: UiRect::horizontal(Val::Px(12.0)),
                    flex_shrink: 0.0,
                    ..default()
                },
                BackgroundColor(Color::srgb(0.15, 0.15, 0.15)),
            ))
            .with_children(|panel| {
                panel.spawn((
                    Text::new("FPS: 0.0"),
                    TextFont { font_size: 20.0, ..default() },
                    TextColor(Color::WHITE),
                    FpsText { timer: Timer::from_seconds(0.05, TimerMode::Repeating) },
                ));

                // Graph: absolutely positioned so it sits at exactly 25vw
                // from the window edge and fills the panel height. Taking it
                // out of flex flow keeps FPS/counters anchored at the panel
                // edges via SpaceBetween regardless of graph width.
                panel.spawn((
                    GraphImage,
                    ImageNode::new(graph_image.clone()),
                    Node {
                        position_type: PositionType::Absolute,
                        left: Val::Vw(25.0),
                        top: Val::Px(0.0),
                        width: Val::Vw(50.0),
                        height: Val::Percent(100.0),
                        ..default()
                    },
                    Pickable::IGNORE,
                ));

                panel
                    .spawn(Node {
                        flex_direction: FlexDirection::Column,
                        align_items: AlignItems::FlexEnd,
                        ..default()
                    })
                    .with_children(|right| {
                        right.spawn((
                            Text::new("Phototrophic: 0"),
                            TextFont { font_size: 20.0, ..default() },
                            TextColor(Color::WHITE),
                            PhotoCountText,
                        ));
                        right.spawn((
                            Text::new("Heterotrophic: 0"),
                            TextFont { font_size: 20.0, ..default() },
                            TextColor(Color::WHITE),
                            HeteroCountText,
                        ));
                    });
            });
        });
}


fn divider_drag_start(
    _ev: On<Pointer<DragStart>>,
    panel_q: Query<&ComputedNode, With<StatisticsPanel>>,
    mut state: ResMut<DividerDragState>,
) {
    if let Ok(panel) = panel_q.single() {
        // ComputedNode reports physical pixels. Convert to logical so it
        // composes with Drag::distance (which is in logical pixels) and the
        // Val::Px height we'll write back.
        state.initial_panel_height = panel.size().y * panel.inverse_scale_factor;
    }
}

fn divider_drag(
    ev: On<Pointer<Drag>>,
    state: Res<DividerDragState>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut panel_q: Query<&mut Node, With<StatisticsPanel>>,
) {
    let Ok(window) = windows.single() else { return };
    let Ok(mut panel_node) = panel_q.single_mut() else { return };
    let max_h = (window.height() - DIVIDER_HEIGHT - VIEWPORT_MIN_PX).max(PANEL_MIN_PX);
    let new_h = (state.initial_panel_height - ev.distance.y).clamp(PANEL_MIN_PX, max_h);
    panel_node.height = Val::Px(new_h);
}


// One-shot: redirect the existing Camera3d (spawned by player_plugin) to
// render into our off-screen image. Runs every Update until it succeeds, then
// the bound flag short-circuits.
fn bind_main_camera_to_viewport(
    mut commands: Commands,
    mut viewport: ResMut<ViewportRender>,
    cameras: Query<Entity, With<Camera3d>>,
) {
    if viewport.bound {
        return;
    }
    let Some(cam) = cameras.iter().next() else { return };
    commands
        .entity(cam)
        .insert(RenderTarget::Image(viewport.image.clone().into()));
    viewport.bound = true;
}


// Match the GPU texture size to the viewport pane's actual layout size so
// the camera renders at native resolution without stretch.
fn resize_render_target(
    mut images: ResMut<Assets<Image>>,
    mut viewport: ResMut<ViewportRender>,
    viewport_node: Query<&ComputedNode, With<ViewportImage>>,
) {
    let Ok(node) = viewport_node.single() else { return };
    let size = node.size();
    let new_w = size.x.max(1.0).round() as u32;
    let new_h = size.y.max(1.0).round() as u32;
    let new_size = UVec2::new(new_w, new_h);
    if new_size == viewport.current_size {
        return;
    }
    if let Some(image) = images.get_mut(&viewport.image) {
        image.resize(Extent3d {
            width: new_w,
            height: new_h,
            depth_or_array_layers: 1,
        });
        viewport.current_size = new_size;
    }
}


fn update_fps_text(
    time: Res<Time>,
    diagnostics: Res<DiagnosticsStore>,
    mut query: Query<(&mut Text, &mut FpsText)>,
) {
    for (mut text, mut fps_marker) in &mut query {
        fps_marker.timer.tick(time.delta());
        if fps_marker.timer.just_finished() {
            if let Some(fps_diag) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) {
                if let Some(fps_value) = fps_diag.smoothed() {
                    text.0 = format!("FPS: {:.1}", fps_value);
                }
            }
        }
    }
}


fn track_organism_births(
    photo_added: Query<Entity, Added<Photoautotroph>>,
    hetero_added: Query<Entity, Added<Heterotroph>>,
    mut counts: ResMut<OrganismCounts>,
) {
    let photo_n = photo_added.iter().count() as i32;
    let hetero_n = hetero_added.iter().count() as i32;
    if photo_n > 0 {
        counts.photo += photo_n;
    }
    if hetero_n > 0 {
        counts.hetero += hetero_n;
    }
}

fn track_organism_deaths(
    mut photo_removed: RemovedComponents<Photoautotroph>,
    mut hetero_removed: RemovedComponents<Heterotroph>,
    mut counts: ResMut<OrganismCounts>,
) {
    let photo_n = photo_removed.read().count() as i32;
    let hetero_n = hetero_removed.read().count() as i32;
    if photo_n > 0 {
        counts.photo -= photo_n;
    }
    if hetero_n > 0 {
        counts.hetero -= hetero_n;
    }
}

fn update_counter_texts(
    counts: Res<OrganismCounts>,
    mut photo_q: Query<&mut Text, (With<PhotoCountText>, Without<HeteroCountText>)>,
    mut hetero_q: Query<&mut Text, (With<HeteroCountText>, Without<PhotoCountText>)>,
) {
    if !counts.is_changed() {
        return;
    }
    for mut text in &mut photo_q {
        text.0 = format!("Phototrophic: {}", counts.photo);
    }
    for mut text in &mut hetero_q {
        text.0 = format!("Heterotrophic: {}", counts.hetero);
    }
}


// 1 Hz: capture a snapshot of OrganismCounts, push it onto the front of the
// ring buffer, drop the oldest sample once we exceed 60 seconds. Marks the
// graph dirty so the redraw system rasterises on this frame.
fn graph_tick(
    time: Res<Time>,
    counts: Res<OrganismCounts>,
    mut graph: ResMut<GraphState>,
) {
    graph.timer.tick(time.delta());
    if !graph.timer.just_finished() {
        return;
    }
    graph.samples.push_front((counts.photo, counts.hetero));
    while graph.samples.len() > GRAPH_SAMPLE_CAP {
        graph.samples.pop_back();
    }
    graph.dirty = true;
}

// CPU-side rasteriser. Touches the GPU only when (a) the node's pixel size
// changed (resize → reallocate texture buffer, redraw), or (b) graph_tick
// pushed a new sample (1 Hz redraw at most). In all other frames this system
// returns after a single `==` check on UVec2 — no allocation, no upload.
fn graph_redraw(
    mut images: ResMut<Assets<Image>>,
    mut graph: ResMut<GraphState>,
    node_q: Query<&ComputedNode, With<GraphImage>>,
) {
    let Ok(node) = node_q.single() else { return };
    let size = node.size();
    let new_w = size.x.max(2.0).round() as u32;
    let new_h = size.y.max(2.0).round() as u32;
    let new_size = UVec2::new(new_w, new_h);
    let resized = new_size != graph.image_size;

    if !graph.dirty && !resized {
        return;
    }

    let Some(image) = images.get_mut(&graph.image) else { return };
    if resized {
        image.resize(Extent3d {
            width: new_w,
            height: new_h,
            depth_or_array_layers: 1,
        });
        graph.image_size = new_size;
    }

    let Some(buf) = image.data.as_mut() else { return };
    rasterise_graph(buf, new_w as i32, new_h as i32, &graph.samples);
    graph.dirty = false;
}

// Pure pixel-twiddling routine. ~ (w * h * 4) bytes for the background fill
// plus ~60 Bresenham segments per line, called at most once per second
// (or on resize). No allocations.
fn rasterise_graph(buf: &mut [u8], w: i32, h: i32, samples: &VecDeque<(i32, i32)>) {
    // Background.
    for chunk in buf.chunks_exact_mut(4) {
        chunk.copy_from_slice(&GRAPH_BG);
    }
    if samples.len() < 2 {
        return;
    }

    // Auto-scale y to the largest count currently on screen, with a floor.
    let mut y_max_i: i32 = 0;
    for &(p, het) in samples.iter() {
        if p > y_max_i { y_max_i = p; }
        if het > y_max_i { y_max_i = het; }
    }
    let y_max = (y_max_i as f32).max(GRAPH_Y_MIN_RANGE);

    // x-axis covers exactly GRAPH_DURATION_SECS samples → (w-1) px.
    let x_scale = (w - 1) as f32 / GRAPH_DURATION_SECS as f32;
    let y_scale = (h - 1) as f32 / y_max;

    // Walk consecutive samples once and draw both lines per segment to keep
    // memory access patterns tight.
    let mut prev_x = 0;
    let mut prev_photo_y = h - 1 - ((samples[0].0 as f32 * y_scale).round() as i32).clamp(0, h - 1);
    let mut prev_hetero_y = h - 1 - ((samples[0].1 as f32 * y_scale).round() as i32).clamp(0, h - 1);
    for i in 1..samples.len() {
        let (p, het) = samples[i];
        let x = ((i as f32) * x_scale).round() as i32;
        let py = h - 1 - ((p as f32 * y_scale).round() as i32).clamp(0, h - 1);
        let hy = h - 1 - ((het as f32 * y_scale).round() as i32).clamp(0, h - 1);
        draw_line(buf, w, h, prev_x, prev_photo_y, x, py, GRAPH_PHOTO_COLOR);
        draw_line(buf, w, h, prev_x, prev_hetero_y, x, hy, GRAPH_HETERO_COLOR);
        prev_x = x;
        prev_photo_y = py;
        prev_hetero_y = hy;
    }
}

#[inline(always)]
fn put_pixel(buf: &mut [u8], w: i32, h: i32, x: i32, y: i32, color: [u8; 4]) {
    if (x as u32) >= w as u32 || (y as u32) >= h as u32 {
        return;
    }
    let idx = ((y * w + x) as usize) * 4;
    buf[idx..idx + 4].copy_from_slice(&color);
}

// Bresenham. Handles every octant.
fn draw_line(buf: &mut [u8], w: i32, h: i32, x0: i32, y0: i32, x1: i32, y1: i32, color: [u8; 4]) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let (mut x, mut y) = (x0, y0);
    loop {
        put_pixel(buf, w, h, x, y, color);
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}


pub fn draw_orientation_gizmos(
    // Only queries entities explicitly tagged with ShowGizmo —
    // terrain chunks, UI, lights etc. are never included.
    query: Query<&Transform, With<ShowGizmo>>,
    mut gizmos: Gizmos,
    viewport_settings: Res<ViewportSettings>,
) {
    if !viewport_settings.show_advanced_viewport {
        return;
    }

    for transform in &query {
        let pos = transform.translation;
        let length = 1.5; // Scaled to cell size (~1 unit), not terrain scale

        gizmos.arrow(pos, pos + transform.right()   * length, Color::srgb(1.0, 0.0, 0.0));
        gizmos.arrow(pos, pos + transform.up()      * length, Color::srgb(1.0, 1.0, 0.0));
        gizmos.arrow(pos, pos + transform.forward() * length, Color::srgb(0.0, 0.0, 1.0));
    }
}
