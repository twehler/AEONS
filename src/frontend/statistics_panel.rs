// Statistics panel — the bottom UI strip.
//
// FPS counter and live cell-count on the left, photo/hetero population graph
// in the centre, current photo/hetero counts on the right. Built once during
// `frontend::setup_panes` via `spawn_panel`, then driven by the systems
// declared at the bottom of this file (registered by `FrontendPlugin`).
//
// The panel's vertical extent is owned by `frontend.rs::divider_drag` —
// dragging the divider above this panel shrinks/grows it.

use std::collections::VecDeque;
use std::fmt::Write as _;

use bevy::prelude::*;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::render::render_resource::Extent3d;
use bevy::render::render_resource::TextureFormat;

use crate::colony::{Photoautotroph, Heterotroph, Organism, OrganismRoot, SaveRequested};
use crate::frontend::PANEL_BG_COLOR;
use crate::simulation_settings::SimulationRunning;


// ── Tunables ─────────────────────────────────────────────────────────────────

/// Live-graph window in seconds. The graph stores `GRAPH_DURATION_SECS + 1`
/// samples (one per second + the freshly-pushed front sample).
pub const GRAPH_DURATION_SECS: usize = 60;
const GRAPH_SAMPLE_CAP: usize = GRAPH_DURATION_SECS + 1;

/// Graph background colour (sRGB-encoded bytes — see image format note in
/// `frontend.rs::setup_panes`).
const GRAPH_BG: [u8; 4] = [20, 20, 20, 255];
/// Photoautotroph line colour (green) and heterotroph line colour (red).
const GRAPH_PHOTO_COLOR:  [u8; 4] = [40, 220, 60, 255];
const GRAPH_HETERO_COLOR: [u8; 4] = [220, 50, 50, 255];

/// Floor for the auto-scaled y-axis so a 1-organism world doesn't slam to
/// the top of the chart.
const GRAPH_Y_MIN_RANGE: f32 = 10.0;

/// Vertical line separating the FPS/Cells block from the rest of the
/// panel — width in logical pixels.
const VERTICAL_LINE_WIDTH:  f32 = 2.0;
const VERTICAL_LINE_COLOR:  Color = Color::srgb(0.35, 0.35, 0.35);

/// Symmetric gap (logical px) above AND below the Start/Stop button. The
/// distance from the divider to the button's top equals this; the
/// distance from the button's bottom to the graph equals this too — the
/// graph "reaches up" until it sits the same small inset below the
/// button as the button sits below the grey separator.
const BUTTON_GAP_PX: f32 = 8.0;

/// Start/Stop button styling.
const BUTTON_WIDTH_PX:        f32 = 110.0;
const BUTTON_HEIGHT_PX:       f32 = 36.0;
const BUTTON_COLOR_RUN:       Color = Color::srgb(0.55, 0.18, 0.18); // red — "Pause"
const BUTTON_COLOR_PAUSED:    Color = Color::srgb(0.18, 0.55, 0.22); // green — "Start"
const BUTTON_COLOR_HOVER:     Color = Color::srgb(0.30, 0.30, 0.30);

/// Save button styling — sits in the bottom-right of the panel.
const BUTTON_COLOR_SAVE:      Color = Color::srgb(0.20, 0.40, 0.65); // calm blue


// ── Resources ────────────────────────────────────────────────────────────────

/// Live tally of organisms by trophic strategy. Updated by
/// `track_organism_births` / `track_organism_deaths`; read by
/// `update_counter_texts` and `graph_tick`.
#[derive(Resource, Default)]
pub struct OrganismCounts {
    pub photo: i32,
    pub hetero: i32,
}

/// Live-graph state. `samples[0]` is the most recent sample (rendered at
/// x=0); the deque caps at `GRAPH_SAMPLE_CAP` and `pop_back` drops samples
/// older than `GRAPH_DURATION_SECS` seconds. `dirty` is set both on tick
/// and on resize; the redraw system clears it after rasterising. Keeping
/// the dirty flag means we don't touch the GPU texture in frames where
/// nothing has changed.
#[derive(Resource)]
pub struct GraphState {
    samples:    VecDeque<(i32, i32)>,
    timer:      Timer,
    image:      Handle<Image>,
    image_size: UVec2,
    dirty:      bool,
}

impl GraphState {
    pub fn new() -> Self {
        Self {
            samples:    VecDeque::with_capacity(GRAPH_SAMPLE_CAP),
            timer:      Timer::from_seconds(1.0, TimerMode::Repeating),
            image:      Handle::default(),
            image_size: UVec2::ZERO,
            dirty:      false,
        }
    }
}


// ── Marker components ────────────────────────────────────────────────────────

/// Marker on the bottom panel — `frontend::divider_drag` queries on this to
/// resize the panel as the divider is dragged.
#[derive(Component)]
pub struct StatisticsPanel;

// Marker structs are `pub` because the systems registered by
// `FrontendPlugin` (defined in this file) reference them in their `Query`
// signatures. Rust's `private_interfaces` lint rejects pub-fn / private-type
// mismatches even when the type is never consumed outside the crate.
#[derive(Component)]
pub struct FpsText { timer: Timer }

#[derive(Component)]
pub struct CellCountText { timer: Timer }

#[derive(Component)]
pub struct PhotoCountText;

#[derive(Component)]
pub struct HeteroCountText;

#[derive(Component)]
pub struct GraphImage;

/// Marker on the Start/Stop button itself.
#[derive(Component)]
pub struct StartStopButton;

/// Marker on the Text entity inside the Start/Stop button so the label
/// update system can find it without re-traversing children.
#[derive(Component)]
pub struct StartStopButtonText;

/// Marker on the Save button (bottom-right of the statistics panel).
#[derive(Component)]
pub struct SaveButton;


// ── Spawning ─────────────────────────────────────────────────────────────────

/// Append the statistics-panel UI subtree as a child of the layout root.
/// Called from `frontend::setup_panes`; expects to be invoked inside the
/// `with_children` closure for the root node. Returns the panel's height in
/// logical pixels for use by other systems if needed (currently unused).
pub fn spawn_panel(
    root:                   &mut ChildSpawnerCommands,
    images:                 &mut Assets<Image>,
    graph:                  &mut GraphState,
    initial_panel_logical:  f32,
) {
    // Graph texture. Same sRGB-view trick as the viewport: storage is
    // Rgba8Unorm but the sampler treats bytes as already-sRGB, so the
    // colour bytes we write match what the user sees on a sRGB swapchain.
    // 1×1 placeholder; `graph_redraw` resizes to the node's real pixel
    // size on the first frame after layout.
    let graph_image = images.add(Image::new_target_texture(
        1, 1,
        TextureFormat::Rgba8Unorm,
        Some(TextureFormat::Rgba8UnormSrgb),
    ));
    graph.image = graph_image.clone();

    root.spawn((
        StatisticsPanel,
        Node {
            width:          Val::Percent(100.0),
            height:         Val::Px(initial_panel_logical),
            flex_direction: FlexDirection::Row,
            align_items:    AlignItems::FlexStart,
            padding:        UiRect {
                left:   Val::Px(12.0),
                right:  Val::Px(12.0),
                top:    Val::Px(8.0),
                bottom: Val::Px(8.0),
            },
            flex_shrink: 0.0,
            ..default()
        },
        BackgroundColor(PANEL_BG_COLOR),
    ))
    .with_children(|panel| {
        // ── Left section: FPS + Cells (anchored to top-left). ──────────
        panel
            .spawn(Node {
                flex_direction: FlexDirection::Column,
                align_items:    AlignItems::FlexStart,
                margin:         UiRect::right(Val::Px(12.0)),
                ..default()
            })
            .with_children(|left| {
                left.spawn((
                    Text::new("FPS: 0.0"),
                    TextFont { font_size: 20.0, ..default() },
                    TextColor(Color::WHITE),
                    FpsText { timer: Timer::from_seconds(0.05, TimerMode::Repeating) },
                ));
                left.spawn((
                    Text::new("Cells: 0"),
                    TextFont { font_size: 20.0, ..default() },
                    TextColor(Color::WHITE),
                    CellCountText { timer: Timer::from_seconds(0.2, TimerMode::Repeating) },
                ));
            });

        // ── Vertical line separating the FPS/Cells block from the rest.
        panel.spawn((
            Node {
                width:       Val::Px(VERTICAL_LINE_WIDTH),
                height:      Val::Percent(100.0),
                flex_shrink: 0.0,
                ..default()
            },
            BackgroundColor(VERTICAL_LINE_COLOR),
        ));

        // ── Spacer so the right section pushes to the panel's right
        //    edge while the graph + button float over it via absolute
        //    positioning.
        panel.spawn(Node {
            flex_grow: 1.0,
            ..default()
        });

        // ── Right section: Photo / Hetero counts. ──────────────────────
        panel
            .spawn(Node {
                flex_direction: FlexDirection::Column,
                align_items:    AlignItems::FlexEnd,
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

        // ── Graph (absolute). The top edge sits BUTTON_GAP_PX below the
        //    button — the same pixel inset the button itself uses
        //    against the grey divider above the panel — so the graph's
        //    distance to the button mirrors the button's distance to
        //    the separator. The graph fills downward to the panel
        //    bottom via `bottom: Px(0)`.
        panel.spawn((
            GraphImage,
            ImageNode::new(graph_image.clone()),
            Node {
                position_type: PositionType::Absolute,
                left:   Val::Vw(25.0),
                top:    Val::Px(BUTTON_GAP_PX + BUTTON_HEIGHT_PX + BUTTON_GAP_PX),
                width:  Val::Vw(50.0),
                bottom: Val::Px(0.0),
                ..default()
            },
            Pickable::IGNORE,
        ));

        // ── Save button (absolute, bottom-right of the panel). ─────
        panel
            .spawn((
                SaveButton,
                Button,
                Node {
                    position_type: PositionType::Absolute,
                    bottom: Val::Px(BUTTON_GAP_PX),
                    right:  Val::Px(BUTTON_GAP_PX),
                    width:  Val::Px(BUTTON_WIDTH_PX),
                    height: Val::Px(BUTTON_HEIGHT_PX),
                    justify_content: JustifyContent::Center,
                    align_items:     AlignItems::Center,
                    ..default()
                },
                BackgroundColor(BUTTON_COLOR_SAVE),
            ))
            .with_children(|btn| {
                btn.spawn((
                    Text::new("Save"),
                    TextFont { font_size: 18.0, ..default() },
                    TextColor(Color::WHITE),
                    Pickable::IGNORE,
                ));
            });

        // ── Start/Stop button (absolute, top-centred in the lounge
        //    space above the graph). ────────────────────────────────
        panel
            .spawn((
                StartStopButton,
                Button,
                Node {
                    position_type: PositionType::Absolute,
                    top:    Val::Px(BUTTON_GAP_PX),
                    // Centre horizontally: pull the left edge to the
                    // window's mid-point, then offset by half the
                    // button's width via a negative margin.
                    left:   Val::Vw(50.0),
                    margin: UiRect::left(Val::Px(-BUTTON_WIDTH_PX / 2.0)),
                    width:  Val::Px(BUTTON_WIDTH_PX),
                    height: Val::Px(BUTTON_HEIGHT_PX),
                    justify_content: JustifyContent::Center,
                    align_items:     AlignItems::Center,
                    ..default()
                },
                BackgroundColor(BUTTON_COLOR_PAUSED),
            ))
            .with_children(|btn| {
                btn.spawn((
                    StartStopButtonText,
                    Text::new("Start"),
                    TextFont { font_size: 18.0, ..default() },
                    TextColor(Color::WHITE),
                    Pickable::IGNORE,
                ));
            });
    });
}


// ── Systems ──────────────────────────────────────────────────────────────────

pub fn update_fps_text(
    time:        Res<Time>,
    diagnostics: Res<DiagnosticsStore>,
    mut query:   Query<(&mut Text, &mut FpsText)>,
) {
    for (mut text, mut fps_marker) in &mut query {
        fps_marker.timer.tick(time.delta());
        if fps_marker.timer.just_finished() {
            if let Some(fps_diag) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) {
                if let Some(fps_value) = fps_diag.smoothed() {
                    // `text.0.clear() + write!` reuses the existing String
                    // buffer instead of allocating a new one each tick.
                    text.0.clear();
                    let _ = write!(text.0, "FPS: {:.1}", fps_value);
                }
            }
        }
    }
}


pub fn update_cell_count_text(
    time:      Res<Time>,
    organisms: Query<&Organism, With<OrganismRoot>>,
    mut query: Query<(&mut Text, &mut CellCountText)>,
) {
    for (mut text, mut marker) in &mut query {
        marker.timer.tick(time.delta());
        if marker.timer.just_finished() {
            // Sum the cached counts directly — `grown_cell_count`
            // walks `body_parts.iter().filter(...).map(|bp| bp.ocg.len())`
            // every call. With photo+non_photo cached on Organism we
            // avoid the body-part walk entirely.
            let total: i64 = organisms.iter()
                .map(|o| (o.photo_cell_count + o.non_photo_cell_count) as i64)
                .sum();
            text.0.clear();
            let _ = write!(text.0, "Cells: {}", total);
        }
    }
}


pub fn track_organism_births(
    photo_added:  Query<Entity, Added<Photoautotroph>>,
    hetero_added: Query<Entity, Added<Heterotroph>>,
    mut counts:   ResMut<OrganismCounts>,
) {
    let photo_n  = photo_added.iter().count() as i32;
    let hetero_n = hetero_added.iter().count() as i32;
    if photo_n > 0  { counts.photo  += photo_n;  }
    if hetero_n > 0 { counts.hetero += hetero_n; }
}

pub fn track_organism_deaths(
    mut photo_removed:  RemovedComponents<Photoautotroph>,
    mut hetero_removed: RemovedComponents<Heterotroph>,
    mut counts:         ResMut<OrganismCounts>,
) {
    let photo_n  = photo_removed.read().count() as i32;
    let hetero_n = hetero_removed.read().count() as i32;
    if photo_n > 0  { counts.photo  -= photo_n;  }
    if hetero_n > 0 { counts.hetero -= hetero_n; }
}

pub fn update_counter_texts(
    counts:       Res<OrganismCounts>,
    mut photo_q:  Query<&mut Text, (With<PhotoCountText>, Without<HeteroCountText>)>,
    mut hetero_q: Query<&mut Text, (With<HeteroCountText>, Without<PhotoCountText>)>,
) {
    if !counts.is_changed() { return; }
    for mut text in &mut photo_q  {
        text.0.clear();
        let _ = write!(text.0, "Phototrophic: {}",  counts.photo);
    }
    for mut text in &mut hetero_q {
        text.0.clear();
        let _ = write!(text.0, "Heterotrophic: {}", counts.hetero);
    }
}


/// 1 Hz: capture a snapshot of `OrganismCounts`, push it onto the front of
/// the ring buffer, drop the oldest sample once we exceed `GRAPH_DURATION_SECS`
/// seconds. Marks the graph dirty so the redraw system rasterises this frame.
pub fn graph_tick(
    time:      Res<Time>,
    counts:    Res<OrganismCounts>,
    mut graph: ResMut<GraphState>,
) {
    graph.timer.tick(time.delta());
    if !graph.timer.just_finished() { return; }
    graph.samples.push_front((counts.photo, counts.hetero));
    while graph.samples.len() > GRAPH_SAMPLE_CAP {
        graph.samples.pop_back();
    }
    graph.dirty = true;
}

/// CPU-side rasteriser. Touches the GPU only when (a) the node's pixel size
/// changed (resize → reallocate texture buffer, redraw) or (b) `graph_tick`
/// pushed a new sample (1 Hz redraw at most). In all other frames this
/// system returns after a single `==` check on UVec2 — no allocation, no
/// upload.
pub fn graph_redraw(
    mut images: ResMut<Assets<Image>>,
    mut graph:  ResMut<GraphState>,
    node_q:     Query<&ComputedNode, With<GraphImage>>,
) {
    let Ok(node) = node_q.single() else { return };
    let size = node.size();
    let new_w = size.x.max(2.0).round() as u32;
    let new_h = size.y.max(2.0).round() as u32;
    let new_size = UVec2::new(new_w, new_h);
    let resized  = new_size != graph.image_size;

    if !graph.dirty && !resized { return; }

    let Some(image) = images.get_mut(&graph.image) else { return };
    if resized {
        image.resize(Extent3d {
            width:  new_w,
            height: new_h,
            depth_or_array_layers: 1,
        });
        graph.image_size = new_size;
    }

    let Some(buf) = image.data.as_mut() else { return };
    rasterise_graph(buf, new_w as i32, new_h as i32, &graph.samples);
    graph.dirty = false;
}


// ── Rasteriser ───────────────────────────────────────────────────────────────

/// Pure pixel-twiddling routine. ~ (w * h * 4) bytes for the background
/// fill plus ~60 Bresenham segments per line, called at most once per
/// second (or on resize). No allocations.
fn rasterise_graph(buf: &mut [u8], w: i32, h: i32, samples: &VecDeque<(i32, i32)>) {
    // Background.
    for chunk in buf.chunks_exact_mut(4) {
        chunk.copy_from_slice(&GRAPH_BG);
    }
    if samples.len() < 2 { return; }

    // Auto-scale y to the largest count currently on screen, with a floor.
    let mut y_max_i: i32 = 0;
    for &(p, het) in samples.iter() {
        if p   > y_max_i { y_max_i = p;   }
        if het > y_max_i { y_max_i = het; }
    }
    let y_max = (y_max_i as f32).max(GRAPH_Y_MIN_RANGE);

    // x-axis covers exactly GRAPH_DURATION_SECS samples → (w-1) px.
    let x_scale = (w - 1) as f32 / GRAPH_DURATION_SECS as f32;
    let y_scale = (h - 1) as f32 / y_max;

    // Walk consecutive samples once and draw both lines per segment.
    let mut prev_x = 0;
    let mut prev_photo_y  = h - 1 - ((samples[0].0 as f32 * y_scale).round() as i32).clamp(0, h - 1);
    let mut prev_hetero_y = h - 1 - ((samples[0].1 as f32 * y_scale).round() as i32).clamp(0, h - 1);
    for i in 1..samples.len() {
        let (p, het) = samples[i];
        let x  = ((i as f32) * x_scale).round() as i32;
        let py = h - 1 - ((p   as f32 * y_scale).round() as i32).clamp(0, h - 1);
        let hy = h - 1 - ((het as f32 * y_scale).round() as i32).clamp(0, h - 1);
        draw_line(buf, w, h, prev_x, prev_photo_y,  x, py, GRAPH_PHOTO_COLOR);
        draw_line(buf, w, h, prev_x, prev_hetero_y, x, hy, GRAPH_HETERO_COLOR);
        prev_x = x;
        prev_photo_y  = py;
        prev_hetero_y = hy;
    }
}

#[inline(always)]
fn put_pixel(buf: &mut [u8], w: i32, h: i32, x: i32, y: i32, color: [u8; 4]) {
    if (x as u32) >= w as u32 || (y as u32) >= h as u32 { return; }
    let idx = ((y * w + x) as usize) * 4;
    buf[idx..idx + 4].copy_from_slice(&color);
}

// ── Start/Stop button systems ────────────────────────────────────────────────

/// Click + hover handler for the Start/Stop button. On `Pressed`, toggles
/// `SimulationRunning` and the matching `Time<Virtual>` pause state.
/// Hover state colour is applied here too so the visuals are in one
/// place. Player controls (capture state) are independent of pause —
/// the user can fly around a frozen world to inspect it.
pub fn handle_start_stop_button(
    mut interactions:  Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<StartStopButton>)>,
    mut sim_running:   ResMut<SimulationRunning>,
    mut virtual_time:  ResMut<Time<Virtual>>,
) {
    for (interaction, mut bg) in &mut interactions {
        match *interaction {
            Interaction::Pressed => {
                sim_running.0 = !sim_running.0;
                if sim_running.0 {
                    virtual_time.unpause();
                    *bg = BackgroundColor(BUTTON_COLOR_RUN);
                } else {
                    virtual_time.pause();
                    *bg = BackgroundColor(BUTTON_COLOR_PAUSED);
                }
            }
            Interaction::Hovered => {
                *bg = BackgroundColor(BUTTON_COLOR_HOVER);
            }
            Interaction::None => {
                *bg = BackgroundColor(if sim_running.0 {
                    BUTTON_COLOR_RUN
                } else {
                    BUTTON_COLOR_PAUSED
                });
            }
        }
    }
}

/// Click handler for the Save button. On `Pressed`, pauses the
/// simulation and flips `SaveRequested(true)` — the actual file write
/// happens in `colony.rs::save_colony_system` on the next Update tick,
/// since the click handler doesn't have access to organism queries.
pub fn handle_save_button(
    mut interactions:   Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<SaveButton>)>,
    mut sim_running:    ResMut<SimulationRunning>,
    mut virtual_time:   ResMut<Time<Virtual>>,
    mut save_requested: ResMut<SaveRequested>,
) {
    for (interaction, mut bg) in &mut interactions {
        match *interaction {
            Interaction::Pressed => {
                if sim_running.0 {
                    sim_running.0 = false;
                    virtual_time.pause();
                }
                save_requested.0 = true;
                *bg = BackgroundColor(BUTTON_COLOR_HOVER);
            }
            Interaction::Hovered => *bg = BackgroundColor(BUTTON_COLOR_HOVER),
            Interaction::None    => *bg = BackgroundColor(BUTTON_COLOR_SAVE),
        }
    }
}


/// Sync the button's text label AND background colour with the current
/// simulation state. Runs every frame; cheap because it only writes
/// when `SimulationRunning` has changed (which on the very first tick
/// covers the resource being newly inserted, so the spawn-time defaults
/// in `spawn_panel` are guaranteed to be overwritten before the user
/// sees the panel).
pub fn update_start_stop_label(
    sim_running:  Res<SimulationRunning>,
    mut texts:    Query<&mut Text, With<StartStopButtonText>>,
    mut buttons:  Query<&mut BackgroundColor, With<StartStopButton>>,
) {
    if !sim_running.is_changed() { return; }
    let label = if sim_running.0 { "Pause" } else { "Start" };
    let color = if sim_running.0 { BUTTON_COLOR_RUN } else { BUTTON_COLOR_PAUSED };
    for mut text in &mut texts {
        text.0 = label.to_string();
    }
    for mut bg in &mut buttons {
        *bg = BackgroundColor(color);
    }
}


/// Bresenham. Handles every octant.
fn draw_line(buf: &mut [u8], w: i32, h: i32, x0: i32, y0: i32, x1: i32, y1: i32, color: [u8; 4]) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let (mut x, mut y) = (x0, y0);
    loop {
        put_pixel(buf, w, h, x, y, color);
        if x == x1 && y == y1 { break; }
        let e2 = 2 * err;
        if e2 >= dy { err += dy; x += sx; }
        if e2 <= dx { err += dx; y += sy; }
    }
}
