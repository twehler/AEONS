// Statistics panel — the bottom UI strip. Built once via
// `frontend::setup_panes::spawn_panel`, driven by the systems at the
// bottom of this file. Vertical extent owned by `frontend.rs::divider_drag`.

use std::collections::VecDeque;
use std::fmt::Write as _;

use bevy::prelude::*;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::input::keyboard::KeyboardInput;
use bevy::render::render_resource::Extent3d;
use bevy::render::render_resource::TextureFormat;

use crate::colony::{Photoautotroph, Heterotroph, Organism, OrganismRoot, SaveRequested};
use crate::frontend::PANEL_BG_COLOR;
use crate::simulation_settings::{SimulationRunning, TimeSpeed, MaxPhotoautotrophs};

use rand::prelude::*;


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

/// Symmetric gap (logical px) above AND below the Start/Stop button; the
/// graph's top inset below the button mirrors the button's inset below the
/// grey separator.
const BUTTON_GAP_PX: f32 = 8.0;

/// Start/Stop button styling.
const BUTTON_WIDTH_PX:        f32 = 110.0;
const BUTTON_HEIGHT_PX:       f32 = 36.0;
const BUTTON_COLOR_RUN:       Color = Color::srgb(0.55, 0.18, 0.18); // red — "Pause"
const BUTTON_COLOR_PAUSED:    Color = Color::srgb(0.18, 0.55, 0.22); // green — "Start"
const BUTTON_COLOR_HOVER:     Color = Color::srgb(0.30, 0.30, 0.30);

/// Save button styling — sits in the bottom-right of the panel.
const BUTTON_COLOR_SAVE:      Color = Color::srgb(0.20, 0.40, 0.65); // calm blue

/// Speed-input visual constants. Two background colours signal when the
/// field is editable.
const SPEED_INPUT_WIDTH_PX:   f32   = 88.0;
const SPEED_INPUT_HEIGHT_PX:  f32   = 26.0;
const SPEED_INPUT_BG_IDLE:    Color = Color::srgb(0.20, 0.20, 0.22);
const SPEED_INPUT_BG_FOCUSED: Color = Color::srgb(0.32, 0.30, 0.18);
/// Max editable-buffer length; bounds paste length so formatting can't clog.
const SPEED_BUFFER_MAX_LEN:   usize = 8;

/// Max-organisms input field — same widget pattern as the speed input.
const MAX_PHOTO_INPUT_WIDTH_PX:  f32   = 88.0;
const MAX_PHOTO_INPUT_HEIGHT_PX: f32   = 26.0;
const MAX_PHOTO_BG_IDLE:         Color = Color::srgb(0.20, 0.20, 0.22);
const MAX_PHOTO_BG_FOCUSED:      Color = Color::srgb(0.32, 0.30, 0.18);
const MAX_PHOTO_BUFFER_MAX_LEN:  usize = 8;

/// Font size for the photo / hetero count texts.
const COUNT_FONT_SIZE:         f32 = 14.0;

/// Spacing between the Max-Organisms input row and the counters below.
const COUNT_TOP_MARGIN_PX:     f32 = 14.0;

/// Cull-notification orange (matches the speed-input warning hue).
const CULL_MSG_COLOR:          Color = Color::srgb(0.95, 0.55, 0.15);

/// How long the orange "{N} organisms have been removed…" toast stays
/// visible after a successful cull.
const CULL_MSG_VISIBLE_SECS:   f32 = 8.0;


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
/// older than `GRAPH_DURATION_SECS`. `dirty` (set on tick + resize, cleared
/// after rasterising) avoids touching the GPU texture on idle frames.
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

// Marker structs are `pub` because the pub systems below reference them in
// `Query` signatures — `private_interfaces` rejects pub-fn / private-type
// mismatches even when the type is crate-internal.
#[derive(Component)]
pub struct FpsText { timer: Timer }

#[derive(Component)]
pub struct CellCountText { timer: Timer }

/// Marker on the camera-coordinate text beneath the FPS counter. No timer:
/// updated purely on `Changed<Transform>` of the `FlyCam`.
#[derive(Component)]
pub struct CameraCoordsText;

/// Marker on the simulation-clock text. Reads `Time<Virtual>` so it pauses
/// with the sim; the timer throttles formatting to 2 Hz.
#[derive(Component)]
pub struct SimTimerText { timer: Timer }

/// Marker on the clickable speed-input box (outer Node + Button); used to
/// detect "click on input" vs "click outside".
#[derive(Component)]
pub struct TimeSpeedInput;

/// Marker on the Text child inside the speed-input box (shows the committed
/// value or the in-progress edit buffer).
#[derive(Component)]
pub struct TimeSpeedText;

/// Marker on the orange "(Be careful with this!)" warning below the
/// speed-input field. Hidden unless the field is being edited.
#[derive(Component)]
pub struct TimeSpeedWarning;

/// Edit-state for the speed-input field. A resource (not on the Node) so
/// sibling systems can inspect `focused`.
#[derive(Resource, Default)]
pub struct TimeSpeedEditState {
    /// Live edit buffer (only meaningful when `focused`).
    pub buffer:  String,
    /// True while typing. Click-on-input → true; Enter/Escape/click-outside → false.
    pub focused: bool,
}

#[derive(Component)]
pub struct PhotoCountText;

#[derive(Component)]
pub struct HeteroCountText;

/// Click target for the "Max Organisms" integer-input field.
#[derive(Component)]
pub struct MaxPhotoautotrophsInput;

/// Marker on the Text child inside the Max-Organisms input box.
#[derive(Component)]
pub struct MaxPhotoautotrophsText;

/// Edit-state for the Max-Organisms input. Same model as `TimeSpeedEditState`.
#[derive(Resource, Default)]
pub struct MaxPhotoautotrophsEditState {
    pub buffer:  String,
    pub focused: bool,
}

/// Marker on the orange cull-notification Text node. Hidden by default;
/// `update_cull_message` flips Display while the message is live.
#[derive(Component)]
pub struct CullMessageText;

/// Clickable square of the "AI-training mode" checkbox sitting
/// directly to the right of the population graph. Toggles the
/// `AiTrainingMode` resource on `Interaction::Pressed`.
#[derive(Component)]
pub struct AiTrainingCheckbox;

/// Inner filled square that visualises the checkbox state — its
/// `Display` is flipped between `Flex` (on) and `None` (off) by
/// `update_ai_training_checkbox_mark`.
#[derive(Component)]
pub struct AiTrainingCheckboxMark;

/// Outer Button-tagged Node for the "Max Herbivores" integer input.
#[derive(Component)]
pub struct MaxHerbivoresInput;

/// Text child inside the Max-Herbivores input box.
#[derive(Component)]
pub struct MaxHerbivoresText;

/// Edit-state for the Max-Herbivores input — mirrors `MaxPhotoautotrophsEditState`.
#[derive(Resource, Default)]
pub struct MaxHerbivoresEditState {
    pub buffer:  String,
    pub focused: bool,
}

/// Notification state for the random-cull toast. Set by
/// `apply_max_phototrophs_cull`; auto-cleared after `CULL_MSG_VISIBLE_SECS`.
#[derive(Resource, Default)]
pub struct CullMessage {
    /// True while the message should be visible.
    pub active: bool,
    /// Most recent cull count; used to format the message.
    pub count:  usize,
    /// Counts up while `active`; flips inactive at `CULL_MSG_VISIBLE_SECS`.
    pub elapsed: f32,
}

#[derive(Component)]
pub struct GraphImage;

/// Marker on the Start/Stop button itself.
#[derive(Component)]
pub struct StartStopButton;

/// Marker on the Text entity inside the Start/Stop button.
#[derive(Component)]
pub struct StartStopButtonText;

/// Marker on the Save button (bottom-right of the statistics panel).
#[derive(Component)]
pub struct SaveButton;

/// Marker on the "Export Simulation Dataset" button — sits in the
/// top strip immediately right of the Start/Stop button. Click
/// pauses the simulation, opens a native save dialog, and writes
/// the chosen `.csv` path into `ExportDatasetRequested`.
#[derive(Component)]
pub struct ExportDatasetButton;


// ── Spawning ─────────────────────────────────────────────────────────────────

/// Append the statistics-panel UI subtree as a child of the layout root.
/// Must be invoked inside the root node's `with_children` closure.
pub fn spawn_panel(
    root:                   &mut ChildSpawnerCommands,
    images:                 &mut Assets<Image>,
    graph:                  &mut GraphState,
    initial_panel_logical:  f32,
) {
    // Graph texture. Same sRGB-view trick as the viewport: storage is
    // Rgba8Unorm but the sampler treats bytes as already-sRGB, so written
    // colour bytes match what's seen on a sRGB swapchain. 1×1 placeholder;
    // `graph_redraw` resizes to the node's real pixel size after layout.
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
                    // Updated every 0.1 s of REAL time (see `update_fps_text`);
                    // neither cadence nor reading scales with `TimeSpeed`.
                    Text::new("FPS: 0.0"),
                    TextFont { font_size: 20.0, ..default() },
                    TextColor(Color::WHITE),
                    FpsText { timer: Timer::from_seconds(0.1, TimerMode::Repeating) },
                ));
                left.spawn((
                    // Camera position; refreshed only on FlyCam transform change.
                    Text::new("Cam: -, -, -"),
                    TextFont { font_size: 14.0, ..default() },
                    TextColor(Color::srgb(0.7, 0.7, 0.7)),
                    CameraCoordsText,
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

        // ── Simulation clock. Sits left of the absolutely-positioned graph
        //    (which starts at 25vw); fixed width so it never overflows into it.
        panel
            .spawn(Node {
                flex_direction: FlexDirection::Column,
                align_items:    AlignItems::FlexStart,
                margin:         UiRect::left(Val::Px(12.0)),
                flex_shrink:    0.0,
                ..default()
            })
            .with_children(|col| {
                col.spawn((
                    Text::new("Sim Time"),
                    TextFont { font_size: 12.0, ..default() },
                    TextColor(Color::srgb(0.70, 0.70, 0.70)),
                    Pickable::IGNORE,
                ));
                col.spawn((
                    Text::new("000:000:00:00:00"),
                    TextFont { font_size: 20.0, ..default() },
                    TextColor(Color::WHITE),
                    SimTimerText {
                        timer: Timer::from_seconds(0.5, TimerMode::Repeating),
                    },
                ));

                // ── Speed-input field. ───────────────────────────────
                col.spawn((
                    Text::new("Speed"),
                    TextFont { font_size: 12.0, ..default() },
                    TextColor(Color::srgb(0.70, 0.70, 0.70)),
                    Node { margin: UiRect::top(Val::Px(6.0)), ..default() },
                    Pickable::IGNORE,
                ));
                col
                    .spawn((
                        TimeSpeedInput,
                        Button,
                        Node {
                            width:   Val::Px(SPEED_INPUT_WIDTH_PX),
                            height:  Val::Px(SPEED_INPUT_HEIGHT_PX),
                            padding: UiRect::axes(Val::Px(6.0), Val::Px(2.0)),
                            align_items:     AlignItems::Center,
                            justify_content: JustifyContent::FlexStart,
                            ..default()
                        },
                        BackgroundColor(SPEED_INPUT_BG_IDLE),
                    ))
                    .with_children(|btn| {
                        btn.spawn((
                            TimeSpeedText,
                            Text::new("1.00x"),
                            TextFont { font_size: 14.0, ..default() },
                            TextColor(Color::WHITE),
                            Pickable::IGNORE,
                        ));
                    });
                col.spawn((
                    TimeSpeedWarning,
                    Text::new("(Be careful with this!)"),
                    TextFont { font_size: 11.0, ..default() },
                    TextColor(Color::srgb(0.95, 0.55, 0.15)),
                    // Starts hidden — `update_time_speed_text` shows it while
                    // the speed field is focused.
                    Node {
                        margin:  UiRect::top(Val::Px(2.0)),
                        display: Display::None,
                        ..default()
                    },
                    Pickable::IGNORE,
                ));
            });

        // ── Spacer so the right section pushes to the panel's right
        //    edge while the graph + button float over it via absolute
        //    positioning.
        panel.spawn(Node {
            flex_grow: 1.0,
            ..default()
        });

        // ── Right section: Max-Organisms input on top, Photo / Hetero
        //    counts pushed down below it. ──────────────────────────────
        panel
            .spawn(Node {
                flex_direction: FlexDirection::Column,
                align_items:    AlignItems::FlexEnd,
                ..default()
            })
            .with_children(|right| {
                // ── "Max Phototrophic Organisms:" row (label + integer input). ──
                right
                    .spawn(Node {
                        flex_direction: FlexDirection::Row,
                        align_items:    AlignItems::Center,
                        ..default()
                    })
                    .with_children(|row| {
                        row.spawn((
                            Text::new("Max Phototrophic Organisms:"),
                            TextFont { font_size: 14.0, ..default() },
                            TextColor(Color::WHITE),
                            Node { margin: UiRect::right(Val::Px(8.0)), ..default() },
                            Pickable::IGNORE,
                        ));
                        row
                            .spawn((
                                MaxPhotoautotrophsInput,
                                Button,
                                Node {
                                    width:   Val::Px(MAX_PHOTO_INPUT_WIDTH_PX),
                                    height:  Val::Px(MAX_PHOTO_INPUT_HEIGHT_PX),
                                    padding: UiRect::axes(Val::Px(6.0), Val::Px(2.0)),
                                    align_items:     AlignItems::Center,
                                    justify_content: JustifyContent::FlexStart,
                                    ..default()
                                },
                                BackgroundColor(MAX_PHOTO_BG_IDLE),
                            ))
                            .with_children(|btn| {
                                btn.spawn((
                                    MaxPhotoautotrophsText,
                                    Text::new("0"),
                                    TextFont { font_size: 14.0, ..default() },
                                    TextColor(Color::WHITE),
                                    Pickable::IGNORE,
                                ));
                            });
                    });

                right.spawn((
                    Text::new("Phototrophic: 0"),
                    TextFont { font_size: COUNT_FONT_SIZE, ..default() },
                    TextColor(Color::WHITE),
                    Node { margin: UiRect::top(Val::Px(COUNT_TOP_MARGIN_PX)), ..default() },
                    PhotoCountText,
                ));
                right.spawn((
                    Text::new("Heterotrophic: 0"),
                    TextFont { font_size: COUNT_FONT_SIZE, ..default() },
                    TextColor(Color::WHITE),
                    HeteroCountText,
                ));

                // ── Orange cull notification (initially hidden). ──
                right.spawn((
                    CullMessageText,
                    Text::new(""),
                    TextFont { font_size: 13.0, ..default() },
                    TextColor(CULL_MSG_COLOR),
                    Node {
                        margin:    UiRect::top(Val::Px(8.0)),
                        max_width: Val::Px(380.0),
                        display:   Display::None,
                        ..default()
                    },
                    Pickable::IGNORE,
                ));
            });

        // ── Graph (absolute). Top edge sits BUTTON_GAP_PX below the button,
        //    mirroring the button's inset below the separator; fills down to
        //    the panel bottom via `bottom: Px(0)`.
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

        // ── Right-of-graph control stack (absolute): AI-training checkbox +
        //    Max-Herbivores field. Anchored at the graph's right edge
        //    (`left = 75vw`) to avoid the bottom-right Save button.
        const AI_CHECKBOX_SIZE_PX: f32 = 16.0;
        panel
            .spawn(Node {
                position_type:  PositionType::Absolute,
                left:           Val::Vw(75.0),
                top:            Val::Px(BUTTON_GAP_PX + BUTTON_HEIGHT_PX + BUTTON_GAP_PX),
                margin:         UiRect::left(Val::Px(10.0)),
                flex_direction: FlexDirection::Column,
                align_items:    AlignItems::FlexStart,
                ..default()
            })
            .with_children(|stack| {
                // ── AI-training mode checkbox row ───────────────────
                stack
                    .spawn(Node {
                        flex_direction: FlexDirection::Row,
                        align_items:    AlignItems::Center,
                        ..default()
                    })
                    .with_children(|row| {
                        row.spawn((
                            AiTrainingCheckbox,
                            Button,
                            Node {
                                width:           Val::Px(AI_CHECKBOX_SIZE_PX),
                                height:          Val::Px(AI_CHECKBOX_SIZE_PX),
                                margin:          UiRect::right(Val::Px(8.0)),
                                align_items:     AlignItems::Center,
                                justify_content: JustifyContent::Center,
                                flex_shrink:     0.0,
                                ..default()
                            },
                            BackgroundColor(Color::srgb(0.10, 0.10, 0.10)),
                        ))
                        .with_children(|cb| {
                            cb.spawn((
                                AiTrainingCheckboxMark,
                                Node {
                                    width:   Val::Px(AI_CHECKBOX_SIZE_PX - 6.0),
                                    height:  Val::Px(AI_CHECKBOX_SIZE_PX - 6.0),
                                    // Hidden until the user toggles on.
                                    display: Display::None,
                                    ..default()
                                },
                                BackgroundColor(Color::srgb(0.85, 0.85, 0.85)),
                                Pickable::IGNORE,
                            ));
                        });
                        row.spawn((
                            Text::new("AI-training mode"),
                            TextFont { font_size: 14.0, ..default() },
                            TextColor(Color::WHITE),
                            Pickable::IGNORE,
                        ));
                    });

                // ── "Max Herbivores: [N]" integer-input row. Same widget as
                //    Max Organisms but unclamped on the upper side — gates
                //    reproduction, no GPU batch dim involved.
                stack
                    .spawn(Node {
                        flex_direction: FlexDirection::Row,
                        align_items:    AlignItems::Center,
                        margin:         UiRect::top(Val::Px(8.0)),
                        ..default()
                    })
                    .with_children(|row| {
                        row.spawn((
                            Text::new("Max Herbivores:"),
                            TextFont { font_size: 14.0, ..default() },
                            TextColor(Color::WHITE),
                            Node { margin: UiRect::right(Val::Px(8.0)), ..default() },
                            Pickable::IGNORE,
                        ));
                        row
                            .spawn((
                                MaxHerbivoresInput,
                                Button,
                                Node {
                                    width:   Val::Px(MAX_PHOTO_INPUT_WIDTH_PX),
                                    height:  Val::Px(MAX_PHOTO_INPUT_HEIGHT_PX),
                                    padding: UiRect::axes(Val::Px(6.0), Val::Px(2.0)),
                                    align_items:     AlignItems::Center,
                                    justify_content: JustifyContent::FlexStart,
                                    ..default()
                                },
                                BackgroundColor(MAX_PHOTO_BG_IDLE),
                            ))
                            .with_children(|btn| {
                                btn.spawn((
                                    MaxHerbivoresText,
                                    Text::new("0"),
                                    TextFont { font_size: 14.0, ..default() },
                                    TextColor(Color::WHITE),
                                    Pickable::IGNORE,
                                ));
                            });
                    });
            });

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
                    // Centre horizontally: left edge at mid-point, offset
                    // back by half the button's width via negative margin.
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

        // ── "Export Simulation Dataset" button (absolute, top strip). Same
        //    Vw(50) anchor as Start/Stop but with a positive
        //    `BUTTON_WIDTH_PX/2 + gap` margin, landing it one gap past
        //    Start/Stop's right edge. Wider so the full label fits.
        const EXPORT_BUTTON_WIDTH_PX: f32 = 220.0;
        const EXPORT_BUTTON_COLOR:    Color = Color::srgb(0.22, 0.46, 0.46); // teal
        panel
            .spawn((
                ExportDatasetButton,
                Button,
                Node {
                    position_type: PositionType::Absolute,
                    top:    Val::Px(BUTTON_GAP_PX),
                    left:   Val::Vw(50.0),
                    margin: UiRect::left(Val::Px(BUTTON_WIDTH_PX / 2.0 + BUTTON_GAP_PX)),
                    width:  Val::Px(EXPORT_BUTTON_WIDTH_PX),
                    height: Val::Px(BUTTON_HEIGHT_PX),
                    justify_content: JustifyContent::Center,
                    align_items:     AlignItems::Center,
                    ..default()
                },
                BackgroundColor(EXPORT_BUTTON_COLOR),
            ))
            .with_children(|btn| {
                btn.spawn((
                    Text::new("Export Simulation Dataset"),
                    TextFont { font_size: 16.0, ..default() },
                    TextColor(Color::WHITE),
                    Pickable::IGNORE,
                ));
            });
    });
}


// ── Systems ──────────────────────────────────────────────────────────────────

pub fn update_fps_text(
    // `Time<Real>` (not the default `Time<Virtual>`) so the counter keeps
    // ticking at wall-clock pace through pause and high `TimeSpeed`.
    time:        Res<Time<bevy::time::Real>>,
    diagnostics: Res<DiagnosticsStore>,
    mut query:   Query<(&mut Text, &mut FpsText)>,
) {
    for (mut text, mut fps_marker) in &mut query {
        fps_marker.timer.tick(time.delta());
        if fps_marker.timer.just_finished() {
            if let Some(fps_diag) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) {
                if let Some(fps_value) = fps_diag.smoothed() {
                    // clear + write! reuses the buffer (no per-tick alloc).
                    text.0.clear();
                    let _ = write!(text.0, "FPS: {:.1}", fps_value);
                }
            }
        }
    }
}


/// Refresh the camera-coordinate readout beneath the FPS counter. Driven by
/// `Changed<Transform>` on the `FlyCam`, so idle frames are ~free; the text
/// buffer is rewritten in place (no alloc).
pub fn update_camera_coords_text(
    camera: Query<&Transform, (With<crate::player_plugin::FlyCam>, Changed<Transform>)>,
    mut query: Query<&mut Text, With<CameraCoordsText>>,
) {
    let Ok(cam_xf) = camera.single() else { return };
    let t = cam_xf.translation;
    for mut text in &mut query {
        text.0.clear();
        let _ = write!(text.0, "Cam: {:.0}, {:.0}, {:.0}", t.x, t.y, t.z);
    }
}


/// Format `total_secs` as `YYY:DDD:HH:MM:SS`. Year = 365 days, no leap
/// correction (simulation, not a calendar). Year field is `{:03}` below
/// 1000, switching to `{:.2e}` scientific at/above 1000 (worst case
/// `1.84e19` for u64::MAX, so the field stays ≤ 7 chars).
fn format_dhms(total_secs: u64) -> String {
    const SECS_PER_DAY:  u64 = 86_400;
    const SECS_PER_YEAR: u64 = SECS_PER_DAY * 365;

    let years = total_secs / SECS_PER_YEAR;
    let rem   = total_secs % SECS_PER_YEAR;
    let days  = rem / SECS_PER_DAY;
    let hours = (rem / 3_600) % 24;
    let mins  = (rem / 60)    % 60;
    let secs  = rem           % 60;

    if years >= 1_000 {
        format!(
            "{:.2e}:{:03}:{:02}:{:02}:{:02}",
            years as f64, days, hours, mins, secs,
        )
    } else {
        format!(
            "{:03}:{:03}:{:02}:{:02}:{:02}",
            years, days, hours, mins, secs,
        )
    }
}


pub fn update_sim_timer_text(
    real_time:    Res<Time>,
    virtual_time: Res<Time<Virtual>>,
    mut query:    Query<(&mut Text, &mut SimTimerText)>,
) {
    for (mut text, mut marker) in &mut query {
        // Throttle on REAL time so the refresh keeps firing while paused;
        // the displayed value uses virtual elapsed secs (frozen while paused).
        marker.timer.tick(real_time.delta());
        if !marker.timer.just_finished() { continue; }
        let total_secs = virtual_time.elapsed_secs() as u64;
        text.0.clear();
        text.0.push_str(&format_dhms(total_secs));
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
            // Sum the cached photo+non_photo counts to avoid a body-part walk.
            let total: i64 = organisms.iter()
                .map(|o| (o.photo_cell_count + o.non_photo_cell_count) as i64)
                .sum();
            text.0.clear();
            let _ = write!(text.0, "Cells: {}", total);
        }
    }
}


/// Direct population count — the source of truth. Recomputed from an
/// archetype query each frame (microseconds for a few-thousand-organism
/// world) instead of incremental `Added`/`RemovedComponents` tracking, which
/// drifts when `RemovedComponents`' bounded buffer drops events at high
/// `TimeSpeed`. The resource write is gated on a change check.
pub fn track_organism_births(
    photo_q:    Query<(), With<Photoautotroph>>,
    hetero_q:   Query<(), With<Heterotroph>>,
    mut counts: ResMut<OrganismCounts>,
) {
    let photo  = photo_q.iter().count()  as i32;
    let hetero = hetero_q.iter().count() as i32;
    if counts.photo != photo {
        counts.photo = photo;
    }
    if counts.hetero != hetero {
        counts.hetero = hetero;
    }
}

/// Public no-op kept so the `FrontendPlugin` registration site is unchanged;
/// `track_organism_births` now resyncs the full population (births + deaths).
pub fn track_organism_deaths() {}


/// TEMPORARY DIAGNOSTIC — breaks the photoautotroph population down by what's
/// visible, logged every 5 real seconds. Note `Mesh3d` lives on child
/// body-part entities (carrying `BodyPartIndex`), not the `OrganismRoot`, so
/// this walks each root's children. Reported buckets: root with a
/// mesh-bearing body-part child (rendering normally); root with body-part
/// children but no mesh (phantom plant); root with no body-part child at all.
pub fn diag_photo_breakdown(
    real_time:           Res<Time<bevy::time::Real>>,
    mut timer:           Local<Option<Timer>>,
    photo_roots:         Query<(Entity, &Organism, Option<&Children>), With<Photoautotroph>>,
    body_part_idx:       Query<&crate::cell::BodyPartIndex>,
    mesh_query:          Query<(), With<Mesh3d>>,
    all_bp_entities:     Query<Entity, With<crate::cell::BodyPartIndex>>,
    organism_mesh_q:     Query<Entity, With<crate::cell::OrganismMesh>>,
) {
    let timer = timer.get_or_insert_with(|| {
        Timer::from_seconds(5.0, TimerMode::Repeating)
    });
    if !timer.tick(real_time.delta()).just_finished() { return; }

    let mut total                 = 0usize;
    let mut alive_bp_in_data      = 0usize;
    let mut empty_bp_vec          = 0usize;
    let mut has_children_comp     = 0usize;
    let mut has_any_child         = 0usize;
    let mut has_bp_child          = 0usize;
    let mut has_mesh_bp_child     = 0usize;

    for (_root, organism, children) in photo_roots.iter() {
        total += 1;
        if organism.body_parts.is_empty() { empty_bp_vec += 1; }
        if organism.alive_body_part_count() > 0 { alive_bp_in_data += 1; }

        let mut any_child       = false;
        let mut bp_child        = false;
        let mut mesh_bp_child   = false;
        if let Some(children) = children {
            has_children_comp += 1;
            for c in children.iter() {
                any_child = true;
                if body_part_idx.get(c).is_ok() {
                    bp_child = true;
                    if mesh_query.get(c).is_ok() {
                        mesh_bp_child = true;
                        break;
                    }
                }
            }
        }
        if any_child      { has_any_child     += 1; }
        if bp_child       { has_bp_child      += 1; }
        if mesh_bp_child  { has_mesh_bp_child += 1; }
    }

    let total_bp_entities       = all_bp_entities.iter().count();
    let total_organism_mesh_ent = organism_mesh_q.iter().count();

    info!(
        "PHOTO_DIAG | total:{}  alive_bp(data):{}  empty_bp_vec:{}  \
         Children-comp:{}  any-child:{}  bp-child:{}  mesh-bp-child:{}  \
         |  world_bp_entities:{}  world_organism_meshes:{}",
        total, alive_bp_in_data, empty_bp_vec,
        has_children_comp, has_any_child, has_bp_child, has_mesh_bp_child,
        total_bp_entities, total_organism_mesh_ent,
    );
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


/// 1 Hz: push an `OrganismCounts` snapshot onto the front of the ring buffer,
/// drop samples beyond `GRAPH_DURATION_SECS`, mark the graph dirty.
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

/// CPU-side rasteriser. Touches the GPU only on resize or after `graph_tick`
/// dirties the graph (≤ 1 Hz); otherwise returns after a UVec2 size check.
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

/// Pure pixel-twiddling routine (background fill + Bresenham segments). No
/// allocations; called at most once per second (or on resize).
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
/// `SimulationRunning` and the matching `Time<Virtual>` pause state. Player
/// controls are independent of pause (fly around a frozen world to inspect).
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

/// Click handler for the Save button. On `Pressed`, pauses the sim, opens a
/// native "Save As" dialog (rfd), and stores the path in `SaveRequested` for
/// `colony.rs::save_colony_system` to write next tick (cancel = no write, sim
/// stays paused). rfd's `save_file()` blocks, stalling the sim for the dialog.
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

                let initial_dir = std::env::current_dir()
                    .unwrap_or_else(|_| std::path::PathBuf::from("."));
                let default_name = format!(
                    "colony_{}.colony",
                    chrono::Local::now().format("%d-%m-%Y-%H-%M-%S"),
                );
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("AEONS colony (.colony)", &["colony"])
                    .set_directory(initial_dir)
                    .set_file_name(default_name)
                    .save_file()
                {
                    save_requested.0 = Some(path);
                }
                *bg = BackgroundColor(BUTTON_COLOR_HOVER);
            }
            Interaction::Hovered => *bg = BackgroundColor(BUTTON_COLOR_HOVER),
            Interaction::None    => *bg = BackgroundColor(BUTTON_COLOR_SAVE),
        }
    }
}


/// Sync the button label + background colour with `SimulationRunning`. Writes
/// only on change; the first tick (resource newly inserted) overwrites the
/// spawn-time defaults before the panel is shown.
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


// ── Time-speed input handlers ───────────────────────────────────────────────

/// Click + keyboard router for the speed-input field.
///   * LMB on box → focus, prefill buffer with committed value.
///   * LMB outside while focused → commit (or discard) and unfocus.
///   * Focused keys: Enter → commit; Escape → cancel; Backspace → del;
///     digit/'.' → append (bounded by SPEED_BUFFER_MAX_LEN).
/// Commit writes `TimeSpeed`; `apply_time_speed` mirrors it onto
/// `Time<Virtual>`, propagating to every `Res<Time>` reader.
pub fn handle_time_speed_input(
    mouse:           Res<ButtonInput<MouseButton>>,
    mut keyboard:    MessageReader<KeyboardInput>,
    interaction_q:   Query<&Interaction, With<TimeSpeedInput>>,
    mut state:       ResMut<TimeSpeedEditState>,
    mut speed:       ResMut<TimeSpeed>,
) {
    let click_on_input = mouse.just_pressed(MouseButton::Left)
        && interaction_q.iter().any(|i| matches!(i, Interaction::Pressed));
    let click_outside  = mouse.just_pressed(MouseButton::Left) && !click_on_input;

    // ── Focus on click-on-input. ───────────────────────────────────
    if click_on_input && !state.focused {
        state.focused = true;
        state.buffer.clear();
        let _ = write!(state.buffer, "{:.2}", speed.0);
    }

    // ── Commit on click-outside. Keyboard is always drained below so
    //    events don't pile up while unfocused.
    if click_outside && state.focused {
        commit_time_speed(&mut state, &mut speed);
    }

    if !state.focused {
        // Drain to avoid event-buffer growth.
        for _ in keyboard.read() {}
        return;
    }

    for ev in keyboard.read() {
        if !ev.state.is_pressed() { continue; }
        match ev.key_code {
            KeyCode::Enter | KeyCode::NumpadEnter => {
                commit_time_speed(&mut state, &mut speed);
            }
            KeyCode::Escape => {
                state.focused = false;
                state.buffer.clear();
            }
            KeyCode::Backspace => {
                state.buffer.pop();
            }
            _ => {
                if let Some(text) = ev.text.as_ref() {
                    for c in text.chars() {
                        if state.buffer.len() >= SPEED_BUFFER_MAX_LEN { break; }
                        // Allow digits and a single decimal point.
                        if c.is_ascii_digit() {
                            state.buffer.push(c);
                        } else if c == '.' && !state.buffer.contains('.') {
                            state.buffer.push(c);
                        }
                    }
                }
            }
        }
    }
}

/// Parse the buffer; if it's a valid non-negative finite number,
/// write it into `TimeSpeed`. Always unfocus + clear the buffer.
fn commit_time_speed(state: &mut TimeSpeedEditState, speed: &mut TimeSpeed) {
    if let Ok(v) = state.buffer.parse::<f32>() {
        if v.is_finite() && v >= 0.0 {
            speed.0 = v;
        }
    }
    state.focused = false;
    state.buffer.clear();
}

/// Sync the input box's text + background colour with `TimeSpeed` and edit
/// state. Early-returns on `!is_changed()`.
pub fn update_time_speed_text(
    state:      Res<TimeSpeedEditState>,
    speed:      Res<TimeSpeed>,
    mut text_q: Query<&mut Text, With<TimeSpeedText>>,
    mut bg_q:   Query<&mut BackgroundColor, With<TimeSpeedInput>>,
    mut warn_q: Query<&mut Node, With<TimeSpeedWarning>>,
) {
    if !state.is_changed() && !speed.is_changed() { return; }

    let display = if state.focused {
        // Cursor indicator so the user can tell where typing lands.
        format!("{}_", state.buffer)
    } else {
        format!("{:.2}x", speed.0)
    };
    for mut text in &mut text_q {
        text.0 = display.clone();
    }

    let bg = if state.focused { SPEED_INPUT_BG_FOCUSED } else { SPEED_INPUT_BG_IDLE };
    for mut b in &mut bg_q {
        if b.0 != bg { *b = BackgroundColor(bg); }
    }

    // Warning shown only while the field is focused for editing.
    let want = if state.focused { Display::Flex } else { Display::None };
    for mut node in &mut warn_q {
        if node.display != want { node.display = want; }
    }
}

/// Mirror `TimeSpeed` onto `Time<Virtual>::set_relative_speed` on change
/// (incl. the first frame, which seeds 1.0 — harmless).
pub fn apply_time_speed(
    speed:           Res<TimeSpeed>,
    mut virtual_time: ResMut<Time<Virtual>>,
) {
    if !speed.is_changed() { return; }
    let s = speed.0.max(0.0);
    virtual_time.set_relative_speed(s);
}


// ── Max-Organisms input handlers ────────────────────────────────────────────

/// Click + keyboard router for the Max-Organisms field. Like
/// `handle_time_speed_input` but digits-only, committing a `usize` into
/// `MaxPhotoautotrophs`. Intentionally NOT clamped to `OrganismPoolSize`:
/// organisms spawned past the pool size just get no brain slot (harmless —
/// they keep their last brain output until recycled into a freed slot).
pub fn handle_max_phototrophs_input(
    mouse:            Res<ButtonInput<MouseButton>>,
    mut keyboard:     MessageReader<KeyboardInput>,
    interaction_q:    Query<&Interaction, With<MaxPhotoautotrophsInput>>,
    mut state:        ResMut<MaxPhotoautotrophsEditState>,
    mut max_photo:      ResMut<MaxPhotoautotrophs>,
) {
    let click_on_input = mouse.just_pressed(MouseButton::Left)
        && interaction_q.iter().any(|i| matches!(i, Interaction::Pressed));
    let click_outside  = mouse.just_pressed(MouseButton::Left) && !click_on_input;

    if click_on_input && !state.focused {
        state.focused = true;
        state.buffer.clear();
        let _ = write!(state.buffer, "{}", max_photo.0);
    }

    if click_outside && state.focused {
        commit_max_phototrophs(&mut state, &mut max_photo);
    }

    if !state.focused {
        for _ in keyboard.read() {}
        return;
    }

    for ev in keyboard.read() {
        if !ev.state.is_pressed() { continue; }
        match ev.key_code {
            KeyCode::Enter | KeyCode::NumpadEnter => {
                commit_max_phototrophs(&mut state, &mut max_photo);
            }
            KeyCode::Escape => {
                state.focused = false;
                state.buffer.clear();
            }
            KeyCode::Backspace => {
                state.buffer.pop();
            }
            _ => {
                if let Some(text) = ev.text.as_ref() {
                    for c in text.chars() {
                        if state.buffer.len() >= MAX_PHOTO_BUFFER_MAX_LEN { break; }
                        if c.is_ascii_digit() {
                            state.buffer.push(c);
                        }
                    }
                }
            }
        }
    }
}

/// Parse the buffer as `usize` and write into the resource (no clamp; see
/// `handle_max_phototrophs_input`). Always unfocus + clear the buffer.
fn commit_max_phototrophs(state: &mut MaxPhotoautotrophsEditState, max_photo: &mut MaxPhotoautotrophs) {
    if let Ok(v) = state.buffer.parse::<usize>() {
        if max_photo.0 != v {
            max_photo.0 = v;
        }
    }
    state.focused = false;
    state.buffer.clear();
}

/// Sync the input box's text + background colour with the current
/// committed `MaxPhotoautotrophs` value and edit state.
pub fn update_max_phototrophs_text(
    state:      Res<MaxPhotoautotrophsEditState>,
    max_photo:    Res<MaxPhotoautotrophs>,
    mut text_q: Query<&mut Text, With<MaxPhotoautotrophsText>>,
    mut bg_q:   Query<&mut BackgroundColor, With<MaxPhotoautotrophsInput>>,
) {
    if !state.is_changed() && !max_photo.is_changed() { return; }

    let display = if state.focused {
        format!("{}_", state.buffer)
    } else {
        format!("{}", max_photo.0)
    };
    for mut text in &mut text_q {
        text.0 = display.clone();
    }

    let bg = if state.focused { MAX_PHOTO_BG_FOCUSED } else { MAX_PHOTO_BG_IDLE };
    for mut b in &mut bg_q {
        if b.0 != bg { *b = BackgroundColor(bg); }
    }
}


// ── Max-Phototrophs enforcement (random cull of photoautotrophs only) ──────

/// Randomly despawn `(count - max)` photoautotrophs whenever the live count
/// exceeds `MaxPhotoautotrophs`. Heterotrophs are untouched (own `MaxHerbivores`
/// cap). Selection is a partial Fisher–Yates over the entity vector (O(k) swaps).
/// Populates the orange notification for `update_cull_message`.
pub fn apply_max_phototrophs_cull(
    mut commands:   Commands,
    max_photo:      Res<MaxPhotoautotrophs>,
    photos:         Query<(Entity, Has<crate::colony::BallPlankton>),
                          (With<OrganismRoot>, With<Photoautotroph>)>,
    mut message:    ResMut<CullMessage>,
) {
    // Enforce on the live count, NOT only on resource change: a disk-loaded
    // colony can spawn far above the cap, and an `is_changed`-only guard would
    // leave that excess (never trimmed by reproduction) as a silent FPS sink.
    // Cheap under the cap: count + early return.
    let cap = max_photo.0;
    let total = photos.iter().count();
    if total <= cap { return; }

    // NEVER cull the auto-spawned `ball_plankton` FLOOR (the persistent
    // `BallPlankton` marker — NOT `ImportedSpeciesOrigin`, which speciation
    // strips after ~1 s, blinding this exclusion). `auto_spawn_plankton` keeps a
    // minimum of these as the prey field; if the random cull despawned them, the
    // count would drop below the floor, the floor would respawn the deficit, that
    // would push back over the cap, and the cull would fire again — the perpetual
    // plankton spawn/despawn/respawn churn. Only the non-floor phototrophs (other
    // species + unmarked reproduced plankton descendants) are cullable. With a
    // sane cap (≥ the floor) the total still trims exactly to the cap; if the cap
    // is set BELOW the floor the floor wins (and the loop still terminates —
    // nothing respawns over the cap).
    let mut cullable: Vec<Entity> = photos.iter()
        .filter(|(_, is_floor)| !is_floor)
        .map(|(e, _)| e)
        .collect();

    let to_remove = (total - cap).min(cullable.len());
    if to_remove == 0 { return; }
    let mut rng = rand::rng();

    // Partial Fisher–Yates: the first `to_remove` entries become a uniform
    // random subset of the cullable phototrophs.
    let n = cullable.len();
    for i in 0..to_remove {
        let j = rng.random_range(i..n);
        cullable.swap(i, j);
    }
    for &entity in &cullable[..to_remove] {
        commands.entity(entity).despawn();
    }

    message.active  = true;
    message.count   = to_remove;
    message.elapsed = 0.0;
}

/// Drives the orange notification: ticks visibility on real time (so it
/// clears even while paused), hides the node after `CULL_MSG_VISIBLE_SECS`,
/// and reformats the string when `CullMessage` is freshly populated.
pub fn update_cull_message(
    real_time:    Res<Time>,
    mut message:  ResMut<CullMessage>,
    mut node_q:   Query<&mut Node, With<CullMessageText>>,
    mut text_q:   Query<&mut Text, With<CullMessageText>>,
) {
    // Tick first so a brand-new message (elapsed = 0) is shown this frame.
    let was_active = message.active;
    if message.active {
        message.elapsed += real_time.delta_secs();
        if message.elapsed >= CULL_MSG_VISIBLE_SECS {
            message.active = false;
        }
    }

    // Update the Text content on activation transitions / first frame.
    if message.is_changed() {
        for mut text in &mut text_q {
            text.0.clear();
            let _ = write!(
                text.0,
                "{} organisms have been removed randomly to meet the newly specified maximum organism count.",
                message.count,
            );
        }
    }

    // Show / hide the node; only touch Display on actual change to avoid
    // bumping downstream change-detection.
    let want = if message.active { Display::Flex } else { Display::None };
    let need_node_update = message.is_changed() || (was_active && !message.active);
    if need_node_update {
        for mut node in &mut node_q {
            if node.display != want { node.display = want; }
        }
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


// ── AI-training mode checkbox handlers ──────────────────────────────────────

/// Toggle the `AiTrainingMode` resource on click of the checkbox.
pub fn handle_ai_training_checkbox(
    interactions: Query<&Interaction, (Changed<Interaction>, With<AiTrainingCheckbox>)>,
    mut mode:     ResMut<crate::simulation_settings::AiTrainingMode>,
) {
    for interaction in &interactions {
        if matches!(interaction, Interaction::Pressed) {
            mode.0 = !mode.0;
        }
    }
}

/// Sync the inner check-mark's visibility to the resource value.
/// Mirrors the navigator's `update_checkbox_mark` pattern.
pub fn update_ai_training_checkbox_mark(
    mode:       Res<crate::simulation_settings::AiTrainingMode>,
    mut mark_q: Query<&mut Node, With<AiTrainingCheckboxMark>>,
) {
    if !mode.is_changed() { return; }
    for mut node in &mut mark_q {
        node.display = if mode.0 { Display::Flex } else { Display::None };
    }
}


// ── Max-Herbivores integer-input field ─────────────────────────────────────
//
// Same widget as Max-Organisms, but commits into `MaxHerbivores` (gates
// reproduction, no GPU batch dim) and is never upper-clamped.

pub fn handle_max_herbivores_input(
    mouse:         Res<ButtonInput<MouseButton>>,
    mut keyboard:  MessageReader<KeyboardInput>,
    interaction_q: Query<&Interaction, With<MaxHerbivoresInput>>,
    mut state:     ResMut<MaxHerbivoresEditState>,
    mut max_herb:  ResMut<crate::simulation_settings::MaxHerbivores>,
) {
    let click_on_input = mouse.just_pressed(MouseButton::Left)
        && interaction_q.iter().any(|i| matches!(i, Interaction::Pressed));
    let click_outside  = mouse.just_pressed(MouseButton::Left) && !click_on_input;

    if click_on_input && !state.focused {
        state.focused = true;
        state.buffer.clear();
        let _ = write!(state.buffer, "{}", max_herb.0);
    }

    if click_outside && state.focused {
        commit_max_herbivores(&mut state, &mut max_herb);
    }

    if !state.focused {
        for _ in keyboard.read() {}
        return;
    }

    for ev in keyboard.read() {
        if !ev.state.is_pressed() { continue; }
        match ev.key_code {
            KeyCode::Enter | KeyCode::NumpadEnter => {
                commit_max_herbivores(&mut state, &mut max_herb);
            }
            KeyCode::Escape => {
                state.focused = false;
                state.buffer.clear();
            }
            KeyCode::Backspace => {
                state.buffer.pop();
            }
            _ => {
                if let Some(text) = ev.text.as_ref() {
                    for c in text.chars() {
                        if state.buffer.len() >= MAX_PHOTO_BUFFER_MAX_LEN { break; }
                        if c.is_ascii_digit() {
                            state.buffer.push(c);
                        }
                    }
                }
            }
        }
    }
}

fn commit_max_herbivores(
    state:    &mut MaxHerbivoresEditState,
    max_herb: &mut crate::simulation_settings::MaxHerbivores,
) {
    if let Ok(v) = state.buffer.parse::<usize>() {
        if max_herb.0 != v {
            max_herb.0 = v;
        }
    }
    state.focused = false;
    state.buffer.clear();
}

pub fn update_max_herbivores_text(
    state:      Res<MaxHerbivoresEditState>,
    max_herb:   Res<crate::simulation_settings::MaxHerbivores>,
    mut text_q: Query<&mut Text, With<MaxHerbivoresText>>,
    mut bg_q:   Query<&mut BackgroundColor, With<MaxHerbivoresInput>>,
) {
    if !state.is_changed() && !max_herb.is_changed() { return; }

    let display = if state.focused {
        format!("{}_", state.buffer)
    } else {
        format!("{}", max_herb.0)
    };
    for mut text in &mut text_q {
        text.0 = display.clone();
    }

    let bg = if state.focused { MAX_PHOTO_BG_FOCUSED } else { MAX_PHOTO_BG_IDLE };
    for mut b in &mut bg_q {
        if b.0 != bg { *b = BackgroundColor(bg); }
    }
}


// ── Export Simulation Dataset button ────────────────────────────────────────

/// Idle / hover backgrounds for the Export-Dataset button.
const EXPORT_BUTTON_BG_IDLE:  Color = Color::srgb(0.22, 0.46, 0.46);
const EXPORT_BUTTON_BG_HOVER: Color = Color::srgb(0.30, 0.30, 0.30);

/// Click handler for the Export-Dataset button. Like `handle_save_button`:
/// pauses the sim, opens a blocking rfd dialog, and writes the path into
/// `ExportDatasetRequested` for `dataset_export::export_dataset_system` to
/// write the CSV next tick. Cancel = no write, sim stays paused.
pub fn handle_export_dataset_button(
    mut interactions: Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<ExportDatasetButton>)>,
    mut sim_running:  ResMut<SimulationRunning>,
    mut virtual_time: ResMut<Time<Virtual>>,
    mut request:      ResMut<crate::dataset_export::ExportDatasetRequested>,
) {
    for (interaction, mut bg) in &mut interactions {
        match *interaction {
            Interaction::Pressed => {
                if sim_running.0 {
                    sim_running.0 = false;
                    virtual_time.pause();
                }

                let initial_dir = std::env::current_dir()
                    .unwrap_or_else(|_| std::path::PathBuf::from("."));
                let default_name = format!(
                    "simulation_dataset_{}.csv",
                    chrono::Local::now().format("%d-%m-%Y-%H-%M-%S"),
                );
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("CSV (.csv)", &["csv"])
                    .set_directory(initial_dir)
                    .set_file_name(default_name)
                    .save_file()
                {
                    request.0 = Some(path);
                }
                *bg = BackgroundColor(EXPORT_BUTTON_BG_HOVER);
            }
            Interaction::Hovered => *bg = BackgroundColor(EXPORT_BUTTON_BG_HOVER),
            Interaction::None    => *bg = BackgroundColor(EXPORT_BUTTON_BG_IDLE),
        }
    }
}
