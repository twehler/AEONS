// Individuum navigator + per-organism identity labels.
//
// Two coupled features live here so they share state (the
// `IndividualIdentifier` component) and a single set of book-keeping systems:
//
//   1. Each heterotroph receives a procedurally-generated name on spawn
//      ("Heterotroph 1", "Heterotroph 2", …). The name is stored in the
//      `IndividualIdentifier` component on the OrganismRoot entity.
//
//   2. A small floating UI label is rendered above each named organism in
//      the 3D viewport. The label always faces the camera (it's a screen-
//      space UI Node positioned per frame from `Camera::world_to_viewport`)
//      and displays the organism's identifier.
//
//   3. A right-side "Individuum Navigator" panel lists every named
//      organism as a button. The panel mirrors the statistics panel's
//      visual language (same background colour, same drag-handle
//      affordance for resize) and lives inside the same top row as the
//      viewport, so its height tracks whatever the statistics-panel
//      divider is currently set to.
//
//   4. A "Show Individual Identifiers" checkbox at the top of the
//      panel toggles the in-viewport labels (the panel itself, the
//      counter, and the button list are unaffected). Identifier
//      assignment is independent of the toggle — turning labels off
//      only hides the visualisation, never the underlying data.
//
// Only heterotrophs are tracked for now. Photoautotrophs and Krishi do
// not receive identifiers.

use bevy::prelude::*;
use bevy::input::keyboard::KeyboardInput;
use bevy::input::mouse::MouseWheel;
use bevy::ui::{ComputedNode, UiGlobalTransform};
use bevy::window::PrimaryWindow;
use std::fmt::Write as _;

use crate::colony::{Heterotroph, IntelligenceLevel, Organism, OrganismRoot};
use crate::frontend::{PANEL_BG_COLOR, ViewportImage};
use crate::simulation_settings::{AutoSpawnHeteros, MinHeteroCount, MinHeteroCountEditState};


// ── Tunables ─────────────────────────────────────────────────────────────────

/// Initial navigator-panel width (logical px). Picked to fit a single
/// "Heterotroph N" button comfortably.
pub const NAV_INITIAL_WIDTH_PX: f32 = 220.0;
/// Lower bound on the navigator panel's width while dragging.
pub const NAV_MIN_WIDTH_PX:     f32 = 120.0;
/// Lower bound on the viewport's width while dragging the vertical
/// divider — symmetrical with `frontend::VIEWPORT_MIN_PX` for the
/// horizontal divider.
pub const NAV_VIEWPORT_MIN_PX:  f32 = 50.0;
/// Thickness of the vertical drag-handle separating viewport from
/// navigator panel (logical px).
pub const NAV_DIVIDER_WIDTH_PX: f32 = 6.0;

const PANEL_PADDING_PX:    f32   = 8.0;
/// Height of one navigator-list row. Sized to fit two stacked text
/// lines (name on top, intelligence level below) plus a little breathing
/// room.
const NAV_BUTTON_HEIGHT_PX:f32   = 44.0;
const NAV_BUTTON_GAP_PX:   f32   = 4.0;
/// Font size of the secondary "Intelligence: Level N" line. Smaller
/// than the name so the visual hierarchy reads name-first at a glance.
const NAV_BUTTON_SUBTEXT_FONT: f32 = 11.0;
/// Subtext colour — dimmer than the name to underscore the hierarchy.
const NAV_BUTTON_SUBTEXT_COLOR: Color = Color::srgb(0.70, 0.70, 0.70);
const CHECKBOX_SIZE_PX:    f32   = 16.0;
const LABEL_FONT_SIZE:     f32   = 13.0;
/// Vertical world-space offset between the organism's transform and
/// the projected anchor point used for the floating label. Small
/// because cells are roughly 1.0 unit and the label should hover just
/// above the body.
const LABEL_WORLD_LIFT:    f32   = 2.0;
/// Pixel padding between the projected anchor point and the bottom
/// edge of the label (the label sits ABOVE the anchor by this amount).
const LABEL_ANCHOR_GAP_PX: f32   = 4.0;
/// Logical pixels scrolled per mouse-wheel notch.
const SCROLL_STEP_PX:      f32   = 32.0;
/// Floating-label background. Translucent so the world stays readable
/// behind the text.
const LABEL_BG: Color = Color::srgba(0.0, 0.0, 0.0, 0.55);


// ── Resources ────────────────────────────────────────────────────────────────

/// Monotonically increasing counter used to mint identifiers. Never
/// decremented — when an organism dies, its number is retired (so labels
/// never reuse a name and the user can refer to a heterotroph by number
/// without ambiguity).
#[derive(Resource, Default)]
pub struct HeterotrophCounter(pub u32);

/// Master toggle for the in-world floating labels. Defaults to `true`
/// so labels are visible from the start. Toggled via the checkbox at
/// the top of the navigator panel.
///
/// Identifier assignment and the navigator-panel button list ignore
/// this flag entirely — turning it off ONLY hides the in-viewport
/// labels.
#[derive(Resource)]
pub struct ShowIndividualIdentifiers(pub bool);

impl Default for ShowIndividualIdentifiers {
    fn default() -> Self { Self(true) }
}

/// Captured at DragStart so subsequent Drag events resolve to an
/// absolute width — same pattern as `frontend::DividerDragState`.
#[derive(Resource, Default)]
struct VerticalDividerDragState {
    initial_panel_width: f32,
}


// ── Components ───────────────────────────────────────────────────────────────

/// Display name attached to an organism's root entity. Currently issued
/// only to heterotrophs (see `assign_individual_identifiers`).
#[derive(Component, Clone)]
pub struct IndividualIdentifier(pub String);

/// Marker on the right-hand navigator panel — `vertical_divider_drag`
/// queries on this to resize the panel as the divider is dragged.
#[derive(Component)]
pub struct NavigatorPanel;

/// Marker on the vertical drag-handle between viewport and navigator
/// panel.
#[derive(Component)]
struct VerticalDivider;

/// Marker on the scrollable list container inside the navigator panel.
#[derive(Component)]
struct NavigatorList;

/// Marker on each per-organism button in the navigator list. `target`
/// references the organism root entity.
#[derive(Component)]
struct NavigatorButton { target: Entity }

#[derive(Component)]
struct NavigatorButtonText;

/// Marker on the "Show Individual Identifiers" checkbox.
#[derive(Component)]
struct IdentifiersCheckbox;

/// Marker on the inner filled square that visualises the checkbox's
/// "checked" state. Hidden via `Display::None` when unchecked.
#[derive(Component)]
struct IdentifiersCheckboxMark;

/// Marker on the "Auto-Spawn Heterotrophs" checkbox at the bottom of
/// the navigator panel.
#[derive(Component)]
struct AutoSpawnCheckbox;

/// Inner-fill marker for the auto-spawn checkbox (mirrors
/// `IdentifiersCheckboxMark`).
#[derive(Component)]
struct AutoSpawnCheckboxMark;

/// Marker on the row containing the "Min count" input field. Hidden via
/// `Display::None` when `AutoSpawnHeteros(false)`.
#[derive(Component)]
struct MinHeteroCountRow;

/// Click-target for the min-heterotroph-count integer-input field.
#[derive(Component)]
struct MinHeteroCountInput;

/// Marker on the Text child inside the min-count input box.
#[derive(Component)]
struct MinHeteroCountText;

/// Max digits accepted in the min-count buffer — bounded so the user
/// can't paste a runaway integer.
const MIN_HETERO_BUFFER_MAX_LEN: usize = 6;
const MIN_HETERO_BG_IDLE:    Color = Color::srgb(0.18, 0.18, 0.18);
const MIN_HETERO_BG_FOCUSED: Color = Color::srgb(0.10, 0.30, 0.10);

/// One per labelled organism. `target` references the OrganismRoot.
/// Labels are spawned as children of the `ViewportImage` so their
/// `Val::Px` positions are already in viewport-local pixel coordinates
/// — the exact space `Camera::world_to_viewport` returns.
#[derive(Component)]
struct IndividualLabel { target: Entity }

#[derive(Component)]
struct IndividualLabelText;


// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct IndividuumNavigatorPlugin;

impl Plugin for IndividuumNavigatorPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<HeterotrophCounter>()
            .init_resource::<ShowIndividualIdentifiers>()
            .init_resource::<VerticalDividerDragState>()
            .add_systems(Update, (
                assign_individual_identifiers,
                manage_label_lifecycle,
                manage_navigator_list,
                update_label_positions,
                update_checkbox_mark,
                handle_identifiers_checkbox,
                navigator_scroll,
                handle_auto_spawn_checkbox,
                update_auto_spawn_checkbox_mark,
                update_min_hetero_row_visibility,
                handle_min_hetero_input,
                update_min_hetero_text,
            ));
    }
}


// ── Layout helpers (called from frontend::setup_panes) ──────────────────────

/// Spawn the vertical drag-handle that separates the viewport from the
/// navigator panel. Called inside the top row's `with_children`.
pub fn spawn_vertical_divider(parent: &mut ChildSpawnerCommands) {
    parent.spawn((
        VerticalDivider,
        Node {
            width:       Val::Px(NAV_DIVIDER_WIDTH_PX),
            height:      Val::Percent(100.0),
            flex_shrink: 0.0,
            ..default()
        },
        BackgroundColor(Color::srgb(0.35, 0.35, 0.35)),
    ))
    .observe(vertical_divider_drag_start)
    .observe(vertical_divider_drag);
}

/// Spawn the navigator panel as a child of the parent (the layout's top
/// row). The panel's width starts at `NAV_INITIAL_WIDTH_PX` and is later
/// resized by the vertical-divider observers.
pub fn spawn_navigator_panel(parent: &mut ChildSpawnerCommands) {
    parent.spawn((
        NavigatorPanel,
        Node {
            width:          Val::Px(NAV_INITIAL_WIDTH_PX),
            height:         Val::Percent(100.0),
            flex_direction: FlexDirection::Column,
            flex_shrink:    0.0,
            padding:        UiRect::all(Val::Px(PANEL_PADDING_PX)),
            ..default()
        },
        BackgroundColor(PANEL_BG_COLOR),
    ))
    .with_children(|panel| {
        // ── Header row: checkbox + label ────────────────────────────
        panel.spawn(Node {
            flex_direction: FlexDirection::Row,
            align_items:    AlignItems::Center,
            margin:         UiRect::bottom(Val::Px(8.0)),
            ..default()
        })
        .with_children(|header| {
            header.spawn((
                IdentifiersCheckbox,
                Button,
                Node {
                    width:           Val::Px(CHECKBOX_SIZE_PX),
                    height:          Val::Px(CHECKBOX_SIZE_PX),
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
                    IdentifiersCheckboxMark,
                    Node {
                        width:  Val::Px(CHECKBOX_SIZE_PX - 6.0),
                        height: Val::Px(CHECKBOX_SIZE_PX - 6.0),
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.85, 0.85, 0.85)),
                    Pickable::IGNORE,
                ));
            });

            header.spawn((
                Text::new("Show Individual Identifiers"),
                TextFont { font_size: 12.0, ..default() },
                TextColor(Color::WHITE),
                Pickable::IGNORE,
            ));
        });

        // ── Title ──────────────────────────────────────────────────────
        panel.spawn((
            Text::new("Individuum Navigator"),
            TextFont { font_size: 15.0, ..default() },
            TextColor(Color::srgb(0.92, 0.92, 0.92)),
            Node {
                margin: UiRect::bottom(Val::Px(6.0)),
                ..default()
            },
            Pickable::IGNORE,
        ));

        // ── Scrollable list of per-organism buttons ────────────────────
        // `Overflow::scroll_y()` enables clipping + scroll-position
        // honouring; the actual mouse-wheel handling lives in
        // `navigator_scroll`.
        panel.spawn((
            NavigatorList,
            Node {
                flex_grow:      1.0,
                flex_basis:     Val::Px(0.0),
                min_height:     Val::Px(0.0),
                flex_direction: FlexDirection::Column,
                overflow:       Overflow::scroll_y(),
                ..default()
            },
            ScrollPosition::default(),
        ));

        // ── Footer: Auto-Spawn checkbox + (conditional) min-count input ─
        // Sits below the scrollable list and pinned to the bottom of
        // the panel by `flex_shrink: 0`.
        panel.spawn(Node {
            flex_direction: FlexDirection::Column,
            margin:         UiRect::top(Val::Px(8.0)),
            flex_shrink:    0.0,
            ..default()
        })
        .with_children(|footer| {
            // Checkbox row.
            footer.spawn(Node {
                flex_direction: FlexDirection::Row,
                align_items:    AlignItems::Center,
                ..default()
            })
            .with_children(|row| {
                row.spawn((
                    AutoSpawnCheckbox,
                    Button,
                    Node {
                        width:           Val::Px(CHECKBOX_SIZE_PX),
                        height:          Val::Px(CHECKBOX_SIZE_PX),
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
                        AutoSpawnCheckboxMark,
                        Node {
                            width:   Val::Px(CHECKBOX_SIZE_PX - 6.0),
                            height:  Val::Px(CHECKBOX_SIZE_PX - 6.0),
                            display: Display::None, // off by default
                            ..default()
                        },
                        BackgroundColor(Color::srgb(0.85, 0.85, 0.85)),
                        Pickable::IGNORE,
                    ));
                });

                row.spawn((
                    Text::new("Auto-Spawn Heterotrophs"),
                    TextFont { font_size: 12.0, ..default() },
                    TextColor(Color::WHITE),
                    Pickable::IGNORE,
                ));
            });

            // Min-count row — collapsed by default, revealed when the
            // checkbox is ticked.
            footer.spawn((
                MinHeteroCountRow,
                Node {
                    flex_direction: FlexDirection::Row,
                    align_items:    AlignItems::Center,
                    margin:         UiRect::top(Val::Px(6.0)),
                    display:        Display::None,
                    ..default()
                },
            ))
            .with_children(|row| {
                row.spawn((
                    Text::new("Min count:"),
                    TextFont { font_size: 12.0, ..default() },
                    TextColor(Color::WHITE),
                    Node { margin: UiRect::right(Val::Px(6.0)), ..default() },
                    Pickable::IGNORE,
                ));
                row.spawn((
                    MinHeteroCountInput,
                    Button,
                    Node {
                        flex_grow:       1.0,
                        height:          Val::Px(22.0),
                        padding:         UiRect::axes(Val::Px(6.0), Val::Px(2.0)),
                        align_items:     AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(MIN_HETERO_BG_IDLE),
                ))
                .with_children(|btn| {
                    btn.spawn((
                        MinHeteroCountText,
                        Text::new(""),
                        TextFont { font_size: 12.0, ..default() },
                        TextColor(Color::WHITE),
                        Pickable::IGNORE,
                    ));
                });
            });
        });
    });
}

// ── Systems ──────────────────────────────────────────────────────────────────

/// Stamp `IndividualIdentifier("Heterotroph N")` onto every newly-
/// observed heterotroph root. Runs every frame; the `Added` filter
/// keeps it cheap (zero work in steady state).
fn assign_individual_identifiers(
    mut commands: Commands,
    mut counter:  ResMut<HeterotrophCounter>,
    new_q:        Query<
        Entity,
        (Added<Heterotroph>, With<OrganismRoot>, Without<IndividualIdentifier>),
    >,
) {
    for entity in &new_q {
        counter.0 += 1;
        // `try_insert` matches the convention used elsewhere when
        // mutating arbitrary entities — a despawn race wouldn't panic.
        commands.entity(entity)
            .try_insert(IndividualIdentifier(format!("Heterotroph {}", counter.0)));
    }
}

/// Mirror identifier set ↔ label set:
///   * spawn a label entity (as a child of the `ViewportImage`) for any
///     newly-tagged organism,
///   * despawn labels whose target organism has been despawned.
///
/// Parenting under `ViewportImage` is what keeps the projection math
/// trivial: `world_to_viewport` returns coords relative to the
/// viewport's top-left, and the label's `Val::Px(left/top)` is also
/// relative to its parent's top-left — so the two systems share the
/// same origin and the per-frame update is just a scale-factor divide.
fn manage_label_lifecycle(
    mut commands:    Commands,
    viewport_q:      Query<Entity, With<ViewportImage>>,
    new_id_q:        Query<
        (Entity, &IndividualIdentifier),
        (With<Heterotroph>, Added<IndividualIdentifier>),
    >,
    label_q:         Query<(Entity, &IndividualLabel)>,
    extant_q:        Query<(), With<IndividualIdentifier>>,
) {
    let Ok(viewport) = viewport_q.single() else { return };

    for (entity, ident) in &new_id_q {
        let label_entity = commands.spawn((
            IndividualLabel { target: entity },
            Node {
                position_type: PositionType::Absolute,
                // Off-screen until the first frame's projection lands.
                top:           Val::Px(-9999.0),
                left:          Val::Px(-9999.0),
                padding:       UiRect::axes(Val::Px(5.0), Val::Px(2.0)),
                ..default()
            },
            BackgroundColor(LABEL_BG),
            Pickable::IGNORE,
            Visibility::Hidden,
        ))
        .with_children(|wrap| {
            wrap.spawn((
                IndividualLabelText,
                Text::new(ident.0.clone()),
                TextFont { font_size: LABEL_FONT_SIZE, ..default() },
                TextColor(Color::WHITE),
                Pickable::IGNORE,
            ));
        })
        .id();
        commands.entity(viewport).add_child(label_entity);
    }

    for (label_entity, label) in &label_q {
        if !extant_q.contains(label.target) {
            commands.entity(label_entity).despawn();
        }
    }
}

/// Project every label-target organism's world position into the
/// viewport pane and set the label's absolute UI position. Labels that
/// fall outside the viewport rect (off-screen, behind camera, or
/// occluded by a UI panel) are hidden via `Visibility::Hidden` so they
/// never bleed onto the navigator / statistics panels.
fn update_label_positions(
    show:        Res<ShowIndividualIdentifiers>,
    cameras:     Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    viewport_q:  Query<&ComputedNode, With<ViewportImage>>,
    organism_q:  Query<&GlobalTransform, With<OrganismRoot>>,
    mut labels:  Query<(&IndividualLabel, &ComputedNode, &mut Node, &mut Visibility)>,
) {
    if !show.0 {
        for (_, _, _, mut v) in &mut labels {
            *v = Visibility::Hidden;
        }
        return;
    }

    let Ok((camera, cam_xf)) = cameras.single() else { return };
    let Ok(viewport_node)    = viewport_q.single() else { return };

    // Labels are children of the `ViewportImage`, so their `Val::Px`
    // (logical) origin coincides with the viewport's top-left.
    // `Camera::world_to_viewport` returns coords in the same image-
    // local space (top-left origin, Y down) in physical pixels (image
    // targets default to scale_factor=1.0). Multiplying by the node's
    // inverse_scale_factor lands us in window-logical pixels, which
    // is what `Val::Px` consumes. No `UiGlobalTransform`, no manual
    // window-half offset.
    //
    // `world_to_viewport` returns Err for points behind the camera or
    // beyond the depth range — those cases hide the label for free.
    // Points outside the viewport's X/Y range still return Ok with
    // out-of-rect coords; we hide those with the explicit bounds
    // check against `viewport_node.size()`.
    let inv_scale         = viewport_node.inverse_scale_factor;
    let viewport_size     = viewport_node.size();

    for (label, label_node, mut node, mut vis) in &mut labels {
        let Ok(target_xf) = organism_q.get(label.target) else {
            *vis = Visibility::Hidden;
            continue;
        };

        let world_pos = target_xf.translation() + Vec3::new(0.0, LABEL_WORLD_LIFT, 0.0);
        let Ok(vp) = camera.world_to_viewport(cam_xf, world_pos) else {
            *vis = Visibility::Hidden;
            continue;
        };

        if vp.x < 0.0 || vp.x > viewport_size.x
            || vp.y < 0.0 || vp.y > viewport_size.y
        {
            *vis = Visibility::Hidden;
            continue;
        }

        // Image-local physical px → viewport-local logical px (Val::Px).
        let anchor_logical = vp * inv_scale;
        // Centre the label horizontally on the anchor and seat its
        // bottom edge a few pixels above the anchor point.
        let label_size_logical = label_node.size() * label_node.inverse_scale_factor;
        node.left = Val::Px(anchor_logical.x - label_size_logical.x * 0.5);
        node.top  = Val::Px(anchor_logical.y - label_size_logical.y - LABEL_ANCHOR_GAP_PX);
        *vis = Visibility::Inherited;
    }
}

/// Sync the navigator-list buttons with the current set of labelled
/// organisms. Same pattern as `manage_label_lifecycle`: spawn for every
/// new identifier, despawn when the target organism is gone.
///
/// Each button stacks two rows:
///   1. The procedurally-minted name ("Heterotroph N").
///   2. "Intelligence: Level N" — read once at spawn time. The
///      `intelligence_level` field is set when the organism is
///      created and inherited verbatim by offspring, so it never
///      mutates and a one-shot read on `Added<IndividualIdentifier>`
///      is sufficient.
fn manage_navigator_list(
    mut commands:    Commands,
    list_q:          Query<Entity, With<NavigatorList>>,
    new_q:           Query<
        (Entity, &IndividualIdentifier, &Organism),
        (With<Heterotroph>, Added<IndividualIdentifier>),
    >,
    buttons_q:       Query<(Entity, &NavigatorButton)>,
    extant_q:        Query<(), With<IndividualIdentifier>>,
) {
    let Ok(list) = list_q.single() else { return };

    for (entity, ident, organism) in &new_q {
        let intel_label = match organism.intelligence_level {
            IntelligenceLevel::Level0 => "Intelligence: Level 0",
            IntelligenceLevel::Level1 => "Intelligence: Level 1",
            IntelligenceLevel::Level2 => "Intelligence: Level 2",
            IntelligenceLevel::Level3 => "Intelligence: Level 3",
        };

        let btn = commands.spawn((
            NavigatorButton { target: entity },
            Button,
            Node {
                width:           Val::Percent(100.0),
                height:          Val::Px(NAV_BUTTON_HEIGHT_PX),
                margin:          UiRect::bottom(Val::Px(NAV_BUTTON_GAP_PX)),
                padding:         UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                flex_direction:  FlexDirection::Column,
                align_items:     AlignItems::FlexStart,
                justify_content: JustifyContent::Center,
                flex_shrink:     0.0,
                ..default()
            },
            BackgroundColor(Color::srgb(0.25, 0.25, 0.25)),
        ))
        .with_children(|btn| {
            btn.spawn((
                NavigatorButtonText,
                Text::new(ident.0.clone()),
                TextFont { font_size: 13.0, ..default() },
                TextColor(Color::WHITE),
                Pickable::IGNORE,
            ));
            btn.spawn((
                Text::new(intel_label),
                TextFont { font_size: NAV_BUTTON_SUBTEXT_FONT, ..default() },
                TextColor(NAV_BUTTON_SUBTEXT_COLOR),
                Pickable::IGNORE,
            ));
        })
        .id();
        commands.entity(list).add_child(btn);
    }

    for (btn_entity, btn) in &buttons_q {
        if !extant_q.contains(btn.target) {
            commands.entity(btn_entity).despawn();
        }
    }
}

/// Show / hide the inner check mark to reflect `ShowIndividualIdentifiers`.
fn update_checkbox_mark(
    show:       Res<ShowIndividualIdentifiers>,
    mut mark_q: Query<&mut Node, With<IdentifiersCheckboxMark>>,
) {
    if !show.is_changed() { return; }
    for mut node in &mut mark_q {
        node.display = if show.0 { Display::Flex } else { Display::None };
    }
}

/// Toggle the master flag on a click of the checkbox.
fn handle_identifiers_checkbox(
    interactions: Query<&Interaction, (Changed<Interaction>, With<IdentifiersCheckbox>)>,
    mut show:     ResMut<ShowIndividualIdentifiers>,
) {
    for interaction in &interactions {
        if matches!(interaction, Interaction::Pressed) {
            show.0 = !show.0;
        }
    }
}

/// Mouse-wheel scroll handler for the navigator list. Walks all
/// MouseWheel events for the frame and applies them when the cursor is
/// inside the navigator panel's screen-space rect. The bevy_picking
/// `Pointer<Scroll>` event isn't a thing in Bevy 0.18, so we handle
/// the wheel-vs-cursor-rect check ourselves.
fn navigator_scroll(
    mut wheel_evs: MessageReader<MouseWheel>,
    windows:       Query<&Window, With<PrimaryWindow>>,
    panel_q:       Query<(&ComputedNode, &UiGlobalTransform), With<NavigatorPanel>>,
    mut list_q:    Query<&mut ScrollPosition, With<NavigatorList>>,
) {
    let mut delta_y = 0.0;
    for ev in wheel_evs.read() {
        delta_y += ev.y;
    }
    if delta_y == 0.0 { return; }

    let Ok(window) = windows.single() else { return };
    let Some(cursor_logical) = window.cursor_position() else { return };
    let Ok((panel_node, panel_ui_xf)) = panel_q.single() else { return };

    // `Window::cursor_position` is window-LOGICAL pixels with origin
    // at the top-left. `UiGlobalTransform.translation` is the node
    // centre in PHYSICAL pixels with the same top-left origin (the
    // UI projection matrix in 0.18 is `orthographic_rh(0,w,h,0)`).
    // So one multiply by `inverse_scale_factor` lands the panel rect
    // in the same units as the cursor.
    let inv_scale = panel_node.inverse_scale_factor;
    let size      = panel_node.size() * inv_scale;
    let centre    = panel_ui_xf.translation * inv_scale;
    let min       = centre - size * 0.5;
    let max       = min + size;

    if cursor_logical.x < min.x || cursor_logical.x > max.x
        || cursor_logical.y < min.y || cursor_logical.y > max.y
    {
        return;
    }

    for mut sp in &mut list_q {
        sp.y = (sp.y - delta_y * SCROLL_STEP_PX).max(0.0);
    }
}


// ── Vertical-divider drag observers ─────────────────────────────────────────

fn vertical_divider_drag_start(
    _ev:        On<Pointer<DragStart>>,
    panel_q:    Query<&ComputedNode, With<NavigatorPanel>>,
    mut state:  ResMut<VerticalDividerDragState>,
) {
    if let Ok(panel) = panel_q.single() {
        state.initial_panel_width = panel.size().x * panel.inverse_scale_factor;
    }
}

fn vertical_divider_drag(
    ev:           On<Pointer<Drag>>,
    state:        Res<VerticalDividerDragState>,
    windows:      Query<&Window, With<PrimaryWindow>>,
    mut panel_q:  Query<&mut Node, With<NavigatorPanel>>,
) {
    let Ok(window) = windows.single() else { return };
    let Ok(mut node) = panel_q.single_mut() else { return };
    let max_w = (window.width() - NAV_DIVIDER_WIDTH_PX - NAV_VIEWPORT_MIN_PX).max(NAV_MIN_WIDTH_PX);
    // Drag right ⇒ shrink panel; drag left ⇒ expand. ev.distance is
    // cumulative from DragStart, so anchoring on `state.initial_panel_width`
    // produces a stable absolute width regardless of drag speed.
    let new_w = (state.initial_panel_width - ev.distance.x).clamp(NAV_MIN_WIDTH_PX, max_w);
    node.width = Val::Px(new_w);
}


// ── Auto-spawn footer handlers ──────────────────────────────────────────────

/// Toggle `AutoSpawnHeteros` on click.
fn handle_auto_spawn_checkbox(
    interactions: Query<&Interaction, (Changed<Interaction>, With<AutoSpawnCheckbox>)>,
    mut auto:     ResMut<AutoSpawnHeteros>,
) {
    for interaction in &interactions {
        if matches!(interaction, Interaction::Pressed) {
            auto.0 = !auto.0;
        }
    }
}

/// Mirror `AutoSpawnHeteros` onto the inner-fill display state.
fn update_auto_spawn_checkbox_mark(
    auto:       Res<AutoSpawnHeteros>,
    mut mark_q: Query<&mut Node, With<AutoSpawnCheckboxMark>>,
) {
    if !auto.is_changed() { return; }
    for mut node in &mut mark_q {
        node.display = if auto.0 { Display::Flex } else { Display::None };
    }
}

/// Reveal / hide the min-count input row based on the master flag.
/// Edge-triggered on `auto` change so the layout cost is paid only at
/// toggle events.
fn update_min_hetero_row_visibility(
    auto:      Res<AutoSpawnHeteros>,
    mut row_q: Query<&mut Node, With<MinHeteroCountRow>>,
) {
    if !auto.is_changed() { return; }
    for mut node in &mut row_q {
        node.display = if auto.0 { Display::Flex } else { Display::None };
    }
}

/// Click + keyboard router for the min-count integer-input field.
/// Mirrors `handle_max_organisms_input` in the statistics panel:
/// click-on-input enters edit mode, Enter commits, Escape cancels,
/// click-outside commits. Digits-only.
fn handle_min_hetero_input(
    mouse:         Res<ButtonInput<MouseButton>>,
    mut keyboard:  MessageReader<KeyboardInput>,
    auto:          Res<AutoSpawnHeteros>,
    interaction_q: Query<&Interaction, With<MinHeteroCountInput>>,
    mut state:     ResMut<MinHeteroCountEditState>,
    mut min_count: ResMut<MinHeteroCount>,
) {
    // When the row is hidden the user can't see / click the field, but
    // we still drain the keyboard reader so events don't accumulate.
    if !auto.0 {
        for _ in keyboard.read() {}
        if state.focused {
            state.focused = false;
            state.buffer.clear();
        }
        return;
    }

    let click_on_input = mouse.just_pressed(MouseButton::Left)
        && interaction_q.iter().any(|i| matches!(i, Interaction::Pressed));
    let click_outside  = mouse.just_pressed(MouseButton::Left) && !click_on_input;

    if click_on_input && !state.focused {
        state.focused = true;
        state.buffer.clear();
        let _ = write!(state.buffer, "{}", min_count.0);
    }

    if click_outside && state.focused {
        commit_min_hetero(&mut state, &mut min_count);
    }

    if !state.focused {
        for _ in keyboard.read() {}
        return;
    }

    for ev in keyboard.read() {
        if !ev.state.is_pressed() { continue; }
        match ev.key_code {
            KeyCode::Enter | KeyCode::NumpadEnter => {
                commit_min_hetero(&mut state, &mut min_count);
            }
            KeyCode::Escape => {
                state.focused = false;
                state.buffer.clear();
            }
            KeyCode::Backspace => { state.buffer.pop(); }
            _ => {
                if let Some(text) = ev.text.as_ref() {
                    for c in text.chars() {
                        if state.buffer.len() >= MIN_HETERO_BUFFER_MAX_LEN { break; }
                        if c.is_ascii_digit() { state.buffer.push(c); }
                    }
                }
            }
        }
    }
}

fn commit_min_hetero(state: &mut MinHeteroCountEditState, min_count: &mut MinHeteroCount) {
    if let Ok(v) = state.buffer.parse::<usize>() {
        if min_count.0 != v { min_count.0 = v; }
    }
    state.focused = false;
    state.buffer.clear();
}

fn update_min_hetero_text(
    state:      Res<MinHeteroCountEditState>,
    min_count:  Res<MinHeteroCount>,
    mut text_q: Query<&mut Text, With<MinHeteroCountText>>,
    mut bg_q:   Query<&mut BackgroundColor, With<MinHeteroCountInput>>,
) {
    if !state.is_changed() && !min_count.is_changed() { return; }
    let display = if state.focused {
        format!("{}_", state.buffer)
    } else {
        format!("{}", min_count.0)
    };
    for mut text in &mut text_q { text.0 = display.clone(); }

    let bg = if state.focused { MIN_HETERO_BG_FOCUSED } else { MIN_HETERO_BG_IDLE };
    for mut b in &mut bg_q {
        if b.0 != bg { *b = BackgroundColor(bg); }
    }
}
