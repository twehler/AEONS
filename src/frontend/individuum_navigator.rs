// Individuum navigator + per-organism identity labels.
//
// Coupled features sharing the `IndividualIdentifier` component: per-heterotroph
// name on spawn; a per-frame `Camera::world_to_viewport`-positioned floating UI
// label above each named organism; a right-side navigator panel of buttons; and
// a "Show Individual Identifiers" checkbox toggling only the labels (identifier
// assignment is independent — off hides the visualisation, not the data).
// Only heterotrophs are tracked (no photoautotrophs).

use bevy::prelude::*;
use bevy::input::keyboard::KeyboardInput;
use bevy::input::mouse::MouseWheel;
use bevy::ui::{ComputedNode, UiGlobalTransform};
use bevy::window::PrimaryWindow;
use std::collections::HashMap;
use std::fmt::Write as _;

use crate::colony::{Heterotroph, Organism, OrganismRoot};
use crate::frontend::{PANEL_BG_COLOR, ViewportImage};
use crate::simulation_settings::{AutoSpawnHeteros, MinHeteroCount, MinHeteroCountEditState};


// ── Tunables ─────────────────────────────────────────────────────────────────

/// Initial navigator-panel width (logical px).
pub const NAV_INITIAL_WIDTH_PX: f32 = 220.0;
/// Lower bound on the navigator panel's width while dragging.
pub const NAV_MIN_WIDTH_PX:     f32 = 120.0;
/// Lower bound on viewport width while dragging the vertical divider —
/// symmetrical with `frontend::VIEWPORT_MIN_PX`.
pub const NAV_VIEWPORT_MIN_PX:  f32 = 50.0;
/// Vertical drag-handle thickness (logical px).
pub const NAV_DIVIDER_WIDTH_PX: f32 = 6.0;

const PANEL_PADDING_PX:    f32   = 8.0;
/// Navigator-list row height; fits two stacked text lines.
const NAV_BUTTON_HEIGHT_PX:f32   = 44.0;
const NAV_BUTTON_GAP_PX:   f32   = 4.0;
const NAV_BUTTON_SUBTEXT_FONT: f32 = 11.0;
const NAV_BUTTON_SUBTEXT_COLOR: Color = Color::srgb(0.70, 0.70, 0.70);
const CHECKBOX_SIZE_PX:    f32   = 16.0;
const LABEL_FONT_SIZE:     f32   = 13.0;
/// World-space lift of the floating-label anchor above the organism
/// transform (cells are ~1.0 unit).
const LABEL_WORLD_LIFT:    f32   = 2.0;
/// Pixel gap above the anchor; the label sits ABOVE the anchor by this much.
const LABEL_ANCHOR_GAP_PX: f32   = 4.0;
/// Logical pixels scrolled per mouse-wheel notch.
const SCROLL_STEP_PX:      f32   = 32.0;
/// Floating-label background; translucent so the world stays readable.
const LABEL_BG: Color = Color::srgba(0.0, 0.0, 0.0, 0.55);


// ── Resources ────────────────────────────────────────────────────────────────

/// Per-species running index allocator (1-based, monotonic, never reused), so
/// `species#N` is a stable handle. Name TRACKS live species: reclassification
/// re-issues a fresh index in the new species.
#[derive(Resource, Default)]
pub struct SpeciesIndexCounters {
    next: std::collections::HashMap<u32, u32>,
}

/// Species an organism is currently NAMED under + the index issued within it.
/// Re-issued when `Organism::species_id` changes. Absence = not yet named.
#[derive(Component)]
pub struct SpeciesIndex {
    pub species_id: u32,
    pub index:      u32,
}

/// Heterotroph selected by viewport left-click; its navigator button is
/// highlighted and scrolled into view. `None` clears on empty/terrain click.
#[derive(Resource, Default)]
pub struct SelectedOrganism(pub Option<Entity>);

/// Emitted by `frontend::viewport_click` on a Simulation-mode viewport
/// left-click (window-space cursor). Consumed by `pick_organism` (ray-cast).
#[derive(Message, Clone, Copy)]
pub struct ViewportPick {
    pub cursor: Vec2,
}

/// Master toggle for the in-world floating labels (default `true`).
/// Identifier assignment and the navigator button list ignore this flag —
/// turning it off ONLY hides the in-viewport labels.
#[derive(Resource)]
pub struct ShowIndividualIdentifiers(pub bool);

impl Default for ShowIndividualIdentifiers {
    fn default() -> Self { Self(true) }
}

/// Captured at DragStart so subsequent Drag events resolve to an absolute
/// width (same pattern as `frontend::DividerDragState`).
#[derive(Resource, Default)]
struct VerticalDividerDragState {
    initial_panel_width: f32,
}


// ── Components ───────────────────────────────────────────────────────────────

/// Display name on an organism's root entity. Heterotrophs only.
#[derive(Component, Clone)]
pub struct IndividualIdentifier(pub String);

/// Marker on the right-hand navigator panel.
#[derive(Component)]
pub struct NavigatorPanel;

/// Marker on the vertical drag-handle between viewport and navigator panel.
#[derive(Component)]
pub struct VerticalDivider;

/// Marker on the scrollable list container inside the navigator panel.
#[derive(Component)]
struct NavigatorList;

/// Per-organism button in the navigator list; `target` = organism root.
#[derive(Component)]
struct NavigatorButton { target: Entity }

/// Per-row "Export species" button. Click → file-save dialog →
/// `(entity, path)` into `ExportSpeciesRequested` for the worker.
#[derive(Component)]
struct NavigatorExportButton { target: Entity }

/// One-shot request: `Some((entity, path))` between dialog and worker run.
#[derive(Resource, Default)]
struct ExportSpeciesRequested(Option<(Entity, std::path::PathBuf)>);

#[derive(Component)]
struct NavigatorButtonText { target: Entity }

/// Marker on the "Show Individual Identifiers" checkbox.
#[derive(Component)]
struct IdentifiersCheckbox;

/// Inner filled square showing "checked"; hidden via `Display::None`.
#[derive(Component)]
struct IdentifiersCheckboxMark;

/// Marker on the "Auto-Spawn Heterotrophs" checkbox.
#[derive(Component)]
struct AutoSpawnCheckbox;

/// Inner-fill marker for the auto-spawn checkbox.
#[derive(Component)]
struct AutoSpawnCheckboxMark;

/// "Min count" input row; hidden via `Display::None` when `AutoSpawnHeteros(false)`.
#[derive(Component)]
struct MinHeteroCountRow;

/// Click-target for the min-heterotroph-count integer-input field.
#[derive(Component)]
struct MinHeteroCountInput;

/// Marker on the Text child inside the min-count input box.
#[derive(Component)]
struct MinHeteroCountText;

/// Max digits in the min-count buffer (bounds runaway paste).
const MIN_HETERO_BUFFER_MAX_LEN: usize = 6;
const MIN_HETERO_BG_IDLE:    Color = Color::srgb(0.18, 0.18, 0.18);
const MIN_HETERO_BG_FOCUSED: Color = Color::srgb(0.10, 0.30, 0.10);

/// The "Species:" rename row (hidden until an organism with a species is
/// selected). Toggled by `update_species_rename_row_visibility`.
#[derive(Component)]
struct SpeciesRenameRow;

/// Click-target for the species-name rename text field.
#[derive(Component)]
struct SpeciesRenameInput;

/// Marker on the Text child inside the rename input box.
#[derive(Component)]
struct SpeciesRenameText;

/// Max chars in the species-name buffer.
const SPECIES_RENAME_BUFFER_MAX_LEN: usize = 32;

/// Focus + edit buffer for the species-rename field (mirrors
/// `MinHeteroCountEditState`).
#[derive(Resource, Default)]
struct SpeciesRenameEditState {
    focused: bool,
    buffer:  String,
}

/// One per labelled organism (`target` = OrganismRoot). Spawned as a child of
/// `ViewportImage` so `Val::Px` positions are viewport-local pixels — the exact
/// space `Camera::world_to_viewport` returns.
#[derive(Component)]
struct IndividualLabel { target: Entity }

#[derive(Component)]
struct IndividualLabelText { target: Entity }

/// Species-name sub-text inside the floating label; kept in sync with the
/// target organism's `species_id` by `update_label_species_text`.
#[derive(Component)]
struct IndividualLabelSpecies { target: Entity }


// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct IndividuumNavigatorPlugin;

impl Plugin for IndividuumNavigatorPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<SpeciesIndexCounters>()
            .init_resource::<SelectedOrganism>()
            .init_resource::<ShowIndividualIdentifiers>()
            .init_resource::<VerticalDividerDragState>()
            .init_resource::<ExportSpeciesRequested>()
            .init_resource::<SpeciesRenameEditState>()
            .add_message::<ViewportPick>()
            .add_systems(Update, (
                assign_individual_identifiers,
                update_navigator_button_text,
                update_label_name_text,
                pick_organism,
                apply_selection_highlight,
                manage_label_lifecycle,
                manage_navigator_list,
                update_label_positions,
                update_label_species_text,
                update_checkbox_mark,
                handle_identifiers_checkbox,
                navigator_scroll,
                handle_auto_spawn_checkbox,
                update_auto_spawn_checkbox_mark,
                update_min_hetero_row_visibility,
                handle_min_hetero_input,
                update_min_hetero_text,
                handle_export_buttons,
                dispatch_export_species_requests,
                (
                    update_species_rename_row_visibility,
                    handle_species_rename_input,
                    update_species_rename_text,
                ),
            ));
    }
}


// ── Layout helpers (called from frontend::setup_panes) ──────────────────────

/// Spawn the vertical drag-handle between viewport and navigator panel.
/// Called inside the top row's `with_children`.
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

/// Spawn the navigator panel as a child of the layout's top row.
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

        // ── Selected-species rename field (hidden until an organism with a
        //    classified species is selected; click to edit, Enter to commit). ─
        panel.spawn((
            SpeciesRenameRow,
            Node {
                flex_direction: FlexDirection::Row,
                align_items:    AlignItems::Center,
                margin:         UiRect::bottom(Val::Px(6.0)),
                display:        Display::None,
                ..default()
            },
        ))
        .with_children(|row| {
            row.spawn((
                Text::new("Species:"),
                TextFont { font_size: 12.0, ..default() },
                TextColor(Color::srgb(0.75, 0.75, 0.75)),
                Node { margin: UiRect::right(Val::Px(6.0)), flex_shrink: 0.0, ..default() },
                Pickable::IGNORE,
            ));
            row.spawn((
                SpeciesRenameInput,
                Button,
                Node {
                    flex_grow: 1.0,
                    min_width: Val::Px(0.0),
                    padding:   UiRect::axes(Val::Px(6.0), Val::Px(3.0)),
                    ..default()
                },
                BackgroundColor(MIN_HETERO_BG_IDLE),
            ))
            .with_children(|btn| {
                btn.spawn((
                    SpeciesRenameText,
                    Text::new(""),
                    TextFont { font_size: 12.0, ..default() },
                    TextColor(Color::WHITE),
                    Pickable::IGNORE,
                ));
            });
        });

        // ── Scrollable list of per-organism buttons ────────────────────
        // `Overflow::scroll_y()` enables clipping; mouse-wheel handling
        // lives in `navigator_scroll`.
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

            // Min-count row — collapsed until the checkbox is ticked.
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

/// Assign each heterotroph an `IndividualIdentifier` `<species name>#<index>`
/// (live `Species::name` + per-species running index). (Re)named on first
/// classification and on reclassification into a DIFFERENT species (fresh index
/// in the new one). No `species_id` ⇒ no identifier (and no row/label), a ~1 s
/// window after spawn. Per-organism work is an O(1) "species changed?" check.
fn assign_individual_identifiers(
    mut commands:  Commands,
    mut counters:  ResMut<SpeciesIndexCounters>,
    registry:      Res<crate::lineages::species::SpeciesRegistry>,
    mut q:         Query<
        (Entity, &Organism, Option<&mut SpeciesIndex>, Option<&mut IndividualIdentifier>),
        (With<Heterotroph>, With<OrganismRoot>),
    >,
) {
    for (entity, organism, maybe_idx, maybe_ident) in &mut q {
        let Some(sid) = organism.species_id else { continue };
        let Some(species) = registry.get(sid) else { continue };

        // Only (re)issue when never named, or when the live species changed.
        let needs_reissue = match &maybe_idx {
            Some(si) => si.species_id != sid,
            None     => true,
        };
        if !needs_reissue { continue; }

        let index = {
            let n = counters.next.entry(sid).or_insert(1);
            let v = *n;
            *n += 1;
            v
        };
        let name = format!("{}#{}", species.name, index);

        match maybe_idx {
            Some(mut si) => { si.species_id = sid; si.index = index; }
            None => {
                commands.entity(entity).try_insert(SpeciesIndex { species_id: sid, index });
            }
        }
        match maybe_ident {
            Some(mut id) => { if id.0 != name { id.0 = name; } }
            None => {
                commands.entity(entity).try_insert(IndividualIdentifier(name));
            }
        }
    }
}

/// Keep each navigator button's label in sync with its target's
/// `IndividualIdentifier` (the name changes on reclassification).
fn update_navigator_button_text(
    idents:    Query<&IndividualIdentifier>,
    mut texts: Query<(&NavigatorButtonText, &mut Text)>,
) {
    for (marker, mut text) in &mut texts {
        if let Ok(id) = idents.get(marker.target) {
            if text.0 != id.0 { text.0 = id.0.clone(); }
        }
    }
}

/// Same, for the in-world floating label's individual-name line.
fn update_label_name_text(
    idents:    Query<&IndividualIdentifier>,
    mut texts: Query<(&IndividualLabelText, &mut Text)>,
) {
    for (marker, mut text) in &mut texts {
        if let Ok(id) = idents.get(marker.target) {
            if text.0 != id.0 { text.0 = id.0.clone(); }
        }
    }
}

/// Resolve a viewport left-click to the heterotroph under the cursor and select
/// it. The camera renders off-screen, so the window cursor is remapped into
/// image-local pixels before `viewport_to_world` builds the ray; `SpatialQuery`
/// returns the closest collider. The hit resolves to its `OrganismRoot`
/// (kinematic root, or body part → parent root for limb organisms). Only a
/// heterotroph root selects; terrain / photoautotroph / miss clears.
fn pick_organism(
    mut picks:    MessageReader<ViewportPick>,
    cameras:      Query<(&Camera, &GlobalTransform), With<crate::player_plugin::FlyCam>>,
    viewport_q:   Query<(&ComputedNode, &UiGlobalTransform), With<ViewportImage>>,
    rapier:       bevy_rapier3d::prelude::ReadRapierContext,
    root_q:       Query<(), With<OrganismRoot>>,
    hetero_q:     Query<(), With<Heterotroph>>,
    childof_q:    Query<&ChildOf>,
    mut selected: ResMut<SelectedOrganism>,
) {
    // Only the most recent click in the frame matters.
    let Some(pick) = picks.read().last().copied() else { return };
    let Ok((camera, cam_xf)) = cameras.single() else { return };

    // Window cursor → viewport-image-local pixels (image-target camera).
    let cursor_vp = match viewport_q.single() {
        Ok((node, ui_xf)) => {
            let inv      = node.inverse_scale_factor;
            let size     = node.size() * inv;
            let top_left = ui_xf.translation * inv - size * 0.5;
            pick.cursor - top_left
        }
        Err(_) => pick.cursor,
    };

    let Ok(ray) = camera.viewport_to_world(cam_xf, cursor_vp) else { return };
    let Ok(ctx) = rapier.single() else { return };
    let hit = ctx.cast_ray(
        ray.origin,
        ray.direction.as_vec3(),
        1000.0,
        true,
        bevy_rapier3d::prelude::QueryFilter::default(),
    );

    let mut new_sel = None;
    if let Some((e, _toi)) = hit {
        let root = if root_q.contains(e) {
            Some(e)
        } else if let Ok(co) = childof_q.get(e) {
            let p = co.parent();
            root_q.contains(p).then_some(p)
        } else {
            None
        };
        if let Some(root) = root {
            if hetero_q.contains(root) {
                new_sel = Some(root);
            }
        }
    }
    if selected.0 != new_sel {
        selected.0 = new_sel;
    }
}

/// On selection change: paint the selected button green, reset others to grey,
/// scroll the list so the selected button is in view.
fn apply_selection_highlight(
    selected:      Res<SelectedOrganism>,
    mut buttons:   Query<(&NavigatorButton, &mut BackgroundColor)>,
    list_q:        Query<&Children, With<NavigatorList>>,
    button_lookup: Query<&NavigatorButton>,
    mut scroll_q:  Query<&mut ScrollPosition, With<NavigatorList>>,
) {
    if !selected.is_changed() { return; }

    for (btn, mut bg) in &mut buttons {
        *bg = if Some(btn.target) == selected.0 {
            BackgroundColor(Color::srgb(0.20, 0.55, 0.25)) // selected → green
        } else {
            BackgroundColor(Color::srgb(0.25, 0.25, 0.25)) // default grey
        };
    }

    let Some(target) = selected.0 else { return };
    let Ok(children)  = list_q.single() else { return };
    let Ok(mut scroll) = scroll_q.single_mut() else { return };
    for i in 0..children.len() {
        let child = children[i];
        if let Ok(btn) = button_lookup.get(child) {
            if btn.target == target {
                scroll.y = i as f32 * (NAV_BUTTON_HEIGHT_PX + NAV_BUTTON_GAP_PX);
                break;
            }
        }
    }
}

/// Mirror identifier set ↔ label set: spawn a label (child of `ViewportImage`)
/// per newly-tagged organism, despawn labels whose target is gone.
/// Parenting under `ViewportImage` keeps projection trivial: both
/// `world_to_viewport` and the label's `Val::Px` share the viewport top-left
/// origin, so the per-frame update is just a scale-factor divide.
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
                // Two-line label (name over species); column so species wraps beneath.
                flex_direction: FlexDirection::Column,
                align_items:    AlignItems::Center,
                ..default()
            },
            BackgroundColor(LABEL_BG),
            Pickable::IGNORE,
            Visibility::Hidden,
        ))
        .with_children(|wrap| {
            wrap.spawn((
                IndividualLabelText { target: entity },
                Text::new(ident.0.clone()),
                TextFont { font_size: LABEL_FONT_SIZE, ..default() },
                TextColor(Color::WHITE),
                Pickable::IGNORE,
            ));
            // Species sub-line; empty until classified, then kept in sync
            // by `update_label_species_text`.
            wrap.spawn((
                IndividualLabelSpecies { target: entity },
                Text::new(String::new()),
                TextFont {
                    font_size: LABEL_FONT_SIZE * 0.85,
                    ..default()
                },
                TextColor(Color::srgb(0.75, 0.85, 1.0)),
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

/// Project each label-target's world position into the viewport and set the
/// label's absolute UI position. Labels outside the viewport rect (off-screen /
/// behind camera) are hidden so they never bleed onto the side panels.
fn update_label_positions(
    show:        Res<ShowIndividualIdentifiers>,
    window_mode: Res<crate::simulation_settings::WindowMode>,
    cameras:     Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    viewport_q:  Query<&ComputedNode, With<ViewportImage>>,
    organism_q:  Query<&GlobalTransform, With<OrganismRoot>>,
    // Base-body-part transforms, keyed by parent OrganismRoot. For LIMB
    // organisms the root transform is frozen at spawn and only the dynamic
    // parts move, so the label tracks the trunk part (`BodyPartIndex(0)`),
    // the true location. For sliding organisms the trunk's GlobalTransform
    // equals the root's — correct on both paths.
    //
    // We read the part's pose from the AUTHORITATIVE Rapier body (when present),
    // not its `GlobalTransform`: for swimmer multibody links bevy_rapier writes a
    // garbage Transform during writeback that is only corrected in `PostUpdate`
    // (`sync_multibody_link_transforms`), so the `Update`-time `GlobalTransform`
    // read here is wrong (was the cause of ghost labels at random locations).
    base_part_q: Query<(&ChildOf, &crate::cell::BodyPartIndex, &GlobalTransform,
                        Option<&bevy_rapier3d::prelude::RapierRigidBodyHandle>)>,
    rb_set_q:    Query<&bevy_rapier3d::prelude::RapierRigidBodySet>,
    mut labels:  Query<(&IndividualLabel, &ComputedNode, &mut Node, &mut Visibility)>,
) {
    // Hide all labels outside Simulation mode. The labels are UI nodes under
    // the viewport image, so `RenderLayers` doesn't filter them like 3D meshes
    // — they need an explicit visibility gate.
    let labels_disabled = !show.0
        || *window_mode != crate::simulation_settings::WindowMode::Simulation;
    if labels_disabled {
        for (_, _, _, mut v) in &mut labels {
            *v = Visibility::Hidden;
        }
        return;
    }

    let Ok((camera, cam_xf)) = cameras.single() else { return };
    let Ok(viewport_node)    = viewport_q.single() else { return };

    // Labels are children of `ViewportImage`, so their `Val::Px` origin is the
    // viewport top-left. `world_to_viewport` returns physical pixels in the same
    // top-left/Y-down space (image targets default scale_factor=1.0); multiply
    // by inverse_scale_factor for window-logical px (`Val::Px`). It returns Err
    // behind the camera / beyond depth (hidden for free); out-of-X/Y still
    // returns Ok, so the explicit bounds check below hides those.
    let inv_scale         = viewport_node.inverse_scale_factor;
    let viewport_size     = viewport_node.size();

    // Map each OrganismRoot to its trunk part's (`BodyPartIndex(0)`) world
    // position — the moving anchor for limb organisms.
    let rb_set = rb_set_q.single().ok();
    let mut base_pos: HashMap<Entity, Vec3> = HashMap::new();
    for (parent, idx, gx, handle) in &base_part_q {
        if idx.0 != 0 { continue; }
        // Authoritative Rapier pose if the part is a registered body; else the
        // propagated GlobalTransform (correct for non-physics / sliding paths).
        let pos = handle
            .zip(rb_set)
            .and_then(|(h, set)| set.bodies.get(h.0))
            .map(|rb| {
                let t = rb.position().translation;
                Vec3::new(t.x, t.y, t.z)
            })
            .unwrap_or_else(|| gx.translation());
        base_pos.insert(parent.parent(), pos);
    }

    for (label, label_node, mut node, mut vis) in &mut labels {
        // Prefer the trunk part; fall back to the root transform.
        let anchor = match base_pos.get(&label.target) {
            Some(p) => *p,
            None => match organism_q.get(label.target) {
                Ok(xf) => xf.translation(),
                Err(_) => {
                    *vis = Visibility::Hidden;
                    continue;
                }
            },
        };

        let world_pos = anchor + Vec3::new(0.0, LABEL_WORLD_LIFT, 0.0);
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
        // Centre on the anchor, bottom edge a few px above it.
        let label_size_logical = label_node.size() * label_node.inverse_scale_factor;
        node.left = Val::Px(anchor_logical.x - label_size_logical.x * 0.5);
        node.top  = Val::Px(anchor_logical.y - label_size_logical.y - LABEL_ANCHOR_GAP_PX);
        *vis = Visibility::Inherited;
    }
}

/// Sync the navigator-list buttons with the labelled-organism set (spawn on new
/// identifier, despawn when the target is gone).
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

    for (entity, ident, _organism) in &new_q {
        // Row: identifier text left, "Export" button right. Outer container is
        // layout-only; the Export sub-button is the only interactive child.
        let btn = commands.spawn((
            NavigatorButton { target: entity },
            Node {
                width:           Val::Percent(100.0),
                height:          Val::Px(NAV_BUTTON_HEIGHT_PX),
                margin:          UiRect::bottom(Val::Px(NAV_BUTTON_GAP_PX)),
                padding:         UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                flex_direction:  FlexDirection::Row,
                align_items:     AlignItems::Center,
                justify_content: JustifyContent::SpaceBetween,
                flex_shrink:     0.0,
                ..default()
            },
            BackgroundColor(Color::srgb(0.25, 0.25, 0.25)),
        ))
        .with_children(|row| {
            row.spawn((
                NavigatorButtonText { target: entity },
                Text::new(ident.0.clone()),
                TextFont { font_size: 13.0, ..default() },
                TextColor(Color::WHITE),
                Pickable::IGNORE,
            ));
            row.spawn((
                NavigatorExportButton { target: entity },
                Button,
                Node {
                    height:          Val::Px(24.0),
                    padding:         UiRect::axes(Val::Px(8.0), Val::Px(2.0)),
                    align_items:     AlignItems::Center,
                    justify_content: JustifyContent::Center,
                    flex_shrink:     0.0,
                    ..default()
                },
                BackgroundColor(Color::srgb(0.22, 0.46, 0.46)),
            ))
            .with_children(|b| {
                b.spawn((
                    Text::new("Export"),
                    TextFont { font_size: 11.0, ..default() },
                    TextColor(Color::WHITE),
                    Pickable::IGNORE,
                ));
            });
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

/// Mouse-wheel scroll for the navigator list; applied when the cursor is inside
/// the panel rect. (No `Pointer<Scroll>` in Bevy 0.18 — we do the rect check.)
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

    // `cursor_position` is window-logical px (top-left origin);
    // `UiGlobalTransform.translation` is the node centre in physical px (same
    // origin). One multiply by `inverse_scale_factor` matches their units.
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
    // Drag right ⇒ shrink, left ⇒ expand. ev.distance is cumulative from
    // DragStart; anchoring on initial width gives a stable absolute width.
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

/// Reveal / hide the min-count input row on `auto` change (edge-triggered).
fn update_min_hetero_row_visibility(
    auto:      Res<AutoSpawnHeteros>,
    mut row_q: Query<&mut Node, With<MinHeteroCountRow>>,
) {
    if !auto.is_changed() { return; }
    for mut node in &mut row_q {
        node.display = if auto.0 { Display::Flex } else { Display::None };
    }
}

/// Click + keyboard router for the min-count field (mirrors
/// `handle_max_organisms_input`): click focuses, Enter / click-outside commits,
/// Escape cancels. Digits-only.
fn handle_min_hetero_input(
    mouse:         Res<ButtonInput<MouseButton>>,
    mut keyboard:  MessageReader<KeyboardInput>,
    auto:          Res<AutoSpawnHeteros>,
    interaction_q: Query<&Interaction, With<MinHeteroCountInput>>,
    mut state:     ResMut<MinHeteroCountEditState>,
    mut min_count: ResMut<MinHeteroCount>,
) {
    // Row hidden: drain the keyboard reader so events don't accumulate.
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


/// Show the "Species:" rename row only while an organism WITH a classified
/// species is selected; hide + unfocus otherwise.
fn update_species_rename_row_visibility(
    selected:  Res<SelectedOrganism>,
    organisms: Query<&crate::organism::Organism>,
    mut row_q: Query<&mut Node, With<SpeciesRenameRow>>,
    mut state: ResMut<SpeciesRenameEditState>,
) {
    let has_species = selected.0
        .and_then(|e| organisms.get(e).ok())
        .and_then(|o| o.species_id)
        .is_some();
    for mut node in &mut row_q {
        let want = if has_species { Display::Flex } else { Display::None };
        if node.display != want { node.display = want; }
    }
    if !has_species && state.focused {
        state.focused = false;
        state.buffer.clear();
    }
}

/// Click + keyboard router for the species-rename field. Click focuses (seeds
/// the buffer with the current name), Enter / click-outside commits the rename
/// into `SpeciesRegistry` (renaming the selected organism's whole species —
/// every member shares the name, and the per-species brain, keyed by id, is
/// untouched), Escape cancels. Accepts any printable character.
fn handle_species_rename_input(
    mouse:         Res<ButtonInput<MouseButton>>,
    mut keyboard:  MessageReader<KeyboardInput>,
    selected:      Res<SelectedOrganism>,
    organisms:     Query<&crate::organism::Organism>,
    mut registry:  ResMut<crate::lineages::species::SpeciesRegistry>,
    interaction_q: Query<&Interaction, With<SpeciesRenameInput>>,
    mut state:     ResMut<SpeciesRenameEditState>,
) {
    let species_id = selected.0
        .and_then(|e| organisms.get(e).ok())
        .and_then(|o| o.species_id);
    let Some(species_id) = species_id else {
        for _ in keyboard.read() {}
        if state.focused { state.focused = false; state.buffer.clear(); }
        return;
    };

    let click_on_input = mouse.just_pressed(MouseButton::Left)
        && interaction_q.iter().any(|i| matches!(i, Interaction::Pressed));
    let click_outside  = mouse.just_pressed(MouseButton::Left) && !click_on_input;

    if click_on_input && !state.focused {
        state.focused = true;
        state.buffer.clear();
        if let Some(s) = registry.get(species_id) {
            state.buffer.push_str(&s.name);
        }
    }
    if click_outside && state.focused {
        commit_species_rename(&mut state, &mut registry, species_id);
    }
    if !state.focused {
        for _ in keyboard.read() {}
        return;
    }

    for ev in keyboard.read() {
        if !ev.state.is_pressed() { continue; }
        match ev.key_code {
            KeyCode::Enter | KeyCode::NumpadEnter => {
                commit_species_rename(&mut state, &mut registry, species_id);
            }
            KeyCode::Escape => { state.focused = false; state.buffer.clear(); }
            KeyCode::Backspace => { state.buffer.pop(); }
            _ => {
                if let Some(text) = ev.text.as_ref() {
                    for c in text.chars() {
                        if state.buffer.len() >= SPECIES_RENAME_BUFFER_MAX_LEN { break; }
                        if !c.is_control() { state.buffer.push(c); }
                    }
                }
            }
        }
    }
}

fn commit_species_rename(
    state:      &mut SpeciesRenameEditState,
    registry:   &mut crate::lineages::species::SpeciesRegistry,
    species_id: u32,
) {
    let name = state.buffer.trim().to_string();
    if !name.is_empty() {
        if let Some(s) = registry.get_mut(species_id) {
            if s.name != name { s.name = name; }
        }
    }
    state.focused = false;
    state.buffer.clear();
}

/// Refresh the rename field's text: the live species name when idle, the edit
/// buffer (with a cursor) while focused; background reflects focus.
fn update_species_rename_text(
    selected:   Res<SelectedOrganism>,
    organisms:  Query<&crate::organism::Organism>,
    registry:   Res<crate::lineages::species::SpeciesRegistry>,
    state:      Res<SpeciesRenameEditState>,
    mut text_q: Query<&mut Text, With<SpeciesRenameText>>,
    mut bg_q:   Query<&mut BackgroundColor, With<SpeciesRenameInput>>,
) {
    let current = selected.0
        .and_then(|e| organisms.get(e).ok())
        .and_then(|o| o.species_id)
        .and_then(|id| registry.get(id).map(|s| s.name.clone()))
        .unwrap_or_default();
    let display = if state.focused { format!("{}_", state.buffer) } else { current };
    for mut text in &mut text_q {
        if text.0 != display { text.0 = display.clone(); }
    }
    let bg = if state.focused { MIN_HETERO_BG_FOCUSED } else { MIN_HETERO_BG_IDLE };
    for mut b in &mut bg_q {
        if b.0 != bg { *b = BackgroundColor(bg); }
    }
}


/// Refresh each label's species sub-text from `Organism::species_id` via
/// `SpeciesRegistry`. Writes only on change (avoids `Changed<Text>` churn).
fn update_label_species_text(
    organisms:    Query<&crate::organism::Organism>,
    registry:     Res<crate::lineages::species::SpeciesRegistry>,
    mut texts:    Query<(&IndividualLabelSpecies, &mut Text)>,
) {
    for (marker, mut text) in &mut texts {
        let new = match organisms.get(marker.target) {
            Ok(org) => match org.species_id {
                Some(id) => registry.get(id)
                    .map(|s| s.name.clone())
                    .unwrap_or_default(),
                None => String::new(),
            },
            Err(_) => String::new(),
        };
        if text.0 != new { text.0 = new; }
    }
}


// ── Export Trained Species — per-row button handlers ───────────────────────
//
// Two-stage workflow:
//   1. `handle_export_buttons` — pause the sim, open a blocking save dialog,
//      stash `(entity, path)` in `ExportSpeciesRequested`. Pause is so the
//      step-2 brain snapshot isn't smeared by an in-progress training tick.
//   2. `dispatch_export_species_requests` — pull the brain, extract body plan +
//      (bilateral-right-half) OCG, write a `.species` v3 file. Resets to `None`.

/// Per-row Export button handler. Blocks on a save dialog; on success writes
/// `(entity, path)` into `ExportSpeciesRequested` for the worker.
fn handle_export_buttons(
    interactions: Query<
        (&Interaction, &NavigatorExportButton),
        Changed<Interaction>,
    >,
    mut sim_running:  ResMut<crate::simulation_settings::SimulationRunning>,
    mut virtual_time: ResMut<Time<Virtual>>,
    mut request:      ResMut<ExportSpeciesRequested>,
) {
    for (interaction, btn) in &interactions {
        if !matches!(*interaction, Interaction::Pressed) { continue; }

        if sim_running.0 {
            sim_running.0 = false;
            virtual_time.pause();
        }

        let initial_dir = std::env::current_dir()
            .ok()
            .map(|d| d.join("species"))
            .unwrap_or_else(|| std::path::PathBuf::from("species"));
        let default_name = format!(
            "trained_species_{}.species",
            chrono::Local::now().format("%d-%m-%Y-%H-%M-%S"),
        );
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("AEONS species (.species)", &["species"])
            .set_directory(initial_dir)
            .set_file_name(default_name)
            .save_file()
        {
            request.0 = Some((btn.target, path));
        }
    }
}

/// Worker — consumes one pending export request, snapshots the organism's brain
/// from whichever pool it belongs to (sliding herbivore_1, a limb walker pool,
/// or the swim pool), encodes a v10 `.species` payload, writes to disk.
///
/// Sliding herbivores keep the legacy single bilateral right-half "Base Body"
/// export. Walkers and swimmers are exported as their LITERAL multi-part body
/// plan with `symmetry = NoSymmetry` (each runtime part verbatim) so respawn
/// rebuilds the exact structure the PPO policy was trained on.
fn dispatch_export_species_requests(
    mut request: ResMut<ExportSpeciesRequested>,
    pool_sl:     NonSend<crate::intelligence_level_herbivore_1_sliding::BrainPoolHerbivore1>,
    pool_lh:     NonSend<crate::intelligence_level_herbivore_1_limb::BrainPoolHerbivore1Limb>,
    pool_l2:     NonSend<crate::intelligence_level_2_limb::BrainPoolL2Limb>,
    pool_l3:     NonSend<crate::intelligence_level_3_limb::BrainPoolL3Limb>,
    pool_sw:     NonSend<crate::intelligence_level_1_swimming::BrainPoolSwim1>,
    query:       Query<
        (
            &Organism,
            Has<crate::colony::Photoautotroph>,
            Has<Heterotroph>,
            Has<crate::colony::Carnivore>,
            Option<&crate::intelligence_level_herbivore_1_sliding::BrainSlotHerbivore1>,
            Option<&crate::intelligence_level_herbivore_1_limb::BrainSlotHerbivore1Limb>,
            Option<&crate::intelligence_level_2_limb::BrainSlotL2Limb>,
            Option<&crate::intelligence_level_3_limb::BrainSlotL3Limb>,
            Option<&crate::intelligence_level_1_swimming::BrainSlotSwim1>,
        ),
        With<OrganismRoot>,
    >,
) {
    use crate::species_editor::session::{Classification, Metabolism};
    use crate::species_editor::save::LoadedBrain;
    use crate::organism::{Symmetry, MovementMode};

    let Some((entity, path)) = request.0.take() else { return };

    let Ok((org, is_photo, is_hetero, is_carn, sl, lh, l2, l3, sw)) = query.get(entity) else {
        warn!("export species: entity {:?} not found", entity);
        return;
    };
    let metabolism = if is_photo {
        Metabolism::Photoautotroph
    } else if is_hetero {
        Metabolism::Heterotroph
    } else {
        warn!("export species: entity {:?} has neither Photo nor Hetero marker", entity);
        return;
    };
    let classification = if is_carn { Classification::Carnivore } else { Classification::Herbivore };
    if org.body_parts.is_empty() {
        warn!("export species: entity {:?} has no body parts", entity);
        return;
    }

    // Per pool: build the body-plan `parts`, symmetry, movement, ground, and the
    // brain payload. `None` ⇒ this organism has no exportable trained brain yet.
    type Parts = Vec<(String, crate::cell::BodyPartKind, usize, Vec<(usize, bevy::math::Vec3, crate::cell::CellType)>)>;
    let export: Option<(Parts, Symmetry, MovementMode, bool, LoadedBrain)> = if let Some(slot) = sl {
        // SLIDING herbivore_1 — single bilateral right-half "Base Body" (legacy).
        let ocg_full = &org.body_parts[0].ocg;
        let ocg: Vec<_> = match org.symmetry {
            Symmetry::NoSymmetry => ocg_full.clone(),
            Symmetry::Bilateral  => ocg_full.iter().filter(|(_, p, _)| p.x > 0.0).copied().collect(),
        };
        if ocg.is_empty() { None } else {
            Some((vec![("Base Body".to_string(), crate::cell::BodyPartKind::Body, 0, ocg)],
                  org.symmetry, MovementMode::Sliding, true,
                  LoadedBrain::Sliding(pool_sl.extract_slot(slot.0))))
        }
    } else {
        // PPO walker / swimmer — literal multi-part body plan, NoSymmetry.
        let restore = if lh.is_some() { pool_lh.0.snapshot().extract(entity) }
            else if l2.is_some()      { pool_l2.0.snapshot().extract(entity) }
            else if l3.is_some()      { pool_l3.0.snapshot().extract(entity) }
            else if sw.is_some()      { pool_sw.0.extract_species_for(entity) }
            else { None };
        restore.map(|r| {
            let parts: Parts = org.body_parts.iter().enumerate().map(|(i, bp)| {
                let name = if i == 0 { "Base Body".to_string() } else { format!("Part {i}") };
                // Literal runtime kind: a trained NoSymmetry export rebuilds the
                // exact structure (no re-mirroring), so the kind round-trips as-is.
                let parent  = bp.attachment.as_ref().map(|a| a.parent_idx).unwrap_or(0);
                (name, bp.kind, parent, bp.ocg.clone())
            }).collect();
            (parts, Symmetry::NoSymmetry, org.movement_mode, org.ground_based, LoadedBrain::Ppo(r))
        })
    };

    let Some((parts, symmetry, movement, ground_based, brain)) = export else {
        warn!(
            "export species: entity {:?} has no exportable trained brain — needs a \
             live brain slot in the sliding-L1-herbivore, a walker (limb), or the \
             swimmer pool (Level0 / sessile / not-yet-enrolled organisms can't be exported)",
            entity,
        );
        return;
    };

    let bytes = crate::species_editor::save::encode_species_with_brain(
        metabolism, symmetry, org.intelligence_level, org.has_variable_form, org.is_sessile,
        classification, movement, ground_based, &parts, &brain,
    );

    // Ensure target directory exists (user may have picked outside ./species/).
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                error!("export species: failed to create dir {}: {}", parent.display(), e);
                return;
            }
        }
    }

    let cells: usize = parts.iter().map(|p| p.3.len()).sum();
    match std::fs::write(&path, &bytes) {
        Ok(()) => info!(
            "trained species exported to {} — {} parts, {} cells, {} bytes",
            path.display(), parts.len(), cells, bytes.len(),
        ),
        Err(e) => error!("export species: write to {} failed: {}", path.display(), e),
    }
}
