// Species editor — Body-part index panel (right side).
//
// Vertical list of body parts (index 0 = base body, later = appendages).
// Click a row to make it the active placement target; click the active row
// again to rename (Enter commits, Escape cancels). "Begin New Body-Part"
// appends an appendage, makes it active, and selects the blue Placeholder
// type; its first cell attaches by contact (see `placement`).

use bevy::input::keyboard::KeyboardInput;
use bevy::prelude::*;
use bevy::ui::{ComputedNode, UiGlobalTransform};

use crate::cell::{BodyPartKind, CellType};
use crate::frontend::PANEL_BG_COLOR;
use crate::simulation_settings::WindowMode;

use super::session::{EditorBodyPart, SpeciesSession};
use super::TOP_PANEL_HEIGHT_PX;


// ── Tunables ─────────────────────────────────────────────────────────────────

pub const BODY_PART_PANEL_WIDTH_PX: f32 = 240.0;

const ROW_HEIGHT_PX:   f32 = 30.0;
const RENAME_MAX_LEN:  usize = 28;

const ROW_BG_IDLE:     Color = Color::srgb(0.18, 0.18, 0.18);
const ROW_BG_ACTIVE:   Color = Color::srgb(0.20, 0.40, 0.62);
const ROW_BG_HOVER:    Color = Color::srgb(0.30, 0.30, 0.30);
const ROW_BG_RENAME:   Color = Color::srgb(0.45, 0.40, 0.12);

const BEGIN_BG:        Color = Color::srgb(0.20, 0.45, 0.70);
const BEGIN_BG_HOVER:  Color = Color::srgb(0.30, 0.55, 0.80);
const BEGIN_BG_DISABLED: Color = Color::srgb(0.18, 0.18, 0.18);

const KIND_BTN_WIDTH_PX: f32 = 78.0;
const KIND_BG:           Color = Color::srgb(0.26, 0.30, 0.42);
const KIND_BG_HOVER:     Color = Color::srgb(0.34, 0.40, 0.55);
const KIND_OPT_BG:       Color = Color::srgb(0.20, 0.22, 0.28);
const KIND_OPT_BG_HOVER: Color = Color::srgb(0.30, 0.40, 0.55);
const KIND_OPT_BG_SEL:   Color = Color::srgb(0.20, 0.40, 0.62);

/// The three appendage kinds the dropdown offers, in display order.
const KIND_OPTIONS: [BodyPartKind; 3] =
    [BodyPartKind::Limb, BodyPartKind::Segment, BodyPartKind::Static];

/// Display label for an appendage kind (base body uses "Body").
fn kind_label(k: BodyPartKind) -> &'static str {
    match k {
        BodyPartKind::Body    => "Body",
        BodyPartKind::Limb    => "Limb",
        BodyPartKind::Organ   => "Organ",
        BodyPartKind::Segment => "Segment",
        BodyPartKind::Static  => "Static",
    }
}


// ── Markers ──────────────────────────────────────────────────────────────────

#[derive(Component)]
pub struct BodyPartPanel;

/// Column container the per-part rows are parented under.
#[derive(Component)]
pub struct BodyPartList;

#[derive(Component)]
pub struct BeginNewBodyPartButton;

/// One row (layout Node, not a Button); its select-button + limb-toggle
/// children carry the interactions. Holds the body-part index.
#[derive(Component, Clone, Copy)]
pub struct BodyPartRow(pub usize);

/// Click target for selecting / renaming a body part. One per row.
#[derive(Component, Clone, Copy)]
pub struct BodyPartSelectButton(pub usize);

/// Per-row "Kind" dropdown button (shows the current kind, opens the menu).
/// Hidden for the base body (index 0).
#[derive(Component, Clone, Copy)]
pub struct BodyPartKindButton(pub usize);

/// Text child of a `BodyPartKindButton`, so the sync system updates its label.
#[derive(Component, Clone, Copy)]
pub struct BodyPartKindLabel(pub usize);

/// The open kind-dropdown overlay (root UI node), tagged with the body-part
/// index it belongs to.
#[derive(Component, Clone, Copy)]
pub struct KindDropdownMenu(pub usize);

/// One option button inside an open kind dropdown.
#[derive(Component, Clone, Copy)]
pub struct KindDropdownOption {
    pub part: usize,
    pub kind: BodyPartKind,
}

/// Text child of a select button, so the label-sync system finds it directly.
#[derive(Component, Clone, Copy)]
pub struct BodyPartRowLabel(pub usize);


// ── Spawn ────────────────────────────────────────────────────────────────────

pub fn spawn_body_part_panel(parent: &mut ChildSpawnerCommands, top_offset_px: f32) {
    parent
        .spawn((
            BodyPartPanel,
            super::SpeciesEditorPanel,
            Node {
                position_type: PositionType::Absolute,
                top:    Val::Px(top_offset_px + TOP_PANEL_HEIGHT_PX),
                right:  Val::Px(0.0),
                bottom: Val::Px(super::BOTTOM_PANEL_HEIGHT_PX),
                width:  Val::Px(BODY_PART_PANEL_WIDTH_PX),
                padding: UiRect::all(Val::Px(8.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(6.0),
                display: Display::None,   // shown only in SpeciesEditor mode
                ..default()
            },
            BackgroundColor(PANEL_BG_COLOR),
        ))
        .with_children(|panel| {
            panel.spawn((
                Text::new("Body part index"),
                TextFont { font_size: 15.0, ..default() },
                TextColor(Color::WHITE),
                Pickable::IGNORE,
            ));

            panel
                .spawn((
                    BeginNewBodyPartButton,
                    Button,
                    Node {
                        width:  Val::Percent(100.0),
                        height: Val::Px(32.0),
                        align_items:     AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        ..default()
                    },
                    BackgroundColor(BEGIN_BG),
                ))
                .with_children(|b| {
                    b.spawn((
                        Text::new("Begin New Body-Part"),
                        TextFont { font_size: 13.0, ..default() },
                        TextColor(Color::WHITE),
                        Pickable::IGNORE,
                    ));
                });

            panel.spawn((
                BodyPartList,
                Node {
                    width: Val::Percent(100.0),
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(4.0),
                    overflow: Overflow::scroll_y(),
                    ..default()
                },
            ));
        });
}


// ── "Begin New Body-Part" ─────────────────────────────────────────────────────

pub fn handle_begin_new_body_part(
    mode:             Res<WindowMode>,
    mut interactions: Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<BeginNewBodyPartButton>)>,
    mut session:      ResMut<SpeciesSession>,
) {
    if *mode != WindowMode::SpeciesEditor { return; }

    for (interaction, mut bg) in &mut interactions {
        let enabled = session.first_cell_spawned;   // disabled until base body exists
        match *interaction {
            Interaction::Pressed if enabled => {
                let n = session.body_parts.len();
                // No fixed parent: it attaches to whichever part its FIRST
                // cell touches (decided by contact in `placement.rs`);
                // `parent = 0` is a placeholder until then.
                session.body_parts.push(EditorBodyPart {
                    name: format!("Body Part {n}"),
                    ocg:  Vec::new(),
                    kind: BodyPartKind::Static,   // default; user re-assigns via the Kind dropdown
                    parent: 0,
                });
                session.active_body_part = n;
                session.renaming_body_part = None;
                session.kind_dropdown_open = None;
                // New parts are sketched in the blue Placeholder type.
                session.selected_cell_type = Some(CellType::Placeholder);
                session.dirty = true;
                *bg = BackgroundColor(BEGIN_BG_HOVER);
            }
            Interaction::Pressed => *bg = BackgroundColor(BEGIN_BG_DISABLED),
            Interaction::Hovered if enabled => *bg = BackgroundColor(BEGIN_BG_HOVER),
            _ => *bg = BackgroundColor(if enabled { BEGIN_BG } else { BEGIN_BG_DISABLED }),
        }
    }
}


// ── Row list lifecycle ────────────────────────────────────────────────────────

/// Rebuild the row entities whenever the number of body parts changes.
/// Per-frame label text + highlight are handled by `sync_body_part_rows`,
/// so this only fires on add/remove.
pub fn manage_body_part_list(
    session:    Res<SpeciesSession>,
    mut commands: Commands,
    list_q:     Query<Entity, With<BodyPartList>>,
    rows_q:     Query<(Entity, &BodyPartRow)>,
) {
    let Ok(list) = list_q.single() else { return };
    let want = session.body_parts.len();
    let have = rows_q.iter().count();
    if want == have { return; }

    for (e, _) in &rows_q { commands.entity(e).despawn(); }

    commands.entity(list).with_children(|list| {
        for (i, part) in session.body_parts.iter().enumerate() {
            list.spawn((
                BodyPartRow(i),
                Node {
                    width:  Val::Percent(100.0),
                    height: Val::Px(ROW_HEIGHT_PX),
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Stretch,
                    column_gap: Val::Px(4.0),
                    ..default()
                },
            ))
            .with_children(|row| {
                // Name select button — clicking it selects this part
                // as the active placement target (and starts rename
                // on a second click).
                row.spawn((
                    BodyPartSelectButton(i),
                    Button,
                    Node {
                        flex_grow: 1.0,
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::FlexStart,
                        padding: UiRect::horizontal(Val::Px(8.0)),
                        ..default()
                    },
                    BackgroundColor(ROW_BG_IDLE),
                ))
                .with_children(|b| {
                    b.spawn((
                        BodyPartRowLabel(i),
                        Text::new(part.name.clone()),
                        TextFont { font_size: 13.0, ..default() },
                        TextColor(Color::WHITE),
                        Pickable::IGNORE,
                    ));
                });

                // "Kind" dropdown button — appendages only; the base body is
                // always Body. Clicking opens a Limb / Segment / Static menu.
                if i != 0 {
                    row.spawn((
                        BodyPartKindButton(i),
                        Button,
                        Node {
                            width:  Val::Px(KIND_BTN_WIDTH_PX),
                            align_items: AlignItems::Center,
                            justify_content: JustifyContent::Center,
                            ..default()
                        },
                        BackgroundColor(KIND_BG),
                    ))
                    .with_children(|b| {
                        b.spawn((
                            BodyPartKindLabel(i),
                            Text::new(kind_label(part.kind)),
                            TextFont { font_size: 11.0, ..default() },
                            TextColor(Color::WHITE),
                            Pickable::IGNORE,
                        ));
                    });
                }
            });
        }
    });
}


// ── Row clicks (select / begin rename) ────────────────────────────────────────

pub fn handle_body_part_row_clicks(
    mode:             Res<WindowMode>,
    mut interactions: Query<(&Interaction, &BodyPartSelectButton), Changed<Interaction>>,
    mut session:      ResMut<SpeciesSession>,
) {
    if *mode != WindowMode::SpeciesEditor { return; }

    for (interaction, sel) in &mut interactions {
        if !matches!(*interaction, Interaction::Pressed) { continue; }
        let i = sel.0;
        if i >= session.body_parts.len() { continue; }

        if session.active_body_part == i && session.renaming_body_part != Some(i) {
            // Second click on the already-active row → start renaming.
            session.renaming_body_part = Some(i);
            session.rename_buffer = session.body_parts[i].name.clone();
        } else {
            // Select this part as the active placement target.
            session.active_body_part = i;
            session.renaming_body_part = None;
        }
        session.kind_dropdown_open = None;   // selecting a row closes any open dropdown
    }
}


// ── "Kind" dropdown ─────────────────────────────────────────────────────────

/// Click the per-row Kind button: toggle the dropdown open/closed for that part.
pub fn handle_kind_button_clicks(
    mode:             Res<WindowMode>,
    mut interactions: Query<(&Interaction, &BodyPartKindButton), Changed<Interaction>>,
    mut session:      ResMut<SpeciesSession>,
) {
    if *mode != WindowMode::SpeciesEditor { return; }
    for (interaction, btn) in &mut interactions {
        if !matches!(*interaction, Interaction::Pressed) { continue; }
        let i = btn.0;
        if i == 0 { continue; }   // base body has no kind
        session.kind_dropdown_open =
            if session.kind_dropdown_open == Some(i) { None } else { Some(i) };
    }
}

/// Spawn / despawn / reposition the open Kind dropdown overlay. The menu is a
/// root UI node (above the panels) anchored just below its row's Kind button —
/// rooted (not a child of the scrollable list) so `overflow: scroll` can't clip it.
pub fn manage_kind_dropdown(
    mode:        Res<WindowMode>,
    mut session: ResMut<SpeciesSession>,
    mut commands: Commands,
    buttons:     Query<(&BodyPartKindButton, &UiGlobalTransform, &ComputedNode)>,
    mut menus:   Query<(Entity, &KindDropdownMenu, &mut Node)>,
) {
    // Outside the editor: force-close and drop every overlay.
    if *mode != WindowMode::SpeciesEditor {
        if session.kind_dropdown_open.is_some() { session.kind_dropdown_open = None; }
        for (e, _, _) in &menus { commands.entity(e).despawn(); }
        return;
    }
    let open = session.kind_dropdown_open;

    // Screen position (logical px) of the open part's button, if visible.
    let target = open.and_then(|part| {
        if part == 0 { return None; }
        buttons.iter().find(|(b, _, _)| b.0 == part).map(|(_, xf, node)| {
            let inv = node.inverse_scale_factor;
            let size = node.size() * inv;
            let centre = xf.translation * inv;
            (part, Vec2::new(centre.x - size.x * 0.5, centre.y + size.y * 0.5))
        })
    });

    // Reconcile existing menus: keep+reposition the one for `open`, drop the rest.
    let mut have_for: Option<usize> = None;
    for (e, m, mut node) in &mut menus {
        if Some(m.0) == open {
            have_for = Some(m.0);
            if let Some((_, pos)) = target {
                node.left = Val::Px(pos.x);
                node.top  = Val::Px(pos.y);
            }
        } else {
            commands.entity(e).despawn();
        }
    }

    // Spawn the menu the first frame it opens (and the button position is known).
    if let Some((part, pos)) = target {
        if have_for != Some(part) {
            spawn_kind_menu(&mut commands, part, pos);
        }
    }
}

fn spawn_kind_menu(commands: &mut Commands, part: usize, pos: Vec2) {
    commands
        .spawn((
            KindDropdownMenu(part),
            Node {
                position_type: PositionType::Absolute,
                left:  Val::Px(pos.x),
                top:   Val::Px(pos.y),
                width: Val::Px(KIND_BTN_WIDTH_PX + 8.0),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            GlobalZIndex(200),
        ))
        .with_children(|menu| {
            for kind in KIND_OPTIONS {
                menu.spawn((
                    KindDropdownOption { part, kind },
                    Button,
                    Node {
                        width:  Val::Percent(100.0),
                        height: Val::Px(ROW_HEIGHT_PX - 2.0),
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        border: UiRect::all(Val::Px(1.0)),
                        ..default()
                    },
                    BackgroundColor(KIND_OPT_BG),
                    BorderColor::all(Color::srgb(0.12, 0.12, 0.14)),
                ))
                .with_children(|b| {
                    b.spawn((
                        Text::new(kind_label(kind)),
                        TextFont { font_size: 12.0, ..default() },
                        TextColor(Color::WHITE),
                        Pickable::IGNORE,
                    ));
                });
            }
        });
}

/// Click an option in the open dropdown: set the part's kind and close.
pub fn handle_kind_option_clicks(
    mode:             Res<WindowMode>,
    mut interactions: Query<(&Interaction, &KindDropdownOption), Changed<Interaction>>,
    mut session:      ResMut<SpeciesSession>,
) {
    if *mode != WindowMode::SpeciesEditor { return; }
    for (interaction, opt) in &mut interactions {
        if !matches!(*interaction, Interaction::Pressed) { continue; }
        if opt.part != 0 {
            if let Some(part) = session.body_parts.get_mut(opt.part) {
                if part.kind != opt.kind {
                    part.kind = opt.kind;
                    session.dirty = true;
                }
            }
        }
        session.kind_dropdown_open = None;
    }
}


// ── Rename keyboard input ──────────────────────────────────────────────────────

pub fn handle_rename_input(
    mode:        Res<WindowMode>,
    mut keyboard: MessageReader<KeyboardInput>,
    mut session: ResMut<SpeciesSession>,
) {
    let renaming = if *mode == WindowMode::SpeciesEditor { session.renaming_body_part } else { None };
    let Some(i) = renaming else {
        for _ in keyboard.read() {}   // drain
        return;
    };
    if i >= session.body_parts.len() {
        session.renaming_body_part = None;
        for _ in keyboard.read() {}
        return;
    }

    for ev in keyboard.read() {
        if !ev.state.is_pressed() { continue; }
        match ev.key_code {
            KeyCode::Enter | KeyCode::NumpadEnter => {
                let buf = session.rename_buffer.trim().to_string();
                if !buf.is_empty() {
                    session.body_parts[i].name = buf;
                    session.dirty = true;
                }
                session.renaming_body_part = None;
                session.rename_buffer.clear();
            }
            KeyCode::Escape => {
                session.renaming_body_part = None;
                session.rename_buffer.clear();
            }
            KeyCode::Backspace => { session.rename_buffer.pop(); }
            _ => {
                if let Some(text) = ev.text.as_ref() {
                    for c in text.chars() {
                        if session.rename_buffer.len() >= RENAME_MAX_LEN { break; }
                        // Printable, non-control characters only.
                        if !c.is_control() {
                            session.rename_buffer.push(c);
                        }
                    }
                }
            }
        }
    }
}


// ── Row label + highlight sync ─────────────────────────────────────────────────

#[allow(clippy::type_complexity)]
pub fn sync_body_part_rows(
    session:      Res<SpeciesSession>,
    mut labels:   Query<(&BodyPartRowLabel, &mut Text), Without<BodyPartKindLabel>>,
    mut selects:  Query<(&BodyPartSelectButton, &Interaction, &mut BackgroundColor), Without<BodyPartKindButton>>,
    mut kind_btns: Query<(&BodyPartKindButton, &Interaction, &mut BackgroundColor), Without<BodyPartSelectButton>>,
    mut kind_lbls: Query<(&BodyPartKindLabel, &mut Text), Without<BodyPartRowLabel>>,
    mut options:  Query<(&KindDropdownOption, &Interaction, &mut BackgroundColor),
                        (Without<BodyPartSelectButton>, Without<BodyPartKindButton>)>,
) {
    // Labels (renaming overlays show the live buffer with a cursor).
    for (label, mut text) in &mut labels {
        let i = label.0;
        let Some(part) = session.body_parts.get(i) else { continue };
        let new = if session.renaming_body_part == Some(i) {
            format!("{}_", session.rename_buffer)
        } else if part.parent != 0 {
            // Show what this part hangs off, so the limb→sub-limb hierarchy
            // (and what "Begin New Body-Part" will attach to) is visible.
            let parent_name = session.body_parts.get(part.parent)
                .map(|p| p.name.as_str()).unwrap_or("?");
            format!("{}  ↳ {}", part.name, parent_name)
        } else {
            part.name.clone()
        };
        if text.0 != new { text.0 = new; }
    }

    // Select-button background: rename > active > hover > idle.
    for (sel, interaction, mut bg) in &mut selects {
        let i = sel.0;
        let target = if session.renaming_body_part == Some(i) {
            ROW_BG_RENAME
        } else if session.active_body_part == i {
            ROW_BG_ACTIVE
        } else if matches!(*interaction, Interaction::Hovered | Interaction::Pressed) {
            ROW_BG_HOVER
        } else {
            ROW_BG_IDLE
        };
        *bg = BackgroundColor(target);
    }

    // Kind-button background (hover variant).
    for (btn, interaction, mut bg) in &mut kind_btns {
        let _ = btn;
        let hovered = matches!(*interaction, Interaction::Hovered | Interaction::Pressed);
        *bg = BackgroundColor(if hovered { KIND_BG_HOVER } else { KIND_BG });
    }
    // Kind-button label reflects the part's current kind.
    for (lbl, mut text) in &mut kind_lbls {
        let Some(part) = session.body_parts.get(lbl.0) else { continue };
        let new = kind_label(part.kind);
        if text.0 != new { text.0 = new.to_string(); }
    }
    // Open dropdown option backgrounds: selected kind highlighted, hover variant.
    for (opt, interaction, mut bg) in &mut options {
        let selected = session.body_parts.get(opt.part).map(|p| p.kind) == Some(opt.kind);
        let hovered = matches!(*interaction, Interaction::Hovered | Interaction::Pressed);
        let target = if selected { KIND_OPT_BG_SEL }
                     else if hovered { KIND_OPT_BG_HOVER }
                     else { KIND_OPT_BG };
        *bg = BackgroundColor(target);
    }
}
