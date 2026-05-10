// Bottom UI strip — organism-creation cyclers.
//
// Three cycler buttons (Metabolism / Intelligence / Symmetry) plus a
// hint label. Click a cycler to advance through its enum's variants;
// the snapshot lives in `EditorSession::draft` and is consumed by the
// left-click placement system in `placement.rs`. There is no
// "Create" button — left-clicking on the map IS the create action.

use bevy::prelude::*;

use crate::organism::Symmetry;
use crate::colony_editor::session::{EditorSession, DraftOrganism};
use crate::colony_editor::template::{
    Form, form_cycle, form_label, intel_cycle, intel_label, sym_cycle, sym_label,
};
use crate::colony_editor::layout::PANEL_BG_COLOR;


// ── Tunables ─────────────────────────────────────────────────────────────────

pub const BOTTOM_PANEL_HEIGHT_PX: f32 = 90.0;
const ROW_PADDING_PX:    f32 = 12.0;
const FIELD_GAP_PX:      f32 = 12.0;
/// Slightly narrower than the original 200 px so all four cyclers
/// fit comfortably in the bottom panel alongside the right-side
/// hint label.
const CYCLER_WIDTH_PX:   f32 = 180.0;
const CYCLER_HEIGHT_PX:  f32 = 36.0;
const LABEL_FONT_SIZE:   f32 = 12.0;
const VALUE_FONT_SIZE:   f32 = 15.0;

const CYCLER_BG_COLOR:     Color = Color::srgb(0.22, 0.22, 0.22);
const CYCLER_HOVER_COLOR:  Color = Color::srgb(0.32, 0.32, 0.32);
const HINT_COLOR:          Color = Color::srgb(0.65, 0.70, 0.55);


// ── Marker components ───────────────────────────────────────────────────────

#[derive(Component)]
pub struct CreationPanel;

#[derive(Component, Clone, Copy)]
enum CyclerKind { Metabolism, Intelligence, Symmetry, Form }

#[derive(Component)]
struct Cycler { kind: CyclerKind }

#[derive(Component)]
struct CyclerValueText { kind: CyclerKind }


// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct CreationPanelPlugin;

impl Plugin for CreationPanelPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (
            handle_cycler_clicks,
            sync_cycler_text,
        ));
    }
}


// ── Spawning ────────────────────────────────────────────────────────────────

pub fn spawn(parent: &mut ChildSpawnerCommands, draft: DraftOrganism) {
    parent.spawn((
        CreationPanel,
        Node {
            position_type: PositionType::Absolute,
            left:   Val::Px(0.0),
            right:  Val::Px(0.0),
            bottom: Val::Px(0.0),
            height: Val::Px(BOTTOM_PANEL_HEIGHT_PX),
            flex_direction: FlexDirection::Row,
            align_items:    AlignItems::Center,
            padding:        UiRect::all(Val::Px(ROW_PADDING_PX)),
            ..default()
        },
        BackgroundColor(PANEL_BG_COLOR),
    ))
    .with_children(|panel| {
        cycler_field(panel, "Metabolism",   draft.metabolism.label(),        CyclerKind::Metabolism);
        cycler_field(panel, "Intelligence", intel_label(draft.intelligence), CyclerKind::Intelligence);
        cycler_field(panel, "Symmetry",     sym_label(draft.symmetry),       CyclerKind::Symmetry);
        cycler_field(panel, "Form",         form_label(draft.form),          CyclerKind::Form);

        // Spacer pushes the hint label to the right edge.
        panel.spawn(Node { flex_grow: 1.0, ..default() });

        // Hint reminding the user how to create. Replaces the old
        // "Create" button — placement is now driven by left-click
        // on the map surface.
        panel
            .spawn(Node {
                flex_direction: FlexDirection::Column,
                align_items:    AlignItems::FlexEnd,
                justify_content: JustifyContent::Center,
                ..default()
            })
            .with_children(|hint| {
                hint.spawn((
                    Text::new("Left-click on the map to place"),
                    TextFont { font_size: 14.0, ..default() },
                    TextColor(HINT_COLOR),
                    Pickable::IGNORE,
                ));
                hint.spawn((
                    Text::new("Right-click an organism to delete it"),
                    TextFont { font_size: 12.0, ..default() },
                    TextColor(Color::srgb(0.55, 0.55, 0.55)),
                    Pickable::IGNORE,
                ));
            });
    });
}

fn cycler_field(parent: &mut ChildSpawnerCommands, label: &str, value: &str, kind: CyclerKind) {
    parent
        .spawn(Node {
            flex_direction: FlexDirection::Column,
            justify_content: JustifyContent::Center,
            align_items:     AlignItems::FlexStart,
            margin:          UiRect::right(Val::Px(FIELD_GAP_PX)),
            ..default()
        })
        .with_children(|col| {
            col.spawn((
                Text::new(label.to_string()),
                TextFont { font_size: LABEL_FONT_SIZE, ..default() },
                TextColor(Color::srgb(0.75, 0.75, 0.75)),
                Node { margin: UiRect::bottom(Val::Px(2.0)), ..default() },
                Pickable::IGNORE,
            ));
            col.spawn((
                Cycler { kind },
                Button,
                Node {
                    width:  Val::Px(CYCLER_WIDTH_PX),
                    height: Val::Px(CYCLER_HEIGHT_PX),
                    align_items:     AlignItems::Center,
                    justify_content: JustifyContent::Center,
                    ..default()
                },
                BackgroundColor(CYCLER_BG_COLOR),
            ))
            .with_children(|c| {
                c.spawn((
                    CyclerValueText { kind },
                    Text::new(value.to_string()),
                    TextFont { font_size: VALUE_FONT_SIZE, ..default() },
                    TextColor(Color::WHITE),
                    Pickable::IGNORE,
                ));
            });
        });
}


// ── Systems ──────────────────────────────────────────────────────────────────

fn handle_cycler_clicks(
    mut interactions: Query<(&Interaction, &Cycler, &mut BackgroundColor), Changed<Interaction>>,
    mut session:      ResMut<EditorSession>,
) {
    for (interaction, cycler, mut bg) in &mut interactions {
        match *interaction {
            Interaction::Pressed => {
                match cycler.kind {
                    CyclerKind::Metabolism => {
                        session.draft.metabolism = session.draft.metabolism.cycle();
                    }
                    CyclerKind::Intelligence => {
                        session.draft.intelligence = intel_cycle(session.draft.intelligence);
                    }
                    CyclerKind::Symmetry => {
                        let new_sym = sym_cycle(session.draft.symmetry);
                        session.draft.symmetry = new_sym;
                        // Simulation invariant: `has_variable_form ⇒
                        // NoSymmetry`. Cycling to Bilateral while in
                        // Variable form would violate it, so we drop
                        // the form to Fixed in that case.
                        if matches!(new_sym, Symmetry::Bilateral)
                            && session.draft.form.is_variable()
                        {
                            session.draft.form = Form::Fixed;
                        }
                    }
                    CyclerKind::Form => {
                        let new_form = form_cycle(session.draft.form);
                        session.draft.form = new_form;
                        // Other half of the invariant: cycling to
                        // Variable while symmetry is Bilateral
                        // would violate it. Fall back to NoSymmetry.
                        if new_form.is_variable()
                            && matches!(session.draft.symmetry, Symmetry::Bilateral)
                        {
                            session.draft.symmetry = Symmetry::NoSymmetry;
                        }
                    }
                }
                *bg = BackgroundColor(CYCLER_HOVER_COLOR);
            }
            Interaction::Hovered => *bg = BackgroundColor(CYCLER_HOVER_COLOR),
            Interaction::None    => *bg = BackgroundColor(CYCLER_BG_COLOR),
        }
    }
}

fn sync_cycler_text(
    session:    Res<EditorSession>,
    mut texts:  Query<(&CyclerValueText, &mut Text)>,
) {
    if !session.is_changed() { return; }
    for (marker, mut text) in &mut texts {
        let s = match marker.kind {
            CyclerKind::Metabolism   => session.draft.metabolism.label().to_string(),
            CyclerKind::Intelligence => intel_label(session.draft.intelligence).to_string(),
            CyclerKind::Symmetry     => sym_label(session.draft.symmetry).to_string(),
            CyclerKind::Form         => form_label(session.draft.form).to_string(),
        };
        text.0 = s;
    }
}
