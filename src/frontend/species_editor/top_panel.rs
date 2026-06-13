// Species editor — top panel: metabolism/intelligence/symmetry/form/movement
// cyclers + Import/Load/Save actions.
//
// Cyclers stay editable after cells exist (so a loaded species can be
// re-tuned), except Symmetry, which locks once cells exist because it
// changes how the stored OCG is interpreted.

use bevy::prelude::*;
use std::path::PathBuf;

use crate::cell::CellType;
use crate::colony::{IntelligenceLevel, Symmetry};
use crate::organism::MovementMode;
use crate::frontend::PANEL_BG_COLOR;
use crate::simulation_settings::WindowMode;

use super::session::{
    cycle_intelligence, cycle_symmetry, intelligence_label, symmetry_label,
    Classification, DraftSpecies, EditorBodyPart, Form, Metabolism, Mobility,
    SpeciesSession,
};
use super::save::{load_species, LoadedSpecies};
use super::mesh_import::{ImportMeshButton, IMPORT_BG};
use super::TOP_PANEL_HEIGHT_PX;


// ── Colors ───────────────────────────────────────────────────────────────────

const CYCLER_BG_ACTIVE:    Color = Color::srgb(0.28, 0.28, 0.28);
const CYCLER_BG_LOCKED:    Color = Color::srgb(0.16, 0.16, 0.16);
const CYCLER_BG_HOVER:     Color = Color::srgb(0.38, 0.38, 0.38);
const ACTION_BG_SUCCESS:   Color = Color::srgb(0.20, 0.55, 0.25);
const ACTION_BG_HOVER:     Color = Color::srgb(0.30, 0.55, 0.80);
const ACTION_BG_DISABLED:  Color = Color::srgb(0.18, 0.18, 0.18);
const ACTION_BG_LOAD:      Color = Color::srgb(0.45, 0.35, 0.62); // purple — "Load"


// ── Markers ──────────────────────────────────────────────────────────────────

#[derive(Component)]
pub struct SpeciesTopPanel;

#[derive(Component, Clone, Copy, PartialEq, Eq)]
pub enum CyclerKind { Metabolism, Mobility, Classification, Intelligence, Symmetry, Form, Movement, Grounding }

#[derive(Component, Clone, Copy)]
pub struct Cycler(pub CyclerKind);

/// Text-child marker so update systems can find a cycler's display
/// label without traversing every child.
#[derive(Component, Clone, Copy)]
pub struct CyclerValueText(pub CyclerKind);

#[derive(Component)]
pub struct CreateSpeciesButton;

#[derive(Component)]
pub struct LoadSpeciesButton;

// ── Load-confirmation modal markers ─────────────────────────────────────────
#[derive(Component)]
pub struct LoadModalRoot;
#[derive(Component)]
pub struct LoadModalYesButton;
#[derive(Component)]
pub struct LoadModalNoButton;


// ── Spawn ────────────────────────────────────────────────────────────────────

pub fn spawn_top_panel(parent: &mut ChildSpawnerCommands, top_offset_px: f32) {
    parent
        .spawn((
            SpeciesTopPanel,
            super::SpeciesEditorPanel,
            Node {
                position_type:   PositionType::Absolute,
                top:             Val::Px(top_offset_px),
                left:            Val::Px(0.0),
                right:           Val::Px(0.0),
                height:          Val::Px(TOP_PANEL_HEIGHT_PX),
                padding:         UiRect::all(Val::Px(6.0)),
                flex_direction:  FlexDirection::Row,
                align_items:     AlignItems::Center,
                justify_content: JustifyContent::SpaceEvenly,
                display:         Display::None,   // shown only in SpeciesEditor mode
                ..default()
            },
            BackgroundColor(PANEL_BG_COLOR),
        ))
        .with_children(|bar| {
            let draft = DraftSpecies::default();
            cycler_button(bar, CyclerKind::Metabolism,     draft.metabolism.label());
            cycler_button(bar, CyclerKind::Mobility,       draft.mobility.label());
            cycler_button(bar, CyclerKind::Classification, draft.classification.label());
            cycler_button(bar, CyclerKind::Intelligence,   intelligence_label(draft.intelligence));
            cycler_button(bar, CyclerKind::Symmetry,       symmetry_label(draft.symmetry));
            cycler_button(bar, CyclerKind::Form,           draft.form.label());
            cycler_button(bar, CyclerKind::Movement,       draft.movement.label());
            cycler_button(bar, CyclerKind::Grounding,      grounding_label(draft.ground_based));

            // The base cell is auto-seeded from the metabolism cycler.
            action_button::<ImportMeshButton>(bar, "Import Mesh (.glb)", IMPORT_BG);
            action_button::<LoadSpeciesButton>(bar, "Load Species",   ACTION_BG_LOAD);
            action_button::<CreateSpeciesButton>(bar, "Save Species", ACTION_BG_SUCCESS);
        });
}

fn cycler_button(parent: &mut ChildSpawnerCommands, kind: CyclerKind, initial_label: &str) {
    parent
        .spawn((
            Cycler(kind),
            Button,
            Node {
                flex_grow:       0.0,
                width:           Val::Px(150.0),
                height:          Val::Px(36.0),
                margin:          UiRect::horizontal(Val::Px(4.0)),
                align_items:     AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            BackgroundColor(CYCLER_BG_ACTIVE),
        ))
        .with_children(|btn| {
            btn.spawn((
                CyclerValueText(kind),
                Text::new(initial_label.to_string()),
                TextFont { font_size: 14.0, ..default() },
                TextColor(Color::WHITE),
                Pickable::IGNORE,
            ));
        });
}

fn action_button<M: Component + Default>(parent: &mut ChildSpawnerCommands, label: &str, bg: Color) {
    parent
        .spawn((
            M::default(),
            Button,
            Node {
                flex_grow:       0.0,
                width:           Val::Px(160.0),
                height:          Val::Px(36.0),
                margin:          UiRect::horizontal(Val::Px(4.0)),
                align_items:     AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            BackgroundColor(bg),
        ))
        .with_children(|btn| {
            btn.spawn((
                Text::new(label.to_string()),
                TextFont { font_size: 14.0, ..default() },
                TextColor(Color::WHITE),
                Pickable::IGNORE,
            ));
        });
}

impl Default for CreateSpeciesButton   { fn default() -> Self { Self } }
impl Default for LoadSpeciesButton     { fn default() -> Self { Self } }


// ── Cycler click handler ────────────────────────────────────────────────────

/// Canonical starting intelligence for a (mobility, classification) pair,
/// so toggling either knob snaps intelligence to a valid value.
fn default_intelligence_for(m: Mobility, c: Classification) -> IntelligenceLevel {
    match (m, c) {
        (Mobility::Sessile,  _)                       => IntelligenceLevel::Level0,
        (Mobility::Mobile, Classification::Herbivore) => IntelligenceLevel::Level1,
        (Mobility::Mobile, Classification::Carnivore) => IntelligenceLevel::Level2,
    }
}

/// Cycle through the THREE author-able movement modes in order
/// Sliding → LimbBasedWalking → Swimming → (back to Sliding). `Flying` is a
/// placeholder and deliberately excluded; any non-author-able value snaps
/// back to `Sliding`.
fn cycle_movement(m: MovementMode) -> MovementMode {
    match m {
        MovementMode::Sliding          => MovementMode::LimbBasedWalking,
        MovementMode::LimbBasedWalking => MovementMode::Swimming,
        MovementMode::Swimming         => MovementMode::Sliding,
        MovementMode::Flying           => MovementMode::Sliding,
    }
}

/// Display label for the Grounding cycler (`DraftSpecies::ground_based`).
fn grounding_label(ground_based: bool) -> &'static str {
    if ground_based { "Ground-Based" } else { "Water-Based" }
}

/// Canonical default symmetry for a metabolism: heterotrophs Bilateral
/// (animal-like), photoautotrophs NoSymmetry (plant-like). Re-cyclable.
fn default_symmetry_for(m: Metabolism) -> Symmetry {
    match m {
        Metabolism::Heterotroph    => Symmetry::Bilateral,
        Metabolism::Photoautotroph => Symmetry::NoSymmetry,
    }
}

pub fn handle_cycler_clicks(
    mode:             Res<WindowMode>,
    mut interactions: Query<(&Interaction, &Cycler, &mut BackgroundColor), Changed<Interaction>>,
    mut session:      ResMut<SpeciesSession>,
) {
    if *mode != WindowMode::SpeciesEditor { return; }

    for (interaction, cycler, mut bg) in &mut interactions {
        // Per-cycler lock (must mirror `sync_cycler_lock_state`):
        //   * Metabolism + Symmetry lock once cells are appended beyond the
        //     base seed — both reinterpret the base in ways incompatible with
        //     an already-built body.
        //   * Intelligence + Movement lock while Sessile (sessile ⇒ Level0 +
        //     Sliding).
        //   * Grounding locks for everything EXCEPT phototrophs — only a
        //     phototroph may be made water-based (floating algae); every
        //     other organism derives `ground_based` from its movement mode.
        // Locked cyclers `continue` to keep the greyed colour.
        let locked = match cycler.0 {
            CyclerKind::Metabolism | CyclerKind::Symmetry => session.has_appended_cells(),
            CyclerKind::Intelligence | CyclerKind::Movement => {
                session.draft.mobility == Mobility::Sessile
            }
            CyclerKind::Grounding => {
                session.draft.metabolism != Metabolism::Photoautotroph
            }
            _ => false,
        };
        if locked { continue; }

        match *interaction {
            Interaction::Pressed => {
                match cycler.0 {
                    CyclerKind::Metabolism => {
                        // Toggle metabolism, snap symmetry to its canonical
                        // default, and re-seed the base so colour/position update.
                        session.draft.metabolism = session.draft.metabolism.cycle();
                        session.draft.symmetry = default_symmetry_for(session.draft.metabolism);
                        // Grounding is a phototroph-only override: leaving
                        // Photoautotroph snaps it back to the movement-mode
                        // default so the (now greyed-out) cycler shows the
                        // value the spawn will actually use.
                        if session.draft.metabolism != Metabolism::Photoautotroph {
                            session.draft.ground_based =
                                session.draft.movement.default_ground_based();
                        }
                        session.seed_base();
                    }
                    CyclerKind::Mobility => {
                        // Enforce Mobility ↔ Intelligence coupling.
                        session.draft.mobility = session.draft.mobility.cycle();
                        session.draft.intelligence = default_intelligence_for(
                            session.draft.mobility, session.draft.classification,
                        );
                        // Sessile must be on the sliding path (Kinematic root);
                        // snap the movement cycler so the UI reflects what
                        // `spawn_organism` will coerce anyway.
                        if session.draft.mobility == Mobility::Sessile {
                            session.draft.movement = MovementMode::Sliding;
                        }
                    }
                    CyclerKind::Classification => {
                        // Re-apply the intelligence default (Carnivore needs ≥ Level2).
                        session.draft.classification = session.draft.classification.cycle();
                        session.draft.intelligence = default_intelligence_for(
                            session.draft.mobility, session.draft.classification,
                        );
                    }
                    CyclerKind::Intelligence => {
                        // Sessile locked at Level0; herbivore cycles {L1,L2,L3},
                        // carnivore cycles {L2,L3}.
                        if session.draft.mobility == Mobility::Sessile { continue; }
                        let next = cycle_intelligence(session.draft.intelligence);
                        // Skip Level0 (sessile-only) and, for carnivores, Level1.
                        let mut candidate = next;
                        for _ in 0..4 {
                            if candidate == IntelligenceLevel::Level0
                               || (session.draft.classification == Classification::Carnivore
                                   && candidate == IntelligenceLevel::Level1)
                            {
                                candidate = cycle_intelligence(candidate);
                            } else { break; }
                        }
                        session.draft.intelligence = candidate;
                    }
                    CyclerKind::Symmetry => {
                        // Re-seed the base at the correct position for the new symmetry.
                        session.draft.symmetry = cycle_symmetry(session.draft.symmetry);
                        session.seed_base();
                    }
                    CyclerKind::Form     => session.draft.form     = session.draft.form.cycle(),
                    CyclerKind::Movement => {
                        // Sessile is locked to Sliding (Kinematic root).
                        if session.draft.mobility == Mobility::Sessile { continue; }
                        // 3-way cycle: Sliding → LimbBasedWalking → Swimming.
                        session.draft.movement = cycle_movement(session.draft.movement);
                        // Non-phototrophs derive Grounding from the movement
                        // mode — keep the greyed-out cycler label honest.
                        if session.draft.metabolism != Metabolism::Photoautotroph {
                            session.draft.ground_based =
                                session.draft.movement.default_ground_based();
                        }
                    }
                    CyclerKind::Grounding => {
                        // Phototroph-only (locked otherwise, see above):
                        // toggle Ground-Based ↔ Water-Based (floating algae).
                        session.draft.ground_based = !session.draft.ground_based;
                    }
                }
                // Mark unsaved so Load/Clear confirm before discarding.
                session.dirty = true;
                *bg = BackgroundColor(CYCLER_BG_HOVER);
            }
            Interaction::Hovered => *bg = BackgroundColor(CYCLER_BG_HOVER),
            Interaction::None    => *bg = BackgroundColor(CYCLER_BG_ACTIVE),
        }
    }
}

/// Sync cycler labels with `session.draft`. Edge-triggered on session change.
pub fn sync_cycler_labels(
    session:   Res<SpeciesSession>,
    mut texts: Query<(&CyclerValueText, &mut Text)>,
) {
    if !session.is_changed() { return; }
    for (marker, mut text) in &mut texts {
        let new = match marker.0 {
            CyclerKind::Metabolism     => session.draft.metabolism.label().to_string(),
            CyclerKind::Mobility       => session.draft.mobility.label().to_string(),
            CyclerKind::Classification => session.draft.classification.label().to_string(),
            CyclerKind::Intelligence   => intelligence_label(session.draft.intelligence).to_string(),
            CyclerKind::Symmetry       => symmetry_label(session.draft.symmetry).to_string(),
            CyclerKind::Form           => session.draft.form.label().to_string(),
            CyclerKind::Movement       => session.draft.movement.label().to_string(),
            CyclerKind::Grounding      => grounding_label(session.draft.ground_based).to_string(),
        };
        if text.0 != new { text.0 = new; }
    }
}

/// Grey-out locked cyclers (appended cells → Metabolism+Symmetry; sessile →
/// Intelligence+Movement). Mirrors the lock in `handle_cycler_clicks`.
pub fn sync_cycler_lock_state(
    session:     Res<SpeciesSession>,
    mut cyclers: Query<(&Cycler, &mut BackgroundColor)>,
) {
    if !session.is_changed() { return; }
    let appended  = session.has_appended_cells();
    let sessile   = session.draft.mobility == Mobility::Sessile;
    let not_photo = session.draft.metabolism != Metabolism::Photoautotroph;
    for (cycler, mut bg) in &mut cyclers {
        let locked = match cycler.0 {
            CyclerKind::Metabolism | CyclerKind::Symmetry => appended,
            CyclerKind::Intelligence | CyclerKind::Movement => sessile,
            // Grounding is editable ONLY while constructing a phototroph.
            CyclerKind::Grounding => not_photo,
            _ => false,
        };
        *bg = BackgroundColor(if locked { CYCLER_BG_LOCKED } else { CYCLER_BG_ACTIVE });
    }
}


// ── Create Species handler ──────────────────────────────────────────────────

pub fn handle_create_species(
    mode:             Res<WindowMode>,
    mut interactions: Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<CreateSpeciesButton>)>,
    mut session:      ResMut<SpeciesSession>,
) {
    if *mode != WindowMode::SpeciesEditor { return; }

    for (interaction, mut bg) in &mut interactions {
        let can_save = session.base_part().is_some_and(|p| !p.ocg.is_empty());
        match *interaction {
            Interaction::Pressed if can_save => {
                let _ = CellType::Photo; // (silence unused-import lint if any)
                let initial_dir = std::env::current_dir()
                    .unwrap_or_else(|_| PathBuf::from("."));
                let default_name = format!(
                    "species_{}.species",
                    chrono::Local::now().format("%d-%m-%Y-%H-%M-%S"),
                );
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("AEONS species (.species)", &["species"])
                    .set_directory(initial_dir)
                    .set_file_name(default_name)
                    .save_file()
                {
                    session.save_requested = Some(path);
                }
                *bg = BackgroundColor(ACTION_BG_HOVER);
            }
            Interaction::Pressed => *bg = BackgroundColor(ACTION_BG_DISABLED),
            Interaction::Hovered if can_save => *bg = BackgroundColor(ACTION_BG_HOVER),
            _ => {
                *bg = BackgroundColor(if can_save { ACTION_BG_SUCCESS } else { ACTION_BG_DISABLED });
            }
        }
    }
}


// ── Load Species ──────────────────────────────────────────────────────────

/// Replace the live session with a decoded `.species`. The `ResMut` change
/// drives the mesh/cycler/lock refreshers, so the editor re-renders. Brain
/// weights are dropped (the editor works on morphology + options).
fn apply_loaded_species(session: &mut SpeciesSession, loaded: LoadedSpecies) {
    session.reset();
    session.draft = DraftSpecies {
        metabolism:     loaded.metabolism,
        intelligence:   loaded.intelligence,
        symmetry:       loaded.symmetry,
        form:           if loaded.has_variable_form { Form::Variable } else { Form::Fixed },
        mobility:       if loaded.is_sessile { Mobility::Sessile } else { Mobility::Mobile },
        classification: loaded.classification,
        movement:       loaded.movement,
        ground_based:   loaded.ground_based,
    };
    session.body_parts = loaded.body_parts.into_iter()
        .map(|p| EditorBodyPart { name: p.name, ocg: p.ocg, is_limb: p.is_limb, parent: p.parent })
        .collect();
    session.active_body_part = 0;
    // A loaded body has appended cells → Metabolism + Symmetry lock.
    session.first_cell_spawned = !session.body_parts.is_empty();
    // Freshly loaded state matches disk → not dirty.
    session.dirty = false;
}

/// Open a native file dialog, decode the chosen `.species`, and load it.
/// Shared by the direct (non-dirty) path and the modal's "Yes".
fn do_load(session: &mut SpeciesSession) {
    let initial_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    if let Some(path) = rfd::FileDialog::new()
        .add_filter("AEONS species (.species)", &["species"])
        .set_directory(initial_dir)
        .pick_file()
    {
        match load_species(&path) {
            Ok(loaded) => {
                info!("species editor: loaded species from {}", path.display());
                apply_loaded_species(session, loaded);
            }
            Err(e) => error!("species editor: failed to load {}: {}", path.display(), e),
        }
    }
}

/// "Load Species" button. If there are unsaved changes, raise the confirmation
/// modal first (the load happens on "Yes"); otherwise load directly.
pub fn handle_load_species(
    mode:             Res<WindowMode>,
    mut interactions: Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<LoadSpeciesButton>)>,
    mut session:      ResMut<SpeciesSession>,
) {
    if *mode != WindowMode::SpeciesEditor { return; }

    for (interaction, mut bg) in &mut interactions {
        match *interaction {
            Interaction::Pressed => {
                if session.dirty {
                    session.show_load_modal = true;
                } else {
                    do_load(&mut session);
                }
                *bg = BackgroundColor(ACTION_BG_HOVER);
            }
            Interaction::Hovered => *bg = BackgroundColor(ACTION_BG_HOVER),
            Interaction::None    => *bg = BackgroundColor(ACTION_BG_LOAD),
        }
    }
}


// ── Load confirmation modal (unsaved changes) ───────────────────────────────

const MODAL_BACKDROP:    Color = Color::srgba(0.0, 0.0, 0.0, 0.55);
const MODAL_CARD:        Color = Color::srgb(0.15, 0.15, 0.18);
const MODAL_CARD_BORDER: Color = Color::srgb(0.40, 0.40, 0.45);
const MODAL_YES:         Color = Color::srgb(0.55, 0.18, 0.18); // muted red — discards changes
const MODAL_YES_HOVER:   Color = Color::srgb(0.68, 0.22, 0.22);
const MODAL_NO:          Color = Color::srgb(0.24, 0.56, 0.36); // green — safe default
const MODAL_NO_HOVER:    Color = Color::srgb(0.32, 0.66, 0.42);
const MODAL_NO_BORDER:   Color = Color::srgb(0.95, 0.95, 0.95);

/// Spawn / despawn the modal in lock-step with `show_load_modal`.
pub fn manage_load_modal_visibility(
    mut commands: Commands,
    session:      Res<SpeciesSession>,
    existing:     Query<Entity, With<LoadModalRoot>>,
) {
    let want = session.show_load_modal;
    let is   = !existing.is_empty();
    if want && !is {
        spawn_load_modal(&mut commands);
    } else if !want && is {
        for e in &existing { commands.entity(e).despawn(); }
    }
}

fn spawn_load_modal(commands: &mut Commands) {
    commands
        .spawn((
            LoadModalRoot,
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(0.0), left: Val::Px(0.0),
                width: Val::Percent(100.0), height: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                align_items:     AlignItems::Center,
                ..default()
            },
            BackgroundColor(MODAL_BACKDROP),
            GlobalZIndex(120),
        ))
        .with_children(|root| {
            root.spawn((
                Node {
                    width: Val::Px(560.0),
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Center,
                    padding: UiRect::all(Val::Px(22.0)),
                    border:  UiRect::all(Val::Px(1.0)),
                    ..default()
                },
                BackgroundColor(MODAL_CARD),
                BorderColor::all(MODAL_CARD_BORDER),
            ))
            .with_children(|card| {
                card.spawn((
                    Text::new("There have been changes since the last save. Are you sure?"),
                    TextFont { font_size: 15.0, ..default() },
                    TextColor(Color::srgb(0.9, 0.9, 0.9)),
                    Node { margin: UiRect::bottom(Val::Px(20.0)), ..default() },
                    Pickable::IGNORE,
                ));
                card.spawn(Node {
                    flex_direction:  FlexDirection::Row,
                    justify_content: JustifyContent::Center,
                    align_items:     AlignItems::Center,
                    ..default()
                })
                .with_children(|row| {
                    // "No" first (left), the safe default.
                    row.spawn((
                        LoadModalNoButton,
                        Button,
                        Node {
                            width: Val::Px(110.0), height: Val::Px(36.0),
                            margin: UiRect::right(Val::Px(16.0)),
                            border: UiRect::all(Val::Px(2.0)),
                            align_items: AlignItems::Center,
                            justify_content: JustifyContent::Center,
                            ..default()
                        },
                        BackgroundColor(MODAL_NO),
                        BorderColor::all(MODAL_NO_BORDER),
                    ))
                    .with_children(|b| { b.spawn((
                        Text::new("No"),
                        TextFont { font_size: 16.0, ..default() },
                        TextColor(Color::WHITE), Pickable::IGNORE,
                    )); });
                    // "Yes" — continue loading (discards unsaved changes).
                    row.spawn((
                        LoadModalYesButton,
                        Button,
                        Node {
                            width: Val::Px(110.0), height: Val::Px(36.0),
                            align_items: AlignItems::Center,
                            justify_content: JustifyContent::Center,
                            ..default()
                        },
                        BackgroundColor(MODAL_YES),
                    ))
                    .with_children(|b| { b.spawn((
                        Text::new("Yes"),
                        TextFont { font_size: 16.0, ..default() },
                        TextColor(Color::WHITE), Pickable::IGNORE,
                    )); });
                });
            });
        });
}

/// Modal Yes/No. Yes → drop the flag and continue the load (file dialog +
/// decode). No → just drop the flag.
pub fn handle_load_modal_buttons(
    mut yes_q:   Query<(&Interaction, &mut BackgroundColor),
                       (Changed<Interaction>, With<LoadModalYesButton>, Without<LoadModalNoButton>)>,
    mut no_q:    Query<(&Interaction, &mut BackgroundColor),
                       (Changed<Interaction>, With<LoadModalNoButton>, Without<LoadModalYesButton>)>,
    mut session: ResMut<SpeciesSession>,
) {
    for (interaction, mut bg) in &mut yes_q {
        match *interaction {
            Interaction::Pressed => {
                session.show_load_modal = false;
                do_load(&mut session);
                *bg = BackgroundColor(MODAL_YES_HOVER);
            }
            Interaction::Hovered => *bg = BackgroundColor(MODAL_YES_HOVER),
            Interaction::None    => *bg = BackgroundColor(MODAL_YES),
        }
    }
    for (interaction, mut bg) in &mut no_q {
        match *interaction {
            Interaction::Pressed => {
                session.show_load_modal = false;
                *bg = BackgroundColor(MODAL_NO_HOVER);
            }
            Interaction::Hovered => *bg = BackgroundColor(MODAL_NO_HOVER),
            Interaction::None    => *bg = BackgroundColor(MODAL_NO),
        }
    }
}
