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
    Classification, DraftSpecies, EditorBodyPart, Form, Grounding, Metabolism, Mobility,
    SpeciesSession,
};
use super::save::{load_species, LoadedSpecies};
use super::mesh_import::{ImportMeshButton, IMPORT_BG};
use super::TOP_PANEL_HEIGHT_PX;
use crate::ui_modal::{
    self, ConfirmModalSpec, NoButtonStyle, YES_BTN_COLOR, YES_BTN_HOVER,
};


// ── Colors ───────────────────────────────────────────────────────────────────

// Cycler-card palette (slate cards on the dark panel). The handler systems
// reference these three by name, so changing the VALUES restyles the cards
// without touching any handler logic.
const CYCLER_BG_ACTIVE:    Color = Color::srgb(0.21, 0.22, 0.26);
const CYCLER_BG_LOCKED:    Color = Color::srgb(0.13, 0.13, 0.15);
const CYCLER_BG_HOVER:     Color = Color::srgb(0.30, 0.33, 0.40);
const ACTION_BG_SUCCESS:   Color = Color::srgb(0.20, 0.55, 0.25);
const ACTION_BG_HOVER:     Color = Color::srgb(0.30, 0.55, 0.80);
const ACTION_BG_DISABLED:  Color = Color::srgb(0.18, 0.18, 0.18);
const ACTION_BG_LOAD:      Color = Color::srgb(0.45, 0.35, 0.62); // purple — "Load"
// Captioned-card text + section-header accents.
const CAPTION_COLOR:       Color = Color::srgb(0.56, 0.61, 0.72); // property-name caption
const SECTION_LABEL_COLOR: Color = Color::srgb(0.45, 0.74, 0.80); // teal group header
const CARD_RADIUS:         Val   = Val::Px(5.0);


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
                padding:         UiRect::axes(Val::Px(12.0), Val::Px(6.0)),
                flex_direction:  FlexDirection::Row,
                align_items:     AlignItems::Center,
                // Attribute groups packed left, File actions pushed right.
                justify_content: JustifyContent::SpaceBetween,
                column_gap:      Val::Px(10.0),
                display:         Display::None,   // shown only in SpeciesEditor mode
                ..default()
            },
            BackgroundColor(PANEL_BG_COLOR),
        ))
        .with_children(|bar| {
            let draft = DraftSpecies::default();
            // Left: the species-attribute cyclers, grouped by concept so each
            // control reads as "<PROPERTY>: <value>" under a section header.
            bar.spawn((Node {
                flex_direction: FlexDirection::Row,
                align_items:    AlignItems::Center,
                column_gap:     Val::Px(4.0),
                ..default()
            },))
            .with_children(|attrs| {
                cycler_group(attrs, "IDENTITY", |g| {
                    cycler_card(g, CyclerKind::Metabolism,     "METABOLISM", draft.metabolism.label());
                    cycler_card(g, CyclerKind::Classification, "DIET",       draft.classification.label());
                });
                cycler_group(attrs, "BODY PLAN", |g| {
                    cycler_card(g, CyclerKind::Symmetry, "SYMMETRY", symmetry_label(draft.symmetry));
                    cycler_card(g, CyclerKind::Form,     "FORM",     draft.form.label());
                    cycler_card(g, CyclerKind::Mobility, "MOBILITY", draft.mobility.label());
                });
                cycler_group(attrs, "LOCOMOTION", |g| {
                    cycler_card(g, CyclerKind::Movement,  "MOVEMENT",  draft.movement.label());
                    cycler_card(g, CyclerKind::Grounding, "GROUNDING", draft.grounding.label());
                });
                cycler_group(attrs, "MIND", |g| {
                    cycler_card(g, CyclerKind::Intelligence, "INTELLIGENCE", intelligence_label(draft.intelligence));
                });
            });

            // Right: file actions (the base cell auto-seeds from Metabolism).
            cycler_group(bar, "FILE", |g| {
                action_button::<ImportMeshButton>(g, "Import .glb", IMPORT_BG);
                action_button::<LoadSpeciesButton>(g, "Load",        ACTION_BG_LOAD);
                action_button::<CreateSpeciesButton>(g, "Save",      ACTION_BG_SUCCESS);
            });
        });
}

/// A labeled section: a small teal header over a row of cards/buttons. Pure
/// layout — no markers — so handlers are unaffected.
fn cycler_group(
    parent: &mut ChildSpawnerCommands,
    title:  &str,
    build:  impl FnOnce(&mut ChildSpawnerCommands),
) {
    parent
        .spawn((Node {
            flex_direction: FlexDirection::Column,
            align_items:    AlignItems::FlexStart,
            row_gap:        Val::Px(3.0),
            margin:         UiRect::horizontal(Val::Px(6.0)),
            ..default()
        },))
        .with_children(|grp| {
            grp.spawn((
                Text::new(title.to_string()),
                TextFont { font_size: 10.0, ..default() },
                TextColor(SECTION_LABEL_COLOR),
                Pickable::IGNORE,
            ));
            grp.spawn((Node {
                flex_direction: FlexDirection::Row,
                align_items:    AlignItems::Center,
                column_gap:     Val::Px(5.0),
                ..default()
            },))
            .with_children(|row| build(row));
        });
}

/// A captioned cycler card: the static property name over the live value
/// (`CyclerValueText`). The card root keeps the `Cycler` + `Button` markers and
/// owns the `BackgroundColor` the click/lock handlers drive — so behaviour is
/// identical to the old flat button; only the look changes.
fn cycler_card(parent: &mut ChildSpawnerCommands, kind: CyclerKind, caption: &str, value: &str) {
    parent
        .spawn((
            Cycler(kind),
            Button,
            Node {
                width:           Val::Px(112.0),
                height:          Val::Px(40.0),
                padding:         UiRect::axes(Val::Px(9.0), Val::Px(4.0)),
                flex_direction:  FlexDirection::Column,
                align_items:     AlignItems::FlexStart,
                justify_content: JustifyContent::Center,
                row_gap:         Val::Px(1.0),
                border_radius:   BorderRadius::all(CARD_RADIUS),
                ..default()
            },
            BackgroundColor(CYCLER_BG_ACTIVE),
        ))
        .with_children(|card| {
            card.spawn((
                Text::new(caption.to_string()),
                TextFont { font_size: 9.0, ..default() },
                TextColor(CAPTION_COLOR),
                Pickable::IGNORE,
            ));
            card.spawn((
                CyclerValueText(kind),
                Text::new(value.to_string()),
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
                width:           Val::Px(96.0),
                height:          Val::Px(40.0),
                align_items:     AlignItems::Center,
                justify_content: JustifyContent::Center,
                border_radius:   BorderRadius::all(CARD_RADIUS),
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
        //   * Grounding locks for SWIMMERS only: phototrophs cycle all three
        //     (ground / water / ocean-floor), and ground-moving heterotrophs
        //     cycle terrestrial ↔ ocean-floor (a benthic slider on the seafloor);
        //     a swimmer is water-based, derived from its movement mode.
        // Locked cyclers `continue` to keep the greyed colour.
        let locked = match cycler.0 {
            CyclerKind::Metabolism | CyclerKind::Symmetry => session.has_appended_cells(),
            CyclerKind::Intelligence | CyclerKind::Movement => {
                session.draft.mobility == Mobility::Sessile
            }
            CyclerKind::Grounding => {
                session.draft.metabolism != Metabolism::Photoautotroph
                    && !session.draft.movement.default_ground_based()
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
                        // Leaving Photoautotroph: re-derive grounding for a
                        // heterotroph — swimmers are water-based; ground-moving
                        // heterotrophs keep their ground-anchored choice
                        // (terrestrial OR ocean-floor benthic), defaulting a
                        // stale water value back to terrestrial.
                        if session.draft.metabolism != Metabolism::Photoautotroph {
                            session.draft.grounding =
                                if !session.draft.movement.default_ground_based() {
                                    Grounding::WaterBased
                                } else if session.draft.grounding == Grounding::WaterBased {
                                    Grounding::GroundBased
                                } else {
                                    session.draft.grounding
                                };
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
                        // Heterotroph grounding follows the new movement mode:
                        // a swimmer becomes water-based; switching between ground
                        // modes preserves the ground-anchored choice (terrestrial
                        // OR ocean-floor benthic), so a benthic slider stays benthic.
                        if session.draft.metabolism != Metabolism::Photoautotroph {
                            session.draft.grounding =
                                if !session.draft.movement.default_ground_based() {
                                    Grounding::WaterBased
                                } else if session.draft.grounding == Grounding::WaterBased {
                                    Grounding::GroundBased
                                } else {
                                    session.draft.grounding
                                };
                        }
                    }
                    CyclerKind::Grounding => {
                        // Phototrophs cycle all three (Ground → Water → Ocean-Floor);
                        // ground-moving heterotrophs (locked for swimmers) toggle
                        // terrestrial ↔ ocean-floor — both ground-anchored, the
                        // latter a benthic slider that crawls on the ocean floor.
                        session.draft.grounding = match session.draft.metabolism {
                            Metabolism::Photoautotroph => session.draft.grounding.cycle(),
                            Metabolism::Heterotroph => match session.draft.grounding {
                                Grounding::OceanFloor => Grounding::GroundBased,
                                _                     => Grounding::OceanFloor,
                            },
                        };
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
            CyclerKind::Grounding      => session.draft.grounding.label().to_string(),
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
            // Editable for phototrophs (3-way) + ground-moving heterotrophs
            // (terrestrial ↔ ocean-floor benthic slider); locked for swimmers.
            CyclerKind::Grounding => not_photo && !session.draft.movement.default_ground_based(),
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
        grounding:      Grounding::from_flags(loaded.ground_based, loaded.ocean_floor),
    };
    session.body_parts = loaded.body_parts.into_iter()
        .enumerate()
        .map(|(i, p)| EditorBodyPart {
            name: p.name,
            ocg: p.ocg,
            // Base body (index 0) is always Body; appendages keep their kind.
            kind: if i == 0 { crate::cell::BodyPartKind::Body } else { p.kind },
            parent: p.parent,
        })
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

/// Spawn / despawn the modal in lock-step with `show_load_modal`.
pub fn manage_load_modal_visibility(
    mut commands: Commands,
    session:      Res<SpeciesSession>,
    existing:     Query<Entity, With<LoadModalRoot>>,
) {
    ui_modal::sync_modal_visibility(
        &mut commands, session.show_load_modal, &existing, spawn_load_modal);
}

fn spawn_load_modal(commands: &mut Commands) {
    ui_modal::spawn_confirm_modal(
        commands,
        &ConfirmModalSpec {
            title:      None,
            body:       "There have been changes since the last save. Are you sure?".to_string(),
            body_font_size: ui_modal::BODY_FONT_SIZE_LG,
            body_color:     ui_modal::BODY_COLOR_LG,
            card_width: 560.0,
            z_index:    120,
            no_style:   NoButtonStyle::Safe,
        },
        LoadModalRoot,
        LoadModalNoButton,
        LoadModalYesButton,
    );
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
        if ui_modal::modal_button_pressed(interaction, &mut bg, YES_BTN_COLOR, YES_BTN_HOVER) {
            session.show_load_modal = false;
            do_load(&mut session);
        }
    }
    let no_style = NoButtonStyle::Safe;
    for (interaction, mut bg) in &mut no_q {
        if ui_modal::modal_button_pressed(interaction, &mut bg, no_style.base(), no_style.hover()) {
            session.show_load_modal = false;
        }
    }
}
