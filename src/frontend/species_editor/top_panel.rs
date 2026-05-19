// Species editor — top panel.
//
// Six buttons laid out left-to-right:
//   1. Metabolism cycler        (Photoautotroph / Heterotroph)
//   2. Intelligence cycler      (Level 0 / 1 / 2 / 3)
//   3. Symmetry cycler          (No Symmetry / Bilateral)
//   4. Form cycler              (Variable / Fixed)
//   5. Spawn first Cell         (action)
//   6. Create Species           (action — opens Save-As dialog)
//
// The four cyclers are disabled (greyed-out) once the first cell has
// been spawned — once cells exist, the body-plan flags are locked and
// the user is in "growth mode."
//
// Spawn first Cell creates the OCG's first entry at the local origin
// for NoSymmetry, or at `(MIN_X_BILATERAL, 0, 0)` for Bilateral.
// Create Species opens a native file dialog and stashes the chosen
// path in `SpeciesSession::save_requested`; the actual file write
// happens in `save.rs::dispatch_save_requests`.

use bevy::prelude::*;
use std::path::PathBuf;

use crate::cell::CellType;
use crate::colony::{IntelligenceLevel, Symmetry};
use crate::frontend::PANEL_BG_COLOR;
use crate::simulation_settings::WindowMode;

use super::session::{
    cycle_intelligence, cycle_symmetry, intelligence_label, symmetry_label,
    Classification, DraftSpecies, Mobility, SpeciesSession,
};
use super::TOP_PANEL_HEIGHT_PX;


// ── Colors ───────────────────────────────────────────────────────────────────

const CYCLER_BG_ACTIVE:    Color = Color::srgb(0.28, 0.28, 0.28);
const CYCLER_BG_LOCKED:    Color = Color::srgb(0.16, 0.16, 0.16);
const CYCLER_BG_HOVER:     Color = Color::srgb(0.38, 0.38, 0.38);
const ACTION_BG_PRIMARY:   Color = Color::srgb(0.20, 0.45, 0.70);
const ACTION_BG_SUCCESS:   Color = Color::srgb(0.20, 0.55, 0.25);
const ACTION_BG_HOVER:     Color = Color::srgb(0.30, 0.55, 0.80);
const ACTION_BG_DISABLED:  Color = Color::srgb(0.18, 0.18, 0.18);


// ── Markers ──────────────────────────────────────────────────────────────────

#[derive(Component)]
pub struct SpeciesTopPanel;

#[derive(Component, Clone, Copy, PartialEq, Eq)]
pub enum CyclerKind { Metabolism, Mobility, Classification, Intelligence, Symmetry, Form }

#[derive(Component, Clone, Copy)]
pub struct Cycler(pub CyclerKind);

/// Text-child marker so update systems can find a cycler's display
/// label without traversing every child.
#[derive(Component, Clone, Copy)]
pub struct CyclerValueText(pub CyclerKind);

#[derive(Component)]
pub struct SpawnFirstCellButton;

#[derive(Component)]
pub struct CreateSpeciesButton;


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

            // Action buttons — wider than cyclers.
            action_button::<SpawnFirstCellButton>(bar, "Spawn first Cell", ACTION_BG_PRIMARY);
            action_button::<CreateSpeciesButton>(bar, "Save Species",      ACTION_BG_SUCCESS);
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

impl Default for SpawnFirstCellButton  { fn default() -> Self { Self } }
impl Default for CreateSpeciesButton   { fn default() -> Self { Self } }


// ── Cycler click handler ────────────────────────────────────────────────────

/// Pick the canonical starting intelligence level for a given
/// (mobility, classification) pair. Used by both the Mobility and
/// Classification click handlers so toggling either knob snaps the
/// intelligence to a valid value.
fn default_intelligence_for(m: Mobility, c: Classification) -> IntelligenceLevel {
    match (m, c) {
        (Mobility::Sessile,  _)                       => IntelligenceLevel::Level0,
        (Mobility::Mobile, Classification::Herbivore) => IntelligenceLevel::Level1,
        (Mobility::Mobile, Classification::Carnivore) => IntelligenceLevel::Level2,
    }
}

pub fn handle_cycler_clicks(
    mode:             Res<WindowMode>,
    mut interactions: Query<(&Interaction, &Cycler, &mut BackgroundColor), Changed<Interaction>>,
    mut session:      ResMut<SpeciesSession>,
) {
    if *mode != WindowMode::SpeciesEditor { return; }
    if session.first_cell_spawned { return; } // locked after first cell

    for (interaction, cycler, mut bg) in &mut interactions {
        match *interaction {
            Interaction::Pressed => {
                match cycler.0 {
                    CyclerKind::Metabolism => session.draft.metabolism = session.draft.metabolism.cycle(),
                    CyclerKind::Mobility => {
                        // Toggle, then enforce the Mobility ↔ Intelligence
                        // coupling: sessile species are auto-Level0;
                        // newly-mobile species default to Level1
                        // (herbivore) or Level2 (carnivore).
                        session.draft.mobility = session.draft.mobility.cycle();
                        session.draft.intelligence = default_intelligence_for(
                            session.draft.mobility, session.draft.classification,
                        );
                    }
                    CyclerKind::Classification => {
                        // Toggle classification, then re-apply the
                        // intelligence default for the new pairing.
                        // Carnivore must be ≥ Level2; if the current
                        // intelligence is < Level2 we'd otherwise be
                        // out of valid range.
                        session.draft.classification = session.draft.classification.cycle();
                        session.draft.intelligence = default_intelligence_for(
                            session.draft.mobility, session.draft.classification,
                        );
                    }
                    CyclerKind::Intelligence => {
                        // Sessile → locked at Level0 (no cycling).
                        // Mobile herbivore → cycles {L1, L2, L3}.
                        // Mobile carnivore → cycles {L2, L3} (L1 not valid).
                        if session.draft.mobility == Mobility::Sessile { continue; }
                        let next = cycle_intelligence(session.draft.intelligence);
                        // Skip Level0 (sessile-only) and, for carnivores,
                        // also skip Level1.
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
                    CyclerKind::Symmetry => session.draft.symmetry = cycle_symmetry(session.draft.symmetry),
                    CyclerKind::Form     => session.draft.form     = session.draft.form.cycle(),
                }
                *bg = BackgroundColor(CYCLER_BG_HOVER);
            }
            Interaction::Hovered => *bg = BackgroundColor(CYCLER_BG_HOVER),
            Interaction::None    => *bg = BackgroundColor(CYCLER_BG_ACTIVE),
        }
    }
}

/// Sync cycler labels with `session.draft` whenever the session
/// changes. Edge-triggered so the per-frame cost is essentially zero.
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
        };
        if text.0 != new { text.0 = new; }
    }
}

/// Visually grey-out cyclers based on lock state:
///   * After first cell spawned → ALL cyclers locked (body plan
///     committed).
///   * While sessile (Mobility = Sessile) → Intelligence cycler locked
///     (Level0 only).
/// Other cyclers stay active in the active state.
pub fn sync_cycler_lock_state(
    session:     Res<SpeciesSession>,
    mut cyclers: Query<(&Cycler, &mut BackgroundColor)>,
) {
    if !session.is_changed() { return; }
    let all_locked = session.first_cell_spawned;
    let intel_locked_by_sessility = session.draft.mobility == Mobility::Sessile;
    for (cycler, mut bg) in &mut cyclers {
        let locked = all_locked
            || (cycler.0 == CyclerKind::Intelligence && intel_locked_by_sessility);
        *bg = BackgroundColor(if locked { CYCLER_BG_LOCKED } else { CYCLER_BG_ACTIVE });
    }
}


// ── Spawn First Cell handler ────────────────────────────────────────────────

pub fn handle_spawn_first_cell(
    mode:             Res<WindowMode>,
    mut interactions: Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<SpawnFirstCellButton>)>,
    mut session:      ResMut<SpeciesSession>,
) {
    if *mode != WindowMode::SpeciesEditor { return; }

    for (interaction, mut bg) in &mut interactions {
        let already_spawned = session.first_cell_spawned;
        match *interaction {
            Interaction::Pressed if !already_spawned => {
                let starter = session.draft.metabolism.starter_cell_type();
                let pos = match session.draft.symmetry {
                    Symmetry::NoSymmetry => Vec3::ZERO,
                    Symmetry::Bilateral  => Vec3::new(crate::body_part::MIN_X_BILATERAL, 0.0, 0.0),
                };
                session.ocg.clear();
                session.ocg.push((0usize, pos, starter));
                session.first_cell_spawned = true;
                session.dirty = true;
                *bg = BackgroundColor(ACTION_BG_HOVER);
            }
            Interaction::Pressed => *bg = BackgroundColor(ACTION_BG_DISABLED),
            Interaction::Hovered if !already_spawned => *bg = BackgroundColor(ACTION_BG_HOVER),
            _ => {
                *bg = BackgroundColor(if already_spawned { ACTION_BG_DISABLED } else { ACTION_BG_PRIMARY });
            }
        }
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
        let can_save = !session.ocg.is_empty();
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
