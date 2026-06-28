// "Load Colony" flow + unsaved-work confirmation modal.
//
// The inventory-panel button sets `load_requested`. `dispatch_load_request`
// loads immediately if the session is clean, or raises `show_load_modal` if
// there are unsaved changes. The modal's Yes → confirm load, No → cancel.
// `perform_colony_load` opens an rfd picker, REPLACES the current colony with
// the chosen `.colony` file's organisms, and clears `dirty`.
//
// Loaded organisms become `NoSymmetry` templates carrying their geometry
// verbatim (root OCG + appendages, attachment origins reconstructed). The save
// path expands bilateral organisms into mirrored runtime parts, which can't be
// reliably collapsed back to right-half templates — so the geometry round-trips
// exactly, but a `Bilateral` colony re-saves as `NoSymmetry`.

use bevy::prelude::*;

use crate::cell::CellType;
use crate::colony::{OrganismMaterials, OrganismKind};
use crate::colony_editor::placement::respawn_template;
use crate::colony_editor::session::EditorSession;
use crate::colony_editor::template::{Form, Metabolism, OrganismTemplate};
use crate::environment::WaterLevel;
use crate::organism::{OrganismRoot, Symmetry};
use crate::simulation_settings::Smoothing;
use crate::ui_modal::{
    self, ConfirmModalSpec, NoButtonStyle,
    MODAL_BACKDROP_COLOR, MODAL_CARD_COLOR, MODAL_CARD_BORDER,
    MODAL_BTN_WIDTH, MODAL_BTN_HEIGHT,
    YES_BTN_COLOR, YES_BTN_HOVER, NO_BTN_HOVER_PLAIN, NO_BTN_COLOR_PLAIN,
};


// ── Tunables (the error modal reuses the shared palette) ─────────────────────

const MODAL_CARD_WIDTH:   f32 = 480.0;
const MODAL_CARD_PADDING: f32 = 22.0;


// ── Marker components ───────────────────────────────────────────────────────

#[derive(Component)]
struct LoadModalRoot;

#[derive(Component)]
struct LoadModalYesButton;

#[derive(Component)]
struct LoadModalNoButton;

#[derive(Component)]
struct LoadErrorModalRoot;

#[derive(Component)]
struct LoadErrorOkButton;


// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct LoadModalPlugin;

impl Plugin for LoadModalPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (
            dispatch_load_request,
            manage_modal_visibility,
            handle_modal_buttons,
            perform_colony_load,
            manage_error_modal_visibility,
            handle_error_ok_button,
        ));
    }
}


// ── Dispatcher: button click → modal-or-load ─────────────────────────────────

fn dispatch_load_request(mut session: ResMut<EditorSession>) {
    if !session.load_requested { return; }
    session.load_requested = false;

    if session.dirty {
        // Defer the load until the user resolves the modal.
        session.show_load_modal = true;
    } else {
        session.load_confirmed = true;
    }
}


// ── Modal lifecycle: spawn / despawn based on `show_load_modal` ──────────────

fn manage_modal_visibility(
    mut commands: Commands,
    session:      Res<EditorSession>,
    existing:     Query<Entity, With<LoadModalRoot>>,
) {
    ui_modal::sync_modal_visibility(
        &mut commands, session.show_load_modal, &existing, spawn_modal);
}

fn spawn_modal(commands: &mut Commands) {
    ui_modal::spawn_confirm_modal(
        commands,
        &ConfirmModalSpec {
            title:      Some("Your work is not saved.".to_string()),
            body:       "Loading a colony will discard the current one. Continue?".to_string(),
            body_font_size: ui_modal::BODY_FONT_SIZE,
            body_color:     ui_modal::BODY_COLOR,
            card_width: MODAL_CARD_WIDTH,
            z_index:    100,
            no_style:   NoButtonStyle::Plain,
        },
        LoadModalRoot,
        LoadModalNoButton,
        LoadModalYesButton,
    );
}


// ── Modal button handlers ────────────────────────────────────────────────────

fn handle_modal_buttons(
    mut yes_q: Query<(&Interaction, &mut BackgroundColor),
                     (Changed<Interaction>, With<LoadModalYesButton>, Without<LoadModalNoButton>)>,
    mut no_q:  Query<(&Interaction, &mut BackgroundColor),
                     (Changed<Interaction>, With<LoadModalNoButton>, Without<LoadModalYesButton>)>,
    mut session: ResMut<EditorSession>,
) {
    for (interaction, mut bg) in &mut yes_q {
        if ui_modal::modal_button_pressed(interaction, &mut bg, YES_BTN_COLOR, YES_BTN_HOVER) {
            session.show_load_modal = false;
            session.load_confirmed  = true;
        }
    }
    let no_style = NoButtonStyle::Plain;
    for (interaction, mut bg) in &mut no_q {
        if ui_modal::modal_button_pressed(interaction, &mut bg, no_style.base(), no_style.hover()) {
            session.show_load_modal = false;
        }
    }
}


// ── Load: pick file → replace colony ─────────────────────────────────────────

/// Consumes `load_confirmed`: opens a blocking rfd picker (sim is paused in
/// editor mode), then REPLACES the current colony with the chosen file's
/// organisms. Cancelling the picker leaves the current colony untouched.
fn perform_colony_load(
    mut session:    ResMut<EditorSession>,
    mut commands:   Commands,
    mut meshes:     ResMut<Assets<Mesh>>,
    mut materials:  ResMut<Assets<StandardMaterial>>,
    org_materials:  Option<Res<OrganismMaterials>>,
    smoothing:      Option<Res<Smoothing>>,
    mut water:      ResMut<WaterLevel>,
    organisms_q:    Query<Entity, With<OrganismRoot>>,
) {
    if !session.load_confirmed { return; }
    session.load_confirmed = false;

    let initial_dir = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let path = match rfd::FileDialog::new()
        .add_filter("AEONS colony (.colony)", &["colony"])
        .set_directory(initial_dir)
        .pick_file()
    {
        Some(p) => p,
        None    => return, // cancelled — keep the current colony
    };

    let path_str = path.to_string_lossy().into_owned();
    let (_elapsed, loaded_water, records) =
        match crate::colony_save_load::load_colony_from_file(&path_str) {
            Ok(t)  => t,
            Err(e) => {
                // Surface the failure in the UI instead of crashing; the load
                // ran BEFORE any teardown, so the current colony is intact.
                error!("failed to load colony {}: {}", path.display(), e);
                session.load_error = Some(format!(
                    "Could not load colony:\n\n{e}\n\nThe current colony was kept."
                ));
                return;
            }
        };

    // ── Replace: tear down the current colony (mirror clear_modal). ──
    // Despawn template visuals before draining so a half-cleared frame can't
    // re-spawn rows for dead entities.
    let removed: Vec<OrganismTemplate> = session.templates.drain(..).collect();
    for t in &removed { commands.entity(t.entity).despawn(); }
    // Recursive despawn of any live organisms (merged mode); RemovedComponents
    // observers reclaim brain slots / counters.
    for e in &organisms_q { commands.entity(e).despawn(); }
    session.active_id = None;

    // Apply the file's water level so a re-save round-trips it.
    water.0 = loaded_water;

    let smoothing_on = smoothing.as_deref().map(|s| s.0).unwrap_or(true);
    let org_mats = org_materials.as_deref();

    let count = records.len();
    for record in &records {
        session.next_id += 1;
        let mut template = template_from_record(session.next_id, record);
        let entity = respawn_template(
            &template, &mut commands, &mut meshes, &mut materials, org_mats, smoothing_on,
        );
        template.entity = entity;
        session.templates.push(template);
    }

    // Freshly loaded ⇒ matches the on-disk file ⇒ not dirty.
    session.dirty = false;
    info!("loaded colony from {} — {} organism(s)", path.display(), count);
}


/// Build an editor `OrganismTemplate` from one decoded `.colony` record.
///
/// Loaded as `NoSymmetry` with geometry verbatim: the root OCG is taken as-is
/// (already combined for a saved bilateral organism), and each appendage's OCG
/// is reconstructed in organism space by adding back its attachment pivot
/// (`write_colony` rebased Limb/Segment/Static OCGs to their first cell). Parent
/// indices map 1:1 (0 = root), so chained appendages are preserved.
fn template_from_record(
    id:     u32,
    record: &crate::colony_save_load::LoadedRecord,
) -> OrganismTemplate {
    let org = &record.organism;

    let metabolism = match record.kind {
        OrganismKind::Photoautotroph => Metabolism::Photoautotroph,
        OrganismKind::Heterotroph    => Metabolism::Heterotroph,
    };
    let form = if org.has_variable_form { Form::Variable } else { Form::Fixed };

    // Carry any saved brain back into the template so a re-save keeps the
    // trained weights (sliding block → Sliding; limb/swim block → Ppo).
    let brain = record.brain.clone().map(crate::species_editor::save::LoadedBrain::Sliding)
        .or_else(|| record.brain_limb.clone().map(crate::species_editor::save::LoadedBrain::Ppo));

    let root_ocg = org.body_parts.first().map(|bp| bp.ocg.clone()).unwrap_or_default();

    let appendages: Vec<(Vec<(usize, Vec3, CellType)>, crate::cell::BodyPartKind, usize)> =
        org.body_parts.iter().skip(1).map(|bp| {
            let (parent, origin) = match &bp.attachment {
                Some(a) => (a.parent_idx, a.origin_local),
                None    => (0, Vec3::ZERO),
            };
            let ocg: Vec<(usize, Vec3, CellType)> =
                bp.ocg.iter().map(|(i, p, ct)| (*i, *p + origin, *ct)).collect();
            (ocg, bp.kind, parent)
        }).collect();

    OrganismTemplate {
        id,
        metabolism,
        intelligence: org.intelligence_level,
        // Geometry is already expanded on disk; load verbatim as NoSymmetry so
        // it re-saves byte-for-byte (a Bilateral colony loses its symmetry flag).
        symmetry:     Symmetry::NoSymmetry,
        form,
        position:     record.pos,
        entity:       Entity::PLACEHOLDER,
        custom_ocg:   Some(root_ocg),
        custom_appendages: appendages,
        // The `.colony` format carries no species name for editor-authored
        // colonies (empty ⇒ None) and no Carnivore flag, so neither is recovered.
        species_name: record.species_name.clone(),
        is_carnivore: false,
        movement_mode: org.movement_mode,
        is_sessile:   org.is_sessile,
        ground_based: org.ground_based,
        brain,
    }
}


// ── Error modal: shown when a load fails (bad / corrupt / unsupported file) ──

fn manage_error_modal_visibility(
    mut commands: Commands,
    session:      Res<EditorSession>,
    existing:     Query<Entity, With<LoadErrorModalRoot>>,
) {
    let want_visible = session.load_error.is_some();
    let is_visible   = !existing.is_empty();

    if want_visible && !is_visible {
        if let Some(msg) = session.load_error.as_deref() {
            spawn_error_modal(&mut commands, msg);
        }
    } else if !want_visible && is_visible {
        for e in &existing { commands.entity(e).despawn(); }
    }
}

fn spawn_error_modal(commands: &mut Commands, message: &str) {
    commands
        .spawn((
            LoadErrorModalRoot,
            Node {
                position_type: PositionType::Absolute,
                top:    Val::Px(0.0),
                left:   Val::Px(0.0),
                width:  Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                align_items:     AlignItems::Center,
                ..default()
            },
            BackgroundColor(MODAL_BACKDROP_COLOR),
            GlobalZIndex(110),
        ))
        .with_children(|root| {
            root
                .spawn((
                    Node {
                        width:  Val::Px(MODAL_CARD_WIDTH),
                        flex_direction: FlexDirection::Column,
                        align_items:    AlignItems::Center,
                        padding: UiRect::all(Val::Px(MODAL_CARD_PADDING)),
                        border:  UiRect::all(Val::Px(1.0)),
                        ..default()
                    },
                    BackgroundColor(MODAL_CARD_COLOR),
                    BorderColor::all(MODAL_CARD_BORDER),
                ))
                .with_children(|card| {
                    card.spawn((
                        Text::new("Load failed"),
                        TextFont { font_size: 18.0, ..default() },
                        TextColor(Color::srgb(0.95, 0.55, 0.55)),
                        Node { margin: UiRect::bottom(Val::Px(10.0)), ..default() },
                        Pickable::IGNORE,
                    ));
                    card.spawn((
                        Text::new(message.to_string()),
                        TextFont { font_size: 13.0, ..default() },
                        TextColor(Color::srgb(0.88, 0.88, 0.88)),
                        Node { margin: UiRect::bottom(Val::Px(20.0)), ..default() },
                        Pickable::IGNORE,
                    ));
                    card.spawn((
                        LoadErrorOkButton,
                        Button,
                        Node {
                            width:  Val::Px(MODAL_BTN_WIDTH),
                            height: Val::Px(MODAL_BTN_HEIGHT),
                            align_items:     AlignItems::Center,
                            justify_content: JustifyContent::Center,
                            ..default()
                        },
                        BackgroundColor(NO_BTN_COLOR_PLAIN),
                    ))
                    .with_children(|btn| {
                        btn.spawn((
                            Text::new("OK"),
                            TextFont { font_size: 16.0, ..default() },
                            TextColor(Color::WHITE),
                            Pickable::IGNORE,
                        ));
                    });
                });
        });
}

fn handle_error_ok_button(
    mut ok_q:    Query<(&Interaction, &mut BackgroundColor),
                       (Changed<Interaction>, With<LoadErrorOkButton>)>,
    mut session: ResMut<EditorSession>,
) {
    for (interaction, mut bg) in &mut ok_q {
        if ui_modal::modal_button_pressed(interaction, &mut bg, NO_BTN_COLOR_PLAIN, NO_BTN_HOVER_PLAIN) {
            session.load_error = None;
        }
    }
}
