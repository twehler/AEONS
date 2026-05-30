// Colony editor — pre-generate organism layouts that AEONS can later
// load via the `.colony` save format.
//
// The editor is launched as a separate process (forked from the
// launcher). Its plugin set is intentionally tiny — no behaviour, no
// energy, no brains — so adding new editor features doesn't risk
// regressing the simulation, and so the editor starts up quickly.
//
// Subsystem layout:
//
//   * camera.rs            — WASD + hold-LMB look flycam
//   * layout.rs            — UI 2D camera + root + panel spawn
//   * creation_panel.rs    — bottom: cyclers + Create button
//   * inventory_panel.rs   — right: list + Save button
//   * placement.rs         — right-click ray-vs-heightmap placement
//   * session.rs           — shared resource: templates, active id,
//                            draft form fields, save_requested flag
//   * template.rs          — OrganismTemplate type + helpers
//   * template_marker.rs   — component on each placed visual mesh
//   * save.rs              — v003 .colony writer

pub mod clear_modal;
pub mod creation_panel;
pub mod exit_modal;
pub mod inventory_panel;
pub mod layout;
pub mod placement;
pub mod save;
pub mod session;
pub mod template;
pub mod template_marker;
pub mod species_panel;
// tool_panel.rs was retired in favour of `species_panel.rs`, which
// merges the old left-side Bulk-Add UI with the new Species Navigator.
// The file is intentionally not declared here so it doesn't compile.
pub mod undo;

use bevy::prelude::*;

use crate::camera::{self, EditorCameraPlugin};
use crate::colony_editor::session::EditorSession;
use crate::simulation_settings::WindowMode;


/// Top-level plugin that wires every editor subsystem into a single
/// `add_plugins(...)` call. Combine with Bevy's DefaultPlugins,
/// `WorldPlugin`, and `WaterPlugin` in `main.rs` to get a working
/// editor.
pub struct ColonyEditorPlugin;

impl Plugin for ColonyEditorPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<EditorSession>()
            .add_plugins(EditorCameraPlugin)
            .add_plugins(layout::EditorLayoutPlugin)
            .add_plugins(creation_panel::CreationPanelPlugin)
            .add_plugins(inventory_panel::InventoryPanelPlugin)
            .add_plugins(species_panel::SpeciesPanelPlugin)
            .add_plugins(placement::PlacementPlugin)
            .add_plugins(exit_modal::ExitModalPlugin)
            .add_plugins(clear_modal::ClearModalPlugin)
            .add_plugins(undo::UndoPlugin)
            .add_systems(Startup, spawn_editor_lighting)
            .add_systems(Update, (dispatch_save_requests, dispatch_load_species_requests));
    }
}


/// Single directional light + ambient. Mirrors the simulation's
/// lighting parameters so the editor's preview matches what the
/// player will see in-game.
fn spawn_editor_lighting(mut commands: Commands) {
    commands.insert_resource(bevy::light::DirectionalLightShadowMap { size: 1024 });
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        bevy::light::CascadeShadowConfigBuilder {
            num_cascades:             1,
            minimum_distance:         0.1,
            maximum_distance:         200.0,
            first_cascade_far_bound:  200.0,
            overlap_proportion:       0.2,
        }.build(),
        Transform::from_rotation(Quat::from_euler(
            EulerRot::XYZ,
            -std::f32::consts::FRAC_PI_4,
            std::f32::consts::FRAC_PI_4,
            0.0,
        )),
    ));
}


/// Consumes `EditorSession::save_requested`. When set, opens an rfd
/// save-file dialog on the calling thread, then writes a v003 .colony
/// file. rfd is blocking, so the simulation effectively stalls during
/// the dialog — fine for a debug tool with no real-time constraints.
fn dispatch_save_requests(mut session: ResMut<EditorSession>) {
    if !session.save_requested { return; }
    session.save_requested = false;

    if session.templates.is_empty() {
        warn!("no organisms to save — create some first");
        return;
    }

    let initial_dir = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let path = match rfd::FileDialog::new()
        .add_filter("AEONS colony (.colony)", &["colony"])
        .set_directory(initial_dir)
        .set_file_name("editor.colony")
        .save_file()
    {
        Some(p) => p,
        None    => { return; }
    };

    let path_str = path.to_string_lossy().into_owned();
    match save::write_colony(&path_str, &session.templates) {
        Ok(()) => {
            info!(
                "wrote .colony file: {} ({} organism(s))",
                path_str, session.templates.len(),
            );
            // Successful save → state is now persisted, so the
            // unsaved-work modal won't fire on the next exit.
            session.dirty = false;
        }
        Err(e) => error!("failed to write .colony file at {}: {}", path_str, e),
    }
}


/// Read one `.species` file, decode it, bilateral-expand the OCG if
/// applicable, and append a `LoadedSpecies` to the session. Returns
/// the new species id on success, `None` on any error (missing file,
/// bad magic, truncation — error logged). Does NOT touch
/// `selected_species_id`; callers decide whether to auto-select.
fn load_species_into_session(
    session: &mut crate::colony_editor::session::EditorSession,
    path:    &std::path::Path,
) -> Option<u32> {
    let loaded = match crate::species_editor::save::load_species(path) {
        Ok(l)  => l,
        Err(e) => {
            error!("failed to load species file {}: {}", path.display(), e);
            return None;
        }
    };

    // Map species-editor's local enums to colony's canonical ones.
    let metabolism = match loaded.metabolism {
        crate::species_editor::session::Metabolism::Photoautotroph
            => crate::colony_editor::template::Metabolism::Photoautotroph,
        crate::species_editor::session::Metabolism::Heterotroph
            => crate::colony_editor::template::Metabolism::Heterotroph,
    };
    let form = if loaded.has_variable_form {
        crate::colony_editor::template::Form::Variable
    } else {
        crate::colony_editor::template::Form::Fixed
    };

    // Base body (part 0). Bilateral species store RIGHT-half only, so
    // expand to right + mirrored-left + re-indexed sequentially. The
    // raw right-half OCGs of any appendage parts are carried through
    // verbatim and expanded at spawn time.
    let expand = |raw: &[(usize, bevy::math::Vec3, crate::cell::CellType)]|
        -> Vec<(usize, bevy::math::Vec3, crate::cell::CellType)> {
        match loaded.symmetry {
            crate::organism::Symmetry::NoSymmetry => raw.to_vec(),
            crate::organism::Symmetry::Bilateral  => {
                let left = crate::body_part::mirror_right_to_left(raw);
                let mut combined: Vec<(usize, bevy::math::Vec3, crate::cell::CellType)> =
                    raw.iter().chain(left.iter()).copied().collect();
                for (i, entry) in combined.iter_mut().enumerate() { entry.0 = i; }
                combined
            }
        }
    };
    let base_raw = loaded.body_parts.first().map(|p| p.ocg.clone()).unwrap_or_default();
    let ocg = expand(&base_raw);
    let appendages: Vec<(Vec<(usize, bevy::math::Vec3, crate::cell::CellType)>, bool)> =
        loaded.body_parts.iter().skip(1)
            .map(|p| (p.ocg.clone(), p.is_limb))
            .collect();

    let display_name = path.file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "Species".to_string());

    let is_carnivore = matches!(
        loaded.classification,
        crate::species_editor::session::Classification::Carnivore,
    );
    let sliding_movement = matches!(
        loaded.movement,
        crate::species_editor::session::SpeciesMovement::Sliding,
    );

    session.next_species_id += 1;
    let id = session.next_species_id;
    session.loaded_species.push(crate::colony_editor::session::LoadedSpecies {
        id,
        name:         display_name.clone(),
        metabolism,
        symmetry:     loaded.symmetry,
        intelligence: loaded.intelligence,
        form,
        is_sessile:   loaded.is_sessile,
        is_carnivore,
        sliding_movement,
        ocg,
        appendages,
        // v3 brain payload, if the file carried one. Spawn paths
        // duplicate this into a per-organism component at
        // placement time.
        brain: loaded.brain,
    });
    info!("loaded species: {}", display_name);
    Some(id)
}

/// Consumes `EditorSession::load_species_path` (set by the Species
/// Navigator's "Load Species" button). Loads and appends, and
/// auto-selects the freshly-loaded species so the next placement
/// click uses it without an extra step.
fn dispatch_load_species_requests(mut session: ResMut<EditorSession>) {
    let Some(path) = session.load_species_path.take() else { return };
    if let Some(id) = load_species_into_session(&mut session, &path) {
        session.selected_species_id = Some(id);
    }
}

/// Startup scan of the `species/` directory next to the executable.
/// Every `*.species` file found is loaded and appended to the
/// session, in filename-sorted order so the navigator list is
/// deterministic across runs. No species is auto-selected — the
/// user picks one by clicking a row. Missing or empty `species/`
/// directories are silently ignored (logged at debug level).
fn autoload_species_folder(mut session: ResMut<EditorSession>) {
    let dir = std::path::Path::new("species");
    if !dir.is_dir() {
        debug!("no species/ directory next to the binary — nothing to autoload");
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        warn!("species/ directory exists but couldn't be read — skipping autoload");
        return;
    };
    let mut paths: Vec<std::path::PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("species"))
        .collect();
    paths.sort();
    for path in paths {
        let _ = load_species_into_session(&mut session, &path);
    }
}


// ── Merged-mode (in-simulation) editor overlay ───────────────────────────────
//
// The simulation uses this plugin instead of `ColonyEditorPlugin`. It
// adds every editor system the standalone editor needs but skips the
// pieces that would clash with the simulation:
//   * No Camera2d/Camera3d (sim has its own).
//   * No editor lighting (sim has its own DirectionalLight).
//   * No layout-root spawning — editor panels are inserted directly
//     as children of the simulation's existing UI tree via
//     `spawn_overlay_panels`, called from `frontend.rs::setup_panes`.
//
// Input handlers (placement, undo, save-shortcut, exit-modal, editor
// camera input) are gated on `WindowMode::EditColony` so they don't
// fire while the user is in Simulation mode.
//
// `dispatch_save_requests` is the same blocking save flow as the
// standalone editor; in merged mode the simulation is already paused
// when the user enters Edit Colony, so the stall is unproblematic.

/// Marker on every editor panel (creation, tool, inventory) attached
/// to the simulation's UI tree. The mode-transition system in
/// `frontend.rs` queries on this marker to flip `Display` between
/// `None` (Simulation) and `Flex` (EditColony).
#[derive(Component)]
pub struct EditorOverlayPanel;

/// Insert the editor's three panels as children of the simulation's
/// layout root. `top_offset_px` is the mode-bar height — keeps the
/// tool/inventory panels from extending up under the bar.
pub fn spawn_overlay_panels(parent: &mut ChildSpawnerCommands, top_offset_px: f32) {
    species_panel::spawn_with_offset(parent, top_offset_px);
    inventory_panel::spawn_with_offset(parent, top_offset_px);
    creation_panel::spawn_with_offset(parent, session::DraftOrganism::default(), top_offset_px);

    // Tag the three panel roots after the fact so the mode-transition
    // system can flip their `Display`. We can't pass the marker as a
    // bundle entry through the existing spawn functions without
    // touching every panel's spawn signature, so we register a small
    // one-shot startup that walks the panel queries and inserts the
    // marker.
    // The marker is inserted by `tag_editor_overlay_panels` (a
    // Startup-PostStartup system below) once the spawned entities
    // are present.
    let _ = parent;
}

/// Run after Startup to attach `EditorOverlayPanel` to every editor
/// panel root that lives under the simulation's UI tree. We can't
/// rely on the `Added<...>` filters in plugin spawn order because
/// the panels are added inside `setup_panes` (Startup) and we want
/// the marker present from the very next Update.
fn tag_editor_overlay_panels(
    mut commands: Commands,
    creation:     Query<Entity, (With<creation_panel::CreationPanel>, Without<EditorOverlayPanel>)>,
    species:      Query<Entity, (With<species_panel::SpeciesPanel>,   Without<EditorOverlayPanel>)>,
    inventory:    Query<Entity, (With<inventory_panel::InventoryPanel>, Without<EditorOverlayPanel>)>,
) {
    for e in &creation  { commands.entity(e).insert(EditorOverlayPanel); }
    for e in &species   { commands.entity(e).insert(EditorOverlayPanel); }
    for e in &inventory { commands.entity(e).insert(EditorOverlayPanel); }
}

/// Initial-display patcher: every editor panel boots `Display::None`
/// (we start in Simulation mode). Runs once at PostStartup so it
/// fires AFTER `setup_panes` has spawned the panels AND
/// `tag_editor_overlay_panels` has attached the `EditorOverlayPanel`
/// marker — querying through the shared marker keeps the system's
/// `&mut Node` access non-overlapping (three separate panel-typed
/// queries triggered Bevy's `B0001` query-conflict assertion).
fn initial_hide_editor_panels(
    mut panels:  Query<&mut Node, With<EditorOverlayPanel>>,
    window_mode: Res<WindowMode>,
) {
    if *window_mode == WindowMode::EditColony { return; }
    for mut n in &mut panels { n.display = Display::None; }
}

/// In-editor run condition for systems that should only fire during
/// `WindowMode::EditColony`. Exposed so the editor's submodule
/// systems (which live in this crate) can share the same predicate
/// without re-implementing it.
pub fn in_edit_colony_mode(mode: Res<WindowMode>) -> bool {
    *mode == WindowMode::EditColony
}

/// The merged-mode plugin. Add this from `run_simulation` in
/// `main.rs`. Standalone editor entry (`run_editor`) keeps using
/// `ColonyEditorPlugin` — the two plugins don't share systems by
/// design so each path stays self-contained.
pub struct EditorOverlayPlugin;

impl Plugin for EditorOverlayPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<EditorSession>()
            // Mark the top mode-bar strip so the editor's right-click
            // delete doesn't treat clicks on the bar as viewport
            // clicks. Standalone editor doesn't insert this, so
            // `top_strip_px` falls back to 0.
            .insert_resource(camera::CursorTopReservedPx(crate::frontend::TOP_BAR_HEIGHT_PX))
            // ViewportClick messages are produced by the picking
            // observer in `frontend.rs::viewport_click` and consumed
            // by `placement::handle_left_click`. Camera rotation is
            // driven by `player_plugin`'s `player_look`, gated on
            // `EditorLookActive` in EditColony — the editor's own
            // `handle_mouse_look` / `handle_keyboard_move` /
            // `handle_wheel_speed` are NOT used in merged mode (they
            // tracked yaw/pitch in a separate component which
            // desynced from the player camera's Transform).
            .add_message::<camera::ViewportClick>()
            .add_plugins(creation_panel::CreationPanelPlugin)
            .add_plugins(inventory_panel::InventoryPanelPlugin)
            .add_plugins(species_panel::SpeciesPanelPlugin)
            .add_plugins(exit_modal::ExitModalPlugin)
            .add_plugins(clear_modal::ClearModalPlugin)
            .add_plugins(undo::UndoPlugin)
            .add_plugins(placement::PlacementPlugin)
            // Editor save dialog dispatcher — only fires when
            // `session.save_requested` is set, which itself only
            // happens via the editor UI. Safe to run in any mode.
            .add_systems(Update, dispatch_save_requests.run_if(in_edit_colony_mode))
            // Species-load dispatcher — reads `.species` files chosen
            // via the Species Navigator's Load Species button and
            // appends them to `loaded_species`. Without this, the
            // path stash is set on click but never consumed in
            // merged mode — the species never appears in the
            // navigator list and the bulk-spawn button stays
            // disabled.
            .add_systems(Update, dispatch_load_species_requests.run_if(in_edit_colony_mode))
            // After Startup we attach the EditorOverlayPanel marker
            // and force-hide the editor panels (we boot in Simulation).
            .add_systems(PostStartup, (tag_editor_overlay_panels, initial_hide_editor_panels).chain())
            // Walk `species/` at startup and load every `.species`
            // file into the session so the navigator list is
            // populated the moment the user enters the colony
            // editor. Runs after `init_resource::<EditorSession>`
            // is committed (any Startup system).
            .add_systems(Startup, autoload_species_folder);
    }
}
