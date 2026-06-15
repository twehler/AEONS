// Colony editor — pre-generate organism layouts loadable via the
// `.colony` save format. Launched as a separate process; its plugin set
// is intentionally tiny (no behaviour/energy/brains) so editor changes
// can't regress the simulation.

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
pub mod undo;

use bevy::prelude::*;

use crate::camera::{self, EditorCameraPlugin};
use crate::colony_editor::session::EditorSession;
use crate::simulation_settings::WindowMode;


/// Top-level plugin wiring every editor subsystem. Combine with
/// DefaultPlugins + `WorldPlugin` + `WaterPlugin` for a working editor.
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


/// Directional light mirroring the simulation's lighting so the
/// editor preview matches in-game.
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


/// Consumes `EditorSession::save_requested`: opens a blocking rfd
/// save dialog (sim stalls — fine for a debug tool) and writes a .colony.
fn dispatch_save_requests(
    mut session: ResMut<EditorSession>,
    water: Res<crate::environment::WaterLevel>,
) {
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
    match save::write_colony(&path_str, &session.templates, water.0) {
        Ok(()) => {
            info!(
                "wrote .colony file: {} ({} organism(s))",
                path_str, session.templates.len(),
            );
            // Persisted ⇒ unsaved-work modal won't fire on next exit.
            session.dirty = false;
        }
        Err(e) => error!("failed to write .colony file at {}: {}", path_str, e),
    }
}


/// Decode one `.species` file (bilateral-expanding the OCG if needed)
/// and append a `LoadedSpecies`. Returns the new id, or `None` on error
/// (logged). Does NOT touch `selected_species_id`.
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

    // Map species-editor enums to colony's canonical ones.
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

    // Bilateral species store the RIGHT half only: expand base body to
    // right + mirrored-left, re-indexed. Appendage right-halves are
    // carried verbatim and expanded at spawn time.
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
    let appendages: Vec<(Vec<(usize, bevy::math::Vec3, crate::cell::CellType)>, crate::cell::BodyPartKind, usize)> =
        loaded.body_parts.iter().skip(1)
            // Preserve the full kind so Segment/Static fuse (not split) at spawn/save.
            .map(|p| (p.ocg.clone(), p.kind, p.parent))
            .collect();

    let display_name = path.file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "Species".to_string());

    let is_carnivore = matches!(
        loaded.classification,
        crate::species_editor::session::Classification::Carnivore,
    );
    let movement_mode = loaded.movement;

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
        movement_mode,
        ground_based: loaded.ground_based,
        ocg,
        appendages,
        // v3 brain payload if present; duplicated per-organism at spawn.
        brain: loaded.brain,
    });
    info!("loaded species: {}", display_name);
    Some(id)
}

/// Consumes `EditorSession::load_species_path`, loads + appends, and
/// auto-selects the new species.
fn dispatch_load_species_requests(mut session: ResMut<EditorSession>) {
    // Read-only check FIRST (immutable deref): `.take()` mutably derefs `session`
    // and would mark `EditorSession` CHANGED every frame even when there's nothing
    // to load — which makes every `session.is_changed()`-gated rebuild (the species
    // palette, etc.) despawn+respawn its rows each frame → massive UI flicker. Only
    // touch it mutably when a load is actually pending.
    if session.load_species_path.is_none() { return; }
    let Some(path) = session.load_species_path.take() else { return };
    if let Some(id) = load_species_into_session(&mut session, &path) {
        session.selected_species_id = Some(id);
    }
}

/// Startup scan of `species/` next to the executable: loads every
/// `*.species` in filename-sorted order (deterministic). No auto-select.
/// Missing/unreadable dir is ignored.
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
// Used by the simulation instead of `ColonyEditorPlugin`: skips the
// pieces that would clash (no Camera2d/3d, no editor lighting, no
// layout root — panels are inserted into the sim's UI tree via
// `spawn_overlay_panels`). Input handlers gate on `WindowMode::EditColony`.

/// Marker on every editor panel attached to the simulation's UI tree.
/// The mode-transition system in `frontend.rs` flips its `Display`
/// between `None` (Simulation) and `Flex` (EditColony).
#[derive(Component)]
pub struct EditorOverlayPanel;

/// Insert the editor's three panels into the simulation's layout root.
/// `top_offset_px` (mode-bar height) keeps panels clear of the bar.
/// The `EditorOverlayPanel` marker is attached later by
/// `tag_editor_overlay_panels` once the entities exist.
pub fn spawn_overlay_panels(parent: &mut ChildSpawnerCommands, top_offset_px: f32) {
    species_panel::spawn_with_offset(parent, top_offset_px);
    inventory_panel::spawn_with_offset(parent, top_offset_px);
    creation_panel::spawn_with_offset(parent, session::DraftOrganism::default(), top_offset_px);
    let _ = parent;
}

/// Attach `EditorOverlayPanel` to every editor panel root after Startup
/// (panels are spawned inside `setup_panes`).
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

/// Boot every editor panel to `Display::None` (we start in Simulation).
/// Runs at PostStartup, after spawn + tagging. Queries through the
/// shared marker — three panel-typed `&mut Node` queries tripped Bevy's
/// `B0001` query-conflict assertion.
fn initial_hide_editor_panels(
    mut panels:  Query<&mut Node, With<EditorOverlayPanel>>,
    window_mode: Res<WindowMode>,
) {
    if *window_mode == WindowMode::EditColony { return; }
    for mut n in &mut panels { n.display = Display::None; }
}

/// Run condition: fire only during `WindowMode::EditColony`.
pub fn in_edit_colony_mode(mode: Res<WindowMode>) -> bool {
    *mode == WindowMode::EditColony
}

/// Merged-mode plugin (added from `run_simulation`). Standalone
/// (`run_editor`) uses `ColonyEditorPlugin`; the two share no systems
/// by design.
pub struct EditorOverlayPlugin;

impl Plugin for EditorOverlayPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<EditorSession>()
            // Reserve the top mode-bar strip so right-click delete
            // doesn't treat bar clicks as viewport clicks (standalone
            // omits this, so `top_strip_px` is 0 there).
            .insert_resource(camera::CursorTopReservedPx(crate::frontend::TOP_BAR_HEIGHT_PX))
            // ViewportClick is produced by `frontend.rs::viewport_click`
            // and consumed by `placement::handle_left_click`. In merged
            // mode camera rotation is `player_plugin::player_look`; the
            // editor's own look/move/wheel handlers are NOT used (they
            // desynced from the player camera's Transform).
            .add_message::<camera::ViewportClick>()
            .add_plugins(creation_panel::CreationPanelPlugin)
            .add_plugins(inventory_panel::InventoryPanelPlugin)
            .add_plugins(species_panel::SpeciesPanelPlugin)
            .add_plugins(exit_modal::ExitModalPlugin)
            .add_plugins(clear_modal::ClearModalPlugin)
            .add_plugins(undo::UndoPlugin)
            .add_plugins(placement::PlacementPlugin)
            .add_systems(Update, dispatch_save_requests.run_if(in_edit_colony_mode))
            // Consumes the Load Species path stash; without it the
            // species never reaches the navigator list in merged mode.
            .add_systems(Update, dispatch_load_species_requests.run_if(in_edit_colony_mode))
            // After Startup: tag panels, then hide them (boot in Simulation).
            .add_systems(PostStartup, (tag_editor_overlay_panels, initial_hide_editor_panels).chain())
            // Populate the navigator list from `species/` at startup.
            .add_systems(Startup, autoload_species_folder);
    }
}
