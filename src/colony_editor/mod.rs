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

pub mod creation_panel;
pub mod exit_modal;
pub mod inventory_panel;
pub mod layout;
pub mod placement;
pub mod save;
pub mod session;
pub mod template;
pub mod template_marker;

use bevy::prelude::*;

use crate::camera::EditorCameraPlugin;
use crate::colony_editor::session::EditorSession;


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
            .add_plugins(placement::PlacementPlugin)
            .add_plugins(exit_modal::ExitModalPlugin)
            .add_systems(Startup, spawn_editor_lighting)
            .add_systems(Update, dispatch_save_requests);
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
