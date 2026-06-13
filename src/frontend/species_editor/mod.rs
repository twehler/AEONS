// Species editor — alternate `WindowMode` to build an organism cell-by-cell and
// save a `.species` binary. Lives far from the simulation in world space
// (`SPECIES_EDITOR_ORIGIN`) so its transforms never collide with simulation
// systems (collision broad-phase, world-model hash, predation). Simulation
// systems keep running aside from the standard virtual-time pause.

pub mod body_part_panel;
pub mod bottom_panel;
pub mod camera;
pub mod clear_modal;
pub mod convert_to_cellular_mesh;
pub mod deletion;
pub mod mesh_import;
pub mod placement;
pub mod save;
pub mod session;
pub mod top_panel;
pub mod undo;

use bevy::prelude::*;


// ── Module-wide constants ────────────────────────────────────────────────────

/// World position where the species editor lives, far from the simulation so its
/// transforms don't collide with simulation systems (collision broad-phase,
/// world-model hash, etc.).
pub const SPECIES_EDITOR_ORIGIN: Vec3 = Vec3::new(100_000.0, 1_000.0, 100_000.0);

/// `RenderLayers` index for species-editor visuals. The shared camera renders
/// only this layer in SpeciesEditor mode (layer 0 otherwise), fully isolating
/// the view. The directional light is dual-layered (0 + 1) so it reaches species
/// cells too.
pub const SPECIES_EDITOR_LAYER: usize = 1;

/// Pixel heights of the two panels.
pub const TOP_PANEL_HEIGHT_PX:    f32 = 52.0;
pub const BOTTOM_PANEL_HEIGHT_PX: f32 = 90.0;


// ── Markers ──────────────────────────────────────────────────────────────────

/// Tag on every UI panel root that belongs to the species editor.
/// `frontend::apply_mode_transition` queries on this marker to flip
/// `Display::None` ↔ `Display::Flex` when the mode changes.
#[derive(Component)]
pub struct SpeciesEditorPanel;


// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct SpeciesEditorPlugin;

impl Plugin for SpeciesEditorPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_plugins(clear_modal::ClearModalPlugin)
            .init_resource::<session::SpeciesSession>()
            .init_resource::<placement::PlacementSnap>()
            .init_resource::<mesh_import::MeshImport>()
            .init_resource::<deletion::DeletionTarget>()
            .init_resource::<camera::StashedSimCameraTransform>()
            .init_resource::<undo::SpeciesUndo>()
            // Top-panel button + cycler handlers.
            .add_systems(Update, (
                top_panel::handle_cycler_clicks,
                top_panel::sync_cycler_labels,
                top_panel::sync_cycler_lock_state,
                top_panel::handle_create_species,
                top_panel::handle_load_species,
                top_panel::manage_load_modal_visibility,
                top_panel::handle_load_modal_buttons,
            ))
            // Bottom-panel tile picker.
            .add_systems(Update, (
                bottom_panel::handle_tile_clicks,
                bottom_panel::sync_tile_borders,
            ))
            // Body-part index panel.
            .add_systems(Update, (
                body_part_panel::handle_begin_new_body_part,
                body_part_panel::manage_body_part_list,
                body_part_panel::handle_body_part_row_clicks,
                body_part_panel::handle_limb_toggle,
                body_part_panel::handle_rename_input,
                body_part_panel::sync_body_part_rows,
            ))
            // Camera + placement.
            .add_systems(Update, (
                camera::snap_camera_on_mode_entry,
                camera::orbit_camera_input,
                placement::refresh_species_mesh,
                placement::refresh_bilateral_axis,
                placement::update_preview_cell,
                placement::handle_left_click_place,
            ))
            // Cell-Deletion Mode: toggle button, hover highlight, click-delete.
            // Hover runs before click so the click reads a fresh target.
            .add_systems(Update, (
                deletion::handle_cell_deletion_button,
                (deletion::update_deletion_hover, deletion::handle_deletion_click).chain(),
            ))
            // Temporary `.glb` mesh import + Blender-style scaling.
            .add_systems(Update, (
                mesh_import::handle_import_button,
                mesh_import::manage_warning_modal_visibility,
                mesh_import::handle_warning_modal_buttons,
                mesh_import::apply_imported_mesh_render_layers,
                mesh_import::apply_mesh_scale,
                mesh_import::manage_overlay_entities,
                mesh_import::update_scale_interaction,
                mesh_import::cleanup_on_mode_exit,
                mesh_import::sync_with_session,
            ))
            // Voxelize an imported mesh into AEONS cells.
            .add_systems(Update, (
                convert_to_cellular_mesh::manage_convert_button_visibility,
                convert_to_cellular_mesh::handle_convert_button,
            ))
            // Undo (Ctrl+Z). `track` records prior states; `handle`
            // restores. `track` is ordered AFTER the mutating systems and
            // the undo handler so a restore is synced, not re-captured.
            .add_systems(Update, (
                undo::handle_species_undo_shortcut,
                undo::track_species_undo,
            ).chain())
            // Save dispatcher.
            .add_systems(Update, save::dispatch_save_requests);
    }
}


// ── Layout entry point ───────────────────────────────────────────────────────

/// Spawn the species-editor's top + bottom panels as children of the
/// passed UI root (called from `frontend::setup_panes`).
pub fn spawn_overlay_panels(parent: &mut ChildSpawnerCommands, top_offset_px: f32) {
    top_panel::spawn_top_panel(parent, top_offset_px);
    bottom_panel::spawn_bottom_panel(parent);
    body_part_panel::spawn_body_part_panel(parent, top_offset_px);
    clear_modal::spawn_clear_new_button(parent);
    deletion::spawn_cell_deletion_button(parent);
    convert_to_cellular_mesh::spawn_convert_button(parent);
}
