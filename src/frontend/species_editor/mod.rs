// Species editor — alternate `WindowMode` that lets the user build an
// organism cell-by-cell and save the result as a `.species` binary.
//
// The mode lives far from the simulation world in coordinate space
// (`SPECIES_EDITOR_ORIGIN` at world (100_000, 1000, 100_000)) so its
// visuals never overlap the running simulation and its cells never
// enter the world model / brain pools / predation events. The
// simulation systems continue to run (we don't pause anything besides
// the standard `WindowMode::is_changed` virtual-time pause).
//
// Subsystem layout:
//
//   * session.rs        — state (cyclers, OCG, selected cell type)
//   * top_panel.rs      — 6 buttons: 4 cyclers + Spawn First + Create
//   * bottom_panel.rs   — cell-type swatch picker
//   * camera.rs         — orbit camera (middle-mouse drag, scroll-zoom)
//   * placement.rs      — preview cell + snap + click-to-place +
//                         bilateral axis + body mesh refresh
//   * save.rs           — .species binary writer (rfd Save-As dialog)

pub mod body_part_panel;
pub mod bottom_panel;
pub mod camera;
pub mod clear_modal;
pub mod placement;
pub mod save;
pub mod session;
pub mod top_panel;
pub mod undo;

use bevy::prelude::*;


// ── Module-wide constants ────────────────────────────────────────────────────

/// World position where the species editor lives. With `RenderLayers`
/// isolation in place (simulation entities on layer 0, species editor
/// on `SPECIES_EDITOR_LAYER`), placement is no longer needed to hide
/// the simulation visually — but the editor still operates in its
/// own corner of world-space so its transforms don't accidentally
/// collide with simulation systems (collision broad-phase, world
/// model spatial hash, etc.).
pub const SPECIES_EDITOR_ORIGIN: Vec3 = Vec3::new(100_000.0, 1_000.0, 100_000.0);

/// `RenderLayers` index reserved for species-editor visuals. The
/// shared 3D camera is switched to render only this layer while
/// `WindowMode::SpeciesEditor` is active (and switched back to the
/// default layer 0 otherwise), giving the editor a fully isolated
/// view — simulation entities, terrain, water, etc. are completely
/// invisible no matter how close the camera is parked. The
/// directional light is dual-layered (0 + 1) so its illumination
/// reaches species cells too.
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
            .init_resource::<camera::StashedSimCameraTransform>()
            .init_resource::<undo::SpeciesUndo>()
            // Top-panel button + cycler handlers.
            .add_systems(Update, (
                top_panel::handle_cycler_clicks,
                top_panel::sync_cycler_labels,
                top_panel::sync_cycler_lock_state,
                top_panel::handle_spawn_first_cell,
                top_panel::handle_create_species,
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
}
