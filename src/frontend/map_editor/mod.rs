// Map editor — alternate `WindowMode` for painting the terrain. On entry the
// shared camera parks top-down/orthographic over the whole map, all organisms
// and the water plane are hidden (reversibly), and the simulation pauses. The
// user paints the terrain's vertex colours with a brush or recolours the whole
// map at once. Painting is VISUAL-ONLY and RUNTIME-ONLY — it never touches the
// simulation or any save format.

pub mod bottom_panel;
pub mod brush_cursor;
pub mod camera;
pub mod export;
pub mod gpu_paint;
pub mod material;
pub mod paint_upload;
pub mod terrain_paint;
pub mod tool_panel;
pub mod uv_unwrap;
pub mod visibility;

use bevy::prelude::*;

use crate::world_geometry::WorldMesh;

pub use material::MapEditorSession;
pub use terrain_paint::TerrainSceneRoot;


// ── Module-wide constants ────────────────────────────────────────────────────

/// Pixel heights of the two side/bottom panels (mirroring the species editor).
pub const TOP_PANEL_HEIGHT_PX:    f32 = 66.0;
pub const BOTTOM_PANEL_HEIGHT_PX: f32 = 112.0;


// ── Marker ───────────────────────────────────────────────────────────────────

/// Tag on every UI panel root that belongs to the map editor. The visibility
/// toggle (`visibility::toggle_map_editor_visuals`) flips `Display::None` ↔
/// `Display::Flex` on this marker when the mode changes.
#[derive(Component)]
pub struct MapEditorPanel;


// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct MapEditorPlugin;

impl Plugin for MapEditorPlugin {
    fn build(&self, app: &mut App) {
        // Painting stamps into an authoritative CPU mirror of the shared paint
        // `Image`, then ships ONLY the dirty sub-rectangle to the RenderApp for an
        // IN-PLACE `RenderQueue::write_texture` against the persistent `GpuImage`
        // (no `AssetEvent::Modified`, so no extract clone / GPU realloc / material
        // bind-group rebuild). See `gpu_paint::brush_stroke` + `paint_upload`.
        app
            .add_plugins(paint_upload::PaintUploadPlugin)
            // Brush-radius cursor ring (invert-blend outline) — custom UI material.
            .add_plugins(UiMaterialPlugin::<brush_cursor::BrushCursorMaterial>::default())
            .init_resource::<MapEditorSession>()
            .init_resource::<camera::StashedMapCamera>()
            .init_resource::<terrain_paint::TerrainPaintTargets>()
            .init_resource::<gpu_paint::PaintState>()
            .init_resource::<gpu_paint::BrushSizeEditState>()
            .init_resource::<gpu_paint::SoftnessEditState>()
            // Mode-transition: panel Display + organism/water hide, and the
            // top-down ortho camera. The camera snap is ordered AFTER the species
            // editor's snap so a direct SpeciesEditor→MapEditor switch lands on
            // our pose (both write the shared `Transform`).
            .add_systems(Update, (
                visibility::toggle_map_editor_visuals,
                visibility::enforce_water_hidden_in_map_editor,
                camera::snap_map_camera_on_mode_entry
                    .after(crate::species_editor::camera::snap_camera_on_mode_entry),
            ))
            // Bottom-panel swatches.
            .add_systems(Update, (
                bottom_panel::handle_tile_clicks,
                bottom_panel::sync_tile_borders,
            ))
            // Left tool panel: brush dropdown + brush-size field + "Color All".
            .add_systems(Update, (
                tool_panel::handle_brush_dropdown_clicks,
                tool_panel::sync_brush_widget,
                tool_panel::handle_brush_size_input,
                tool_panel::update_brush_size_text,
                tool_panel::handle_softness_input,
                tool_panel::update_softness_text,
                tool_panel::handle_color_all_click,
            ))
            // Terrain prep + brush painting (paint reads the async `WorldMesh` and
            // the prepared atlas meshes).
            .add_systems(Update, (
                terrain_paint::prepare_terrain_paint,
                gpu_paint::brush_stroke
                    .after(terrain_paint::prepare_terrain_paint)
                    .run_if(resource_exists::<WorldMesh>),
                // Bottom-right "Export" → write a `.aeonsw` file.
                export::handle_export_click,
                // Sim-side "Save World" (stats panel) → combined terrain+colony
                // `.aeonsw`; runs in every mode (self-gated by the button's presence).
                export::handle_save_world_click,
                // Brush-radius cursor ring: spawn-once + per-frame follow/gate.
                brush_cursor::spawn_brush_cursor_ring,
                brush_cursor::update_brush_cursor_ring,
            ));
    }
}


// ── Layout entry point ───────────────────────────────────────────────────────

/// Spawn the map-editor's bottom + left panels as children of the passed UI root
/// (called from `frontend::setup_panes`). Both boot hidden.
pub fn spawn_overlay_panels(parent: &mut ChildSpawnerCommands, top_offset_px: f32) {
    bottom_panel::spawn_bottom_panel(parent);
    tool_panel::spawn_tool_panel(parent, top_offset_px);
    export::spawn_export_button(parent);
}
