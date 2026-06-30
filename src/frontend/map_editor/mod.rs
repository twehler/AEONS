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
                // Recompute the ground-nutrient table from the live painted atlas
                // when the user LEAVES the Map Editor, so the resource reflects edits.
                recompute_nutrients_on_map_editor_exit,
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


/// On LEAVING the Map Editor, rebuild the `TerrainProperties` (ground nutrients)
/// resource from the LIVE painted atlas (`PaintState.mirror`) + the prepared
/// submesh geometry + the heightmap, so the in-memory table tracks the edits the
/// user just painted. No-ops cleanly when the atlas/heightmap/resource isn't ready
/// (e.g. `.glb`/flat worlds, pre-load frames). The `.aeonsw` save path recomputes
/// independently (`export::write_combined_world`).
#[allow(clippy::too_many_arguments)]
fn recompute_nutrients_on_map_editor_exit(
    mode:       Res<crate::simulation_settings::WindowMode>,
    targets:    Res<terrain_paint::TerrainPaintTargets>,
    paint:      Res<gpu_paint::PaintState>,
    meshes:     Res<Assets<Mesh>>,
    transforms: Query<&GlobalTransform>,
    heightmap:  Option<Res<crate::world_geometry::HeightmapSampler>>,
    props:      Option<ResMut<crate::terrain_properties::TerrainProperties>>,
) {
    use bevy::mesh::{Indices, VertexAttributeValues};
    use crate::simulation_settings::WindowMode;

    // Only on the transition AWAY from MapEditor.
    if !mode.is_changed() || *mode == WindowMode::MapEditor { return; }
    let (Some(hm), Some(mut props)) = (heightmap, props) else { return };
    if !targets.prepared { return; }
    let edge = paint.atlas_edge;
    if edge == 0 || paint.mirror.len() < (edge as usize) * (edge as usize) * 4 { return; }

    // Gather the prepared atlas submeshes (own the buffers so the borrows outlive
    // the build call). Same geometry the Export path serialises.
    let mut owned: Vec<(Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<u32>, Mat4)> = Vec::new();
    for (mesh_handle, entity) in &targets.meshes {
        let Some(mesh) = meshes.get(mesh_handle) else { continue };
        let Some(VertexAttributeValues::Float32x3(pos)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else { continue };
        let Some(VertexAttributeValues::Float32x2(uv)) = mesh.attribute(Mesh::ATTRIBUTE_UV_0)
        else { continue };
        if uv.len() != pos.len() { continue; }
        let idx: Vec<u32> = match mesh.indices() {
            Some(Indices::U32(v)) => v.clone(),
            Some(Indices::U16(v)) => v.iter().map(|&i| i as u32).collect(),
            None                  => (0..pos.len() as u32).collect(),
        };
        let model = transforms.get(*entity).copied().unwrap_or_default()
            .compute_transform().to_matrix();
        owned.push((pos.clone(), uv.clone(), idx, model));
    }
    if owned.is_empty() { return; }

    let pm: Vec<crate::terrain_properties::PropMesh> = owned
        .iter()
        .map(|(p, u, i, m)| crate::terrain_properties::PropMesh {
            model: *m, positions: p, uv0: u, indices: i,
        })
        .collect();
    *props = crate::terrain_properties::build_terrain_properties(
        hm.width, hm.depth, hm.min_x, hm.min_z, &hm.heights, &paint.mirror, edge, &pm,
    );
}


// ── Layout entry point ───────────────────────────────────────────────────────

/// Spawn the map-editor's bottom + left panels as children of the passed UI root
/// (called from `frontend::setup_panes`). Both boot hidden.
pub fn spawn_overlay_panels(parent: &mut ChildSpawnerCommands, top_offset_px: f32) {
    bottom_panel::spawn_bottom_panel(parent);
    tool_panel::spawn_tool_panel(parent, top_offset_px);
    export::spawn_export_button(parent);
}
