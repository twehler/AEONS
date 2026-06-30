// Map editor — "Export" button: write the current painted terrain to a `.aeonsw`.
//
// A one-shot button at the BOTTOM-RIGHT of the Map Editor viewport. On press (once
// the terrain is prepared) it gathers every prepared atlas submesh's geometry
// (POSITION/NORMAL/UV0/indices) in SUBMESH-LOCAL space plus its entity's world
// `GlobalTransform`, the painted atlas `Image` bytes, and `MapSize`, then writes a
// `.aeonsw` file (`aeonsw_format::write_aeonsw`). The output path is the loaded
// world path with its extension swapped to `.aeonsw`, or `assets/world_export.aeonsw`
// when the source is unknown. No-op (logged) if the terrain is not prepared.

use bevy::mesh::{Indices, Mesh, VertexAttributeValues};
use bevy::prelude::*;

use crate::simulation_settings::WindowMode;
use crate::aeonsw_format::{self, PaintTextureData, AeonswData, WorldMeshEntry};
use crate::world_geometry::{HeightmapSampler, LoadedWorldPath, MapSize};

use super::gpu_paint::PaintState;
use super::terrain_paint::TerrainPaintTargets;
use super::{BOTTOM_PANEL_HEIGHT_PX, MapEditorPanel};


// ── Tunables ──────────────────────────────────────────────────────────────────

const EXPORT_BG:    Color = Color::srgb(0.20, 0.45, 0.40);
const EXPORT_HOVER: Color = Color::srgb(0.28, 0.58, 0.50);


// ── Marker ────────────────────────────────────────────────────────────────────

#[derive(Component)]
pub struct ExportWorldButton;


// ── Spawn (called from `map_editor::spawn_overlay_panels`) ──────────────────────

/// Spawn the bottom-right "Export" button. Like every `MapEditorPanel` node it
/// boots `Display::None` and is flipped visible by `visibility::toggle_map_editor_visuals`.
pub fn spawn_export_button(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((
            ExportWorldButton,
            Button,
            MapEditorPanel,
            Node {
                position_type:   PositionType::Absolute,
                right:           Val::Px(8.0),
                bottom:          Val::Px(BOTTOM_PANEL_HEIGHT_PX + 8.0),
                width:           Val::Px(110.0),
                height:          Val::Px(30.0),
                align_items:     AlignItems::Center,
                justify_content: JustifyContent::Center,
                display:         Display::None, // shown only in MapEditor mode
                ..default()
            },
            BackgroundColor(EXPORT_BG),
        ))
        .with_children(|b| {
            b.spawn((
                Text::new("Export"),
                TextFont { font_size: 13.0, ..default() },
                TextColor(Color::WHITE),
                Pickable::IGNORE,
            ));
        });
}


// ── Handler ─────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub fn handle_export_click(
    mode:        Res<WindowMode>,
    targets:     Res<TerrainPaintTargets>,
    paint:       Res<PaintState>,
    map_size:    Res<MapSize>,
    loaded_path: Option<Res<LoadedWorldPath>>,
    meshes:      Res<Assets<Mesh>>,
    images:      Res<Assets<Image>>,
    transforms:  Query<&GlobalTransform>,
    water:       Res<crate::environment::WaterLevel>,
    heightmap:   Option<Res<HeightmapSampler>>,
    colony:      crate::colony_save_load::ColonySerializeParams,
    mut buttons: Query<(&Interaction, &mut BackgroundColor),
                       (Changed<Interaction>, With<ExportWorldButton>)>,
) {
    if *mode != WindowMode::MapEditor { return; }

    for (interaction, mut bg) in &mut buttons {
        match *interaction {
            Interaction::Hovered => { *bg = BackgroundColor(EXPORT_HOVER); }
            Interaction::None    => { *bg = BackgroundColor(EXPORT_BG); }
            Interaction::Pressed => {
                *bg = BackgroundColor(EXPORT_HOVER);
                // Embed the live colony (terrain + colony in one `.aeonsw`); an
                // empty colony (count 0) writes no block → loads to Colony Editor.
                let (colony_bytes, n) = colony.to_colony_bytes();
                let embedded = if n > 0 { Some(colony_bytes) } else { None };
                export_world(&targets, &paint, &map_size, loaded_path.as_deref(),
                             &meshes, &images, &transforms, water.0,
                             heightmap.as_deref(), embedded);
            }
        }
    }
}

/// Marker for the sim-side "Save World" button (lives in the stats panel; written
/// here so it shares `write_combined_world` with the Map Editor Export).
#[derive(Component)]
pub struct SaveWorldButton;

/// Sim-side "Save World": rfd "Save As" → write a combined terrain+colony `.aeonsw`.
/// Runs in every mode (the button only exists/visible in Simulation's stats panel).
#[allow(clippy::too_many_arguments)]
pub fn handle_save_world_click(
    targets:     Res<TerrainPaintTargets>,
    paint:       Res<PaintState>,
    map_size:    Res<MapSize>,
    meshes:      Res<Assets<Mesh>>,
    images:      Res<Assets<Image>>,
    transforms:  Query<&GlobalTransform>,
    water:       Res<crate::environment::WaterLevel>,
    heightmap:   Option<Res<HeightmapSampler>>,
    colony:      crate::colony_save_load::ColonySerializeParams,
    mut buttons: Query<(&Interaction, &mut BackgroundColor),
                       (Changed<Interaction>, With<SaveWorldButton>)>,
) {
    for (interaction, mut bg) in &mut buttons {
        match *interaction {
            Interaction::Hovered => { *bg = BackgroundColor(EXPORT_HOVER); }
            Interaction::None    => { *bg = BackgroundColor(EXPORT_BG); }
            Interaction::Pressed => {
                *bg = BackgroundColor(EXPORT_HOVER);
                let Some(path) = rfd::FileDialog::new()
                    .add_filter("AEONS world (.aeonsw)", &["aeonsw"])
                    .set_file_name("world.aeonsw")
                    .save_file()
                else { return };
                let out = path.to_string_lossy().into_owned();
                let (colony_bytes, n) = colony.to_colony_bytes();
                let embedded = if n > 0 { Some(colony_bytes) } else { None };
                write_combined_world(&out, &targets, &paint, &map_size, &meshes, &images,
                                     &transforms, water.0, heightmap.as_deref(), embedded);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn export_world(
    targets:     &TerrainPaintTargets,
    paint:       &PaintState,
    map_size:    &MapSize,
    loaded_path: Option<&LoadedWorldPath>,
    meshes:      &Assets<Mesh>,
    images:      &Assets<Image>,
    transforms:  &Query<&GlobalTransform>,
    water_level: f32,
    heightmap:   Option<&HeightmapSampler>,
    embedded_colony: Option<Vec<u8>>,
) {
    // Derive the output path from the loaded world (extension → `.aeonsw`).
    let out = match loaded_path {
        Some(p) => std::path::Path::new(&p.0)
            .with_extension("aeonsw").to_string_lossy().into_owned(),
        None => "assets/world_export.aeonsw".to_string(),
    };
    write_combined_world(&out, targets, paint, map_size, meshes, images, transforms,
                         water_level, heightmap, embedded_colony);
}

/// Gather the prepared terrain (submesh geometry + painted atlas) + `MapSize` +
/// water level + the embedded colony, and write a v2 `.aeonsw` to `out`. Shared by
/// the Map Editor Export button and the sim-side "Save World" action. No-op (warns)
/// if the terrain atlas isn't prepared yet (open the Map Editor once first).
#[allow(clippy::too_many_arguments)]
pub fn write_combined_world(
    out:         &str,
    targets:     &TerrainPaintTargets,
    paint:       &PaintState,
    map_size:    &MapSize,
    meshes:      &Assets<Mesh>,
    images:      &Assets<Image>,
    transforms:  &Query<&GlobalTransform>,
    water_level: f32,
    heightmap:   Option<&HeightmapSampler>,
    embedded_colony: Option<Vec<u8>>,
) {
    // Readiness.
    let Some(image_handle) = paint.paint_image.as_ref() else {
        warn!("Save World: terrain not prepared (open the Map Editor once first).");
        return;
    };
    if !targets.prepared {
        warn!("Save World: terrain not prepared (open the Map Editor once first).");
        return;
    }

    // 1. Gather submeshes (submesh-LOCAL geometry + entity GlobalTransform).
    let mut entries: Vec<WorldMeshEntry> = Vec::new();
    for (mesh_handle, entity) in &targets.meshes {
        let Some(mesh) = meshes.get(mesh_handle) else { continue };

        let Some(VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else { continue };
        let positions: Vec<[f32; 3]> = positions.clone();
        let n = positions.len();

        let Some(VertexAttributeValues::Float32x2(uvs)) = mesh.attribute(Mesh::ATTRIBUTE_UV_0)
        else { continue };
        if uvs.len() != n { continue; }
        let uv0: Vec<[f32; 2]> = uvs.clone();

        // Normals optional — synthesise flat up if absent / mismatched.
        let normals: Vec<[f32; 3]> = match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
            Some(VertexAttributeValues::Float32x3(nm)) if nm.len() == n => nm.clone(),
            _ => vec![[0.0, 1.0, 0.0]; n],
        };

        let indices: Vec<u32> = match mesh.indices() {
            Some(Indices::U32(v)) => v.clone(),
            Some(Indices::U16(v)) => v.iter().map(|&i| i as u32).collect(),
            None                  => (0..n as u32).collect(),
        };

        // The submesh entity is a descendant of the normalising TerrainSceneRoot,
        // so only the GLOBAL transform captures final world space.
        let global = transforms.get(*entity).copied().unwrap_or_default();
        let transform = global.compute_transform();

        entries.push(WorldMeshEntry { transform, positions, normals, uv0, indices });
    }

    // 2. Gather the paint texture bytes.
    let Some(img) = images.get(image_handle) else {
        warn!("Export: paint Image asset missing.");
        return;
    };
    let width  = img.width();
    let height = img.height();
    let bytes  = img.data.clone().unwrap_or_default();
    if bytes.len() != (width as usize) * (height as usize) * 4 {
        warn!(
            "Export: paint texture byte length {} != {}x{}x4 — aborting.",
            bytes.len(), width, height
        );
        return;
    }

    // Recompute the per-cell nutrient table from the LIVE painted atlas + geometry
    // so the saved table is fresh (the atlas is square, so `width` is the edge).
    // Borrows `entries`/`bytes` — done BEFORE they move into `AeonswData`.
    let nutrients_table: Option<Vec<f32>> = heightmap.map(|hm| {
        let pm: Vec<crate::terrain_properties::PropMesh> = entries
            .iter()
            .map(|e| crate::terrain_properties::PropMesh {
                model:     e.transform.to_matrix(),
                positions: &e.positions,
                uv0:       &e.uv0,
                indices:   &e.indices,
            })
            .collect();
        let props = crate::terrain_properties::build_terrain_properties(
            hm.width, hm.depth, hm.min_x, hm.min_z, &hm.heights, &bytes, width, &pm,
        );
        props.cells.iter().flat_map(|c| [c.nitrogen, c.calcium]).collect()
    });

    let data = AeonswData {
        map_x:   map_size.x,
        map_z:   map_size.z,
        texture: PaintTextureData { width, height, bytes },
        meshes:  entries,
        water_level: Some(water_level),
        embedded_colony,
        nutrients_table,
    };

    match aeonsw_format::write_aeonsw(out, &data) {
        Ok(()) => {
            let abs = std::fs::canonicalize(out)
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|_| out.to_string());
            info!(
                "Wrote .aeonsw: {abs} ({} of {} submeshes, {}x{} texture, colony {})",
                data.meshes.len(), targets.meshes.len(), width, height,
                if data.embedded_colony.is_some() { "embedded" } else { "none" },
            );
        }
        Err(e) => error!("Save World: failed to write .aeonsw '{out}': {e}"),
    }
}
