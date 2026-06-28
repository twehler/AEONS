// Map editor — "Export" button: write the current painted terrain to a `.world`.
//
// A one-shot button at the BOTTOM-RIGHT of the Map Editor viewport. On press (once
// the terrain is prepared) it gathers every prepared atlas submesh's geometry
// (POSITION/NORMAL/UV0/indices) in SUBMESH-LOCAL space plus its entity's world
// `GlobalTransform`, the painted atlas `Image` bytes, and `MapSize`, then writes a
// `.world` file (`world_format::write_world`). The output path is the loaded
// world path with its extension swapped to `.world`, or `assets/world_export.world`
// when the source is unknown. No-op (logged) if the terrain is not prepared.

use bevy::mesh::{Indices, Mesh, VertexAttributeValues};
use bevy::prelude::*;

use crate::simulation_settings::WindowMode;
use crate::world_format::{self, PaintTextureData, WorldData, WorldMeshEntry};
use crate::world_geometry::{LoadedWorldPath, MapSize};

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
                export_world(&targets, &paint, &map_size, loaded_path.as_deref(),
                             &meshes, &images, &transforms);
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
) {
    // Readiness.
    let Some(image_handle) = paint.paint_image.as_ref() else {
        warn!("Export: world not ready to export (no paint texture).");
        return;
    };
    if !targets.prepared {
        warn!("Export: world not ready to export (terrain not prepared).");
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

    // 3. Derive the output path.
    let out = match loaded_path {
        Some(p) => std::path::Path::new(&p.0)
            .with_extension("world")
            .to_string_lossy()
            .into_owned(),
        None => "assets/world_export.world".to_string(),
    };

    let data = WorldData {
        map_x:   map_size.x,
        map_z:   map_size.z,
        texture: PaintTextureData { width, height, bytes },
        meshes:  entries,
    };

    match world_format::write_world(&out, &data) {
        Ok(()) => {
            let abs = std::fs::canonicalize(&out)
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or(out);
            info!(
                "Exported .world: {abs} ({} of {} submeshes, {}x{} texture)",
                data.meshes.len(), targets.meshes.len(), width, height
            );
        }
        Err(e) => error!("Export: failed to write .world '{out}': {e}"),
    }
}
