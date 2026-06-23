// Species editor — Deletion Mode. While the editor-mode dropdown is set to
// Deletion, the hovered cell is highlighted deep-red and a left-click removes
// it. The hovered cell is picked via the shared `editor_mode::nearest_rendered_cell`
// (nearest RENDERED cell to the cursor within a pixel radius). For Bilateral the
// rendered set includes the mirrored left half, so the returned index is already
// mapped back via `combined_index % part.ocg.len()` — deleting removes the cell
// and its mirror together. Placement is suppressed outside Addition (see
// `placement.rs`); the mode itself is chosen in `editor_mode.rs`.

use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::cell::CellType;
use crate::frontend::ViewportImage;
use crate::player_plugin::FlyCam;
use crate::simulation_settings::WindowMode;
use crate::volumetric_growth::build_uncolored_mesh_from_ocg;

use super::editor_mode::nearest_rendered_cell;
use super::session::SpeciesSession;
use super::SPECIES_EDITOR_LAYER;


// ── Tunables ────────────────────────────────────────────────────────────────

/// Deep-red overlay colour painted over the hovered cell.
const DELETION_HILITE: Color = Color::srgb(0.80, 0.04, 0.04);

/// Max pixel distance from cursor to a cell's screen projection to count as
/// hovered (deletable).
const PICK_RADIUS_PX: f32 = 45.0;


// ── Markers / resources ───────────────────────────────────────────────────────

#[derive(Component)]
pub struct SpeciesDeletionHighlight;

/// The cell the cursor is currently hovering, as `(body_part_index,
/// ocg_index)`. Written by `update_deletion_hover`, consumed by
/// `handle_deletion_click`. `None` when nothing is hovered.
#[derive(Resource, Default)]
pub struct DeletionTarget {
    pub target: Option<(usize, usize)>,
}


// ── Hover highlight ───────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub fn update_deletion_hover(
    mode:          Res<WindowMode>,
    session:       Res<SpeciesSession>,
    mut target:    ResMut<DeletionTarget>,
    mut commands:  Commands,
    mut meshes:    ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    cameras:       Query<(&Camera, &GlobalTransform), With<FlyCam>>,
    windows:       Query<&Window, With<PrimaryWindow>>,
    viewport_q:    Query<&bevy::ui::ComputedNode, With<ViewportImage>>,
    mut hl_q:      Query<(Entity, &mut Transform, &mut Visibility), With<SpeciesDeletionHighlight>>,
) {
    if *mode != WindowMode::SpeciesEditor || !session.is_deletion() {
        target.target = None;
        for (e, _, _) in &hl_q { commands.entity(e).despawn(); }
        return;
    }

    let Ok((camera, cam_xf)) = cameras.single()      else { return };
    let Ok(window)           = windows.single()       else { return };
    let Ok(viewport_node)    = viewport_q.single()    else { return };
    let Some(cursor)         = window.cursor_position() else {
        target.target = None;
        for (_, _, mut v) in &mut hl_q { *v = Visibility::Hidden; }
        return;
    };
    let inv_scale = viewport_node.inverse_scale_factor;

    let hovered = nearest_rendered_cell(&session, camera, cam_xf, cursor, inv_scale, PICK_RADIUS_PX);

    match hovered {
        Some((pi, oi, world, _ct)) => {
            target.target = Some((pi, oi));
            if let Ok((_, mut t, mut v)) = hl_q.single_mut() {
                t.translation = world;
                *v = Visibility::Inherited;
            } else {
                // Single-cell overlay, scaled up slightly to cover the
                // underlying cell (avoids z-fighting), unlit for clean red.
                let mesh = meshes.add(build_uncolored_mesh_from_ocg(&[(0, Vec3::ZERO, CellType::DigestionCell)]));
                let mat  = materials.add(StandardMaterial {
                    base_color: DELETION_HILITE,
                    unlit:      true,
                    ..default()
                });
                commands.spawn((
                    SpeciesDeletionHighlight,
                    Mesh3d(mesh),
                    MeshMaterial3d(mat),
                    Transform::from_translation(world).with_scale(Vec3::splat(1.12)),
                    RenderLayers::layer(SPECIES_EDITOR_LAYER),
                    bevy::light::NotShadowCaster,
                ));
            }
        }
        None => {
            target.target = None;
            for (_, _, mut v) in &mut hl_q { *v = Visibility::Hidden; }
        }
    }
}


// ── Click-to-delete ───────────────────────────────────────────────────────────

pub fn handle_deletion_click(
    mode:            Res<WindowMode>,
    mouse:           Res<ButtonInput<MouseButton>>,
    target:          Res<DeletionTarget>,
    ui_interactions: Query<&Interaction>,
    mut session:     ResMut<SpeciesSession>,
) {
    if *mode != WindowMode::SpeciesEditor || !session.is_deletion() { return; }
    if !mouse.just_pressed(MouseButton::Left) { return; }
    // Don't delete on a click that a UI button claimed (e.g. the dropdown).
    if ui_interactions.iter().any(|i| matches!(i, Interaction::Pressed)) { return; }

    let Some((pi, oi)) = target.target else { return };
    let Some(part) = session.body_parts.get_mut(pi) else { return };
    if oi >= part.ocg.len() { return; }

    part.ocg.remove(oi);
    // Re-number to contiguous 0..N (the growth / bilateral pipelines require it).
    for (i, e) in part.ocg.iter_mut().enumerate() { e.0 = i; }
    session.dirty = true;
}
