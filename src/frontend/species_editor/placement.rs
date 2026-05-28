// Species editor — placement system.
//
// Two visual elements are managed here:
//   * The committed-organism mesh: rebuilt from `session.ocg` (and its
//     mirror for Bilateral) whenever the OCG changes. Uses the same
//     materials as runtime organisms so the editor preview matches
//     what the user will see in the simulation.
//   * The cursor preview cell: a translucent blue sphere-of-cells that
//     follows the mouse, snapping to the nearest valid lattice
//     position from `volumetric_growth::candidate_centers_for_ocg`.
//     Left-click commits the cell, extends the OCG, mesh rebuilds.
//   * The yellow bilateral axis: a tall thin yellow cylinder along
//     local Y at x = 0 in editor-local space (= world `SPECIES_EDITOR_ORIGIN`).
//     Visible only when `session.draft.symmetry == Bilateral`.
//
// All editor-spawned entities are tagged with `SpeciesEditorEntity`
// so they can be cleared on mode exit if needed. Currently we keep
// them across mode switches so the user's work isn't lost.

use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::body_part::{bilateral_body_part_from_right_ocg, MIN_X_BILATERAL};
use crate::cell::CellType;
use crate::colony::Symmetry;
use crate::player_plugin::FlyCam;
use crate::simulation_settings::WindowMode;
use crate::volumetric_growth::{build_mesh_from_ocg, candidate_centers_for_ocg};

use super::session::SpeciesSession;
use super::{SPECIES_EDITOR_ORIGIN, SPECIES_EDITOR_LAYER};
use crate::frontend::ViewportImage;


// ── Tunables ─────────────────────────────────────────────────────────────────

/// Maximum pixel distance from cursor to a candidate's screen
/// projection at which the preview cell will snap to that candidate.
/// Beyond this, no snap (preview hidden).
const SNAP_RADIUS_PX: f32 = 60.0;

const PREVIEW_BLUE:        Color = Color::srgba(0.20, 0.45, 0.95, 0.55);
const BILATERAL_AXIS_HEIGHT: f32 = 12.0;
const BILATERAL_AXIS_RADIUS: f32 = 0.06;
const BILATERAL_AXIS_COLOR:  Color = Color::srgba(0.95, 0.92, 0.20, 0.90);


// ── Markers ──────────────────────────────────────────────────────────────────

#[derive(Component)] pub struct SpeciesEditorMesh;
#[derive(Component)] pub struct SpeciesPreviewCell;
#[derive(Component)] pub struct SpeciesBilateralAxis;

/// Per-frame cache populated by the placement system: world-space
/// candidate positions and (post-cursor-projection) the snapped target
/// if one is active. Stored on the session for the click-handler to
/// read.
#[derive(Resource, Default)]
pub struct PlacementSnap {
    pub snapped_local: Option<Vec3>,
}


// ── Mesh refresh ─────────────────────────────────────────────────────────────

/// Rebuild the body mesh from the current OCG whenever the session
/// changes (cycler flips, first-cell spawn, new cell placed). Always
/// runs in SpeciesEditor mode; the `is_changed` gate keeps the cost
/// at zero between mutations.
pub fn refresh_species_mesh(
    mode:          Res<WindowMode>,
    session:       Res<SpeciesSession>,
    mut commands:  Commands,
    mut meshes:    ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    existing:      Query<Entity, With<SpeciesEditorMesh>>,
) {
    if *mode != WindowMode::SpeciesEditor { return; }
    if !session.is_changed() { return; }
    if session.body_parts.is_empty() { return; }

    // Despawn the old mesh entities if present. The session's meshes are
    // small and rebuilt entirely (no entity reuse needed).
    for e in &existing { commands.entity(e).despawn(); }

    // One mesh entity per body part. Each part's OCG is mirrored
    // (Bilateral) via `combined_ocg`. Per-cell colour comes from the
    // mesh's `ATTRIBUTE_COLOR` attribute (one colour per source cell),
    // so the material is a single white that lets vertex colours show
    // through unmultiplied — cells of different types within the same
    // body part now display their own colours individually.
    let mat_handle = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        ..default()
    });
    for part in &session.body_parts {
        if part.ocg.is_empty() { continue; }
        let combined = session.combined_ocg(&part.ocg);
        let mesh_handle = meshes.add(build_mesh_from_ocg(&combined));
        commands.spawn((
            SpeciesEditorMesh,
            Mesh3d(mesh_handle),
            MeshMaterial3d(mat_handle.clone()),
            Transform::from_translation(SPECIES_EDITOR_ORIGIN),
            RenderLayers::layer(SPECIES_EDITOR_LAYER),
            bevy::light::NotShadowCaster,
        ));
    }
}


// ── Bilateral axis ──────────────────────────────────────────────────────────

pub fn refresh_bilateral_axis(
    mode:          Res<WindowMode>,
    session:       Res<SpeciesSession>,
    mut commands:  Commands,
    mut meshes:    ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    existing:      Query<Entity, With<SpeciesBilateralAxis>>,
) {
    if !mode.is_changed() && !session.is_changed() { return; }

    let want_axis = *mode == WindowMode::SpeciesEditor
                 && session.draft.symmetry == Symmetry::Bilateral;

    for e in &existing { commands.entity(e).despawn(); }
    if !want_axis { return; }

    let mesh = meshes.add(Cylinder::new(BILATERAL_AXIS_RADIUS, BILATERAL_AXIS_HEIGHT).mesh());
    let mat  = materials.add(StandardMaterial {
        base_color:        BILATERAL_AXIS_COLOR,
        alpha_mode:        AlphaMode::Blend,
        unlit:             true,
        ..default()
    });
    commands.spawn((
        SpeciesBilateralAxis,
        Mesh3d(mesh),
        MeshMaterial3d(mat),
        // Local origin at x = 0; cylinder default axis is Y → already
        // upright. Translate to species-editor world origin.
        Transform::from_translation(SPECIES_EDITOR_ORIGIN),
        RenderLayers::layer(SPECIES_EDITOR_LAYER),
        bevy::light::NotShadowCaster,
    ));
}


// ── Preview cell follow + snap ──────────────────────────────────────────────

/// Run every frame in SpeciesEditor mode. Reads the cursor position,
/// projects it onto the local XZ plane through the species-editor
/// origin, finds the nearest candidate centre in screen space, and
/// positions the preview-cell entity there. Also (re)creates the
/// preview entity when `selected_cell_type` flips from None to Some.
#[allow(clippy::too_many_arguments)]
pub fn update_preview_cell(
    mode:           Res<WindowMode>,
    session:        Res<SpeciesSession>,
    mut snap:       ResMut<PlacementSnap>,
    mut commands:   Commands,
    mut meshes:     ResMut<Assets<Mesh>>,
    mut materials:  ResMut<Assets<StandardMaterial>>,
    cameras:        Query<(&Camera, &GlobalTransform), With<FlyCam>>,
    windows:        Query<&Window, With<PrimaryWindow>>,
    viewport_q:     Query<&bevy::ui::ComputedNode, With<ViewportImage>>,
    mut preview_q:  Query<(Entity, &mut Transform, &mut Visibility), With<SpeciesPreviewCell>>,
) {
    let inactive = *mode != WindowMode::SpeciesEditor
                || session.selected_cell_type.is_none()
                || !session.first_cell_spawned;

    if inactive {
        snap.snapped_local = None;
        for (e, _, _) in &preview_q { commands.entity(e).despawn(); }
        return;
    }

    let Ok((camera, cam_xf)) = cameras.single() else { return };
    let Ok(window)           = windows.single() else { return };
    let Ok(viewport_node)    = viewport_q.single() else { return };
    let Some(cursor_window)  = window.cursor_position() else {
        // No cursor on screen — hide preview.
        snap.snapped_local = None;
        for (e, _, _) in &preview_q { commands.entity(e).despawn(); }
        return;
    };

    // The cursor is in window-logical px (top-left origin). The
    // viewport image's ComputedNode gives its physical pixel rect
    // (centre + size). Convert cursor → viewport-local logical px.
    let inv_scale = viewport_node.inverse_scale_factor;
    let _ = inv_scale; // viewport-local conversion below uses world_to_viewport

    // Candidates: list of LOCAL-frame centres (relative to species editor origin).
    // Bilateral placement allows the midline (x = 0) and the +X half; the
    // constraint only rejects −X lattice slots (which belong to the
    // auto-generated left half). Midline cells bridge the two halves with
    // real shared faces — without them the halves only meet at points.
    let min_x_constraint = match session.draft.symmetry {
        Symmetry::Bilateral  => Some(0.0),
        Symmetry::NoSymmetry => None,
    };
    // Candidates come from the ACTIVE part's frontier. A freshly-begun
    // appendage has no cells yet, so its first cell attaches to the BASE
    // body's frontier instead.
    let active_empty = session.active_part().map_or(true, |p| p.ocg.is_empty());
    let source_ocg: Vec<(usize, Vec3, CellType)> = if active_empty {
        session.base_part().map(|p| p.ocg.clone()).unwrap_or_default()
    } else {
        session.active_part().map(|p| p.ocg.clone()).unwrap_or_default()
    };
    let candidates_local = candidate_centers_for_ocg(&source_ocg, min_x_constraint);
    if candidates_local.is_empty() {
        snap.snapped_local = None;
        for (e, _, _) in &preview_q { commands.entity(e).despawn(); }
        return;
    }

    // Project each candidate to viewport coords; pick the one closest
    // to the cursor in pixels.
    let mut best: Option<(f32, Vec3)> = None;
    for &local in &candidates_local {
        let world = SPECIES_EDITOR_ORIGIN + local;
        let Ok(vp_phys) = camera.world_to_viewport(cam_xf, world) else { continue };
        // `vp_phys` is in physical pixels relative to viewport's
        // top-left. Convert to window-logical px the same way
        // `individuum_navigator` does it: multiply by the viewport
        // node's inverse_scale_factor.
        let inv_scale = viewport_node.inverse_scale_factor;
        let vp_logical = vp_phys * inv_scale;
        // Now translate to window-logical coords using the viewport's
        // top-left. The simulation's viewport occupies the full
        // window above the bottom panel; for this editor we don't
        // have a viewport node carving — the ImageNode covers the
        // whole content area, so the cursor's window position == the
        // viewport's logical position (modulo top-bar offset).
        // Use the cursor position directly minus the panel offsets.
        let dx = vp_logical.x - cursor_window.x;
        let dy = vp_logical.y - cursor_window.y;
        let d2 = dx * dx + dy * dy;
        if best.map_or(true, |(b, _)| d2 < b) {
            best = Some((d2, local));
        }
    }

    let snapped_local = match best {
        Some((d2, local)) if d2.sqrt() <= SNAP_RADIUS_PX => Some(local),
        _ => None,
    };
    snap.snapped_local = snapped_local;

    // Spawn / update the preview entity. We cache a single cell's
    // worth of OCG (`[(0, local, ct)]`) and build its mesh per change
    // — single-cell mesh is just one rhombic dodecahedron, ~tens of
    // microseconds.
    let preview_local = snapped_local.unwrap_or(candidates_local[0]);
    let preview_world = SPECIES_EDITOR_ORIGIN + preview_local;
    let visible = snapped_local.is_some();

    if let Ok((_, mut transform, mut vis)) = preview_q.single_mut() {
        transform.translation = preview_world;
        *vis = if visible { Visibility::Inherited } else { Visibility::Hidden };
        return;
    }

    // No existing preview entity — spawn one. Use a single-cell mesh
    // WITHOUT per-vertex colours so the translucent blue material isn't
    // tinted by the snapped cell type's colour.
    let preview_ocg = vec![(0usize, Vec3::ZERO, session.selected_cell_type.unwrap_or(CellType::NonPhoto))];
    let mesh        = crate::volumetric_growth::build_uncolored_mesh_from_ocg(&preview_ocg);
    let mesh_handle = meshes.add(mesh);
    let mat_handle  = materials.add(StandardMaterial {
        base_color:  PREVIEW_BLUE,
        alpha_mode:  AlphaMode::Blend,
        unlit:       true,
        ..default()
    });
    commands.spawn((
        SpeciesPreviewCell,
        Mesh3d(mesh_handle),
        MeshMaterial3d(mat_handle),
        Transform::from_translation(preview_world),
        if visible { Visibility::Inherited } else { Visibility::Hidden },
        RenderLayers::layer(SPECIES_EDITOR_LAYER),
        bevy::light::NotShadowCaster,
    ));
}


// ── Left-click placement ────────────────────────────────────────────────────

/// Commit a cell at the currently-snapped candidate. Triggered by a
/// left-click anywhere the cursor is "snapped" (the preview cell is
/// visible). Does NOT consume the click globally — buttons in the top
/// / bottom panels still receive their Interaction events first via
/// Bevy's UI picking. We just opportunistically place on a frame where
/// no UI element claimed the click.
pub fn handle_left_click_place(
    mode:        Res<WindowMode>,
    mouse:       Res<ButtonInput<MouseButton>>,
    snap:        Res<PlacementSnap>,
    ui_interactions: Query<&Interaction>,
    mut session: ResMut<SpeciesSession>,
) {
    if *mode != WindowMode::SpeciesEditor { return; }
    if !mouse.just_pressed(MouseButton::Left) { return; }
    let Some(target_local) = snap.snapped_local else { return };
    let Some(ct) = session.selected_cell_type else { return };

    // If ANY UI interaction is in `Pressed` state this frame, the
    // click is owned by a button. Skip placement.
    if ui_interactions.iter().any(|i| matches!(i, Interaction::Pressed)) { return; }

    // Bilateral guard: the right-half OCG must stay at x >= 0 (midline
    // allowed); −X slots belong to the mirrored left half.
    if session.draft.symmetry == Symmetry::Bilateral
        && target_local.x < -MIN_X_BILATERAL * 0.5
    {
        return;
    }

    // Place into the ACTIVE body part. Bail if none exists yet (the base
    // is seeded by "Spawn first Cell" before placement is possible).
    let active = session.active_body_part;
    let Some(part) = session.body_parts.get_mut(active) else { return };
    let idx = part.ocg.len();
    part.ocg.push((idx, target_local, ct));
    let part_ocg = part.ocg.clone();
    session.dirty = true;

    // For bilateral, exercise the welding pipeline at edit time as a
    // sanity check; result discarded (the mesh refresh does the same
    // work). Geometric validation is `MIN_X_BILATERAL` keeping the
    // right half off the YZ plane.
    if session.draft.symmetry == Symmetry::Bilateral {
        let _ = bilateral_body_part_from_right_ocg(&part_ocg);
    }
}
