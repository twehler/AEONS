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
    if session.ocg.is_empty()  { return; }

    // Despawn the old mesh entity if present. The session's mesh is
    // small and rebuilt entirely (no entity reuse needed).
    for e in &existing { commands.entity(e).despawn(); }

    // Build mesh from the combined OCG (mirrored automatically for
    // Bilateral via `mesh_ocg()`).
    let combined = session.mesh_ocg();
    let mesh = build_mesh_from_ocg(&combined);
    let mesh_handle = meshes.add(mesh);

    // Pick a base colour matching the metabolism. Mixed-type bodies
    // still get a single material because the runtime organism mesh
    // shares one material per body part — the cell colours show
    // through via per-vertex colour data in the mesh.
    let base = match session.draft.metabolism {
        super::session::Metabolism::Photoautotroph => Color::srgb(0.2, 0.8, 0.2),
        super::session::Metabolism::Heterotroph    => Color::srgb(0.8, 0.2, 0.2),
    };
    let mat_handle = materials.add(StandardMaterial {
        base_color: base,
        ..default()
    });

    commands.spawn((
        SpeciesEditorMesh,
        Mesh3d(mesh_handle),
        MeshMaterial3d(mat_handle),
        Transform::from_translation(SPECIES_EDITOR_ORIGIN),
        RenderLayers::layer(SPECIES_EDITOR_LAYER),
        bevy::light::NotShadowCaster,
    ));
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
    let min_x_constraint = match session.draft.symmetry {
        Symmetry::Bilateral  => Some(MIN_X_BILATERAL),
        Symmetry::NoSymmetry => None,
    };
    let candidates_local = candidate_centers_for_ocg(&session.ocg, min_x_constraint);
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

    // No existing preview entity — spawn one. Use a single-cell mesh.
    let preview_ocg = vec![(0usize, Vec3::ZERO, session.selected_cell_type.unwrap_or(CellType::NonPhoto))];
    let mesh        = build_mesh_from_ocg(&preview_ocg);
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

    // Bilateral guard: the right-half OCG must stay at x >= MIN_X_BILATERAL.
    if session.draft.symmetry == Symmetry::Bilateral
        && target_local.x < MIN_X_BILATERAL - 1e-3
    {
        return;
    }

    let idx = session.ocg.len();
    session.ocg.push((idx, target_local, ct));
    session.dirty = true;

    // For bilateral, double-check the result by running through
    // `bilateral_body_part_from_right_ocg` — if the welding fails for
    // some reason (e.g. seed cell mis-aligned), at least the user
    // sees the mesh refresh and can decide to start over. The real
    // validation is geometric: `MIN_X_BILATERAL` keeps the right half
    // off the YZ plane.
    if session.draft.symmetry == Symmetry::Bilateral {
        // The dummy call exercises the welding pipeline at edit time;
        // we discard the result — `mesh_ocg()` does the same work for
        // the actual mesh refresh.
        let _ = bilateral_body_part_from_right_ocg(&session.ocg);
    }
}
