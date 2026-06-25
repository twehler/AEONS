// Species editor — placement system. Manages the committed-organism mesh
// (rebuilt from each part's OCG, mirrored for Bilateral), the cursor preview
// cell (snaps to lattice frontier candidates, left-click commits), and the
// yellow bilateral axes (along local Y and local X = mirror-normal, shown only
// for Bilateral symmetry).

use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::body_part::{bilateral_body_part_from_right_ocg, MIN_X_BILATERAL};
use crate::cell::CellType;
use crate::colony::Symmetry;
use crate::player_plugin::FlyCam;
use crate::simulation_settings::WindowMode;
use crate::volumetric_growth::{build_part_meshes, candidate_centers_for_ocg};

use super::session::SpeciesSession;
use super::mesh_import::MeshImport;
use super::{SPECIES_EDITOR_ORIGIN, SPECIES_EDITOR_LAYER};
use crate::frontend::ViewportImage;


// ── Tunables ─────────────────────────────────────────────────────────────────

/// Max pixel distance from cursor to a candidate's screen projection to snap;
/// beyond it the preview is hidden.
const SNAP_RADIUS_PX: f32 = 60.0;

const PREVIEW_BLUE:        Color = Color::srgba(0.20, 0.45, 0.95, 0.55);
const BILATERAL_AXIS_HEIGHT: f32 = 12.0;
// Thin reference bars — ¼ the old 0.06 radius (the front arrow below is the
// bold heading indicator).
const BILATERAL_AXIS_RADIUS: f32 = 0.015;
const BILATERAL_AXIS_COLOR:  Color = Color::srgba(0.95, 0.92, 0.20, 0.90);
// FRONT (heading) arrow along +Z — the organism's forward axis (the limb/swim
// brains treat body +Z as "forward"). A distinct cyan so the body's front reads
// at a glance against the yellow mirror axes.
const FRONT_ARROW_COLOR:       Color = Color::srgba(0.15, 0.85, 0.95, 0.95);
const FRONT_ARROW_LEN:         f32 = 6.0;
const FRONT_ARROW_HEAD_RADIUS: f32 = 0.30;
const FRONT_ARROW_HEAD_HEIGHT: f32 = 1.00;


// ── Markers ──────────────────────────────────────────────────────────────────

#[derive(Component)] pub struct SpeciesEditorMesh;
#[derive(Component)] pub struct SpeciesPreviewCell;
#[derive(Component)] pub struct SpeciesBilateralAxis;

/// Per-frame cache: the snapped local target, if any. Written by the placement
/// system, read by the click handler.
#[derive(Resource, Default)]
pub struct PlacementSnap {
    pub snapped_local: Option<Vec3>,
}


// ── Mesh refresh ─────────────────────────────────────────────────────────────

/// Rebuild the body mesh from the current OCG whenever the session changes.
/// The `is_changed` gate keeps the cost at zero between mutations.
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

    // Despawn old mesh entities before the empty-check, so deleting all cells
    // clears the rendered body instead of leaving it stranded.
    for e in &existing { commands.entity(e).despawn(); }
    if session.body_parts.is_empty() { return; }

    // Per-cell colour comes from the mesh's `ATTRIBUTE_COLOR`, so the material is
    // a white that lets vertex colours show through unmultiplied. Each body part
    // splits into an opaque mesh and (if it has translucent cells like Gelly) a
    // separate alpha-blended mesh — see `build_part_meshes`. Sharing one Blend
    // material for the whole part would drop the solid cells into the transparent
    // pass too, making them render glassy.
    let opaque_mat = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        ..default()
    });
    let mut blend_mat: Option<Handle<StandardMaterial>> = None;
    for part in &session.body_parts {
        if part.ocg.is_empty() { continue; }
        let combined = session.combined_ocg(&part.ocg);
        let (opaque_mesh, translucent_mesh) = build_part_meshes(&combined, false);
        if let Some(mesh) = opaque_mesh {
            commands.spawn((
                SpeciesEditorMesh,
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(opaque_mat.clone()),
                Transform::from_translation(SPECIES_EDITOR_ORIGIN),
                RenderLayers::layer(SPECIES_EDITOR_LAYER),
                bevy::light::NotShadowCaster,
            ));
        }
        if let Some(mesh) = translucent_mesh {
            let mat = blend_mat.get_or_insert_with(|| materials.add(StandardMaterial {
                base_color: Color::WHITE,
                alpha_mode: AlphaMode::Blend,
                ..default()
            })).clone();
            commands.spawn((
                SpeciesEditorMesh,
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(mat),
                Transform::from_translation(SPECIES_EDITOR_ORIGIN),
                RenderLayers::layer(SPECIES_EDITOR_LAYER),
                bevy::light::NotShadowCaster,
            ));
        }
    }
}


// ── Bilateral axis ──────────────────────────────────────────────────────────

pub fn refresh_bilateral_axis(
    mode:          Res<WindowMode>,
    session:       Res<SpeciesSession>,
    mesh_import:   Res<MeshImport>,
    mut commands:  Commands,
    mut meshes:    ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    existing:      Query<Entity, With<SpeciesBilateralAxis>>,
) {
    if !mode.is_changed() && !session.is_changed() && !mesh_import.is_changed() { return; }

    let want_axis = *mode == WindowMode::SpeciesEditor
                 && session.draft.symmetry == Symmetry::Bilateral
                 && !mesh_import.active();

    for e in &existing { commands.entity(e).despawn(); }
    if !want_axis { return; }

    // One cylinder mesh + yellow material shared by both axis bars.
    let mesh = meshes.add(Cylinder::new(BILATERAL_AXIS_RADIUS, BILATERAL_AXIS_HEIGHT).mesh());
    let mat  = materials.add(StandardMaterial {
        base_color:        BILATERAL_AXIS_COLOR,
        alpha_mode:        AlphaMode::Blend,
        unlit:             true,
        ..default()
    });
    // Y-axis bar: cylinder default axis is Y, already upright, at x = 0.
    commands.spawn((
        SpeciesBilateralAxis,
        Mesh3d(mesh.clone()),
        MeshMaterial3d(mat.clone()),
        Transform::from_translation(SPECIES_EDITOR_ORIGIN),
        RenderLayers::layer(SPECIES_EDITOR_LAYER),
        bevy::light::NotShadowCaster,
    ));
    // X-axis bar: same cylinder rotated 90° about Z so its long axis lies along
    // X (the bilateral mirror-normal).
    commands.spawn((
        SpeciesBilateralAxis,
        Mesh3d(mesh),
        MeshMaterial3d(mat),
        Transform::from_translation(SPECIES_EDITOR_ORIGIN)
            .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
        RenderLayers::layer(SPECIES_EDITOR_LAYER),
        bevy::light::NotShadowCaster,
    ));

    // FRONT (heading) arrow along +Z: a thin shaft + cone in a distinct cyan, so
    // the body's "front" is unambiguous (+Z is the canonical forward axis).
    let front_mat = materials.add(StandardMaterial {
        base_color: FRONT_ARROW_COLOR,
        alpha_mode: AlphaMode::Blend,
        unlit:      true,
        ..default()
    });
    // Shaft: cylinder (long axis +Y) rotated +90° about X so it lies along +Z,
    // centred at z = LEN/2 → spans the origin out to +Z·LEN.
    commands.spawn((
        SpeciesBilateralAxis,
        Mesh3d(meshes.add(Cylinder::new(BILATERAL_AXIS_RADIUS * 2.0, FRONT_ARROW_LEN).mesh())),
        MeshMaterial3d(front_mat.clone()),
        Transform::from_translation(SPECIES_EDITOR_ORIGIN + Vec3::new(0.0, 0.0, FRONT_ARROW_LEN * 0.5))
            .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
        RenderLayers::layer(SPECIES_EDITOR_LAYER),
        bevy::light::NotShadowCaster,
    ));
    // Arrowhead: cone (apex +Y) rotated +90° about X so its apex points +Z, set
    // just past the shaft tip.
    commands.spawn((
        SpeciesBilateralAxis,
        Mesh3d(meshes.add(Cone { radius: FRONT_ARROW_HEAD_RADIUS, height: FRONT_ARROW_HEAD_HEIGHT }.mesh())),
        MeshMaterial3d(front_mat),
        Transform::from_translation(
            SPECIES_EDITOR_ORIGIN + Vec3::new(0.0, 0.0, FRONT_ARROW_LEN + FRONT_ARROW_HEAD_HEIGHT * 0.5))
            .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
        RenderLayers::layer(SPECIES_EDITOR_LAYER),
        bevy::light::NotShadowCaster,
    ));
}


// ── Preview cell follow + snap ──────────────────────────────────────────────

/// Each frame: project candidate centres to screen, snap the preview cell to
/// the one nearest the cursor, and (re)create/destroy the preview entity as
/// `selected_cell_type` changes.
#[allow(clippy::too_many_arguments)]
pub fn update_preview_cell(
    mode:           Res<WindowMode>,
    session:        Res<SpeciesSession>,
    mesh_import:    Res<MeshImport>,
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
                || !session.first_cell_spawned
                || !session.is_addition() // only Addition mode places cells
                || mesh_import.active();  // imported mesh suspends cell placement

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

    let inv_scale = viewport_node.inverse_scale_factor;
    let _ = inv_scale; // viewport-local conversion below uses world_to_viewport

    // Candidates are LOCAL-frame centres (relative to the editor origin).
    // Bilateral allows the midline (x = 0) and +X half, rejecting −X slots (the
    // auto-generated left half). Midline cells bridge the halves with shared
    // faces; without them the halves meet only at points.
    let min_x_constraint = match session.draft.symmetry {
        Symmetry::Bilateral  => Some(0.0),
        Symmetry::NoSymmetry => None,
    };
    // Candidates come from the ACTIVE part's frontier once it has cells. A
    // freshly-begun part (no cells) can attach its first cell to the frontier of
    // ANY existing part (union of all parts' cells); contact decides its parent
    // in `handle_left_click_place`.
    let active = session.active_body_part;
    let active_empty = session.active_part().map_or(true, |p| p.ocg.is_empty());
    let source_ocg: Vec<(usize, Vec3, CellType)> = if active_empty {
        let mut all: Vec<(usize, Vec3, CellType)> = Vec::new();
        for (pi, p) in session.body_parts.iter().enumerate() {
            if pi == active { continue; }
            for &(_, pos, ct) in &p.ocg { let i = all.len(); all.push((i, pos, ct)); }
        }
        all
    } else {
        session.active_part().map(|p| p.ocg.clone()).unwrap_or_default()
    };
    let mut candidates_local = candidate_centers_for_ocg(&source_ocg, min_x_constraint);
    if candidates_local.is_empty() {
        if source_ocg.is_empty() {
            // Body fully deleted — bootstrap the base seed position so the user
            // can place a fresh first cell.
            let seed = match session.draft.symmetry {
                Symmetry::Bilateral  => Vec3::new(MIN_X_BILATERAL, 0.0, 0.0),
                Symmetry::NoSymmetry => Vec3::ZERO,
            };
            candidates_local = vec![seed];
        } else {
            snap.snapped_local = None;
            for (e, _, _) in &preview_q { commands.entity(e).despawn(); }
            return;
        }
    }

    // Project each candidate to viewport coords; pick the closest to the cursor.
    let mut best: Option<(f32, Vec3)> = None;
    for &local in &candidates_local {
        let world = SPECIES_EDITOR_ORIGIN + local;
        let Ok(vp_phys) = camera.world_to_viewport(cam_xf, world) else { continue };
        // Physical → window-logical px (same as `individuum_navigator`): multiply
        // by the viewport node's inverse_scale_factor. The ImageNode covers the
        // whole content area, so viewport-logical == cursor window position.
        let inv_scale = viewport_node.inverse_scale_factor;
        let vp_logical = vp_phys * inv_scale;
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

    // Spawn / update the preview entity (single-cell mesh, cheap to rebuild).
    let preview_local = snapped_local.unwrap_or(candidates_local[0]);
    let preview_world = SPECIES_EDITOR_ORIGIN + preview_local;
    let visible = snapped_local.is_some();

    if let Ok((_, mut transform, mut vis)) = preview_q.single_mut() {
        transform.translation = preview_world;
        *vis = if visible { Visibility::Inherited } else { Visibility::Hidden };
        return;
    }

    // No preview entity yet — spawn one. Uncolored mesh so the translucent blue
    // material isn't tinted by the cell type's vertex colour.
    let preview_ocg = vec![(0usize, Vec3::ZERO, session.selected_cell_type.unwrap_or(CellType::AbsorptionCell))];
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

/// Commit a cell at the snapped candidate on left-click. Does NOT consume the
/// click globally — UI buttons get Interaction events first; we only place on a
/// frame where no UI element claimed the click.
pub fn handle_left_click_place(
    mode:        Res<WindowMode>,
    mouse:       Res<ButtonInput<MouseButton>>,
    snap:        Res<PlacementSnap>,
    mesh_import: Res<MeshImport>,
    ui_interactions: Query<&Interaction>,
    mut session: ResMut<SpeciesSession>,
) {
    if *mode != WindowMode::SpeciesEditor { return; }
    if mesh_import.active() { return; } // imported mesh suspends cell placement
    if !session.is_addition() { return; } // only Addition mode places cells
    if !mouse.just_pressed(MouseButton::Left) { return; }
    let Some(target_local) = snap.snapped_local else { return };
    let Some(ct) = session.selected_cell_type else { return };

    // If any UI interaction is `Pressed` this frame, a button owns the click.
    if ui_interactions.iter().any(|i| matches!(i, Interaction::Pressed)) { return; }

    // Bilateral guard: right-half OCG must stay at x >= 0 (midline allowed); −X
    // slots belong to the mirrored left half.
    if session.draft.symmetry == Symmetry::Bilateral
        && target_local.x < -MIN_X_BILATERAL * 0.5
    {
        return;
    }

    // Place into the ACTIVE body part.
    let active = session.active_body_part;

    // Parent-by-contact: for the FIRST cell of the active part, parent is
    // whichever OTHER part has a cell adjacent to it, so a limb attaches to the
    // body or another limb purely by placement. Computed before the mutable
    // borrow below.
    let is_first = session.body_parts.get(active).map_or(false, |p| p.ocg.is_empty());
    let parent_by_contact = if is_first {
        let mut best: Option<(f32, usize)> = None;
        for (pi, p) in session.body_parts.iter().enumerate() {
            if pi == active { continue; }
            for &(_, pos, _) in &p.ocg {
                let d2 = (pos - target_local).length_squared();
                if best.map_or(true, |(b, _)| d2 < b) { best = Some((d2, pi)); }
            }
        }
        // Within one RD-adjacency step (same window physiology uses, ≤ 6.0).
        match best { Some((d2, pi)) if d2 <= 6.0 => Some(pi), _ => Some(0) }
    } else {
        None
    };

    let Some(part) = session.body_parts.get_mut(active) else { return };
    let idx = part.ocg.len();
    part.ocg.push((idx, target_local, ct));
    if let Some(p) = parent_by_contact { part.parent = p; }
    let part_ocg = part.ocg.clone();
    session.dirty = true;

    // For bilateral, exercise the welding pipeline at edit time as a sanity
    // check; result discarded.
    if session.draft.symmetry == Symmetry::Bilateral {
        let _ = bilateral_body_part_from_right_ocg(&part_ocg);
    }
}
