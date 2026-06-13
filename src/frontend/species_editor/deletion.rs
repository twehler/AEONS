// Species editor — Cell-Deletion Mode. While active, the hovered cell is
// highlighted deep-red and a left-click removes it. Picking mirrors
// `placement::update_preview_cell` (nearest RENDERED cell to the cursor within a
// pixel radius). For Bilateral the rendered set includes the mirrored left half,
// so a combined-index maps back via `combined_index % part.ocg.len()` and
// deleting removes the cell and its mirror together. Placement is suppressed
// while this mode is on (see `placement.rs`).

use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::cell::CellType;
use crate::frontend::ViewportImage;
use crate::player_plugin::FlyCam;
use crate::simulation_settings::WindowMode;
use crate::volumetric_growth::build_uncolored_mesh_from_ocg;

use super::session::SpeciesSession;
use super::{BOTTOM_PANEL_HEIGHT_PX, SpeciesEditorPanel, SPECIES_EDITOR_LAYER, SPECIES_EDITOR_ORIGIN};


// ── Tunables ────────────────────────────────────────────────────────────────

const BTN_MARGIN:      f32 = 12.0;
const CLEAR_BTN_WIDTH: f32 = 160.0; // mirrors clear_modal::BTN_WIDTH
const DEL_BTN_GAP:     f32 = 8.0;
const DEL_BTN_WIDTH:   f32 = 170.0;
const DEL_BTN_HEIGHT:  f32 = 36.0;

const DEL_BTN_NORMAL:  Color = Color::srgb(0.55, 0.16, 0.16); // red (idle)
const DEL_BTN_ACTIVE:  Color = Color::srgb(0.85, 0.15, 0.15); // bright red (mode on)
const DEL_BTN_HOVER:   Color = Color::srgb(0.68, 0.20, 0.20);

/// Deep-red overlay colour painted over the hovered cell.
const DELETION_HILITE: Color = Color::srgb(0.80, 0.04, 0.04);

/// Max pixel distance from cursor to a cell's screen projection to count as
/// hovered (deletable).
const PICK_RADIUS_PX:  f32 = 45.0;


// ── Markers / resources ───────────────────────────────────────────────────────

#[derive(Component)]
pub struct CellDeletionButton;

#[derive(Component)]
pub struct SpeciesDeletionHighlight;

/// The cell the cursor is currently hovering, as `(body_part_index,
/// ocg_index)`. Written by `update_deletion_hover`, consumed by
/// `handle_deletion_click`. `None` when nothing is hovered.
#[derive(Resource, Default)]
pub struct DeletionTarget {
    pub target: Option<(usize, usize)>,
}


// ── Button spawn ──────────────────────────────────────────────────────────────

/// Spawn the "Cell-Deletion Mode" button, just left of Clear/New.
pub fn spawn_cell_deletion_button(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((
            CellDeletionButton,
            SpeciesEditorPanel,
            Button,
            Node {
                position_type:   PositionType::Absolute,
                bottom:          Val::Px(BOTTOM_PANEL_HEIGHT_PX + BTN_MARGIN),
                right:           Val::Px(BTN_MARGIN + CLEAR_BTN_WIDTH + DEL_BTN_GAP),
                width:           Val::Px(DEL_BTN_WIDTH),
                height:          Val::Px(DEL_BTN_HEIGHT),
                align_items:     AlignItems::Center,
                justify_content: JustifyContent::Center,
                display:         Display::None, // shown only in SpeciesEditor mode
                ..default()
            },
            BackgroundColor(DEL_BTN_NORMAL),
        ))
        .with_children(|btn| {
            btn.spawn((
                Text::new("Cell-Deletion Mode"),
                TextFont { font_size: 14.0, ..default() },
                TextColor(Color::WHITE),
                Pickable::IGNORE,
            ));
        });
}


// ── Toggle handler ────────────────────────────────────────────────────────────

pub fn handle_cell_deletion_button(
    mode:             Res<WindowMode>,
    mut interactions: Query<(&Interaction, &mut BackgroundColor),
                            (Changed<Interaction>, With<CellDeletionButton>)>,
    mut session:      ResMut<SpeciesSession>,
) {
    if *mode != WindowMode::SpeciesEditor { return; }

    for (interaction, mut bg) in &mut interactions {
        match *interaction {
            Interaction::Pressed => {
                session.deletion_mode = !session.deletion_mode;
                *bg = BackgroundColor(DEL_BTN_HOVER);
            }
            Interaction::Hovered => *bg = BackgroundColor(DEL_BTN_HOVER),
            Interaction::None    => {
                *bg = BackgroundColor(if session.deletion_mode { DEL_BTN_ACTIVE } else { DEL_BTN_NORMAL });
            }
        }
    }
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
    let inactive = *mode != WindowMode::SpeciesEditor || !session.deletion_mode;
    if inactive {
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

    // Nearest RENDERED cell to the cursor. `ci % n` maps the combined (mirrored)
    // index back to the right-half OCG entry the deletion removes.
    let mut best: Option<(f32, usize, usize, Vec3)> = None; // (d², part, ocg_index, world)
    for (pi, part) in session.body_parts.iter().enumerate() {
        if part.ocg.is_empty() { continue; }
        let n = part.ocg.len();
        for (ci, entry) in session.combined_ocg(&part.ocg).iter().enumerate() {
            let world = SPECIES_EDITOR_ORIGIN + entry.1;
            let Ok(vp_phys) = camera.world_to_viewport(cam_xf, world) else { continue };
            let vp = vp_phys * inv_scale;
            let dx = vp.x - cursor.x;
            let dy = vp.y - cursor.y;
            let d2 = dx * dx + dy * dy;
            if best.map_or(true, |(b, ..)| d2 < b) {
                best = Some((d2, pi, ci % n, world));
            }
        }
    }

    let hovered = match best {
        Some((d2, pi, oi, world)) if d2.sqrt() <= PICK_RADIUS_PX => Some((pi, oi, world)),
        _ => None,
    };

    match hovered {
        Some((pi, oi, world)) => {
            target.target = Some((pi, oi));
            if let Ok((_, mut t, mut v)) = hl_q.single_mut() {
                t.translation = world;
                *v = Visibility::Inherited;
            } else {
                // Single-cell overlay, scaled up slightly to cover the
                // underlying cell (avoids z-fighting), unlit for clean red.
                let mesh = meshes.add(build_uncolored_mesh_from_ocg(&[(0, Vec3::ZERO, CellType::NonPhoto)]));
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
    if *mode != WindowMode::SpeciesEditor || !session.deletion_mode { return; }
    if !mouse.just_pressed(MouseButton::Left) { return; }
    // Don't delete on a click that a UI button claimed (e.g. the toggle).
    if ui_interactions.iter().any(|i| matches!(i, Interaction::Pressed)) { return; }

    let Some((pi, oi)) = target.target else { return };
    let Some(part) = session.body_parts.get_mut(pi) else { return };
    if oi >= part.ocg.len() { return; }

    part.ocg.remove(oi);
    // Re-number to contiguous 0..N (the growth / bilateral pipelines require it).
    for (i, e) in part.ocg.iter_mut().enumerate() { e.0 = i; }
    session.dirty = true;
}
