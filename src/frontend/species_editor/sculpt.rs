// Species editor — Sculpt Mode. A bulk-authoring brush: an LMB-drag fills (Add)
// or clears (Erase) a SPHERE of cells on the RD lattice, centred on the nearest
// rendered cell under the cursor. The bulk counterpart to one-cell `Addition`
// and the one-shot `Wrap` shell.
//
// - Centre: snaps to the nearest rendered cell of ANY part (the cursor projects
//   onto it) — `editor_mode::nearest_rendered_cell`.
// - Operation: Add (grows the ACTIVE part) or Erase (removes from EVERY part),
//   chosen by the tool-panel Add/Erase toggle.
// - Stroke: continuous — re-stamps every frame LMB is held; re-meshes via the
//   `session.dirty → refresh_species_mesh` path. The whole drag coalesces into
//   ONE undo entry (see `undo.rs` + the stroke-active flag here).
// - Radius unit: cell lengths (`SCULPT_CELL_LENGTH`). The on-screen ring is a
//   fixed-px cursor guide (`SCULPT_RING_PX`), decoupled from the cell radius.
//
// No save-format change: sculpted cells are ordinary OCG entries.

use bevy::input::keyboard::KeyboardInput;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use std::collections::HashSet;
use std::fmt::Write as _;

use crate::body_part::{bilateral_body_part_from_right_ocg, BILATERAL_MIDLINE_EPS};
use crate::cell::CELL_SPACING;
use crate::colony::Symmetry;
use crate::frontend::ViewportImage;
use crate::player_plugin::FlyCam;
use crate::simulation_settings::{
    SCULPT_CELL_LENGTH, SCULPT_MAX_CELLS_PER_STAMP, SCULPT_PICK_RADIUS_PX,
    SCULPT_RADIUS_CELLS_MAX, SCULPT_RADIUS_CELLS_MIN, WindowMode,
};
use crate::volumetric_growth::dodecahedron::{center_scale, SLOT_DIRS};
use crate::volumetric_growth::lattice_key;

use super::editor_mode::{
    nearest_rendered_cell, SculptRadiusInput, SculptRadiusText, SculptOpText, SculptOpToggle,
    SCULPT_FIELD_BG_FOCUSED, SCULPT_FIELD_BG_IDLE, SCULPT_OP_ADD_BG, SCULPT_OP_ERASE_BG,
};
use super::mesh_import::MeshImport;
use super::session::{SculptOp, SculptRadiusEditState, SpeciesSession};
use super::SPECIES_EDITOR_ORIGIN;


// ── Tunables ──────────────────────────────────────────────────────────────────

/// Max chars in the radius edit buffer.
const RADIUS_BUF_MAX: usize = 8;

/// EDGE_LEN the lattice uses (mirrors `volumetric_growth::EDGE_LEN`, private
/// there) — both derive from the master `GEOMETRY_SCALE`.
const EDGE_LEN: f32 = crate::simulation_settings::GEOMETRY_SCALE;

/// Two cells "coincide" (one in the way of the other) when closer than half the
/// lattice spacing — mirrors `wrap::COINCIDE_DIST`.
const COINCIDE_DIST: f32 = CELL_SPACING * 0.5;


// ── Radius field handler (mirrors gpu_paint::handle_brush_size_input) ──────────

/// Click + keyboard router for the sculpt "Brush radius (cell lengths)" field.
///   * LMB on box → focus, prefill buffer with the committed value.
///   * LMB outside while focused → commit and unfocus.
///   * Focused keys: Enter → commit; Escape → cancel; Backspace → del;
///     digit/'.' → append (bounded by `RADIUS_BUF_MAX`).
pub fn handle_sculpt_radius_input(
    mode:          Res<WindowMode>,
    mouse:         Res<ButtonInput<MouseButton>>,
    mut keyboard:  MessageReader<KeyboardInput>,
    interaction_q: Query<&Interaction, With<SculptRadiusInput>>,
    mut state:     ResMut<SculptRadiusEditState>,
    mut session:   ResMut<SpeciesSession>,
) {
    if *mode != WindowMode::SpeciesEditor {
        if state.focused { state.focused = false; state.buffer.clear(); }
        for _ in keyboard.read() {}
        return;
    }

    let click_on_input = mouse.just_pressed(MouseButton::Left)
        && interaction_q.iter().any(|i| matches!(i, Interaction::Pressed));
    let click_outside  = mouse.just_pressed(MouseButton::Left) && !click_on_input;

    if click_on_input && !state.focused {
        state.focused = true;
        state.buffer.clear();
        let _ = write!(state.buffer, "{:.1}", session.sculpt_radius_cells);
    }

    if click_outside && state.focused {
        commit_radius(&mut state, &mut session);
    }

    if !state.focused {
        for _ in keyboard.read() {}   // drain to avoid event-buffer growth
        return;
    }

    for ev in keyboard.read() {
        if !ev.state.is_pressed() { continue; }
        match ev.key_code {
            KeyCode::Enter | KeyCode::NumpadEnter => commit_radius(&mut state, &mut session),
            KeyCode::Escape => { state.focused = false; state.buffer.clear(); }
            KeyCode::Backspace => { state.buffer.pop(); }
            _ => {
                if let Some(text) = ev.text.as_ref() {
                    for c in text.chars() {
                        if state.buffer.len() >= RADIUS_BUF_MAX { break; }
                        if c.is_ascii_digit() {
                            state.buffer.push(c);
                        } else if c == '.' && !state.buffer.contains('.') {
                            state.buffer.push(c);
                        }
                    }
                }
            }
        }
    }
}

/// Parse + clamp the buffer into `sculpt_radius_cells`. Always unfocus + clear.
fn commit_radius(state: &mut SculptRadiusEditState, session: &mut SpeciesSession) {
    if let Ok(v) = state.buffer.parse::<f32>() {
        if v.is_finite() {
            session.sculpt_radius_cells =
                v.clamp(SCULPT_RADIUS_CELLS_MIN, SCULPT_RADIUS_CELLS_MAX);
        }
    }
    state.focused = false;
    state.buffer.clear();
}

/// Sync the radius field's text + background colour with state.
pub fn update_sculpt_radius_text(
    mode:       Res<WindowMode>,
    state:      Res<SculptRadiusEditState>,
    session:    Res<SpeciesSession>,
    mut text_q: Query<&mut Text, With<SculptRadiusText>>,
    mut bg_q:   Query<&mut BackgroundColor, With<SculptRadiusInput>>,
) {
    if *mode != WindowMode::SpeciesEditor { return; }
    if !state.is_changed() && !session.is_changed() { return; }

    let display = if state.focused {
        format!("{}_", state.buffer)
    } else {
        format!("{:.1}", session.sculpt_radius_cells)
    };
    for mut text in &mut text_q {
        if text.0 != display { text.0 = display.clone(); }
    }

    let bg = if state.focused { SCULPT_FIELD_BG_FOCUSED } else { SCULPT_FIELD_BG_IDLE };
    for mut b in &mut bg_q {
        if b.0 != bg { *b = BackgroundColor(bg); }
    }
}


// ── Add/Erase toggle ──────────────────────────────────────────────────────────

/// Flip `session.sculpt_op` when the Add/Erase button is pressed.
pub fn handle_sculpt_op_toggle(
    mode:         Res<WindowMode>,
    interactions: Query<&Interaction, (Changed<Interaction>, With<SculptOpToggle>)>,
    mut session:  ResMut<SpeciesSession>,
) {
    if *mode != WindowMode::SpeciesEditor { return; }
    for interaction in &interactions {
        if matches!(interaction, Interaction::Pressed) {
            session.sculpt_op = session.sculpt_op.toggle();
        }
    }
}

/// Sync the Add/Erase toggle's label + background tint with `sculpt_op`.
pub fn sync_sculpt_op_text(
    mode:       Res<WindowMode>,
    session:    Res<SpeciesSession>,
    mut text_q: Query<&mut Text, With<SculptOpText>>,
    mut bg_q:   Query<&mut BackgroundColor, With<SculptOpToggle>>,
) {
    if *mode != WindowMode::SpeciesEditor { return; }
    if !session.is_changed() { return; }

    let label = session.sculpt_op.label();
    for mut t in &mut text_q {
        if t.0 != label { t.0 = label.to_string(); }
    }
    let bg = match session.sculpt_op {
        SculptOp::Add   => SCULPT_OP_ADD_BG,
        SculptOp::Erase => SCULPT_OP_ERASE_BG,
    };
    for mut b in &mut bg_q {
        if b.0 != bg { *b = BackgroundColor(bg); }
    }
}


// ── Stroke-active flag (undo coalescing) ──────────────────────────────────────

/// `true` from LMB-press to LMB-release across a sculpt drag. While set, the undo
/// tracker (`track_species_undo`) skips per-frame snapshots so the WHOLE stroke
/// is one undo entry; the pre-stroke baseline is captured on `just_pressed`.
#[derive(Resource, Default)]
pub struct SculptStrokeActive(pub bool);


// ── Core stroke system ────────────────────────────────────────────────────────

/// Continuous LMB-drag sculpt. Re-stamps the sphere every frame the button is
/// held; mutates the OCG and sets `session.dirty` ONLY when a cell actually
/// changed (so a stationary hold doesn't re-mesh).
#[allow(clippy::too_many_arguments)]
pub fn apply_sculpt_stroke(
    mode:            Res<WindowMode>,
    mouse:           Res<ButtonInput<MouseButton>>,
    mesh_import:     Res<MeshImport>,
    radius_edit:     Res<SculptRadiusEditState>,
    mut stroke:      ResMut<SculptStrokeActive>,
    ui_interactions: Query<&Interaction>,
    cameras:         Query<(&Camera, &GlobalTransform), With<FlyCam>>,
    windows:         Query<&Window, With<PrimaryWindow>>,
    viewport_q:      Query<&bevy::ui::ComputedNode, With<ViewportImage>>,
    mut session:     ResMut<SpeciesSession>,
) {
    // Maintain the stroke-active flag regardless of mode/gates so a release is
    // never missed (the undo tracker keys off it).
    if mouse.just_pressed(MouseButton::Left) { stroke.0 = true; }
    if !mouse.pressed(MouseButton::Left)      { stroke.0 = false; }

    if *mode != WindowMode::SpeciesEditor { return; }
    if !session.is_sculpt() { return; }
    if mesh_import.active() { return; }
    if radius_edit.focused { return; }          // typing in the radius field
    if !mouse.pressed(MouseButton::Left) { return; }
    // A UI button claimed this frame's click — never sculpt through a panel.
    if ui_interactions.iter().any(|i| matches!(i, Interaction::Pressed)) { return; }

    let Ok((camera, cam_xf)) = cameras.single()       else { return };
    let Ok(window)           = windows.single()        else { return };
    let Ok(viewport_node)    = viewport_q.single()     else { return };
    let Some(cursor)         = window.cursor_position() else { return };
    let inv_scale = viewport_node.inverse_scale_factor;

    // 1. Centre = nearest rendered cell under the cursor (a real lattice point).
    let Some((_pi, _oi, world_pos, _ct)) =
        nearest_rendered_cell(&session, camera, cam_xf, cursor, inv_scale, SCULPT_PICK_RADIUS_PX)
    else { return };
    let center = world_pos - SPECIES_EDITOR_ORIGIN;

    // 2. World radius.
    let r_world = session.sculpt_radius_cells * SCULPT_CELL_LENGTH;

    // 3. Enumerate the lattice sphere (flood-fill over all 18 slot dirs).
    let sphere = lattice_sphere(center, r_world);
    if sphere.is_empty() { return; }

    // 4. Dispatch. Mutate through `bypass_change_detection` and flag the session
    //    changed ONLY when a cell actually changed — otherwise a stationary hold
    //    (or a drag over already-filled/empty lattice) would trip `is_changed`
    //    every frame and make `refresh_species_mesh` rebuild all meshes for
    //    nothing. (`get_mut`/`iter_mut` inside the helpers would otherwise mark
    //    the ResMut changed unconditionally.) Undo is unaffected: it diffs
    //    `body_parts` by value, not via change detection.
    let op = session.sculpt_op;
    let changed = match op {
        SculptOp::Add   => sculpt_add(session.bypass_change_detection(), center, &sphere),
        SculptOp::Erase => sculpt_erase(session.bypass_change_detection(), center, r_world),
    };
    if changed {
        session.set_changed(); // drives `refresh_species_mesh` this frame
    }
    // 5. Re-mesh is automatic via the `is_changed` path when `changed`.
}


/// Flood-fill the RD lattice from `center`, collecting every lattice point within
/// `r_world`. Dedups float drift via `lattice_key`; bounded by
/// `SCULPT_MAX_CELLS_PER_STAMP` (logs + breaks if exceeded).
fn lattice_sphere(center: Vec3, r_world: f32) -> Vec<Vec3> {
    let step = center_scale(EDGE_LEN);
    let r2   = r_world * r_world;
    let r_exp2 = (r_world + step) * (r_world + step);

    let mut seen: HashSet<_> = HashSet::new();
    seen.insert(lattice_key(center));
    let mut queue: Vec<Vec3> = vec![center];
    let mut sphere: Vec<Vec3> = Vec::new();

    while let Some(p) = queue.pop() {
        if (p - center).length_squared() <= r2 {
            sphere.push(p);
        }
        for dir in SLOT_DIRS {
            let q = p + dir * step;
            if (q - center).length_squared() <= r_exp2 && seen.insert(lattice_key(q)) {
                queue.push(q);
            }
        }
        if seen.len() > SCULPT_MAX_CELLS_PER_STAMP {
            warn!(
                "sculpt: lattice-sphere flood exceeded {} points (radius too large); truncating",
                SCULPT_MAX_CELLS_PER_STAMP
            );
            break;
        }
    }
    sphere
}


/// ADD: grow the ACTIVE part with the selected cell type. Bilateral folds −X
/// points to the right half so the auto-mirror reproduces the left side; skips
/// occupied slots in the active part and slots colliding with other parts.
fn sculpt_add(session: &mut SpeciesSession, _center: Vec3, sphere: &[Vec3]) -> bool {
    let Some(ct) = session.selected_cell_type else { return false };
    let active = session.active_body_part;
    let bilateral = session.draft.symmetry == Symmetry::Bilateral;

    // Existing right-half occupancy of the active part.
    let mut occupied: HashSet<_> = session
        .body_parts
        .get(active)
        .map(|p| p.ocg.iter().map(|&(_, c, _)| lattice_key(c)).collect())
        .unwrap_or_default();

    // Cells of every OTHER part (collision skip), in the shared editor frame.
    let others: Vec<Vec3> = session.body_parts.iter().enumerate()
        .filter(|(i, _)| *i != active)
        .flat_map(|(_, p)| p.ocg.iter().map(|&(_, pos, _)| pos))
        .collect();
    let coincide2 = COINCIDE_DIST * COINCIDE_DIST;

    let Some(part) = session.body_parts.get_mut(active) else { return false };
    let mut added = false;
    for &p in sphere {
        // Bilateral: fold the left (−X) half onto the right; midline cells stay.
        let mut q = p;
        if bilateral && q.x < -BILATERAL_MIDLINE_EPS {
            q = Vec3::new(-q.x, q.y, q.z);
        }
        // Drop anything that still lands on the −X side of the mirror plane
        // (would be discarded by `mirror_right_to_left` anyway).
        if bilateral && q.x < -BILATERAL_MIDLINE_EPS { continue; }

        let key = lattice_key(q);
        if !occupied.insert(key) { continue; } // already in the active part
        if others.iter().any(|o| o.distance_squared(q) < coincide2) { continue; }

        let idx = part.ocg.len();
        part.ocg.push((idx, q, ct));
        added = true;
    }

    if added {
        // Keep indices contiguous 0..N (growth / bilateral pipelines need it).
        for (i, e) in part.ocg.iter_mut().enumerate() { e.0 = i; }
        // Bilateral: exercise the weld pipeline as a sanity check (result discarded).
        if bilateral {
            let part_ocg = part.ocg.clone();
            let _ = bilateral_body_part_from_right_ocg(&part_ocg);
        }
        session.dirty = true;
    }
    added
}


/// ERASE: remove cells inside the sphere from EVERY part. Bilateral also tests
/// each right-half cell's mirror, so erasing on the rendered left side removes the
/// right-half genome entry it mirrors from. Renumbers each modified part.
fn sculpt_erase(session: &mut SpeciesSession, center: Vec3, r_world: f32) -> bool {
    let bilateral = session.draft.symmetry == Symmetry::Bilateral;
    let r2 = r_world * r_world;
    let mut removed_any = false;

    for part in session.body_parts.iter_mut() {
        let before = part.ocg.len();
        part.ocg.retain(|&(_, p, _)| {
            let mut hit = (p - center).length_squared() <= r2;
            if bilateral && p.x > BILATERAL_MIDLINE_EPS {
                let m = Vec3::new(-p.x, p.y, p.z);
                hit |= (m - center).length_squared() <= r2;
            }
            !hit
        });
        if part.ocg.len() != before {
            // Renumber to contiguous 0..N (growth / bilateral pipelines need it).
            for (i, e) in part.ocg.iter_mut().enumerate() { e.0 = i; }
            removed_any = true;
        }
    }

    if removed_any {
        session.dirty = true;
    }
    removed_any
}
