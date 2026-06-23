// Species editor — "Wrap" tool. One-shot action (a button in the tool panel):
// coats the ACTIVE body part in a single-cell shell of the currently-selected
// cell type. A shell cell that would land on top of ANOTHER body part's cell is
// skipped, so wrapping never overlaps neighbouring parts.
//
// The shell positions are the active part's empty FACE-adjacent lattice slots
// (`face_adjacent_centers_for_ocg`) — a single, uniform one-cell coating. (The
// full frontier also includes the 6 axis-aligned next-nearest slots, a √2
// farther out; adding those too made the wrap look two layers thick.)

use bevy::prelude::*;

use crate::cell::CELL_SPACING;
use crate::colony::Symmetry;
use crate::simulation_settings::WindowMode;
use crate::volumetric_growth::face_adjacent_centers_for_ocg;

use super::session::SpeciesSession;


// ── Tunables ──────────────────────────────────────────────────────────────────

const WRAP_BTN_BG:    Color = Color::srgb(0.20, 0.45, 0.40);
const WRAP_BTN_HOVER: Color = Color::srgb(0.28, 0.58, 0.50);

/// Two cells "coincide" (one is in the way of the other) when closer than half
/// the lattice spacing — distinct lattice slots are a full `CELL_SPACING` apart,
/// so this cleanly separates "same slot" from "adjacent slot".
const COINCIDE_DIST: f32 = CELL_SPACING * 0.5;


// ── Marker ──────────────────────────────────────────────────────────────────

#[derive(Component)]
pub struct WrapButton;


// ── Spawn (called from `editor_mode::spawn_tool_panel`) ───────────────────────

pub fn spawn_wrap_button(panel: &mut ChildSpawnerCommands) {
    panel
        .spawn((
            WrapButton,
            Button,
            Node {
                width:           Val::Percent(100.0),
                height:          Val::Px(30.0),
                align_items:     AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            BackgroundColor(WRAP_BTN_BG),
        ))
        .with_children(|b| {
            b.spawn((
                Text::new("Wrap"),
                TextFont { font_size: 13.0, ..default() },
                TextColor(Color::WHITE),
                Pickable::IGNORE,
            ));
        });
}


// ── Click handler ─────────────────────────────────────────────────────────────

pub fn handle_wrap_click(
    mode:             Res<WindowMode>,
    mut interactions: Query<(&Interaction, &mut BackgroundColor),
                            (Changed<Interaction>, With<WrapButton>)>,
    mut session:      ResMut<SpeciesSession>,
) {
    if *mode != WindowMode::SpeciesEditor { return; }

    for (interaction, mut bg) in &mut interactions {
        match *interaction {
            Interaction::Pressed => {
                wrap_active_part(&mut session);
                *bg = BackgroundColor(WRAP_BTN_HOVER);
            }
            Interaction::Hovered => *bg = BackgroundColor(WRAP_BTN_HOVER),
            Interaction::None    => *bg = BackgroundColor(WRAP_BTN_BG),
        }
    }
}


// ── Wrap action ─────────────────────────────────────────────────────────────

/// Add a one-cell shell of `selected_cell_type` around the active body part,
/// skipping shell positions that collide with another part. No-op when no cell
/// type is selected or the active part has no cells.
fn wrap_active_part(session: &mut SpeciesSession) {
    let Some(ct) = session.selected_cell_type else { return };
    let active = session.active_body_part;
    let Some(part) = session.body_parts.get(active) else { return };
    if part.ocg.is_empty() { return; }

    // Empty FACE-adjacent slots of the active part = the one-cell shell. Bilateral
    // keeps the right half + midline (x >= 0); the left half is mirrored at render.
    let min_x = match session.draft.symmetry {
        Symmetry::Bilateral  => Some(0.0),
        Symmetry::NoSymmetry => None,
    };
    let candidates = face_adjacent_centers_for_ocg(&part.ocg, min_x);
    if candidates.is_empty() { return; }

    // Cells of every OTHER part, in the shared editor frame — a shell cell that
    // coincides with any of these is "in the way" and gets skipped.
    let others: Vec<Vec3> = session.body_parts.iter().enumerate()
        .filter(|(i, _)| *i != active)
        .flat_map(|(_, p)| p.ocg.iter().map(|&(_, pos, _)| pos))
        .collect();
    let coincide2 = COINCIDE_DIST * COINCIDE_DIST;

    let part = session.body_parts.get_mut(active).expect("active part exists");
    let mut added = false;
    for c in candidates {
        if others.iter().any(|o| o.distance_squared(c) < coincide2) { continue; }
        let idx = part.ocg.len();
        part.ocg.push((idx, c, ct));
        added = true;
    }

    if added {
        // Keep indices contiguous 0..N (the growth / bilateral pipelines need it).
        for (i, e) in part.ocg.iter_mut().enumerate() { e.0 = i; }
        session.dirty = true;
    }
}
