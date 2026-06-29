// Map editor — show the bare map, hide everything else, reversibly.
//
// On a mode change this system does three jobs (all keyed on `mode.is_changed()`
// so it runs once per transition, not per frame):
//   1. Flip every map-editor panel's `Display` (Flex in MapEditor, None else).
//   2. Hide / restore organisms — recursing the `OrganismRoot` AND all its
//      descendants. Body-part meshes are spawned `Visibility::Visible` (NOT
//      `Inherited`), so toggling the root alone is a no-op; the recursion also
//      covers the markerless translucent overlay meshes.
//   3. Hide / restore the water plane.
//
// The sim is PAUSED inside the editor (no spawns), so a blanket restore-to-
// `Visible` on exit is safe and matches the descendants' original spawn state.

use bevy::prelude::*;

use crate::organism::OrganismRoot;
use crate::simulation_settings::WindowMode;
use crate::water::WaterPlane;

use super::MapEditorPanel;


/// Toggle panel `Display` + organism/water `Visibility` on a mode change.
///
/// A SINGLE `&mut Visibility` query handles both organisms and the water plane —
/// adding a second `&mut Visibility` query (even with a disjoint `With<>`) risks
/// a B0001 archetype overlap at runtime, so the water entities are fed in via a
/// read-only `Query<Entity, With<WaterPlane>>`.
pub fn toggle_map_editor_visuals(
    mode:        Res<WindowMode>,
    children_q:  Query<&Children>,
    roots:       Query<Entity, With<OrganismRoot>>,
    water_ents:  Query<Entity, With<WaterPlane>>,
    mut vis_q:   Query<&mut Visibility>,
    mut panels:  Query<&mut Node, With<MapEditorPanel>>,
) {
    if !mode.is_changed() { return; }

    let entering = *mode == WindowMode::MapEditor;
    let target   = if entering { Visibility::Hidden } else { Visibility::Visible };

    // 1. Panels.
    let disp = if entering { Display::Flex } else { Display::None };
    for mut n in &mut panels {
        if n.display != disp { n.display = disp; }
    }

    // 2. Organisms: root + ALL descendants (children spawn explicit `Visible`,
    //    so a root-only toggle is a no-op; recursion also covers the markerless
    //    translucent overlay meshes).
    for root in &roots {
        let mut stack = vec![root];
        while let Some(e) = stack.pop() {
            if let Ok(mut v) = vis_q.get_mut(e) { *v = target; }
            if let Ok(ch) = children_q.get(e) { stack.extend(ch.iter()); }
        }
    }

    // 3. Water plane (root entities, fed in read-only to keep ONE `&mut Visibility`).
    for e in &water_ents {
        if let Ok(mut v) = vis_q.get_mut(e) { *v = target; }
    }
}


/// Keep the water plane hidden for as long as the Map Editor is open.
///
/// `toggle_map_editor_visuals` hides the water once, on the mode change — but the
/// water plane spawns asynchronously (gated on `HeightmapSampler`), so on the
/// `--setup` path that boots straight into the Map Editor the plane can spawn
/// (visible) AFTER that one-shot toggle already ran, leaving it on screen. This
/// runs every frame, but early-returns instantly outside the Map Editor and only
/// writes on a mismatch, so it is effectively free. Restore-to-`Visible` on exit
/// stays with `toggle_map_editor_visuals`.
pub fn enforce_water_hidden_in_map_editor(
    mode:    Res<WindowMode>,
    mut water_q: Query<&mut Visibility, With<WaterPlane>>,
) {
    if *mode != WindowMode::MapEditor { return; }
    for mut v in &mut water_q {
        if *v != Visibility::Hidden { *v = Visibility::Hidden; }
    }
}
