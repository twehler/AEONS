// Map-click handlers.
//
// Two click semantics:
//
//   * LEFT click on the map → spawn a NEW organism at the heightmap
//     hit point with the session's current draft (Metabolism /
//     Intelligence / Symmetry). The new template becomes the active
//     selection.
//
//   * RIGHT click on an organism → delete it. The click is hit-tested
//     against every template's visual sphere; if the ray pierces one
//     the closest hit is removed from the session and its mesh entity
//     is despawned. Clicks that don't hit any organism are silently
//     ignored — there's no "right-click on the ground" action.
//
// Left-click events come in as `ViewportClick` messages emitted by
// `camera.rs`, which distinguishes a clean LMB tap from a camera-
// look drag using the same physical button.

use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::camera::{EditorCamera, ViewportClick};
use crate::colony_editor::session::EditorSession;
use crate::colony_editor::template::OrganismTemplate;
use crate::colony_editor::template_marker::EditorTemplateMarker;
use crate::volumetric_growth::build_smoothed_mesh_from_ocg;
use crate::world_geometry::{HeightmapSampler, HEIGHTMAP_CELL_SIZE, MAP_MAX_X, MAP_MAX_Z};


/// Maximum march distance along the camera ray. Past this we give up
/// and treat the click as missing the world. Generous enough to cover
/// even the corner-to-corner diagonal of a 2048² map from above.
const MAX_RAY_DISTANCE: f32 = 4_000.0;
/// March step size (world units). Half the heightmap cell size to
/// guarantee we never skip across a single column without sampling.
const STEP_SIZE:        f32 = HEIGHTMAP_CELL_SIZE * 0.5;


pub struct PlacementPlugin;

impl Plugin for PlacementPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (handle_right_click, handle_left_click)
                .run_if(resource_exists::<HeightmapSampler>),
        );
    }
}


// ── Right-click: delete the organism under the cursor ──────────────────────

fn handle_right_click(
    buttons:     Res<ButtonInput<MouseButton>>,
    windows:     Query<&Window, With<PrimaryWindow>>,
    cam_q:       Query<(&Camera, &GlobalTransform), With<EditorCamera>>,
    mut session: ResMut<EditorSession>,
    mut commands: Commands,
) {
    if !buttons.just_pressed(MouseButton::Right) { return; }

    // Suppress right-click when the modal is open or when the cursor
    // is over a UI panel — same intent as the LMB suppression in
    // `camera.rs`. We don't want a stray right-click in the inventory
    // panel deleting the last-clicked organism.
    if session.show_exit_modal { return; }
    let Ok(window)      = windows.single() else { return };
    let Some(cursor_px) = window.cursor_position() else { return };
    if crate::camera::cursor_over_ui_panel_for_test(cursor_px, window) { return; }

    let Ok((camera, cam_xf)) = cam_q.single() else { return };
    let Ok(ray) = camera.viewport_to_world(cam_xf, cursor_px) else { return };
    let dir = ray.direction;

    // Closest-hit pick across every template. Returns `(index, t)`
    // where `t` is the ray parameter of the entry point — smaller t
    // means closer to the camera.
    let mut best: Option<(usize, f32)> = None;
    for (i, tpl) in session.templates.iter().enumerate() {
        if let Some(t) = ray_hits_sphere(ray.origin, *dir, tpl.position, tpl.pick_radius()) {
            best = match best {
                Some((_, bt)) if bt <= t => best,
                _ => Some((i, t)),
            };
        }
    }

    let Some((idx, _)) = best else { return };
    let removed = session.templates.swap_remove(idx);
    commands.entity(removed.entity).despawn();
    if session.active_id == Some(removed.id) {
        session.active_id = None;
    }
    session.dirty = true;
}


/// Standard ray-vs-sphere intersection. Returns the smaller of the
/// two roots when the ray pierces the sphere (or grazes it), else
/// `None`. Skips intersections behind the camera.
fn ray_hits_sphere(
    origin:    Vec3,
    direction: Vec3,
    centre:    Vec3,
    radius:    f32,
) -> Option<f32> {
    let oc  = centre - origin;
    let t_c = oc.dot(direction);
    let d2  = oc.length_squared() - t_c * t_c;
    let r2  = radius * radius;
    if d2 > r2 { return None; }
    let half_chord = (r2 - d2).sqrt();
    let t_near     = t_c - half_chord;
    let t_far      = t_c + half_chord;
    // Pick the nearest non-negative root (positive ⇒ in front of
    // camera). If both roots are negative, the sphere is behind us.
    if      t_near >= 0.0 { Some(t_near) }
    else if t_far  >= 0.0 { Some(t_far)  }
    else                  { None         }
}


// ── Left-click: create a new template at the surface point ─────────────────

fn handle_left_click(
    mut clicks:    MessageReader<ViewportClick>,
    cam_q:         Query<(&Camera, &GlobalTransform), With<EditorCamera>>,
    heightmap:     Res<HeightmapSampler>,
    mut session:   ResMut<EditorSession>,
    mut commands:  Commands,
    mut meshes:    ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Drain everything; only the most recent click is meaningful, but
    // we don't want stale events accumulating.
    let Some(click) = clicks.read().last().copied() else { return };

    // Suppress clicks that landed on a UI panel — those are panel
    // clicks the picking system should handle, not world clicks. We
    // don't have access to the panels' rects here, but the click
    // events from the camera only fire when LMB is RELEASED without
    // a drag, AND placement uses the cursor position at PRESS time.
    // Bevy's UI picking still consumes the press for buttons via
    // `Interaction`, so panel clicks won't reach here in practice.
    // What CAN reach here is a click on a region not covered by any
    // button — that's exactly what we want.

    let Ok((camera, cam_xf)) = cam_q.single() else { return };
    let Ok(ray) = camera.viewport_to_world(cam_xf, click.cursor) else { return };
    let Some(hit) = ray_hit_heightmap(ray.origin, *ray.direction, &heightmap) else { return };

    spawn_template_at(hit, &mut session, &mut commands, &mut meshes, &mut materials);
}


/// Push a new `OrganismTemplate` onto the session's list, spawn its
/// visual marker entity at `position`, and select it as the active
/// placement target.
///
/// The visual is the same rhombic-dodecahedron mesh the simulation
/// uses for an adult organism — built by feeding the template's OCG
/// (single cell or bilateral pair) through
/// `build_smoothed_mesh_from_ocg`. This lets the user see exactly
/// how the organism will look in-sim while choosing positions.
fn spawn_template_at(
    position:  Vec3,
    session:   &mut ResMut<EditorSession>,
    commands:  &mut Commands,
    meshes:    &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
) {
    let draft = session.draft;
    session.next_id += 1;
    let id = session.next_id;

    // Build the OCG (1 or 2 cells) and turn it into the same mesh
    // the simulation would render for an adult organism with this
    // body plan. `build_smoothed_mesh_from_ocg` runs the Jacobi
    // smoother once at construction time — no per-frame cost.
    let preview = OrganismTemplate {
        id,
        metabolism:   draft.metabolism,
        intelligence: draft.intelligence,
        symmetry:     draft.symmetry,
        form:         draft.form,
        position,
        // The entity field is filled in below; this placeholder is
        // never read because we only call `build_ocg` on it here.
        entity:       Entity::PLACEHOLDER,
    };
    let ocg = preview.build_ocg();
    let mesh    = meshes.add(build_smoothed_mesh_from_ocg(&ocg));
    let colour  = draft.metabolism.preview_colour();
    let material = materials.add(StandardMaterial {
        base_color: colour,
        ..default()
    });

    let entity = commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(material),
        Transform::from_translation(position),
        EditorTemplateMarker(id),
    )).id();

    session.templates.push(OrganismTemplate { entity, ..preview });
    session.active_id = Some(id);
    session.dirty     = true;
}


/// Forward-march the ray and return the first point at or below the
/// heightmap. Returns `None` when the ray never crosses the surface
/// within `MAX_RAY_DISTANCE` (e.g. the ray points up into the sky, or
/// horizontally past the map edge).
///
/// We only consider hits inside `[0, MAP_MAX_X] × [0, MAP_MAX_Z]` —
/// the heightmap clamps to its border outside this rect, so a hit
/// reported outside the map's footprint would feel wrong.
fn ray_hit_heightmap(
    origin:    Vec3,
    direction: Vec3,
    heightmap: &HeightmapSampler,
) -> Option<Vec3> {
    if direction.length_squared() < 1e-12 { return None; }
    let dir = direction.normalize();

    let mut t = 0.0_f32;
    let mut prev_above: Option<(Vec3, f32)> = None;

    while t < MAX_RAY_DISTANCE {
        let p = origin + dir * t;
        let in_bounds = (0.0..=MAP_MAX_X).contains(&p.x)
                     && (0.0..=MAP_MAX_Z).contains(&p.z);
        if in_bounds {
            let ground = heightmap.height_at(p.x, p.z);
            if p.y <= ground {
                if let Some((prev_p, prev_ground)) = prev_above {
                    let dy_prev = prev_p.y - prev_ground;
                    let dy_curr = p.y     - ground;
                    let span    = dy_prev - dy_curr;
                    let alpha = if span.abs() > 1e-6 { dy_prev / span } else { 0.5 };
                    let alpha = alpha.clamp(0.0, 1.0);
                    let interp = prev_p.lerp(p, alpha);
                    let final_y = heightmap.height_at(interp.x, interp.z) + 0.5;
                    return Some(Vec3::new(interp.x, final_y, interp.z));
                } else {
                    return Some(Vec3::new(p.x, ground + 0.5, p.z));
                }
            }
            prev_above = Some((p, ground));
        } else {
            prev_above = None;
        }
        t += STEP_SIZE;
    }
    None
}
