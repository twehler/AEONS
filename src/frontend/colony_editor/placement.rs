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

use crate::camera::{CursorTopReservedPx, EditorCamera, ViewportClick};
use crate::cell::CellType;
use crate::colony::{root_body_part_from_ocg, spawn_organism, OrganismMaterials};
use crate::energy::MAX_ENERGY_PER_CELL;
use crate::organism::{OrganismKind, Symmetry};
use crate::player_plugin::FlyCam;
use crate::simulation_settings::Smoothing;
use crate::colony_editor::session::EditorSession;
use crate::colony_editor::template::OrganismTemplate;
use crate::colony_editor::template_marker::EditorTemplateMarker;
use crate::colony_editor::undo::{EditorAction, UndoStack};
use crate::organism::{Organism, OrganismRoot};
use crate::volumetric_growth::build_smoothed_mesh_from_ocg;
use crate::world_geometry::{HeightmapSampler, HEIGHTMAP_CELL_SIZE, MapSize};


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
                .run_if(resource_exists::<HeightmapSampler>)
                // In merged mode (WindowMode resource present) only fire
                // while EditColony is active; in standalone editor mode
                // (no WindowMode resource) always fire.
                .run_if(in_edit_mode_or_standalone),
        );
    }
}

fn in_edit_mode_or_standalone(mode: Option<Res<crate::simulation_settings::WindowMode>>) -> bool {
    match mode {
        Some(m) => *m == crate::simulation_settings::WindowMode::EditColony,
        None    => true,
    }
}


// ── Right-click: delete the organism under the cursor ──────────────────────

fn handle_right_click(
    buttons:        Res<ButtonInput<MouseButton>>,
    windows:        Query<&Window, With<PrimaryWindow>>,
    // Standalone editor uses `EditorCamera` (its own Camera3d
    // entity); merged-mode reuses `FlyCam` from the simulation. The
    // two markers never coexist on the same entity, so a single
    // query with `Or` filter finds the right one in either path.
    cam_q:          Query<(&Camera, &GlobalTransform), Or<(With<EditorCamera>, With<FlyCam>)>>,
    mut session:    ResMut<EditorSession>,
    mut commands:   Commands,
    mut undo_stack: ResMut<UndoStack>,
    top_reserved:   Option<Res<CursorTopReservedPx>>,
    viewport_q:     Query<(&bevy::ui::ComputedNode, &bevy::ui::UiGlobalTransform), With<crate::frontend::ViewportImage>>,
    organisms_q:    Query<(Entity, &GlobalTransform, &Organism), With<OrganismRoot>>,
) {
    if !buttons.just_pressed(MouseButton::Right) { return; }

    // Suppress right-click when the modal is open or when the cursor
    // is over a UI panel — same intent as the LMB suppression in
    // `camera.rs`. We don't want a stray right-click in the inventory
    // panel deleting the last-clicked organism.
    if session.show_exit_modal { return; }
    let Ok(window)      = windows.single() else { return };
    let Some(cursor_px) = window.cursor_position() else { return };
    let top_strip       = top_reserved.map(|r| r.0).unwrap_or(0.0);
    if crate::camera::cursor_over_ui_panel_with_top(cursor_px, window, top_strip) { return; }

    let Ok((camera, cam_xf)) = cam_q.single() else { return };
    let cursor_vp = adjust_cursor_to_viewport(cursor_px, &viewport_q);
    let Ok(ray) = camera.viewport_to_world(cam_xf, cursor_vp) else { return };
    let dir = ray.direction;

    // Closest-hit pick across BOTH editor templates (preview meshes
    // the user placed during this Edit-Colony session) AND live
    // simulation organisms. Whichever is closer to the camera wins —
    // a template stacked in front of a heterotroph deletes the
    // template; right-clicking a wild organism with no template in
    // the way deletes the organism. Both sets use a conservative
    // bounding sphere; near-perfect mesh accuracy isn't worth it for
    // a destructive verb the user can undo (templates) or just shrug
    // off and re-issue (organisms).
    #[derive(Clone, Copy)]
    enum Hit {
        Template { index: usize },
        Organism { entity: Entity },
    }

    let mut best: Option<(Hit, f32)> = None;
    let mut keep_closer = |h: Hit, t: f32| {
        best = match best {
            Some((_, bt)) if bt <= t => best,
            _ => Some((h, t)),
        };
    };

    for (i, tpl) in session.templates.iter().enumerate() {
        if let Some(t) = ray_hits_sphere(ray.origin, *dir, tpl.position, tpl.pick_radius()) {
            keep_closer(Hit::Template { index: i }, t);
        }
    }
    for (e, gxf, org) in &organisms_q {
        // Bounding radius is in the organism's local frame; the root
        // entity has no scale in the simulation, so adding it to the
        // world translation gives a tight enough sphere for pick
        // purposes. Cells extend at most `bounding_radius` from the
        // root, so the test is conservative.
        let radius = org.bounding_radius().max(1.0);
        if let Some(t) = ray_hits_sphere(ray.origin, *dir, gxf.translation(), radius) {
            keep_closer(Hit::Organism { entity: e }, t);
        }
    }

    match best {
        Some((Hit::Template { index }, _)) => {
            let removed = session.templates.swap_remove(index);
            commands.entity(removed.entity).despawn();
            if session.active_id == Some(removed.id) {
                session.active_id = None;
            }
            session.dirty = true;
            // Snapshot the removed template (including its trait set
            // and last-seen position) so Ctrl+Z can rebuild it.
            undo_stack.push(EditorAction::Deleted(removed));
        }
        Some((Hit::Organism { entity }, _)) => {
            // Despawning the root recursively drops every body-part
            // child + visual mesh entity. RemovedComponents observers
            // (brain-pool slot reclaim, statistics death counter,
            // auto-spawn-heteros) all fire automatically, so the
            // simulation stays internally consistent without us
            // poking each subsystem. No undo entry — wild organisms
            // aren't part of the editor's reversible-action model.
            commands.entity(entity).despawn();
        }
        None => {}
    }
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
    mut clicks:     MessageReader<ViewportClick>,
    // Standalone editor uses `EditorCamera` (its own Camera3d
    // entity); merged-mode reuses `FlyCam` from the simulation. The
    // two markers never coexist on the same entity, so a single
    // query with `Or` filter finds the right one in either path.
    cam_q:          Query<(&Camera, &GlobalTransform), Or<(With<EditorCamera>, With<FlyCam>)>>,
    heightmap:      Res<HeightmapSampler>,
    map_size:       Res<MapSize>,
    mut session:    ResMut<EditorSession>,
    mut commands:   Commands,
    mut meshes:     ResMut<Assets<Mesh>>,
    mut materials:  ResMut<Assets<StandardMaterial>>,
    mut undo_stack: ResMut<UndoStack>,
    viewport_q:     Query<(&bevy::ui::ComputedNode, &bevy::ui::UiGlobalTransform), With<crate::frontend::ViewportImage>>,
    org_materials:  Option<Res<OrganismMaterials>>,
    smoothing:      Option<Res<Smoothing>>,
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
    // In merged mode the 3D camera renders into the `ViewportImage`
    // texture rather than directly into the window. `viewport_to_world`
    // for an image-target camera expects cursor coordinates relative
    // to the IMAGE — so we subtract the image's top-left screen
    // position. In standalone editor mode the ViewportImage entity
    // doesn't exist (no frontend layout), `viewport_q` returns
    // empty, and we fall through to using the raw cursor (which
    // already IS the window — the camera renders directly to it).
    let cursor = adjust_cursor_to_viewport(click.cursor, &viewport_q);
    let Ok(ray) = camera.viewport_to_world(cam_xf, cursor) else { return };
    let Some(hit) = ray_hit_heightmap(ray.origin, *ray.direction, &heightmap, *map_size) else { return };

    // In merged mode `OrganismMaterials` is inserted by `spawn_colony`
    // once the heightmap has loaded; the editor only reaches this
    // point after the user has paused the running simulation, so the
    // resource is always present in merged mode. In standalone the
    // option stays `None` and the template falls back to a preview
    // mesh.
    let smoothing_on = smoothing.as_deref().map(|s| s.0).unwrap_or(true);
    // Species-driven placement: if no species is selected, the call
    // returns `None` and the click is a no-op. The Species Navigator
    // panel is where the user picks one.
    let Some(id) = spawn_template_at(
        hit,
        &mut session,
        &mut commands,
        &mut meshes,
        &mut materials,
        org_materials.as_deref(),
        smoothing_on,
    ) else { return; };
    undo_stack.push(EditorAction::Created(vec![id]));
}

/// If a `ViewportImage` is present (merged-mode editor), translate
/// `cursor_window` into image-local coordinates. Otherwise (standalone
/// editor) return it unchanged.
fn adjust_cursor_to_viewport(
    cursor_window: Vec2,
    viewport_q:    &Query<(&bevy::ui::ComputedNode, &bevy::ui::UiGlobalTransform), With<crate::frontend::ViewportImage>>,
) -> Vec2 {
    let Ok((node, ui_xf)) = viewport_q.single() else { return cursor_window };
    let inv_scale = node.inverse_scale_factor;
    let size      = node.size() * inv_scale;
    let centre    = ui_xf.translation * inv_scale;
    let top_left  = centre - size * 0.5;
    cursor_window - top_left
}


/// Create the visual entity for a given template snapshot.
///
/// Two modes:
///   * **Merged mode** (`org_materials` present): spawn a real,
///     fully-functional `Organism` via `colony::spawn_organism`. The
///     returned entity is the `OrganismRoot`, so it gets picked up
///     by movement, photosynthesis, energy, brains, statistics —
///     i.e. it actually lives in the simulation. Editing the colony
///     during a paused simulation and resuming it now produces real
///     evolution events.
///   * **Standalone editor** (`org_materials == None`): same legacy
///     preview-mesh entity as before — the editor's job there is to
///     author `.colony` files, not to run simulation systems.
///
/// `pub(super)` so the undo handler in `undo.rs` can reuse it to
/// restore a deleted template (the undo path re-inserts the
/// snapshot into `session.templates` itself, with a fresh entity
/// returned by this function).
pub(super) fn respawn_template(
    template:      &OrganismTemplate,
    commands:      &mut Commands,
    meshes:        &mut ResMut<Assets<Mesh>>,
    materials:     &mut ResMut<Assets<StandardMaterial>>,
    org_materials: Option<&OrganismMaterials>,
    smoothing:     bool,
) -> Entity {
    if let Some(org_mats) = org_materials {
        return spawn_real_organism(template, commands, meshes, org_mats, smoothing);
    }

    let ocg = template.build_ocg();
    // Use the uncolored mesh so the marker takes its `preview_colour`
    // material verbatim (no per-cell tinting in the fallback marker).
    let mesh    = meshes.add(crate::volumetric_growth::build_uncolored_mesh_from_ocg(&ocg));
    let colour  = template.metabolism.preview_colour();
    let material = materials.add(StandardMaterial {
        base_color: colour,
        ..default()
    });
    commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(material),
        Transform::from_translation(template.position),
        EditorTemplateMarker(template.id),
    )).id()
}

/// Merged-mode helper: translate the editor's `OrganismTemplate` into
/// the parameter set `colony::spawn_organism` expects. The resulting
/// entity is an `OrganismRoot` with the trophic marker, brain-pool
/// hookups, and body-part mesh hierarchy — indistinguishable from a
/// reproduced or loaded organism. Initial energy is half the max
/// capacity (same convention as `.colony` save records and the
/// initial colony cohort), so the new organism doesn't immediately
/// starve.
/// Build a STATIC appendage `BodyPart` from a full OCG (already mirrored
/// if needed). The entity sits at the organism's local origin and cells
/// render at their authored absolute positions — no per-frame rotation.
/// Use `limb_body_part` instead when the appendage is flagged as a Limb.
fn appendage_body_part(ocg: Vec<(usize, Vec3, CellType)>, parent_idx: usize) -> crate::cell::BodyPart {
    let cells = ocg.iter()
        .map(|(_, p, ct)| crate::cell::Cell::new(*p, *ct))
        .collect();
    crate::cell::BodyPart {
        kind:         crate::cell::BodyPartKind::Organ,
        local_offset: Vec3::ZERO,
        cells,
        ocg,
        attachment:   Some(crate::body_part::Attachment {
            parent_idx,
            origin_local: Vec3::ZERO,
            rotation:     Quat::IDENTITY,
        }),
        consumed:     false,
        debug_blue:   false,
        regrowable:   true,
    }
}

/// Build a LIMB `BodyPart` from a full OCG: the entity is positioned at
/// the first cell of the appendage and the cells are rebased relative
/// to that point. Rotating the entity then rotates the limb around the
/// first cell (the attachment seed), exactly as the species-editor
/// "Limb" toggle promises. Tagged `BodyPartKind::Limb` so
/// `spawn_organism` attaches the `LimbAnimation` marker.
fn limb_body_part(ocg: Vec<(usize, Vec3, CellType)>, parent_idx: usize) -> crate::cell::BodyPart {
    let pivot = ocg.first().map(|(_, p, _)| *p).unwrap_or(Vec3::ZERO);
    let shifted_ocg: Vec<(usize, Vec3, CellType)> = ocg.iter()
        .map(|(i, p, ct)| (*i, *p - pivot, *ct))
        .collect();
    let cells = shifted_ocg.iter()
        .map(|(_, p, ct)| crate::cell::Cell::new(*p, *ct))
        .collect();
    crate::cell::BodyPart {
        kind:         crate::cell::BodyPartKind::Limb,
        local_offset: Vec3::ZERO,
        cells,
        ocg:          shifted_ocg,
        attachment:   Some(crate::body_part::Attachment {
            parent_idx,
            origin_local: pivot,
            rotation:     Quat::IDENTITY,
        }),
        consumed:     false,
        debug_blue:   false,
        regrowable:   true,
    }
}

fn spawn_real_organism(
    template:  &OrganismTemplate,
    commands:  &mut Commands,
    meshes:    &mut ResMut<Assets<Mesh>>,
    org_mats:  &OrganismMaterials,
    smoothing: bool,
) -> Entity {
    let ocg = template.build_ocg();
    // Base body (part 0, attachment None). Appendage parts attach to it
    // (parent_idx 0). For bilateral species each appendage is a mirrored
    // pair of runtime parts; for NoSymmetry it is one part.
    let mut body_parts = vec![root_body_part_from_ocg(&ocg)];
    // Map an EDITOR body-part index (0 = base, a+1 = appendage `a`) to its
    // runtime body-part index. NoSymmetry is 1:1; Bilateral expands each
    // appendage into a right+left pair, so we track each side separately so
    // a sub-limb's right half attaches to its parent's right half (and left
    // to left). A sub-limb is always authored AFTER its parent, so the
    // parent's runtime indices already exist when we reach the sub-limb.
    let make = |o: Vec<(usize, Vec3, CellType)>, is_limb: bool, parent_idx: usize| -> crate::cell::BodyPart {
        if is_limb { limb_body_part(o, parent_idx) } else { appendage_body_part(o, parent_idx) }
    };
    match template.symmetry {
        Symmetry::NoSymmetry => {
            // Editor index e → runtime index e (base 0, then appendages in order).
            for (app_raw, is_limb, parent) in &template.custom_appendages {
                body_parts.push(make(app_raw.clone(), *is_limb, *parent));
            }
        }
        Symmetry::Bilateral => {
            // runtime[e] right/left indices for each editor index.
            let mut right_of: Vec<usize> = vec![0]; // editor 0 (base) → runtime 0
            let mut left_of:  Vec<usize> = vec![0];
            for (app_raw, is_limb, parent) in &template.custom_appendages {
                let p_right = right_of[*parent];
                let p_left  = left_of[*parent];
                let r_idx = body_parts.len();
                body_parts.push(make(app_raw.clone(), *is_limb, p_right));
                let l_idx = body_parts.len();
                body_parts.push(make(crate::body_part::mirror_right_to_left(app_raw), *is_limb, p_left));
                right_of.push(r_idx);
                left_of.push(l_idx);
            }
        }
    }
    let kind = match template.metabolism {
        crate::colony_editor::template::Metabolism::Photoautotroph => OrganismKind::Photoautotroph,
        crate::colony_editor::template::Metabolism::Heterotroph    => OrganismKind::Heterotroph,
    };
    // Cell count drives the half-tank initial-energy budget — sum across
    // every body part so multi-part species get a proportionate tank.
    let cell_count = body_parts.iter().map(|bp| bp.cells.len()).sum::<usize>() as f32;
    let _ = CellType::Photo; // silence unused-import lint without removing the symbol
    let initial_energy = cell_count * MAX_ENERGY_PER_CELL * 0.5;
    let mut rng = rand::rng();
    let entity = spawn_organism(
        template.position,
        body_parts,
        kind,
        template.symmetry,
        template.has_variable_form(),
        template.is_sessile(),
        template.intelligence,
        smoothing,
        initial_energy,
        // Movement paradigm from the species-editor toggle, threaded
        // through `OrganismTemplate::sliding_movement`. Legacy /
        // cycler-derived templates set this to `true` (sliding);
        // limb-based species set it to `false`, routing the spawn
        // into the Avian dynamic-body path in `spawn_organism`.
        template.sliding_movement,
        commands,
        meshes,
        org_mats,
        &mut rng,
    );
    if template.is_carnivore {
        commands.entity(entity).try_insert(crate::colony::Carnivore);
    }
    // Tag organisms that originated from a `.species` file so the
    // speciation system treats them as a fresh founder lineage rather
    // than classifying them into the nearest existing species. The
    // marker carries the filename stem (e.g. "herbivore1"), which the
    // registry then uses as the species' display name — the floating
    // label sub-line reads `Species::name` directly so this is what
    // the user sees in the viewport.
    if let Some(ref species_name) = template.species_name {
        commands.entity(entity).try_insert(
            crate::lineages::species::ImportedSpeciesOrigin {
                name: species_name.clone(),
            },
        );
    }
    entity
}

/// Push a new `OrganismTemplate` (built from the current draft) onto
/// the session's list, spawn its visual marker entity at `position`,
/// and select it as the active placement target. Returns the new
/// template's id so callers can record a `Created` entry on the
/// undo stack.
///
/// `pub(super)` so the tool panel's Bulk-Add button can call this
/// from its sibling module — left-click placement and bulk-add
/// share the exact same per-template spawn pipeline.
pub(super) fn spawn_template_at(
    position:      Vec3,
    session:       &mut ResMut<EditorSession>,
    commands:      &mut Commands,
    meshes:        &mut ResMut<Assets<Mesh>>,
    materials:     &mut ResMut<Assets<StandardMaterial>>,
    org_materials: Option<&OrganismMaterials>,
    smoothing:     bool,
) -> Option<u32> {
    // Placement is species-driven: every spawn requires the user to
    // have a `.species` file selected in the navigator. The four
    // trait cyclers that used to drive a default-shape fallback
    // were retired together with the bottom panel's button strip.
    let species_id = session.selected_species_id?;
    let species = session.loaded_species.iter().find(|s| s.id == species_id).cloned()?;
    Some(spawn_species_template_at(
        position, &species, session, commands, meshes, materials, org_materials, smoothing,
    ))
}

/// Spawn one instance of a loaded species at `position`. The
/// resulting `OrganismTemplate` carries the species's full OCG and
/// its body-plan flags verbatim. Per-organism hyperparameter genes
/// (curiosity, K_EAT, σ, …) are NOT set here — the brain pool's
/// `assign_brains_l1_hetero` system samples them from
/// `simulation_settings.rs`'s `L1_*_RANGE` ranges when the slot is
/// assigned next PreUpdate, so every species spawn produces a unique
/// gene set.
pub(super) fn spawn_species_template_at(
    position:      Vec3,
    species:       &crate::colony_editor::session::LoadedSpecies,
    session:       &mut ResMut<EditorSession>,
    commands:      &mut Commands,
    meshes:        &mut ResMut<Assets<Mesh>>,
    materials:     &mut ResMut<Assets<StandardMaterial>>,
    org_materials: Option<&OrganismMaterials>,
    smoothing:     bool,
) -> u32 {
    session.next_id += 1;
    let id = session.next_id;

    let template = OrganismTemplate {
        id,
        metabolism:   species.metabolism,
        intelligence: species.intelligence,
        symmetry:     species.symmetry,
        form:         species.form,
        position,
        entity:       Entity::PLACEHOLDER,
        custom_ocg:   Some(species.ocg.clone()),
        custom_appendages: species.appendages.clone(),
        species_name: Some(species.name.clone()),
        is_carnivore: species.is_carnivore,
        sliding_movement: species.sliding_movement,
        is_sessile:   species.is_sessile,
    };
    let entity = respawn_template(&template, commands, meshes, materials, org_materials, smoothing);

    // If the loaded species carried a trained brain, attach a copy
    // of the restore payload to the freshly-spawned organism. The
    // herbivore pool's `assign_brains_herbivore_1` (running next
    // PreUpdate) sees the component, allocates a slot, restores the
    // saved weights into that slot, and removes the component. All
    // bulk-spawned instances get the SAME payload — they boot with
    // identical brains and diverge through training, as designed.
    if let Some(brain) = &species.brain {
        commands.entity(entity).try_insert(brain.clone());
    }

    session.templates.push(OrganismTemplate { entity, ..template });
    session.active_id = Some(id);
    session.dirty     = true;
    id
}


/// Forward-march the ray and return the first point at or below the
/// heightmap. Returns `None` when the ray never crosses the surface
/// within `MAX_RAY_DISTANCE` (e.g. the ray points up into the sky, or
/// horizontally past the map edge).
///
/// We only consider hits inside `[0, map_size.x] × [0, map_size.z]` —
/// the heightmap clamps to its border outside this rect, so a hit
/// reported outside the map's footprint would feel wrong.
fn ray_hit_heightmap(
    origin:    Vec3,
    direction: Vec3,
    heightmap: &HeightmapSampler,
    map_size:  MapSize,
) -> Option<Vec3> {
    if direction.length_squared() < 1e-12 { return None; }
    let dir = direction.normalize();

    let mut t = 0.0_f32;
    let mut prev_above: Option<(Vec3, f32)> = None;

    while t < MAX_RAY_DISTANCE {
        let p = origin + dir * t;
        let in_bounds = (0.0..=map_size.x).contains(&p.x)
                     && (0.0..=map_size.z).contains(&p.z);
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
