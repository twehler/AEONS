// Map-click handlers.
//
//   * LEFT click on the map → spawn a NEW organism at the heightmap hit
//     point with the current draft; the new template becomes the active
//     selection. Left-click events arrive as `ViewportClick` messages
//     from `camera.rs` (a clean LMB tap, distinguished from a look-drag).
//   * RIGHT click → closest-hit delete of a template or live organism
//     (bounding-sphere ray test). A miss is a no-op.

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


/// Max march distance along the camera ray; past this the click misses
/// the world. Covers the diagonal of a 2048² map viewed from above.
const MAX_RAY_DISTANCE: f32 = 4_000.0;
/// March step (world units). Half the heightmap cell size so we never
/// skip a column without sampling it.
const STEP_SIZE:        f32 = HEIGHTMAP_CELL_SIZE * 0.5;


pub struct PlacementPlugin;

impl Plugin for PlacementPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (handle_right_click, handle_left_click)
                .run_if(resource_exists::<HeightmapSampler>)
                // Merged mode: only while EditColony active. Standalone
                // (no WindowMode resource): always.
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
    // Standalone uses `EditorCamera`; merged-mode reuses sim `FlyCam`.
    // Never coexist, so one `Or`-filter query finds the right one.
    cam_q:          Query<(&Camera, &GlobalTransform), Or<(With<EditorCamera>, With<FlyCam>)>>,
    mut session:    ResMut<EditorSession>,
    mut commands:   Commands,
    mut undo_stack: ResMut<UndoStack>,
    top_reserved:   Option<Res<CursorTopReservedPx>>,
    viewport_q:     Query<(&bevy::ui::ComputedNode, &bevy::ui::UiGlobalTransform), With<crate::frontend::ViewportImage>>,
    organisms_q:    Query<(Entity, &GlobalTransform, &Organism), With<OrganismRoot>>,
) {
    if !buttons.just_pressed(MouseButton::Right) { return; }

    // Suppress while a modal is open or the cursor is over a UI panel —
    // a stray panel right-click must not delete an organism.
    if session.show_exit_modal { return; }
    let Ok(window)      = windows.single() else { return };
    let Some(cursor_px) = window.cursor_position() else { return };
    let top_strip       = top_reserved.map(|r| r.0).unwrap_or(0.0);
    if crate::camera::cursor_over_ui_panel_with_top(cursor_px, window, top_strip) { return; }

    let Ok((camera, cam_xf)) = cam_q.single() else { return };
    let cursor_vp = adjust_cursor_to_viewport(cursor_px, &viewport_q);
    let Ok(ray) = camera.viewport_to_world(cam_xf, cursor_vp) else { return };
    let dir = ray.direction;

    // Closest-hit pick across BOTH editor templates and live organisms;
    // whichever is nearer the camera wins. Conservative bounding sphere —
    // exact mesh accuracy isn't worth it for an undoable/re-issuable verb.
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
        // Root has no scale, so local bounding_radius about its world
        // translation is a conservative pick sphere (cells extend at
        // most that far from the root).
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
            // Snapshot so Ctrl+Z can rebuild it.
            undo_stack.push(EditorAction::Deleted(removed));
        }
        Some((Hit::Organism { entity }, _)) => {
            // Recursive despawn drops body-part children + mesh;
            // RemovedComponents observers (brain-slot reclaim, stats,
            // auto-spawn) fire automatically. No undo entry — wild
            // organisms aren't in the editor's reversible model.
            commands.entity(entity).despawn();
        }
        None => {}
    }
}


/// Ray-vs-sphere intersection; nearest non-negative root (in front of
/// the camera), else `None`.
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
    // Nearest non-negative root; both negative ⇒ sphere behind camera.
    if      t_near >= 0.0 { Some(t_near) }
    else if t_far  >= 0.0 { Some(t_far)  }
    else                  { None         }
}


// ── Left-click: create a new template at the surface point ─────────────────

fn handle_left_click(
    mut clicks:     MessageReader<ViewportClick>,
    // Standalone uses `EditorCamera`; merged-mode reuses sim `FlyCam`.
    // Never coexist, so one `Or`-filter query finds the right one.
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
    // Only the most recent click matters; drain the rest.
    let Some(click) = clicks.read().last().copied() else { return };

    // Panel clicks don't reach here: Bevy UI picking consumes button
    // presses via `Interaction`, and these events only fire on a
    // no-drag LMB release. Non-button regions fall through, as intended.

    let Ok((camera, cam_xf)) = cam_q.single() else { return };
    // Merged mode: 3D camera renders into the `ViewportImage` texture,
    // so cursor coords must be image-relative — `adjust_cursor_to_viewport`
    // subtracts the image's top-left. Standalone: no ViewportImage, raw
    // cursor IS the window coordinate.
    let cursor = adjust_cursor_to_viewport(click.cursor, &viewport_q);
    let Ok(ray) = camera.viewport_to_world(cam_xf, cursor) else { return };
    let Some(hit) = ray_hit_heightmap(ray.origin, *ray.direction, &heightmap, *map_size) else { return };

    // Merged mode: `OrganismMaterials` is always present (inserted by
    // `spawn_colony`). Standalone: `None`, falling back to a preview mesh.
    let smoothing_on = smoothing.as_deref().map(|s| s.0).unwrap_or(true);
    // Species-driven: no species selected ⇒ `None` ⇒ click is a no-op.
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


/// Create the visual entity for a template snapshot.
///   * Merged mode (`org_materials` present): spawn a real `Organism`
///     via `colony::spawn_organism` (returns the `OrganismRoot`).
///   * Standalone (`None`): a preview-mesh entity only.
/// `pub(super)` so `undo.rs` can reuse it to restore a deleted template.
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
    // Uncolored mesh so the marker takes its `preview_colour` verbatim.
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

/// Build an appendage `BodyPart` from a full OCG (already mirrored if needed).
/// REBASED to the first cell exactly like `limb_body_part` (entity at the
/// first-cell pivot, cells shifted relative to it): once every part became a
/// dynamic spherical-jointed body, an appendage left at `origin_local = ZERO`
/// with authored-position cells was pinned at the ROOT ORIGIN with its COM far
/// away — a large lever arm that spun it violently (the sub-limbed-swimmer
/// "flung into the air" bug). Rebasing pivots the joint at the contact point
/// and keeps the COM near the origin. Rendering is unchanged (mesh from the
/// shifted OCG, entity at the pivot). `kind = Organ` (vs `limb_body_part`'s
/// `Limb`) is the only remaining difference.
fn appendage_body_part(ocg: Vec<(usize, Vec3, CellType)>, parent_idx: usize) -> crate::cell::BodyPart {
    let pivot = ocg.first().map(|(_, p, _)| *p).unwrap_or(Vec3::ZERO);
    let shifted_ocg: Vec<(usize, Vec3, CellType)> = ocg.iter()
        .map(|(i, p, ct)| (*i, *p - pivot, *ct))
        .collect();
    let cells = shifted_ocg.iter()
        .map(|(_, p, ct)| crate::cell::Cell::new(*p, *ct))
        .collect();
    crate::cell::BodyPart {
        kind:         crate::cell::BodyPartKind::Organ,
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

/// Build a LIMB `BodyPart` from a full OCG: entity positioned at the
/// first cell, cells rebased relative to it, so rotating the entity
/// rotates the limb about that pivot. Tagged `BodyPartKind::Limb` so
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

/// Merged-mode helper: translate `OrganismTemplate` into the args
/// `colony::spawn_organism` expects, yielding an `OrganismRoot`
/// indistinguishable from a reproduced/loaded organism. Initial energy
/// is half-tank (same convention as save records / the initial cohort).
fn spawn_real_organism(
    template:  &OrganismTemplate,
    commands:  &mut Commands,
    meshes:    &mut ResMut<Assets<Mesh>>,
    org_mats:  &OrganismMaterials,
    smoothing: bool,
) -> Entity {
    let ocg = template.build_ocg();
    // Base body = part 0 (attachment None); appendages attach to it.
    let mut body_parts = vec![root_body_part_from_ocg(&ocg)];
    // Map EDITOR body-part index (0 = base, a+1 = appendage a) to runtime
    // index. NoSymmetry is 1:1; Bilateral expands each appendage into a
    // right+left pair, tracked separately so a sub-limb's right half
    // attaches to its parent's right (left to left). Sub-limbs are always
    // authored after their parent, so parent indices exist by then.
    let make = |o: Vec<(usize, Vec3, CellType)>, is_limb: bool, parent_idx: usize| -> crate::cell::BodyPart {
        if is_limb { limb_body_part(o, parent_idx) } else { appendage_body_part(o, parent_idx) }
    };
    match template.symmetry {
        Symmetry::NoSymmetry => {
            for (app_raw, is_limb, parent) in &template.custom_appendages {
                body_parts.push(make(app_raw.clone(), *is_limb, *parent));
            }
        }
        Symmetry::Bilateral => {
            // Per-editor-index right/left runtime indices.
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
    // Sum cells across all parts so multi-part species get a proportionate tank.
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
        template.movement_mode,
        template.ground_based,
        commands,
        meshes,
        org_mats,
        &mut rng,
    );
    if template.is_carnivore {
        commands.entity(entity).try_insert(crate::colony::Carnivore);
    }
    // Mark `.species`-sourced organisms so speciation treats them as a
    // fresh founder lineage; the marker's filename stem becomes the
    // species display name shown in the viewport label.
    if let Some(ref species_name) = template.species_name {
        commands.entity(entity).try_insert(
            crate::lineages::species::ImportedSpeciesOrigin {
                name: species_name.clone(),
            },
        );
    }
    entity
}

/// Push a new `OrganismTemplate` onto the session, spawn its marker at
/// `position`, select it active, and return its id (for an undo entry).
/// `pub(super)` so bulk-add shares the same per-template spawn pipeline.
pub(super) fn spawn_template_at(
    position:      Vec3,
    session:       &mut ResMut<EditorSession>,
    commands:      &mut Commands,
    meshes:        &mut ResMut<Assets<Mesh>>,
    materials:     &mut ResMut<Assets<StandardMaterial>>,
    org_materials: Option<&OrganismMaterials>,
    smoothing:     bool,
) -> Option<u32> {
    // Species-driven: a spawn requires a `.species` selected in the navigator.
    let species_id = session.selected_species_id?;
    let species = session.loaded_species.iter().find(|s| s.id == species_id).cloned()?;
    Some(spawn_species_template_at(
        position, &species, session, commands, meshes, materials, org_materials, smoothing,
    ))
}

/// Spawn one instance of a loaded species at `position`, carrying its
/// OCG + body-plan flags verbatim. Per-organism hyperparameter genes are
/// NOT set here — the brain pool samples them from `L1_*_RANGE` at slot
/// assignment, so each spawn gets a unique gene set.
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
        movement_mode: species.movement_mode,
        is_sessile:   species.is_sessile,
        ground_based: species.ground_based,
    };
    let entity = respawn_template(&template, commands, meshes, materials, org_materials, smoothing);

    // If the species carried a trained brain, attach a copy of the
    // restore payload; `assign_brains_herbivore_1` consumes it next
    // PreUpdate. All bulk instances share the payload and diverge
    // through training, as designed.
    if let Some(brain) = &species.brain {
        commands.entity(entity).try_insert(brain.clone());
    }

    session.templates.push(OrganismTemplate { entity, ..template });
    session.active_id = Some(id);
    session.dirty     = true;
    id
}


/// March the ray and return the first point at/below the heightmap, or
/// `None` if it never crosses within `MAX_RAY_DISTANCE`. Hits are only
/// accepted inside `[0,map_size.x] × [0,map_size.z]` — outside that the
/// heightmap clamps to its border, so a reported hit would feel wrong.
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
