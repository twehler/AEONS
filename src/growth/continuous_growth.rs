// Continuous growth — periodic mutation for variable-form organisms.
//
// Reproduction grows offspring by exactly one cell at birth. That works for
// motile (Bilateral) creatures that scatter their genes through reproduction
// rounds, but plant-like (variable-form, sessile) organisms need to grow
// IN PLACE so they can develop a recognisable canopy over their lifetime.
//
// `ContinuousGrowthPlugin` ticks once per second and applies one round of
// mutation to every alive organism whose `has_variable_form == true` and
// whose total grown-cell count is below `MAX_CELLS`. The same 80/20 split
// reproduction uses (extend the root part vs. spawn a new branch) is
// reused here, so a tree-shaped organism naturally develops branches over
// time as well as a fattening trunk.
//
// Variable-form organisms can keep going through normal reproduction too:
// each generation inherits the parent's body plan plus one mutation. With
// continuous growth on top, parents become large structures whose offspring
// scatter to new spots — the same loop that already runs for any other
// NoSymmetry organism.

use bevy::prelude::*;
use rand::prelude::*;

use crate::body_part::{self, Attachment};
use crate::cell::*;
use crate::colony::*;
use crate::frontend::ShowGizmo;
use crate::physiology;
use crate::volumetric_growth::{build_mesh_from_ocg, MAX_CELLS};


/// Effective growth cadence per organism. Each variable-form organism
/// receives one growth tick every `CONTINUOUS_GROWTH_INTERVAL` seconds.
/// 1.0 s gives a noticeable "growing" silhouette over ~30 seconds for
/// a fresh seed reaching the 30-cell cap.
const CONTINUOUS_GROWTH_INTERVAL: f32 = 1.0;

/// Number of phase slices the per-second growth workload is sliced into.
/// At 30, the system fires every `1/30` s ≈ 33 ms and each tick
/// processes only the organisms whose entity-index modulo 30 matches
/// the rotating phase counter — roughly 1/30th of the variable-form
/// population per tick. Total work per second is unchanged; the
/// per-tick allocator + Bevy command-buffer spike that was visible
/// every second goes away. Aligned with the 30 Hz brain tick so both
/// throttled subsystems share the same timing rhythm.
const GROWTH_PHASE_PERIOD: u32 = 30;

/// Wall-clock interval between phase steps. `CONTINUOUS_GROWTH_INTERVAL
/// / GROWTH_PHASE_PERIOD` so the per-organism cadence is preserved.
const GROWTH_PHASE_STEP_SECS: f32 =
    CONTINUOUS_GROWTH_INTERVAL / GROWTH_PHASE_PERIOD as f32;


// ── Resources ────────────────────────────────────────────────────────────────

#[derive(Resource)]
pub struct ContinuousGrowthTimer(pub Timer);

impl Default for ContinuousGrowthTimer {
    fn default() -> Self {
        Self(Timer::from_seconds(GROWTH_PHASE_STEP_SECS, TimerMode::Repeating))
    }
}

/// Rotating phase counter (mod `GROWTH_PHASE_PERIOD`). Incremented by
/// the system on every fired tick. An organism is processed this tick
/// iff `entity.index() % GROWTH_PHASE_PERIOD == counter`. Using the
/// entity index as the phase saves one byte per organism vs. storing a
/// `growth_phase` field; the modulo of sequentially-allocated indices
/// distributes evenly across phases at our scales.
#[derive(Resource, Default)]
pub struct GrowthPhaseCounter(pub u32);


// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct ContinuousGrowthPlugin;

impl Plugin for ContinuousGrowthPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ContinuousGrowthTimer>()
            .init_resource::<GrowthPhaseCounter>()
            .add_systems(Update, grow_variable_form_organisms);
    }
}


// ── Per-tick growth ──────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn grow_variable_form_organisms(
    time:                Res<Time<Virtual>>,
    mut timer:           ResMut<ContinuousGrowthTimer>,
    mut phase_counter:   ResMut<GrowthPhaseCounter>,
    smoothing:           Res<crate::simulation_settings::Smoothing>,
    mut commands:        Commands,
    mut meshes:          ResMut<Assets<Mesh>>,
    materials:           ResMut<Assets<StandardMaterial>>,
    organism_mats:       Option<Res<OrganismMaterials>>,
    mut organisms:       Query<
        (Entity, &mut Organism, Has<Photoautotroph>, Has<Heterotroph>),
        With<OrganismRoot>,
    >,
    children_q:          Query<&Children>,
    body_part_idx_q:     Query<&BodyPartIndex>,
    mesh3d_q:            Query<&Mesh3d>,
) {
    // `materials` is kept in the signature for symmetry with other
    // spawn sites and so that any future code path inside this
    // system that needs to mint a one-off material has direct
    // access. Currently every branch reuses the shared
    // `OrganismMaterials` resource.
    let _ = materials;

    timer.0.tick(time.delta());
    if !timer.0.just_finished() { return; }

    // Read the current phase, then advance the counter for the next
    // tick. Every organism whose entity-index mod period equals
    // `current_phase` will be processed this tick; all others wait
    // for their slot in the rotation.
    let current_phase = phase_counter.0;
    phase_counter.0 = (phase_counter.0 + 1) % GROWTH_PHASE_PERIOD;

    let mut rng = rand::rng();
    // Reuse the shared `OrganismMaterials` resource (populated by
    // `spawn_colony`). The previous code lazily called
    // `OrganismMaterials::new(&mut materials)` on every branch tick
    // and that minted three fresh `StandardMaterial` assets into
    // `Assets<StandardMaterial>` — they leaked over the simulation's
    // lifetime because each spawned body-part held a strong handle
    // to one of them. After tens of thousands of branch growth
    // events the asset arena had grown enough to be the dominant
    // VRAM driver. Reading from the Resource means three handles,
    // ever.
    let Some(organism_materials) = organism_mats.as_deref() else {
        // Heightmap / colony hasn't initialised yet — skip the
        // whole tick, the resource will be present next time.
        return;
    };

    for (root_entity, mut organism, is_photo, is_hetero) in &mut organisms {
        // Phase gate: only ~1/`GROWTH_PHASE_PERIOD` of variable-form
        // organisms run their growth step this tick. Per-organism
        // effective rate stays at 1 / `CONTINUOUS_GROWTH_INTERVAL`
        // because the phase counter rotates through every value over
        // exactly that interval.
        if root_entity.index().index() % GROWTH_PHASE_PERIOD != current_phase { continue; }

        if !organism.has_variable_form { continue; }
        if organism.grown_cell_count() >= MAX_CELLS { continue; }
        if organism.body_parts.is_empty() { continue; }

        let kind = if is_photo {
            OrganismKind::Photoautotroph
        } else if is_hetero {
            OrganismKind::Heterotroph
        } else {
            continue;
        };

        if body_part::should_branch(&mut rng) {
            grow_new_branch(
                &mut organism, root_entity, kind,
                &mut rng, &mut commands, &mut meshes,
                organism_materials, &children_q, &body_part_idx_q,
            );
        } else {
            extend_root_part(
                &mut organism, root_entity,
                &mut rng, &mut meshes,
                &children_q, &body_part_idx_q, &mesh3d_q,
            );
        }

        // Adult-transition check. Runs on whichever tick first pushes
        // the organism over `MAX_CELLS`. Once `adult` flips, this
        // branch never fires for that organism again — the
        // `grown_cell_count() >= MAX_CELLS` guard above also skips it
        // from the growth loop entirely. So the smoothing pass is
        // strictly one-time per organism over its entire lifetime.
        // The smoothing operation itself is gated on `Smoothing` —
        // when the resource is `false` the `adult` flag still flips
        // (so future smoothing-on toggles don't re-fire), but the
        // mesh stays faceted.
        if !organism.adult && organism.grown_cell_count() >= MAX_CELLS {
            organism.adult = true;
            if smoothing.0 {
                smooth_all_body_part_meshes(
                    &organism, root_entity, &mut meshes,
                    &children_q, &body_part_idx_q, &mesh3d_q,
                );
            }
        }
    }
}


/// Replace every alive, regrowable body-part mesh on `organism` with
/// the smoothed version of its OCG. Called exactly once per organism's
/// lifetime — on the tick where its `adult` flag flips from false to
/// true. Iterates only the body parts that have an actual mesh
/// (`regrowable && !ocg.is_empty()`).
fn smooth_all_body_part_meshes(
    organism:        &Organism,
    root_entity:     Entity,
    meshes:          &mut Assets<Mesh>,
    children_q:      &Query<&Children>,
    body_part_idx_q: &Query<&BodyPartIndex>,
    mesh3d_q:        &Query<&Mesh3d>,
) {
    for (idx, bp) in organism.body_parts.iter().enumerate() {
        if !bp.is_alive() || !bp.regrowable || bp.ocg.is_empty() { continue; }
        let Some(part_entity) = find_body_part_entity(
            root_entity, idx, children_q, body_part_idx_q,
        ) else { continue };
        let Ok(mesh3d) = mesh3d_q.get(part_entity) else { continue };
        if let Some(mesh) = meshes.get_mut(&mesh3d.0) {
            *mesh = crate::volumetric_growth::build_smoothed_mesh_from_ocg(&bp.ocg);
        }
    }
}


/// Append a new branch body part attached to body_parts[0]. The branch
/// starts with 1 seed cell, is grown to 2 via mutation (mirroring the
/// reproduction-time branch path so newly-spawned branches are visibly
/// non-trivial), gets its physiology recomputed, then is materialised
/// as a new Bevy child entity parented to the root body-part's entity.
#[allow(clippy::too_many_arguments)]
fn grow_new_branch(
    organism:        &mut Organism,
    root_entity:     Entity,
    kind:            OrganismKind,
    rng:             &mut impl Rng,
    commands:        &mut Commands,
    meshes:          &mut Assets<Mesh>,
    materials:       &OrganismMaterials,
    children_q:      &Query<&Children>,
    body_part_idx_q: &Query<&BodyPartIndex>,
) {
    const PARENT_IDX: usize = 0;
    let parent_ocg = organism.body_parts[PARENT_IDX].ocg.clone();
    if parent_ocg.is_empty() { return; }

    let (origin_local, outward) = body_part::pick_attachment(&parent_ocg, rng);
    let seed_ct = parent_ocg[0].2;
    let attachment = Attachment {
        parent_idx: PARENT_IDX,
        origin_local,
        rotation: Quat::IDENTITY,
    };
    let mut new_part = body_part::create_branch_body_part(seed_ct, attachment, outward);

    // Grow 1→2 cells for visibility (matches reproduction's branch path).
    let grown_ocg = crate::mutation::mutate_ocg(&new_part.ocg, rng);
    if grown_ocg.is_empty() { return; }
    new_part.cells = grown_ocg.iter().map(|(_, p, ct)| Cell::new(*p, *ct)).collect();
    new_part.ocg = grown_ocg;
    physiology::recompute_body_part(&mut new_part);

    // Update the cached cell counts on the organism — composition has
    // changed, and predation / energy / physiology read from the cache.
    for cell in &new_part.cells {
        match cell.cell_type {
            CellType::Photo    => organism.photo_cell_count    += 1,
            CellType::NonPhoto => organism.non_photo_cell_count += 1,
        }
    }

    let new_idx = organism.body_parts.len();
    let mesh_handle = meshes.add(build_mesh_from_ocg(&new_part.ocg));
    let mat = materials.handle_for(kind, &new_part);

    let child_entity = commands.spawn((
        Mesh3d(mesh_handle),
        MeshMaterial3d(mat),
        Transform { translation: origin_local, rotation: Quat::IDENTITY, ..default() },
        Visibility::Visible,
        OrganismMesh,
        BodyPartIndex(new_idx),
        ShowGizmo,
        bevy::light::NotShadowCaster,
    )).id();

    // Flat hierarchy — new branches are direct children of the
    // OrganismRoot, not nested under body_parts[PARENT_IDX]'s entity.
    // Mirrors the fix in `spawn_organism`: when branches were nested,
    // predation-despawning body_parts[0] cascaded through recursive
    // try_despawn and killed every branch, leaving the root entity
    // alive with `prey_dead == false` and zero children — a phantom.
    // body_parts[PARENT_IDX]'s transform is identity for procedural
    // organisms, so the world position is unchanged.
    let _ = (children_q, body_part_idx_q, PARENT_IDX);
    commands.entity(root_entity).add_child(child_entity);

    organism.body_parts.push(new_part);
    // The new branch extends the cell envelope further out from the
    // root, so the cached bounding radius needs refreshing before the
    // next per-frame movement / floor / collision tick reads it.
    // O(cells) — only runs at growth events.
    organism.recompute_bounding_radius();
}


/// Run one mutation step on body_parts[0]'s OCG, replace the part's cells
/// to match, recompute its physiology, and update the existing child mesh
/// asset in-place so the visual catches up.
fn extend_root_part(
    organism:        &mut Organism,
    root_entity:     Entity,
    rng:             &mut impl Rng,
    meshes:          &mut Assets<Mesh>,
    children_q:      &Query<&Children>,
    body_part_idx_q: &Query<&BodyPartIndex>,
    mesh3d_q:        &Query<&Mesh3d>,
) {
    let root_part_ocg = organism.body_parts[0].ocg.clone();
    let grown_ocg = crate::mutation::mutate_ocg(&root_part_ocg, rng);
    if grown_ocg.len() <= root_part_ocg.len() {
        // No valid growth candidate this tick — frontier might be
        // fully enclosed for the current cell layout. Try again next
        // tick.
        return;
    }

    let new_cell_type = grown_ocg.last().unwrap().2;

    organism.body_parts[0].cells = grown_ocg.iter()
        .map(|(_, p, ct)| Cell::new(*p, *ct))
        .collect();
    organism.body_parts[0].ocg = grown_ocg.clone();
    physiology::recompute_body_part(&mut organism.body_parts[0]);

    match new_cell_type {
        CellType::Photo    => organism.photo_cell_count    += 1,
        CellType::NonPhoto => organism.non_photo_cell_count += 1,
    }
    // Cached bounding radius needs refreshing — the new cell may extend
    // the envelope. Only one body part has changed; the function still
    // walks all alive body parts but at 30 cells max it's cheap.
    organism.recompute_bounding_radius();

    // Replace the existing mesh asset's contents in place. Bevy's asset
    // change detection flags the asset, the renderer re-uploads the new
    // vertex/index buffers, and the existing Mesh3d handle keeps
    // pointing at the right thing — no entity churn.
    if let Some(part_entity) = find_body_part_entity(root_entity, 0, children_q, body_part_idx_q) {
        if let Ok(mesh3d) = mesh3d_q.get(part_entity) {
            if let Some(mesh) = meshes.get_mut(&mesh3d.0) {
                *mesh = build_mesh_from_ocg(&grown_ocg);
            }
        }
    }
}


/// Walk the entity tree starting at `parent` and return the first
/// descendant whose `BodyPartIndex` matches `target_idx`. Body parts may
/// be nested (a branch's entity is a child of its attachment-target body
/// part's entity), so the search is recursive.
fn find_body_part_entity(
    parent:     Entity,
    target_idx: usize,
    children_q: &Query<&Children>,
    bp_idx_q:   &Query<&BodyPartIndex>,
) -> Option<Entity> {
    let Ok(children) = children_q.get(parent) else { return None; };
    for child in children.iter() {
        if let Ok(idx) = bp_idx_q.get(child) {
            if idx.0 == target_idx { return Some(child); }
        }
        if let Some(found) = find_body_part_entity(child, target_idx, children_q, bp_idx_q) {
            return Some(found);
        }
    }
    None
}
