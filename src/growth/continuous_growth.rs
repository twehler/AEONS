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


/// How often the continuous-growth tick runs. 1.0 s gives a noticeable
/// "growing" silhouette over ~30 seconds for a fresh seed reaching the
/// 30-cell cap. Tune higher for slower growth, lower for faster.
const CONTINUOUS_GROWTH_INTERVAL: f32 = 1.0;


// ── Resources ────────────────────────────────────────────────────────────────

#[derive(Resource)]
pub struct ContinuousGrowthTimer(pub Timer);

impl Default for ContinuousGrowthTimer {
    fn default() -> Self {
        Self(Timer::from_seconds(CONTINUOUS_GROWTH_INTERVAL, TimerMode::Repeating))
    }
}


// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct ContinuousGrowthPlugin;

impl Plugin for ContinuousGrowthPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ContinuousGrowthTimer>()
            .add_systems(Update, grow_variable_form_organisms);
    }
}


// ── Per-tick growth ──────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn grow_variable_form_organisms(
    time:                Res<Time<Virtual>>,
    mut timer:           ResMut<ContinuousGrowthTimer>,
    mut commands:        Commands,
    mut meshes:          ResMut<Assets<Mesh>>,
    mut materials:       ResMut<Assets<StandardMaterial>>,
    mut organisms:       Query<
        (Entity, &mut Organism, Has<Photoautotroph>, Has<Heterotroph>),
        With<OrganismRoot>,
    >,
    children_q:          Query<&Children>,
    body_part_idx_q:     Query<&BodyPartIndex>,
    mesh3d_q:            Query<&Mesh3d>,
) {
    timer.0.tick(time.delta());
    if !timer.0.just_finished() { return; }

    let mut rng = rand::rng();
    // Built once per tick, not per organism — `materials.add()` is cheap
    // but still a heap allocation per call, and per-tick is plenty.
    let materials_helper = OrganismMaterials::new(&mut materials);

    for (root_entity, mut organism, is_photo, is_hetero) in &mut organisms {
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
                &materials_helper, &children_q, &body_part_idx_q,
            );
        } else {
            extend_root_part(
                &mut organism, root_entity,
                &mut rng, &mut meshes,
                &children_q, &body_part_idx_q, &mesh3d_q,
            );
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
    )).id();

    if let Some(parent_entity) = find_body_part_entity(
        root_entity, PARENT_IDX, children_q, body_part_idx_q,
    ) {
        commands.entity(parent_entity).add_child(child_entity);
    }

    organism.body_parts.push(new_part);
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
