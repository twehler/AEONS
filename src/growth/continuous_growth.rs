// Continuous growth — periodic mutation for variable-form organisms.
//
// Reproduction grows offspring by one cell at birth, fine for motile Bilateral
// creatures. Plant-like (variable-form, sessile) organisms instead grow IN
// PLACE so they develop a canopy over their lifetime. `ContinuousGrowthPlugin`
// ticks ~1 Hz, applying one mutation to each alive `has_variable_form` organism
// below `MAX_CELLS`, using reproduction's 80/20 split (extend root vs. new
// branch). Variable-form organisms still reproduce normally on top of this.

use bevy::prelude::*;
use rand::prelude::*;

use crate::body_part::{self, Attachment};
use crate::cell::*;
use crate::colony::*;
use crate::frontend::ShowGizmo;
use crate::physiology;
use crate::volumetric_growth::{build_mesh_from_ocg, MAX_CELLS};


use crate::simulation_settings::{
    GROWTH_PHASE_PERIOD, GROWTH_PHASE_STEP_SECS,
};


// ── Resources ────────────────────────────────────────────────────────────────

#[derive(Resource)]
pub struct ContinuousGrowthTimer(pub Timer);

impl Default for ContinuousGrowthTimer {
    fn default() -> Self {
        Self(Timer::from_seconds(GROWTH_PHASE_STEP_SECS, TimerMode::Repeating))
    }
}

/// Rotating phase counter (mod `GROWTH_PHASE_PERIOD`), advanced each tick. An
/// organism runs this tick iff `entity.index() % GROWTH_PHASE_PERIOD == counter`.
/// Using the entity index as phase avoids storing a per-organism field; the
/// modulo of sequential indices spreads evenly at our scales.
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
    // `materials` is unused — every branch reuses the shared `OrganismMaterials`.
    let _ = materials;

    timer.0.tick(time.delta());
    if !timer.0.just_finished() { return; }

    // Read current phase, then advance for next tick.
    let current_phase = phase_counter.0;
    phase_counter.0 = (phase_counter.0 + 1) % GROWTH_PHASE_PERIOD;

    let mut rng = rand::rng();
    // Reuse the shared `OrganismMaterials` (populated by `spawn_colony`) rather
    // than minting fresh `StandardMaterial`s per branch — those leaked (each
    // body-part holds a strong handle), growing the asset arena unbounded.
    let Some(organism_materials) = organism_mats.as_deref() else {
        // Colony not initialised yet — skip this tick.
        return;
    };

    for (root_entity, mut organism, is_photo, is_hetero) in &mut organisms {
        // Phase gate: ~1/`GROWTH_PHASE_PERIOD` of organisms run per tick; the
        // rotation keeps each organism's effective rate at 1/CONTINUOUS_GROWTH_INTERVAL.
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

        // The body just changed shape (its lowest cell may have moved), so re-seat
        // it on the floor: drop the one-shot placement marker and
        // `place_sessile_organisms` re-runs once next frame. Variable-form ⇒
        // sessile, so every grown organism is a seated plant. Cost is bounded to
        // the few organisms that grew this tick — not per frame.
        commands.entity(root_entity).remove::<crate::movement::SurfacePlaced>();

        // Adult transition: fires once, on the tick that crosses `MAX_CELLS`
        // (the guard above then excludes the organism forever). Smoothing is
        // gated on `Smoothing`; when off, `adult` still flips (so a later
        // toggle-on doesn't re-fire) but the mesh stays faceted.
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


/// Replace each alive, regrowable, non-empty body-part mesh with the smoothed
/// version of its OCG. Called once per organism, when `adult` flips true.
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


/// Append a new branch body part attached to body_parts[0]: seed 1 cell, grow
/// to 2 via mutation (matches reproduction's branch path), recompute physiology,
/// then materialise as a new Bevy child entity.
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

    // Grow 1→2 cells for visibility.
    let grown_ocg = crate::mutation::mutate_ocg(&new_part.ocg, rng);
    if grown_ocg.is_empty() { return; }
    new_part.cells = grown_ocg.iter().map(|(_, p, ct)| Cell::new(*p, *ct)).collect();
    new_part.ocg = grown_ocg;
    physiology::recompute_body_part(&mut new_part);

    // Update cached cell counts + upkeep weight — energy/physiology read the cache.
    for cell in &new_part.cells {
        if cell.cell_type.is_photo() {
            organism.photo_cell_count += 1;
        } else {
            organism.non_photo_cell_count += 1;
        }
        organism.upkeep_weight += cell.cell_type.upkeep_mult();
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

    // Flat hierarchy: branches are direct children of OrganismRoot, not nested
    // under body_parts[PARENT_IDX]. Nesting let a predation despawn of
    // body_parts[0] cascade-kill every branch, leaving a phantom root. Parent
    // transform is identity for procedural organisms, so position is unchanged.
    let _ = (children_q, body_part_idx_q, PARENT_IDX);
    commands.entity(root_entity).add_child(child_entity);

    organism.body_parts.push(new_part);
    // New branch extends the envelope — refresh cached bounding radius before
    // the next movement/floor/collision tick reads it.
    organism.recompute_bounding_radius();
}


/// One mutation step on body_parts[0]'s OCG: update cells, recompute physiology,
/// update the existing child mesh asset in place.
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
        // No valid candidate (frontier may be fully enclosed) — retry next tick.
        return;
    }

    let new_cell_type = grown_ocg.last().unwrap().2;

    organism.body_parts[0].cells = grown_ocg.iter()
        .map(|(_, p, ct)| Cell::new(*p, *ct))
        .collect();
    organism.body_parts[0].ocg = grown_ocg.clone();
    physiology::recompute_body_part(&mut organism.body_parts[0]);

    if new_cell_type.is_photo() {
        organism.photo_cell_count += 1;
    } else {
        organism.non_photo_cell_count += 1;
    }
    organism.upkeep_weight += new_cell_type.upkeep_mult();
    // Refresh cached bounding radius — the new cell may extend the envelope.
    organism.recompute_bounding_radius();

    // Replace the mesh asset's contents in place: change detection re-uploads
    // the buffers and the Mesh3d handle stays valid — no entity churn.
    if let Some(part_entity) = find_body_part_entity(root_entity, 0, children_q, body_part_idx_q) {
        if let Ok(mesh3d) = mesh3d_q.get(part_entity) {
            if let Some(mesh) = meshes.get_mut(&mesh3d.0) {
                *mesh = build_mesh_from_ocg(&grown_ocg);
            }
        }
    }
}


/// First descendant of `parent` whose `BodyPartIndex` matches `target_idx`.
/// Recursive because body parts may be nested.
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
