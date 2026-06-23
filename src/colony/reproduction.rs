// Reproduction. Each birth clones the parent's body plan and applies one
// mutation step (NoSymmetry: 20% new branch / 80% extend root; Bilateral:
// 20% new appendage pair / 80% grow base by a mirrored cell pair). Cell
// types inherited from the seed, preserving trophic identity. The
// `reproduced` flag enforces a per-class reproduction cap.

use bevy::prelude::*;
use rand::prelude::*;

use crate::body_part::{self, Attachment};
use crate::cell::*;
use crate::colony::*;
use crate::energy::MAX_ENERGY_PER_CELL;
use crate::world_geometry::{HeightmapSampler, MapSize};


// ── Constants ────────────────────────────────────────────────────────────────

use crate::simulation_settings::{
    REPRODUCTION_CHECK_INTERVAL, OFFSPRING_ENERGY_FRACTION,
    REPRODUCTION_ENERGY_THRESHOLD, HETEROTROPH_REPRODUCTION_CAP,
};


// ── Timer resource ───────────────────────────────────────────────────────────

#[derive(Resource)]
pub struct ReproductionTimer {
    pub timer: Timer,
}

impl Default for ReproductionTimer {
    fn default() -> Self {
        Self { timer: Timer::from_seconds(REPRODUCTION_CHECK_INTERVAL, TimerMode::Repeating) }
    }
}


// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct ReproductionPlugin;

impl Plugin for ReproductionPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ReproductionTimer::default());
        app.add_systems(Update, reproduction_system);
    }
}




// ── Reproduction system ──────────────────────────────────────────────────────

fn reproduction_system(
    time:           Res<Time>,
    virtual_time:   Res<Time<Virtual>>,
    mut timer:      ResMut<ReproductionTimer>,
    mut commands:   Commands,
    mut meshes:     ResMut<Assets<Mesh>>,
    materials:      ResMut<Assets<StandardMaterial>>,
    organism_mats:  Option<Res<OrganismMaterials>>,
    heightmap:      Option<Res<HeightmapSampler>>,
    water_level:    Option<Res<crate::environment::WaterLevel>>,
    smoothing:      Res<crate::simulation_settings::Smoothing>,
    map_size:       Res<MapSize>,
    max_photoautotrophs: Res<crate::simulation_settings::MaxPhotoautotrophs>,
    max_herbivores:      Res<crate::simulation_settings::MaxHerbivores>,
    ai_training:         Res<crate::simulation_settings::AiTrainingMode>,
    mut query:      Query<
        (Entity, &mut Organism, &Transform, Has<Photoautotroph>, Has<Heterotroph>, Has<Carnivore>),
        With<OrganismRoot>,
    >,
) {
    // Kept in the signature though every birth reuses the shared
    // `OrganismMaterials` resource.
    let _ = materials;
    timer.timer.tick(time.delta());
    if !timer.timer.just_finished() { return; }

    // Independent per-class caps. `OrganismPoolSize` still bounds
    // heterotrophs implicitly: offspring beyond it spawn but get no
    // brain slot until one frees.
    let photo_cap  = max_photoautotrophs.0;
    let herb_cap   = max_herbivores.0;
    let live_photos = query.iter()
        .filter(|(_, _, _, is_photo, _, _)| *is_photo)
        .count();
    let live_herbivores = query.iter()
        .filter(|(_, _, _, _, is_hetero, is_carn)| *is_hetero && !*is_carn)
        .count();
    let mut pending_photo_births:     usize = 0;
    let mut pending_herbivore_births: usize = 0;

    let mut pending_births: Vec<PendingBirth> = Vec::new();
    let mut rng = rand::rng();

    for (parent_entity, mut organism, transform, is_photo, is_hetero, is_carn) in &mut query {
        if organism.reproduced { continue; }

        // Skip when reproducing would push the class at/above its cap.
        // Carnivores are unbounded.
        let is_herbivore = is_hetero && !is_carn;
        // Training scaffold: LIMB herbivores don't reproduce during AI training.
        // A mid-run offspring spawned while the steering assist is rotating the
        // cohort hit the rotation in a half-initialised state and momentarily
        // separated its joints (G2). Suppressing limb-herbivore births keeps the
        // learning cohort fixed and G2 clean — and is decoupled from
        // `--max-herbivores` (which also sizes the GPU brain pool / FPS), unlike
        // setting the cap to 0.
        if is_herbivore && !organism.movement_mode.is_sliding()
            && ai_training.0
            && crate::simulation_settings::DISABLE_LIMB_HERBIVORE_REPRODUCTION
        {
            continue;
        }
        if is_photo
            && (live_photos + pending_photo_births) >= photo_cap
        {
            continue;
        }
        if is_herbivore
            && (live_herbivores + pending_herbivore_births) >= herb_cap
        {
            continue;
        }

        let max_energy = (organism.grown_cell_count() as f32) * MAX_ENERGY_PER_CELL;
        if max_energy <= 0.0 { continue; }
        if organism.energy < max_energy * REPRODUCTION_ENERGY_THRESHOLD { continue; }

        let offspring_energy = max_energy * OFFSPRING_ENERGY_FRACTION;
        organism.energy      -= offspring_energy;
        organism.reproductions = organism.reproductions.saturating_add(1);
        // RL reward: reproduction grants max dopamine. Child starts at 0.
        organism.dopamine = 1.0;

        // PHOTOTROPHS reproduce WITHOUT a per-individual limit: the population is
        // bounded by `MaxPhotoautotrophs` (the reproduction gate above) + energy/
        // sunlight, and a depleted prey field is refilled by `auto_spawn_plankton`.
        // So never flip `reproduced` for a phototroph — it can breed every time it
        // recharges to the energy threshold. Heterotrophs keep their per-individual cap.
        if !is_photo && organism.reproductions >= HETEROTROPH_REPRODUCTION_CAP {
            organism.reproduced = true;
        }

        let kind = if is_photo {
            OrganismKind::Photoautotroph
        } else if is_hetero {
            OrganismKind::Heterotroph
        } else {
            continue;
        };

        // Build the child's body parts via the growth pipeline for the
        // parent's symmetry (inherited verbatim). AI-training mode: no
        // mutation — clone the parent so the cohort keeps fixed morphology.
        let child_body_parts = if ai_training.0 {
            organism.body_parts.clone()
        } else { match organism.symmetry {
            Symmetry::NoSymmetry => {
                // 20%: new branch body part. 80%: extend root OCG by one cell.
                if body_part::should_branch(&mut rng) {
                    let mut parts = organism.body_parts.clone();
                    let parent_ocg = &parts[0].ocg;
                    if parent_ocg.is_empty() { continue; }
                    let (origin_local, outward) =
                        body_part::pick_attachment(parent_ocg, &mut rng);
                    let seed_ct = parent_ocg[0].2;
                    let attachment = Attachment {
                        parent_idx:   0,
                        origin_local,
                        rotation:     Quat::IDENTITY,
                    };
                    let new_part = body_part::create_branch_body_part(
                        seed_ct, attachment, outward,
                    );
                    // Grow the new part 1 → 2 cells.
                    let grown_ocg = crate::mutation::mutate_ocg(
                        &new_part.ocg, &mut rng,
                    );
                    if grown_ocg.is_empty() { continue; }
                    let mut grown_part = new_part;
                    grown_part.cells = grown_ocg.iter()
                        .map(|(_, p, ct)| Cell::new(*p, *ct))
                        .collect();
                    grown_part.ocg = grown_ocg;
                    parts.push(grown_part);
                    parts
                } else {
                    let mut parts = organism.body_parts.clone();
                    let grown_ocg = crate::mutation::mutate_ocg(
                        &parts[0].ocg, &mut rng,
                    );
                    if grown_ocg.is_empty() { continue; }
                    parts[0].cells = grown_ocg.iter()
                        .map(|(_, p, ct)| Cell::new(*p, *ct))
                        .collect();
                    parts[0].ocg = grown_ocg;
                    parts
                }
            }
            Symmetry::Bilateral => {
                // Bilateral has ONE base part (index 0) whose OCG holds
                // both halves. Each birth: 20% (if a pair fits under
                // MAX_BODY_PARTS) grow a NEW MIRRORED PAIR of appendages
                // on the base (both parent_idx=0, never on another
                // appendage); else extend the base by one mirrored cell
                // pair (mutate the right half, mirror, reassemble —
                // the combined-OCG mesh build welds the seam).
                let mut parts = organism.body_parts.clone();
                if parts.is_empty() { continue; }

                // A pair adds TWO parts, so only branch while two more fit.
                if parts.len() + 2 <= body_part::MAX_BODY_PARTS
                    && body_part::should_branch(&mut rng)
                {
                    let base_ocg = &parts[0].ocg;
                    if base_ocg.is_empty() { continue; }

                    // Pick the attachment from the base's right half only,
                    // so the +X limb and its mirror land symmetrically.
                    let right_base: Vec<(usize, Vec3, CellType)> = base_ocg.iter()
                        .filter(|(_, p, _)| p.x > 0.0)
                        .enumerate()
                        .map(|(i, (_, p, ct))| (i, *p, *ct))
                        .collect();
                    if right_base.is_empty() { continue; }

                    let seed_ct = base_ocg[0].2;
                    let (origin_r, outward_r) =
                        body_part::pick_attachment(&right_base, &mut rng);

                    // Build + grow (1→2) the right appendage.
                    let right_attach = Attachment {
                        parent_idx:   0,
                        origin_local: origin_r,
                        rotation:     Quat::IDENTITY,
                    };
                    let right_seed = body_part::create_branch_body_part(
                        seed_ct, right_attach, outward_r,
                    );
                    let right_ocg = crate::mutation::mutate_ocg(
                        &right_seed.ocg, &mut rng,
                    );
                    if right_ocg.is_empty() { continue; }
                    let mut right_part = right_seed;
                    right_part.cells = right_ocg.iter()
                        .map(|(_, p, ct)| Cell::new(*p, *ct))
                        .collect();
                    right_part.ocg = right_ocg.clone();

                    // Mirror across YZ for the left appendage (cell ledger,
                    // attachment origin, outward seed direction). The mesh
                    // builder emits canonical outward faces, so it's not
                    // inside-out.
                    let left_ocg = body_part::mirror_ocg_x(&right_ocg);
                    let origin_l = Vec3::new(-origin_r.x, origin_r.y, origin_r.z);
                    let outward_l = Vec3::new(-outward_r.x, outward_r.y, outward_r.z);
                    let left_attach = Attachment {
                        parent_idx:   0,
                        origin_local: origin_l,
                        rotation:     Quat::IDENTITY,
                    };
                    let mut left_part = body_part::create_branch_body_part(
                        seed_ct, left_attach, outward_l,
                    );
                    left_part.cells = left_ocg.iter()
                        .map(|(_, p, ct)| Cell::new(*p, *ct))
                        .collect();
                    left_part.ocg = left_ocg;

                    parts.push(right_part);
                    parts.push(left_part);
                    parts
                } else {
                    // Extract the right half (midline + +X) with fresh
                    // sequential indices — `mutate_bilateral` needs a
                    // contiguous [0..N) ledger. `x > -EPS` keeps midline
                    // cells, drops the mirrored −X cells.
                    let right_ocg: Vec<(usize, Vec3, CellType)> = parts[0].ocg.iter()
                        .filter(|(_, p, _)| p.x > -body_part::BILATERAL_MIDLINE_EPS)
                        .enumerate()
                        .map(|(i, (_, p, ct))| (i, *p, *ct))
                        .collect();
                    if right_ocg.is_empty() { continue; }

                    let Some(grown_right) =
                        crate::mutation::mutate_bilateral(&right_ocg, &mut rng)
                        else { continue; };
                    let grown_left = body_part::mirror_right_to_left(&grown_right);

                    // Reassemble both halves into one contiguous-index OCG.
                    let combined: Vec<(usize, Vec3, CellType)> = grown_right.iter()
                        .chain(grown_left.iter())
                        .enumerate()
                        .map(|(i, (_, p, ct))| (i, *p, *ct))
                        .collect();
                    parts[0].cells = combined.iter()
                        .map(|(_, p, ct)| Cell::new(*p, *ct))
                        .collect();
                    parts[0].ocg = combined;
                    parts
                }
            }
        } };

        // Inside the WORLD_SAFETY_MARGIN inset (same band as the initial
        // cohort and `apply_world_bounds`).
        let spawn_x = rng.random_range(
            crate::world_geometry::WORLD_SAFETY_MARGIN
                ..(map_size.x - crate::world_geometry::WORLD_SAFETY_MARGIN),
        );
        let spawn_z = rng.random_range(
            crate::world_geometry::WORLD_SAFETY_MARGIN
                ..(map_size.z - crate::world_geometry::WORLD_SAFETY_MARGIN),
        );
        // Ground + 1.0 clearance (as `spawn_colony`). Fall back to the
        // parent's Y before the heightmap resource exists (rare early tick).
        // SWIMMER offspring instead spawn SUBMERGED above the flat floor
        // cuboid (`submerged_spawn_y`): the walker convention `terrain + 1.0`
        // puts them INSIDE the floor collider wherever the local terrain is
        // below the map-centre height, and Rapier's depenetration ejects the
        // body violently upward ("flung away on spawn").
        let spawn_y = match heightmap.as_ref() {
            Some(hm) if organism.movement_mode.is_swimming() => {
                let water = water_level.as_ref().map(|w| w.0).unwrap_or(0.0);
                crate::colony::submerged_spawn_y(
                    hm.height_at(spawn_x, spawn_z),
                    crate::colony::limb_floor_top(hm),
                    water,
                )
            }
            Some(hm) => hm.height_at(spawn_x, spawn_z) + 1.0,
            None     => transform.translation.y,
        };
        let spawn_pos = Vec3::new(spawn_x, spawn_y, spawn_z);

        if is_herbivore { pending_herbivore_births += 1; }
        if is_photo     { pending_photo_births     += 1; }

        pending_births.push(PendingBirth {
            pos:        spawn_pos,
            body_parts: child_body_parts,
            energy:     offspring_energy,
            kind,
            // Inherited verbatim from the parent.
            symmetry:           organism.symmetry,
            has_variable_form:  organism.has_variable_form,
            is_sessile:         organism.is_sessile,
            movement_mode:      organism.movement_mode,
            ground_based:       organism.ground_based,
            intelligence_level: organism.intelligence_level,
            // For copying parent → child brain weights when
            // `assign_brains_*` runs. Skipped for Level0 (no MLP rows).
            parent_for_brain_inheritance: if matches!(
                organism.intelligence_level,
                crate::organism::IntelligenceLevel::Level0
            ) {
                None
            } else {
                Some(parent_entity)
            },
            parent_species_id: organism.species_id,
            lineage_parent:    parent_entity,
        });
    }

    if pending_births.is_empty() { return; }
    // Reuse the shared `OrganismMaterials` resource — minting fresh
    // StandardMaterials per birth leaks VRAM (assets only GC when the
    // last strong handle drops, and every body part holds one).
    let Some(organism_materials) = organism_mats.as_deref() else {
        warn!("reproduction_system: OrganismMaterials resource missing — \
               spawn_colony hasn't run yet. Skipping this tick's births.");
        return;
    };

    for birth in pending_births {
        let child_root = spawn_organism(
            birth.pos,
            birth.body_parts,
            birth.kind,
            birth.symmetry,
            birth.has_variable_form,
            birth.is_sessile,
            birth.intelligence_level,
            smoothing.0,
            birth.energy,
            birth.movement_mode,
            birth.ground_based,
            &mut commands,
            &mut meshes,
            organism_materials,
            &mut rng,
        );
        if let Some(parent) = birth.parent_for_brain_inheritance {
            // `assign_brains_*` copies the parent's weights into the new
            // slot and removes this marker. If the parent already died,
            // the lookup fails silently (offspring keeps recycled weights).
            commands.entity(child_root)
                .try_insert(crate::rl_helpers::BrainInheritance(parent));
        }

        // Lineage record with `parent_id = Some(parent)`; attaching it
        // here makes the default `assign_lineage_records` skip this entity.
        let spawn_time = virtual_time.elapsed_secs();
        commands.entity(child_root).try_insert(
            crate::organism::LineageRecord::new_offspring(
                birth.lineage_parent,
                spawn_time,
            ),
        );

        // Bump the parent's reproduction counter via Commands (the
        // active `&mut Organism` query already borrows the parent, so a
        // direct mutation would conflict). Tolerates a since-dead parent.
        let parent_entity_for_queue = birth.lineage_parent;
        commands.queue(move |world: &mut World| {
            if let Ok(mut entity_ref) = world.get_entity_mut(parent_entity_for_queue) {
                if let Some(mut lr) =
                    entity_ref.get_mut::<crate::organism::LineageRecord>()
                {
                    lr.times_reproduced_self =
                        lr.times_reproduced_self.saturating_add(1);
                }
            }
        });
        // Inherit parent's species id; the next speciation tick may
        // re-classify if brain genes have drifted past threshold.
        if let Some(sid) = birth.parent_species_id {
            commands.queue(move |world: &mut World| {
                if let Ok(mut entity_ref) = world.get_entity_mut(child_root) {
                    if let Some(mut org) = entity_ref.get_mut::<crate::organism::Organism>() {
                        org.species_id = Some(sid);
                    }
                }
            });
        }
    }
}


struct PendingBirth {
    pos:                Vec3,
    body_parts:         Vec<BodyPart>,
    energy:             f32,
    kind:               OrganismKind,
    symmetry:           Symmetry,
    has_variable_form:  bool,
    is_sessile:         bool,
    /// Inherited verbatim from the parent (movement paradigm).
    movement_mode:      MovementMode,
    /// Inherited verbatim from the parent (ground- vs water-based).
    ground_based:       bool,
    /// Inherited verbatim from the parent.
    intelligence_level: IntelligenceLevel,
    /// Parent entity for `BrainInheritance` (copy parent's brain row
    /// into the offspring's slot). `None` for Level0 (no pool).
    parent_for_brain_inheritance: Option<Entity>,
    /// Parent's species id at reproduction, inherited onto the child;
    /// the next speciation tick may re-classify. `None` if the parent
    /// wasn't yet classified.
    parent_species_id: Option<u32>,
    /// Parent entity for the lineage `parent_id` link — always present
    /// (distinct from `parent_for_brain_inheritance`, gated on level).
    lineage_parent: Entity,
}
