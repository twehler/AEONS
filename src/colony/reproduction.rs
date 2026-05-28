// Reproduction.
//
// Each birth clones the parent's body plan and applies one mutation step:
//   * NoSymmetry: 20% append a new branch body part, else extend the root
//     part's OCG by one cell.
//   * Bilateral: 20% (under MAX_BODY_PARTS) append a new appendage on the
//     base body, else grow the base by one mirrored cell pair.
// Cell types are inherited from the seed entry, preserving trophic identity.
//
// The `reproduced` boolean on `Organism` enforces a species-specific
// reproduction cap (`reproductions` counter vs. the cap): heterotrophs
// reproduce at most twice, photoautotrophs at most twice.

use bevy::prelude::*;
use rand::prelude::*;

use crate::body_part::{self, Attachment};
use crate::cell::*;
use crate::colony::*;
use crate::energy::MAX_ENERGY_PER_CELL;
use crate::world_geometry::{HeightmapSampler, MapSize};


// ── Constants ────────────────────────────────────────────────────────────────

const REPRODUCTION_CHECK_INTERVAL: f32 = 2.0;

/// Energy split between parent and offspring at reproduction (50/50).
const OFFSPRING_ENERGY_FRACTION: f32 = 0.5;

/// Threshold (fraction of `max_energy`) above which an organism becomes a
/// reproduction candidate.
const REPRODUCTION_ENERGY_THRESHOLD: f32 = 0.8;

const HETEROTROPH_REPRODUCTION_CAP:    u8 = 2;
const PHOTOAUTOTROPH_REPRODUCTION_CAP: u8 = 2;


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
    smoothing:      Res<crate::simulation_settings::Smoothing>,
    map_size:       Res<MapSize>,
    max_photoautotrophs: Res<crate::simulation_settings::MaxPhotoautotrophs>,
    max_herbivores:      Res<crate::simulation_settings::MaxHerbivores>,
    mut query:      Query<
        (Entity, &mut Organism, &Transform, Has<Photoautotroph>, Has<Heterotroph>, Has<Carnivore>),
        With<OrganismRoot>,
    >,
) {
    // Silence unused-mut warning on `materials` — kept in the
    // signature for the unlikely case that a future code path inside
    // this system needs to mint a material from scratch (e.g. user
    // dropping in a new species at runtime). Currently every birth
    // reuses the shared `OrganismMaterials` resource.
    let _ = materials;
    timer.timer.tick(time.delta());
    if !timer.timer.just_finished() { return; }

    // Per-class caps. The photoautotroph cap and the herbivore cap
    // are INDEPENDENT — each class fills its own budget without
    // competing for shared slots. The global brain pool size
    // (`OrganismPoolSize`) still bounds heterotrophs implicitly: an
    // offspring beyond pool size spawns but doesn't get a brain slot
    // until one frees up.
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

        // Per-class budget: skip the parent when reproducing would
        // push the relevant class's running total at/above its cap.
        // Carnivores are unbounded — they reproduce freely subject
        // only to `reproduced` flag and energy threshold.
        let is_herbivore = is_hetero && !is_carn;
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
        // RL reward: reproduction grants the maximum dopamine reward
        // outright (the strongest evolutionary signal the agent can
        // receive). Child starts at 0 — clean slate.
        organism.dopamine = 1.0;

        let cap = if is_photo { PHOTOAUTOTROPH_REPRODUCTION_CAP } else { HETEROTROPH_REPRODUCTION_CAP };
        if organism.reproductions >= cap { organism.reproduced = true; }

        let kind = if is_photo {
            OrganismKind::Photoautotroph
        } else if is_hetero {
            OrganismKind::Heterotroph
        } else {
            continue;
        };

        // Build the child's body parts from the parent's plan, using the
        // growth pipeline appropriate to the parent's symmetry. Offspring
        // inherit the parent's symmetry verbatim.
        let child_body_parts = match organism.symmetry {
            Symmetry::NoSymmetry => {
                // Legacy path. 20% chance: spawn a new branch body part.
                // 80%: extend the root part's OCG by one cell.
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
                    // Grow the new part from 1 cell to 2.
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
                // Bilateral organisms have ONE base body part (index 0)
                // whose OCG contains both halves (right cells with
                // x > 0 plus their mirrors). On each birth, either:
                //   * 20% (when a symmetric pair still fits under
                //     MAX_BODY_PARTS): grow a NEW MIRRORED PAIR of
                //     appendages on the base body — one on the +X
                //     side, its X-mirror on the −X side. Both attach
                //     to the base body (parent_idx = 0), never to
                //     another appendage. Pairs keep the body plan
                //     bilaterally symmetric, like real limbs.
                //   * otherwise: extend the bilateral base by one
                //     mirrored cell pair — extract the right half,
                //     mutate it (constrained to x ≥ MIN_X_BILATERAL),
                //     mirror, and reassemble. The combined-OCG mesh
                //     build welds the seam and drops interior faces.
                let mut parts = organism.body_parts.clone();
                if parts.is_empty() { continue; }

                // A pair adds TWO parts, so only branch while two more
                // still fit under the cap (with MAX_BODY_PARTS = 3 this
                // means exactly when the organism has only its base).
                if parts.len() + 2 <= body_part::MAX_BODY_PARTS
                    && body_part::should_branch(&mut rng)
                {
                    let base_ocg = &parts[0].ocg;
                    if base_ocg.is_empty() { continue; }

                    // Pick the right-side appendage attachment from the
                    // base's right half only, so the +X limb and its
                    // mirror land symmetrically off each flank.
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

                    // Mirror it across the YZ plane to make the left
                    // appendage: mirror the cell ledger, the attachment
                    // origin, and the outward seed direction. Each cell
                    // is still emitted as a canonical outward-faced RD
                    // by the mesh builder, so the mirrored mesh is not
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
                    // Extract the right half (midline + +X cells) with
                    // fresh sequential indices — `mutate_bilateral`
                    // expects a contiguous [0..N) ledger as input. The
                    // `x > -EPS` filter keeps midline cells (x = 0) while
                    // dropping the mirrored −X (left) cells.
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

                    // Reassemble both halves into a single OCG with
                    // contiguous indices.
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
        };

        // Stay inside the WORLD_SAFETY_MARGIN inset — same band the
        // initial cohort and `apply_world_bounds` agree on.
        let spawn_x = rng.random_range(
            crate::world_geometry::WORLD_SAFETY_MARGIN
                ..(map_size.x - crate::world_geometry::WORLD_SAFETY_MARGIN),
        );
        let spawn_z = rng.random_range(
            crate::world_geometry::WORLD_SAFETY_MARGIN
                ..(map_size.z - crate::world_geometry::WORLD_SAFETY_MARGIN),
        );
        // Spawn on the ground at the chosen XZ, with a 1.0-unit clearance
        // mirroring the initial colony spawn (`spawn_colony`). Falling back
        // to the parent's Y if the heightmap resource hasn't been inserted
        // yet keeps reproduction usable in the (rare) early-tick window
        // before the world finishes loading.
        let spawn_y = match heightmap.as_ref() {
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
            // Offspring inherit the parent's symmetry, sessility,
            // variable-form, and intelligence_level traits verbatim.
            symmetry:           organism.symmetry,
            has_variable_form:  organism.has_variable_form,
            is_sessile:         organism.is_sessile,
            intelligence_level: organism.intelligence_level,
            // Captured here so the brain pool can copy parent → child
            // weights when `assign_brains_*` runs next PreUpdate.
            // Skip the inheritance link for Level0: the sessile
            // organisms have no MLP rows to copy, and attaching the
            // marker would just leak a stale Entity reference until
            // the entity is despawned.
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
    // Reuse the shared `OrganismMaterials` resource populated by
    // `spawn_colony`. The earlier code minted three fresh
    // `StandardMaterial` assets per reproduction tick — those
    // assets leaked over the simulation's lifetime because
    // `Assets<StandardMaterial>` only GC's when the LAST strong
    // handle drops, and every spawned body part held a strong
    // copy. Over 60 minutes of reproduction every 2 s, that's
    // ~5400 dead-but-retained materials in VRAM. The same fix
    // applies to `continuous_growth.rs`.
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
            &mut commands,
            &mut meshes,
            organism_materials,
            &mut rng,
        );
        if let Some(parent) = birth.parent_for_brain_inheritance {
            // The brain pool's PreUpdate `assign_brains_*` will see
            // this component, look up the parent's slot, copy its
            // weights into the new slot, and remove the marker. If
            // the parent has died before that runs (rare), the
            // lookup fails silently and the offspring just keeps
            // the recycled slot's previous tenant's weights.
            commands.entity(child_root)
                .try_insert(crate::rl_helpers::BrainInheritance(parent));
        }

        // Lineage record on the offspring. Attached here so the
        // default `assign_lineage_records` (which only handles
        // entities without a record) skips this one — we want
        // `parent_id = Some(parent)`, not the default `None`.
        let spawn_time = virtual_time.elapsed_secs();
        commands.entity(child_root).try_insert(
            crate::organism::LineageRecord::new_offspring(
                birth.lineage_parent,
                spawn_time,
            ),
        );

        // Bump the parent's reproduction counter. The
        // `LineageRecord` mutation goes through Commands so it
        // doesn't conflict with the active `&mut Organism` query
        // (Bevy's reproduction loop holds the parent's Organism
        // borrow already). The closure handles the case where the
        // parent died before this tick's command queue flushed.
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
        // Inherit parent's species classification — the speciation
        // tick will re-evaluate next time it runs and split off a
        // new lineage if the brain genes have drifted past the
        // threshold. We patch the Organism component directly via
        // the same commands queue.
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
    /// Inherited verbatim from the parent — see the doc on
    /// `IntelligenceLevel`. Initial spawn rolls this; reproduction
    /// passes it through.
    intelligence_level: IntelligenceLevel,
    /// Parent entity, captured so that after the offspring's root
    /// entity is spawned we can attach `BrainInheritance(parent)` and
    /// the brain pool's `assign_brains_*` system will copy parent's
    /// row into the offspring's slot. `None` for Level0 offspring
    /// (no pool to inherit from).
    parent_for_brain_inheritance: Option<Entity>,
    /// Parent's species id at the moment of reproduction. Inherited
    /// verbatim onto the offspring's `Organism::species_id` field;
    /// the next speciation tick re-evaluates whether the child is
    /// still close enough to that species' centroid or needs to
    /// fork off its own. `None` if the parent hadn't been classified
    /// yet (the very first frame after initial spawn).
    parent_species_id: Option<u32>,
    /// Parent entity, captured unconditionally for the lineage
    /// `parent_id` link (distinct from `parent_for_brain_inheritance`,
    /// which is gated on intelligence level). Every offspring needs
    /// a lineage parent regardless of whether brain weights are
    /// inheritable.
    lineage_parent: Entity,
}
