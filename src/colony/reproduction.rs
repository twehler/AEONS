// Reproduction.
//
// Offspring are exact copies of their parent's body plan — same body parts,
// same cell types, no mutation. Each newborn is a rhombic-dodecahedron
// organism (one cell per body part), just like every parent.
//
// The `reproduced` boolean on `Organism` enforces a species-specific
// reproduction cap: heterotrophs reproduce at most once, photoautotrophs at
// most twice.

use bevy::prelude::*;
use rand::prelude::*;

use crate::body_part::{self, Attachment};
use crate::cell::*;
use crate::colony::*;
use crate::energy::MAX_ENERGY_PER_CELL;
use crate::world_geometry::{HeightmapSampler, MAP_MAX_X, MAP_MAX_Z};


// ── Constants ────────────────────────────────────────────────────────────────

const REPRODUCTION_CHECK_INTERVAL: f32 = 2.0;

/// Energy split between parent and offspring at reproduction (50/50).
const OFFSPRING_ENERGY_FRACTION: f32 = 0.5;

/// Threshold (fraction of `max_energy`) above which an organism becomes a
/// reproduction candidate.
const REPRODUCTION_ENERGY_THRESHOLD: f32 = 0.8;

const HETEROTROPH_REPRODUCTION_CAP:    u8 = 1;
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
    mut timer:      ResMut<ReproductionTimer>,
    mut commands:   Commands,
    mut meshes:     ResMut<Assets<Mesh>>,
    mut materials:  ResMut<Assets<StandardMaterial>>,
    heightmap:      Option<Res<HeightmapSampler>>,
    smoothing:      Res<crate::simulation_settings::Smoothing>,
    mut query:      Query<
        (Entity, &mut Organism, &Transform, Has<Photoautotroph>, Has<Heterotroph>),
        With<OrganismRoot>,
    >,
) {
    timer.timer.tick(time.delta());
    if !timer.timer.just_finished() { return; }

    let current_pop = query.iter().count();
    if current_pop >= MAXIMUM_ORGANISMS { return; }
    let spawn_budget = MAXIMUM_ORGANISMS - current_pop;

    let mut pending_births: Vec<PendingBirth> = Vec::new();
    let mut rng = rand::rng();

    for (parent_entity, mut organism, transform, is_photo, is_hetero) in &mut query {
        if pending_births.len() >= spawn_budget { break; }

        if organism.reproduced { continue; }

        // Heterotroph-movement RL debug mode: heterotrophs do not reproduce
        // so the training population stays at the seeded count. Photoautotrophs
        // still reproduce normally so prey continues to repopulate.
        if is_hetero
            && !crate::simulation_settings::HETEROTROPH_MOVEMENT_AI_DEBUGGING
        {
            continue;
        }

        let max_energy = (organism.grown_cell_count() as f32) * MAX_ENERGY_PER_CELL;
        if max_energy <= 0.0 { continue; }
        if organism.energy < max_energy * REPRODUCTION_ENERGY_THRESHOLD { continue; }

        let offspring_energy = max_energy * OFFSPRING_ENERGY_FRACTION;
        organism.energy      -= offspring_energy;
        organism.reproductions = organism.reproductions.saturating_add(1);

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
                // Bilateral organisms have ONE body part whose OCG
                // contains both halves (right cells with x > 0 plus
                // their mirrors). Growth: extract the right half from
                // the combined OCG, mutate it (constrained to
                // x ≥ MIN_X_BILATERAL), mirror, and reassemble. The
                // combined-OCG mesh build then welds the seam and
                // drops interior faces automatically. No branching
                // for bilateral organisms.
                let mut parts = organism.body_parts.clone();
                if parts.is_empty() { continue; }

                // Extract right-half cells with fresh sequential
                // indices — `mutate_bilateral` expects a contiguous
                // [0..N) ledger as input.
                let right_ocg: Vec<(usize, Vec3, CellType)> = parts[0].ocg.iter()
                    .filter(|(_, p, _)| p.x > 0.0)
                    .enumerate()
                    .map(|(i, (_, p, ct))| (i, *p, *ct))
                    .collect();
                if right_ocg.is_empty() { continue; }

                let Some(grown_right) =
                    crate::mutation::mutate_bilateral(&right_ocg, &mut rng)
                    else { continue; };
                let grown_left = body_part::mirror_ocg_x(&grown_right);

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
        };

        let spawn_x = rng.random_range(0.0..MAP_MAX_X);
        let spawn_z = rng.random_range(0.0..MAP_MAX_Z);
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
        });
    }

    // Skip the asset registration entirely on ticks where no births
    // are pending. Each `OrganismMaterials::new` call inserts 3
    // StandardMaterial assets — cheap individually but every 2-second
    // tick was creating them unconditionally.
    if pending_births.is_empty() { return; }
    let organism_materials = OrganismMaterials::new(&mut materials);

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
            &organism_materials,
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
}
