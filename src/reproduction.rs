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

use crate::cell::*;
use crate::colony::*;
use crate::energy::MAX_ENERGY_PER_CELL;
use crate::world_geometry::{MAP_MAX_X, MAP_MAX_Z};


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


// ── Inheritance ──────────────────────────────────────────────────────────────

/// Build a child genome by cloning each alive parent body part verbatim.
fn inherit_body_parts(parent: &[BodyPart]) -> Vec<BodyPart> {
    parent.iter()
        .filter(|b| b.is_alive())
        .cloned()
        .collect()
}


// ── Reproduction system ──────────────────────────────────────────────────────

fn reproduction_system(
    time:           Res<Time>,
    mut timer:      ResMut<ReproductionTimer>,
    mut commands:   Commands,
    mut meshes:     ResMut<Assets<Mesh>>,
    mut materials:  ResMut<Assets<StandardMaterial>>,
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

    for (_entity, mut organism, transform, is_photo, is_hetero) in &mut query {
        if pending_births.len() >= spawn_budget { break; }

        if organism.reproduced { continue; }

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

        let child_parts = inherit_body_parts(&organism.body_parts);
        if child_parts.is_empty() { continue; }

        let spawn_x = rng.random_range(0.0..MAP_MAX_X);
        let spawn_z = rng.random_range(0.0..MAP_MAX_Z);
        let spawn_pos = Vec3::new(spawn_x, transform.translation.y, spawn_z);

        pending_births.push(PendingBirth {
            pos:    spawn_pos,
            parts:  child_parts,
            energy: offspring_energy,
            kind,
        });
    }

    let shared_material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        ..default()
    });

    for birth in pending_births {
        spawn_organism(
            birth.pos,
            birth.parts,
            birth.kind,
            birth.energy,
            &mut commands,
            &mut meshes,
            &shared_material,
            &mut rng,
        );
    }
}


struct PendingBirth {
    pos:    Vec3,
    parts:  Vec<BodyPart>,
    energy: f32,
    kind:   OrganismKind,
}
