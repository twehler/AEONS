use bevy::prelude::*;
use crate::colony::*;
use crate::cell::CellType;

// ── Constants ────────────────────────────────────────────────────────────────

const PREDATION_CHECK_INTERVAL: f32 = 0.5;
const PREDATION_RANGE: f32 = 15.0;      // organism centers must be this close
const BITE_DAMAGE: f32 = 5.0;            // base damage per predation tick
const ENERGY_TRANSFER_RATE: f32 = 0.1;   // 10% trophic efficiency on damage dealt

// ── Timer resource ───────────────────────────────────────────────────────────

#[derive(Resource)]
pub struct PredationTimer {
    pub timer: Timer,
}

impl Default for PredationTimer {
    fn default() -> Self {
        Self {
            timer: Timer::from_seconds(PREDATION_CHECK_INTERVAL, TimerMode::Repeating),
        }
    }
}

// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct PredationPlugin;

impl Plugin for PredationPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(PredationTimer::default());
        app.add_systems(Last, predation_system);
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

// Count total digestion power (from GutCells) and armor (from HardCells) in an organism
fn organism_stats(organism: &Organism) -> (f32, f32) {
    let mut digestion = 0.0f32;
    let mut armor = 0.0f32;
    for entry in &organism.ocg[..organism.grown_cell_count] {
        let props = entry.cell_type.properties();
        digestion += props.digestion;
        armor += props.armor;
    }
    (digestion, armor)
}

// ── System ───────────────────────────────────────────────────────────────────

fn predation_system(
    time: Res<Time>,
    mut timer: ResMut<PredationTimer>,
    mut query: Query<(Entity, &mut Organism, &Transform), With<OrganismRoot>>,
) {
    timer.timer.tick(time.delta());
    if !timer.timer.just_finished() {
        return;
    }

    // Snapshot positions and stats to avoid borrow conflicts
    let snapshots: Vec<(Entity, Vec3, f32, f32, f32)> = query.iter()
        .map(|(entity, organism, transform)| {
            let (digestion, armor) = organism_stats(organism);
            (entity, transform.translation, digestion, armor, organism.energy)
        })
        .collect();

    // Check all pairs — predator needs digestion > 0
    let mut energy_changes: Vec<(Entity, f32)> = Vec::new();

    for i in 0..snapshots.len() {
        let (entity_a, pos_a, dig_a, armor_a, energy_a) = snapshots[i];

        for j in (i + 1)..snapshots.len() {
            let (entity_b, pos_b, dig_b, armor_b, energy_b) = snapshots[j];

            let dist = pos_a.distance(pos_b);
            if dist >= PREDATION_RANGE {
                continue;
            }

            // A attacks B (if A has digestion)
            if dig_a > 0.0 && energy_b > 0.0 {
                let effective_damage = (BITE_DAMAGE * dig_a * (1.0 - armor_b * 0.5)).max(0.0);
                let actual_damage = effective_damage.min(energy_b);
                let gained = actual_damage * ENERGY_TRANSFER_RATE;
                energy_changes.push((entity_b, -actual_damage));
                energy_changes.push((entity_a, gained));
            }

            // B attacks A (if B has digestion)
            if dig_b > 0.0 && energy_a > 0.0 {
                let effective_damage = (BITE_DAMAGE * dig_b * (1.0 - armor_a * 0.5)).max(0.0);
                let actual_damage = effective_damage.min(energy_a);
                let gained = actual_damage * ENERGY_TRANSFER_RATE;
                energy_changes.push((entity_a, -actual_damage));
                energy_changes.push((entity_b, gained));
            }
        }
    }

    // Apply energy changes
    for (entity, delta) in energy_changes {
        if let Ok((_, mut organism, _)) = query.get_mut(entity) {
            organism.energy = (organism.energy + delta).max(0.0);
        }
    }
}
