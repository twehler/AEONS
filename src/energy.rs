use bevy::prelude::*;
use crate::colony::*;
use crate::cell::CellType;

// ── Constants ────────────────────────────────────────────────────────────────

const ENERGY_TICK_INTERVAL: f32 = 0.5; // seconds between energy updates
const METABOLIC_EXPONENT: f32 = 1.1;   // Kleiber's law scaling
const METABOLIC_BASE: f32 = 0.1;       // base metabolic rate multiplier
const MAX_ENERGY_PER_CELL: f32 = 10.0; // energy capacity per cell
const LIGHT_LEVEL: f32 = 1.0;          // global light multiplier (future: day/night)
const CORPSE_ENERGY_FRACTION: f32 = 0.5; // fraction of max energy left as corpse
const CORPSE_DECAY_TIME: f32 = 30.0;   // seconds before corpse despawns

// ── Components ───────────────────────────────────────────────────────────────

#[derive(Component)]
pub struct Corpse {
    pub energy: f32,
    pub decay_timer: Timer,
}

// ── Timer resource ───────────────────────────────────────────────────────────

#[derive(Resource)]
pub struct EnergyTickTimer {
    pub timer: Timer,
}

impl Default for EnergyTickTimer {
    fn default() -> Self {
        Self {
            timer: Timer::from_seconds(ENERGY_TICK_INTERVAL, TimerMode::Repeating),
        }
    }
}

// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct EnergyPlugin;

impl Plugin for EnergyPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(EnergyTickTimer::default());
        app.add_systems(Update, (
            energy_tick,
            corpse_decay,
        ));
    }
}

// ── Systems ──────────────────────────────────────────────────────────────────

fn energy_tick(
    time: Res<Time>,
    mut timer: ResMut<EnergyTickTimer>,
    mut commands: Commands,
    mut query: Query<(Entity, &mut Organism, &Transform), With<OrganismRoot>>,
) {
    timer.timer.tick(time.delta());
    if !timer.timer.just_finished() {
        return;
    }

    let dt = ENERGY_TICK_INTERVAL;

    for (entity, mut organism, transform) in &mut query {
        let cell_count = organism.grown_cell_count as f32;
        if cell_count < 1.0 { continue; }

        // ── Max energy capacity ──────────────────────────────────────────
        let max_energy = cell_count * MAX_ENERGY_PER_CELL;

        // ── Metabolic cost (Kleiber's law: scales as N^1.1) ─────────────
        let metabolic_cost = METABOLIC_BASE * cell_count.powf(METABOLIC_EXPONENT) * dt;

        // ── Per-cell maintenance cost ────────────────────────────────────
        let mut maintenance_cost = 0.0f32;
        let mut photo_income = 0.0f32;

        for entry in &organism.ocg[..organism.grown_cell_count] {
            let props = entry.cell_type.properties();
            maintenance_cost += props.energy_cost * dt;
            photo_income += props.photosynthesis * LIGHT_LEVEL * dt;
        }

        // ── Apply energy changes ─────────────────────────────────────────
        organism.energy += photo_income;
        organism.energy -= metabolic_cost + maintenance_cost;
        organism.energy = organism.energy.clamp(0.0, max_energy);

        // ── Starvation → death ───────────────────────────────────────────
        if organism.energy <= 0.0 {
            let corpse_energy = max_energy * CORPSE_ENERGY_FRACTION;
            commands.entity(entity).remove::<Organism>();
            commands.entity(entity).remove::<OrganismRoot>();
            commands.entity(entity).insert(Corpse {
                energy: corpse_energy,
                decay_timer: Timer::from_seconds(CORPSE_DECAY_TIME, TimerMode::Once),
            });
        }
    }
}

fn corpse_decay(
    time: Res<Time>,
    mut commands: Commands,
    mut query: Query<(Entity, &mut Corpse)>,
) {
    for (entity, mut corpse) in &mut query {
        corpse.decay_timer.tick(time.delta());
        if corpse.decay_timer.just_finished() {
            commands.entity(entity).despawn();
        }
    }
}
