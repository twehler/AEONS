use bevy::prelude::*;
use crate::colony::*;
use crate::cell::CellType;

// ── Constants ─────────────────────────────────────────────────────────────────

const ENERGY_TICK_INTERVAL: f32  = 0.5;  // seconds between energy updates
const METABOLIC_EXPONENT:   f32  = 1.1;  // Kleiber's law body-size scaling
const METABOLIC_BASE:       f32  = 0.1;  // base metabolic rate multiplier
const MAX_ENERGY_PER_CELL:  f32  = 10.0; // energy capacity per grown cell
const CORPSE_ENERGY_FRACTION: f32 = 0.5; // fraction of max energy stored in corpse
const CORPSE_DECAY_TIME:    f32  = 30.0; // seconds before a corpse despawns

/// Global light multiplier applied to all PhotoCell income.
/// 1.0 = full daylight. Set < 1.0 to simulate shade or night.
const LIGHT_LEVEL: f32 = 1.0;

// ── Components ────────────────────────────────────────────────────────────────

/// Spawned when an organism starves. Carries stored energy that predation.rs
/// can transfer to a feeding heterotroph, then decays after CORPSE_DECAY_TIME.
#[derive(Component)]
pub struct Corpse {
    pub energy:      f32,
    pub decay_timer: Timer,
}

// ── Timer resource ────────────────────────────────────────────────────────────

#[derive(Resource)]
pub struct EnergyTickTimer {
    pub timer: Timer,
}

impl Default for EnergyTickTimer {
    fn default() -> Self {
        Self { timer: Timer::from_seconds(ENERGY_TICK_INTERVAL, TimerMode::Repeating) }
    }
}

// ── Plugin ────────────────────────────────────────────────────────────────────

pub struct EnergyPlugin;

impl Plugin for EnergyPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(EnergyTickTimer::default());
        app.add_systems(Update, (
            photoautotroph_energy_tick,
            heterotroph_energy_tick,
            corpse_decay,
        ));
    }
}

// ── Photoautotroph energy system ──────────────────────────────────────────────
//
// Income:  PhotoCells perform photosynthesis scaled by LIGHT_LEVEL.
//          Each PhotoCell has photosynthesis = 0.5 (from cell.rs).
//          Income per tick = Σ(photosynthesis × LIGHT_LEVEL) × dt
//
// Costs:   Per-cell maintenance (cell_type.properties().energy_cost × dt)
//          + Kleiber metabolic overhead (METABOLIC_BASE × N^1.1 × dt)
//
// Death:   When energy reaches 0 the organism is despawned and a Corpse is left.
//
fn photoautotroph_energy_tick(
    time:     Res<Time>,
    mut timer: ResMut<EnergyTickTimer>,
    mut commands: Commands,
    mut query: Query<(Entity, &mut Organism, &Transform), (With<OrganismRoot>, With<Photoautotroph>)>,
) {
    // The timer is shared with the heterotroph system; only one of the two
    // systems advances it — we let photoautotroph_energy_tick own the tick.
    // (heterotroph_energy_tick checks just_finished without ticking again.)
    timer.timer.tick(time.delta());
    if !timer.timer.just_finished() { return; }

    let dt = ENERGY_TICK_INTERVAL;

    for (entity, mut organism, transform) in &mut query {
        let cell_count = organism.grown_cell_count as f32;
        if cell_count < 1.0 { continue; }

        let max_energy = cell_count * MAX_ENERGY_PER_CELL;

        // ── Metabolic overhead ────────────────────────────────────────────
        let metabolic_cost = METABOLIC_BASE * cell_count.powf(METABOLIC_EXPONENT) * dt;

        // ── Per-cell photosynthesis income and maintenance cost ────────────
        let mut photo_income     = 0.0_f32;
        let mut maintenance_cost = 0.0_f32;

        for entry in &organism.ocg[..organism.grown_cell_count] {
            let props = entry.cell_type.properties();
            photo_income     += props.photosynthesis * LIGHT_LEVEL * dt;
            maintenance_cost += props.energy_cost * dt;
        }

        organism.energy += photo_income;
        organism.energy -= metabolic_cost + maintenance_cost;
        organism.energy  = organism.energy.clamp(0.0, max_energy);

        // ── Starvation → corpse ───────────────────────────────────────────
        if organism.energy <= 0.0 {
            let corpse_energy = cell_count * MAX_ENERGY_PER_CELL * CORPSE_ENERGY_FRACTION;
            commands.entity(entity).despawn();
            commands.spawn((
                Transform::from_translation(transform.translation),
                Corpse {
                    energy:      corpse_energy,
                    decay_timer: Timer::from_seconds(CORPSE_DECAY_TIME, TimerMode::Once),
                },
            ));
        }
    }
}

// ── Heterotroph energy system ─────────────────────────────────────────────────
//
// Income:  Heterotrophs have no PhotoCells, so photosynthesis = 0.
//          Their energy income comes entirely from predation.rs (which reads
//          GutCell digestion values to calculate how much prey-energy is absorbed
//          per contact tick). This system handles only the cost side.
//
// Costs:   Same structure as photoautotrophs — per-cell maintenance + metabolic.
//          HardCell, FinCell, FootCell, GutCell, RedCell all pay energy_cost × dt.
//
// Death:   Starvation works identically — despawn + Corpse.
//          Heterotrophs can also die from being consumed by a larger predator;
//          that is handled externally by predation.rs calling commands.entity().despawn().
//
fn heterotroph_energy_tick(
    // NOTE: does NOT tick the timer — photoautotroph_energy_tick does that.
    timer: Res<EnergyTickTimer>,
    mut commands: Commands,
    mut query: Query<(Entity, &mut Organism, &Transform), (With<OrganismRoot>, With<Heterotroph>)>,
) {
    if !timer.timer.just_finished() { return; }

    let dt = ENERGY_TICK_INTERVAL;

    for (entity, mut organism, transform) in &mut query {
        let cell_count = organism.grown_cell_count as f32;
        if cell_count < 1.0 { continue; }

        let max_energy = cell_count * MAX_ENERGY_PER_CELL;

        // ── Metabolic overhead ────────────────────────────────────────────
        let metabolic_cost = METABOLIC_BASE * cell_count.powf(METABOLIC_EXPONENT) * dt;

        // ── Per-cell maintenance cost (no photo income) ───────────────────
        let mut maintenance_cost = 0.0_f32;
        for entry in &organism.ocg[..organism.grown_cell_count] {
            maintenance_cost += entry.cell_type.properties().energy_cost * dt;
        }

        // No photosynthesis income — heterotrophs must hunt to survive.
        // TEMPORARILY DISABLED FOR AI-TRAINING
        // organism.energy -= metabolic_cost + maintenance_cost;
        organism.energy = max_energy;
        organism.energy  = organism.energy.clamp(0.0, max_energy);

        // ── Starvation → corpse ───────────────────────────────────────────
        if organism.energy <= 0.0 {
            let corpse_energy = cell_count * MAX_ENERGY_PER_CELL * CORPSE_ENERGY_FRACTION;
            commands.entity(entity).despawn();
            commands.spawn((
                Transform::from_translation(transform.translation),
                Corpse {
                    energy:      corpse_energy,
                    decay_timer: Timer::from_seconds(CORPSE_DECAY_TIME, TimerMode::Once),
                },
            ));
        }
    }
}

// ── Corpse decay ──────────────────────────────────────────────────────────────

fn corpse_decay(
    time:         Res<Time>,
    mut commands: Commands,
    mut query:    Query<(Entity, &mut Corpse)>,
) {
    for (entity, mut corpse) in &mut query {
        corpse.decay_timer.tick(time.delta());
        if corpse.decay_timer.just_finished() {
            commands.entity(entity).despawn();
        }
    }
} 
