use bevy::prelude::*;
use crate::colony::*;

// ── Constants ─────────────────────────────────────────────────────────────────

const ENERGY_TICK_INTERVAL: f32 = 0.5;   // seconds between energy updates
const MAX_ENERGY_PER_CELL: f32 = 10.0;   // energy capacity per grown cell
const PHOTO_INCOME: f32 = 2.0;           // constant energy gained by phototrophs per tick
const HETERO_STARVATION: f32 = 1.0;      // constant energy lost by heterotrophs per tick

// ── Timer Resource ────────────────────────────────────────────────────────────

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

// ── Plugin ────────────────────────────────────────────────────────────────────

pub struct EnergyPlugin;

impl Plugin for EnergyPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EnergyTickTimer>();
        app.add_systems(Update, manage_energy);
    }
}

// ── Systems ───────────────────────────────────────────────────────────────────

fn manage_energy(
    mut commands: Commands,
    time: Res<Time>,
    mut timer: ResMut<EnergyTickTimer>,
    mut photos: Query<&mut Organism, (With<Photoautotroph>, Without<Heterotroph>)>,
    mut heteros: Query<(Entity, &mut Organism), With<Heterotroph>>,
) {
    timer.timer.tick(time.delta());
    if !timer.timer.just_finished() {
        return;
    }

    // 1. Phototrophs get constant energy from photosynthesis
    for mut organism in photos.iter_mut() {
        let max_energy = organism.grown_cell_count as f32 * MAX_ENERGY_PER_CELL;
        organism.energy += PHOTO_INCOME;
        organism.energy = organism.energy.clamp(0.0, max_energy);
    }

    // 2. Heterotrophs starve over time at a constant rate
    for (entity, mut organism) in heteros.iter_mut() {
        let max_energy = organism.grown_cell_count as f32 * MAX_ENERGY_PER_CELL;
        organism.energy -= HETERO_STARVATION;
        organism.energy = organism.energy.clamp(0.0, max_energy);

        // If they starve completely, they despawn and die
        if organism.energy <= 0.0 {
            commands.entity(entity).despawn();
        }
    }
}
