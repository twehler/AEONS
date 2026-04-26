use bevy::prelude::*;
use crate::colony::*;
use crate::cell::CellType;
use crate::environment::WATER_LEVEL; // NEW: Needed to determine fluid vs ground drag

// ── Constants ─────────────────────────────────────────────────────────────────

const ENERGY_TICK_INTERVAL: f32 = 0.5;
pub const MAX_ENERGY_PER_CELL: f32 = 10.0;

const PHOTO_PRODUCTION_PER_CELL: f32 = 0.2;    
const NON_PHOTO_CONSUMPTION_PER_CELL: f32 = 0.4; 

// NEW: Biomechanical movement cost constants
// These are tuned so that max speed (20) doesn't instantly kill them, 
// but still heavily punishes sprinting, especially for heavy creatures.
const K_GROUND_FRICTION: f32 = 0.05;  
const K_FLUID_DRAG: f32 = 0.0005;     

// ── Helper Functions ──────────────────────────────────────────────────────────

pub fn get_max_energy(organism: &Organism) -> f32 {
    organism.grown_cell_count as f32 * MAX_ENERGY_PER_CELL
}

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
    // Added &Transform to accurately measure submersion depth
    mut organisms: Query<(Entity, &mut Organism, &Transform), With<OrganismRoot>>,
) {
    timer.timer.tick(time.delta());
    if !timer.timer.just_finished() {
        return;
    }

    for (entity, mut organism, transform) in organisms.iter_mut() {
        let max_energy = get_max_energy(&organism);

        // 1. Tally the cells
        let mut photo_count = 0;
        let mut non_photo_count = 0;

        for (_, cell_type) in &organism.active_cells {
            if *cell_type == CellType::PhotoCell {
                photo_count += 1;
            } else {
                non_photo_count += 1;
            }
        }

        // 2. Calculate Submersion (0.0 = completely on ground, 1.0 = completely underwater)
        let depth = WATER_LEVEL - transform.translation.y;
        let submersion = (depth / organism.bounding_radius.max(1.0)).clamp(0.0, 1.0);
        let ground_fraction = 1.0 - submersion;

        // 3. Movement Energy Cost (Power = Force x Velocity)
        let speed = organism.movement_speed; // The effort the organism is exerting
        let weight = organism.weight.max(1.0); // Safety against NaN from zero-weight

        // Friction: scales linearly with Weight and Speed
        let friction_power = K_GROUND_FRICTION * weight * speed;
        let friction_cost = ground_fraction * friction_power * ENERGY_TICK_INTERVAL;

        // Fluid Drag: scales with Weight^(2/3) [Area] and Speed^3
        let fluid_power = K_FLUID_DRAG * weight.powf(2.0 / 3.0) * speed.powi(3);
        let fluid_cost = submersion * fluid_power * ENERGY_TICK_INTERVAL;

        let total_movement_cost = friction_cost + fluid_cost;

        // 4. Calculate net energy shift
        let production = photo_count as f32 * PHOTO_PRODUCTION_PER_CELL;
        let consumption = (non_photo_count as f32 * NON_PHOTO_CONSUMPTION_PER_CELL) + total_movement_cost;

        // 5. Apply physics
        organism.energy += production - consumption;
        organism.energy = organism.energy.clamp(0.0, max_energy);

        // 6. Starvation Check
        if organism.energy <= 0.0 {
            commands.entity(entity).despawn();
        }
    }
}
