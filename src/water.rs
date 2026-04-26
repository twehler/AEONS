use bevy::prelude::*;
use crate::colony::*;
use crate::environment::WATER_LEVEL;
use crate::world_geometry::HeightmapSampler;

// ── Constants ────────────────────────────────────────────────────────────────

const WATER_COLOR: Color = Color::srgba(0.1, 0.3, 0.8, 0.35);
const BUOYANCY_STRENGTH: f32 = 12.0;  

// NEW: True aerodynamic/hydrodynamic drag coefficient
const TRUE_WATER_DRAG_COEF: f32 = 0.05; 

// ── Components & Plugin (Keep your existing WaterPlane and Plugin) ───────────
#[derive(Component)]
pub struct WaterPlane;

pub struct WaterPlugin;
impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (spawn_water_plane, apply_buoyancy));
    }
}

// ── Systems ──────────────────────────────────────────────────────────────────

// (Keep your existing spawn_water_plane system here!)
// fn spawn_water_plane(...) { ... }

fn spawn_water_plane(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    heightmap: Res<HeightmapSampler>,
    query: Query<&WaterPlane>,
    mut spawned: Local<bool>,
) {
    if *spawned { return; }
    if !query.is_empty() { *spawned = true; return; }
    *spawned = true;

    // Size the plane to cover the entire world
    let world_width = heightmap.width as f32;
    let world_depth = heightmap.depth as f32;
    let center_x = heightmap.min_x as f32 + world_width / 2.0;
    let center_z = heightmap.min_z as f32 + world_depth / 2.0;

    commands.spawn((
        Mesh3d(meshes.add(Plane3d::new(Vec3::Y, Vec2::new(world_width / 2.0, world_depth / 2.0)))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: WATER_COLOR,
            alpha_mode: AlphaMode::Blend,
            double_sided: true,
            cull_mode: None,
            perceptual_roughness: 0.1,
            metallic: 0.0,
            ..default()
        })),
        Transform::from_translation(Vec3::new(center_x, WATER_LEVEL, center_z)),
        WaterPlane,
    ));
}



fn apply_buoyancy(
    time: Res<Time>,
    mut query: Query<(&mut Transform, &mut Organism), With<OrganismRoot>>,
) {
    let dt = time.delta_secs();

    for (mut transform, mut organism) in &mut query {
        if transform.translation.y >= WATER_LEVEL {
            continue;
        }

        let depth = WATER_LEVEL - transform.translation.y;
        let submersion = (depth / organism.bounding_radius.max(1.0)).clamp(0.0, 1.0);

        // Buoyancy: upward force proportional to submersion
        organism.velocity.y += BUOYANCY_STRENGTH * submersion * dt;

        // ──────────────────────────────────────────────────────────────────
        // NEW: Physically Accurate Fluid Drag (F = C_d * Area * V^2)
        // ──────────────────────────────────────────────────────────────────
        let weight = organism.weight.max(1.0);
        let area = weight.powf(2.0 / 3.0); // Square-Cube law for frontal area

        // 1. Apply drag to physical gravity/push velocity
        let vel_sq = organism.velocity.length_squared();
        if vel_sq > 0.001 {
            let drag_force = TRUE_WATER_DRAG_COEF * area * vel_sq * submersion;
            let drag_accel = drag_force / weight; // Newton's Second Law: a = F/m
            
            let current_speed = organism.velocity.length();
            let new_speed = (current_speed - drag_accel * dt).max(0.0);
            organism.velocity = organism.velocity.normalize() * new_speed;
        }

        // 2. Apply drag to biological movement speed 
        let mov_sq = organism.movement_speed.powi(2);
        if mov_sq > 0.001 {
            let drag_force = TRUE_WATER_DRAG_COEF * area * mov_sq * submersion;
            let drag_accel = drag_force / weight; 
            
            organism.movement_speed = (organism.movement_speed - drag_accel * dt).max(0.0);
        }

        // FinCell bonus: organisms with fins move faster underwater
        let mut fin_thrust = 0.0f32;
        // Using iter().take() to safely read only the grown cells
        for entry in organism.ocg.iter().take(organism.grown_cell_count) {
            fin_thrust += entry.cell_type.properties().thrust;
        }
        
        // Add fin thrust back into the movement speed
        organism.movement_speed += fin_thrust * submersion * dt;
    }
}
