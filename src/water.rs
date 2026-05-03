use bevy::prelude::*;
use crate::colony::*;
use crate::environment::WATER_LEVEL;
use crate::world_geometry::HeightmapSampler;

const WATER_COLOR: Color = Color::srgba(0.1, 0.3, 0.8, 0.35);
const BUOYANCY_STRENGTH:    f32 = 12.0;
const TRUE_WATER_DRAG_COEF: f32 = 0.05;

#[derive(Component)]
pub struct WaterPlane;

pub struct WaterPlugin;
impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (
            spawn_water_plane.run_if(resource_exists::<HeightmapSampler>),
            apply_buoyancy,
        ));
    }
}

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

    // Size the plane to cover the entire world.
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
    mut query: Query<(&Transform, &mut Organism), With<OrganismRoot>>,
) {
    let dt = time.delta_secs();

    for (transform, mut organism) in &mut query {
        if transform.translation.y >= WATER_LEVEL {
            continue;
        }

        let bounding = organism.bounding_radius().max(1.0);
        let depth      = WATER_LEVEL - transform.translation.y;
        let submersion = (depth / bounding).clamp(0.0, 1.0);

        // Buoyancy: upward force proportional to submersion.
        organism.velocity.y += BUOYANCY_STRENGTH * submersion * dt;

        // Quadratic fluid drag (F = C_d * A * v²). Frontal area scales as
        // weight^(2/3) by the square-cube law.
        let weight = organism.weight();
        let area   = weight.powf(2.0 / 3.0);

        // Drag on physical velocity (gravity / collision push-back).
        let vel_sq = organism.velocity.length_squared();
        if vel_sq > 0.001 {
            let drag_accel = TRUE_WATER_DRAG_COEF * area * vel_sq * submersion / weight;
            let current    = organism.velocity.length();
            let next_speed = (current - drag_accel * dt).max(0.0);
            organism.velocity = organism.velocity.normalize() * next_speed;
        }

        // Drag on commanded biological movement speed.
        let mov_sq = organism.movement_speed.powi(2);
        if mov_sq > 0.001 {
            let drag_accel = TRUE_WATER_DRAG_COEF * area * mov_sq * submersion / weight;
            organism.movement_speed = (organism.movement_speed - drag_accel * dt).max(0.0);
        }

        // The new architecture has only Photo and NonPhoto cell types — no
        // FinCell — so there's no per-cell thrust bonus to apply here.
    }
}
