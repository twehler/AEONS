use bevy::prelude::*;
use crate::colony::*;
use crate::environment::WATER_LEVEL;
use crate::world_geometry::HeightmapSampler;

// ── Constants ────────────────────────────────────────────────────────────────

const WATER_COLOR: Color = Color::srgba(0.1, 0.3, 0.8, 0.35);
const BUOYANCY_STRENGTH: f32 = 12.0;  // upward force when submerged
const WATER_DRAG: f32 = 0.92;          // velocity damping per frame underwater

// ── Components ───────────────────────────────────────────────────────────────

#[derive(Component)]
pub struct WaterPlane;

// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct WaterPlugin;

impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (
            spawn_water_plane,
            apply_buoyancy,
        ));
    }
}

// ── Systems ──────────────────────────────────────────────────────────────────

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

        // Submersion depth: how far below water surface
        let depth = WATER_LEVEL - transform.translation.y;
        let submersion = (depth / (organism.bounding_radius.max(1.0))).clamp(0.0, 1.0);

        // Buoyancy: upward force proportional to submersion
        organism.velocity.y += BUOYANCY_STRENGTH * submersion * dt;

        // Water drag: slow down all movement
        organism.velocity *= WATER_DRAG;
        organism.movement_speed *= WATER_DRAG;

        // FinCell bonus: organisms with fins move faster underwater
        let mut fin_thrust = 0.0f32;
        for entry in &organism.ocg[..organism.grown_cell_count] {
            fin_thrust += entry.cell_type.properties().thrust;
        }
        if fin_thrust > 0.0 {
            let thrust_bonus = fin_thrust * 0.1 * dt;
            transform.translation += organism.movement_direction * thrust_bonus;
        }
    }
}
