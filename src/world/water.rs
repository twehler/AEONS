use bevy::prelude::*;
use crate::colony::*;
use crate::environment::WaterLevel;
use crate::world_geometry::{HeightmapSampler, HEIGHTMAP_CELL_SIZE};
use bevy::light::NotShadowCaster;

const WATER_COLOR: Color = Color::srgba(0.1, 0.3, 0.8, 0.35);
use crate::simulation_settings::{BUOYANCY_STRENGTH, TRUE_WATER_DRAG_COEF};

#[derive(Component)]
pub struct WaterPlane;

pub struct WaterPlugin;
impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        // `init_resource` is idempotent: keeps any argv-/colony-set value.
        app.init_resource::<WaterLevel>();
        app.add_systems(Update, (
            spawn_water_plane.run_if(resource_exists::<HeightmapSampler>),
            // Move the visible plane when a colony load (or UI) changes the level.
            reposition_water_plane.run_if(resource_changed::<WaterLevel>),
            apply_buoyancy,
        ));
    }
}

fn spawn_water_plane(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    heightmap: Res<HeightmapSampler>,
    water: Res<WaterLevel>,
    query: Query<&WaterPlane>,
    mut spawned: Local<bool>,
) {
    if *spawned { return; }
    if !query.is_empty() { *spawned = true; return; }
    *spawned = true;

    // Cover the whole world. heightmap.{width,depth,min_x,min_z} are in CELL
    // units; multiply by HEIGHTMAP_CELL_SIZE for world units.
    let world_width = heightmap.width as f32 * HEIGHTMAP_CELL_SIZE;
    let world_depth = heightmap.depth as f32 * HEIGHTMAP_CELL_SIZE;
    let center_x = heightmap.min_x as f32 * HEIGHTMAP_CELL_SIZE + world_width / 2.0;
    let center_z = heightmap.min_z as f32 * HEIGHTMAP_CELL_SIZE + world_depth / 2.0;

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
        Transform::from_translation(Vec3::new(center_x, water.0, center_z)),
        // World-spanning transparent plane: worst case for shadow overdraw and
        // cascade fitting, and shadows nothing meaningfully.
        NotShadowCaster,
        WaterPlane,
    ));
}


/// Keep the visible water plane's Y in sync with the `WaterLevel` resource
/// (a colony load or UI edit changes it). Gated by `resource_changed`.
fn reposition_water_plane(
    water: Res<WaterLevel>,
    mut query: Query<&mut Transform, With<WaterPlane>>,
) {
    for mut transform in &mut query {
        transform.translation.y = water.0;
    }
}


fn apply_buoyancy(
    time: Res<Time>,
    water: Res<WaterLevel>,
    mut query: Query<(&Transform, &mut Organism), With<OrganismRoot>>,
) {
    let dt = time.delta_secs();
    let water_y = water.0;

    // Per-organism buoyancy/drag is entity-disjoint (own velocity + speed, no
    // Commands), so it fans out over `ComputeTaskPool`.
    query.par_iter_mut().for_each(|(transform, mut organism)| {
        // GROUND-ANCHORED organisms (land AND ocean-floor sliders + phototrophs)
        // never float: gravity sinks them and the heightmap floor-clamp holds
        // them on the terrain — exactly like a land slider. Buoyancy is for
        // WATER-BASED bodies (floating algae / swimmers) only; applying it to a
        // submerged benthic organism is what made ocean-floor life rise to the
        // surface instead of staying on the seafloor.
        if organism.ground_based {
            return;
        }
        if transform.translation.y >= water_y {
            return;
        }

        let bounding = organism.bounding_radius().max(1.0);
        let depth      = water_y - transform.translation.y;
        let submersion = (depth / bounding).clamp(0.0, 1.0);

        // Buoyancy: upward force proportional to submersion.
        organism.velocity.y += BUOYANCY_STRENGTH * submersion * dt;

        // Quadratic fluid drag (F = C_d·A·v²); frontal area ~ weight^(2/3) by
        // the square-cube law.
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

        // No FinCell type exists, so no per-cell thrust bonus applies here.
    });
}
