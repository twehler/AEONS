use bevy::prelude::*;
use bevy::transform::TransformSystems;

use crate::world_geometry::{HeightmapSampler, BlockWorld};
use crate::colony::*;
use crate::cell::{CellType, GLOBAL_CELL_SIZE};
use std::collections::HashSet;
use rand::prelude::*;
use crate::organism_collision;

const MIN_DIRECTION_INTERVAL: f32 = 1.0;
const MAX_DIRECTION_INTERVAL: f32 = 10.0;

pub struct MovementPlugin {
    pub mode: MovementMode,
}

impl MovementPlugin {
    pub fn with_mode(mode: MovementMode) -> Self {
        Self { mode }
    }
}

impl Plugin for MovementPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(self.mode);
        app.insert_resource(organism_collision::OrganismCollisionTimer::default());

        app.add_systems(PostUpdate, (
            apply_movement,
            apply_rotation // NEW: Apply rotation smoothly alongside movement
        ).before(TransformSystems::Propagate));
        
        app.add_systems(PostUpdate, (
            apply_floor_collision,
            apply_world_bounds,
        ).chain().after(TransformSystems::Propagate));

        app.add_systems(Last, organism_collision::apply_organism_collision);

        match self.mode {
            MovementMode::TwoD => {
                // NEW: Add random_2d_rotation to 2D Mode
                app.add_systems(PostUpdate, (random_2d_direction, random_2d_rotation).before(apply_movement));
                app.add_systems(PostUpdate, apply_gravity
                    .after(random_2d_direction)
                    .before(apply_movement));
            }
            MovementMode::ThreeD => {
                // NEW: Add random_3d_rotation to 3D Mode
                app.add_systems(PostUpdate, (random_3d_direction, random_3d_rotation).before(apply_movement));
            }
        }
    }
}

// ── Movement mode selection (mutually exclusive) ─────────────────────────

#[derive(Resource, PartialEq, Clone, Copy)]
pub enum MovementMode {
    TwoD,   
    ThreeD, 
}

impl Default for MovementMode {
    fn default() -> Self {
        Self::TwoD 
    }
}

// ── System: Random 2D direction (XZ plane only) ──────────────────────────

fn random_2d_direction(
    time: Res<Time>,
    mut query: Query<(&mut Organism, &mut DirectionTimer), With<OrganismRoot>>,
) {
    let dt = time.delta();
    for (mut organism, mut timer) in &mut query {
        timer.timer.tick(dt);
        if timer.timer.just_finished() {
            let angle = rand::random::<f32>() * std::f32::consts::TAU;
            organism.movement_direction = Vec3::new(angle.cos(), 0.0, angle.sin()).normalize();
            organism.movement_speed = rand::random::<f32>() * 20.0;
            let new_interval = MIN_DIRECTION_INTERVAL + rand::random::<f32>() * (MAX_DIRECTION_INTERVAL - MIN_DIRECTION_INTERVAL);
            timer.timer = Timer::from_seconds(new_interval, TimerMode::Repeating);
        }
    }
}

// ── NEW System: Random 2D Rotation (Yaw only) ────────────────────────────

fn random_2d_rotation(
    time: Res<Time>,
    mut query: Query<(&mut Organism, &mut RotationTimer), With<OrganismRoot>>,
) {
    let dt = time.delta();
    for (mut organism, mut timer) in &mut query {
        timer.timer.tick(dt);
        if timer.timer.just_finished() {
            // Pick a random angle around the Y-axis
            let angle = rand::random::<f32>() * std::f32::consts::TAU;
            organism.target_rotation = Quat::from_rotation_y(angle);
            
            let new_interval = MIN_DIRECTION_INTERVAL + rand::random::<f32>() * (MAX_DIRECTION_INTERVAL - MIN_DIRECTION_INTERVAL);
            timer.timer = Timer::from_seconds(new_interval, TimerMode::Repeating);
        }
    }
}

// ── System: Random 3D direction (full XYZ, no gravity) ───────────────────

fn random_3d_direction(
    time: Res<Time>,
    mut query: Query<(&mut Organism, &mut DirectionTimer), With<OrganismRoot>>,
) {
    let dt = time.delta();
    for (mut organism, mut timer) in &mut query {
        timer.timer.tick(dt);
        if timer.timer.just_finished() {
            let theta = rand::random::<f32>() * std::f32::consts::TAU;
            let phi = rand::random::<f32>() * std::f32::consts::PI;
            
            let x = theta.cos() * phi.sin();
            let y = phi.cos();
            let z = theta.sin() * phi.sin();
            
            organism.movement_direction = Vec3::new(x, y, z).normalize();
            organism.movement_speed = rand::random::<f32>() * 20.0;

            let new_interval = MIN_DIRECTION_INTERVAL + rand::random::<f32>() * (MAX_DIRECTION_INTERVAL - MIN_DIRECTION_INTERVAL);
            timer.timer = Timer::from_seconds(new_interval, TimerMode::Repeating);
        }
    }
}

// ── NEW System: Random 3D Rotation (Full XYZ axes) ───────────────────────

fn random_3d_rotation(
    time: Res<Time>,
    mut query: Query<(&mut Organism, &mut RotationTimer), With<OrganismRoot>>,
) {
    let dt = time.delta();
    for (mut organism, mut timer) in &mut query {
        timer.timer.tick(dt);
        if timer.timer.just_finished() {
            // Generate a random Quat rotation for full 3D tumbling/spinning
            let u = rand::random::<f32>();
            let v = rand::random::<f32>();
            let w = rand::random::<f32>();
            organism.target_rotation = Quat::from_euler(
                EulerRot::XYZ, 
                u * std::f32::consts::TAU, 
                v * std::f32::consts::TAU, 
                w * std::f32::consts::TAU
            );
            
            let new_interval = MIN_DIRECTION_INTERVAL + rand::random::<f32>() * (MAX_DIRECTION_INTERVAL - MIN_DIRECTION_INTERVAL);
            timer.timer = Timer::from_seconds(new_interval, TimerMode::Repeating);
        }
    }
}

// ── Constants ────────────────────────────────────────────────────────────────

const GRAVITY: f32 = 9.8;
const MAX_CLIMB_HEIGHT: f32 = 4.0; 

// ── Movement & Rotation Application ──────────────────────────────────────────

fn apply_movement(
    time: Res<Time>,
    blockworld: Res<BlockWorld>,
    heightmap: Res<HeightmapSampler>,
    mut query: Query<(&mut Transform, &mut Organism), With<OrganismRoot>>,
) {
    let dt = time.delta_secs();

    for (mut transform, mut organism) in &mut query {
        let move_vector = organism.movement_direction * organism.movement_speed * dt;

        let floor_y = heightmap.height_at(transform.translation.x, transform.translation.z);
        if transform.translation.y - organism.bounding_radius > floor_y + 2.0 {
            organism.is_climbing = false;
            transform.translation.x += move_vector.x;
            transform.translation.z += move_vector.z;
            continue;
        }

        let mut is_blocked = false;
        let mut climb_needed = 0.0f32;

        for (local_pos, cell_type) in &organism.active_cells {
            let world_pos = transform.transform_point(*local_pos);
            
            let cell_x = world_pos.x.floor() as i32;
            let cell_y = world_pos.y.floor() as i32;
            let cell_z = world_pos.z.floor() as i32;
            let half_height = cell_type.size() / 2.0;
            
            if move_vector.x != 0.0 || move_vector.z != 0.0 {
                let step_x = if move_vector.x > 0.0 { 1 } else if move_vector.x < 0.0 { -1 } else { 0 };
                let step_z = if move_vector.z > 0.0 { 1 } else if move_vector.z < 0.0 { -1 } else { 0 };
                
                let check_x = cell_x + step_x;
                let check_z = cell_z + step_z;
                
                if blockworld.is_solid(check_x, cell_y, check_z) {
                    is_blocked = true;
                    let block_top = cell_y as f32 + 1.0;
                    let needed = block_top - (world_pos.y - half_height);
                    if needed > climb_needed {
                        climb_needed = needed;
                    }
                }
            }
        }
        
        organism.is_climbing = is_blocked;
        
        if is_blocked && climb_needed > 0.0 && climb_needed <= MAX_CLIMB_HEIGHT {
            let climb_amount = (organism.movement_speed * dt).min(climb_needed);
            transform.translation.y += climb_amount;
        } else if is_blocked {
            organism.is_climbing = false;
        } else {
            transform.translation.x += move_vector.x;
            transform.translation.z += move_vector.z;
        }
    }
}

// ── NEW System: Apply Rotation ───────────────────────────────────────────────

fn apply_rotation(
    time: Res<Time>,
    mut query: Query<(&mut Transform, &Organism), With<OrganismRoot>>,
) {
    let dt = time.delta_secs();
    for (mut transform, organism) in &mut query {
        // Smoothly rotate the organism from its current rotation to its target_rotation
        transform.rotation = transform.rotation.slerp(organism.target_rotation, organism.rotation_speed * dt);
    }
}

// ── Gravity & Collisions ─────────────────────────────────────────────────────

fn apply_gravity(
    mode: Res<MovementMode>,
    time: Res<Time>,
    mut query: Query<(&mut Transform, &mut Organism), With<OrganismRoot>>,
) {
    if *mode == MovementMode::ThreeD {
        return;
    }

    let dt = time.delta_secs();

    for (mut transform, mut organism) in &mut query {
        if !organism.is_climbing {
            organism.velocity.y -= GRAVITY * dt;
            transform.translation.y += organism.velocity.y * dt;
        }
    }
}

fn apply_floor_collision(
    heightmap: Res<HeightmapSampler>,
    mut query: Query<(&mut Transform, &mut Organism), With<OrganismRoot>>,
    joint_query: Query<&GlobalTransform, Without<OrganismRoot>>,
) {
    for (mut root_transform, mut organism) in &mut query {
        let mut max_penetration = 0.0f32;

        for (collection_id, offset) in &organism.floor_cells {
            if let Some(&joint_entity) = organism.joint_entities.get(collection_id) {
                if let Ok(joint_global) = joint_query.get(joint_entity) {
                    let world_pos = joint_global.transform_point(*offset);
                    let cell_bottom = world_pos.y - GLOBAL_CELL_SIZE / 2.0;
                    let floor_y = heightmap.height_at(world_pos.x, world_pos.z);
                    let penetration = floor_y - cell_bottom;
                    if penetration > max_penetration {
                        max_penetration = penetration;
                    }
                }
            }
        }

        if max_penetration > 0.0 {
            root_transform.translation.y += max_penetration;
            if organism.velocity.y < 0.0 {
                organism.velocity.y = 0.0;
            }
        }
    }
}

fn apply_world_bounds(
    heightmap: Res<HeightmapSampler>,
    mut query: Query<&mut Transform, With<OrganismRoot>>,
) {
    let min_x = heightmap.min_x as f32;
    let max_x = (heightmap.min_x + heightmap.width as i32 - 1) as f32;
    let min_z = heightmap.min_z as f32;
    let max_z = (heightmap.min_z + heightmap.depth as i32 - 1) as f32;

    for mut transform in &mut query {
        transform.translation.x = transform.translation.x.clamp(min_x, max_x);
        transform.translation.z = transform.translation.z.clamp(min_z, max_z);
    }
}
