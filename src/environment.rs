use bevy::prelude::*;
use crate::world_geometry::BlockWorld;

// ── Environment query ────────────────────────────────────────────────────────

// Procedurally computed environment at a world position.
// No per-block storage needed — derived from position and block type.
pub struct EnvironmentSample {
    pub temperature: f32,  // 0.0 (freezing) to 1.0 (hot)
    pub acidity: f32,      // 0.0 (neutral) to 1.0 (highly acidic)
    pub light: f32,        // 0.0 (dark) to 1.0 (full light)
    pub hardness: f32,     // ground hardness 0.0-1.0
    pub metal: f32,        // metal concentration 0.0-1.0
    pub humidity: f32,     // 0.0 (dry) to 1.0 (wet)
    pub underwater: bool,  // true if below water level
}

// ── Constants ────────────────────────────────────────────────────────────────

pub const WATER_LEVEL: f32 = 4.9; // world Y of water surface

// Temperature gradient: warmer at lower altitudes, cooler at peaks.
const TEMP_BASE: f32 = 0.7;
const TEMP_HEIGHT_SCALE: f32 = 0.005; // per unit of height
pub const MAP_MAX_X: f32 = 1024.0;
pub const MAP_MAX_Z: f32 = 1024.0;

// Simple noise-like variation using sin/cos of world position
// (deterministic, no extra crate needed)
fn pseudo_noise(x: f32, z: f32, seed: f32) -> f32 {
    let v = (x * 0.01 + seed).sin() * (z * 0.013 + seed * 1.7).cos();
    (v + 1.0) * 0.5 // map to 0..1
}

// ── Block type → material properties ────────────────────────────────────────

// The block_metadata stores block type as u8. This maps type to material props.
// Based on generate_world.rs block assignments:
//   0-9: various stone types (hard, sometimes metallic)
//   10-19: soil/dirt types (soft, neutral)
//   20-29: sand types (soft, neutral to acidic near water)
//   30-39: organic/vegetation (soft, neutral)
//   40+: special (ores, crystals)

fn block_hardness(block_type: u8) -> f32 {
    match block_type {
        0..=9   => 0.8 + (block_type as f32 * 0.02),  // stone: hard
        10..=19 => 0.3,                                 // soil: soft
        20..=29 => 0.1,                                 // sand: very soft
        30..=39 => 0.2,                                 // organic: soft
        40..=49 => 0.9,                                 // ores: very hard
        _       => 0.5,                                 // default: medium
    }
}

fn block_metal(block_type: u8) -> f32 {
    match block_type {
        40..=44 => 0.7 + (block_type as f32 - 40.0) * 0.06, // metallic ores
        45..=49 => 0.3,                                       // crystal/semi-metallic
        0..=4   => 0.1,                                       // trace minerals in stone
        _       => 0.0,
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

pub fn sample_environment(pos: Vec3, block_world: &BlockWorld) -> EnvironmentSample {
    let underwater = pos.y < WATER_LEVEL;

    // Temperature: altitude-based with spatial variation
    let altitude_factor = (TEMP_BASE - pos.y * TEMP_HEIGHT_SCALE).clamp(0.0, 1.0);
    let spatial_var = pseudo_noise(pos.x, pos.z, 42.0) * 0.2;
    let temperature = (altitude_factor + spatial_var).clamp(0.0, 1.0);

    // Acidity: higher near water, varies spatially
    let water_proximity = if underwater { 0.3 } else {
        (1.0 - ((pos.y - WATER_LEVEL).abs() * 0.05)).clamp(0.0, 0.3)
    };
    let acidity = (water_proximity + pseudo_noise(pos.x, pos.z, 137.0) * 0.2).clamp(0.0, 1.0);

    // Light: full above ground, diminishes underwater / underground
    let light = if underwater {
        (1.0 - (WATER_LEVEL - pos.y) * 0.03).clamp(0.1, 0.8)
    } else {
        1.0
    };

    // Humidity: high near water, varies with position
    let humidity = if underwater {
        1.0
    } else {
        let base = (1.0 - (pos.y - WATER_LEVEL) * 0.02).clamp(0.0, 0.8);
        (base + pseudo_noise(pos.x, pos.z, 271.0) * 0.2).clamp(0.0, 1.0)
    };

    // Ground properties: look up block directly below organism
    let ground_x = pos.x.floor() as i32;
    let ground_z = pos.z.floor() as i32;
    let ground_y = (pos.y - 1.0).floor() as i32;

    let (hardness, metal) = if let Some(bt) = block_world.block_type(ground_x, ground_y, ground_z) {
        (block_hardness(bt), block_metal(bt))
    } else {
        (0.5, 0.0) // default for air/void
    };

    EnvironmentSample {
        temperature,
        acidity,
        light,
        hardness,
        metal,
        humidity,
        underwater,
    }
}
