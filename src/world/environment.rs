// Shared environment constants used across systems.

use bevy::prelude::Resource;

/// Default world-space Y of the global water surface. Seeds `WaterLevel` and the
/// launcher's "Water Level" field.
pub const DEFAULT_WATER_LEVEL: f32 = 200.0;

/// Runtime world-space Y of the global water surface. Replaces the old WATER_LEVEL const. Set from the launcher --water-level flag, overridden by a loaded .colony file unless --adjust-colony-dimensions was passed.
#[derive(Resource, Clone, Copy, Debug)]
pub struct WaterLevel(pub f32);
impl Default for WaterLevel { fn default() -> Self { Self(DEFAULT_WATER_LEVEL) } }
