// Per-organism sunlight check.
//
// A photoautotroph is "in direct sunlight" when the line from its position
// to the (fixed) sun is not occluded by terrain. We test this by marching a
// ray from the organism toward the sun and querying the heightmap at each
// step — if the terrain is above the ray's Y at any sampled (x, z), the
// organism is in shadow.
//
// The check is throttled to 10 Hz: terrain doesn't move and organisms
// wander on the order of seconds, so a fresh result every frame would just
// burn cycles for a value that hasn't changed.

use bevy::prelude::*;
use bevy::time::common_conditions::on_timer;
use std::time::Duration;

use crate::colony::{Organism, OrganismRoot};
use crate::world_geometry::HeightmapSampler;


/// Direction *toward* the sun, as a unit vector.
///
/// Mirrors the directional light orientation in `main.rs`:
/// `Quat::from_euler(EulerRot::XYZ, -π/4, π/4, 0)` applied to Bevy's default
/// directional light forward (`-Z`) yields a light pointing roughly
/// `(-0.5, -√2/2, -0.5)`. The opposite of that — the direction *toward* the
/// light source — is the unit vector below.
pub const SUN_DIRECTION: Vec3 = Vec3::new(0.5, std::f32::consts::FRAC_1_SQRT_2, 0.5);

const SHADOW_CHECK_INTERVAL: Duration = Duration::from_millis(100);

/// Step length of the shadow raymarch in world units. Chosen to match the
/// heightmap cell size (1.0) so each step samples a fresh terrain cell.
const RAY_STEP_SIZE: f32 = 1.0;

/// Maximum number of steps before declaring the ray escaped to the sky.
/// With sun y-component √2/2 ≈ 0.707 and step size 1.0, 300 steps lift
/// the ray ~210 units above its origin — comfortably above any plausible
/// terrain peak in normalised worlds.
const MAX_RAY_STEPS: usize = 300;


pub struct PhotosynthesisPlugin;

impl Plugin for PhotosynthesisPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PreUpdate,
            update_sunlight
                .run_if(resource_exists::<HeightmapSampler>)
                .run_if(on_timer(SHADOW_CHECK_INTERVAL))
        );
    }
}

/// Updates `Organism::in_sunlight` for every organism, including
/// heterotrophs — the field is cheap to maintain and may become useful for
/// future systems (e.g. thermoregulation). The brain only consumes it for
/// photoautotrophs in `intelligence_level_1.rs`.
fn update_sunlight(
    heightmap: Res<HeightmapSampler>,
    mut query: Query<(&mut Organism, &Transform), With<OrganismRoot>>,
) {
    let sun      = SUN_DIRECTION.normalize();
    let escape_y = heightmap.max_height + 5.0;

    for (mut organism, transform) in query.iter_mut() {
        organism.in_sunlight = is_lit(transform.translation, &heightmap, sun, escape_y);
    }
}

/// Marches a ray from `origin` toward the sun. Returns `true` once the ray
/// rises above any plausible terrain peak; returns `false` the first time
/// the heightmap is above the ray at the sampled (x, z).
fn is_lit(origin: Vec3, heightmap: &HeightmapSampler, sun: Vec3, escape_y: f32) -> bool {
    for i in 1..=MAX_RAY_STEPS {
        let p = origin + sun * (i as f32 * RAY_STEP_SIZE);
        if p.y > escape_y { return true; }
        if heightmap.height_at(p.x, p.z) > p.y { return false; }
    }
    true
}
