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
use std::collections::HashMap;

use crate::colony::{Organism, OrganismRoot};
use crate::world_geometry::HeightmapSampler;


pub use crate::simulation_settings::SUN_DIRECTION;

use crate::simulation_settings::SHADOW_CHECK_INTERVAL;

use crate::simulation_settings::RAY_STEP_SIZE;

use crate::simulation_settings::MAX_RAY_STEPS;


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
    mut query: Query<(Entity, &mut Organism, &Transform), With<OrganismRoot>>,
    // Trunk-part (`BodyPartIndex(0)`) world transforms, keyed by their
    // parent OrganismRoot. For LIMB-based organisms the root transform
    // is frozen at spawn — only the per-part `RigidBody::Dynamic`
    // children move — so the ray-march must originate from the trunk
    // part's true world position, otherwise a walking phototroph's
    // sun-occlusion would be evaluated at its birthplace forever. For
    // sliding organisms the trunk sits at identity-local under a moving
    // root, so its world position equals the root's and this is a no-op.
    base_part_q: Query<(&ChildOf, &crate::cell::BodyPartIndex, &GlobalTransform)>,
) {
    let sun      = SUN_DIRECTION.normalize();
    let escape_y = heightmap.max_height + 5.0;

    // Map each OrganismRoot to its trunk part's world position.
    let mut base_pos: HashMap<Entity, Vec3> = HashMap::new();
    for (parent, idx, gx) in &base_part_q {
        if idx.0 == 0 {
            base_pos.insert(parent.parent(), gx.translation());
        }
    }

    for (entity, mut organism, transform) in query.iter_mut() {
        // Prefer the trunk part's world position; fall back to the root
        // transform if (briefly) no trunk part is registered.
        let origin = base_pos.get(&entity).copied()
            .unwrap_or(transform.translation);
        organism.in_sunlight = is_lit(origin, &heightmap, sun, escape_y);
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
