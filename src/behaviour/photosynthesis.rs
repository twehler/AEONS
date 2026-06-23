// Per-organism sunlight check.
//
// "In direct sunlight" = the line to the (fixed) sun isn't occluded by
// terrain; tested by ray-marching toward the sun and sampling the
// heightmap. Throttled to 10 Hz since terrain is static and organisms
// move slowly.

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

/// Updates `Organism::in_sunlight` for every organism (heterotrophs
/// included; only photoautotroph brains consume it).
fn update_sunlight(
    heightmap: Res<HeightmapSampler>,
    mut query: Query<(Entity, &mut Organism, &Transform), With<OrganismRoot>>,
    // Trunk-part (`BodyPartIndex(0)`) world transforms, keyed by parent
    // OrganismRoot. For limb organisms the root is frozen at spawn (only
    // per-part dynamic bodies move), so the ray-march must originate from
    // the trunk's true world position. For sliding organisms the trunk
    // is identity-local under a moving root, so this is a no-op.
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
