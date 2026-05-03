use bevy::prelude::*;
use bevy::time::common_conditions::on_timer;
use std::time::Duration;

use crate::intelligence_level_1::{
    BrainPoolL1, assign_brains_l1, free_brains_l1, apply_intelligence_level_1,
};
use crate::intelligence_level_3::{
    BrainPool, assign_brains, free_brains, apply_intelligence_level_3,
};
use crate::photosynthesis::PhotosynthesisPlugin;


// Heterotroph brain tick rate. Each tick fires a single batched forward
// pass at `[N, IN]`, computing every heterotroph's private MLP in parallel
// on the GPU. Between ticks, organisms continue to consume their last
// commanded `movement_*` per frame in `apply_movement`, so the simulation
// stays smooth.
//
// `on_timer` ticks with `Time<Virtual>::delta()`, so this naturally pauses
// when the simulation is paused.
const BRAIN_TICK_INTERVAL: Duration = Duration::from_millis(5);


pub struct BehaviourPlugin;

impl Plugin for BehaviourPlugin {
    fn build(&self, app: &mut App) {
        // Photoautotroph sunlight check — runs in PreUpdate at 10 Hz so
        // every Level 1 brain tick reads a recent value of `in_sunlight`.
        app.add_plugins(PhotosynthesisPlugin);

        // ── Brain-slot lifecycle ─────────────────────────────────────────
        //
        // assign_brains* and free_brains* live in PreUpdate. The reason:
        // both `predation::predation_system` and `energy::manage_energy`
        // run in Update and despawn organisms (eaten prey, starved roots).
        // If assign ran in Update too, it could query an organism that's
        // alive at start-of-Update, queue `insert<BrainSlot…>` on it, and
        // then the end-of-Update flush would land that organism's despawn
        // first — leaving the insert to hit a dead entity and panic. By
        // running assign/free in PreUpdate, the PreUpdate→Update schedule
        // boundary guarantees every assigned slot lands on a live entity,
        // and free is collocated so the pool's bookkeeping stays grouped.
        //
        // apply_intelligence_* still runs in Update — it's the brain
        // *tick*, which writes movement_speed/movement_direction on the
        // Organism component, and Update is the natural place for that.

        app.init_non_send_resource::<BrainPoolL1>();
        app.add_systems(PreUpdate, (assign_brains_l1, free_brains_l1).chain());
        app.add_systems(Update, apply_intelligence_level_1);

        app.init_non_send_resource::<BrainPool>();
        app.add_systems(PreUpdate, (assign_brains, free_brains).chain());
        app.add_systems(Update, apply_intelligence_level_3.run_if(on_timer(BRAIN_TICK_INTERVAL)));
    }
}
