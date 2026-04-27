use bevy::prelude::*;
use crate::intelligence_level_1::{BrainResourceLevel1, apply_intelligence_level_1};
use crate::intelligence_level_3::{BrainResourceLevel3, ScannerTimer, initialize_local_graphs, scan_environment, apply_intelligence_level_3};


// ── Bevy Plugin ─────────────────────────────────────────────────────────────

pub struct BehaviourPlugin;

impl Plugin for BehaviourPlugin {
    fn build(&self, app: &mut App) {
        app.init_non_send_resource::<BrainResourceLevel1>();
        app.add_systems(Update, apply_intelligence_level_1);


        app.init_non_send_resource::<BrainResourceLevel3>();
        app.init_resource::<ScannerTimer>();
        app.add_systems(Update, (
            initialize_local_graphs,
            scan_environment,
            apply_intelligence_level_3
        ).chain());

    }
}


