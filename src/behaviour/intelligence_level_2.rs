// Level 2 intelligence — placeholder.
//
// Mirrors `intelligence_level_0.rs` exactly: a single marker
// component, no pool, no systems, no GPU resources. Reserved for a
// future heterotroph brain tier; until something is implemented
// here, the simulation routes initial-spawn L2 rolls back to L1
// (see `IntelligenceLevel::for_initial_spawn`).

use bevy::prelude::*;


/// Marker for organisms whose intelligence level is 2.
///
/// Currently never inserted on any entity — the L1 rewrite removed
/// the corresponding brain pool. The type stays around so save files
/// with `Level2` organisms still load, and so future work can hang
/// real systems off this marker without renaming the enum variant.
#[derive(Component, Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct BrainLevel2;
