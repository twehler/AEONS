// Level 0 intelligence — the "do nothing" tier.
//
// Sessile organisms (`Organism::is_sessile == true`) get the
// `BrainLevel0` marker component automatically at spawn (see
// `colony.rs::spawn_organism` / `spawn_loaded_organism`). For now the
// level has no associated systems, no resources, no plugin — anything
// that operates on intelligence ladder filters via the marker so a
// sessile body part is never the subject of a brain forward / backward
// pass it could not act on anyway.
//
// Future per-cell or per-organism rules that should run only on
// stationary organisms will be added inside this file. Until then,
// nothing is called and nothing is executed for level 0.

use bevy::prelude::*;


/// Marker for organisms whose intelligence level is 0.
///
/// Inserted on the `OrganismRoot` whenever `Organism::is_sessile == true`.
/// Higher-level brain systems (`apply_intelligence_level_1`,
/// `apply_intelligence_level_3`, and their `assign_*` siblings) filter
/// against this marker via `Without<BrainLevel0>` so a sessile organism
/// is never wired into a higher pool. There is no opposite "BrainLevel1
/// / BrainLevel3" marker — those pools select by trophic kind
/// (`Photoautotroph` / `Heterotroph`) AND `Without<BrainLevel0>`.
#[derive(Component, Clone, Copy, Debug)]
pub struct BrainLevel0;
