// Level 0 intelligence — the "do nothing" tier.
//
// Sessile organisms (`Organism::is_sessile == true`) get the
// `BrainLevel0` marker at spawn. The level has no systems/resources;
// higher pools filter via the marker so a sessile organism is never
// run through a brain pass it couldn't act on.

use bevy::prelude::*;


/// Marker for intelligence level 0. Inserted on the `OrganismRoot` when
/// `Organism::is_sessile == true`. Higher pools filter via
/// `Without<BrainLevel0>` (plus trophic kind); there is no opposite
/// level-1/3 marker.
#[derive(Component, Clone, Copy, Debug)]
pub struct BrainLevel0;
