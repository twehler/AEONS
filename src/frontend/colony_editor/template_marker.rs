// Marker component on the visual mesh entity for each placed
// `OrganismTemplate`. Carries the template's id so the inventory
// panel can highlight the active row by its visual.

use bevy::prelude::*;

#[derive(Component)]
#[allow(dead_code)] // Back-reference to OrganismTemplate.id; reserved
                    // for future "click visual to select template" UX.
pub struct EditorTemplateMarker(pub u32);
