// Marker on a placed template's visual mesh, carrying the template id.

use bevy::prelude::*;

#[derive(Component)]
#[allow(dead_code)] // Reserved for future "click visual to select template" UX.
pub struct EditorTemplateMarker(pub u32);
