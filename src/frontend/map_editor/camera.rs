// Map editor — camera: a top-down ORTHOGRAPHIC view framing the whole terrain.
//
// On entry to `MapEditor` the shared `FlyCam`'s `Transform` and `Projection` are
// stashed and the camera is parked centred high above the map looking straight
// down, with an orthographic projection sized to fit the full `MapSize`. On exit
// to any other mode the stash is restored. Flycam movement/look is disabled in
// this mode (see `player_plugin`), so the parked pose is stable.
//
// This is a PARALLEL system to the species editor's `snap_camera_on_mode_entry`
// (which also fires on every mode change) — it keeps its OWN stash and only
// touches the camera on MapEditor entry/exit, so the two never mis-restore each
// other. It is ordered `.after` the species snap so that on a direct
// SpeciesEditor→MapEditor switch the MapEditor pose wins the shared `Transform`.

use bevy::camera::ScalingMode;
use bevy::prelude::*;

use crate::player_plugin::FlyCam;
use crate::simulation_settings::WindowMode;
use crate::world_geometry::MapSize;


// ── Tunables ──────────────────────────────────────────────────────────────────

/// Camera height above the map plane. Well above any plausible terrain Y so the
/// whole surface is in front of the ortho near-plane.
const CAMERA_HEIGHT: f32 = 5000.0;


/// Stash for the `FlyCam`'s pre-editor `Transform` + `Projection` while the map
/// editor is active. `Projection` is not `Copy`, so it is cloned.
#[derive(Resource, Default)]
pub struct StashedMapCamera {
    pub transform:  Option<Transform>,
    pub projection: Option<Projection>,
}


/// On entering `MapEditor`, stash the camera and park it top-down/orthographic
/// framing the whole `MapSize`; on leaving, restore the stash.
pub fn snap_map_camera_on_mode_entry(
    mode:        Res<WindowMode>,
    map_size:    Res<MapSize>,
    mut stash:   ResMut<StashedMapCamera>,
    mut cam_q:   Query<(&mut Transform, &mut Projection), With<FlyCam>>,
) {
    if !mode.is_changed() { return; }

    let Ok((mut transform, mut projection)) = cam_q.single_mut() else { return };

    match *mode {
        WindowMode::MapEditor => {
            // Stash the pre-editor pose for restore on exit (defensive against
            // re-entry without restore: don't overwrite an existing stash).
            if stash.transform.is_none() {
                stash.transform  = Some(*transform);
                stash.projection = Some(projection.clone());
            }

            let cx = map_size.x * 0.5;
            let cz = map_size.z * 0.5;
            // Looking straight down (-Y), so the up vector must be Z — +Y is
            // gimbal-degenerate when the look direction is the Y axis.
            *transform = Transform::from_translation(Vec3::new(cx, CAMERA_HEIGHT, cz))
                .looking_at(Vec3::new(cx, 0.0, cz), Vec3::Z);

            // `AutoMin` guarantees the full `MapSize` box is framed regardless of
            // the viewport's aspect ratio (the rendered extent may exceed it).
            *projection = Projection::Orthographic(OrthographicProjection {
                scaling_mode: ScalingMode::AutoMin {
                    min_width:  map_size.x,
                    min_height: map_size.z,
                },
                scale: 1.0,
                near:  0.0,
                far:   CAMERA_HEIGHT + 5000.0, // span camera height + terrain depth
                ..OrthographicProjection::default_3d()
            });
        }
        // Any non-MapEditor mode: restore the stash if we own one.
        _ => {
            if let Some(t) = stash.transform.take()  { *transform = t; }
            if let Some(p) = stash.projection.take() { *projection = p; }
        }
    }
}
