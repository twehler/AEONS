// Map editor — camera: an ANGLED PERSPECTIVE overview that the user then flies.
//
// On entry to `MapEditor` the shared `FlyCam`'s `Transform` and `Projection` are
// stashed and the camera is moved to a perspective overview pose: above and
// behind the map centre, looking down at it at ~50°. From there the flycam is
// live (see `player_plugin`): WASD travels, hold-middle-mouse rotates, LMB stays
// free to paint. On exit to any other mode the stash is restored (so the
// simulation gets its tight far-plane projection back).
//
// The entry rotation is `looking_at(centre, +Y)` — NOT straight down — so it is
// free of gimbal degeneracy AND of roll, which means `player_look`'s YXZ
// decompose→rebuild reproduces it exactly: the first middle-mouse rotate does not
// snap. The flycam's default far-plane (300) is far too tight to see a whole map
// from an overview, so we widen `far` to fit the terrain span.
//
// This is a PARALLEL system to the species editor's `snap_camera_on_mode_entry`
// (which also fires on every mode change) — it keeps its OWN stash and only
// touches the camera on MapEditor entry/exit, so the two never mis-restore each
// other. It is ordered `.after` the species snap so that on a direct
// SpeciesEditor→MapEditor switch the MapEditor pose wins the shared `Transform`.

use bevy::prelude::*;

use crate::player_plugin::FlyCam;
use crate::simulation_settings::WindowMode;
use crate::world_geometry::MapSize;


/// Stash for the `FlyCam`'s pre-editor `Transform` + `Projection` while the map
/// editor is active. `Projection` is not `Copy`, so it is cloned.
#[derive(Resource, Default)]
pub struct StashedMapCamera {
    pub transform:  Option<Transform>,
    pub projection: Option<Projection>,
}


/// On entering `MapEditor`, stash the camera and place it at an angled perspective
/// overview of the whole `MapSize`; on leaving, restore the stash.
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

            let cx     = map_size.x * 0.5;
            let cz     = map_size.z * 0.5;
            let centre = Vec3::new(cx, 0.0, cz);
            let span   = map_size.x.max(map_size.z).max(1.0);

            // Eye above (+Y) and behind (+Z) the centre: ~52° downward pitch
            // (atan(span / (0.8·span))). `up = +Y` ⇒ zero roll ⇒ no first-rotate
            // snap (see header).
            let eye = Vec3::new(cx, span, cz + span * 0.8);
            *transform = Transform::from_translation(eye).looking_at(centre, Vec3::Y);

            // Perspective (the flycam's native projection). Widen `far` so the
            // whole terrain stays inside the frustum from this distance
            // (eye→far-corner ≈ 2.5·span; ×4 + buffer is comfortable headroom).
            *projection = Projection::Perspective(PerspectiveProjection {
                far:  span * 4.0 + 500.0,
                near: 0.1,
                ..default()
            });
        }
        // Any non-MapEditor mode: restore the stash if we own one.
        _ => {
            if let Some(t) = stash.transform.take()  { *transform = t; }
            if let Some(p) = stash.projection.take() { *projection = p; }
        }
    }
}
