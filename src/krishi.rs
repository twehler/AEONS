// Krishi — a fixed-mesh heterotroph variant.
//
// Krishi is a special predator class introduced for testing / observation.
// It plugs into the existing simulation as a heterotroph (so it inherits
// the Level 3 brain pool's prey-pursuit behaviour and is recognised as a
// predator by `predation::predation_system`), but differs from the
// procedurally-meshed heterotrophs in three ways:
//
//   1. **Visual.** Instead of a procedurally-generated body-part mesh,
//      every Krishi renders the glb scene at `assets/krishi.glb`. The
//      `Organism::body_parts` data still exists on the entity (with one
//      placeholder cell) so the collision and movement systems work
//      transparently, but no body-part mesh children are spawned — the
//      glb scene is the only visible geometry.
//
//   2. **No mutation.** `Organism::reproduced = true` from the moment a
//      Krishi spawns, which permanently gates `reproduction_system` off
//      for it. Krishi never reproduce, so the mutation pipeline is
//      structurally inaccessible.
//
//   3. **Spawned in a fixed-size cohort once the world is ready**, not
//      mixed into the photoautotroph / heterotroph initial colony in
//      `colony.rs`.
//
// Krishi is identified by the `Krishi` marker component (in addition to
// the `Heterotroph` marker that drives behaviour). Future systems that
// need to treat Krishi differently (e.g. a custom mesh-rebuild skip or a
// special predation rule) can filter on `With<Krishi>`.

use bevy::prelude::*;
use bevy::scene::SceneRoot;
use rand::prelude::*;

use crate::cell::*;
use crate::colony::*;
use crate::movement::DirectionTimer;
use crate::world_geometry::{HeightmapSampler, MAP_MAX_X, MAP_MAX_Z};


// ── Tunables ─────────────────────────────────────────────────────────────────

// Initial cohort size lives in `colony.rs` as `INITIAL_KRISHI`, alongside
// the other startup-population knobs. Krishi spawn once when the heightmap
// becomes available and are never replenished (they don't reproduce).

/// Asset path of the Krishi glb. Resolved relative to Bevy's asset root
/// (`assets/`). The `#Scene0` fragment selects the default scene inside the
/// glb — Bevy's GLTF loader exposes scenes as sub-assets of the Gltf asset.
const KRISHI_SCENE_ASSET: &str = "krishi.glb#Scene0";

/// Pixels-of-spawn-altitude above the heightmap floor, mirroring the
/// initial heightmap-clearance the colony uses for the procedural
/// organisms.
const KRISHI_SPAWN_ALTITUDE: f32 = 1.0;

/// Uniform size multiplier applied to BOTH the visual (the glb SceneRoot
/// child Transform) AND the collision footprint (the body part's cell
/// layout in `make_krishi_body`). Visual and collision stay locked
/// together so a Krishi that *looks* a certain size also *touches* prey
/// at that size.
///
/// Note on energy: Krishi consumption scales linearly with cell count,
/// so a scaled Krishi spends ~7x more energy per tick than a 1-cell
/// heterotroph. Starvation TIME is unchanged (the 0.5 starting energy
/// fraction also scales with cell count), but the predator must eat
/// ~7x more frequently to stay alive. Tune `KRISHI_SCALE` together with
/// the energy constants in `energy.rs` if you change ecological pressure.
const KRISHI_SCALE: f32 = 6.0;


// ── Components ───────────────────────────────────────────────────────────────

/// Marker on the Krishi root entity. Distinguishes Krishi from the
/// procedurally-meshed heterotrophs for any system that needs to special-
/// case them (currently none beyond the spawn pipeline itself).
#[derive(Component, Clone, Copy)]
pub struct Krishi;


// ── Resources ────────────────────────────────────────────────────────────────

/// Holds the glb scene handle so every Krishi spawn shares the same
/// asset rather than re-loading from disk per organism.
#[derive(Resource)]
struct KrishiAssets {
    scene: Handle<Scene>,
}


// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct KrishiPlugin;

impl Plugin for KrishiPlugin {
    fn build(&self, app: &mut App) {
        // Asset load is fire-and-forget — the actual `SceneRoot` spawn
        // tolerates the handle still being in the loading state; Bevy
        // materialises the visual once the asset finishes decoding.
        app.add_systems(Startup, load_krishi_assets);
        app.add_systems(
            Update,
            spawn_krishi_cohort.run_if(resource_exists::<HeightmapSampler>),
        );
    }
}


fn load_krishi_assets(mut commands: Commands, asset_server: Res<AssetServer>) {
    let scene: Handle<Scene> = asset_server.load(KRISHI_SCENE_ASSET);
    commands.insert_resource(KrishiAssets { scene });
}


// ── Body / collision construction ────────────────────────────────────────────

/// Build the single body part used for a Krishi's collision footprint at
/// the given uniform `scale`. The body has one centre cell plus six
/// axis-aligned cells at `±scale * 0.5` along each principal axis (an
/// octahedron of cells around the origin). Combined with each cell's
/// `CELL_COLLISION_RADIUS` sphere in `organism_collision.rs`, this yields
/// a roughly spherical collision footprint of radius
/// `≈ scale * 0.5 + CELL_COLLISION_RADIUS` — close enough to a sphere
/// inscribed in the scaled glb mesh that predator-prey contacts trigger
/// when the visuals visibly overlap rather than only when the prey
/// touches the Krishi's invisible centre.
///
/// At very small scales (≤ 1.2) the axis cells would overlap the centre,
/// so we fall back to the original single-cell footprint — the centre
/// cell alone already covers a sphere of radius `CELL_COLLISION_RADIUS`
/// which is adequate for that range.
fn make_krishi_body(scale: f32) -> BodyPart {
    let mut cells = vec![Cell {
        local_pos: Vec3::ZERO,
        cell_type: CellType::NonPhoto,
    }];

    if scale > 1.2 {
        let r = scale * 0.5;
        for dir in [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z] {
            cells.push(Cell { local_pos: dir * r, cell_type: CellType::NonPhoto });
        }
    }

    BodyPart {
        kind:          BodyPartKind::Body,
        local_offset:  Vec3::ZERO,
        cells,
        consumed:      false,
    }
}


// ── Spawning ─────────────────────────────────────────────────────────────────

fn spawn_krishi_cohort(
    mut commands:  Commands,
    krishi:        Res<KrishiAssets>,
    heightmap:     Res<HeightmapSampler>,
    mut spawned:   Local<bool>,
) {
    if *spawned { return; }
    *spawned = true;

    let mut rng = rand::rng();

    for _ in 0..INITIAL_KRISHI {
        let x = rng.random_range(0.0_f32..MAP_MAX_X);
        let z = rng.random_range(0.0_f32..MAP_MAX_Z);
        let y = heightmap.height_at(x, z) + KRISHI_SPAWN_ALTITUDE;
        spawn_krishi(Vec3::new(x, y, z), &krishi, &mut commands, &mut rng);
    }
}


/// Spawn one Krishi at `pos`. The hierarchy produced is:
///
///   OrganismRoot (Heterotroph + Krishi + Organism + DirectionTimer)
///   └── SceneRoot child (the krishi.glb scene)
///
/// Note the absence of body-part mesh children — Krishi's visual is the
/// glb scene only. The single body-part cell in `Organism::body_parts`
/// exists purely so movement / collision queries find a footprint to test
/// against the world mesh and prey cells.
fn spawn_krishi(
    pos:      Vec3,
    assets:   &KrishiAssets,
    commands: &mut Commands,
    rng:      &mut impl rand::Rng,
) -> Entity {
    let angle     = rng.random::<f32>() * std::f32::consts::TAU;
    let direction = Vec3::new(angle.cos(), 0.0, angle.sin());
    // Match the standard heterotroph speed range so Level 3's brain tick
    // produces movement at familiar magnitudes.
    let speed     = 15.0 + rng.random::<f32>() * 10.0;

    // Body part scaled to `KRISHI_SCALE`. See `make_krishi_body` for the
    // cell-layout strategy. Cells are `NonPhoto` (heterotroph-style); the
    // type doesn't affect behaviour (Heterotroph marker drives that), only
    // the per-cell consumption tally in `energy.rs`.
    let body_parts = vec![make_krishi_body(KRISHI_SCALE)];

    let max_energy = (body_parts.iter().map(|b| b.cells.len()).sum::<usize>() as f32)
                     * crate::energy::MAX_ENERGY_PER_CELL;

    let ocg: Vec<(usize, Vec3, CellType)> = body_parts.iter()
        .flat_map(|bp| bp.cells.iter().map(|c| (bp.local_offset + c.local_pos, c.cell_type)))
        .enumerate()
        .map(|(i, (pos, ct))| (i, pos, ct))
        .collect();

    let organism = Organism {
        body_parts: body_parts.clone(),
        ocg,
        energy:             max_energy * 0.5,
        in_sunlight:        false,
        // `reproduced = true` permanently gates `reproduction_system` off.
        // This is how mutation is "switched off" for Krishi: the mutation
        // pipeline only runs as part of reproduction, and reproduction
        // never fires for an organism whose `reproduced` flag is set.
        reproduced:         true,
        reproductions:      0,
        movement_speed:     speed,
        movement_direction: direction,
        velocity:           Vec3::ZERO,
        is_climbing:        false,
        climb_energy_debt:  0.0,
    };

    let direction_interval = 1.0 + rng.random::<f32>() * 9.0;
    let spawn_rotation     = Quat::from_rotation_y(angle);

    let root = commands
        .spawn((
            Transform::from_translation(pos).with_rotation(spawn_rotation),
            Visibility::Visible,
            OrganismRoot,
            Heterotroph, // Routes Krishi through the Level 3 brain pool.
            Krishi,
            organism,
            DirectionTimer::new(direction_interval),
        ))
        .id();

    // Visual: spawn the glb scene as a child, scaled by `KRISHI_SCALE` so
    // the rendered model matches the collision footprint built by
    // `make_krishi_body(KRISHI_SCALE)`. `Visibility::Inherited` lets the
    // root's visibility cascade if Krishi ever needs to be hidden
    // wholesale (debug toggles, future fade-outs, etc).
    let scene_child = commands
        .spawn((
            SceneRoot(assets.scene.clone()),
            Transform::from_scale(Vec3::splat(KRISHI_SCALE)),
            Visibility::Inherited,
        ))
        .id();
    commands.entity(root).add_child(scene_child);

    root
}
