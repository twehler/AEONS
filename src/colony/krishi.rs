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
use crate::world_geometry::{HeightmapSampler, MapSize};


// ── Tunables ─────────────────────────────────────────────────────────────────

// Initial cohort size lives in `colony.rs` as `INITIAL_KRISHI`, alongside
// the other startup-population knobs. Krishi spawn once when the heightmap
// becomes available and are never replenished (they don't reproduce).

/// Asset path of the Krishi glb. Resolved relative to Bevy's asset root
/// (`assets/`). The `#Scene0` fragment selects the default scene inside the
/// glb — Bevy's GLTF loader exposes scenes as sub-assets of the Gltf asset.
const KRISHI_SCENE_ASSET: &str = "krishi.glb#Scene0";

use crate::simulation_settings::KRISHI_SPAWN_ALTITUDE;

use crate::simulation_settings::KRISHI_SCALE;


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
    let mut cells = vec![Cell::new(Vec3::ZERO, CellType::NonPhoto)];

    if scale > 1.2 {
        let r = scale * 0.5;
        for dir in [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z] {
            cells.push(Cell::new(dir * r, CellType::NonPhoto));
        }
    }

    // Mirror cells into the per-part OCG so collision / energy code sees the
    // same cell catalogue everything else uses.
    let ocg: Vec<(usize, Vec3, CellType)> = cells.iter().enumerate()
        .map(|(i, c)| (i, c.local_pos, c.cell_type))
        .collect();

    BodyPart {
        kind:          BodyPartKind::Body,
        local_offset:  Vec3::ZERO,
        cells,
        ocg,
        attachment:    None,
        consumed:      false,
        debug_blue:    false,
        // Krishi has a hand-built footprint; mutation must never touch it
        // and the visual is the glb scene, not a generated mesh.
        regrowable:    false,
    }
}


// ── Spawning ─────────────────────────────────────────────────────────────────

fn spawn_krishi_cohort(
    mut commands:  Commands,
    krishi:        Res<KrishiAssets>,
    heightmap:     Res<HeightmapSampler>,
    map_size:      Res<MapSize>,
    mut spawned:   Local<bool>,
) {
    if *spawned { return; }
    *spawned = true;

    let mut rng = rand::rng();

    for _ in 0..INITIAL_KRISHI {
        let x = rng.random_range(0.0_f32..map_size.x);
        let z = rng.random_range(0.0_f32..map_size.z);
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

    // Populate per-cell physiology caches even though the body is hand-built
    // and never regrows — `Organism::recompute_cell_counts` reads them, and
    // future physiology rules will too. Krishi's Photo cells (currently
    // none — its layout is all NonPhoto) would still produce energy through
    // the standard photosynthesis path if any were present.
    let mut body_parts = body_parts;
    crate::physiology::recompute_body_parts(&mut body_parts);

    let mut organism = Organism {
        body_parts: body_parts.clone(),
        // Krishi never reproduces (`reproduced: true` below) so the symmetry
        // value is structurally irrelevant — set to NoSymmetry for type
        // consistency with the rest of the population.
        symmetry:             Symmetry::NoSymmetry,
        // Mobile heterotroph → Level 3 brain (large RL pool).
        // Spelled out here because the krishi spawn path doesn't
        // go through `spawn_organism`'s rolling logic.
        intelligence_level:   IntelligenceLevel::Level3,
        // Krishi roams freely (mobile predator); no plant-like form.
        is_sessile:           false,
        has_variable_form:    false,
        // Krishi keeps legacy sliding movement — it has a hand-built
        // static body and no limbs to drive locomotion through.
        sliding_movement:     true,
        limb_targets:         [0.0; 8],
        // Krishi has a hand-built footprint and never grows — adult
        // from spawn. (No mesh smoothing is run because Krishi's body
        // part is `regrowable: false`, so spawn_organism skips mesh
        // creation entirely; its visual is the glb scene.)
        adult:                true,
        photo_cell_count:     0,
        non_photo_cell_count: 0,
        energy:             max_energy * 0.5,
        in_sunlight:        false,
        // `reproduced = true` permanently gates `reproduction_system` off.
        // This is how mutation is "switched off" for Krishi: the mutation
        // pipeline only runs as part of reproduction, and reproduction
        // never fires for an organism whose `reproduced` flag is set.
        reproduced:         true,
        reproductions:      0,
        predations:         0,
        hunger:             0.0,
        dopamine:           0.0,
        target_distance:    crate::sensory::SENSORY_RADIUS,
        movement_speed:     speed,
        movement_direction: direction,
        velocity:           Vec3::ZERO,
        is_climbing:        false,
        climb_energy_debt:  0.0,
        cached_bounding_radius: 0.0,
        // Krishi is structurally a Heterotroph + NoSymmetry + L3
        // organism. Brain-gene slots stay 0 (Krishi doesn't use
        // the L1 hetero pool); the speciation system will park it
        // in its own species on the first tick because its
        // intelligence dim differs from L1 heteros.
        dna: crate::lineages::dna::structural_dna(
            crate::organism::OrganismKind::Heterotroph,
            Symmetry::NoSymmetry,
            false, // has_variable_form
            false, // is_sessile
            IntelligenceLevel::Level3,
        ),
        species_id: None,
    };
    organism.recompute_cell_counts();

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
