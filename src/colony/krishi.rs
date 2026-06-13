// Krishi — a fixed-mesh heterotroph variant (testing/observation predator).
// Plugs in as a heterotroph (Level 3 brain, recognised by predation), but:
//   1. Renders `assets/krishi.glb` instead of a procedural mesh; the single
//      body-part cell exists only so collision/movement have a footprint.
//   2. `reproduced = true` from spawn → never reproduces, so mutation is
//      structurally inaccessible.
//   3. Spawned as a fixed cohort once the world is ready, not via `colony.rs`.
// Identified by the `Krishi` marker (alongside `Heterotroph`).

use bevy::prelude::*;
use bevy::scene::SceneRoot;
use rand::prelude::*;

use crate::cell::*;
use crate::colony::*;
use crate::movement::DirectionTimer;
use crate::world_geometry::{HeightmapSampler, MapSize};


// ── Tunables ─────────────────────────────────────────────────────────────────

// Initial cohort size lives in `colony.rs` as `INITIAL_KRISHI`. Krishi spawn
// once when the heightmap is available and are never replenished.

/// Krishi glb path (relative to `assets/`). `#Scene0` selects the default scene.
const KRISHI_SCENE_ASSET: &str = "krishi.glb#Scene0";

use crate::simulation_settings::KRISHI_SPAWN_ALTITUDE;

use crate::simulation_settings::KRISHI_SCALE;


// ── Components ───────────────────────────────────────────────────────────────

/// Marker on the Krishi root entity.
#[derive(Component, Clone, Copy)]
pub struct Krishi;


// ── Resources ────────────────────────────────────────────────────────────────

/// Shared glb scene handle so every Krishi spawn reuses one asset.
#[derive(Resource)]
struct KrishiAssets {
    scene: Handle<Scene>,
}


// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct KrishiPlugin;

impl Plugin for KrishiPlugin {
    fn build(&self, app: &mut App) {
        // Asset load is fire-and-forget — `SceneRoot` tolerates a still-loading
        // handle; Bevy materialises the visual once decoding finishes.
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

/// Build the single body part for a Krishi's collision footprint at uniform
/// `scale`: a centre cell plus six axis cells at `±scale*0.5`, giving a
/// roughly spherical footprint of radius `≈ scale*0.5 + CELL_COLLISION_RADIUS`
/// so contacts trigger when the visuals overlap, not only at the centre.
/// At scale ≤ 1.2 the axis cells would overlap the centre, so fall back to
/// the single centre cell (already covers `CELL_COLLISION_RADIUS`).
fn make_krishi_body(scale: f32) -> BodyPart {
    let mut cells = vec![Cell::new(Vec3::ZERO, CellType::NonPhoto)];

    if scale > 1.2 {
        let r = scale * 0.5;
        for dir in [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z] {
            cells.push(Cell::new(dir * r, CellType::NonPhoto));
        }
    }

    // Mirror cells into the per-part OCG so collision/energy see the same
    // cell catalogue as everything else.
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
        // Hand-built footprint; mutation must never touch it.
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


/// Spawn one Krishi at `pos`:
///   OrganismRoot (Heterotroph + Krishi + Organism + DirectionTimer)
///   └── SceneRoot child (the krishi.glb scene)
/// No body-part mesh children — the glb scene is the only visual; the single
/// body-part cell exists only as a collision/movement footprint.
fn spawn_krishi(
    pos:      Vec3,
    assets:   &KrishiAssets,
    commands: &mut Commands,
    rng:      &mut impl rand::Rng,
) -> Entity {
    let angle     = rng.random::<f32>() * std::f32::consts::TAU;
    let direction = Vec3::new(angle.cos(), 0.0, angle.sin());
    // Match the standard heterotroph speed range.
    let speed     = 15.0 + rng.random::<f32>() * 10.0;

    // Cells are NonPhoto; type only affects the per-cell consumption tally,
    // not behaviour (the Heterotroph marker drives that).
    let body_parts = vec![make_krishi_body(KRISHI_SCALE)];

    let max_energy = (body_parts.iter().map(|b| b.cells.len()).sum::<usize>() as f32)
                     * crate::energy::MAX_ENERGY_PER_CELL;

    // Populate per-cell physiology caches even though the body never
    // regrows — `recompute_cell_counts` reads them.
    let mut body_parts = body_parts;
    crate::physiology::recompute_body_parts(&mut body_parts);

    let mut organism = Organism {
        body_parts: body_parts.clone(),
        // Irrelevant (never reproduces); NoSymmetry for type consistency.
        symmetry:             Symmetry::NoSymmetry,
        // Set explicitly because krishi spawn skips `spawn_organism`'s roll.
        intelligence_level:   IntelligenceLevel::Level3,
        is_sessile:           false,
        has_variable_form:    false,
        // Hand-built static body, no limbs to drive — sliding movement.
        movement_mode:        MovementMode::Sliding,
        ground_based:         true,
        limb_targets:         [0.0; 10],
        // Never grows → adult from spawn (regrowable: false skips meshing).
        adult:                true,
        photo_cell_count:     0,
        non_photo_cell_count: 0,
        energy:             max_energy * 0.5,
        in_sunlight:        false,
        // Permanently gates `reproduction_system` off → mutation never runs.
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
        // Brain-gene slots stay 0 (no L1 pool); speciation parks Krishi in
        // its own species since its intelligence dim differs from L1 heteros.
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

    // Visual: glb scene child scaled by `KRISHI_SCALE` to match the collision
    // footprint. `Visibility::Inherited` lets the root toggle it wholesale.
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
