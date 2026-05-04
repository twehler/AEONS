use crate::cell::*;
use crate::viewport_settings::ShowGizmo;
use crate::volumetric_growth::build_mesh_from_ocg;
use crate::world_geometry::{HeightmapSampler, MAP_MAX_X, MAP_MAX_Z};
use crate::movement::DirectionTimer;
use bevy::prelude::*;
use rand::prelude::*;

/// Hard cap on simulation population. Both brain pools size their tensors to
/// this constant — exceeding it would silently miss organisms in the batched
/// MLP forward pass.
pub const MAXIMUM_ORGANISMS: usize = 1100;

/// Initial population of each trophic strategy. Photoautotrophs dominate so
/// the food web has plenty of prey for the smaller heterotroph predator pool.
const INITIAL_PHOTOAUTOTROPHS: u32 = 400;
const INITIAL_HETEROTROPHS:    u32 = 200;

/// Initial Krishi cohort size. `pub` so `krishi.rs` reads it directly —
/// keeps every "how many of X spawn at startup" knob in one place.
pub const INITIAL_KRISHI: u32 = 1;


pub struct ColonyPlugin;

impl Plugin for ColonyPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, spawn_colony.run_if(resource_exists::<HeightmapSampler>));
    }
}


// ── Organism ─────────────────────────────────────────────────────────────────
//
// New architecture: an organism is an ordered list of body parts. Each body
// part is a separate child entity rendering its own procedural mesh. Cell
// data lives inside `BodyPart::cells` (typed vertex-cells).
//
// Aggregate quantities (weight, bounding radius, photo/non-photo tally) are
// computed on demand from `body_parts` rather than stored as redundant
// fields — they're only read by ~10 Hz systems and the iteration cost is
// negligible at our scale.

#[derive(Component, Clone)]
pub struct Organism {
    pub body_parts: Vec<BodyPart>,
    /// Order-of-cell-growth ledger: each entry is (index, position relative
    /// to organism root). Populated from body-part cells at spawn time, then
    /// extended by volumetric_growth as the organism grows. Collision geometry
    /// is derived from this vector instead of body_parts[].cells.
    pub ocg: Vec<(usize, Vec3, CellType)>,
    pub energy: f32,
    /// True when the organism's position has an unobstructed line to the sun.
    /// Maintained by `photosynthesis.rs` and consumed by Level 1 brains.
    pub in_sunlight: bool,
    /// Hard gate consulted by `reproduction.rs`: once `true`, this organism
    /// will never spawn another offspring for the rest of its life.
    pub reproduced: bool,
    /// Running count of successful reproductions, used by `reproduction.rs`
    /// to decide when to flip `reproduced` (heterotrophs flip after the
    /// first, photoautotrophs after the second).
    pub reproductions: u8,
    pub movement_speed: f32,
    pub movement_direction: Vec3,
    pub velocity: Vec3,
    pub is_climbing: bool,
    /// Vertical metres climbed since the last energy tick, awaiting payment
    /// at `ELEVATION_ENERGY_PER_UNIT` per unit. Reset to 0 each tick. Krishi
    /// is excluded from the energy system entirely so its debt never drains.
    pub climb_energy_debt: f32,
}

impl Organism {
    /// Total currently-grown cells across alive body parts. Predation-
    /// consumed body parts are skipped — they no longer contribute to
    /// energy, weight or photosynthesis bookkeeping.
    #[inline]
    pub fn grown_cell_count(&self) -> usize {
        self.ocg.len()
    }

    /// (photo_count, non_photo_count) over grown cells of alive body parts.
    pub fn cell_counts(&self) -> (u32, u32) {
        let mut p = 0u32;
        let mut np = 0u32;
        for (_, _, ct) in &self.ocg {
            match ct {
                CellType::Photo    => p  += 1,
                CellType::NonPhoto => np += 1,
            }
        }
        (p, np)
    }

    /// Effective biological mass — proportional to grown cell count of
    /// alive body parts. Floored at 1.0 so single-cell juveniles don't
    /// divide by zero in energy / drag calculations.
    #[inline]
    pub fn weight(&self) -> f32 {
        (self.grown_cell_count() as f32).max(1.0)
    }

    /// Maximum world-space distance from the organism root that any grown
    /// cell on an alive body part can reach. Used by water buoyancy
    /// submersion and movement broad phase.
    pub fn bounding_radius(&self) -> f32 {
        let mut max_r = 2.0 * RD_HALF_SIZE; // single-cell starter floor
        for (_, pos, _) in &self.ocg {
            let r = pos.length() + MESH_PADDING.max(2.0 * RD_HALF_SIZE);
            if r > max_r { max_r = r; }
        }
        max_r
    }

    /// Number of body parts that still have cells and haven't been eaten.
    #[inline]
    pub fn alive_body_part_count(&self) -> usize {
        self.body_parts.iter().filter(|b| b.is_alive()).count()
    }
}


/// Marks an organism as a photoautotroph (energy from photosynthesis).
#[derive(Component, Clone, Copy)]
pub struct Photoautotroph;

/// Marks an organism as a heterotroph (energy from consuming other organisms).
#[derive(Component, Clone, Copy)]
pub struct Heterotroph;

#[derive(Component)]
pub struct OrganismRoot;


/// Trophic strategy chosen at spawn time. Decides which marker component is
/// inserted on the root entity and which colour the starter cell takes.
#[derive(Clone, Copy, Debug)]
pub enum OrganismKind {
    Photoautotroph,
    Heterotroph,
}

impl OrganismKind {
    #[inline]
    pub fn starter_cell_type(self) -> CellType {
        match self {
            OrganismKind::Photoautotroph => CellType::Photo,
            OrganismKind::Heterotroph    => CellType::NonPhoto,
        }
    }
}


// ── Spawning ─────────────────────────────────────────────────────────────────

fn spawn_colony(
    mut commands:  Commands,
    mut meshes:    ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    heightmap:     Res<HeightmapSampler>,
    mut spawned:   Local<bool>,
) {
    if *spawned { return; }
    *spawned = true;

    // One shared white material across every organism. StandardMaterial
    // multiplies base_color by ATTRIBUTE_COLOR, so a white base lets the
    // per-vertex cell colour shine through unchanged. Also: 1100 organisms
    // sharing one material handle keeps GPU bind-group churn minimal.
    let shared_material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        ..default()
    });

    let mut rng = rand::rng();

    for _ in 0..INITIAL_PHOTOAUTOTROPHS {
        let x = rng.random_range(0.0_f32..MAP_MAX_X);
        let z = rng.random_range(0.0_f32..MAP_MAX_Z);
        let y = heightmap.height_at(x, z) + 1.0;
        let ocg = vec![(0usize, Vec3::ZERO, CellType::Photo)];
        let max_e = (ocg.len() as f32) * crate::energy::MAX_ENERGY_PER_CELL;
        spawn_organism(
            Vec3::new(x, y, z),
            ocg,
            OrganismKind::Photoautotroph,
            max_e * 0.5,
            &mut commands,
            &mut meshes,
            &shared_material,
            &mut rng,
        );
    }

    for _ in 0..INITIAL_HETEROTROPHS {
        let x = rng.random_range(0.0_f32..MAP_MAX_X);
        let z = rng.random_range(0.0_f32..MAP_MAX_Z);
        let y = heightmap.height_at(x, z) + 1.0;
        let ocg = vec![(0usize, Vec3::ZERO, CellType::NonPhoto)];
        let max_e = (ocg.len() as f32) * crate::energy::MAX_ENERGY_PER_CELL;
        spawn_organism(
            Vec3::new(x, y, z),
            ocg,
            OrganismKind::Heterotroph,
            max_e * 0.5,
            &mut commands,
            &mut meshes,
            &shared_material,
            &mut rng,
        );
    }
}


/// Construct + register an organism from its OCG genome at world position
/// `pos`. The OCG is the organism's DNA — body shape is fully determined by
/// replaying it through `build_mesh_from_ocg`. Used by both initial colony
/// spawn and reproduction (children are adult clones of their parent's OCG).
///
/// Hierarchy produced:
///   OrganismRoot (transform = pos, has Organism + Photoautotroph/Heterotroph)
///   └── single body-part child (Mesh3d built from OCG replay)
pub fn spawn_organism(
    pos:            Vec3,
    ocg:            Vec<(usize, Vec3, CellType)>,
    kind:           OrganismKind,
    initial_energy: f32,
    commands:       &mut Commands,
    meshes:         &mut ResMut<Assets<Mesh>>,
    material:       &Handle<StandardMaterial>,
    rng:            &mut impl rand::Rng,
) -> Entity {
    let angle     = rng.random::<f32>() * std::f32::consts::TAU;
    let direction = Vec3::new(angle.cos(), 0.0, angle.sin());
    let speed     = match kind {
        OrganismKind::Photoautotroph => 0.0,
        OrganismKind::Heterotroph    => 15.0 + rng.random::<f32>() * 10.0,
    };

    // Single sentinel body part for predation soft-delete bookkeeping.
    let first_ct = ocg.first().map(|(_, _, ct)| *ct).unwrap_or(CellType::NonPhoto);
    let sentinel = BodyPart {
        kind:         BodyPartKind::Body,
        local_offset: Vec3::ZERO,
        cells:        vec![Cell { local_pos: Vec3::ZERO, cell_type: first_ct }],
        consumed:     false,
    };

    let organism = Organism {
        body_parts: vec![sentinel],
        ocg: ocg.clone(),
        energy: initial_energy.max(0.0),
        in_sunlight: false,
        reproduced: false,
        reproductions: 0,
        movement_speed: speed,
        movement_direction: direction,
        velocity: Vec3::ZERO,
        is_climbing: false,
        climb_energy_debt: 0.0,
    };

    let direction_interval = 1.0 + rng.random::<f32>() * 9.0;
    let spawn_rotation     = Quat::from_rotation_y(angle);

    let mut root_cmd = commands.spawn((
        Transform::from_translation(pos).with_rotation(spawn_rotation),
        Visibility::Visible,
        OrganismRoot,
        organism,
        DirectionTimer::new(direction_interval),
    ));
    match kind {
        OrganismKind::Photoautotroph => { root_cmd.insert(Photoautotroph); }
        OrganismKind::Heterotroph    => { root_cmd.insert(Heterotroph); }
    }
    let root = root_cmd.id();

    // One mesh child representing the full body, built by replaying the OCG.
    let mesh_handle = meshes.add(build_mesh_from_ocg(&ocg));
    let child = commands.spawn((
        Mesh3d(mesh_handle),
        MeshMaterial3d(material.clone()),
        Transform::from_translation(Vec3::ZERO),
        Visibility::Visible,
        OrganismMesh,
        BodyPartIndex(0),
        ShowGizmo,
    )).id();
    commands.entity(root).add_child(child);

    root
}
