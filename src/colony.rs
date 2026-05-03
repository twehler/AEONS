use crate::cell::*;
use crate::viewport_settings::ShowGizmo;
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
        self.body_parts.iter().filter(|b| b.is_alive())
            .map(|b| b.cells.len()).sum()
    }

    /// (photo_count, non_photo_count) over grown cells of alive body parts.
    pub fn cell_counts(&self) -> (u32, u32) {
        let mut p = 0u32;
        let mut np = 0u32;
        for bp in self.body_parts.iter().filter(|b| b.is_alive()) {
            let (a, b) = bp.cell_counts();
            p  += a;
            np += b;
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
        for bp in self.body_parts.iter().filter(|b| b.is_alive()) {
            let r = bp.local_offset.length() + bp.local_bounding_radius();
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
    /// Cell type of the organism's first cell. Photoautotrophs start green,
    /// heterotrophs start red — the recognisable initial state before any
    /// mutation kicks in.
    #[inline]
    pub fn starter_cell(self) -> CellType {
        match self {
            OrganismKind::Photoautotroph => CellType::Photo,
            OrganismKind::Heterotroph    => CellType::NonPhoto,
        }
    }
}


// ── Starter body part ────────────────────────────────────────────────────────

/// Construct the canonical 1-cell starter body part. The mesh generator
/// renders this as a rhombic dodecahedron — the "first instance" appearance
/// the spec calls for.
pub fn make_starter_body_part(cell_type: CellType) -> BodyPart {
    BodyPart {
        kind: BodyPartKind::Body,
        local_offset: Vec3::ZERO,
        cells: vec![ Cell { local_pos: Vec3::ZERO, cell_type } ],
        consumed: false,
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
        let parts = vec![ make_starter_body_part(CellType::Photo) ];
        let max_e = (parts.iter().map(|b| b.cells.len()).sum::<usize>() as f32)
                    * crate::energy::MAX_ENERGY_PER_CELL;
        spawn_organism(
            Vec3::new(x, y, z),
            parts,
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
        let parts = vec![ make_starter_body_part(CellType::NonPhoto) ];
        let max_e = (parts.iter().map(|b| b.cells.len()).sum::<usize>() as f32)
                    * crate::energy::MAX_ENERGY_PER_CELL;
        spawn_organism(
            Vec3::new(x, y, z),
            parts,
            OrganismKind::Heterotroph,
            max_e * 0.5,
            &mut commands,
            &mut meshes,
            &shared_material,
            &mut rng,
        );
    }
}


/// Construct + register an organism with the supplied body-part genome at
/// world position `pos`. Used by both initial colony spawn and reproduction.
///
/// Hierarchy produced:
///   OrganismRoot (transform = pos, has Organism + Photoautotroph/Heterotroph)
///   ├── BodyPart child 0 (transform = body_parts[0].local_offset, Mesh3d)
///   ├── BodyPart child 1 (transform = body_parts[1].local_offset, Mesh3d)
///   └── ...
pub fn spawn_organism(
    pos:            Vec3,
    body_parts:     Vec<BodyPart>,
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

    let organism = Organism {
        body_parts: body_parts.clone(),
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

    // One child entity per body part. Each child carries its own mesh,
    // its local offset relative to the root, and a `BodyPartIndex` so
    // systems that walk the hierarchy (mesh rebuild, gizmo overlays) can
    // map back to the slot in `Organism::body_parts` in O(1).
    for (i, bp) in body_parts.iter().enumerate() {
        let mesh_handle = meshes.add(generate_body_part_mesh(bp));
        let child = commands.spawn((
            Mesh3d(mesh_handle),
            MeshMaterial3d(material.clone()),
            Transform::from_translation(bp.local_offset),
            Visibility::Visible,
            OrganismMesh,
            BodyPartIndex(i),
            ShowGizmo,
        )).id();
        commands.entity(root).add_child(child);
    }

    root
}
