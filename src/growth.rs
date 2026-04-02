use bevy::prelude::*;
use crate::colony::*;
use crate::cell::{MeshCell, merge_organism_mesh, OrganismMesh, GLOBAL_CELL_SIZE};
use std::collections::HashMap;

// ── Constants ────────────────────────────────────────────────────────────────

const GROWTH_TICK_INTERVAL: f32 = 0.5;     // seconds between growth checks
const MESH_REBUILD_INTERVAL: f32 = 0.25;   // max 4Hz mesh rebuilds
const ENERGY_PER_CELL_GROWTH: f32 = 2.0;   // energy cost to grow one cell
const CELLS_PER_TICK: usize = 1;            // max cells grown per tick

// ── Components ───────────────────────────────────────────────────────────────

#[derive(Component)]
pub struct MeshDirty;

// ── Timer resources ──────────────────────────────────────────────────────────

#[derive(Resource)]
pub struct GrowthTimer {
    pub timer: Timer,
}

impl Default for GrowthTimer {
    fn default() -> Self {
        Self {
            timer: Timer::from_seconds(GROWTH_TICK_INTERVAL, TimerMode::Repeating),
        }
    }
}

#[derive(Resource)]
pub struct MeshRebuildTimer {
    pub timer: Timer,
}

impl Default for MeshRebuildTimer {
    fn default() -> Self {
        Self {
            timer: Timer::from_seconds(MESH_REBUILD_INTERVAL, TimerMode::Repeating),
        }
    }
}

// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct GrowthPlugin;

impl Plugin for GrowthPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(GrowthTimer::default());
        app.insert_resource(MeshRebuildTimer::default());
        app.add_systems(Update, (
            growth_system,
            mesh_rebuild_system,
        ));
    }
}

// ── Systems ──────────────────────────────────────────────────────────────────

fn growth_system(
    time: Res<Time>,
    mut timer: ResMut<GrowthTimer>,
    mut commands: Commands,
    mut query: Query<(Entity, &mut Organism), With<OrganismRoot>>,
) {
    timer.timer.tick(time.delta());
    if !timer.timer.just_finished() {
        return;
    }

    for (entity, mut organism) in &mut query {
        // Already fully grown
        if organism.grown_cell_count >= organism.ocg.len() {
            if !organism.adult {
                organism.adult = true;
            }
            continue;
        }

        // Grow up to CELLS_PER_TICK cells if energy allows
        let mut cells_grown = 0;
        while cells_grown < CELLS_PER_TICK
            && organism.grown_cell_count < organism.ocg.len()
            && organism.energy >= ENERGY_PER_CELL_GROWTH
        {
            organism.energy -= ENERGY_PER_CELL_GROWTH;
            organism.grown_cell_count += 1;
            cells_grown += 1;
        }

        if cells_grown > 0 {
            // Rebuild active_cells from grown portion of OCG
            organism.active_cells = organism.ocg[..organism.grown_cell_count]
                .iter()
                .filter_map(|entry| {
                    organism.collections.get(&entry.collection_id).map(|coll| {
                        (coll.starter_cell_position + entry.offset, entry.cell_type)
                    })
                })
                .collect();

            // Mark mesh dirty
            commands.entity(entity).insert(MeshDirty);
        }
    }
}

fn mesh_rebuild_system(
    time: Res<Time>,
    mut timer: ResMut<MeshRebuildTimer>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    dirty_query: Query<(Entity, &Organism), (With<OrganismRoot>, With<MeshDirty>)>,
    children_query: Query<&Children>,
    mesh_query: Query<Entity, With<OrganismMesh>>,
) {
    timer.timer.tick(time.delta());
    if !timer.timer.just_finished() {
        return;
    }

    for (entity, organism) in &dirty_query {
        // Build new mesh from grown cells
        let mesh_cells: Vec<MeshCell> = organism.ocg[..organism.grown_cell_count]
            .iter()
            .filter_map(|entry| {
                organism.collections.get(&entry.collection_id).map(|coll| {
                    MeshCell {
                        mesh_space_pos: coll.starter_cell_position + entry.offset,
                        cell_type: entry.cell_type,
                    }
                })
            })
            .collect();

        if mesh_cells.is_empty() { continue; }

        let new_mesh = meshes.add(merge_organism_mesh(mesh_cells));

        // Find and update the existing mesh child entity
        if let Ok(children) = children_query.get(entity) {
            for child in children.iter() {
                if mesh_query.get(child).is_ok() {
                    commands.entity(child).remove::<Mesh3d>();
                    commands.entity(child).insert(Mesh3d(new_mesh.clone()));
                    break;
                }
            }
        }

        // Remove dirty marker
        commands.entity(entity).remove::<MeshDirty>();
    }
}
