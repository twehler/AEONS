use crate::cell;
use bevy::prelude::*;
use avian3d::prelude::*;

pub struct ColonyPlugin;

impl Plugin for ColonyPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_colony);
    }
}



fn spawn_colony(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    cell::spawn_rhombic_dodecahedron(Vec3::new(40.0, 40.0, 40.0), &cell::CellType::BlueCell, &mut commands, &mut meshes, &mut materials);
    cell::spawn_rhombic_dodecahedron(Vec3::new(41.0, 41.0, 40.0), &cell::CellType::RedCell, &mut commands, &mut meshes, &mut materials);
}

