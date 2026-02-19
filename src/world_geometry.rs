use bevy::prelude::*;
use std::fs::File;
use serde::{Deserialize, Serialize};

pub struct WorldPlugin {
    pub map_path: String,
}

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(WorldSettings{map_path: self.map_path.clone()});
        app.add_systems(Startup, load_scene);
        }
}

// struct which represents the entire world, loaded from a binary file
#[derive(Serialize, Deserialize)]
pub struct WorldCache {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
}


#[derive(Resource)]
struct WorldSettings {
    map_path: String,
}

fn load_scene(
    settings: Res<WorldSettings>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>
) {
    // load world file
    let file = File::open(&settings.map_path).expect("Map file not found");

    // Wrap the file in a decoder
    let decoder = zstd::stream::read::Decoder::new(file)
        .expect("Failed to create Zstd decoder");

    let cache: WorldCache = bincode::deserialize_from(decoder)
        .expect("Failed to deserialize compressed world file.");

    let mut mesh = Mesh::new(
        bevy::render::render_resource::PrimitiveTopology::TriangleList,
        bevy::asset::RenderAssetUsages::default(),
    );

    // We take the data directly from the cache
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, cache.positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, cache.normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, cache.uvs);
    mesh.insert_indices(bevy::mesh::Indices::U32(cache.indices));

    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.3, 0.7, 0.3),
            ..default()
        })),
        Transform::IDENTITY,
    ));
}

