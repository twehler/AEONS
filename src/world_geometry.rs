use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;
use std::fs::File;
use serde::{Deserialize, Serialize};

// Number of block types in your stacked texture PNG.
// Stack them vertically: stone (layer 0), dirt (layer 1),
// grass (layer 2), unknown (layer 3).
const BLOCK_TYPE_COUNT: u32 = 4;
const TERRAIN_SHADER_PATH: &str = "shaders/terrain.wgsl";

pub struct WorldPlugin {
    pub map_path:     String,
    pub texture_path: String, // path to the stacked block textures PNG
}

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<TerrainMaterial>::default());
        app.insert_resource(WorldSettings {
            map_path:     self.map_path.clone(),
            texture_path: self.texture_path.clone(),
        });
        app.add_systems(Startup,  begin_load_scene);
        app.add_systems(Update,   finish_load_scene);
    }
}

// --- Custom terrain material ---

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct TerrainMaterial {
    #[texture(0, dimension = "2d_array")]
    #[sampler(1)]
    pub textures: Handle<Image>,
}

impl Material for TerrainMaterial {
    fn fragment_shader() -> ShaderRef {
        TERRAIN_SHADER_PATH.into()
    }
}

// --- Serialized world cache ---

#[derive(Serialize, Deserialize)]
pub struct WorldCache {
    pub positions: Vec<[f32; 3]>,
    pub normals:   Vec<[f32; 3]>,
    pub uvs:       Vec<[f32; 2]>,
    pub colors:    Vec<[f32; 4]>,
    pub indices:   Vec<u32>,
}

// --- Resources ---

#[derive(Resource)]
struct WorldSettings {
    map_path:     String,
    texture_path: String,
}

#[derive(Resource)]
struct PendingTexture {
    handle:    Handle<Image>,
    is_loaded: bool,
    cache:     WorldCache,
}

// --- Systems ---

fn begin_load_scene(
    settings:    Res<WorldSettings>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
) {
    // Deserialize the world cache from disk immediately
    let file    = File::open(&settings.map_path).expect("Map file not found");
    let decoder = zstd::stream::read::Decoder::new(file)
        .expect("Failed to create Zstd decoder");
    let cache: WorldCache = bincode::deserialize_from(decoder)
        .expect("Failed to deserialize world file");

    // Start async loading of the block texture atlas
    let handle: Handle<Image> = asset_server.load(&settings.texture_path);

    commands.insert_resource(PendingTexture {
        handle,
        is_loaded: false,
        cache,
    });
}

fn finish_load_scene(
    mut commands:    Commands,
    mut pending:     ResMut<PendingTexture>,
    mut images:      ResMut<Assets<Image>>,
    mut meshes:      ResMut<Assets<Mesh>>,
    mut materials:   ResMut<Assets<TerrainMaterial>>,
    asset_server:    Res<AssetServer>,
) {
    if pending.is_loaded {
        return;
    }
    if !asset_server.load_state(pending.handle.id()).is_loaded() {
        return;
    }

    pending.is_loaded = true;

    // Reinterpret the stacked PNG as a texture array — one layer per block type
    let image = images.get_mut(&pending.handle).unwrap();
    image.reinterpret_stacked_2d_as_array(BLOCK_TYPE_COUNT);

    // Build the mesh from the cached data
    let cache = &pending.cache;
    let mut mesh = Mesh::new(
        bevy::render::render_resource::PrimitiveTopology::TriangleList,
        bevy::asset::RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, cache.positions.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL,   cache.normals.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0,     cache.uvs.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR,    cache.colors.clone());
    mesh.insert_indices(bevy::mesh::Indices::U32(cache.indices.clone()));

    let material = materials.add(TerrainMaterial {
        textures: pending.handle.clone(),
    });

    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(material),
        Transform::IDENTITY,
    ));
}
