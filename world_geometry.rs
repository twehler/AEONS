use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;
use bevy::image::{ImageSampler, ImageSamplerDescriptor, ImageFilterMode};
use std::fs::File;
use std::collections::HashMap;
use bincode::{Encode, Decode};

const TERRAIN_SHADER_PATH: &str = "shaders/terrain.wgsl";

const TILE_SIZE: u32 = 8;
const ATLAS_COLS: u32 = 16;
const ATLAS_ROWS: u32 = 16;
const BLOCK_TYPE_COUNT: u32 = ATLAS_COLS * ATLAS_ROWS;

// --- Plugin ---

pub struct WorldPlugin {
    pub terrain_path: String,
    pub texture_path: String,
}

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<TerrainMaterial>::default());
        app.insert_resource(WorldSettings {
            terrain_path: self.terrain_path.clone(),
            texture_path: self.texture_path.clone(),
        });
        app.add_systems(Startup, begin_load_scene);
        app.add_systems(Update,  finish_load_scene);
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
// Must exactly match the structs in generate_world.rs

#[derive(Encode, Decode)]
pub struct ChunkCache {
    pub chunk_coord: [i32; 2],
    pub positions:   Vec<[f32; 3]>,
    pub normals:     Vec<[f32; 3]>,
    pub uvs:         Vec<[f32; 2]>,
    pub colors:      Vec<[f32; 4]>,
    pub indices:     Vec<u32>,
}

#[derive(Encode, Decode)]
pub struct WorldCache {
    pub block_metadata: HashMap<[i32; 3], u8>,
    pub chunks:         Vec<ChunkCache>,
}

// --- Chunk component for later dirty-rebuild queries ---

#[derive(Component)]
pub struct TerrainChunk {
    pub chunk_coord: [i32; 2],
    pub dirty:       bool,
}

// --- Resources ---

#[derive(Resource)]
struct WorldSettings {
    terrain_path: String,
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
    settings:     Res<WorldSettings>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
) {
    let file        = File::open(&settings.terrain_path).expect("Map file not found");
    let mut decoder = zstd::stream::read::Decoder::new(file)
        .expect("Failed to create Zstd decoder");

    let cache: WorldCache =
        bincode::decode_from_std_read(&mut decoder, bincode::config::standard())
        .expect("Failed to deserialize world file");

    println!(
        "Loaded {} chunks, {} total blocks in metadata",
        cache.chunks.len(),
        cache.block_metadata.len()
    );

    let handle: Handle<Image> = asset_server.load(&settings.texture_path);

    commands.insert_resource(PendingTexture {
        handle,
        is_loaded: false,
        cache,
    });
}

fn finish_load_scene(
    mut commands:  Commands,
    mut pending:   ResMut<PendingTexture>,
    mut images:    ResMut<Assets<Image>>,
    mut meshes:    ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    asset_server:  Res<AssetServer>,
) {
    if pending.is_loaded {
        return;
    }
    if !asset_server.load_state(pending.handle.id()).is_loaded() {
        return;
    }

    pending.is_loaded = true;

    // Reinterpret the stacked PNG as a texture array
    let image = images.get_mut(&pending.handle).unwrap();
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        mag_filter:    ImageFilterMode::Nearest,
        min_filter:    ImageFilterMode::Nearest,
        mipmap_filter: ImageFilterMode::Nearest,
        ..default()
    });
    reorder_atlas_to_array(image, ATLAS_COLS, ATLAS_ROWS, TILE_SIZE);
    image.reinterpret_stacked_2d_as_array(BLOCK_TYPE_COUNT);

    let material = materials.add(TerrainMaterial {
        textures: pending.handle.clone(),
    });

    // Spawn one mesh entity per chunk
    for chunk in &pending.cache.chunks {
        let mut mesh = Mesh::new(
            bevy::render::render_resource::PrimitiveTopology::TriangleList,
            bevy::asset::RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, chunk.positions.clone());
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL,   chunk.normals.clone());
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0,     chunk.uvs.clone());
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR,    chunk.colors.clone());
        mesh.insert_indices(bevy::mesh::Indices::U32(chunk.indices.clone()));

        commands.spawn((
            Mesh3d(meshes.add(mesh.clone())),
            MeshMaterial3d(material.clone()),
            Transform::IDENTITY,
            RigidBody::Static,
            Collider::trimesh_from_mesh(&mesh)
                .expect("Failed to build terrain collider"),
            TerrainChunk {
                chunk_coord: chunk.chunk_coord,
                dirty:       false,
            },
));
        }

    println!("Spawned {} chunk meshes", pending.cache.chunks.len());
}

// --- Atlas reordering (unchanged) ---

fn reorder_atlas_to_array(
    image:     &mut Image,
    cols:      u32,
    rows:      u32,
    tile_size: u32,
) {
    use bevy::render::render_resource::{Extent3d, TextureFormat};

    image.convert(TextureFormat::Rgba8UnormSrgb);

    let expected_width  = cols * tile_size;
    let expected_height = rows * tile_size;
    let actual_width    = image.texture_descriptor.size.width;
    let actual_height   = image.texture_descriptor.size.height;

    assert_eq!(actual_width,  expected_width,
        "Atlas width mismatch: {}px wide but expected {}px", actual_width,  expected_width);
    assert_eq!(actual_height, expected_height,
        "Atlas height mismatch: {}px tall but expected {}px", actual_height, expected_height);

    let bytes_per_pixel = 4usize;
    let atlas_width     = expected_width as usize;
    let total_tiles     = (cols * rows) as usize;
    let tile_bytes      = tile_size as usize * tile_size as usize * bytes_per_pixel;

    let src = image.data.as_ref().expect("Image has no data").clone();
    let mut dst = vec![0u8; total_tiles * tile_bytes];

    for tile_idx in 0..total_tiles {
        let tile_col   = tile_idx % cols as usize;
        let tile_row   = tile_idx / cols as usize;
        let dst_offset = tile_idx * tile_bytes;

        for py in 0..tile_size as usize {
            for px in 0..tile_size as usize {
                let src_x      = tile_col * tile_size as usize + px;
                let src_y      = tile_row * tile_size as usize + py;
                let src_offset = (src_y * atlas_width + src_x) * bytes_per_pixel;
                let dst_pixel  = dst_offset + (py * tile_size as usize + px) * bytes_per_pixel;
                dst[dst_pixel..dst_pixel + 4]
                    .copy_from_slice(&src[src_offset..src_offset + 4]);
            }
        }
    }

    image.data = Some(dst);
    image.texture_descriptor.size = Extent3d {
        width:                 tile_size,
        height:                tile_size * total_tiles as u32,
        depth_or_array_layers: 1,
    };
}
