use bevy::prelude::*;
use bevy::mesh::Indices;
pub use bevy::render::render_resource::PrimitiveTopology;
pub use bevy::asset::RenderAssetUsages;
use serde::{Serialize, Deserialize};
use image::GenericImageView;
use std::collections::{HashSet, HashMap};
use std::fs::File;
use std::io::BufWriter;
use bevy::prelude::Vec3; // Using Bevy's math for consistency
use std::env;
use std::path::Path;
use rayon::prelude::*; // for multithreading


#[derive(Serialize, Deserialize)]
pub struct WorldCache {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
}


#[derive(Serialize, Deserialize, Debug)]
pub struct TerrainData {
    pub width: usize,
    pub depth: usize,
    pub columns: Vec<Column>,  // Flat 1D vector of all columns
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct Column {
    pub height: u8,      // Height at this position (1-10)
    pub block_type: u8,  // Block type (1 = grass, can expand later)
}



pub struct VoxelBuffer {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
    pub position_list: Vec<[i32; 3]>,
}

impl VoxelBuffer {
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
            position_list: Vec::new()
        }
    }

    pub fn build(self) -> Mesh {
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, self.uvs)
        .with_inserted_indices(Indices::U32(self.indices))
    }
}

pub fn append_world_voxel(
    buffer: &mut VoxelBuffer,
    pos: [i32; 3],
    size: f32,
    occupied: &HashSet<[i32; 3]>,
) {
    let s = size / 2.0; // Half size for centered cube
    let center = Vec3::new(pos[0] as f32, pos[1] as f32, pos[2] as f32);

    let vertex_offset = buffer.positions.len() as u32;


    let directions = [
        ([0, 0, 1],  "front",  Vec3::Z,  [[-s,-s, s], [s,-s, s], [s, s, s], [-s, s, s]]),
        ([0, 0, -1], "back",   -Vec3::Z, [[s,-s,-s], [-s,-s,-s], [-s, s,-s], [s, s,-s]]),
        ([1, 0, 0],  "right",  Vec3::X,  [[s,-s, s], [s,-s,-s], [s, s,-s], [s, s, s]]),
        ([-1, 0, 0], "left",   -Vec3::X, [[-s,-s,-s], [-s,-s, s], [-s, s, s], [-s, s,-s]]),
        ([0, 1, 0],  "top",    Vec3::Y,  [[-s, s, s], [s, s, s], [s, s,-s], [-s, s,-s]]),
        ([0, -1, 0], "bottom", -Vec3::Y, [[-s,-s,-s], [s,-s,-s], [s,-s, s], [-s,-s, s]]),
    ];



    for (offset, _name, normal, face_verts) in directions {
        let neighbor = [pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2]];

        // ONLY append the face if the neighbor position is empty
        if !occupied.contains(&neighbor) {
            let vertex_offset = buffer.positions.len() as u32;

            // Add 4 vertices for this face
            for v in face_verts {
                buffer.positions.push([v[0] + center.x, v[1] + center.y, v[2] + center.z]);
                buffer.normals.push(normal.to_array());
            }

            // Add standard UVs for one face
            buffer.uvs.extend([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);

            // Add 2 triangles (indices) for this face
            buffer.indices.extend([
                vertex_offset, vertex_offset + 1, vertex_offset + 2,
                vertex_offset, vertex_offset + 2, vertex_offset + 3,
            ]);
        }
    }
}


fn add_greedy_quad(
    buffer: &mut VoxelBuffer,
    axis: usize,
    is_front: bool,
    u_coord: i32,
    v_coord: i32,
    slice_coord: i32,
    width: i32,
    height: i32,
) {
    let vertex_offset = buffer.positions.len() as u32;
    let mut v0 = [0.0; 3]; let mut v1 = [0.0; 3];
    let mut v2 = [0.0; 3]; let mut v3 = [0.0; 3];

    let u = (axis + 1) % 3;
    let v = (axis + 2) % 3;

    // Define vertices based on the current sweep axis
    let offset = if is_front { 1.0 } else { 0.0 };
    v0[axis] = slice_coord as f32 + offset; v0[u] = u_coord as f32; v0[v] = v_coord as f32;
    v1[axis] = slice_coord as f32 + offset; v1[u] = (u_coord + width) as f32; v1[v] = v_coord as f32;
    v2[axis] = slice_coord as f32 + offset; v2[u] = (u_coord + width) as f32; v2[v] = (v_coord + height) as f32;
    v3[axis] = slice_coord as f32 + offset; v3[u] = u_coord as f32; v3[v] = (v_coord + height) as f32;

    buffer.positions.extend([v0, v1, v2, v3]);

    let mut normal = [0.0; 3];
    normal[axis] = if is_front { 1.0 } else { -1.0 };
    buffer.normals.extend([normal; 4]);

    // Scaling UVs prevents textures from stretching across the large quad
    buffer.uvs.extend([[0.0, 0.0], [width as f32, 0.0], [width as f32, height as f32], [0.0, height as f32]]);

    if is_front {
        buffer.indices.extend([vertex_offset, vertex_offset + 1, vertex_offset + 2, vertex_offset, vertex_offset + 2, vertex_offset + 3]);
    } else {
        buffer.indices.extend([vertex_offset, vertex_offset + 2, vertex_offset + 1, vertex_offset, vertex_offset + 3, vertex_offset + 2]);
    }
}



fn main() {

    let args: Vec<String> = env::args().collect();

    // args[0] is always the program name
    // args[1] = input path
    // args[2] = max_height
    // args[3] = output_path
    if args.len() < 3 {
        println!("Usage: cargo run --bin generate_world_cache <input.png> <max_height> <output.bin>");
        return;
    }

    let input_path = Path::new(&args[1]);
    let max_height = args[2]
        .parse::<f32>()
        .expect("The second argument must be a valid number.");

    let output_path = Path::new(&args[3]);


    println!("--- Starting Full World Mesh Generation ---");

    // 1. Load Image and generate the occupied voxel set (Your current logic)
    let img = image::open(input_path)
        .expect("Input image not present at specified location.")
        .to_luma8();

    let (width, depth) = img.dimensions();
    let mut occupied = HashSet::new();

    // normalize greyscale pixel values from range 0 - 255 to 1 - custom height
    // in order to not create a world which is 255 blocks high...
    for y in 0..depth {
        for x in 0..width {
            let pixel = img.get_pixel(x, y)[0];
            let height = ((pixel as f32 / 255.0) * max_height).round() as i32 + 1;

            for z_height in 0..height {
                occupied.insert([x as i32, z_height, y as i32]);
            }
        }
    }

    // build the Geometry using Greedy Meshing
    let mut buffer = VoxelBuffer::new();
    
    // 1. Calculate world bounds
    let mut min = [i32::MAX; 3];
    let mut max = [i32::MIN; 3];
    for pos in &occupied {
        for i in 0..3 {
            min[i] = min[i].min(pos[i]);
            max[i] = max[i].max(pos[i]);
        }
    }

    // 2. Greedy Sweep over 3 axes (X=0, Y=1, Z=2)
    for d in 0..3 {
        let u = (d + 1) % 3;
        let v = (d + 2) % 3;
        let mut x = [0; 3];
        let mut q = [0; 3]; q[d] = 1;

        for i in (min[d] - 1)..=(max[d]) {
            let mut mask = std::collections::HashMap::new();
            for j in min[u]..=max[u] {
                for k in min[v]..=max[v] {
                    x[d] = i; x[u] = j; x[v] = k;
                    let current = occupied.contains(&x);
                    let neighbor = occupied.contains(&[x[0]+q[0], x[1]+q[1], x[2]+q[2]]);
                    if current != neighbor { 
                        mask.insert((j, k), current); 
                    }
                }
            }

            // Greedy merge logic for the 2D mask
            for j in min[u]..=max[u] {
                for k in min[v]..=max[v] {
                    if let Some(front) = mask.remove(&(j, k)) {
                        let mut w = 1;
                        while mask.get(&(j + w, k)) == Some(&front) {
                            mask.remove(&(j + w, k)); w += 1;
                        }
                        let mut h = 1;
                        'h_loop: loop {
                            for r in 0..w {
                                if mask.get(&(j + r, k + h)) != Some(&front) { break 'h_loop; }
                            }
                            for r in 0..w { mask.remove(&(j + r, k + h)); }
                            h += 1;
                        }
                        add_greedy_quad(&mut buffer, d, front, j, k, i, w, h);
                    }
                }
            }
        }
    }

    // save the result as a binary cache
    let cache = WorldCache {
        positions: buffer.positions,
        normals: buffer.normals,
        uvs: buffer.uvs,
        indices: buffer.indices,
    };

    let file = File::create(output_path).expect("Failed to create output file.");
    let writer = BufWriter::new(file);

    // binary cache will also be compressed with zstd
    let mut encoder = zstd::stream::write::Encoder::new(writer, 3)
    .expect("Failed to create Zstd encoder");

    bincode::serialize_into(&mut encoder, &cache).expect("Failed to serialize world file.");

    // finish the encoder to flush the remaining compressed bits
    encoder.finish().expect("Failed to finish Zstd encoding");

    println!("World mesh saved! Vertices: {}", cache.positions.len());
}
