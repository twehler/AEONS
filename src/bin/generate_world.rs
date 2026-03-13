use log::warn;
use bevy::prelude::*;
use bevy::mesh::Indices;
pub use bevy::render::render_resource::PrimitiveTopology;
pub use bevy::asset::RenderAssetUsages;
use serde::{Serialize, Deserialize};
use image::GenericImageView;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::env;
use std::path::Path;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

#[derive(Serialize, Deserialize)]
pub struct WorldCache {
    pub positions:   Vec<[f32; 3]>,
    pub block_metadata: Vec<u8>, // block-type
    pub normals:     Vec<[f32; 3]>,
    pub uvs:         Vec<[f32; 2]>,
    pub colors:      Vec<[f32; 4]>, // r = block_type / 255.0, gba unused
    pub indices:     Vec<u32>,
}

pub struct VoxelBuffer {
    pub positions: Vec<[f32; 3]>,
    pub block_metadata: Vec<u8>, 
    pub normals:   Vec<[f32; 3]>,
    pub uvs:       Vec<[f32; 2]>,
    pub colors:    Vec<[f32; 4]>,
    pub indices:   Vec<u32>,
}

impl VoxelBuffer {
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            block_metadata: Vec::new(),
            normals:   Vec::new(),
            uvs:       Vec::new(),
            colors:    Vec::new(),
            indices:   Vec::new(),
        }
    }
}


const BLOCK_CHALK_1: u8 = 0;
const BLOCK_CHALK_2: u8 = 1;
const BLOCK_CHALK_3: u8 = 2;
const BLOCK_GREYSTONE_1: u8 = 3;
const BLOCK_GREYSTONE_2: u8 = 4;
const BLOCK_GREYSTONE_3: u8 = 5;
const BLOCK_GREYSTONE_4: u8 = 6;
const BLOCK_BROWNSTONE: u8 = 7;
const BLOCK_SAND_1: u8 = 8;
const BLOCK_SAND_2: u8 = 9;
const BLOCK_RED_SAND_1: u8 = 10;
const BLOCK_RED_SAND_2: u8 = 11;
const BLOCK_AQUATIC_SAND_1: u8 = 12;
const BLOCK_AQUATIC_SAND_2: u8 = 13;
const BLOCK_AQUATIC_SAND_3: u8 = 14;
const BLOCK_AQUATIC_SAND_4: u8 = 15;

const BLOCK_CHALK_1_BIOLAYER_1: u8 = 32;
const BLOCK_CHALK_2_BIOLAYER_1: u8 = 33;
const BLOCK_CHALK_3_BIOLAYER_1: u8 = 34;
const BLOCK_GREYSTONE_1_BIOLAYER_1: u8 = 35;
const BLOCK_GREYSTONE_2_BIOLAYER_1: u8 = 36;
const BLOCK_GREYSTONE_3_BIOLAYER_1: u8 = 37;
const BLOCK_GREYSTONE_4_BIOLAYER_1: u8 = 38;
const BLOCK_BROWNSTONE_BIOLAYER_1: u8 = 39;
const BLOCK_SAND_1_BIOLAYER_1: u8 = 40;
const BLOCK_SAND_2_BIOLAYER_1: u8 = 41;
const BLOCK_RED_SAND_1_BIOLAYER_1: u8 = 42;
const BLOCK_RED_SAND_2_BIOLAYER_1: u8 = 43;
const BLOCK_AQUATIC_SAND_1_BIOLAYER_1: u8 = 44;
const BLOCK_AQUATIC_SAND_2_BIOLAYER_1: u8 = 45;
const BLOCK_AQUATIC_SAND_3_BIOLAYER_1: u8 = 46;
const BLOCK_AQUATIC_SAND_4_BIOLAYER_1: u8 = 47;

const BLOCK_CHALK_1_BIOLAYER_2: u8 = 64;
const BLOCK_CHALK_2_BIOLAYER_2: u8 = 65;
const BLOCK_CHALK_3_BIOLAYER_2: u8 = 66;
const BLOCK_GREYSTONE_1_BIOLAYER_2: u8 = 67;
const BLOCK_GREYSTONE_2_BIOLAYER_2: u8 = 68;
const BLOCK_GREYSTONE_3_BIOLAYER_2: u8 = 69;
const BLOCK_GREYSTONE_4_BIOLAYER_2: u8 = 70;
const BLOCK_BROWNSTONE_BIOLAYER_2: u8 = 71;
const BLOCK_SAND_1_BIOLAYER_2: u8 = 72;
const BLOCK_SAND_2_BIOLAYER_2: u8 = 73;
const BLOCK_RED_SAND_1_BIOLAYER_2: u8 = 74;
const BLOCK_RED_SAND_2_BIOLAYER_2: u8 = 75;
const BLOCK_AQUATIC_SAND_1_BIOLAYER_2: u8 = 76;
const BLOCK_AQUATIC_SAND_2_BIOLAYER_2: u8 = 77;
const BLOCK_AQUATIC_SAND_3_BIOLAYER_2: u8 = 78;
const BLOCK_AQUATIC_SAND_4_BIOLAYER_2: u8 = 79;


const BLOCK_UNKNOWN: u8 = 255;






fn add_greedy_quad(
    buffer: &mut VoxelBuffer,
    axis: usize,
    is_front: bool,
    u_coord: i32,
    v_coord: i32,
    slice_coord: i32,
    width: i32,
    height: i32,
    block_type: u8,
) {
    let vertex_offset = buffer.positions.len() as u32;
    let u = (axis + 1) % 3;
    let v = (axis + 2) % 3;

    let plane_pos = slice_coord as f32 + 1.0;

    let mut v0 = [0.0f32; 3]; let mut v1 = [0.0f32; 3];
    let mut v2 = [0.0f32; 3]; let mut v3 = [0.0f32; 3];

    v0[axis] = plane_pos; v0[u] = u_coord as f32;           v0[v] = v_coord as f32;
    v1[axis] = plane_pos; v1[u] = (u_coord + width) as f32; v1[v] = v_coord as f32;
    v2[axis] = plane_pos; v2[u] = (u_coord + width) as f32; v2[v] = (v_coord + height) as f32;
    v3[axis] = plane_pos; v3[u] = u_coord as f32;           v3[v] = (v_coord + height) as f32;

    buffer.positions.extend([v0, v1, v2, v3]);

    let mut normal = [0.0f32; 3];
    normal[axis] = if is_front { 1.0 } else { -1.0 };
    buffer.normals.extend([normal; 4]);

    // UVs tile by the quad's size in blocks — no atlas math here at all.
    // fract() in the shader wraps these back to 0..1 per block, giving
    // a clean per-block texture repeat across the entire greedy quad.
    buffer.uvs.extend([
        [0.0,          0.0          ],
        [width as f32, 0.0          ],
        [width as f32, height as f32],
        [0.0,          height as f32],
    ]);

    // Pack block_type into the red channel of vertex color.
    // The shader reads it back as: i32(color.r * 255.0)
    let encoded = block_type as f32 / 255.0;

    buffer.colors.extend([[encoded, 0.0, 0.0, 1.0]; 4]);

    if is_front {
        buffer.indices.extend([
            vertex_offset, vertex_offset+1, vertex_offset+2,
            vertex_offset, vertex_offset+2, vertex_offset+3,
        ]);
    } else {
        buffer.indices.extend([
            vertex_offset, vertex_offset+2, vertex_offset+1,
            vertex_offset, vertex_offset+3, vertex_offset+2,
        ]);
    }
}

fn main() {

    // count all blocks of unknown type (useful at the end of this code)
    let mut unknown_blocks = 0;

    // Closure-function to assign blocks based in their signature color inside the material map
    let mut material_map_color_to_block = |r: u8, g: u8, b: u8| -> u8 {
        match (r, g, b) {
            (0xff, 0xff, 0xff) => BLOCK_CHALK_1,
            (0xf4, 0xe8, 0xe2) => BLOCK_CHALK_2,
            (0xe9, 0xd2, 0xc6) => BLOCK_CHALK_3,
            (0xc8, 0xc8, 0xc8) => BLOCK_GREYSTONE_1,
            (0xaa, 0xaa, 0xaa) => BLOCK_GREYSTONE_2,
            (0x99, 0x99, 0x99) => BLOCK_GREYSTONE_3,
            (0x7e, 0x7e, 0x7e) => BLOCK_GREYSTONE_4,
            (0xae, 0xa3, 0x9b) => BLOCK_BROWNSTONE,
            (0xe0, 0xd3, 0xc6) => BLOCK_SAND_1,
            (0xff, 0xe9, 0xc8) => BLOCK_SAND_2,
            (0xff, 0xdd, 0xa9) => BLOCK_RED_SAND_1,
            (0xf4, 0xca, 0x8b) => BLOCK_RED_SAND_2,
            (0xb4, 0xc1, 0xbf) => BLOCK_AQUATIC_SAND_1,
            (0x9a, 0xb1, 0xd1) => BLOCK_AQUATIC_SAND_2,
            (0x8f, 0x95, 0xaf) => BLOCK_AQUATIC_SAND_3,
            (0x74, 0x77, 0x85) => BLOCK_AQUATIC_SAND_4,

            (0xff, 0xda, 0xc7) => BLOCK_CHALK_1_BIOLAYER_1,
            (0xff, 0xcc, 0xb1) => BLOCK_CHALK_2_BIOLAYER_1,
            (0xf4, 0xba, 0x9b) => BLOCK_CHALK_3_BIOLAYER_1,
            (0xd6, 0x97, 0x76) => BLOCK_GREYSTONE_1_BIOLAYER_1,
            (0xae, 0x7b, 0x61) => BLOCK_GREYSTONE_2_BIOLAYER_1,
            (0x9c, 0x70, 0x5a) => BLOCK_GREYSTONE_3_BIOLAYER_1,
            (0x7d, 0x59, 0x47) => BLOCK_GREYSTONE_4_BIOLAYER_1,
            (0xb0, 0x8a, 0x6e) => BLOCK_BROWNSTONE_BIOLAYER_1,
            (0xda, 0xac, 0x7e) => BLOCK_SAND_1_BIOLAYER_1,
            (0xff, 0xb4, 0x68) => BLOCK_SAND_2_BIOLAYER_1,
            (0xff, 0xa1, 0x42) => BLOCK_RED_SAND_1_BIOLAYER_1,
            (0xff, 0x80, 0x00) => BLOCK_RED_SAND_2_BIOLAYER_1,
            (0xca, 0xeb, 0xe3) => BLOCK_AQUATIC_SAND_1_BIOLAYER_1,
            (0xad, 0xd7, 0xcd) => BLOCK_AQUATIC_SAND_2_BIOLAYER_1,
            (0x6e, 0xa7, 0x9a) => BLOCK_AQUATIC_SAND_3_BIOLAYER_1,
            (0x55, 0x8d, 0x80) => BLOCK_AQUATIC_SAND_4_BIOLAYER_1,

            (0xcc, 0xd8, 0x9b) => BLOCK_CHALK_1_BIOLAYER_2,
            (0x9d, 0xc4, 0x6c) => BLOCK_CHALK_2_BIOLAYER_2,
            (0x7c, 0xb9, 0x4f) => BLOCK_CHALK_3_BIOLAYER_2,
            (0x6d, 0xaf, 0x43) => BLOCK_GREYSTONE_1_BIOLAYER_2,
            (0x71, 0xa0, 0x4e) => BLOCK_GREYSTONE_2_BIOLAYER_2,
            (0x47, 0x7e, 0x22) => BLOCK_GREYSTONE_3_BIOLAYER_2,
            (0x43, 0x7f, 0x1e) => BLOCK_GREYSTONE_4_BIOLAYER_2,
            (0x64, 0x8a, 0x4e) => BLOCK_BROWNSTONE_BIOLAYER_2,
            (0xa1, 0xa5, 0x77) => BLOCK_SAND_1_BIOLAYER_2,
            (0xcc, 0xd3, 0x8e) => BLOCK_SAND_2_BIOLAYER_2,
            (0xa9, 0xb9, 0x69) => BLOCK_RED_SAND_1_BIOLAYER_2,
            (0xb0, 0xbd, 0x71) => BLOCK_RED_SAND_2_BIOLAYER_2,
            (0x8a, 0x9b, 0x73) => BLOCK_AQUATIC_SAND_1_BIOLAYER_2,
            (0x75, 0x92, 0x66) => BLOCK_AQUATIC_SAND_2_BIOLAYER_2,
            (0x6e, 0x8c, 0x5e) => BLOCK_AQUATIC_SAND_3_BIOLAYER_2,
            (0x38, 0x56, 0x35) => BLOCK_AQUATIC_SAND_4_BIOLAYER_2,

            _                  => {
                unknown_blocks += 2; // increment by 2 because the terrain-surface is 2 blocks deep
                BLOCK_UNKNOWN
            }
        }
    };



    let args: Vec<String> = env::args().collect();

    if args.len() < 5 {
        println!(
            "Usage: cargo run --bin generate_world \
             <heightmap.png> <max_height> <material_map.png> <output.bin>"
        );
        return;
    }

    let input_path    = Path::new(&args[1]);
    let max_height    = args[2].parse::<f32>().expect("max_height must be a number");
    let material_path = Path::new(&args[3]);
    let output_path   = Path::new(&args[4]);

    println!("--- Starting World Mesh Generation ---");

    println!("Loading heightmap...");
    let img = image::open(input_path)
        .expect("Heightmap not found")
        .to_luma8();
    let (width, depth) = img.dimensions();

    println!("Loading material map...");
    let mat_img = image::open(material_path)
        .expect("Material map not found")
        .to_rgb8();

    assert_eq!(
        mat_img.dimensions(), (width, depth),
        "Material map must match heightmap dimensions"
    );

    println!("Collecting block positions & types...");
    let start = Instant::now();
    let mut voxel_map: HashMap<[i32; 3], u8> = HashMap::new();


    for y in 0..depth {
        for x in 0..width {
            let pixel      = img.get_pixel(x, y)[0];
            let col_height = ((pixel as f32 / 255.0) * max_height).round() as i32 + 1;

            let px = mat_img.get_pixel(x, y);
            let surface_type = material_map_color_to_block(px[0], px[1], px[2]);

            for z in 0..col_height {
                voxel_map.insert([x as i32, z, y as i32], surface_type);
        }
        }
    }
    println!("Completed in {:?}", start.elapsed());

    let mut min  = [i32::MAX; 3];
    let mut max_b = [i32::MIN; 3];
    for pos in voxel_map.keys() {
        for i in 0..3 {
            min[i]    = min[i].min(pos[i]);
            max_b[i]  = max_b[i].max(pos[i]);
        }
    }

    let voxel_map = Arc::new(voxel_map);

    let mut tasks = Vec::new();
    for d in 0..3usize {
        for i in (min[d] - 1)..=max_b[d] {
            tasks.push((d, i));
        }
    }

    println!("Building terrain...");
    let start_gm = Instant::now();

    let partial_buffers: Vec<VoxelBuffer> = tasks.into_par_iter().map(|(d, i)| {
        let voxel_map = Arc::clone(&voxel_map);
        let u = (d + 1) % 3;
        let v = (d + 2) % 3;
        let mut q = [0i32; 3]; q[d] = 1;

        let mut local_buffer = VoxelBuffer::new();
        let mut mask: HashMap<(i32, i32), (bool, u8)> = HashMap::new();
        let mut x = [0i32; 3];
        x[d] = i;

        for j in min[u]..=max_b[u] {
            for k in min[v]..=max_b[v] {
                x[u] = j; x[v] = k;
                let current_type  = voxel_map.get(&x).copied();
                let neighbor_type = voxel_map.get(
                    &[x[0]+q[0], x[1]+q[1], x[2]+q[2]]
                ).copied();

                match (current_type, neighbor_type) {
                    (Some(ct), None) => { mask.insert((j, k), (true,  ct)); }
                    (None, Some(nt)) => { mask.insert((j, k), (false, nt)); }
                    _ => {}
                }
            }
        }

        for j in min[u]..=max_b[u] {
            for k in min[v]..=max_b[v] {
                if let Some((front, btype)) = mask.remove(&(j, k)) {
                    let mut w = 1i32;
                    while mask.get(&(j + w, k)) == Some(&(front, btype)) {
                        mask.remove(&(j + w, k));
                        w += 1;
                    }
                    let mut h = 1i32;
                    'h_loop: loop {
                        for r in 0..w {
                            if mask.get(&(j + r, k + h)) != Some(&(front, btype)) {
                                break 'h_loop;
                            }
                        }
                        for r in 0..w { mask.remove(&(j + r, k + h)); }
                        h += 1;
                    }
                    add_greedy_quad(&mut local_buffer, d, front, j, k, i, w, h, btype);
                }
            }
        }
        local_buffer
    }).collect();

    println!("Building-process completed in {:?}", start_gm.elapsed());

    let mut buffer = VoxelBuffer::new();
    for mut pb in partial_buffers {
        let offset = buffer.positions.len() as u32;
        buffer.positions.append(&mut pb.positions);
        buffer.normals.append(&mut pb.normals);
        buffer.uvs.append(&mut pb.uvs);
        buffer.colors.append(&mut pb.colors);
        for idx in pb.indices { buffer.indices.push(idx + offset); }
    }

    let cache = WorldCache {
        positions: buffer.positions,
        normals:   buffer.normals,
        uvs:       buffer.uvs,
        colors:    buffer.colors,
        indices:   buffer.indices,
    };

    println!("Saving...");
    let file    = File::create(output_path).expect("Failed to create output file");
    let writer  = BufWriter::new(file);
    let mut enc = zstd::stream::write::Encoder::new(writer, 3)
        .expect("Failed to create Zstd encoder");
    bincode::serialize_into(&mut enc, &cache).expect("Failed to serialize");
    enc.finish().expect("Failed to finish Zstd encoding");

    println!("Done! Vertices: {}", cache.positions.len());

    if unknown_blocks != 0 {
        println!("WARNING: Some colors in the Material-Map could not be matched to known block types! {} blocks with unknown type have been generated in total.", unknown_blocks);
    }
}
