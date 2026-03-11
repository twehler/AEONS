#import bevy_pbr::{
    forward_io::VertexOutput,
    mesh_view_bindings::view,
    pbr_types::{PbrInput, pbr_input_new},
    pbr_functions as fns,
}
#import bevy_core_pipeline::tonemapping::tone_mapping

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var block_textures: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var block_sampler:  sampler;

@fragment
fn fragment(
    @builtin(front_facing) is_front: bool,
    mesh: VertexOutput,
) -> @location(0) vec4<f32> {

    // Decode block_type from the red vertex color channel.
    // generate_world.rs encodes it as: block_type as f32 / 255.0
    // Layer mapping matches the stacked PNG order:
    //   0 = stone, 1 = dirt, 2 = grass, 3 = unknown
    let block_type = i32(round(mesh.color.r * 255.0));
    let layer = block_type - 1; // block types are 1-indexed, layers are 0-indexed

    // fract() wraps the tiled UVs (0..width, 0..height) back to 0..1
    // per block, so the 16x16 texture repeats cleanly across the greedy quad.
    let tiled_uv = fract(mesh.uv);
    let color = textureSample(block_textures, block_sampler, tiled_uv, layer);

    // Feed through Bevy's PBR pipeline so lighting, shadows and fog work correctly
    var pbr_input: PbrInput = pbr_input_new();
    pbr_input.material.base_color = color;
    pbr_input.frag_coord          = mesh.position;
    pbr_input.world_position      = mesh.world_position;
    pbr_input.world_normal        = fns::prepare_world_normal(
        mesh.world_normal, false, is_front
    );
    pbr_input.is_orthographic = view.clip_from_view[3].w == 1.0;
    pbr_input.N = normalize(pbr_input.world_normal);
    pbr_input.V = fns::calculate_view(mesh.world_position, pbr_input.is_orthographic);

    return tone_mapping(fns::apply_pbr_lighting(pbr_input), view.color_grading);
}