struct Uniforms {
    light_view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<storage, read> models: array<mat4x4<f32>>;

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec4<f32>,
};

@vertex
fn vertex_main(input: VertexInput) -> @builtin(position) vec4<f32> {
    let world_pos = models[input.instance_index] * vec4<f32>(input.position.xyz, 1.0);
    return uniforms.light_view_proj * world_pos;
}
