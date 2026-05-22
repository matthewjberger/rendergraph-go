struct PointShadowUniforms {
    view_projection: mat4x4<f32>,
    light_position: vec3<f32>,
    light_range: f32,
};

@group(0) @binding(0) var<uniform>       uniforms: PointShadowUniforms;
@group(1) @binding(0) var<storage, read> models:   array<mat4x4<f32>>;

struct VertexInput {
    @location(0) position: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
};

@vertex
fn vertex_main(input: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let model = models[instance_index];
    let world = model * vec4<f32>(input.position.xyz, 1.0);
    var out: VertexOutput;
    out.world_position = world.xyz;
    out.clip_position = uniforms.view_projection * world;
    return out;
}

// Stores normalized light-to-fragment distance ([0,1]) so the mesh
// shader can compare against the same normalization, matching the
// reference engine's omnidirectional shadow technique.
@fragment
fn fragment_main(in: VertexOutput) -> @location(0) f32 {
    let distance = length(in.world_position - uniforms.light_position);
    return distance / uniforms.light_range;
}
