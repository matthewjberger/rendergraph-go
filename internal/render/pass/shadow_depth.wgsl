struct Uniforms {
    light_view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct MorphDisplacement {
    position: vec3<f32>,
    _pad0: f32,
    normal: vec3<f32>,
    _pad1: f32,
    tangent: vec3<f32>,
    _pad2: f32,
};

struct MorphInstance {
    weights: array<f32, 8>,
    target_count: u32,
    vertex_count: u32,
    _mpad0: u32,
    _mpad1: u32,
};

@group(1) @binding(0) var<storage, read> models: array<mat4x4<f32>>;
@group(1) @binding(1) var<storage, read> morph_displacements: array<MorphDisplacement>;
@group(1) @binding(2) var<storage, read> morph_instances:     array<MorphInstance>;

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
    @location(0) position: vec4<f32>,
};

// Vertex-only depth pass; back faces are culled by the pipeline (cull_mode
// Back), matching nightshade's shadow_depth.wgsl.
@vertex
fn vertex_main(input: VertexInput) -> @builtin(position) vec4<f32> {
    var local_position = input.position.xyz;
    var morph = morph_instances[input.instance_index];
    if (morph.target_count > 0u) {
        for (var t = 0u; t < morph.target_count; t = t + 1u) {
            let w = morph.weights[t];
            if (abs(w) > 0.0001) {
                local_position = local_position + morph_displacements[t * morph.vertex_count + input.vertex_index].position * w;
            }
        }
    }
    let world_pos = models[input.instance_index] * vec4<f32>(local_position, 1.0);
    return uniforms.light_view_proj * world_pos;
}
