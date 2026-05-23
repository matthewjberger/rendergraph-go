// Depth-only prepass for the static mesh path. Reads the cull
// shader's visible_indices, transforms position, writes depth so
// the main pass runs with DepthCompare=LessEqual + DepthWrite=OFF
// for early-Z trim. Blend-mode (alpha_mode 2) fragments discard so
// their depth doesn't occlude the OIT pass.

struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) normal:   vec4<f32>,
    @location(2) tangent:  vec4<f32>,
    @location(3) uv:       vec4<f32>,
    @location(4) color:    vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(flat) alpha_mode: u32,
};

struct Material {
    base_color:               vec4<f32>,
    emissive_factor:          vec3<f32>,
    alpha_mode:               u32,
    base_layer:               u32,
    emissive_layer:           u32,
    normal_layer:             u32,
    metallic_roughness_layer: u32,
    occlusion_layer:          u32,
    normal_scale:             f32,
    occlusion_strength:       f32,
    metallic_factor:          f32,
    roughness_factor:         f32,
    alpha_cutoff:             f32,
    unlit:                    u32,
    ior:                      f32,
    emissive_strength:        f32,
    _pad1a:                   f32,
    _pad1b:                   f32,
    _pad1c:                   f32,
};

@group(0) @binding(0) var<uniform> view_proj: mat4x4<f32>;

@group(1) @binding(0) var<storage, read> models:           array<mat4x4<f32>>;
@group(1) @binding(1) var<storage, read> material_indices: array<u32>;
@group(1) @binding(2) var<storage, read> entity_ids:       array<u32>;
@group(1) @binding(3) var<storage, read> visible_indices:  array<u32>;

@group(2) @binding(0) var<storage, read> materials: array<Material>;

@vertex
fn vertex_main(input: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let slot = visible_indices[instance_index];
    let model = models[slot];
    let material_index = material_indices[slot];
    let mat = materials[material_index];
    var out: VertexOutput;
    let world = model * input.position;
    out.clip_position = view_proj * world;
    out.alpha_mode = mat.alpha_mode;
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) {
    if (in.alpha_mode == 2u) {
        discard;
    }
}
