// Skinned mesh shader. Vertex stage blends position + normal by
// up to 4 joint matrices per vertex (matching the reference
// engine's per-influence accumulation form); fragment stage
// samples the material's base color texture from the shared
// material texture array and shades with a basic Lambert +
// ambient model. Output lands in scene_color so the postprocess
// chain (SSAO, bloom, tonemap) still applies.

struct ViewProj {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> view_proj_uniform: ViewProj;

struct SkinnedUniforms {
    light_direction: vec4<f32>,
    light_color:     vec4<f32>,
    ambient_color:   vec4<f32>,
};

@group(1) @binding(0) var<uniform> skinned_uniforms: SkinnedUniforms;
@group(1) @binding(1) var material_srgb_array: texture_2d_array<f32>;
@group(1) @binding(2) var material_sampler:    sampler;

struct SkinnedInstance {
    base_color:   vec4<f32>,
    entity_id:    u32,
    joint_offset: u32,
    base_layer:   u32,
    alpha_mode:   u32,
};

@group(2) @binding(0) var<storage, read> joint_matrices: array<mat4x4<f32>>;
@group(2) @binding(1) var<storage, read> instances:      array<SkinnedInstance>;

struct VertexInput {
    @location(0) position:      vec4<f32>,
    @location(1) normal:        vec4<f32>,
    @location(2) tangent:       vec4<f32>,
    @location(3) uv:            vec4<f32>,
    @location(4) color:         vec4<f32>,
    @location(5) joint_indices: vec4<u32>,
    @location(6) joint_weights: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) entity_id: u32,
    @location(4) @interpolate(flat) instance: u32,
};

struct FragmentOutput {
    @location(0) color:       vec4<f32>,
    @location(1) entity_id:   u32,
    @location(2) view_normal: vec4<f32>,
};

const NO_TEXTURE_LAYER: u32 = 0xFFFFFFFFu;

@vertex
fn vertex_main(input: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    // Per glTF spec: when a node has a skin, the node's own
    // transform is ignored -- joint matrices already encode every
    // bind-pose-to-world transformation. Per-influence
    // accumulation form (weight * (M * pos) summed over 4) is
    // mathematically equivalent to (sum(weight * M)) * pos but
    // plays nicer with non-affine joint matrices.
    let position = vec4<f32>(input.position.xyz, 1.0);
    let normal = input.normal.xyz;
    let joint_offset = instances[instance_index].joint_offset;
    var skinned_position = vec3<f32>(0.0, 0.0, 0.0);
    var skinned_normal = vec3<f32>(0.0, 0.0, 0.0);
    for (var index = 0u; index < 4u; index = index + 1u) {
        let joint_index = input.joint_indices[index];
        let joint_weight = input.joint_weights[index];
        if (joint_weight > 0.0) {
            let joint_matrix = joint_matrices[joint_offset + joint_index];
            let transformed_pos = joint_matrix * position;
            skinned_position = skinned_position + transformed_pos.xyz * joint_weight;
            let normal_matrix = mat3x3<f32>(joint_matrix[0].xyz, joint_matrix[1].xyz, joint_matrix[2].xyz);
            let transformed_normal = normal_matrix * normal;
            skinned_normal = skinned_normal + transformed_normal * joint_weight;
        }
    }
    if (length(skinned_normal) < 0.0001) {
        skinned_normal = vec3<f32>(0.0, 0.0, 1.0);
    } else {
        skinned_normal = normalize(skinned_normal);
    }

    var out: VertexOutput;
    out.clip_position = view_proj_uniform.view_proj * vec4<f32>(skinned_position, 1.0);
    out.world_normal = skinned_normal;
    out.color = input.color;
    out.uv = input.uv.xy;
    out.entity_id = instances[instance_index].entity_id;
    out.instance = instance_index;
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> FragmentOutput {
    let instance = instances[in.instance];
    if (instance.alpha_mode == 2u) {
        discard;
    }
    var base_color = instance.base_color;
    if (instance.base_layer != NO_TEXTURE_LAYER) {
        let layer = i32(instance.base_layer & 0xFFFFu);
        let sampled = textureSample(material_srgb_array, material_sampler, in.uv, layer);
        base_color = base_color * sampled;
    }
    let albedo = base_color.rgb * in.color.rgb;
    let normal = normalize(in.world_normal);
    let light_dir = -normalize(skinned_uniforms.light_direction.xyz);
    let lambert = max(dot(normal, light_dir), 0.0);
    let lit = skinned_uniforms.ambient_color.rgb + skinned_uniforms.light_color.rgb * lambert;
    var out: FragmentOutput;
    out.color = vec4<f32>(albedo * lit, base_color.a * in.color.a);
    out.entity_id = in.entity_id;
    out.view_normal = vec4<f32>(normal, 1.0);
    return out;
}
