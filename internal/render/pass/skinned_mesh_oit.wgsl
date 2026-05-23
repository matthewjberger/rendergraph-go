
struct ViewProjOit {
    view_proj: mat4x4<f32>,
    z_scale:   f32,
    _pad0:     f32,
    _pad1:     f32,
    _pad2:     f32,
};

@group(0) @binding(0) var<uniform> view_proj_uniform: ViewProjOit;

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
    @location(1) color:        vec4<f32>,
    @location(2) uv:           vec2<f32>,
    @location(3) @interpolate(flat) entity_id: u32,
    @location(4) view_z:       f32,
    @location(5) @interpolate(flat) instance: u32,
};

struct OitOutput {
    @location(0) accum:     vec4<f32>,
    @location(1) reveal:    f32,
    @location(2) entity_id: u32,
};

const NO_TEXTURE_LAYER: u32 = 0xFFFFFFFFu;

@vertex
fn vertex_main(input: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
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
            skinned_normal = skinned_normal + (normal_matrix * normal) * joint_weight;
        }
    }
    if (length(skinned_normal) < 0.0001) {
        skinned_normal = vec3<f32>(0.0, 0.0, 1.0);
    } else {
        skinned_normal = normalize(skinned_normal);
    }

    let clip = view_proj_uniform.view_proj * vec4<f32>(skinned_position, 1.0);
    var out: VertexOutput;
    out.clip_position = clip;
    out.world_normal = skinned_normal;
    out.color = input.color;
    out.uv = input.uv.xy;
    out.entity_id = instances[instance_index].entity_id;
    out.view_z = max(clip.w, 0.0001);
    out.instance = instance_index;
    return out;
}

fn oit_weight(view_z: f32, a: f32) -> f32 {
    let z_scale = max(view_proj_uniform.z_scale, 1.0);
    let z_ratio = view_z / z_scale;
    return a * clamp(0.03 / (1e-5 + pow(z_ratio, 4.0)), 1e-2, 3e3);
}

@fragment
fn fragment_main(in: VertexOutput) -> OitOutput {
    let instance = instances[in.instance];
    if (instance.alpha_mode != 2u) {
        discard;
    }
    var base_color = instance.base_color * in.color;
    if (instance.base_layer != NO_TEXTURE_LAYER) {
        let layer = i32(instance.base_layer & 0xFFFFu);
        base_color = base_color * textureSample(material_srgb_array, material_sampler, in.uv, layer);
    }
    let alpha = base_color.a;
    if (alpha < 0.0039) {
        discard;
    }
    let normal = normalize(in.world_normal);
    let light_dir = -normalize(skinned_uniforms.light_direction.xyz);
    let lambert = max(dot(normal, light_dir), 0.0);
    let lit = skinned_uniforms.ambient_color.rgb + skinned_uniforms.light_color.rgb * lambert;
    let albedo = base_color.rgb * lit;

    let w = oit_weight(in.view_z, alpha);
    var out: OitOutput;
    out.accum = vec4<f32>(albedo * alpha, alpha) * w;
    out.reveal = alpha;
    out.entity_id = in.entity_id;
    return out;
}
