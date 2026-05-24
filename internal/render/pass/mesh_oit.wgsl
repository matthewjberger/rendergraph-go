
struct ViewProj {
    view_proj:       mat4x4<f32>,
    camera_position: vec4<f32>,
    camera_z_far:    f32,
    oit_z_scale:     f32,
    _pad0:           f32,
    _pad1:           f32,
};

struct DirectionalLight {
    direction: vec4<f32>,
    color:     vec4<f32>,
    ambient:   vec4<f32>,
};

struct TextureTransform {
    row0: vec4<f32>,
    row1: vec4<f32>,
};

struct Material {
    base_color:                vec4<f32>,
    emissive_factor:           vec3<f32>,
    alpha_mode:                u32,
    base_layer:                u32,
    emissive_layer:            u32,
    normal_layer:              u32,
    metallic_roughness_layer:  u32,
    occlusion_layer:           u32,
    normal_scale:              f32,
    occlusion_strength:        f32,
    metallic_factor:           f32,
    roughness_factor:          f32,
    alpha_cutoff:              f32,
    unlit:                     u32,
    ior:                       f32,
    emissive_strength:         f32,
    _pad0:                     f32,
    _pad1:                     f32,
    _pad2:                     f32,

    normal_map_flags:     u32,
    specular_factor:      f32,
    specular_layer:       u32,
    specular_color_layer: u32,

    specular_color_factor: vec3<f32>,
    transmission_factor:   f32,

    transmission_layer:   u32,
    thickness:            f32,
    thickness_layer:      u32,
    attenuation_distance: f32,

    attenuation_color: vec3<f32>,
    dispersion:        f32,

    anisotropy_strength:     f32,
    anisotropy_rotation_cos: f32,
    anisotropy_rotation_sin: f32,
    anisotropy_layer:        u32,

    clearcoat_factor:          f32,
    clearcoat_roughness_factor: f32,
    clearcoat_normal_scale:    f32,
    clearcoat_layer:           u32,

    clearcoat_roughness_layer: u32,
    clearcoat_normal_layer:    u32,
    sheen_color_layer:         u32,
    sheen_roughness_layer:     u32,

    sheen_color_factor:     vec3<f32>,
    sheen_roughness_factor: f32,

    iridescence_factor:        f32,
    iridescence_ior:           f32,
    iridescence_thickness_min: f32,
    iridescence_thickness_max: f32,

    iridescence_layer:           u32,
    iridescence_thickness_layer: u32,
    diffuse_transmission_factor: f32,
    diffuse_transmission_color_layer: u32,

    diffuse_transmission_color_factor: vec3<f32>,
    _pad_dt:                           f32,

    base_transform:               TextureTransform,
    normal_transform:             TextureTransform,
    metallic_roughness_transform: TextureTransform,
    occlusion_transform:          TextureTransform,
    emissive_transform:           TextureTransform,
};

@group(0) @binding(0) var<uniform>       view_proj_uniform: ViewProj;
@group(0) @binding(1) var<uniform>       directional:       DirectionalLight;
@group(0) @binding(2) var<storage, read> materials:         array<Material>;
@group(0) @binding(3) var                material_srgb_array: texture_2d_array<f32>;
@group(0) @binding(4) var                material_sampler:    sampler;

@group(1) @binding(0) var<storage, read> models:           array<mat4x4<f32>>;
@group(1) @binding(1) var<storage, read> material_indices: array<u32>;
@group(1) @binding(2) var<storage, read> entity_ids:       array<u32>;
@group(1) @binding(3) var<storage, read> visible_indices:  array<u32>;

struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) normal:   vec4<f32>,
    @location(2) tangent:  vec4<f32>,
    @location(3) uv:       vec4<f32>,
    @location(4) color:    vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal:   vec3<f32>,
    @location(2) uv:             vec2<f32>,
    @location(3) color:          vec4<f32>,
    @location(4) @interpolate(flat) material_index: u32,
    @location(5) @interpolate(flat) entity_id:      u32,
    @location(6) view_z:         f32,
};

struct OitOutput {
    @location(0) accum:     vec4<f32>,
    @location(1) reveal:    f32,
    @location(2) entity_id: u32,
};

const NO_TEXTURE_LAYER: u32 = 0xFFFFFFFFu;

@vertex
fn vertex_main(input: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let slot = visible_indices[instance_index];
    let model = models[slot];
    let world_position = model * vec4<f32>(input.position.xyz, 1.0);
    let world_normal = normalize((model * vec4<f32>(input.normal.xyz, 0.0)).xyz);
    let clip = view_proj_uniform.view_proj * world_position;
    var out: VertexOutput;
    out.clip_position = clip;
    out.world_position = world_position.xyz;
    out.world_normal = world_normal;
    out.uv = input.uv.xy;
    out.color = input.color;
    out.material_index = material_indices[slot];
    out.entity_id = entity_ids[slot];

    out.view_z = max(clip.w, 0.0001);
    return out;
}

fn oit_weight(view_z: f32, a: f32) -> f32 {
    let z_scale = max(view_proj_uniform.oit_z_scale, 1.0);
    let z_ratio = view_z / z_scale;
    return a * clamp(0.03 / (1e-5 + pow(z_ratio, 4.0)), 1e-2, 3e3);
}

@fragment
fn fragment_main(in: VertexOutput) -> OitOutput {
    let mat = materials[in.material_index];
    if (mat.alpha_mode != 2u) {
        discard;
    }

    var base_color = mat.base_color * in.color;
    if (mat.base_layer != NO_TEXTURE_LAYER) {
        let layer = i32(mat.base_layer & 0xFFFFu);
        base_color = base_color * textureSampleLevel(material_srgb_array, material_sampler, in.uv, layer, 0.0);
    }

    var albedo = base_color.rgb;
    let alpha = base_color.a;
    if (alpha < 0.0039) {
        discard;
    }

    if (mat.unlit == 0u) {
        let normal = normalize(in.world_normal);
        let light_dir = -normalize(directional.direction.xyz);
        let lambert = max(dot(normal, light_dir), 0.0);
        albedo = albedo * (directional.ambient.rgb + directional.color.rgb * lambert);
    }
    albedo = albedo + mat.emissive_factor * mat.emissive_strength;

    let w = oit_weight(in.view_z, alpha);

    var out: OitOutput;
    out.accum = vec4<f32>(albedo * alpha, alpha) * w;
    out.reveal = alpha;
    out.entity_id = in.entity_id;
    return out;
}
