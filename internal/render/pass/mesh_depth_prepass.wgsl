
struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) normal:   vec4<f32>,
    @location(2) tangent:  vec4<f32>,
    @location(3) uv:       vec4<f32>,
    @location(4) color:    vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(flat) material_index: u32,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct TextureTransform {
    row0: vec4<f32>,
    row1: vec4<f32>,
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

@group(0) @binding(0) var<uniform> view_proj: mat4x4<f32>;

@group(1) @binding(0) var<storage, read> models:           array<mat4x4<f32>>;
@group(1) @binding(1) var<storage, read> material_indices: array<u32>;
@group(1) @binding(2) var<storage, read> entity_ids:       array<u32>;
@group(1) @binding(3) var<storage, read> visible_indices:  array<u32>;

@group(2) @binding(0) var<storage, read> materials: array<Material>;

@group(3) @binding(0) var material_srgb_array: texture_2d_array<f32>;
@group(3) @binding(1) var material_sampler: sampler;

const NO_LAYER: u32 = 0xFFFFFFFFu;

fn apply_wrap_axis(value: f32, mode: u32) -> f32 {
    if (mode == 0u) {
        return fract(value);
    } else if (mode == 1u) {
        let cycle = value - 2.0 * floor(value * 0.5);
        return min(cycle, 2.0 - cycle);
    }
    return clamp(value, 0.0, 1.0);
}

fn apply_wrap(uv: vec2<f32>, packed: u32) -> vec2<f32> {
    let mode_u = (packed >> 16u) & 0x3u;
    let mode_v = (packed >> 18u) & 0x3u;
    return vec2<f32>(apply_wrap_axis(uv.x, mode_u), apply_wrap_axis(uv.y, mode_v));
}

fn texture_uv(uv: vec2<f32>, transform: TextureTransform) -> vec2<f32> {
    let h = vec3<f32>(uv, 1.0);
    return vec2<f32>(dot(transform.row0.xyz, h), dot(transform.row1.xyz, h));
}

@vertex
fn vertex_main(input: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let slot = visible_indices[instance_index];
    let model = models[slot];
    var out: VertexOutput;
    let world = model * input.position;
    out.clip_position = view_proj * world;
    out.material_index = material_indices[slot];
    out.uv = input.uv.xy;
    out.color = input.color;
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) {
    let mat = materials[in.material_index];
    if (mat.alpha_mode == 2u) {
        discard;
    }
    if (mat.alpha_mode == 1u) {
        var a = mat.base_color.a * in.color.a;
        if (mat.base_layer != NO_LAYER) {
            let packed = mat.base_layer;
            let layer = i32(packed & 0xFFFFu);
            let wrapped = apply_wrap(texture_uv(in.uv, mat.base_transform), packed);
            a = a * textureSampleLevel(material_srgb_array, material_sampler, wrapped, layer, 0.0).a;
        }
        if (a < mat.alpha_cutoff) {
            discard;
        }
    }
}
