
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
    @location(0) color: vec4<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) world_normal: vec3<f32>,
    @location(3) world_tangent: vec4<f32>,
    @location(4) uv: vec4<f32>,
    @location(5) @interpolate(flat) entity_id: u32,
    @location(6) @interpolate(flat) material_index: u32,
    @location(7) view_normal: vec3<f32>,
    @location(8) @interpolate(flat) world_scale_factor: f32,
};

struct Light {
    position:    vec4<f32>,
    direction:   vec4<f32>,
    color:       vec4<f32>,
    light_type:  u32,
    range:       f32,
    inner_cone:  f32,
    outer_cone:  f32,
    shadow_index: i32,
    light_size:  f32,
    cookie_layer: u32,
    _padding:    f32,
};

struct LightGrid {
    offset: u32,
    count:  u32,
};

struct ClusterUniforms {
    inverse_projection: mat4x4<f32>,
    screen_size: vec2<f32>,
    z_near: f32,
    z_far: f32,
    cluster_count: vec4<u32>,
    tile_size: vec2<f32>,
    num_lights: u32,
    num_directional_lights: u32,
    camera_position: vec4<f32>,
    flat_color: vec4<f32>,
    global_unlit: f32,
    flat_pad0: f32,
    flat_pad1: f32,
    flat_pad2: f32,
};

struct TextureTransform {
    row0: vec4<f32>,
    row1: vec4<f32>,
};

struct Material {
    base_color:      vec4<f32>,
    emissive_factor: vec3<f32>,
    alpha_mode:      u32,

    base_layer:               u32,
    emissive_layer:           u32,
    normal_layer:             u32,
    metallic_roughness_layer: u32,

    occlusion_layer:    u32,
    normal_scale:       f32,
    occlusion_strength: f32,
    metallic_factor:    f32,

    roughness_factor: f32,
    alpha_cutoff:     f32,
    unlit:            u32,
    ior:              f32,

    emissive_strength: f32,
    _pad1a:            f32,
    _pad1b:            f32,
    _pad1c:            f32,

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
    blend_opaque_alpha_threshold:      f32,

    base_transform:               TextureTransform,
    normal_transform:             TextureTransform,
    metallic_roughness_transform: TextureTransform,
    occlusion_transform:          TextureTransform,
    emissive_transform:           TextureTransform,

    transmission_transform:              TextureTransform,
    thickness_transform:                 TextureTransform,
    specular_transform:                  TextureTransform,
    specular_color_transform:            TextureTransform,
    clearcoat_transform:                 TextureTransform,
    clearcoat_roughness_transform:       TextureTransform,
    clearcoat_normal_transform:          TextureTransform,
    sheen_color_transform:               TextureTransform,
    sheen_roughness_transform:           TextureTransform,
    iridescence_transform:               TextureTransform,
    iridescence_thickness_transform:     TextureTransform,
    anisotropy_transform:                TextureTransform,
    diffuse_transmission_transform:      TextureTransform,
    diffuse_transmission_color_transform: TextureTransform,
};

@group(0) @binding(0) var<uniform> view_proj: mat4x4<f32>;

@group(1) @binding(0) var<storage, read> lights:        array<Light>;
@group(1) @binding(1) var<storage, read> light_grid:    array<LightGrid>;
@group(1) @binding(2) var<storage, read> light_indices: array<u32>;
@group(1) @binding(3) var<uniform>       cluster_uniforms: ClusterUniforms;
@group(1) @binding(4) var<uniform>       view_matrix:   mat4x4<f32>;
@group(1) @binding(5) var material_srgb_array:   texture_2d_array<f32>;
@group(1) @binding(6) var material_linear_array: texture_2d_array<f32>;
@group(1) @binding(7) var material_sampler:      sampler;
@group(1) @binding(8) var<storage, read> materials: array<Material>;

struct ShadowUniforms {
    cascade_view_projections: array<mat4x4<f32>, 4>,
    cascade_splits: vec4<f32>,
    light_direction: vec4<f32>,
    cascade_texel_world: vec4<f32>,
    light_size: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(1) @binding(9) var<uniform>       shadow_uniforms: ShadowUniforms;
@group(1) @binding(10) var shadow_map:           texture_depth_2d_array;
@group(1) @binding(11) var shadow_sampler:       sampler_comparison;

struct SpotShadowEntry {
    view_proj: mat4x4<f32>,
    atlas_offset: vec2<f32>,
    atlas_scale: vec2<f32>,
    bias: f32,
    enabled: u32,
    light_size: f32,
    _pad1: f32,
};

struct SpotShadowUniforms {
    entries: array<SpotShadowEntry, 4>,
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(1) @binding(12) var<uniform> spot_shadow_uniforms: SpotShadowUniforms;
@group(1) @binding(13) var spot_shadow_atlas: texture_depth_2d;

struct PointShadowEntry {
    position: vec3<f32>,
    range: f32,
    bias: f32,
    light_size: f32,
    _pad1: f32,
    _pad2: f32,
};

struct PointShadowUniforms {
    entries: array<PointShadowEntry, 4>,
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(1) @binding(14) var<uniform> point_shadow_uniforms: PointShadowUniforms;
@group(1) @binding(15) var point_shadow_cube_array: texture_cube_array<f32>;
@group(1) @binding(16) var point_shadow_sampler: sampler;
@group(1) @binding(17) var shadow_raw_sampler: sampler;

var<private> POISSON_16: array<vec2<f32>, 16> = array<vec2<f32>, 16>(
    vec2<f32>(-0.94201624, -0.39906216),
    vec2<f32>(0.94558609, -0.76890725),
    vec2<f32>(-0.094184101, -0.92938870),
    vec2<f32>(0.34495938, 0.29387760),
    vec2<f32>(-0.91588581, 0.45771432),
    vec2<f32>(-0.81544232, -0.87912464),
    vec2<f32>(-0.38277543, 0.27676845),
    vec2<f32>(0.97484398, 0.75648379),
    vec2<f32>(0.44323325, -0.97511554),
    vec2<f32>(0.53742981, -0.47373420),
    vec2<f32>(-0.26496911, -0.41893023),
    vec2<f32>(0.79197514, 0.19090188),
    vec2<f32>(-0.24188840, 0.99706507),
    vec2<f32>(-0.81409955, 0.91437590),
    vec2<f32>(0.19984126, 0.78641367),
    vec2<f32>(0.14383161, -0.14100790),
);

var<private> POISSON_8: array<vec2<f32>, 8> = array<vec2<f32>, 8>(
    vec2<f32>(-0.7071, 0.7071),
    vec2<f32>(0.0, -0.875),
    vec2<f32>(0.5303, 0.5303),
    vec2<f32>(-0.625, 0.0),
    vec2<f32>(0.3536, -0.3536),
    vec2<f32>(0.0, 0.875),
    vec2<f32>(-0.5303, -0.5303),
    vec2<f32>(0.625, 0.0),
);

struct SkinnedInstance {
    base_color:           vec4<f32>,
    entity_id:            u32,
    joint_offset:         u32,
    base_layer:           u32,
    alpha_mode:           u32,
    transmission_factor:  f32,
    ior:                  f32,
    roughness_factor:     f32,
    metallic_factor:      f32,
    dispersion:           f32,
    thickness:            f32,
    attenuation_distance: f32,
    material_index:       u32,
    attenuation_color:    vec3<f32>,
    world_scale_factor:   f32,
};

@group(2) @binding(0) var<storage, read> joint_matrices: array<mat4x4<f32>>;
@group(2) @binding(1) var<storage, read> instances:      array<SkinnedInstance>;

@group(3) @binding(0) var irradiance_map:  texture_cube<f32>;
@group(3) @binding(1) var prefiltered_env: texture_cube<f32>;
@group(3) @binding(2) var brdf_lut:        texture_2d<f32>;
@group(3) @binding(3) var ibl_sampler:     sampler;
@group(3) @binding(4) var transmission_color_texture: texture_2d<f32>;
@group(3) @binding(5) var transmission_color_sampler: sampler;

const NO_LAYER: u32 = 0xFFFFFFFFu;
const COOKIE_LAYER_NONE: u32 = 0xFFFFFFFFu;
const PI: f32 = 3.14159265359;
const MAX_LIGHTS_PER_CLUSTER: u32 = 256u;
const LIGHT_TYPE_DIRECTIONAL: u32 = 0u;
const LIGHT_TYPE_POINT: u32 = 1u;
const LIGHT_TYPE_SPOT: u32 = 2u;
const MAX_REFLECTION_LOD: f32 = 4.0;
const NORMAL_MAP_FLIP_Y: u32 = 1u;
const NORMAL_MAP_TWO_COMPONENT: u32 = 2u;

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq < 0.0001) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return v / sqrt(len_sq);
}

fn get_normal(
    world_normal: vec3<f32>,
    world_tangent: vec4<f32>,
    normal_sample: vec3<f32>,
    has_normal_texture: bool,
    normal_scale: f32,
    flags: u32,
) -> vec3<f32> {
    let n = safe_normalize(world_normal);
    if (!has_normal_texture) {
        return n;
    }

    var sample_xy = normal_sample.xy * 2.0 - 1.0;
    if ((flags & NORMAL_MAP_FLIP_Y) != 0u) {
        sample_xy.y = -sample_xy.y;
    }
    var sample_z: f32;
    if ((flags & NORMAL_MAP_TWO_COMPONENT) != 0u) {
        sample_z = sqrt(max(1.0 - dot(sample_xy, sample_xy), 0.0));
    } else {
        sample_z = normal_sample.z * 2.0 - 1.0;
    }

    let tangent_normal = vec3<f32>(sample_xy * normal_scale, sample_z);
    var t = safe_normalize(world_tangent.xyz);
    t = safe_normalize(t - n * dot(n, t));
    let b = cross(n, t) * world_tangent.w;
    let tbn = mat3x3<f32>(t, b, n);
    return safe_normalize(tbn * tangent_normal);
}

struct FragmentOutput {
    @location(0) color:        vec4<f32>,
    @location(1) entity_id:    u32,
    @location(2) view_normal:  vec4<f32>,
};

fn apply_wrap_axis(uv: f32, mode: u32) -> f32 {
    if (mode == 2u) {
        return clamp(uv, 0.0, 1.0);
    } else if (mode == 1u) {
        let folded = fract(uv * 0.5) * 2.0;
        if (folded > 1.0) {
            return 2.0 - folded;
        }
        return folded;
    }
    return fract(uv);
}

fn apply_wrap(uv: vec2<f32>, packed: u32) -> vec2<f32> {
    let mode_u = (packed >> 16u) & 0x3u;
    let mode_v = (packed >> 18u) & 0x3u;
    return vec2<f32>(apply_wrap_axis(uv.x, mode_u), apply_wrap_axis(uv.y, mode_v));
}

fn sample_srgb_layer(packed: u32, uv: vec2<f32>) -> vec4<f32> {
    let layer = i32(packed & 0xFFFFu);
    let wrapped = apply_wrap(uv, packed);
    return textureSampleLevel(material_srgb_array, material_sampler, wrapped, layer, 0.0);
}

fn sample_linear_layer(packed: u32, uv: vec2<f32>) -> vec4<f32> {
    let layer = i32(packed & 0xFFFFu);
    let wrapped = apply_wrap(uv, packed);
    return textureSampleLevel(material_linear_array, material_sampler, wrapped, layer, 0.0);
}

fn distribution_ggx(n: vec3<f32>, h: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h = max(dot(n, h), 0.0);
    let n_dot_h2 = n_dot_h * n_dot_h;
    let denom = n_dot_h2 * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denom * denom, 0.0001);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

fn geometry_smith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_l = max(dot(n, l), 0.0);
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn apply_ior_to_roughness(roughness: f32, ior: f32) -> f32 {
    return roughness * clamp(ior * 2.0 - 2.0, 0.0, 1.0);
}

fn apply_volume_attenuation(radiance: vec3<f32>, transmission_distance: f32, attenuation_color: vec3<f32>, attenuation_distance: f32) -> vec3<f32> {
    if (attenuation_distance <= 0.0) {
        return radiance;
    }
    let attenuation_coefficient = -log(max(attenuation_color, vec3<f32>(0.0001))) / attenuation_distance;
    let transmittance = exp(-attenuation_coefficient * transmission_distance);
    return transmittance * radiance;
}

fn get_transmission_sample(reflection: vec3<f32>, roughness: f32, ior: f32) -> vec3<f32> {
    let transmission_roughness = apply_ior_to_roughness(roughness, ior);
    return textureSampleLevel(prefiltered_env, ibl_sampler, reflection, transmission_roughness * MAX_REFLECTION_LOD).rgb;
}

fn get_screen_space_transmission(world_pos: vec3<f32>, refracted_dir: vec3<f32>, thickness: f32, ior: f32, roughness: f32) -> vec3<f32> {
    let exit_world = world_pos + refracted_dir * max(thickness, 0.001);
    let exit_clip = view_proj * vec4<f32>(exit_world, 1.0);
    let ibl_sample = get_transmission_sample(refracted_dir, roughness, ior);
    if (exit_clip.w <= 0.0001) {
        return ibl_sample;
    }
    let exit_ndc = exit_clip.xyz / exit_clip.w;
    let uv = vec2<f32>(exit_ndc.x * 0.5 + 0.5, 0.5 - exit_ndc.y * 0.5);
    let clamped_uv = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let scene_sample = textureSampleLevel(transmission_color_texture, transmission_color_sampler, clamped_uv, 0.0).rgb;
    let edge_dist = min(min(uv.x, uv.y), min(1.0 - uv.x, 1.0 - uv.y));
    let blend = clamp(edge_dist * 8.0, 0.0, 1.0);
    return mix(ibl_sample, scene_sample, blend);
}

fn refract_safe(view: vec3<f32>, normal: vec3<f32>, eta: f32) -> vec3<f32> {
    let refraction = refract(-view, normal, eta);
    let len_sq = dot(refraction, refraction);
    if (len_sq > 0.0001) {
        return refraction / sqrt(len_sq);
    }
    return -view;
}

fn get_ibl_volume_refraction(
    normal: vec3<f32>,
    view: vec3<f32>,
    world_pos: vec3<f32>,
    roughness: f32,
    base_color: vec3<f32>,
    f0: vec3<f32>,
    ior: f32,
    dispersion: f32,
    thickness: f32,
    attenuation_color: vec3<f32>,
    attenuation_distance: f32,
    model_scale: vec3<f32>,
) -> vec3<f32> {
    let transmission_distance = thickness * length(model_scale);
    var transmitted_light: vec3<f32>;
    if (dispersion > 0.0) {
        let half_spread = (ior - 1.0) * 0.025 * dispersion;
        let ior_r = ior - half_spread;
        let ior_b = ior + half_spread;
        let dir_r = refract_safe(view, normal, 1.0 / ior_r);
        let dir_g = refract_safe(view, normal, 1.0 / ior);
        let dir_b = refract_safe(view, normal, 1.0 / ior_b);
        let r = get_screen_space_transmission(world_pos, dir_r, transmission_distance, ior_r, roughness).r;
        let g = get_screen_space_transmission(world_pos, dir_g, transmission_distance, ior, roughness).g;
        let b = get_screen_space_transmission(world_pos, dir_b, transmission_distance, ior_b, roughness).b;
        transmitted_light = vec3<f32>(r, g, b);
    } else {
        let dir = refract_safe(view, normal, 1.0 / ior);
        transmitted_light = get_screen_space_transmission(world_pos, dir, transmission_distance, ior, roughness);
    }
    let attenuated_color = apply_volume_attenuation(transmitted_light, transmission_distance, attenuation_color, attenuation_distance);
    let n_dot_v = clamp(dot(normal, view), 0.001, 1.0);
    let brdf = textureSampleLevel(brdf_lut, ibl_sampler, vec2<f32>(n_dot_v, roughness), 0.0).rg;
    let specular_color = f0 * brdf.x + brdf.y;
    return (vec3<f32>(1.0) - specular_color) * attenuated_color * base_color;
}

fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    let invR = vec3<f32>(1.0 - roughness);
    return f0 + (max(invR, f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn select_cascade(view_depth: f32) -> i32 {
    for (var cascade: i32 = 0; cascade < 4; cascade = cascade + 1) {
        if view_depth < shadow_uniforms.cascade_splits[cascade] {
            return cascade;
        }
    }
    return 3;
}

fn sample_cascade(world_pos: vec3<f32>, world_normal: vec3<f32>, cascade: i32) -> f32 {
    let direction_to_light = -normalize(shadow_uniforms.light_direction.xyz);
    let texel_world = shadow_uniforms.cascade_texel_world[cascade];

    let slope = 1.0 - max(dot(world_normal, direction_to_light), 0.0);
    let normal_offset = world_normal * texel_world * (1.8 + slope * 3.0);
    let depth_offset = direction_to_light * 0.008;
    let offset_pos = world_pos + normal_offset + depth_offset;
    let shadow_clip = shadow_uniforms.cascade_view_projections[cascade] * vec4<f32>(offset_pos, 1.0);
    let shadow_ndc = shadow_clip.xyz / shadow_clip.w;
    let shadow_uv = vec2<f32>(shadow_ndc.x * 0.5 + 0.5, -shadow_ndc.y * 0.5 + 0.5);
    if (shadow_uv.x < 0.0 || shadow_uv.x > 1.0 || shadow_uv.y < 0.0 || shadow_uv.y > 1.0) {
        return 1.0;
    }
    if (shadow_ndc.z < 0.0 || shadow_ndc.z > 1.0) {
        return 1.0;
    }
    let depth = shadow_ndc.z;
    let texel = 1.0 / 2048.0;
    let light_size = max(shadow_uniforms.light_size, 0.001);

    let search_bias = 0.001;
    let search_radius = 4.0 * texel * light_size;
    let dims_cascade = vec2<f32>(textureDimensions(shadow_map));
    var blocker_sum = 0.0;
    var blocker_count = 0.0;
    for (var index: i32 = 0; index < 8; index = index + 1) {
        let offset = POISSON_8[index] * search_radius;
        let texel_coords = vec2<i32>(clamp((shadow_uv + offset) * dims_cascade, vec2<f32>(0.0), dims_cascade - vec2<f32>(1.0)));
        let sampled = textureLoad(shadow_map, texel_coords, cascade, 0);
        if (sampled < depth - search_bias) {
            blocker_sum = blocker_sum + sampled;
            blocker_count = blocker_count + 1.0;
        }
    }
    if (blocker_count == 0.0) {
        return 1.0;
    }
    let blocker_depth = blocker_sum / blocker_count;
    let penumbra = (depth - blocker_depth) / max(blocker_depth, 0.0001);
    let filter_radius = clamp(penumbra * 16.0 * texel * light_size, texel, 24.0 * texel);

    var sum = 0.0;
    for (var index: i32 = 0; index < 16; index = index + 1) {
        let offset = POISSON_16[index] * filter_radius;
        sum = sum + textureSampleCompareLevel(shadow_map, shadow_sampler, shadow_uv + offset, cascade, depth);
    }
    return sum / 16.0;
}

fn sample_directional_shadow(world_pos: vec3<f32>, world_normal: vec3<f32>) -> f32 {
    let view_pos = view_matrix * vec4<f32>(world_pos, 1.0);
    let view_depth = -view_pos.z;
    let cascade = select_cascade(view_depth);
    let factor = sample_cascade(world_pos, world_normal, cascade);
    if (cascade < 3) {
        let cascade_end = shadow_uniforms.cascade_splits[cascade];

        let blend_range = cascade_end * 0.25;
        let blend_start = cascade_end - blend_range;
        if (view_depth > blend_start) {
            let next = sample_cascade(world_pos, world_normal, cascade + 1);
            let t = clamp((view_depth - blend_start) / blend_range, 0.0, 1.0);
            return mix(factor, next, t);
        }
    }
    return factor;
}

fn range_attenuation(range: f32, distance: f32) -> f32 {
    if (range <= 0.0) {
        return 1.0;
    }
    let clamped_distance = max(distance, 0.01);
    return max(min(1.0 - pow(distance / range, 4.0), 1.0), 0.0) / (clamped_distance * clamped_distance);
}

fn spot_attenuation(point_to_light: vec3<f32>, spot_direction: vec3<f32>, outer_cone_cos: f32, inner_cone_cos: f32) -> f32 {
    let actual_cos = dot(normalize(spot_direction), normalize(-point_to_light));
    if (actual_cos > outer_cone_cos) {
        if (actual_cos < inner_cone_cos) {
            return smoothstep(outer_cone_cos, inner_cone_cos, actual_cos);
        }
        return 1.0;
    }
    return 0.0;
}

fn sample_spot_cookie(light: Light, world_pos: vec3<f32>) -> vec3<f32> {
    if (light.cookie_layer == COOKIE_LAYER_NONE || light.shadow_index < 0) {
        return vec3<f32>(1.0);
    }
    if (u32(light.shadow_index) >= spot_shadow_uniforms.count) {
        return vec3<f32>(1.0);
    }
    let entry = spot_shadow_uniforms.entries[light.shadow_index];
    let clip = entry.view_proj * vec4<f32>(world_pos, 1.0);
    if (clip.w <= 0.0) {
        return vec3<f32>(0.0);
    }
    let ndc = clip.xyz / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return vec3<f32>(0.0);
    }
    let layer = i32(light.cookie_layer & 0xFFFFu);
    return textureSampleLevel(material_srgb_array, material_sampler, uv, layer, 0.0).rgb;
}

fn light_radiance(light: Light, point_to_light: vec3<f32>) -> vec3<f32> {
    var range_atten = 1.0;
    var spot_atten = 1.0;
    var cookie = vec3<f32>(1.0);
    if (light.light_type != LIGHT_TYPE_DIRECTIONAL) {
        range_atten = range_attenuation(light.range, length(point_to_light));
    }
    if (light.light_type == LIGHT_TYPE_SPOT) {
        spot_atten = spot_attenuation(point_to_light, light.direction.xyz, light.outer_cone, light.inner_cone);
        cookie = sample_spot_cookie(light, light.position.xyz - point_to_light);
    }
    return range_atten * spot_atten * light.color.rgb * cookie;
}

fn get_cluster_index(frag_coord: vec2<f32>, view_depth: f32) -> u32 {
    let tile = vec2<u32>(
        u32(frag_coord.x / cluster_uniforms.tile_size.x),
        u32(frag_coord.y / cluster_uniforms.tile_size.y)
    );
    let log_ratio = log(cluster_uniforms.z_far / cluster_uniforms.z_near);
    let safe_depth = max(view_depth, cluster_uniforms.z_near);
    let slice = u32(log(safe_depth / cluster_uniforms.z_near) / log_ratio * f32(cluster_uniforms.cluster_count.z));
    let clamped_slice = clamp(slice, 0u, cluster_uniforms.cluster_count.z - 1u);
    let clamped_tile_x = clamp(tile.x, 0u, cluster_uniforms.cluster_count.x - 1u);
    let clamped_tile_y = clamp(tile.y, 0u, cluster_uniforms.cluster_count.y - 1u);
    return clamped_tile_x +
           clamped_tile_y * cluster_uniforms.cluster_count.x +
           clamped_slice * cluster_uniforms.cluster_count.x * cluster_uniforms.cluster_count.y;
}

fn sample_spot_shadow(world_pos: vec3<f32>, world_normal: vec3<f32>, shadow_index: i32) -> f32 {
    if (shadow_index < 0 || u32(shadow_index) >= spot_shadow_uniforms.count) {
        return 1.0;
    }
    let entry = spot_shadow_uniforms.entries[shadow_index];
    if (entry.enabled == 0u) {
        return 1.0;
    }
    let bias_offset = world_normal * 0.02;
    let shadow_clip = entry.view_proj * vec4<f32>(world_pos + bias_offset, 1.0);
    if (shadow_clip.w <= 0.0) {
        return 1.0;
    }
    let shadow_ndc = shadow_clip.xyz / shadow_clip.w;
    var slot_uv = vec2<f32>(shadow_ndc.x * 0.5 + 0.5, -shadow_ndc.y * 0.5 + 0.5);
    if (slot_uv.x < 0.0 || slot_uv.x > 1.0 || slot_uv.y < 0.0 || slot_uv.y > 1.0) {
        return 1.0;
    }
    if (shadow_ndc.z < 0.0 || shadow_ndc.z > 1.0) {
        return 1.0;
    }
    let atlas_uv = entry.atlas_offset + slot_uv * entry.atlas_scale;
    let depth = shadow_ndc.z - entry.bias;

    let texel = 1.0 / 2048.0;
    let light_size = max(entry.light_size, 0.001);
    let search_bias = 0.001;
    let search_radius = 4.0 * texel * entry.atlas_scale.x * light_size;
    let dims_spot = vec2<f32>(textureDimensions(spot_shadow_atlas));
    var blocker_sum = 0.0;
    var blocker_count = 0.0;
    for (var index: i32 = 0; index < 8; index = index + 1) {
        let offset = POISSON_8[index] * search_radius;
        let texel_coords = vec2<i32>(clamp((atlas_uv + offset) * dims_spot, vec2<f32>(0.0), dims_spot - vec2<f32>(1.0)));
        let sampled = textureLoad(spot_shadow_atlas, texel_coords, 0);
        if (sampled < depth - search_bias) {
            blocker_sum = blocker_sum + sampled;
            blocker_count = blocker_count + 1.0;
        }
    }
    if (blocker_count == 0.0) {
        return 1.0;
    }
    let blocker_depth = blocker_sum / blocker_count;
    let penumbra = (depth - blocker_depth) / max(blocker_depth, 0.0001);
    let filter_radius = clamp(penumbra * 16.0 * texel * light_size, texel, 16.0 * texel) * entry.atlas_scale.x;

    var sum = 0.0;
    for (var index: i32 = 0; index < 16; index = index + 1) {
        let offset = POISSON_16[index] * filter_radius;
        sum = sum + textureSampleCompareLevel(spot_shadow_atlas, shadow_sampler, atlas_uv + offset, depth);
    }
    return sum / 16.0;
}

fn sample_point_shadow(world_pos: vec3<f32>, world_normal: vec3<f32>, shadow_index: i32) -> f32 {
    if (shadow_index < 0 || u32(shadow_index) >= point_shadow_uniforms.count) {
        return 1.0;
    }
    let entry = point_shadow_uniforms.entries[shadow_index];
    let light_to_frag = world_pos - entry.position;
    let distance = length(light_to_frag);
    if (distance > entry.range) {
        return 1.0;
    }
    let direction = normalize(light_to_frag);
    let normalized = distance / entry.range;
    let light_dir = -direction;
    let slope = 1.0 - abs(dot(world_normal, light_dir));
    let bias = entry.bias * (1.0 + slope * 2.0);
    let reference = normalized - bias;

    var tangent: vec3<f32>;
    var bitangent: vec3<f32>;
    if (direction.z < -0.9999999) {
        tangent = vec3<f32>(0.0, -1.0, 0.0);
        bitangent = vec3<f32>(-1.0, 0.0, 0.0);
    } else {
        let a = 1.0 / (1.0 + direction.z);
        let b = -direction.x * direction.y * a;
        tangent = vec3<f32>(1.0 - direction.x * direction.x * a, b, -direction.x);
        bitangent = vec3<f32>(b, 1.0 - direction.y * direction.y * a, -direction.y);
    }

    let light_size = max(entry.light_size, 0.001);
    let search_bias = 0.002;
    let search_radius = 0.01 * light_size;
    var blocker_sum = 0.0;
    var blocker_count = 0.0;
    for (var index: i32 = 0; index < 8; index = index + 1) {
        let disk = POISSON_8[index] * search_radius;
        let sample_dir = normalize(direction + tangent * disk.x + bitangent * disk.y);
        let sampled = textureSampleLevel(point_shadow_cube_array, point_shadow_sampler, sample_dir, shadow_index, 0.0).r;
        if (sampled < reference - search_bias) {
            blocker_sum = blocker_sum + sampled;
            blocker_count = blocker_count + 1.0;
        }
    }
    if (blocker_count == 0.0) {
        return 1.0;
    }
    let blocker_depth = blocker_sum / blocker_count;
    let penumbra = (reference - blocker_depth) / max(blocker_depth, 0.0001);
    let filter_radius = clamp(penumbra * 0.05 * light_size, 0.003, 0.06);

    var visibility = 0.0;
    for (var index: i32 = 0; index < 16; index = index + 1) {
        let disk = POISSON_16[index] * filter_radius;
        let sample_dir = normalize(direction + tangent * disk.x + bitangent * disk.y);
        let sampled = textureSampleLevel(point_shadow_cube_array, point_shadow_sampler, sample_dir, shadow_index, 0.0).r;
        if (reference <= sampled) {
            visibility = visibility + 1.0;
        }
    }
    return visibility / 16.0;
}

fn d_charlie(roughness: f32, n_dot_h: f32) -> f32 {
    let alpha = max(roughness * roughness, 0.0001);
    let inv_alpha = 1.0 / alpha;
    let cos2h = n_dot_h * n_dot_h;
    let sin2h = max(1.0 - cos2h, 0.0078125);
    return (2.0 + inv_alpha) * pow(sin2h, inv_alpha * 0.5) / (2.0 * PI);
}

fn v_neubelt(n_dot_v: f32, n_dot_l: f32) -> f32 {
    return clamp(1.0 / (4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v)), 0.0, 1.0);
}

fn fresnel_clearcoat(cos_theta: f32) -> f32 {
    return 0.04 + 0.96 * pow(max(1.0 - cos_theta, 0.0), 5.0);
}

fn d_ggx_anisotropic(n_dot_h: f32, t_dot_h: f32, b_dot_h: f32, at: f32, ab: f32) -> f32 {
    let a2 = at * ab;
    let f = vec3<f32>(ab * t_dot_h, at * b_dot_h, a2 * n_dot_h);
    let w2 = a2 / max(dot(f, f), 1.0e-6);
    return a2 * w2 * w2 / PI;
}

fn v_ggx_anisotropic(n_dot_l: f32, n_dot_v: f32, b_dot_v: f32, t_dot_v: f32, b_dot_l: f32, t_dot_l: f32, at: f32, ab: f32) -> f32 {
    let ggx_v = n_dot_l * length(vec3<f32>(at * t_dot_v, ab * b_dot_v, n_dot_v));
    let ggx_l = n_dot_v * length(vec3<f32>(at * t_dot_l, ab * b_dot_l, n_dot_l));
    return clamp(0.5 / max(ggx_v + ggx_l, 1.0e-4), 0.0, 1.0);
}

struct ShadeParams {
    aniso_strength: f32,
    aniso_t: vec3<f32>,
    aniso_b: vec3<f32>,
    sheen_color: vec3<f32>,
    sheen_roughness: f32,
    sheen_scale: f32,
    cc_factor: f32,
    cc_roughness: f32,
    cc_normal: vec3<f32>,
};

fn shade_one_light(light: Light, point_to_light: vec3<f32>, v: vec3<f32>, n: vec3<f32>, albedo: vec3<f32>, f0: vec3<f32>, metallic: f32, roughness: f32, p: ShadeParams) -> vec3<f32> {
    let l = normalize(point_to_light);
    let h = normalize(v + l);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_h = max(dot(n, h), 0.0);
    let radiance = light_radiance(light, point_to_light);
    let f = fresnel_schlick(max(dot(h, v), 0.0), f0);
    var specular: vec3<f32>;
    if (p.aniso_strength > 0.0) {
        let alpha = roughness * roughness;
        let at = mix(alpha, 1.0, p.aniso_strength * p.aniso_strength);
        let ab = alpha;
        let d = d_ggx_anisotropic(n_dot_h, dot(p.aniso_t, h), dot(p.aniso_b, h), at, ab);
        let vv = v_ggx_anisotropic(n_dot_l, n_dot_v, dot(p.aniso_b, v), dot(p.aniso_t, v), dot(p.aniso_b, l), dot(p.aniso_t, l), at, ab);
        specular = f * d * vv;
    } else {
        let ndf = distribution_ggx(n, h, roughness);
        let g = geometry_smith(n, v, l, roughness);
        specular = (ndf * g * f) / (4.0 * n_dot_v * n_dot_l + 0.0001);
    }
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = kd * albedo / PI;
    var contribution = (diffuse + specular) * radiance * n_dot_l;

    let sheen_max = max(p.sheen_color.r, max(p.sheen_color.g, p.sheen_color.b));
    if (sheen_max > 0.0 && n_dot_l > 0.0) {
        let sheen_brdf = p.sheen_color * d_charlie(p.sheen_roughness, n_dot_h) * v_neubelt(n_dot_v, n_dot_l);
        contribution = contribution * p.sheen_scale + sheen_brdf * radiance * n_dot_l;
    }

    if (p.cc_factor > 0.0) {
        let cc_n_dot_l = max(dot(p.cc_normal, l), 0.0);
        if (cc_n_dot_l > 0.0) {
            let cc_n_dot_v = max(dot(p.cc_normal, v), 0.0);
            let cc_n_dot_h = max(dot(p.cc_normal, h), 0.0);
            let cc_v_dot_h = max(dot(v, h), 0.0);
            let cc_alpha = p.cc_roughness * p.cc_roughness;
            let cc_a2 = cc_alpha * cc_alpha;
            let denom = cc_n_dot_h * cc_n_dot_h * (cc_a2 - 1.0) + 1.0;
            let cc_d = cc_a2 / (PI * denom * denom);
            let cc_k = (p.cc_roughness + 1.0) * (p.cc_roughness + 1.0) / 8.0;
            let cc_gv = cc_n_dot_v / (cc_n_dot_v * (1.0 - cc_k) + cc_k);
            let cc_gl = cc_n_dot_l / (cc_n_dot_l * (1.0 - cc_k) + cc_k);
            let cc_fr = fresnel_clearcoat(cc_v_dot_h);
            let cc_lobe = cc_fr * cc_d * (cc_gv * cc_gl) / max(4.0 * cc_n_dot_v * cc_n_dot_l, 0.001);
            contribution = contribution * (1.0 - p.cc_factor * cc_fr) + p.cc_factor * cc_lobe * radiance * cc_n_dot_l;
        }
    }
    return contribution;
}

@vertex
fn vertex_main(input: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let inst = instances[instance_index];
    let position = vec4<f32>(input.position.xyz, 1.0);
    let joint_offset = inst.joint_offset;
    var skinned_position = vec3<f32>(0.0, 0.0, 0.0);
    var skinned_normal = vec3<f32>(0.0, 0.0, 0.0);
    var skinned_tangent = vec3<f32>(0.0, 0.0, 0.0);
    for (var index = 0u; index < 4u; index = index + 1u) {
        let joint_index = input.joint_indices[index];
        let joint_weight = input.joint_weights[index];
        if (joint_weight > 0.0) {
            let joint_matrix = joint_matrices[joint_offset + joint_index];
            skinned_position = skinned_position + (joint_matrix * position).xyz * joint_weight;
            let normal_matrix = mat3x3<f32>(joint_matrix[0].xyz, joint_matrix[1].xyz, joint_matrix[2].xyz);
            skinned_normal = skinned_normal + (normal_matrix * input.normal.xyz) * joint_weight;
            skinned_tangent = skinned_tangent + (normal_matrix * input.tangent.xyz) * joint_weight;
        }
    }
    if (length(skinned_normal) < 0.0001) {
        skinned_normal = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        skinned_normal = normalize(skinned_normal);
    }

    var out: VertexOutput;
    out.clip_position = view_proj * vec4<f32>(skinned_position, 1.0);
    out.world_pos = skinned_position;
    out.world_normal = skinned_normal;
    out.world_tangent = vec4<f32>(safe_normalize(skinned_tangent), input.tangent.w);
    out.uv = input.uv;
    out.color = input.color;
    out.entity_id = inst.entity_id;
    out.material_index = inst.material_index;
    let view_mat3 = mat3x3<f32>(view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz);
    out.view_normal = view_mat3 * skinned_normal;
    out.world_scale_factor = inst.world_scale_factor;
    return out;
}

fn texture_uv(uv: vec4<f32>, transform: TextureTransform) -> vec2<f32> {
    let coords = select(uv.xy, uv.zw, u32(transform.row0.w) == 1u);
    let h = vec3<f32>(coords, 1.0);
    return vec2<f32>(dot(transform.row0.xyz, h), dot(transform.row1.xyz, h));
}

fn fresnel0_to_ior(f0: vec3<f32>) -> vec3<f32> {
    let s = sqrt(clamp(f0, vec3<f32>(0.0), vec3<f32>(0.9999)));
    return (vec3<f32>(1.0) + s) / (vec3<f32>(1.0) - s);
}

fn ior_to_fresnel0_v(transmitted: vec3<f32>, incident: f32) -> vec3<f32> {
    let r = (transmitted - vec3<f32>(incident)) / (transmitted + vec3<f32>(incident));
    return r * r;
}

fn ior_to_fresnel0_f(transmitted: f32, incident: f32) -> f32 {
    let r = (transmitted - incident) / (transmitted + incident);
    return r * r;
}

fn eval_sensitivity(opd: f32, shift: vec3<f32>) -> vec3<f32> {
    let phase = 2.0 * PI * opd * 1.0e-9;
    let val = vec3<f32>(5.4856e-13, 4.4201e-13, 5.2481e-13);
    let pos = vec3<f32>(1.6810e+06, 1.7953e+06, 2.2084e+06);
    let var_ = vec3<f32>(4.3278e+09, 9.3046e+09, 6.6121e+09);
    var xyz = val * sqrt(2.0 * PI * var_) * cos(pos * phase + shift) * exp(-phase * phase * var_);
    xyz.x = xyz.x + 9.7470e-14 * sqrt(2.0 * PI * 4.5282e+09) * cos(2.2399e+06 * phase + shift.x) * exp(-4.5282e+09 * phase * phase);
    xyz = xyz / 1.0685e-7;
    let xyz_to_rec709 = mat3x3<f32>(
        vec3<f32>(3.2404542, -0.9692660, 0.0556434),
        vec3<f32>(-1.5371385, 1.8760108, -0.2040259),
        vec3<f32>(-0.4985314, 0.0415560, 1.0572252),
    );
    return xyz_to_rec709 * xyz;
}

fn eval_iridescence(outside_ior: f32, eta2: f32, cos_theta1: f32, thickness: f32, base_f0: vec3<f32>) -> vec3<f32> {
    let irid_ior = mix(outside_ior, eta2, smoothstep(0.0, 0.03, thickness));
    let sin_theta2_sq = (outside_ior / irid_ior) * (outside_ior / irid_ior) * (1.0 - cos_theta1 * cos_theta1);
    let cos_theta2_sq = 1.0 - sin_theta2_sq;
    if (cos_theta2_sq < 0.0) {
        return vec3<f32>(1.0);
    }
    let cos_theta2 = sqrt(cos_theta2_sq);
    let r0_film = ior_to_fresnel0_f(irid_ior, outside_ior);
    let r12 = r0_film + (1.0 - r0_film) * pow(max(1.0 - cos_theta1, 0.0), 5.0);
    let t121 = 1.0 - r12;
    var phi12 = 0.0;
    if (irid_ior < outside_ior) { phi12 = PI; }
    let phi21 = PI - phi12;
    let base_ior = fresnel0_to_ior(base_f0);
    let r1 = ior_to_fresnel0_v(base_ior, irid_ior);
    let r23 = r1 + (vec3<f32>(1.0) - r1) * pow(max(1.0 - cos_theta2, 0.0), 5.0);
    var phi23 = vec3<f32>(0.0);
    if (base_ior.x < irid_ior) { phi23.x = PI; }
    if (base_ior.y < irid_ior) { phi23.y = PI; }
    if (base_ior.z < irid_ior) { phi23.z = PI; }
    let opd = 2.0 * irid_ior * thickness * cos_theta2;
    let phi = vec3<f32>(phi21) + phi23;
    let r123 = clamp(vec3<f32>(r12) * r23, vec3<f32>(1.0e-5), vec3<f32>(0.9999));
    let r123_sqrt = sqrt(r123);
    let rs = (t121 * t121) * r23 / (vec3<f32>(1.0) - r123);
    var i = vec3<f32>(r12) + rs;
    var cm = rs - vec3<f32>(t121);
    cm = cm * r123_sqrt;
    i = i + cm * 2.0 * eval_sensitivity(opd, phi);
    cm = cm * r123_sqrt;
    i = i + cm * 2.0 * eval_sensitivity(2.0 * opd, 2.0 * phi);
    return max(i, vec3<f32>(0.0));
}

@fragment
fn fragment_main(in: VertexOutput) -> FragmentOutput {
    let mat = materials[in.material_index];

    let view_dir = normalize(cluster_uniforms.camera_position.xyz - in.world_pos);

    let derived_normal = normalize(cross(dpdx(in.world_pos), dpdy(in.world_pos)));
    let n_len = length(in.world_normal);
    var geom_normal: vec3<f32>;
    if (n_len > 0.001) {
        geom_normal = in.world_normal / n_len;
    } else {
        geom_normal = derived_normal;
    }
    var geom_tangent = in.world_tangent;

    if (dot(geom_normal, view_dir) < 0.0) {
        geom_normal = -geom_normal;
        geom_tangent = -geom_tangent;
    }

    var normal_sample = vec3<f32>(0.5, 0.5, 1.0);
    let has_normal_texture = mat.normal_layer != NO_LAYER;
    if (has_normal_texture) {
        normal_sample = sample_linear_layer(mat.normal_layer, texture_uv(in.uv, mat.normal_transform)).xyz;
    }
    var normal = get_normal(geom_normal, geom_tangent, normal_sample, has_normal_texture, mat.normal_scale, 0u);
    if (cluster_uniforms.flat_color.a > 0.0) {
        normal = geom_normal;
    }

    var albedo_sample = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    if (mat.base_layer != NO_LAYER) {
        albedo_sample = sample_srgb_layer(mat.base_layer, texture_uv(in.uv, mat.base_transform));
    }
    if (cluster_uniforms.flat_color.a >= 2.0) {
        discard;
    }
    var base_color = mat.base_color * albedo_sample * in.color;
    if (cluster_uniforms.flat_color.a > 0.0) {
        base_color = vec4<f32>(cluster_uniforms.flat_color.rgb, base_color.a);
    }
    let albedo = base_color.rgb;

    if (mat.alpha_mode == 1u && base_color.a < mat.alpha_cutoff) {
        discard;
    }
    if (mat.alpha_mode == 2u || mat.transmission_factor > 0.0) {
        discard;
    }

    var metallic = mat.metallic_factor;
    var roughness = mat.roughness_factor;
    if (mat.metallic_roughness_layer != NO_LAYER) {
        let mr = sample_linear_layer(mat.metallic_roughness_layer, texture_uv(in.uv, mat.metallic_roughness_transform));
        roughness = roughness * mr.g;
        metallic = metallic * mr.b;
    }
    roughness = clamp(roughness, 0.04, 1.0);
    metallic = clamp(metallic, 0.0, 1.0);
    if (cluster_uniforms.flat_color.a > 0.0) {
        metallic = 0.0;
        roughness = 1.0;
    }

    var occlusion = 1.0;
    if (mat.occlusion_layer != NO_LAYER) {
        occlusion = sample_linear_layer(mat.occlusion_layer, texture_uv(in.uv, mat.occlusion_transform)).r;
    }

    var emissive = mat.emissive_factor * mat.emissive_strength;
    if (mat.emissive_layer != NO_LAYER) {
        emissive = emissive * sample_srgb_layer(mat.emissive_layer, texture_uv(in.uv, mat.emissive_transform)).rgb;
    }
    if (cluster_uniforms.flat_color.a > 0.0) {
        emissive = vec3<f32>(0.0);
    }

    if (mat.unlit != 0u || cluster_uniforms.global_unlit > 0.5) {
        var out_unlit: FragmentOutput;
        out_unlit.color = vec4<f32>(albedo + emissive, 1.0);
        out_unlit.entity_id = in.entity_id;
        out_unlit.view_normal = vec4<f32>(normalize(in.view_normal), 1.0);
        return out_unlit;
    }

    let v = view_dir;
    let n = normal;

    var specular_color = mat.specular_color_factor;
    if (mat.specular_color_layer != NO_LAYER) {
        specular_color = specular_color * sample_srgb_layer(mat.specular_color_layer, texture_uv(in.uv, mat.specular_color_transform)).rgb;
    }
    var specular_strength = mat.specular_factor;
    if (mat.specular_layer != NO_LAYER) {
        specular_strength = specular_strength * sample_linear_layer(mat.specular_layer, texture_uv(in.uv, mat.specular_transform)).a;
    }
    let dielectric_f0 = pow((mat.ior - 1.0) / (mat.ior + 1.0), 2.0);
    let dielectric_specular_f0 = min(vec3<f32>(dielectric_f0) * specular_color, vec3<f32>(1.0));
    var f0 = mix(dielectric_specular_f0, albedo, metallic);

    if (mat.iridescence_factor > 0.0) {
        var irid_factor = mat.iridescence_factor;
        if (mat.iridescence_layer != NO_LAYER) {
            irid_factor = irid_factor * sample_linear_layer(mat.iridescence_layer, texture_uv(in.uv, mat.iridescence_transform)).r;
        }
        var irid_thickness = mat.iridescence_thickness_max;
        if (mat.iridescence_thickness_layer != NO_LAYER) {
            let t = sample_linear_layer(mat.iridescence_thickness_layer, texture_uv(in.uv, mat.iridescence_thickness_transform)).g;
            irid_thickness = mix(mat.iridescence_thickness_min, mat.iridescence_thickness_max, t);
        }
        if (irid_factor > 0.0 && irid_thickness > 0.0) {
            let irid_f0 = eval_iridescence(1.0, mat.iridescence_ior, max(dot(n, v), 0.0), irid_thickness, f0);
            f0 = mix(f0, irid_f0, irid_factor);
        }
    }

    var shade: ShadeParams;
    shade.aniso_strength = mat.anisotropy_strength;
    var aniso_dir = vec2<f32>(mat.anisotropy_rotation_cos, mat.anisotropy_rotation_sin);
    if (mat.anisotropy_layer != NO_LAYER) {
        let a = sample_linear_layer(mat.anisotropy_layer, texture_uv(in.uv, mat.anisotropy_transform));
        let td = a.xy * 2.0 - 1.0;
        aniso_dir = vec2<f32>(
            mat.anisotropy_rotation_cos * td.x - mat.anisotropy_rotation_sin * td.y,
            mat.anisotropy_rotation_sin * td.x + mat.anisotropy_rotation_cos * td.y);
        shade.aniso_strength = shade.aniso_strength * a.b;
    }
    let tangent_w = normalize(geom_tangent.xyz);
    let bitangent_w = cross(n, tangent_w) * geom_tangent.w;
    shade.aniso_t = aniso_dir.x * tangent_w + aniso_dir.y * bitangent_w;
    shade.aniso_b = cross(n, shade.aniso_t);

    var sheen_color = mat.sheen_color_factor;
    if (mat.sheen_color_layer != NO_LAYER) {
        sheen_color = sheen_color * sample_srgb_layer(mat.sheen_color_layer, texture_uv(in.uv, mat.sheen_color_transform)).rgb;
    }
    var sheen_rough = clamp(mat.sheen_roughness_factor, 0.07, 1.0);
    if (mat.sheen_roughness_layer != NO_LAYER) {
        sheen_rough = sheen_rough * sample_linear_layer(mat.sheen_roughness_layer, texture_uv(in.uv, mat.sheen_roughness_transform)).a;
    }
    shade.sheen_color = sheen_color;
    shade.sheen_roughness = sheen_rough;
    let sheen_max = max(sheen_color.r, max(sheen_color.g, sheen_color.b));
    let sheen_e = textureSampleLevel(brdf_lut, ibl_sampler, vec2<f32>(max(dot(n, v), 0.0), sheen_rough), 0.0).b;
    shade.sheen_scale = clamp(1.0 - sheen_max * sheen_e, 0.0, 1.0);

    var cc_factor = mat.clearcoat_factor;
    if (mat.clearcoat_layer != NO_LAYER) {
        cc_factor = cc_factor * sample_linear_layer(mat.clearcoat_layer, texture_uv(in.uv, mat.clearcoat_transform)).r;
    }
    var cc_roughness = clamp(mat.clearcoat_roughness_factor, 0.04, 1.0);
    if (mat.clearcoat_roughness_layer != NO_LAYER) {
        cc_roughness = cc_roughness * sample_linear_layer(mat.clearcoat_roughness_layer, texture_uv(in.uv, mat.clearcoat_roughness_transform)).g;
    }
    var cc_normal = n;
    if (mat.clearcoat_normal_layer != NO_LAYER) {
        let cc_n_sample = sample_linear_layer(mat.clearcoat_normal_layer, texture_uv(in.uv, mat.clearcoat_normal_transform)).xyz;
        let cc_n_tangent_xy = (cc_n_sample.xy * 2.0 - 1.0) * mat.clearcoat_normal_scale;
        let cc_n_tangent = vec3<f32>(cc_n_tangent_xy, sqrt(max(1.0 - dot(cc_n_tangent_xy, cc_n_tangent_xy), 0.0)));
        let cc_t = normalize(in.world_tangent.xyz);
        let cc_b = cross(in.world_normal, cc_t) * in.world_tangent.w;
        let cc_tbn = mat3x3<f32>(cc_t, cc_b, in.world_normal);
        cc_normal = normalize(cc_tbn * cc_n_tangent);
    }
    shade.cc_factor = cc_factor;
    shade.cc_roughness = cc_roughness;
    shade.cc_normal = cc_normal;

    var lo = vec3<f32>(0.0);

    let shadow_factor = sample_directional_shadow(in.world_pos, n);

    for (var i = 0u; i < cluster_uniforms.num_directional_lights; i = i + 1u) {
        let light = lights[i];
        let point_to_light = -light.direction.xyz;
        var contribution = shade_one_light(light, point_to_light, v, n, albedo, f0, metallic, roughness, shade);
        if (i == 0u) {
            contribution = contribution * shadow_factor;
        }
        lo = lo + contribution;
    }

    let view_pos = view_matrix * vec4<f32>(in.world_pos, 1.0);
    let view_depth = -view_pos.z;
    let cluster_idx = get_cluster_index(in.clip_position.xy, view_depth);
    let grid = light_grid[cluster_idx];
    let base = cluster_idx * MAX_LIGHTS_PER_CLUSTER;
    let cluster_count = min(grid.count, MAX_LIGHTS_PER_CLUSTER);
    for (var i = 0u; i < cluster_count; i = i + 1u) {
        let light_idx = light_indices[base + i];
        let light = lights[cluster_uniforms.num_directional_lights + light_idx];
        let point_to_light = light.position.xyz - in.world_pos;
        var contribution = shade_one_light(light, point_to_light, v, n, albedo, f0, metallic, roughness, shade);
        if (light.light_type == LIGHT_TYPE_SPOT && light.shadow_index >= 0) {
            contribution = contribution * sample_spot_shadow(in.world_pos, n, light.shadow_index);
        }
        if (light.light_type == LIGHT_TYPE_POINT && light.shadow_index >= 0) {
            contribution = contribution * sample_point_shadow(in.world_pos, n, light.shadow_index);
        }
        lo = lo + contribution;
    }

    let n_dot_v = max(dot(n, v), 0.0);
    let r = reflect(-v, n);
    let f_ibl = fresnel_schlick_roughness(n_dot_v, f0, roughness);
    let irradiance = textureSampleLevel(irradiance_map, ibl_sampler, n, 0.0).rgb;
    let prefiltered = textureSampleLevel(prefiltered_env, ibl_sampler, r, roughness * MAX_REFLECTION_LOD).rgb;
    let brdf = textureSampleLevel(brdf_lut, ibl_sampler, vec2<f32>(n_dot_v, roughness), 0.0).rg;
    let fss_ess = f_ibl * brdf.x + brdf.y;
    let ems = 1.0 - (brdf.x + brdf.y);
    let f_avg = f0 + (vec3<f32>(1.0) - f0) / 21.0;
    let fms_ems = ems * fss_ess * f_avg / (vec3<f32>(1.0) - f_avg * ems);
    let c_diff = albedo * (1.0 - metallic);
    let kd_ibl = c_diff * (vec3<f32>(1.0) - fss_ess - fms_ems);
    var diffuse_ibl = (fms_ems + kd_ibl) * irradiance;

    if (mat.diffuse_transmission_factor > 0.0) {
        var dt_factor = mat.diffuse_transmission_factor;
        var dt_color = mat.diffuse_transmission_color_factor;
        if (mat.diffuse_transmission_color_layer != NO_LAYER) {
            dt_color = dt_color * sample_srgb_layer(mat.diffuse_transmission_color_layer, texture_uv(in.uv, mat.diffuse_transmission_color_transform)).rgb;
        }
        let back_irradiance = textureSampleLevel(irradiance_map, ibl_sampler, -n, 0.0).rgb;
        let c_diff_back = dt_color * albedo * (1.0 - metallic);
        let dt_diffuse = (fms_ems + c_diff_back * (vec3<f32>(1.0) - fss_ess - fms_ems)) * back_irradiance;
        diffuse_ibl = mix(diffuse_ibl, dt_diffuse, dt_factor);
    }

    if (mat.transmission_factor > 0.0) {
        var trans = mat.transmission_factor;
        if (mat.transmission_layer != NO_LAYER) {
            trans = trans * sample_linear_layer(mat.transmission_layer, texture_uv(in.uv, mat.transmission_transform)).r;
        }
        var thickness = mat.thickness;
        if (mat.thickness_layer != NO_LAYER) {
            thickness = thickness * sample_linear_layer(mat.thickness_layer, texture_uv(in.uv, mat.thickness_transform)).g;
        }
        let transmission = get_ibl_volume_refraction(
            n, v, in.world_pos, roughness, albedo, f0, mat.ior, mat.dispersion,
            thickness, mat.attenuation_color, mat.attenuation_distance,
            vec3<f32>(in.world_scale_factor),
        );
        let transmission_attenuated = transmission * (vec3<f32>(1.0) - fss_ess);
        diffuse_ibl = mix(diffuse_ibl, transmission_attenuated, trans);
    }

    let specular_ibl = prefiltered * fss_ess * specular_strength;
    var ambient = diffuse_ibl + specular_ibl;

    if (mat.clearcoat_factor > 0.0) {
        var cc = mat.clearcoat_factor;
        if (mat.clearcoat_layer != NO_LAYER) {
            cc = cc * sample_linear_layer(mat.clearcoat_layer, texture_uv(in.uv, mat.clearcoat_transform)).r;
        }
        var cc_rough = clamp(mat.clearcoat_roughness_factor, 0.04, 1.0);
        if (mat.clearcoat_roughness_layer != NO_LAYER) {
            cc_rough = cc_rough * sample_linear_layer(mat.clearcoat_roughness_layer, texture_uv(in.uv, mat.clearcoat_roughness_transform)).g;
        }
        let cc_n_dot_v = max(dot(cc_normal, v), 0.0);
        let cc_r = reflect(-v, cc_normal);
        let cc_prefiltered = textureSampleLevel(prefiltered_env, ibl_sampler, cc_r, cc_rough * MAX_REFLECTION_LOD).rgb;
        let cc_f = fresnel_clearcoat(cc_n_dot_v);
        let cc_brdf = textureSampleLevel(brdf_lut, ibl_sampler, vec2<f32>(cc_n_dot_v, cc_rough), 0.0).rg;
        let cc_spec = cc * (cc_f * cc_brdf.x + cc_brdf.y) * cc_prefiltered;
        ambient = ambient * (1.0 - cc * cc_f) + cc_spec;
    }

    ambient = mix(ambient, ambient * occlusion, mat.occlusion_strength);
    let color = ambient + lo + emissive;

    var output: FragmentOutput;
    output.color = vec4<f32>(color, 1.0);
    output.entity_id = in.entity_id;
    output.view_normal = vec4<f32>(normalize(in.view_normal), 1.0);
    return output;
}
