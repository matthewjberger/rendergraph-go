
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
    blend_opaque_alpha_threshold:      f32,

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

@group(2) @binding(0) var prefiltered_env: texture_cube<f32>;
@group(2) @binding(1) var brdf_lut:        texture_2d<f32>;
@group(2) @binding(2) var ibl_sampler:     sampler;
@group(2) @binding(3) var transmission_color_texture: texture_2d<f32>;
@group(2) @binding(4) var transmission_color_sampler: sampler;

const MAX_REFLECTION_LOD: f32 = 4.0;

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
    @location(7) @interpolate(flat) world_scale_factor: f32,
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
    out.world_scale_factor = (length(model[0].xyz) + length(model[1].xyz) + length(model[2].xyz)) / 3.0;
    return out;
}

fn oit_weight(view_z: f32, a: f32) -> f32 {
    let z_scale = max(view_proj_uniform.oit_z_scale, 1.0);
    let z_ratio = view_z / z_scale;
    return a * clamp(0.03 / (1e-5 + pow(z_ratio, 4.0)), 1e-2, 3e3);
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
    let exit_clip = view_proj_uniform.view_proj * vec4<f32>(exit_world, 1.0);
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

@fragment
fn fs_blend_opaque_prepass(in: VertexOutput) {
    let mat = materials[in.material_index];
    if (mat.alpha_mode != 2u) {
        discard;
    }
    if (mat.transmission_factor > 0.0) {
        return;
    }
    var alpha = mat.base_color.a * in.color.a;
    if (mat.base_layer != NO_TEXTURE_LAYER) {
        let layer = i32(mat.base_layer & 0xFFFFu);
        alpha = alpha * textureSampleLevel(material_srgb_array, material_sampler, in.uv, layer, 0.0).a;
    }
    if (alpha < mat.blend_opaque_alpha_threshold) {
        discard;
    }
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

    if (mat.transmission_factor > 0.0) {
        let normal = normalize(in.world_normal);
        let view = normalize(view_proj_uniform.camera_position.xyz - in.world_position);
        let f0 = mix(vec3<f32>(0.04), base_color.rgb, mat.metallic_factor);
        let transmission = get_ibl_volume_refraction(
            normal, view, in.world_position, mat.roughness_factor, base_color.rgb, f0,
            mat.ior, mat.dispersion, mat.thickness, mat.attenuation_color, mat.attenuation_distance,
            vec3<f32>(in.world_scale_factor),
        );
        albedo = mix(albedo, transmission, mat.transmission_factor);
    }

    let w = oit_weight(in.view_z, alpha);

    var out: OitOutput;
    out.accum = vec4<f32>(albedo * alpha, alpha) * w;
    out.reveal = alpha;
    out.entity_id = in.entity_id;
    return out;
}
