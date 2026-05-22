// Mesh pass shader: instanced PBR rendering with shared sRGB +
// linear texture arrays. Material parameters and texture layer
// indices live in a per-handle storage buffer. Each fragment
// samples its material's base color, normal, metallic-roughness,
// occlusion, and emissive maps, then runs Cook-Torrance shading
// against every active light. Ports nightshade's mesh.wgsl PBR
// path, trimmed to the feature set indigo supports today (no
// skinning, no IBL, no shadow mapping, no morph targets).

struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) normal:   vec4<f32>,
    @location(2) tangent:  vec4<f32>,
    @location(3) uv:       vec4<f32>,
    @location(4) color:    vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) world_normal: vec3<f32>,
    @location(3) world_tangent: vec4<f32>,
    @location(4) uv: vec2<f32>,
    @location(5) @interpolate(flat) entity_id: u32,
    @location(6) @interpolate(flat) material_index: u32,
};

struct LightData {
    position:   vec3<f32>,
    light_type: u32,
    direction:  vec3<f32>,
    range:      f32,
    color:      vec3<f32>,
    intensity:  f32,
};

struct Lights {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    data:  array<LightData, 8>,
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
    _pad0:            u32,

    _pad1: vec4<f32>,
};

@group(0) @binding(0) var<uniform> view_proj: mat4x4<f32>;

@group(1) @binding(0) var<uniform> lights: Lights;
@group(1) @binding(1) var material_srgb_array:   texture_2d_array<f32>;
@group(1) @binding(2) var material_linear_array: texture_2d_array<f32>;
@group(1) @binding(3) var material_sampler:      sampler;

@group(2) @binding(0) var<storage, read> models:     array<mat4x4<f32>>;
@group(2) @binding(1) var<storage, read> materials:  array<Material>;
@group(2) @binding(2) var<storage, read> entity_ids: array<u32>;

const NO_LAYER: u32 = 0xFFFFFFFFu;
const PI: f32 = 3.14159265359;

struct FragmentOutput {
    @location(0) color:     vec4<f32>,
    @location(1) entity_id: u32,
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
    return textureSample(material_srgb_array, material_sampler, wrapped, layer);
}

fn sample_linear_layer(packed: u32, uv: vec2<f32>) -> vec4<f32> {
    let layer = i32(packed & 0xFFFFu);
    let wrapped = apply_wrap(uv, packed);
    return textureSample(material_linear_array, material_sampler, wrapped, layer);
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

@vertex
fn vertex_main(input: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let model = models[instance_index];
    var out: VertexOutput;
    let world = model * input.position;
    out.clip_position = view_proj * world;
    out.world_pos = world.xyz;
    out.world_normal = (model * vec4<f32>(input.normal.xyz, 0.0)).xyz;
    let world_tangent = (model * vec4<f32>(input.tangent.xyz, 0.0)).xyz;
    out.world_tangent = vec4<f32>(world_tangent, input.tangent.w);
    out.uv = input.uv.xy;
    out.color = input.color;
    out.entity_id = entity_ids[instance_index];
    out.material_index = instance_index;
    return out;
}

@fragment
fn fragment_main(in: VertexOutput, @builtin(front_facing) front_facing: bool) -> FragmentOutput {
    let mat = materials[in.material_index];

    let derived_normal = normalize(cross(dpdx(in.world_pos), dpdy(in.world_pos)));
    let n_len = length(in.world_normal);
    var geom_normal: vec3<f32>;
    if (n_len > 0.001) {
        geom_normal = in.world_normal / n_len;
    } else {
        geom_normal = derived_normal;
    }
    if (!front_facing) {
        geom_normal = -geom_normal;
    }

    var normal = geom_normal;
    if (mat.normal_layer != NO_LAYER) {
        let n_sample = sample_linear_layer(mat.normal_layer, in.uv).xyz * 2.0 - vec3<f32>(1.0);
        let scaled = vec3<f32>(n_sample.xy * mat.normal_scale, n_sample.z);
        let t_len = length(in.world_tangent.xyz);
        if (t_len > 0.001) {
            let t = in.world_tangent.xyz / t_len;
            let b = normalize(cross(geom_normal, t) * in.world_tangent.w);
            normal = normalize(t * scaled.x + b * scaled.y + geom_normal * scaled.z);
        }
    }

    var albedo_sample = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    if (mat.base_layer != NO_LAYER) {
        albedo_sample = sample_srgb_layer(mat.base_layer, in.uv);
    }
    let base_color = mat.base_color * albedo_sample * in.color;
    let albedo = base_color.rgb;

    if (mat.alpha_mode == 1u && base_color.a < mat.alpha_cutoff) {
        discard;
    }

    var metallic = mat.metallic_factor;
    var roughness = mat.roughness_factor;
    if (mat.metallic_roughness_layer != NO_LAYER) {
        let mr = sample_linear_layer(mat.metallic_roughness_layer, in.uv);
        roughness = roughness * mr.g;
        metallic = metallic * mr.b;
    }
    roughness = clamp(roughness, 0.04, 1.0);
    metallic = clamp(metallic, 0.0, 1.0);

    var occlusion = 1.0;
    if (mat.occlusion_layer != NO_LAYER) {
        let occ_sample = sample_linear_layer(mat.occlusion_layer, in.uv).r;
        occlusion = 1.0 + mat.occlusion_strength * (occ_sample - 1.0);
    }

    var emissive = mat.emissive_factor;
    if (mat.emissive_layer != NO_LAYER) {
        emissive = emissive * sample_srgb_layer(mat.emissive_layer, in.uv).rgb;
    }

    if (mat.unlit != 0u) {
        var out_unlit: FragmentOutput;
        out_unlit.color = vec4<f32>(albedo + emissive, base_color.a);
        out_unlit.entity_id = in.entity_id;
        return out_unlit;
    }

    let v = normalize(-in.world_pos);
    let n = normal;
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    var lo = vec3<f32>(0.0);
    let count = lights.count;
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let light = lights.data[i];
        var l: vec3<f32>;
        var radiance: vec3<f32>;
        if (light.light_type == 0u) {
            l = normalize(-light.direction);
            radiance = light.color * light.intensity;
        } else {
            let to_light = light.position - in.world_pos;
            let dist = length(to_light);
            l = to_light / max(dist, 0.0001);
            let attenuation = 1.0 / max(dist * dist, 0.0001);
            radiance = light.color * light.intensity * attenuation;
        }
        let h = normalize(v + l);
        let n_dot_l = max(dot(n, l), 0.0);
        let ndf = distribution_ggx(n, h, roughness);
        let g = geometry_smith(n, v, l, roughness);
        let f = fresnel_schlick(max(dot(h, v), 0.0), f0);
        let specular_num = ndf * g * f;
        let specular_den = 4.0 * max(dot(n, v), 0.0) * n_dot_l + 0.0001;
        let specular = specular_num / specular_den;
        let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
        let diffuse = kd * albedo / PI;
        lo = lo + (diffuse + specular) * radiance * n_dot_l;
    }

    let ambient = albedo * 0.05 * occlusion;
    let color = (ambient + lo) * occlusion + emissive;

    var output: FragmentOutput;
    output.color = vec4<f32>(color, base_color.a);
    output.entity_id = in.entity_id;
    return output;
}
