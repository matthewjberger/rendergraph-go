// Mesh pass shader: instanced PBR rendering with clustered light
// culling. Directional lights live at the head of the lights
// storage buffer and every fragment iterates them; local lights
// (point + spot) are pre-bucketed per cluster by the
// cluster_light_assign compute pass, and each fragment only
// iterates the lights its cluster overlaps.

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

@group(1) @binding(0) var<storage, read> lights:        array<Light>;
@group(1) @binding(1) var<storage, read> light_grid:    array<LightGrid>;
@group(1) @binding(2) var<storage, read> light_indices: array<u32>;
@group(1) @binding(3) var<uniform>       cluster_uniforms: ClusterUniforms;
@group(1) @binding(4) var<uniform>       view_matrix:   mat4x4<f32>;
@group(1) @binding(5) var material_srgb_array:   texture_2d_array<f32>;
@group(1) @binding(6) var material_linear_array: texture_2d_array<f32>;
@group(1) @binding(7) var material_sampler:      sampler;
@group(1) @binding(8) var<storage, read> materials: array<Material>;

@group(2) @binding(0) var<storage, read> models:           array<mat4x4<f32>>;
@group(2) @binding(1) var<storage, read> material_indices: array<u32>;
@group(2) @binding(2) var<storage, read> entity_ids:       array<u32>;

@group(3) @binding(0) var irradiance_map:  texture_cube<f32>;
@group(3) @binding(1) var prefiltered_env: texture_cube<f32>;
@group(3) @binding(2) var brdf_lut:        texture_2d<f32>;
@group(3) @binding(3) var ibl_sampler:     sampler;

const NO_LAYER: u32 = 0xFFFFFFFFu;
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

// get_normal builds the perturbed surface normal from the
// interpolated geometric normal + tangent and a sampled normal
// map. Re-orthogonalizes T against N to undo the post-
// rasterization interpolation drift, then assembles the TBN
// basis. The world_tangent's .w component is the glTF handedness
// sign for the bitangent (+1 or -1).
//
// flags is the per-material normal_map_flags bitfield: bit 0 =
// FLIP_Y (negate the green channel — common for tools that
// export -Y), bit 1 = TWO_COMPONENT (only XY stored, reconstruct
// Z = sqrt(1 - x^2 - y^2)). Both bits are passed in as 0 today;
// the helper takes the arg so the flag plumbing is in place for
// when materials surface them.
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

// Material sampling uses textureSampleGrad with explicit
// derivatives computed at the top of fragment_main (uniform
// control flow). textureSample requires uniform control flow
// because it picks LOD from implicit derivatives, and we call
// these helpers from inside per-fragment `if (layer != NO_LAYER)`
// branches that WebGPU's strict validation classifies as
// non-uniform. Passing ddx/ddy explicitly keeps full mipmapping
// available while satisfying the rule.
fn sample_srgb_layer(packed: u32, uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>) -> vec4<f32> {
    let layer = i32(packed & 0xFFFFu);
    let wrapped = apply_wrap(uv, packed);
    return textureSampleGrad(material_srgb_array, material_sampler, wrapped, layer, ddx, ddy);
}

fn sample_linear_layer(packed: u32, uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>) -> vec4<f32> {
    let layer = i32(packed & 0xFFFFu);
    let wrapped = apply_wrap(uv, packed);
    return textureSampleGrad(material_linear_array, material_sampler, wrapped, layer, ddx, ddy);
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

// fresnel_schlick_roughness is the roughness-aware variant used
// for IBL specular: pulls f0 up toward (1 - roughness) so rough
// surfaces don't artificially darken at grazing angles. Standard
// trick from Real Shading in Unreal Engine 4 / Karis 2013.
fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    let invR = vec3<f32>(1.0 - roughness);
    return f0 + (max(invR, f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
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

fn light_radiance(light: Light, point_to_light: vec3<f32>) -> vec3<f32> {
    var range_atten = 1.0;
    var spot_atten = 1.0;
    if (light.light_type != LIGHT_TYPE_DIRECTIONAL) {
        range_atten = range_attenuation(light.range, length(point_to_light));
    }
    if (light.light_type == LIGHT_TYPE_SPOT) {
        spot_atten = spot_attenuation(point_to_light, light.direction.xyz, light.outer_cone, light.inner_cone);
    }
    return range_atten * spot_atten * light.color.rgb;
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

fn shade_one_light(light: Light, point_to_light: vec3<f32>, v: vec3<f32>, n: vec3<f32>, albedo: vec3<f32>, f0: vec3<f32>, metallic: f32, roughness: f32) -> vec3<f32> {
    let l = normalize(point_to_light);
    let h = normalize(v + l);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 0.0);
    let radiance = light_radiance(light, point_to_light);
    let ndf = distribution_ggx(n, h, roughness);
    let g = geometry_smith(n, v, l, roughness);
    let f = fresnel_schlick(max(dot(h, v), 0.0), f0);
    let specular = (ndf * g * f) / (4.0 * n_dot_v * n_dot_l + 0.0001);
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = kd * albedo / PI;
    return (diffuse + specular) * radiance * n_dot_l;
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
    out.material_index = material_indices[instance_index];
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> FragmentOutput {
    let mat = materials[in.material_index];

    // Sample-coordinate derivatives, hoisted to uniform control
    // flow at the top of the fragment so the per-layer sample
    // helpers below can run inside conditionals via
    // textureSampleGrad without violating WebGPU's uniform-
    // control-flow rule for implicit-LOD sampling.
    let uv_ddx = dpdx(in.uv);
    let uv_ddy = dpdy(in.uv);

    // View direction in world space, recovered from the camera
    // position uniform (NOT from -world_pos — that only worked
    // when the camera sat at the origin).
    let view_dir = normalize(cluster_uniforms.camera_position.xyz - in.world_pos);

    // Re-derive the geometric normal. Falls back to a screen-space
    // derivative when the interpolated vertex normal is degenerate
    // (e.g., a flat-shaded primitive with all-zero NORMAL data).
    let derived_normal = normalize(cross(dpdx(in.world_pos), dpdy(in.world_pos)));
    let n_len = length(in.world_normal);
    var geom_normal: vec3<f32>;
    if (n_len > 0.001) {
        geom_normal = in.world_normal / n_len;
    } else {
        geom_normal = derived_normal;
    }
    var geom_tangent = in.world_tangent;

    // Back-face flip. Use the view-dir dot product instead of the
    // WGSL front_facing builtin: front_facing depends on culling
    // state and produces wrong results with CullModeNone +
    // double-sided meshes. Flipping the normal also flips the
    // tangent so the TBN basis stays consistent under the flip.
    if (dot(geom_normal, view_dir) < 0.0) {
        geom_normal = -geom_normal;
        geom_tangent = -geom_tangent;
    }

    var normal_sample = vec3<f32>(0.5, 0.5, 1.0);
    let has_normal_texture = mat.normal_layer != NO_LAYER;
    if (has_normal_texture) {
        normal_sample = sample_linear_layer(mat.normal_layer, in.uv, uv_ddx, uv_ddy).xyz;
    }
    let normal = get_normal(geom_normal, geom_tangent, normal_sample, has_normal_texture, mat.normal_scale, 0u);

    var albedo_sample = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    if (mat.base_layer != NO_LAYER) {
        albedo_sample = sample_srgb_layer(mat.base_layer, in.uv, uv_ddx, uv_ddy);
    }
    let base_color = mat.base_color * albedo_sample * in.color;
    let albedo = base_color.rgb;

    if (mat.alpha_mode == 1u && base_color.a < mat.alpha_cutoff) {
        discard;
    }

    var metallic = mat.metallic_factor;
    var roughness = mat.roughness_factor;
    if (mat.metallic_roughness_layer != NO_LAYER) {
        let mr = sample_linear_layer(mat.metallic_roughness_layer, in.uv, uv_ddx, uv_ddy);
        roughness = roughness * mr.g;
        metallic = metallic * mr.b;
    }
    roughness = clamp(roughness, 0.04, 1.0);
    metallic = clamp(metallic, 0.0, 1.0);

    // Raw occlusion sample. Strength is applied later via
    // mix(ambient, ambient*occlusion, strength) so the term only
    // attenuates the ambient IBL contribution. Applying strength
    // here too would double-count it and crush ambient on any
    // material with strength != 1.
    var occlusion = 1.0;
    if (mat.occlusion_layer != NO_LAYER) {
        occlusion = sample_linear_layer(mat.occlusion_layer, in.uv, uv_ddx, uv_ddy).r;
    }

    var emissive = mat.emissive_factor;
    if (mat.emissive_layer != NO_LAYER) {
        emissive = emissive * sample_srgb_layer(mat.emissive_layer, in.uv, uv_ddx, uv_ddy).rgb;
    }

    if (mat.unlit != 0u) {
        var out_unlit: FragmentOutput;
        out_unlit.color = vec4<f32>(albedo + emissive, base_color.a);
        out_unlit.entity_id = in.entity_id;
        return out_unlit;
    }

    // Reuse the view_dir computed above for V — after the back-
    // face flip, geom_normal points toward view_dir, so V is
    // exactly normalize(camera_position - world_pos).
    let v = view_dir;
    let n = normal;
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    var lo = vec3<f32>(0.0);

    // Iterate every directional light unconditionally; they have no
    // bounding volume and affect every cluster.
    for (var i = 0u; i < cluster_uniforms.num_directional_lights; i = i + 1u) {
        let light = lights[i];
        let point_to_light = -light.direction.xyz;
        lo = lo + shade_one_light(light, point_to_light, v, n, albedo, f0, metallic, roughness);
    }

    // Iterate only the local lights that touch this fragment's
    // cluster, looked up via screen-space tile + log-z slice.
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
        lo = lo + shade_one_light(light, point_to_light, v, n, albedo, f0, metallic, roughness);
    }

    // Image-based ambient (split-sum approximation + multi-scatter
    // Fdez-Aguera 2019). Diffuse term samples the Lambertian-filtered
    // irradiance cubemap with N; specular term samples the
    // GGX-prefiltered cubemap with R at LOD = roughness *
    // MAX_REFLECTION_LOD and pairs it with the pre-integrated BRDF
    // LUT sampled at (NdotV, roughness).
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
    let diffuse_ibl = (fms_ems + kd_ibl) * irradiance;
    let specular_ibl = prefiltered * fss_ess;
    var ambient = diffuse_ibl + specular_ibl;
    ambient = mix(ambient, ambient * occlusion, mat.occlusion_strength);
    let color = ambient + lo + emissive;

    var output: FragmentOutput;
    output.color = vec4<f32>(color, base_color.a);
    output.entity_id = in.entity_id;
    return output;
}
