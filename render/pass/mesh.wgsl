struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) uv: vec4<f32>,
    @location(4) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) world_normal: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) @interpolate(flat) entity_id: u32,
};

struct LightData {
    position: vec3<f32>,
    light_type: u32,
    direction: vec3<f32>,
    range: f32,
    color: vec3<f32>,
    intensity: f32,
};

struct Lights {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    data: array<LightData, 8>,
};

struct MaterialData {
    base_color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> view_proj: mat4x4<f32>;
@group(1) @binding(0) var<storage, read> models: array<mat4x4<f32>>;
@group(1) @binding(1) var<storage, read> materials: array<MaterialData>;
@group(1) @binding(2) var<storage, read> entity_ids: array<u32>;
@group(1) @binding(3) var base_color_tex: texture_2d<f32>;
@group(1) @binding(4) var base_color_sampler: sampler;
@group(2) @binding(0) var<uniform> lights: Lights;

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) entity_id: u32,
};

@vertex
fn vertex_main(input: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let world = models[instance_index] * input.position;
    out.position = view_proj * world;
    let normal_world = (models[instance_index] * vec4<f32>(input.normal.xyz, 0.0)).xyz;
    out.world_normal = normal_world;
    out.uv = input.uv.xy;
    out.color = input.color * materials[instance_index].base_color;
    out.world_pos = world.xyz;
    out.entity_id = entity_ids[instance_index];
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> FragmentOutput {
    // Compute the screen-space derivatives unconditionally so WebGPU's
    // strict uniform-control-flow rule for dpdx/dpdy is satisfied,
    // then pick between the vertex normal and the derived face normal
    // afterward.
    let derived = normalize(cross(dpdx(in.world_pos), dpdy(in.world_pos)));
    let n_len = length(in.world_normal);
    var normal: vec3<f32>;
    if (n_len > 0.001) {
        normal = in.world_normal / n_len;
    } else {
        normal = derived;
    }

    let sampled = textureSample(base_color_tex, base_color_sampler, in.uv);
    let albedo = in.color.rgb * sampled.rgb;

    let ambient = vec3<f32>(0.18, 0.18, 0.2);
    var lit = albedo * ambient;

    for (var i: u32 = 0u; i < lights.count; i = i + 1u) {
        let light = lights.data[i];
        var n_dot_l = 0.0;
        if (light.light_type == 0u) {
            n_dot_l = abs(dot(normal, light.direction));
        } else {
            let to_light = light.position - in.world_pos;
            let dist = length(to_light);
            let l = to_light / max(dist, 0.0001);
            n_dot_l = abs(dot(normal, l));
            if (light.range > 0.0) {
                let attenuation = max(0.0, 1.0 - dist / light.range);
                n_dot_l = n_dot_l * attenuation * attenuation;
            }
        }
        lit = lit + albedo * light.color * (light.intensity * n_dot_l);
    }

    var out: FragmentOutput;
    out.color = vec4<f32>(lit, in.color.a * sampled.a);
    out.entity_id = in.entity_id;
    return out;
}
