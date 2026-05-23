// Instanced mesh draw. The world matrix per instance is produced
// by the instanced transform compute (parent * local); the vertex
// stage reads it by instance index and derives the normal matrix
// from its upper 3x3 (valid for uniform scale). Lambert + ambient
// shading, consistent with the engine's other simplified paths.

struct ViewProj {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> view_proj_uniform: ViewProj;

struct InstancedUniforms {
    light_direction: vec4<f32>,
    light_color:     vec4<f32>,
    ambient_color:   vec4<f32>,
};

@group(1) @binding(0) var<uniform> lighting: InstancedUniforms;

struct PerEntity {
    base_color: vec4<f32>,
    entity_id:  u32,
    _pad0:      u32,
    _pad1:      u32,
    _pad2:      u32,
};

@group(2) @binding(0) var<storage, read> world_matrices: array<mat4x4<f32>>;
@group(2) @binding(1) var<uniform>       per_entity:     PerEntity;

struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) normal:   vec4<f32>,
    @location(2) tangent:  vec4<f32>,
    @location(3) uv:       vec4<f32>,
    @location(4) color:    vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color:        vec4<f32>,
    @location(2) @interpolate(flat) entity_id: u32,
};

struct FragmentOutput {
    @location(0) color:       vec4<f32>,
    @location(1) entity_id:   u32,
    @location(2) view_normal: vec4<f32>,
};

@vertex
fn vertex_main(input: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let model = world_matrices[instance_index];
    let world_position = model * vec4<f32>(input.position.xyz, 1.0);
    let normal_matrix = mat3x3<f32>(model[0].xyz, model[1].xyz, model[2].xyz);
    let world_normal = normalize(normal_matrix * input.normal.xyz);

    var out: VertexOutput;
    out.clip_position = view_proj_uniform.view_proj * world_position;
    out.world_normal = world_normal;
    out.color = input.color;
    out.entity_id = per_entity.entity_id;
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> FragmentOutput {
    let albedo = per_entity.base_color.rgb * in.color.rgb;
    let normal = normalize(in.world_normal);
    let light_dir = -normalize(lighting.light_direction.xyz);
    let lambert = max(dot(normal, light_dir), 0.0);
    let lit = lighting.ambient_color.rgb + lighting.light_color.rgb * lambert;
    var out: FragmentOutput;
    out.color = vec4<f32>(albedo * lit, per_entity.base_color.a);
    out.entity_id = in.entity_id;
    out.view_normal = vec4<f32>(normal, 1.0);
    return out;
}
