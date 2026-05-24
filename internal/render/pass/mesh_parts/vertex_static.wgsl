struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) normal:   vec4<f32>,
    @location(2) tangent:  vec4<f32>,
    @location(3) uv:       vec4<f32>,
    @location(4) color:    vec4<f32>,
};
struct MorphInstance {
    weights: array<f32, 8>,
    target_count: u32,
    vertex_count: u32,
    _mpad0: u32,
    _mpad1: u32,
};
@group(2) @binding(0) var<storage, read> models:           array<mat4x4<f32>>;
@group(2) @binding(1) var<storage, read> material_indices: array<u32>;
@group(2) @binding(2) var<storage, read> entity_ids:       array<u32>;
@group(2) @binding(3) var<storage, read> visible_indices:  array<u32>;
@group(2) @binding(4) var<storage, read> morph_displacements: array<MorphDisplacement>;
@group(2) @binding(5) var<storage, read> morph_instances:     array<MorphInstance>;
@vertex
fn vertex_main(input: VertexInput, @builtin(instance_index) instance_index: u32, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let slot = visible_indices[instance_index];
    let model = models[slot];
    var local_position = input.position.xyz;
    var local_normal = input.normal.xyz;
    var local_tangent = input.tangent.xyz;
    var morph = morph_instances[slot];
    if (morph.target_count > 0u) {
        for (var t = 0u; t < morph.target_count; t = t + 1u) {
            let w = morph.weights[t];
            if (abs(w) > 0.0001) {
                let d = morph_displacements[t * morph.vertex_count + vertex_index];
                local_position = local_position + d.position * w;
                local_normal = local_normal + d.normal * w;
                local_tangent = local_tangent + d.tangent * w;
            }
        }
    }
    var out: VertexOutput;
    let world = model * vec4<f32>(local_position, 1.0);
    out.clip_position = view_proj * world;
    out.world_pos = world.xyz;
    out.world_normal = (model * vec4<f32>(local_normal, 0.0)).xyz;
    let world_tangent = (model * vec4<f32>(local_tangent, 0.0)).xyz;
    out.world_tangent = vec4<f32>(world_tangent, input.tangent.w);
    out.uv = input.uv;
    out.color = input.color;
    out.entity_id = entity_ids[slot];
    out.material_index = material_indices[slot];
    let view_mat3 = mat3x3<f32>(view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz);
    out.view_normal = view_mat3 * out.world_normal;
    out.world_scale_factor = (length(model[0].xyz) + length(model[1].xyz) + length(model[2].xyz)) / 3.0;
    return out;
}
