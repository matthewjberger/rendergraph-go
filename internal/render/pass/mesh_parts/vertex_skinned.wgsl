struct VertexInput {
    @location(0) position:      vec4<f32>,
    @location(1) normal:        vec4<f32>,
    @location(2) tangent:       vec4<f32>,
    @location(3) uv:            vec4<f32>,
    @location(4) color:         vec4<f32>,
    @location(5) joint_indices: vec4<u32>,
    @location(6) joint_weights: vec4<f32>,
};
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
    morph_weights:             array<f32, 8>,
    morph_target_count:        u32,
    morph_displacement_offset: u32,
    morph_vertex_count:        u32,
    flip_winding:              u32,
};
@group(2) @binding(0) var<storage, read> joint_matrices: array<mat4x4<f32>>;
@group(2) @binding(1) var<storage, read> instances:      array<SkinnedInstance>;
@group(2) @binding(2) var<storage, read> morph_displacements: array<MorphDisplacement>;
@vertex
fn vertex_main(input: VertexInput, @builtin(instance_index) instance_index: u32, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var inst = instances[instance_index];
    var morphed_position = input.position.xyz;
    var morphed_normal = input.normal.xyz;
    var morphed_tangent = input.tangent.xyz;
    if (inst.morph_target_count > 0u) {
        for (var t = 0u; t < inst.morph_target_count; t = t + 1u) {
            let w = inst.morph_weights[t];
            if (abs(w) > 0.0001) {
                let d = morph_displacements[inst.morph_displacement_offset + t * inst.morph_vertex_count + vertex_index];
                morphed_position = morphed_position + d.position * w;
                morphed_normal = morphed_normal + d.normal * w;
                morphed_tangent = morphed_tangent + d.tangent * w;
            }
        }
    }
    let position = vec4<f32>(morphed_position, 1.0);
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
            skinned_normal = skinned_normal + (normal_matrix * morphed_normal) * joint_weight;
            skinned_tangent = skinned_tangent + (normal_matrix * morphed_tangent) * joint_weight;
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
    // flip_winding is computed on the CPU from the entity's world transform
    // determinant (skinning bakes the transform into joint matrices, so it
    // can't be derived here), matching nightshade's per-object approach.
    out.flip_winding = inst.flip_winding;
    return out;
}
