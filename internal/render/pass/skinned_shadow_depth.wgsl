
struct CascadeUniforms {
    light_view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> cascade: CascadeUniforms;

struct JointOffset {
    value: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(1) @binding(0) var<storage, read> joint_matrices: array<mat4x4<f32>>;
@group(1) @binding(1) var<uniform>       joint_offset:   JointOffset;

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
};

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    let position = vec4<f32>(input.position.xyz, 1.0);
    var skinned_position = vec3<f32>(0.0, 0.0, 0.0);
    for (var index = 0u; index < 4u; index = index + 1u) {
        let joint_weight = input.joint_weights[index];
        if (joint_weight > 0.0) {
            let joint_index = input.joint_indices[index];
            let joint_matrix = joint_matrices[joint_offset.value + joint_index];
            skinned_position = skinned_position + (joint_matrix * position).xyz * joint_weight;
        }
    }
    var out: VertexOutput;
    out.clip_position = cascade.light_view_proj * vec4<f32>(skinned_position, 1.0);
    return out;
}
