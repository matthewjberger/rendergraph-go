struct Uniforms {
    parent_transform: mat4x4<f32>,
    instance_count:   u32,
    output_offset:    u32,
    _pad0:            u32,
    _pad1:            u32,
};

@group(0) @binding(0) var<storage, read>       local_matrices: array<mat4x4<f32>>;
@group(0) @binding(1) var<storage, read_write> world_matrices: array<mat4x4<f32>>;
@group(0) @binding(2) var<uniform>             uniforms:       Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= uniforms.instance_count) {
        return;
    }
    let slot = uniforms.output_offset + index;
    world_matrices[slot] = uniforms.parent_transform * local_matrices[slot];
}
