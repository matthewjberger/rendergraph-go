@group(0) @binding(0) var<storage, read>       bone_transforms:       array<mat4x4<f32>>;
@group(0) @binding(1) var<storage, read>       inverse_bind_matrices: array<mat4x4<f32>>;
@group(0) @binding(2) var<storage, read_write> joint_matrices:        array<mat4x4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let count = arrayLength(&joint_matrices);
    if (i >= count) {
        return;
    }
    joint_matrices[i] = bone_transforms[i] * inverse_bind_matrices[i];
}
