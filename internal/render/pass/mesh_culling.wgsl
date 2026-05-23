struct CullingUniforms {
    frustum_planes:        array<vec4<f32>, 6>,
    view_projection:       mat4x4<f32>,
    screen_size:           vec2<f32>,
    min_screen_pixel_size: f32,
    projection_scale_y:    f32,
    enabled:               u32,
    _pad0:                 u32,
    _pad1:                 u32,
    _pad2:                 u32,
};

struct BucketParams {
    bounds_center: vec3<f32>,
    bounds_radius: f32,
    object_count:  u32,
    _pad0:         u32,
    _pad1:         u32,
    _pad2:         u32,
};

struct DrawIndirect {
    vertex_count:   u32,
    instance_count: atomic<u32>,
    first_vertex:   u32,
    first_instance: u32,
};

@group(0) @binding(0) var<uniform> culling: CullingUniforms;

@group(1) @binding(0) var<uniform>             params:           BucketParams;
@group(1) @binding(1) var<storage, read>       models:           array<mat4x4<f32>>;
@group(1) @binding(2) var<storage, read_write> indirect_command: DrawIndirect;
@group(1) @binding(3) var<storage, read_write> visible_indices:  array<u32>;

fn sphere_in_frustum(center: vec3<f32>, radius: f32) -> bool {
    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = culling.frustum_planes[i];
        let distance = dot(plane.xyz, center) + plane.w;
        if (distance < -radius) {
            return false;
        }
    }
    return true;
}

fn matrix_max_scale(m: mat4x4<f32>) -> f32 {
    let sx = length(vec3<f32>(m[0].x, m[0].y, m[0].z));
    let sy = length(vec3<f32>(m[1].x, m[1].y, m[1].z));
    let sz = length(vec3<f32>(m[2].x, m[2].y, m[2].z));
    return max(sx, max(sy, sz));
}

@compute @workgroup_size(64)
fn cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let instance_index = gid.x;
    if (instance_index >= params.object_count) {
        return;
    }
    if (culling.enabled == 0u) {
        let write_index = atomicAdd(&indirect_command.instance_count, 1u);
        visible_indices[write_index] = instance_index;
        return;
    }
    let model = models[instance_index];
    let world_center = (model * vec4<f32>(params.bounds_center, 1.0)).xyz;
    let world_radius = params.bounds_radius * matrix_max_scale(model);
    if (!sphere_in_frustum(world_center, world_radius)) {
        return;
    }
    if (culling.min_screen_pixel_size > 0.0) {
        let clip = culling.view_projection * vec4<f32>(world_center, 1.0);
        let w = max(abs(clip.w), 1e-4);
        let diameter_px = 2.0 * world_radius * culling.projection_scale_y * culling.screen_size.y / w;
        if (diameter_px < culling.min_screen_pixel_size) {
            return;
        }
    }
    let write_index = atomicAdd(&indirect_command.instance_count, 1u);
    visible_indices[write_index] = instance_index;
}
