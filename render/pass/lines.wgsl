struct LineSegment {
    start: vec4<f32>,
    end:   vec4<f32>,
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> view_proj: mat4x4<f32>;
@group(0) @binding(1) var<storage, read> lines: array<LineSegment>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
    let seg = lines[ii];
    var world: vec3<f32>;
    if (vi == 0u) {
        world = seg.start.xyz;
    } else {
        world = seg.end.xyz;
    }
    var out: VertexOutput;
    out.clip_position = view_proj * vec4<f32>(world, 1.0);
    out.color = seg.color;
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
