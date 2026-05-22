struct GizmoLine {
    start_end: vec4<f32>,
    color: vec4<f32>,
    extra: vec4<f32>,
};

struct Viewport {
    size: vec2<f32>,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> viewport: Viewport;
@group(0) @binding(1) var<storage, read> lines: array<GizmoLine>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
    let line = lines[ii];
    let start = line.start_end.xy;
    let end = line.start_end.zw;
    let thickness = line.extra.x;
    let kind = line.extra.y;

    let dir = end - start;
    let length = max(length(dir), 0.0001);
    let unit = dir / length;
    let normal = vec2<f32>(-unit.y, unit.x) * (thickness * 0.5);

    var pixel: vec2<f32>;
    if (kind < 0.5) {
        // Rectangle (line segment) — six corners spanning a rotated rect.
        var corners = array<vec2<f32>, 6>(
            start + normal,
            start - normal,
            end + normal,
            start - normal,
            end - normal,
            end + normal,
        );
        pixel = corners[vi];
    } else {
        // Filled triangle (arrowhead) — base at start, tip at end.
        var corners = array<vec2<f32>, 6>(
            start + normal,
            start - normal,
            end,
            end,
            end,
            end,
        );
        pixel = corners[vi];
    }

    let ndc_x = (pixel.x / viewport.size.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (pixel.y / viewport.size.y) * 2.0;

    var out: VertexOutput;
    out.clip_position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.color = line.color;
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
