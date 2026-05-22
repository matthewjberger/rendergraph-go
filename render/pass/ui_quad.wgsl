struct UiQuadInstance {
    rect: vec4<f32>,
    color: vec4<f32>,
};

struct Viewport {
    size: vec2<f32>,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> viewport: Viewport;
@group(0) @binding(1) var<storage, read> quads: array<UiQuadInstance>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
    let quad = quads[ii];

    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0),
    );
    let corner = corners[vi];

    let pixel_x = quad.rect.x + corner.x * quad.rect.z;
    let pixel_y = quad.rect.y + corner.y * quad.rect.w;

    let ndc_x = (pixel_x / viewport.size.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (pixel_y / viewport.size.y) * 2.0;

    var out: VertexOutput;
    out.clip_position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.color = quad.color;
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
