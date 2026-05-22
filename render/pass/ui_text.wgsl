struct UiGlyphInstance {
    rect: vec4<f32>,
    color: vec4<f32>,
    atlas: vec4<f32>,
};

struct Viewport {
    size: vec2<f32>,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> viewport: Viewport;
@group(0) @binding(1) var<storage, read> glyphs: array<UiGlyphInstance>;
@group(0) @binding(2) var atlas_tex: texture_2d<f32>;
@group(0) @binding(3) var atlas_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
    let glyph = glyphs[ii];

    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0),
    );
    let corner = corners[vi];

    let pixel_x = glyph.rect.x + corner.x * glyph.rect.z;
    let pixel_y = glyph.rect.y + corner.y * glyph.rect.w;

    let ndc_x = (pixel_x / viewport.size.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (pixel_y / viewport.size.y) * 2.0;

    let u = (glyph.atlas.x + corner.x * glyph.atlas.z) / glyph.atlas.w;
    let v = corner.y;

    var out: VertexOutput;
    out.clip_position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv = vec2<f32>(u, v);
    out.color = glyph.color;
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let coverage = textureSample(atlas_tex, atlas_sampler, in.uv).r;
    return vec4<f32>(in.color.rgb, in.color.a * coverage);
}
