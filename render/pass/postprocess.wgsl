// Postprocess pass: HDR scene_color in, LDR ldr_color out.
// Applies an exposure scale and an ACES filmic tonemap. The
// pipeline can grow extra knobs (auto-exposure, bloom composite,
// color grading) by adding more uniform fields and shader steps;
// the bind layout already routes a uniform buffer at binding 2.

struct Uniform {
    exposure: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var hdr_texture: texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler: sampler;
@group(0) @binding(2) var<uniform> u: Uniform;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let uv = vec2<f32>(
        f32((vertex_index << 1u) & 2u),
        f32(vertex_index & 2u),
    );
    let clip = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    var out: VertexOutput;
    out.clip_position = clip;
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

fn tonemap_aces(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let hdr = textureSample(hdr_texture, hdr_sampler, in.uv);
    let exposed = hdr.rgb * u.exposure;
    return vec4<f32>(tonemap_aces(exposed), hdr.a);
}
