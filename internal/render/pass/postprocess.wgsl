
struct Uniform {
    exposure: f32,
    bloom_intensity: f32,
    bloom_enabled: f32,
    ssao_enabled: f32,
    auto_exposure_enabled: f32,
    auto_exposure_target: f32,
    auto_exposure_min_scale: f32,
    auto_exposure_max_scale: f32,
};

struct AutoExposureBuffer {
    target_log_luminance:  f32,
    current_log_luminance: f32,
    adaptation_rate:       f32,
    delta_time:            f32,
    primed:                u32,
    _pad0:                 u32,
    _pad1:                 u32,
    _pad2:                 u32,
};

@group(0) @binding(0) var hdr_texture: texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler: sampler;
@group(0) @binding(2) var<uniform> u: Uniform;
@group(0) @binding(3) var bloom_texture: texture_2d<f32>;
@group(0) @binding(4) var bloom_sampler: sampler;
@group(0) @binding(5) var ssao_texture: texture_2d<f32>;
@group(0) @binding(6) var ssao_sampler: sampler;
@group(0) @binding(7) var<uniform> auto_exposure: AutoExposureBuffer;

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

fn linear_to_srgb(linear: vec3<f32>) -> vec3<f32> {
    let cutoff = step(linear, vec3<f32>(0.0031308));
    let lower = linear * 12.92;
    let higher = 1.055 * pow(max(linear, vec3<f32>(0.0)), vec3<f32>(1.0/2.4)) - 0.055;
    return mix(higher, lower, cutoff);
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let hdr = textureSample(hdr_texture, hdr_sampler, in.uv);
    var color = hdr.rgb;
    if u.ssao_enabled > 0.5 {
        let ao = textureSample(ssao_texture, ssao_sampler, in.uv).r;
        color = color * ao;
    }
    if u.bloom_enabled > 0.5 {
        let bloom = textureSample(bloom_texture, bloom_sampler, in.uv).rgb;
        color = color + bloom * u.bloom_intensity;
    }
    var effective_exposure = u.exposure;
    if u.auto_exposure_enabled > 0.5 {
        let avg_lum = exp2(auto_exposure.current_log_luminance);
        let target_lum = max(u.auto_exposure_target, 1e-4);
        let auto_scale = clamp(
            target_lum / max(avg_lum, 1e-4),
            u.auto_exposure_min_scale,
            u.auto_exposure_max_scale,
        );
        effective_exposure = effective_exposure * auto_scale;
    }
    let exposed = color * effective_exposure;
    let tonemapped = tonemap_aces(exposed);
    return vec4<f32>(linear_to_srgb(tonemapped), hdr.a);
}
