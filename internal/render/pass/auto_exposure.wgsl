struct Output {
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
@group(0) @binding(1) var<storage, read_write> output: Output;

const WORKGROUP_SIZE: u32 = 256u;
var<workgroup> shared_lum: array<f32, 256>;

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let dims = textureDimensions(hdr_texture, 0);
    let stride_int = 16u;
    let row = lid.x / stride_int;
    let col = lid.x % stride_int;
    let cell_w = max(dims.x / stride_int, 1u);
    let cell_h = max(dims.y / stride_int, 1u);
    let x = min(col * cell_w + cell_w / 2u, dims.x - 1u);
    let y = min(row * cell_h + cell_h / 2u, dims.y - 1u);
    let color = textureLoad(hdr_texture, vec2<i32>(i32(x), i32(y)), 0).rgb;
    let lum = max(luminance(color), 1e-4);

    shared_lum[lid.x] = clamp(log2(lum), -10.0, 4.0);
    workgroupBarrier();

    var stride2 = WORKGROUP_SIZE / 2u;
    loop {
        if (stride2 == 0u) {
            break;
        }
        if (lid.x < stride2) {
            shared_lum[lid.x] = shared_lum[lid.x] + shared_lum[lid.x + stride2];
        }
        workgroupBarrier();
        stride2 = stride2 / 2u;
    }

    if (lid.x == 0u) {
        let mean_log_lum = shared_lum[0] / f32(WORKGROUP_SIZE);
        let alpha = 1.0 - exp(-output.adaptation_rate * output.delta_time);
        var new_current = mean_log_lum;
        if (output.primed != 0u) {
            new_current = output.current_log_luminance + (mean_log_lum - output.current_log_luminance) * alpha;
        }
        output.target_log_luminance = mean_log_lum;
        output.current_log_luminance = new_current;
        output.primed = 1u;
    }
}
