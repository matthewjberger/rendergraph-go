struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct FxaaParams {
    texel_size: vec2<f32>,
    enabled: f32,
    subpixel_quality: f32,
};

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;
@group(0) @binding(2) var<uniform> params: FxaaParams;

@vertex
fn vertex_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32((vertex_index & 1u) << 1u);
    let y = f32((vertex_index & 2u));
    out.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

const EDGE_THRESHOLD_MIN: f32 = 0.0312;
const EDGE_THRESHOLD_MAX: f32 = 0.125;
const ITERATIONS: i32 = 12;
const QUALITY_0: f32 = 1.0;
const QUALITY_1: f32 = 1.0;
const QUALITY_2: f32 = 1.0;
const QUALITY_3: f32 = 1.0;
const QUALITY_4: f32 = 1.0;
const QUALITY_5: f32 = 1.5;
const QUALITY_6: f32 = 2.0;
const QUALITY_7: f32 = 2.0;
const QUALITY_8: f32 = 2.0;
const QUALITY_9: f32 = 2.0;
const QUALITY_10: f32 = 4.0;
const QUALITY_11: f32 = 8.0;

fn luma(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.299, 0.587, 0.114));
}

fn quality(index: i32) -> f32 {
    switch index {
        case 0  { return QUALITY_0; }
        case 1  { return QUALITY_1; }
        case 2  { return QUALITY_2; }
        case 3  { return QUALITY_3; }
        case 4  { return QUALITY_4; }
        case 5  { return QUALITY_5; }
        case 6  { return QUALITY_6; }
        case 7  { return QUALITY_7; }
        case 8  { return QUALITY_8; }
        case 9  { return QUALITY_9; }
        case 10 { return QUALITY_10; }
        default { return QUALITY_11; }
    }
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let center_color = textureSampleLevel(input_texture, input_sampler, in.uv, 0.0);

    if params.enabled < 0.5 {
        return center_color;
    }

    let texel = params.texel_size;
    let luma_center = luma(center_color.rgb);

    let luma_d = luma(textureSampleLevel(input_texture, input_sampler, in.uv + vec2<f32>(0.0, texel.y), 0.0).rgb);
    let luma_u = luma(textureSampleLevel(input_texture, input_sampler, in.uv + vec2<f32>(0.0, -texel.y), 0.0).rgb);
    let luma_l = luma(textureSampleLevel(input_texture, input_sampler, in.uv + vec2<f32>(-texel.x, 0.0), 0.0).rgb);
    let luma_r = luma(textureSampleLevel(input_texture, input_sampler, in.uv + vec2<f32>(texel.x, 0.0), 0.0).rgb);

    let luma_min = min(luma_center, min(min(luma_d, luma_u), min(luma_l, luma_r)));
    let luma_max = max(luma_center, max(max(luma_d, luma_u), max(luma_l, luma_r)));
    let luma_range = luma_max - luma_min;

    if luma_range < max(EDGE_THRESHOLD_MIN, luma_max * EDGE_THRESHOLD_MAX) {
        return center_color;
    }

    let luma_dl = luma(textureSampleLevel(input_texture, input_sampler, in.uv + vec2<f32>(-texel.x, texel.y), 0.0).rgb);
    let luma_dr = luma(textureSampleLevel(input_texture, input_sampler, in.uv + vec2<f32>(texel.x, texel.y), 0.0).rgb);
    let luma_ul = luma(textureSampleLevel(input_texture, input_sampler, in.uv + vec2<f32>(-texel.x, -texel.y), 0.0).rgb);
    let luma_ur = luma(textureSampleLevel(input_texture, input_sampler, in.uv + vec2<f32>(texel.x, -texel.y), 0.0).rgb);

    let luma_du = luma_d + luma_u;
    let luma_lr = luma_l + luma_r;

    let luma_left_corners = luma_dl + luma_ul;
    let luma_down_corners = luma_dl + luma_dr;
    let luma_right_corners = luma_dr + luma_ur;
    let luma_up_corners = luma_ul + luma_ur;

    let edge_horizontal = abs(-2.0 * luma_l + luma_left_corners) +
                          abs(-2.0 * luma_center + luma_du) * 2.0 +
                          abs(-2.0 * luma_r + luma_right_corners);
    let edge_vertical = abs(-2.0 * luma_u + luma_up_corners) +
                        abs(-2.0 * luma_center + luma_lr) * 2.0 +
                        abs(-2.0 * luma_d + luma_down_corners);

    let is_horizontal = edge_horizontal >= edge_vertical;

    var luma_negative: f32;
    var luma_positive: f32;
    if is_horizontal {
        luma_negative = luma_u;
        luma_positive = luma_d;
    } else {
        luma_negative = luma_l;
        luma_positive = luma_r;
    }

    let gradient_negative = abs(luma_negative - luma_center);
    let gradient_positive = abs(luma_positive - luma_center);

    let is_negative = gradient_negative >= gradient_positive;

    var step_length: f32;
    var luma_local_average: f32;
    var gradient: f32;
    if is_horizontal {
        step_length = texel.y;
    } else {
        step_length = texel.x;
    }

    if is_negative {
        step_length = -step_length;
        luma_local_average = 0.5 * (luma_negative + luma_center);
        gradient = gradient_negative;
    } else {
        luma_local_average = 0.5 * (luma_positive + luma_center);
        gradient = gradient_positive;
    }

    var current_uv = in.uv;
    if is_horizontal {
        current_uv.y += step_length * 0.5;
    } else {
        current_uv.x += step_length * 0.5;
    }

    var uv_offset: vec2<f32>;
    if is_horizontal {
        uv_offset = vec2<f32>(texel.x, 0.0);
    } else {
        uv_offset = vec2<f32>(0.0, texel.y);
    }

    var uv_neg = current_uv - uv_offset * quality(0);
    var uv_pos = current_uv + uv_offset * quality(0);

    let scaled_gradient = gradient * 0.25;

    var luma_end_neg = luma(textureSampleLevel(input_texture, input_sampler, uv_neg, 0.0).rgb) - luma_local_average;
    var luma_end_pos = luma(textureSampleLevel(input_texture, input_sampler, uv_pos, 0.0).rgb) - luma_local_average;

    var reached_neg = abs(luma_end_neg) >= scaled_gradient;
    var reached_pos = abs(luma_end_pos) >= scaled_gradient;
    var reached_both = reached_neg && reached_pos;

    if !reached_neg {
        uv_neg -= uv_offset * quality(1);
    }
    if !reached_pos {
        uv_pos += uv_offset * quality(1);
    }

    if !reached_both {
        for (var iteration = 2; iteration < ITERATIONS; iteration++) {
            if !reached_neg {
                luma_end_neg = luma(textureSampleLevel(input_texture, input_sampler, uv_neg, 0.0).rgb) - luma_local_average;
            }
            if !reached_pos {
                luma_end_pos = luma(textureSampleLevel(input_texture, input_sampler, uv_pos, 0.0).rgb) - luma_local_average;
            }

            reached_neg = abs(luma_end_neg) >= scaled_gradient;
            reached_pos = abs(luma_end_pos) >= scaled_gradient;
            reached_both = reached_neg && reached_pos;

            if !reached_neg {
                uv_neg -= uv_offset * quality(iteration);
            }
            if !reached_pos {
                uv_pos += uv_offset * quality(iteration);
            }

            if reached_both {
                break;
            }
        }
    }

    var distance_neg: f32;
    var distance_pos: f32;
    if is_horizontal {
        distance_neg = in.uv.x - uv_neg.x;
        distance_pos = uv_pos.x - in.uv.x;
    } else {
        distance_neg = in.uv.y - uv_neg.y;
        distance_pos = uv_pos.y - in.uv.y;
    }

    let is_closest_neg = distance_neg < distance_pos;
    let distance_final = min(distance_neg, distance_pos);

    let edge_length = distance_neg + distance_pos;
    var edge_blend = 0.5 - distance_final / edge_length;

    let is_luma_center_smaller = luma_center < luma_local_average;
    let correct_variation_neg = (luma_end_neg < 0.0) != is_luma_center_smaller;
    let correct_variation_pos = (luma_end_pos < 0.0) != is_luma_center_smaller;
    let correct_variation = select(correct_variation_pos, correct_variation_neg, is_closest_neg);

    if !correct_variation {
        edge_blend = 0.0;
    }

    let subpixel_luma_range = abs((2.0 * (luma_d + luma_u + luma_l + luma_r) +
                                   (luma_dl + luma_dr + luma_ul + luma_ur)) / 12.0 - luma_center);
    let subpixel_offset = clamp(subpixel_luma_range / luma_range, 0.0, 1.0);
    let subpixel_offset_final = (-2.0 * subpixel_offset + 3.0) * subpixel_offset * subpixel_offset;
    let subpixel_blend = subpixel_offset_final * subpixel_offset_final * params.subpixel_quality;

    let final_blend = max(edge_blend, subpixel_blend);

    var final_uv = in.uv;
    if is_horizontal {
        final_uv.y += step_length * final_blend;
    } else {
        final_uv.x += step_length * final_blend;
    }

    return textureSampleLevel(input_texture, input_sampler, final_uv, 0.0);
}
