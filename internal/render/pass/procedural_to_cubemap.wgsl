
struct ProceduralUniform {
    output_size: u32,
    time:        f32,
    _pad0:       u32,
    _pad1:       u32,
};

@group(0) @binding(0)
var<uniform> params: ProceduralUniform;

@group(0) @binding(1)
var output_texture: texture_storage_2d_array<rgba16float, write>;

fn cube_to_world(face: u32, uv: vec2<f32>) -> vec3<f32> {
    var dir: vec3<f32>;
    let x = 2.0 * uv.x - 1.0;
    let y = 2.0 * uv.y - 1.0;

    switch face {
        case 0u: {
            dir = vec3<f32>(1.0, -y, -x);
        }
        case 1u: {
            dir = vec3<f32>(-1.0, -y, x);
        }
        case 2u: {
            dir = vec3<f32>(x, 1.0, y);
        }
        case 3u: {
            dir = vec3<f32>(x, -1.0, -y);
        }
        case 4u: {
            dir = vec3<f32>(x, -y, 1.0);
        }
        default: {
            dir = vec3<f32>(-x, -y, -1.0);
        }
    }
    return normalize(dir);
}

fn mod289_3(x: vec3<f32>) -> vec3<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn mod289_4(x: vec4<f32>) -> vec4<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn permute(x: vec4<f32>) -> vec4<f32> {
    return mod289_4(((x * 34.0) + 1.0) * x);
}

fn taylor_inv_sqrt(r: vec4<f32>) -> vec4<f32> {
    return 1.79284291400159 - 0.85373472095314 * r;
}

fn snoise(v: vec3<f32>) -> f32 {
    let C = vec2<f32>(1.0 / 6.0, 1.0 / 3.0);
    let D = vec4<f32>(0.0, 0.5, 1.0, 2.0);

    var i = floor(v + dot(v, vec3<f32>(C.y, C.y, C.y)));
    let x0 = v - i + dot(i, vec3<f32>(C.x, C.x, C.x));

    let g = step(x0.yzx, x0.xyz);
    let l = 1.0 - g;
    let i1 = min(g.xyz, l.zxy);
    let i2 = max(g.xyz, l.zxy);

    let x1 = x0 - i1 + C.x;
    let x2 = x0 - i2 + C.y;
    let x3 = x0 - D.yyy;

    i = mod289_3(i);
    let p = permute(permute(permute(
        i.z + vec4<f32>(0.0, i1.z, i2.z, 1.0))
        + i.y + vec4<f32>(0.0, i1.y, i2.y, 1.0))
        + i.x + vec4<f32>(0.0, i1.x, i2.x, 1.0));

    let n_ = 0.142857142857;
    let ns = n_ * D.wyz - D.xzx;

    let j = p - 49.0 * floor(p * ns.z * ns.z);

    let x_ = floor(j * ns.z);
    let y_ = floor(j - 7.0 * x_);

    let x = x_ * ns.x + ns.y;
    let y = y_ * ns.x + ns.y;
    let h = 1.0 - abs(x) - abs(y);

    let b0 = vec4<f32>(x.xy, y.xy);
    let b1 = vec4<f32>(x.zw, y.zw);

    let s0 = floor(b0) * 2.0 + 1.0;
    let s1 = floor(b1) * 2.0 + 1.0;
    let sh = -step(h, vec4<f32>(0.0, 0.0, 0.0, 0.0));

    let a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    let a1 = b1.xzyw + s1.xzyw * sh.zzww;

    var p0 = vec3<f32>(a0.xy, h.x);
    var p1 = vec3<f32>(a0.zw, h.y);
    var p2 = vec3<f32>(a1.xy, h.z);
    var p3 = vec3<f32>(a1.zw, h.w);

    let norm = taylor_inv_sqrt(vec4<f32>(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    var m = max(0.6 - vec4<f32>(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), vec4<f32>(0.0));
    m = m * m;
    return 42.0 * dot(m * m, vec4<f32>(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

fn fbm(p: vec3<f32>, octaves: i32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;

    for (var index = 0; index < octaves; index++) {
        value += amplitude * (snoise(p * frequency) * 0.5 + 0.5);
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    return value;
}

fn apply_clouds(base: vec3<f32>, dir: vec3<f32>, sun_direction: vec3<f32>, time: f32) -> vec3<f32> {
    let height = dir.y;
    if (height <= 0.0) {
        return base;
    }

    let cloud_drift = vec3<f32>(time * 0.012, 0.0, time * 0.006);
    let cloud_pos = dir * 2.5 + cloud_drift;

    let cloud_base = fbm(cloud_pos * 0.8, 5);
    let cloud_detail = fbm(cloud_pos * 2.0, 4) * 0.5;
    let cloud_density = cloud_base + cloud_detail * 0.4;

    let cloud_coverage = 0.62;
    let cloud_shape = smoothstep(cloud_coverage, cloud_coverage + 0.2, cloud_density);

    let height_fade = smoothstep(0.15, 0.45, height) * (1.0 - smoothstep(0.7, 0.95, height));
    let cloud_amount = cloud_shape * height_fade;

    let cloud_sun_lit = max(dot(dir, sun_direction), 0.0);
    let light_factor = pow(cloud_sun_lit, 1.2);

    let cloud_shadow = vec3<f32>(0.65, 0.7, 0.78);
    let cloud_lit = vec3<f32>(1.0, 0.99, 0.97);
    let cloud_bright = vec3<f32>(1.0, 1.0, 0.98);

    var cloud_color = mix(cloud_shadow, cloud_lit, smoothstep(0.0, 0.5, light_factor));
    cloud_color = mix(cloud_color, cloud_bright, smoothstep(0.5, 1.0, light_factor) * 0.5);

    let cloud_thickness = smoothstep(cloud_coverage, cloud_coverage + 0.35, cloud_density);
    cloud_color = mix(cloud_color, cloud_shadow, cloud_thickness * 0.3);

    return mix(base, cloud_color, cloud_amount * 0.85);
}

fn sky_color(dir: vec3<f32>, time: f32) -> vec3<f32> {
    let sky_top_color = vec3<f32>(0.385, 0.454, 0.55);
    let sky_horizon_color = vec3<f32>(0.646, 0.656, 0.67);
    let ground_horizon_color = vec3<f32>(0.646, 0.656, 0.67);
    let ground_bottom_color = vec3<f32>(0.2, 0.169, 0.133);

    let height = dir.y;

    let sky_curve = 0.15;
    let ground_curve = 0.02;

    var color: vec3<f32>;
    if (height > 0.0) {
        let t = 1.0 - pow(1.0 - height, 1.0 / sky_curve);
        color = mix(sky_horizon_color, sky_top_color, clamp(t, 0.0, 1.0));
    } else {
        let t = 1.0 - pow(1.0 + height, 1.0 / ground_curve);
        color = mix(ground_horizon_color, ground_bottom_color, clamp(t, 0.0, 1.0));
    }

    color = color * 1.3;

    let sun_direction = normalize(vec3<f32>(0.0, 0.5, -1.0));
    let sun_angle = acos(clamp(dot(dir, sun_direction), -1.0, 1.0));
    let sun_disk = 1.0 - smoothstep(0.0, 0.02, sun_angle);
    let sun_color = vec3<f32>(1.0, 0.95, 0.8);
    color = mix(color, sun_color, sun_disk * 0.5);

    return apply_clouds(color, dir, sun_direction, time);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = global_id.xy;
    let face = global_id.z;

    if (coords.x >= params.output_size || coords.y >= params.output_size || face >= 6u) {
        return;
    }

    let uv = vec2<f32>(
        (f32(coords.x) + 0.5) / f32(params.output_size),
        (f32(coords.y) + 0.5) / f32(params.output_size)
    );

    let direction = cube_to_world(face, uv);
    let color = sky_color(direction, params.time);

    textureStore(output_texture, coords, face, vec4<f32>(color, 1.0));
}
