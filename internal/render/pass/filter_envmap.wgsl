
const PI: f32 = 3.14159265358979323846;

const DISTRIBUTION_LAMBERTIAN: u32 = 0u;
const DISTRIBUTION_GGX: u32 = 1u;

struct FilterParams {
    face: u32,
    output_size: u32,
    roughness: f32,
    sample_count: u32,
    source_width: u32,
    source_height: u32,
    distribution: u32,
    source_mip_count: u32,
}

@group(0) @binding(0)
var source_cubemap: texture_cube<f32>;

@group(0) @binding(1)
var source_sampler: sampler;

@group(0) @binding(2)
var output_texture: texture_storage_2d_array<rgba16float, write>;

@group(0) @binding(3)
var<uniform> params: FilterParams;

fn hammersley_2d(index: u32, num_samples: u32) -> vec2<f32> {
    var bits = index;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    let rdi = f32(bits) * 2.3283064365386963e-10;
    return vec2<f32>(f32(index) / f32(num_samples), rdi);
}

fn cube_to_world(face: u32, uv: vec2<f32>) -> vec3<f32> {
    let x = 2.0 * uv.x - 1.0;
    let y = 2.0 * uv.y - 1.0;

    var dir: vec3<f32>;
    switch face {
        case 0u: { dir = vec3<f32>(1.0, -y, -x); }
        case 1u: { dir = vec3<f32>(-1.0, -y, x); }
        case 2u: { dir = vec3<f32>(x, 1.0, y); }
        case 3u: { dir = vec3<f32>(x, -1.0, -y); }
        case 4u: { dir = vec3<f32>(x, -y, 1.0); }
        default: { dir = vec3<f32>(-x, -y, -1.0); }
    }
    return normalize(dir);
}

fn orthonormal_basis(n: vec3<f32>) -> mat3x3<f32> {
    let sign = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let tangent = vec3<f32>(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let bitangent = vec3<f32>(b, sign + n.y * n.y * a, -n.y);
    return mat3x3<f32>(tangent, bitangent, n);
}

fn importance_sample_cosine(xi: vec2<f32>, normal: vec3<f32>) -> vec3<f32> {
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt(1.0 - xi.y);
    let sin_theta = sqrt(xi.y);

    let local_dir = vec3<f32>(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta
    );

    let tbn = orthonormal_basis(normal);
    return normalize(tbn * local_dir);
}

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32(seed: u32) -> f32 {
    return f32(pcg_hash(seed)) / 4294967295.0;
}

fn rand_vec2(seed: u32) -> vec2<f32> {
    return vec2<f32>(rand_f32(seed), rand_f32(seed + 1u));
}

fn D_GGX_prefilter(dotNH: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let denom = dotNH * dotNH * (alpha2 - 1.0) + 1.0;
    return alpha2 / (PI * denom * denom);
}

fn importance_sample_ggx(xi: vec2<f32>, roughness: f32, normal: vec3<f32>) -> vec3<f32> {
    let alpha = roughness * roughness;
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (alpha * alpha - 1.0) * xi.y));
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    let local_dir = vec3<f32>(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta
    );

    let tbn = orthonormal_basis(normal);
    return normalize(tbn * local_dir);
}

fn filter_irradiance(normal: vec3<f32>, pixel_seed: u32) -> vec3<f32> {
    var color = vec3<f32>(0.0);
    let num_samples = params.sample_count;
    let random_offset = rand_vec2(pixel_seed);

    for (var sample_index = 0u; sample_index < num_samples; sample_index = sample_index + 1u) {
        var xi = hammersley_2d(sample_index, num_samples);
        xi = fract(xi + random_offset);
        let sample_dir = importance_sample_cosine(xi, normal);
        let sample_color = textureSampleLevel(source_cubemap, source_sampler, sample_dir, 0.0).rgb;
        color = color + min(sample_color, vec3<f32>(3.0));
    }

    return color / f32(num_samples);
}

fn prefilter_env_map(normal: vec3<f32>, roughness: f32, pixel_seed: u32) -> vec3<f32> {
    if (roughness == 0.0) {
        return textureSampleLevel(source_cubemap, source_sampler, normal, 0.0).rgb;
    }

    let n = normal;
    let v = normal;

    var color = vec3<f32>(0.0);
    var total_weight = 0.0;
    let num_samples = params.sample_count;
    let clamped_roughness = max(roughness, 0.001);
    let random_offset = rand_vec2(pixel_seed);

    let env_map_dim = f32(params.source_width);

    for (var sample_index = 0u; sample_index < num_samples; sample_index = sample_index + 1u) {
        var xi = hammersley_2d(sample_index, num_samples);
        xi = fract(xi + random_offset);
        let h = importance_sample_ggx(xi, clamped_roughness, n);
        let l = normalize(2.0 * dot(v, h) * h - v);

        let dot_nl = clamp(dot(n, l), 0.0, 1.0);
        if (dot_nl > 0.0) {
            let dot_nh = clamp(dot(n, h), 0.0, 1.0);
            let dot_vh = clamp(dot(v, h), 0.0, 1.0);

            let pdf = D_GGX_prefilter(dot_nh, clamped_roughness) * dot_nh / (4.0 * dot_vh) + 0.0001;
            let omega_s = 1.0 / (f32(num_samples) * pdf);
            let omega_p = 4.0 * PI / (6.0 * env_map_dim * env_map_dim);
            var mip_level = 0.0;
            if (params.roughness > 0.0) {
                mip_level = max(0.5 * log2(omega_s / omega_p) + 1.0, 0.0);
            }

            let sample_color = textureSampleLevel(source_cubemap, source_sampler, l, mip_level).rgb;
            color = color + min(sample_color, vec3<f32>(4.0)) * dot_nl;
            total_weight = total_weight + dot_nl;
        }
    }

    if (total_weight > 0.0) {
        return color / total_weight;
    }
    return textureSampleLevel(source_cubemap, source_sampler, normal, 0.0).rgb;
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
    let pixel_seed = (coords.x * 2131358057u) ^ (coords.y * 3416869721u) ^ (face * 1199786941u);

    var color: vec3<f32>;
    if (params.distribution == DISTRIBUTION_LAMBERTIAN) {
        color = filter_irradiance(direction, pixel_seed);
    } else {
        color = prefilter_env_map(direction, params.roughness, pixel_seed);
    }

    textureStore(output_texture, coords, face, vec4<f32>(color, 1.0));
}
