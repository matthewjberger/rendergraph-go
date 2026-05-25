struct OitOutput {
    @location(0) accum:     vec4<f32>,
    @location(1) reveal:    f32,
    @location(2) entity_id: u32,
};
fn oit_weight(view_z: f32, a: f32) -> f32 {
    let z_scale = max(cluster_uniforms.z_far * 0.2, 1.0);
    let z_ratio = view_z / z_scale;
    return a * clamp(0.03 / (1e-5 + pow(z_ratio, 4.0)), 1e-2, 3e3);
}
fn oit_output(color: vec3<f32>, alpha: f32, view_z: f32, entity_id: u32) -> OitOutput {
    let w = oit_weight(view_z, alpha);
    var out: OitOutput;
    out.accum = vec4<f32>(color * alpha, alpha) * w;
    out.reveal = alpha;
    out.entity_id = entity_id;
    return out;
}
@fragment
fn fragment_main(in: VertexOutput, @builtin(front_facing) front_facing: bool) -> OitOutput {
    let mat = materials[in.material_index];
    if (mat.double_sided == 0u && !front_facing) {
        discard;
    }

    let view_dir = normalize(cluster_uniforms.camera_position.xyz - in.world_pos);

    let derived_normal = normalize(cross(dpdx(in.world_pos), dpdy(in.world_pos)));
    let n_len = length(in.world_normal);
    var geom_normal: vec3<f32>;
    if (n_len > 0.001) {
        geom_normal = in.world_normal / n_len;
    } else {
        geom_normal = derived_normal;
    }
    var geom_tangent = in.world_tangent;

    if (dot(geom_normal, view_dir) < 0.0) {
        geom_normal = -geom_normal;
        geom_tangent = -geom_tangent;
    }

    var normal_sample = vec3<f32>(0.5, 0.5, 1.0);
    let has_normal_texture = mat.normal_layer != NO_LAYER;
    if (has_normal_texture) {
        normal_sample = sample_linear_layer(mat.normal_layer, texture_uv(in.uv, mat.normal_transform)).xyz;
    }
    var normal = get_normal(geom_normal, geom_tangent, normal_sample, has_normal_texture, mat.normal_scale, 0u);
    if (cluster_uniforms.flat_color.a > 0.0) {
        normal = geom_normal;
    }

    var albedo_sample = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    if (mat.base_layer != NO_LAYER) {
        albedo_sample = sample_srgb_layer(mat.base_layer, texture_uv(in.uv, mat.base_transform));
    }
    if (cluster_uniforms.flat_color.a >= 2.0) {
        discard;
    }
    var base_color = mat.base_color * albedo_sample * in.color;
    if (cluster_uniforms.flat_color.a > 0.0) {
        base_color = vec4<f32>(cluster_uniforms.flat_color.rgb, base_color.a);
    }
    let albedo = base_color.rgb;

    if (mat.alpha_mode != 2u && mat.transmission_factor <= 0.0) {
        discard;
    }
    if (mat.alpha_mode == 1u && base_color.a < mat.alpha_cutoff) {
        discard;
    }
    if (mat.transmission_factor <= 0.0 && base_color.a < 0.0039) {
        discard;
    }

    var metallic = mat.metallic_factor;
    var roughness = mat.roughness_factor;
    if (mat.metallic_roughness_layer != NO_LAYER) {
        let mr = sample_linear_layer(mat.metallic_roughness_layer, texture_uv(in.uv, mat.metallic_roughness_transform));
        roughness = roughness * mr.g;
        metallic = metallic * mr.b;
    }
    roughness = clamp(roughness, 0.04, 1.0);
    metallic = clamp(metallic, 0.0, 1.0);
    if (cluster_uniforms.flat_color.a > 0.0) {
        metallic = 0.0;
        roughness = 1.0;
    }

    var occlusion = 1.0;
    if (mat.occlusion_layer != NO_LAYER) {
        occlusion = sample_linear_layer(mat.occlusion_layer, texture_uv(in.uv, mat.occlusion_transform)).r;
    }

    var emissive = mat.emissive_factor * mat.emissive_strength;
    if (mat.emissive_layer != NO_LAYER) {
        emissive = emissive * sample_srgb_layer(mat.emissive_layer, texture_uv(in.uv, mat.emissive_transform)).rgb;
    }
    if (cluster_uniforms.flat_color.a > 0.0) {
        emissive = vec3<f32>(0.0);
    }

    if (mat.unlit != 0u || cluster_uniforms.global_unlit > 0.5) {
        var unlit_alpha = base_color.a;
        if (mat.transmission_factor > 0.0) {
            unlit_alpha = 1.0;
        }
        let unlit_view_pos = view_matrix * vec4<f32>(in.world_pos, 1.0);
        return oit_output(albedo + emissive, unlit_alpha, -unlit_view_pos.z, in.entity_id);
    }

    let v = view_dir;
    let n = normal;

    var specular_color = mat.specular_color_factor;
    if (mat.specular_color_layer != NO_LAYER) {
        specular_color = specular_color * sample_srgb_layer(mat.specular_color_layer, texture_uv(in.uv, mat.specular_color_transform)).rgb;
    }
    var specular_strength = mat.specular_factor;
    if (mat.specular_layer != NO_LAYER) {
        specular_strength = specular_strength * sample_linear_layer(mat.specular_layer, texture_uv(in.uv, mat.specular_transform)).a;
    }
    let dielectric_f0 = pow((mat.ior - 1.0) / (mat.ior + 1.0), 2.0);
    let dielectric_specular_f0 = min(vec3<f32>(dielectric_f0) * specular_color, vec3<f32>(1.0));
    var f0 = mix(dielectric_specular_f0, albedo, metallic);

    if (mat.iridescence_factor > 0.0) {
        var irid_factor = mat.iridescence_factor;
        if (mat.iridescence_layer != NO_LAYER) {
            irid_factor = irid_factor * sample_linear_layer(mat.iridescence_layer, texture_uv(in.uv, mat.iridescence_transform)).r;
        }
        var irid_thickness = mat.iridescence_thickness_max;
        if (mat.iridescence_thickness_layer != NO_LAYER) {
            let t = sample_linear_layer(mat.iridescence_thickness_layer, texture_uv(in.uv, mat.iridescence_thickness_transform)).g;
            irid_thickness = mix(mat.iridescence_thickness_min, mat.iridescence_thickness_max, t);
        }
        if (irid_factor > 0.0 && irid_thickness > 0.0) {
            let irid_f0 = eval_iridescence(1.0, mat.iridescence_ior, max(dot(n, v), 0.0), irid_thickness, f0);
            f0 = mix(f0, irid_f0, irid_factor);
        }
    }

    var shade: ShadeParams;
    shade.aniso_strength = mat.anisotropy_strength;
    var aniso_dir = vec2<f32>(mat.anisotropy_rotation_cos, mat.anisotropy_rotation_sin);
    if (mat.anisotropy_layer != NO_LAYER) {
        let a = sample_linear_layer(mat.anisotropy_layer, texture_uv(in.uv, mat.anisotropy_transform));
        let td = a.xy * 2.0 - 1.0;
        aniso_dir = vec2<f32>(
            mat.anisotropy_rotation_cos * td.x - mat.anisotropy_rotation_sin * td.y,
            mat.anisotropy_rotation_sin * td.x + mat.anisotropy_rotation_cos * td.y);
        shade.aniso_strength = shade.aniso_strength * a.b;
    }
    let tangent_w = normalize(geom_tangent.xyz);
    let bitangent_w = cross(n, tangent_w) * geom_tangent.w;
    shade.aniso_t = aniso_dir.x * tangent_w + aniso_dir.y * bitangent_w;
    shade.aniso_b = cross(n, shade.aniso_t);

    var sheen_color = mat.sheen_color_factor;
    if (mat.sheen_color_layer != NO_LAYER) {
        sheen_color = sheen_color * sample_srgb_layer(mat.sheen_color_layer, texture_uv(in.uv, mat.sheen_color_transform)).rgb;
    }
    var sheen_rough = clamp(mat.sheen_roughness_factor, 0.07, 1.0);
    if (mat.sheen_roughness_layer != NO_LAYER) {
        sheen_rough = sheen_rough * sample_linear_layer(mat.sheen_roughness_layer, texture_uv(in.uv, mat.sheen_roughness_transform)).a;
    }
    shade.sheen_color = sheen_color;
    shade.sheen_roughness = sheen_rough;
    let sheen_max = max(sheen_color.r, max(sheen_color.g, sheen_color.b));
    let sheen_e = textureSampleLevel(brdf_lut, ibl_sampler, vec2<f32>(max(dot(n, v), 0.0), sheen_rough), 0.0).b;
    shade.sheen_scale = clamp(1.0 - sheen_max * sheen_e, 0.0, 1.0);

    var cc_factor = mat.clearcoat_factor;
    if (mat.clearcoat_layer != NO_LAYER) {
        cc_factor = cc_factor * sample_linear_layer(mat.clearcoat_layer, texture_uv(in.uv, mat.clearcoat_transform)).r;
    }
    var cc_roughness = clamp(mat.clearcoat_roughness_factor, 0.04, 1.0);
    if (mat.clearcoat_roughness_layer != NO_LAYER) {
        cc_roughness = cc_roughness * sample_linear_layer(mat.clearcoat_roughness_layer, texture_uv(in.uv, mat.clearcoat_roughness_transform)).g;
    }
    var cc_normal = n;
    if (mat.clearcoat_normal_layer != NO_LAYER) {
        let cc_n_sample = sample_linear_layer(mat.clearcoat_normal_layer, texture_uv(in.uv, mat.clearcoat_normal_transform)).xyz;
        let cc_n_tangent_xy = (cc_n_sample.xy * 2.0 - 1.0) * mat.clearcoat_normal_scale;
        let cc_n_tangent = vec3<f32>(cc_n_tangent_xy, sqrt(max(1.0 - dot(cc_n_tangent_xy, cc_n_tangent_xy), 0.0)));
        let cc_t = normalize(in.world_tangent.xyz);
        let cc_b = cross(in.world_normal, cc_t) * in.world_tangent.w;
        let cc_tbn = mat3x3<f32>(cc_t, cc_b, in.world_normal);
        cc_normal = normalize(cc_tbn * cc_n_tangent);
    }
    shade.cc_factor = cc_factor;
    shade.cc_roughness = cc_roughness;
    shade.cc_normal = cc_normal;

    var lo = vec3<f32>(0.0);

    let shadow_factor = sample_directional_shadow(in.world_pos, n);

    for (var i = 0u; i < cluster_uniforms.num_directional_lights; i = i + 1u) {
        let light = lights[i];
        let point_to_light = -light.direction.xyz;
        var contribution = shade_one_light(light, point_to_light, v, n, albedo, f0, metallic, roughness, shade);
        if (i == 0u) {
            contribution = contribution * shadow_factor;
        }
        lo = lo + contribution;
    }

    let view_pos = view_matrix * vec4<f32>(in.world_pos, 1.0);
    let view_depth = -view_pos.z;
    let cluster_idx = get_cluster_index(in.clip_position.xy, view_depth);
    let grid = light_grid[cluster_idx];
    let base = cluster_idx * MAX_LIGHTS_PER_CLUSTER;
    let cluster_count = min(grid.count, MAX_LIGHTS_PER_CLUSTER);
    for (var i = 0u; i < cluster_count; i = i + 1u) {
        let light_idx = light_indices[base + i];
        let light = lights[cluster_uniforms.num_directional_lights + light_idx];
        let point_to_light = light.position.xyz - in.world_pos;
        var contribution = shade_one_light(light, point_to_light, v, n, albedo, f0, metallic, roughness, shade);
        if (light.light_type == LIGHT_TYPE_SPOT && light.shadow_index >= 0) {
            contribution = contribution * sample_spot_shadow(in.world_pos, n, light.shadow_index);
        }
        if (light.light_type == LIGHT_TYPE_POINT && light.shadow_index >= 0) {
            contribution = contribution * sample_point_shadow(in.world_pos, n, light.shadow_index);
        }
        lo = lo + contribution;
    }

    let n_dot_v = max(dot(n, v), 0.0);
    let r = reflect(-v, n);
    let f_ibl = fresnel_schlick_roughness(n_dot_v, f0, roughness);
    let irradiance = textureSampleLevel(irradiance_map, ibl_sampler, n, 0.0).rgb;
    let prefiltered = textureSampleLevel(prefiltered_env, ibl_sampler, r, roughness * MAX_REFLECTION_LOD).rgb;
    let brdf = textureSampleLevel(brdf_lut, ibl_sampler, vec2<f32>(n_dot_v, roughness), 0.0).rg;
    let fss_ess = f_ibl * brdf.x + brdf.y;
    let ems = 1.0 - (brdf.x + brdf.y);
    let f_avg = f0 + (vec3<f32>(1.0) - f0) / 21.0;
    let fms_ems = ems * fss_ess * f_avg / (vec3<f32>(1.0) - f_avg * ems);
    let c_diff = albedo * (1.0 - metallic);
    let kd_ibl = c_diff * (vec3<f32>(1.0) - fss_ess - fms_ems);
    var diffuse_ibl = (fms_ems + kd_ibl) * irradiance;

    if (mat.diffuse_transmission_factor > 0.0) {
        var dt_factor = mat.diffuse_transmission_factor;
        var dt_color = mat.diffuse_transmission_color_factor;
        if (mat.diffuse_transmission_color_layer != NO_LAYER) {
            dt_color = dt_color * sample_srgb_layer(mat.diffuse_transmission_color_layer, texture_uv(in.uv, mat.diffuse_transmission_color_transform)).rgb;
        }
        let back_irradiance = textureSampleLevel(irradiance_map, ibl_sampler, -n, 0.0).rgb;
        let c_diff_back = dt_color * albedo * (1.0 - metallic);
        let dt_diffuse = (fms_ems + c_diff_back * (vec3<f32>(1.0) - fss_ess - fms_ems)) * back_irradiance;
        diffuse_ibl = mix(diffuse_ibl, dt_diffuse, dt_factor);
    }

    if (mat.transmission_factor > 0.0) {
        var trans = mat.transmission_factor;
        if (mat.transmission_layer != NO_LAYER) {
            trans = trans * sample_linear_layer(mat.transmission_layer, texture_uv(in.uv, mat.transmission_transform)).r;
        }
        var thickness = mat.thickness;
        if (mat.thickness_layer != NO_LAYER) {
            thickness = thickness * sample_linear_layer(mat.thickness_layer, texture_uv(in.uv, mat.thickness_transform)).g;
        }
        let transmission = get_ibl_volume_refraction(
            n, v, in.world_pos, roughness, albedo, f0, mat.ior, mat.dispersion,
            thickness, mat.attenuation_color, mat.attenuation_distance,
            vec3<f32>(in.world_scale_factor),
        );
        let transmission_attenuated = transmission * (vec3<f32>(1.0) - fss_ess);
        diffuse_ibl = mix(diffuse_ibl, transmission_attenuated, trans);
    }

    let specular_ibl = prefiltered * fss_ess * specular_strength;
    var ambient = diffuse_ibl + specular_ibl;

    if (mat.clearcoat_factor > 0.0) {
        var cc = mat.clearcoat_factor;
        if (mat.clearcoat_layer != NO_LAYER) {
            cc = cc * sample_linear_layer(mat.clearcoat_layer, texture_uv(in.uv, mat.clearcoat_transform)).r;
        }
        var cc_rough = clamp(mat.clearcoat_roughness_factor, 0.04, 1.0);
        if (mat.clearcoat_roughness_layer != NO_LAYER) {
            cc_rough = cc_rough * sample_linear_layer(mat.clearcoat_roughness_layer, texture_uv(in.uv, mat.clearcoat_roughness_transform)).g;
        }
        let cc_n_dot_v = max(dot(cc_normal, v), 0.0);
        let cc_r = reflect(-v, cc_normal);
        let cc_prefiltered = textureSampleLevel(prefiltered_env, ibl_sampler, cc_r, cc_rough * MAX_REFLECTION_LOD).rgb;
        let cc_f = fresnel_clearcoat(cc_n_dot_v);
        let cc_brdf = textureSampleLevel(brdf_lut, ibl_sampler, vec2<f32>(cc_n_dot_v, cc_rough), 0.0).rg;
        let cc_spec = cc * (cc_f * cc_brdf.x + cc_brdf.y) * cc_prefiltered;
        ambient = ambient * (1.0 - cc * cc_f) + cc_spec;
    }

    ambient = mix(ambient, ambient * occlusion, mat.occlusion_strength);
    let color = ambient + lo + emissive;

    var alpha = base_color.a;
    if (mat.transmission_factor > 0.0) {
        alpha = 1.0;
    }
    return oit_output(color, alpha, view_depth, in.entity_id);
}
@fragment
fn fs_blend_opaque_prepass(in: VertexOutput, @builtin(front_facing) front_facing: bool) {
    let mat = materials[in.material_index];
    if (mat.double_sided == 0u && !front_facing) {
        discard;
    }
    if (mat.alpha_mode != 2u && mat.transmission_factor <= 0.0) {
        discard;
    }
    if (mat.transmission_factor > 0.0) {
        return;
    }
    var alpha = mat.base_color.a * in.color.a;
    if (mat.base_layer != NO_LAYER) {
        alpha = alpha * sample_srgb_layer(mat.base_layer, texture_uv(in.uv, mat.base_transform)).a;
    }
    if (alpha < mat.blend_opaque_alpha_threshold) {
        discard;
    }
}
