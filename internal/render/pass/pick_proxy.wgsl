struct Uniforms {
    view_proj: mat4x4<f32>,
    model:     mat4x4<f32>,
    entity_id: u32,
    _pad0:     u32,
    _pad1:     u32,
    _pad2:     u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vertex_main(@location(0) position: vec4<f32>) -> @builtin(position) vec4<f32> {
    return uniforms.view_proj * uniforms.model * vec4<f32>(position.xyz, 1.0);
}

@fragment
fn fragment_main() -> @location(0) u32 {
    return uniforms.entity_id;
}
