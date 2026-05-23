package pass

import (
	"fmt"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/go-gl/mathgl/mgl32"

	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/render/asset"
)

// createViewProjLayout returns the group 0 bind-group layout: one
// uniform buffer holding the camera's view * projection matrix,
// visible to the vertex shader.
func createViewProjLayout(device *wgpu.Device) (*wgpu.BindGroupLayout, error) {
	layout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "mesh view_proj bind group layout",
		Entries: []wgpu.BindGroupLayoutEntry{{
			Binding:    0,
			Visibility: wgpu.ShaderStageVertex,
			Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform},
		}},
	})
	if err != nil {
		return nil, fmt.Errorf("mesh pass: view_proj bind group layout: %w", err)
	}
	return layout, nil
}

// createGlobalBgLayout returns the group 1 bind-group layout: the
// clustered-lighting bindings (lights / light_grid / light_indices
// / cluster_uniforms / view_matrix) plus the shared material
// texture arrays and sampler. All fragment-visible.
func createGlobalBgLayout(device *wgpu.Device) (*wgpu.BindGroupLayout, error) {
	layout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "mesh global bind group layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: wgpu.ShaderStageFragment,
				Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage},
			},
			{
				Binding:    1,
				Visibility: wgpu.ShaderStageFragment,
				Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage},
			},
			{
				Binding:    2,
				Visibility: wgpu.ShaderStageFragment,
				Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage},
			},
			{
				Binding:    3,
				Visibility: wgpu.ShaderStageFragment,
				Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform},
			},
			{
				Binding:    4,
				Visibility: wgpu.ShaderStageVertex | wgpu.ShaderStageFragment,
				Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform},
			},
			{
				Binding:    5,
				Visibility: wgpu.ShaderStageFragment,
				Texture: wgpu.TextureBindingLayout{
					SampleType:    wgpu.TextureSampleTypeFloat,
					ViewDimension: wgpu.TextureViewDimension2DArray,
				},
			},
			{
				Binding:    6,
				Visibility: wgpu.ShaderStageFragment,
				Texture: wgpu.TextureBindingLayout{
					SampleType:    wgpu.TextureSampleTypeFloat,
					ViewDimension: wgpu.TextureViewDimension2DArray,
				},
			},
			{
				Binding:    7,
				Visibility: wgpu.ShaderStageFragment,
				Sampler:    wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeFiltering},
			},
			{
				Binding:    8,
				Visibility: wgpu.ShaderStageFragment,
				Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage},
			},
			{
				Binding:    9,
				Visibility: wgpu.ShaderStageFragment,
				Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform},
			},
			{
				Binding:    10,
				Visibility: wgpu.ShaderStageFragment,
				Texture: wgpu.TextureBindingLayout{
					SampleType:    wgpu.TextureSampleTypeDepth,
					ViewDimension: wgpu.TextureViewDimension2DArray,
				},
			},
			{
				Binding:    11,
				Visibility: wgpu.ShaderStageFragment,
				Sampler:    wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeComparison},
			},
			{
				Binding:    12,
				Visibility: wgpu.ShaderStageFragment,
				Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform},
			},
			{
				Binding:    13,
				Visibility: wgpu.ShaderStageFragment,
				Texture: wgpu.TextureBindingLayout{
					SampleType:    wgpu.TextureSampleTypeDepth,
					ViewDimension: wgpu.TextureViewDimension2D,
				},
			},
			{
				Binding:    14,
				Visibility: wgpu.ShaderStageFragment,
				Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform},
			},
			{
				Binding:    15,
				Visibility: wgpu.ShaderStageFragment,
				Texture: wgpu.TextureBindingLayout{
					SampleType:    wgpu.TextureSampleTypeUnfilterableFloat,
					ViewDimension: wgpu.TextureViewDimensionCubeArray,
				},
			},
			{
				Binding:    16,
				Visibility: wgpu.ShaderStageFragment,
				Sampler:    wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeNonFiltering},
			},
			{
				Binding:    17,
				Visibility: wgpu.ShaderStageFragment,
				Sampler:    wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeNonFiltering},
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("mesh pass: global bind group layout: %w", err)
	}
	return layout, nil
}

// createHandleBgLayout returns the group 2 bind-group layout: the
// per-handle storage buffers (models / material_indices /
// entity_ids). All vertex-only — the vertex stage flat-interpolates
// the material_index into the fragment shader, which then looks the
// MaterialGPU up in the global registry buffer bound at group 1
// binding 8.
func createHandleBgLayout(device *wgpu.Device) (*wgpu.BindGroupLayout, error) {
	layout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "mesh per-handle bind group layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: wgpu.ShaderStageVertex,
				Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage},
			},
			{
				Binding:    1,
				Visibility: wgpu.ShaderStageVertex,
				Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage},
			},
			{
				Binding:    2,
				Visibility: wgpu.ShaderStageVertex,
				Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage},
			},
			{
				Binding:    3,
				Visibility: wgpu.ShaderStageVertex,
				Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage},
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("mesh pass: per-handle bind group layout: %w", err)
	}
	return layout, nil
}

// createIblBgLayout returns the group 3 bind-group layout: the
// irradiance cube, prefiltered specular cube, BRDF LUT, and IBL
// sampler. All fragment-visible.
func createIblBgLayout(device *wgpu.Device) (*wgpu.BindGroupLayout, error) {
	layout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "mesh ibl bind group layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: wgpu.ShaderStageFragment,
				Texture: wgpu.TextureBindingLayout{
					SampleType:    wgpu.TextureSampleTypeFloat,
					ViewDimension: wgpu.TextureViewDimensionCube,
				},
			},
			{
				Binding:    1,
				Visibility: wgpu.ShaderStageFragment,
				Texture: wgpu.TextureBindingLayout{
					SampleType:    wgpu.TextureSampleTypeFloat,
					ViewDimension: wgpu.TextureViewDimensionCube,
				},
			},
			{
				Binding:    2,
				Visibility: wgpu.ShaderStageFragment,
				Texture: wgpu.TextureBindingLayout{
					SampleType:    wgpu.TextureSampleTypeFloat,
					ViewDimension: wgpu.TextureViewDimension2D,
				},
			},
			{
				Binding:    3,
				Visibility: wgpu.ShaderStageFragment,
				Sampler:    wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeFiltering},
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("mesh pass: ibl bind group layout: %w", err)
	}
	return layout, nil
}

// createMeshPipeline builds the render pipeline that ties the four
// bind-group layouts to the WGSL shader module. The pipeline owns
// the only reference to the shader module; the caller can release
// the shader after this function returns. The pipeline layout is
// owned exclusively by this pipeline and released here too.
func createMeshPipeline(
	device *wgpu.Device,
	surfaceFormat wgpu.TextureFormat,
	viewProjLayout, globalBgLayout, handleBgLayout, iblBgLayout *wgpu.BindGroupLayout,
) (*wgpu.RenderPipeline, error) {
	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "mesh shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: meshShader},
	})
	if err != nil {
		return nil, fmt.Errorf("mesh pass: shader: %w", err)
	}
	defer shader.Release()

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "mesh pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{viewProjLayout, globalBgLayout, handleBgLayout, iblBgLayout},
	})
	if err != nil {
		return nil, fmt.Errorf("mesh pass: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	pipeline, err := device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "mesh pipeline",
		Layout: pipelineLayout,
		Vertex: wgpu.VertexState{
			Module:     shader,
			EntryPoint: "vertex_main",
			Buffers: []wgpu.VertexBufferLayout{{
				ArrayStride: uint64(unsafe.Sizeof(asset.MeshVertex{})),
				StepMode:    wgpu.VertexStepModeVertex,
				Attributes: []wgpu.VertexAttribute{
					{Format: wgpu.VertexFormatFloat32x4, Offset: 0, ShaderLocation: 0},
					{Format: wgpu.VertexFormatFloat32x4, Offset: 16, ShaderLocation: 1},
					{Format: wgpu.VertexFormatFloat32x4, Offset: 32, ShaderLocation: 2},
					{Format: wgpu.VertexFormatFloat32x4, Offset: 48, ShaderLocation: 3},
					{Format: wgpu.VertexFormatFloat32x4, Offset: 64, ShaderLocation: 4},
				},
			}},
		},
		Primitive: wgpu.PrimitiveState{
			Topology:  wgpu.PrimitiveTopologyTriangleList,
			FrontFace: wgpu.FrontFaceCCW,
			CullMode:  wgpu.CullModeNone,
		},
		DepthStencil: &wgpu.DepthStencilState{
			Format:            render.DepthFormat,
			DepthWriteEnabled: meshMainDepthWrite,
			DepthCompare:      meshMainDepthCompare,
			StencilFront: wgpu.StencilFaceState{
				Compare:     wgpu.CompareFunctionAlways,
				FailOp:      wgpu.StencilOperationKeep,
				DepthFailOp: wgpu.StencilOperationKeep,
				PassOp:      wgpu.StencilOperationKeep,
			},
			StencilBack: wgpu.StencilFaceState{
				Compare:     wgpu.CompareFunctionAlways,
				FailOp:      wgpu.StencilOperationKeep,
				DepthFailOp: wgpu.StencilOperationKeep,
				PassOp:      wgpu.StencilOperationKeep,
			},
		},
		Multisample: wgpu.MultisampleState{
			Count:                  1,
			Mask:                   0xFFFFFFFF,
			AlphaToCoverageEnabled: false,
		},
		Fragment: &wgpu.FragmentState{
			Module:     shader,
			EntryPoint: "fragment_main",
			Targets: []wgpu.ColorTargetState{
				{
					Format:    surfaceFormat,
					WriteMask: wgpu.ColorWriteMaskAll,
				},
				{
					Format:    render.EntityIdFormat,
					WriteMask: wgpu.ColorWriteMaskAll,
				},
				{
					Format:    render.HdrFormat,
					WriteMask: wgpu.ColorWriteMaskAll,
				},
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("mesh pass: pipeline: %w", err)
	}
	return pipeline, nil
}

// createViewProjBuffer returns a 64-byte uniform buffer that the
// camera writes its view * projection matrix into each frame.
func createViewProjBuffer(device *wgpu.Device) (*wgpu.Buffer, error) {
	buf, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "mesh view_proj buffer",
		Size:  uint64(unsafe.Sizeof(mgl32.Mat4{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("mesh pass: view_proj buffer: %w", err)
	}
	return buf, nil
}

// createViewProjBindGroup wraps the view_proj uniform buffer in a
// group 0 bind group with the matching layout.
func createViewProjBindGroup(device *wgpu.Device, layout *wgpu.BindGroupLayout, buf *wgpu.Buffer) (*wgpu.BindGroup, error) {
	bg, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "mesh view_proj bind group",
		Layout: layout,
		Entries: []wgpu.BindGroupEntry{{
			Binding: 0,
			Buffer:  buf,
			Offset:  0,
			Size:    uint64(unsafe.Sizeof(mgl32.Mat4{})),
		}},
	})
	if err != nil {
		return nil, fmt.Errorf("mesh pass: view_proj bind group: %w", err)
	}
	return bg, nil
}

// createGlobalBindGroup builds the group 1 bind group from the
// cluster compute resources + the material texture arrays + the
// global material registry. The registry buffer can grow at
// runtime; when it does, the caller is responsible for releasing
// this bind group and rebuilding via this helper.
func createGlobalBindGroup(
	device *wgpu.Device,
	layout *wgpu.BindGroupLayout,
	clusters *clusterResources,
	arrays *asset.MaterialTextureArrays,
	registry *asset.MaterialRegistry,
	shadow *Shadow,
	shadowLightVP *wgpu.Buffer,
	spotShadow *SpotShadow,
	pointShadow *PointShadow,
) (*wgpu.BindGroup, error) {
	bg, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "mesh global bind group",
		Layout: layout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: clusters.lights, Offset: 0, Size: wgpu.WholeSize},
			{Binding: 1, Buffer: clusters.lightGrid, Offset: 0, Size: wgpu.WholeSize},
			{Binding: 2, Buffer: clusters.lightIndices, Offset: 0, Size: wgpu.WholeSize},
			{Binding: 3, Buffer: clusters.clusterUniforms, Offset: 0, Size: ClusterUniformsSize},
			{Binding: 4, Buffer: clusters.viewMatrix, Offset: 0, Size: 64},
			{Binding: 5, TextureView: arrays.SRGBView},
			{Binding: 6, TextureView: arrays.LinearView},
			{Binding: 7, Sampler: arrays.Sampler},
			{Binding: 8, Buffer: registry.Buffer(), Offset: 0, Size: wgpu.WholeSize},
			{Binding: 9, Buffer: shadowLightVP, Offset: 0, Size: wgpu.WholeSize},
			{Binding: 10, TextureView: shadow.ArrayView},
			{Binding: 11, Sampler: shadow.Sampler},
			{Binding: 12, Buffer: spotShadow.UniformBuffer, Offset: 0, Size: wgpu.WholeSize},
			{Binding: 13, TextureView: spotShadow.AtlasView},
			{Binding: 14, Buffer: pointShadow.DataBuffer, Offset: 0, Size: wgpu.WholeSize},
			{Binding: 15, TextureView: pointShadow.CubeArrayView},
			{Binding: 16, Sampler: pointShadow.Sampler},
			{Binding: 17, Sampler: shadow.RawSampler},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("mesh pass: global bind group: %w", err)
	}
	return bg, nil
}

// createIblBindGroup wraps the IBL bundle's textures + sampler in
// the group 3 bind group the fragment shader reads from.
func createIblBindGroup(device *wgpu.Device, layout *wgpu.BindGroupLayout, ibl *IBL) (*wgpu.BindGroup, error) {
	bg, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "mesh ibl bind group",
		Layout: layout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, TextureView: ibl.IrradianceView},
			{Binding: 1, TextureView: ibl.PrefilteredView},
			{Binding: 2, TextureView: ibl.BrdfLutView},
			{Binding: 3, Sampler: ibl.Sampler},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("mesh pass: ibl bind group: %w", err)
	}
	return bg, nil
}
