package pass

import (
	_ "embed"
	"fmt"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"

	"indigo/ecs"
	"indigo/render"
)

//go:embed postprocess.wgsl
var postprocessShader string

type postprocessUniform struct {
	Exposure       float32
	BloomIntensity float32
	BloomEnabled   float32
	SsaoEnabled    float32
}

type postprocessPassState struct {
	pipeline        *wgpu.RenderPipeline
	bindGroupLayout *wgpu.BindGroupLayout
	sampler         *wgpu.Sampler
	uniformBuffer   *wgpu.Buffer
	bindGroup       *wgpu.BindGroup
	bloom           *render.Pass
	dummyTexture    *wgpu.Texture
	dummyView       *wgpu.TextureView
	lastSsaoView    *wgpu.TextureView
}

// NewPostProcessPass builds the HDR -> LDR composition pass. The
// fragment shader samples the bound HDR scene_color, applies an
// exposure scale + ACES filmic tonemap, and writes the result
// into the LDR output target. Output format is the surface format
// so subsequent UI / fxaa / present passes can sample it without
// another conversion.
//
// Bind group is rebuilt via the [render.Pass.InvalidateBindGroups]
// hook when the input view changes (resize, transient
// reallocation, external view replacement).
func NewPostProcessPass(device *wgpu.Device, surfaceFormat wgpu.TextureFormat, bloom *render.Pass) (*render.Pass, error) {
	state := &postprocessPassState{bloom: bloom}

	dummyTex, err := device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "postprocess bloom dummy",
		Size: wgpu.Extent3D{
			Width:              1,
			Height:             1,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     wgpu.TextureDimension2D,
		Format:        render.HdrFormat,
		Usage:         wgpu.TextureUsageTextureBinding | wgpu.TextureUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("postprocess: dummy bloom texture: %w", err)
	}
	state.dummyTexture = dummyTex
	dummyView, err := dummyTex.CreateView(nil)
	if err != nil {
		dummyTex.Release()
		return nil, fmt.Errorf("postprocess: dummy bloom view: %w", err)
	}
	state.dummyView = dummyView

	bindGroupLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "postprocess bind group layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: wgpu.ShaderStageFragment,
				Texture: wgpu.TextureBindingLayout{
					SampleType:    wgpu.TextureSampleTypeFloat,
					ViewDimension: wgpu.TextureViewDimension2D,
				},
			},
			{
				Binding:    1,
				Visibility: wgpu.ShaderStageFragment,
				Sampler:    wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeFiltering},
			},
			{
				Binding:    2,
				Visibility: wgpu.ShaderStageFragment,
				Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform},
			},
			{
				Binding:    3,
				Visibility: wgpu.ShaderStageFragment,
				Texture: wgpu.TextureBindingLayout{
					SampleType:    wgpu.TextureSampleTypeFloat,
					ViewDimension: wgpu.TextureViewDimension2D,
				},
			},
			{
				Binding:    4,
				Visibility: wgpu.ShaderStageFragment,
				Sampler:    wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeFiltering},
			},
			{
				Binding:    5,
				Visibility: wgpu.ShaderStageFragment,
				Texture: wgpu.TextureBindingLayout{
					SampleType:    wgpu.TextureSampleTypeFloat,
					ViewDimension: wgpu.TextureViewDimension2D,
				},
			},
			{
				Binding:    6,
				Visibility: wgpu.ShaderStageFragment,
				Sampler:    wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeFiltering},
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("postprocess: bind group layout: %w", err)
	}
	state.bindGroupLayout = bindGroupLayout

	sampler, err := device.CreateSampler(&wgpu.SamplerDescriptor{
		Label:         "postprocess sampler",
		AddressModeU:  wgpu.AddressModeClampToEdge,
		AddressModeV:  wgpu.AddressModeClampToEdge,
		AddressModeW:  wgpu.AddressModeClampToEdge,
		MagFilter:     wgpu.FilterModeLinear,
		MinFilter:     wgpu.FilterModeLinear,
		MipmapFilter:  wgpu.MipmapFilterModeNearest,
		MaxAnisotropy: 1,
	})
	if err != nil {
		return nil, fmt.Errorf("postprocess: sampler: %w", err)
	}
	state.sampler = sampler

	uniformBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "postprocess uniform",
		Size:  uint64(unsafe.Sizeof(postprocessUniform{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("postprocess: uniform buffer: %w", err)
	}
	state.uniformBuffer = uniformBuffer

	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "postprocess shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: postprocessShader},
	})
	if err != nil {
		return nil, fmt.Errorf("postprocess: shader: %w", err)
	}
	defer shader.Release()

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "postprocess pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bindGroupLayout},
	})
	if err != nil {
		return nil, fmt.Errorf("postprocess: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	pipeline, err := device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "postprocess pipeline",
		Layout: pipelineLayout,
		Vertex: wgpu.VertexState{
			Module:     shader,
			EntryPoint: "vertex_main",
		},
		Primitive: wgpu.PrimitiveState{
			Topology:  wgpu.PrimitiveTopologyTriangleList,
			FrontFace: wgpu.FrontFaceCCW,
			CullMode:  wgpu.CullModeNone,
		},
		Multisample: wgpu.MultisampleState{
			Count:                  1,
			Mask:                   0xFFFFFFFF,
			AlphaToCoverageEnabled: false,
		},
		Fragment: &wgpu.FragmentState{
			Module:     shader,
			EntryPoint: "fragment_main",
			Targets: []wgpu.ColorTargetState{{
				Format:    surfaceFormat,
				WriteMask: wgpu.ColorWriteMaskAll,
			}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("postprocess: pipeline: %w", err)
	}
	state.pipeline = pipeline

	return &render.Pass{
		Name:                 "postprocess",
		Reads:                []string{"input"},
		Writes:               []string{"output"},
		State:                state,
		Prepare:              postprocessPrepare,
		Execute:              postprocessExecute,
		InvalidateBindGroups: postprocessInvalidate,
		Release:              postprocessRelease,
	}, nil
}

func postprocessPrepare(s any, context *render.PassContext) error {
	state := s.(*postprocessPassState)

	settings := render.DefaultPostProcessSettings()
	if resource, ok := ecs.Resource[render.PostProcessSettings](context.World); ok {
		settings = *resource
	}

	bloomView := state.dummyView
	bloomEnabled := float32(0.0)
	if settings.BloomEnabled && state.bloom != nil {
		if view := BloomMipView(state.bloom); view != nil {
			bloomView = view
			bloomEnabled = 1.0
		}
	}

	ssaoView := state.dummyView
	ssaoEnabled := float32(0.0)
	if resource, ok := ecs.Resource[SsaoResource](context.World); ok && resource != nil && resource.Result != nil && resource.Result.View != nil {
		ssaoView = resource.Result.View
		ssaoEnabled = 1.0
	}

	uniform := postprocessUniform{
		Exposure:       settings.Exposure,
		BloomIntensity: settings.BloomIntensity,
		BloomEnabled:   bloomEnabled,
		SsaoEnabled:    ssaoEnabled,
	}
	writeBuffer(context.Device, context.Queue, context.Encoder, state.uniformBuffer, 0, bytesOf(&uniform))

	if state.bindGroup != nil && state.lastSsaoView == ssaoView {
		return nil
	}
	if state.bindGroup != nil {
		state.bindGroup.Release()
		state.bindGroup = nil
	}
	state.lastSsaoView = ssaoView

	inputView, err := context.TextureView("input")
	if err != nil {
		return err
	}

	bindGroup, err := context.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "postprocess bind group",
		Layout: state.bindGroupLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, TextureView: inputView},
			{Binding: 1, Sampler: state.sampler},
			{Binding: 2, Buffer: state.uniformBuffer, Offset: 0, Size: uint64(unsafe.Sizeof(postprocessUniform{}))},
			{Binding: 3, TextureView: bloomView},
			{Binding: 4, Sampler: state.sampler},
			{Binding: 5, TextureView: ssaoView},
			{Binding: 6, Sampler: state.sampler},
		},
	})
	if err != nil {
		return fmt.Errorf("postprocess: bind group: %w", err)
	}
	state.bindGroup = bindGroup
	return nil
}

func postprocessExecute(s any, context *render.PassContext) error {
	state := s.(*postprocessPassState)
	attachment, err := context.ColorAttachment("output")
	if err != nil {
		return err
	}
	pass := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		Label:            "postprocess",
		ColorAttachments: []wgpu.RenderPassColorAttachment{attachment},
	})
	pass.SetPipeline(state.pipeline)
	pass.SetBindGroup(0, state.bindGroup, nil)
	pass.Draw(3, 1, 0, 0)
	pass.End()
	pass.Release()
	return nil
}

func postprocessInvalidate(s any) {
	state := s.(*postprocessPassState)
	if state.bindGroup != nil {
		state.bindGroup.Release()
		state.bindGroup = nil
	}
}

func postprocessRelease(s any) {
	state := s.(*postprocessPassState)
	if state.bindGroup != nil {
		state.bindGroup.Release()
	}
	if state.pipeline != nil {
		state.pipeline.Release()
	}
	if state.uniformBuffer != nil {
		state.uniformBuffer.Release()
	}
	if state.sampler != nil {
		state.sampler.Release()
	}
	if state.bindGroupLayout != nil {
		state.bindGroupLayout.Release()
	}
	if state.dummyView != nil {
		state.dummyView.Release()
	}
	if state.dummyTexture != nil {
		state.dummyTexture.Release()
	}
}
