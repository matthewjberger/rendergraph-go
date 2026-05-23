package pass

import (
	"fmt"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/go-gl/mathgl/mgl32"

	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
)

type ssaoBlurPassState struct {
	owner         *ssaoPassState
	pipeline      *wgpu.RenderPipeline
	bgLayout      *wgpu.BindGroupLayout
	depthSampler  *wgpu.Sampler
	paramsBuffer  *wgpu.Buffer
	outputTexture *wgpu.Texture
	outputView    *wgpu.TextureView
	bindGroup     *wgpu.BindGroup
	currentWidth  uint32
	currentHeight uint32
}

func newSsaoBlurState(device *wgpu.Device, owner *ssaoPassState) (*ssaoBlurPassState, error) {
	module, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "ssao blur shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: ssaoBlurShader},
	})
	if err != nil {
		return nil, fmt.Errorf("ssao blur: shader: %w", err)
	}
	defer module.Release()

	bgLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "ssao blur bg layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageFragment, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeFloat, ViewDimension: wgpu.TextureViewDimension2D}},
			{Binding: 1, Visibility: wgpu.ShaderStageFragment, Sampler: wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeFiltering}},
			{Binding: 2, Visibility: wgpu.ShaderStageFragment, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeDepth, ViewDimension: wgpu.TextureViewDimension2D}},
			{Binding: 3, Visibility: wgpu.ShaderStageFragment, Sampler: wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeNonFiltering}},
			{Binding: 4, Visibility: wgpu.ShaderStageFragment, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeFloat, ViewDimension: wgpu.TextureViewDimension2D}},
			{Binding: 5, Visibility: wgpu.ShaderStageFragment, Sampler: wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeFiltering}},
			{Binding: 6, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("ssao blur: bg layout: %w", err)
	}

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "ssao blur pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bgLayout},
	})
	if err != nil {
		return nil, fmt.Errorf("ssao blur: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	pipeline, err := device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "ssao blur pipeline",
		Layout: pipelineLayout,
		Vertex: wgpu.VertexState{Module: module, EntryPoint: "vertex_main"},
		Primitive: wgpu.PrimitiveState{
			Topology: wgpu.PrimitiveTopologyTriangleList,
			CullMode: wgpu.CullModeNone,
		},
		Multisample: wgpu.MultisampleState{Count: 1, Mask: 0xFFFFFFFF},
		Fragment: &wgpu.FragmentState{
			Module:     module,
			EntryPoint: "fragment_main",
			Targets:    []wgpu.ColorTargetState{{Format: ssaoFormat, WriteMask: wgpu.ColorWriteMaskAll}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("ssao blur: pipeline: %w", err)
	}

	depthSampler, err := device.CreateSampler(&wgpu.SamplerDescriptor{
		Label:         "ssao blur depth sampler",
		AddressModeU:  wgpu.AddressModeClampToEdge,
		AddressModeV:  wgpu.AddressModeClampToEdge,
		AddressModeW:  wgpu.AddressModeClampToEdge,
		MagFilter:     wgpu.FilterModeNearest,
		MinFilter:     wgpu.FilterModeNearest,
		MipmapFilter:  wgpu.MipmapFilterModeNearest,
		MaxAnisotropy: 1,
	})
	if err != nil {
		return nil, fmt.Errorf("ssao blur: depth sampler: %w", err)
	}

	paramsBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "ssao blur params",
		Size:  uint64(unsafe.Sizeof(ssaoBlurParams{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("ssao blur: params: %w", err)
	}

	return &ssaoBlurPassState{
		owner:        owner,
		pipeline:     pipeline,
		bgLayout:     bgLayout,
		depthSampler: depthSampler,
		paramsBuffer: paramsBuffer,
	}, nil
}

func ssaoBlurPrepare(state *ssaoBlurPassState, context *render.PassContext) error {
	width, height, err := ssaoSize(context, "depth")
	if err != nil {
		return err
	}
	if state.currentWidth != width || state.currentHeight != height {
		if err := state.recreateOutput(context.Device, width, height); err != nil {
			return err
		}
		state.currentWidth = width
		state.currentHeight = height
		state.bindGroup = nil
	}

	params := ssaoBlurParams{
		ScreenSize:     mgl32.Vec2{float32(width), float32(height)},
		DepthThreshold: 0.05,
		NormalPower:    32.0,
	}
	writeBuffer(context.Device, context.Queue, context.Encoder, state.paramsBuffer, 0, bytesOf(&params))

	if state.bindGroup != nil {
		return nil
	}
	depthView, err := context.TextureView("depth")
	if err != nil {
		return err
	}
	normalView, err := context.TextureView("view_normals")
	if err != nil {
		return err
	}
	if state.owner == nil || state.owner.rawView == nil {
		return nil
	}
	bg, err := context.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "ssao blur bg",
		Layout: state.bgLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, TextureView: state.owner.rawView},
			{Binding: 1, Sampler: state.owner.noiseSampler},
			{Binding: 2, TextureView: depthView},
			{Binding: 3, Sampler: state.depthSampler},
			{Binding: 4, TextureView: normalView},
			{Binding: 5, Sampler: state.owner.noiseSampler},
			{Binding: 6, Buffer: state.paramsBuffer, Offset: 0, Size: uint64(unsafe.Sizeof(ssaoBlurParams{}))},
		},
	})
	if err != nil {
		return fmt.Errorf("ssao blur: bind group: %w", err)
	}
	state.bindGroup = bg

	ecsSetSsaoResource(context, state.outputView)
	return nil
}

func ssaoBlurExecute(state *ssaoBlurPassState, context *render.PassContext) error {
	if state.outputView == nil || state.bindGroup == nil {
		return nil
	}
	passEnc := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		Label: "ssao blur",
		ColorAttachments: []wgpu.RenderPassColorAttachment{{
			View:       state.outputView,
			LoadOp:     wgpu.LoadOpClear,
			StoreOp:    wgpu.StoreOpStore,
			ClearValue: wgpu.Color{R: 1, G: 1, B: 1, A: 1},
		}},
	})
	passEnc.SetPipeline(state.pipeline)
	passEnc.SetBindGroup(0, state.bindGroup, nil)
	passEnc.Draw(3, 1, 0, 0)
	passEnc.End()
	passEnc.Release()
	return nil
}

func ssaoBlurInvalidate(state *ssaoBlurPassState) {
	if state.bindGroup != nil {
		state.bindGroup.Release()
		state.bindGroup = nil
	}
}

func ssaoBlurRelease(state *ssaoBlurPassState) {
	if state.bindGroup != nil {
		state.bindGroup.Release()
	}
	if state.outputView != nil {
		state.outputView.Release()
	}
	if state.outputTexture != nil {
		state.outputTexture.Release()
	}
	if state.paramsBuffer != nil {
		state.paramsBuffer.Release()
	}
	if state.depthSampler != nil {
		state.depthSampler.Release()
	}
	if state.pipeline != nil {
		state.pipeline.Release()
	}
	if state.bgLayout != nil {
		state.bgLayout.Release()
	}
}

func (state *ssaoBlurPassState) recreateOutput(device *wgpu.Device, width, height uint32) error {
	if state.outputView != nil {
		state.outputView.Release()
	}
	if state.outputTexture != nil {
		state.outputTexture.Release()
	}
	tex, err := device.CreateTexture(&wgpu.TextureDescriptor{
		Label:         "ssao blurred",
		Size:          wgpu.Extent3D{Width: width, Height: height, DepthOrArrayLayers: 1},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     wgpu.TextureDimension2D,
		Format:        ssaoFormat,
		Usage:         wgpu.TextureUsageRenderAttachment | wgpu.TextureUsageTextureBinding,
	})
	if err != nil {
		return fmt.Errorf("ssao blur: output texture: %w", err)
	}
	view, err := tex.CreateView(nil)
	if err != nil {
		tex.Release()
		return fmt.Errorf("ssao blur: output view: %w", err)
	}
	state.outputTexture = tex
	state.outputView = view
	return nil
}

func ecsSetSsaoResource(context *render.PassContext, view *wgpu.TextureView) {
	if view == nil {
		return
	}
	if resource, ok := ecs.Resource[SsaoResource](context.World); ok && resource != nil && resource.Result != nil {
		resource.Result.View = view
		return
	}
	ecs.SetResource(context.World, SsaoResource{Result: &SsaoResult{View: view}})
}
