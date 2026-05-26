package pass

import (
	_ "embed"
	"fmt"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/go-gl/mathgl/mgl32"

	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
)

//go:embed ssr.wgsl
var ssrShader string

//go:embed ssr_blur.wgsl
var ssrBlurShader string

const ssrFormat = render.HdrFormat

const (
	ssrFadeStart float32 = 0.8
	ssrFadeEnd   float32 = 1.0
)

type ssrParams struct {
	Projection    mgl32.Mat4
	InvProjection mgl32.Mat4
	View          mgl32.Mat4
	InvView       mgl32.Mat4
	ScreenSize    mgl32.Vec2
	MaxSteps      uint32
	Thickness     float32
	MaxDistance   float32
	Stride        float32
	FadeStart     float32
	FadeEnd       float32
	Intensity     float32
	Enabled       float32
	Padding       mgl32.Vec2
}

type ssrBlurParams struct {
	ScreenSize      mgl32.Vec2
	DepthThreshold  float32
	NormalThreshold float32
}

type SsrResult struct {
	View *wgpu.TextureView
}

type SsrResource struct {
	Result *SsrResult
}

type ssrPassState struct {
	pipeline      *wgpu.RenderPipeline
	bgLayout      *wgpu.BindGroupLayout
	pointSampler  *wgpu.Sampler
	linearSampler *wgpu.Sampler
	paramsBuffer  *wgpu.Buffer
	rawTexture    *wgpu.Texture
	rawView       *wgpu.TextureView
	bindGroup     *wgpu.BindGroup
	currentWidth  uint32
	currentHeight uint32
	aspectFn      func() float32
}

func AddSsrPass(renderer *render.Renderer, aspect func() float32) (*render.Pass, *render.Pass, error) {
	state, err := newSsrState(renderer.Device, aspect)
	if err != nil {
		return nil, nil, err
	}
	pass := &render.Pass{
		Name:                 "ssr",
		Reads:                []string{"depth", "view_normals", "scene_color"},
		Prepare:              func(c *render.PassContext) error { return ssrPrepare(state, c) },
		Execute:              func(c *render.PassContext) error { return ssrExecute(state, c) },
		InvalidateBindGroups: func() { ssrInvalidate(state) },
		Release:              func() { ssrRelease(state) },
	}
	if err := renderer.Graph.AddPass(pass, []render.SlotBinding{
		{Slot: "depth", ResourceID: renderer.DepthID},
		{Slot: "view_normals", ResourceID: renderer.ViewNormalsID},
		{Slot: "scene_color", ResourceID: renderer.SceneColorID},
	}); err != nil {
		return nil, nil, err
	}

	blurState, err := newSsrBlurState(renderer.Device, state)
	if err != nil {
		return nil, nil, err
	}
	blurPass := &render.Pass{
		Name:                 "ssr_blur",
		Reads:                []string{"depth", "view_normals"},
		Prepare:              func(c *render.PassContext) error { return ssrBlurPrepare(blurState, c) },
		Execute:              func(c *render.PassContext) error { return ssrBlurExecute(blurState, c) },
		InvalidateBindGroups: func() { ssrBlurInvalidate(blurState) },
		Release:              func() { ssrBlurRelease(blurState) },
	}
	if err := renderer.Graph.AddPass(blurPass, []render.SlotBinding{
		{Slot: "depth", ResourceID: renderer.DepthID},
		{Slot: "view_normals", ResourceID: renderer.ViewNormalsID},
	}); err != nil {
		return nil, nil, err
	}
	return pass, blurPass, nil
}

func newSsrState(device *wgpu.Device, aspect func() float32) (*ssrPassState, error) {
	module, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "ssr shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: ssrShader},
	})
	if err != nil {
		return nil, fmt.Errorf("ssr: shader: %w", err)
	}
	defer module.Release()

	bgLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "ssr bg layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageFragment, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeDepth, ViewDimension: wgpu.TextureViewDimension2D}},
			{Binding: 1, Visibility: wgpu.ShaderStageFragment, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeFloat, ViewDimension: wgpu.TextureViewDimension2D}},
			{Binding: 2, Visibility: wgpu.ShaderStageFragment, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeFloat, ViewDimension: wgpu.TextureViewDimension2D}},
			{Binding: 3, Visibility: wgpu.ShaderStageFragment, Sampler: wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeNonFiltering}},
			{Binding: 4, Visibility: wgpu.ShaderStageFragment, Sampler: wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeFiltering}},
			{Binding: 5, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("ssr: bg layout: %w", err)
	}

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "ssr pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bgLayout},
	})
	if err != nil {
		return nil, fmt.Errorf("ssr: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	pipeline, err := device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:       "ssr pipeline",
		Layout:      pipelineLayout,
		Vertex:      wgpu.VertexState{Module: module, EntryPoint: "vertex_main"},
		Primitive:   wgpu.PrimitiveState{Topology: wgpu.PrimitiveTopologyTriangleList, CullMode: wgpu.CullModeNone},
		Multisample: wgpu.MultisampleState{Count: 1, Mask: 0xFFFFFFFF},
		Fragment: &wgpu.FragmentState{
			Module:     module,
			EntryPoint: "fragment_main",
			Targets:    []wgpu.ColorTargetState{{Format: ssrFormat, WriteMask: wgpu.ColorWriteMaskAll}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("ssr: pipeline: %w", err)
	}

	pointSampler, err := device.CreateSampler(&wgpu.SamplerDescriptor{
		Label:         "ssr point sampler",
		AddressModeU:  wgpu.AddressModeClampToEdge,
		AddressModeV:  wgpu.AddressModeClampToEdge,
		AddressModeW:  wgpu.AddressModeClampToEdge,
		MagFilter:     wgpu.FilterModeNearest,
		MinFilter:     wgpu.FilterModeNearest,
		MipmapFilter:  wgpu.MipmapFilterModeNearest,
		MaxAnisotropy: 1,
	})
	if err != nil {
		return nil, fmt.Errorf("ssr: point sampler: %w", err)
	}
	linearSampler, err := device.CreateSampler(&wgpu.SamplerDescriptor{
		Label:         "ssr linear sampler",
		AddressModeU:  wgpu.AddressModeClampToEdge,
		AddressModeV:  wgpu.AddressModeClampToEdge,
		AddressModeW:  wgpu.AddressModeClampToEdge,
		MagFilter:     wgpu.FilterModeLinear,
		MinFilter:     wgpu.FilterModeLinear,
		MipmapFilter:  wgpu.MipmapFilterModeNearest,
		MaxAnisotropy: 1,
	})
	if err != nil {
		return nil, fmt.Errorf("ssr: linear sampler: %w", err)
	}

	paramsBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "ssr params",
		Size:  uint64(unsafe.Sizeof(ssrParams{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("ssr: params buffer: %w", err)
	}

	return &ssrPassState{
		pipeline:      pipeline,
		bgLayout:      bgLayout,
		pointSampler:  pointSampler,
		linearSampler: linearSampler,
		paramsBuffer:  paramsBuffer,
		aspectFn:      aspect,
	}, nil
}

func ssrPrepare(state *ssrPassState, context *render.PassContext) error {
	width, height, err := ssaoSize(context, "depth")
	if err != nil {
		return err
	}
	if state.currentWidth != width || state.currentHeight != height {
		if err := state.recreateRawTexture(context.Device, width, height); err != nil {
			return err
		}
		state.currentWidth = width
		state.currentHeight = height
		state.bindGroup = nil
	}

	settings := render.DefaultGraphics().Ssr
	if g, ok := ecs.Resource[render.Graphics](context.World); ok && g != nil {
		settings = g.Ssr
	}
	enabled := float32(0)
	if settings.Enabled {
		enabled = 1
	}

	view, projection := ssaoViewProj(context, state.aspectFn())
	params := ssrParams{
		Projection:    projection,
		InvProjection: projection.Inv(),
		View:          view,
		InvView:       view.Inv(),
		ScreenSize:    mgl32.Vec2{float32(width), float32(height)},
		MaxSteps:      settings.MaxSteps,
		Thickness:     settings.Thickness,
		MaxDistance:   settings.MaxDistance,
		Stride:        settings.Stride,
		FadeStart:     ssrFadeStart,
		FadeEnd:       ssrFadeEnd,
		Intensity:     settings.Intensity,
		Enabled:       enabled,
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
	sceneView, err := context.TextureView("scene_color")
	if err != nil {
		return err
	}
	bg, err := context.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "ssr bg",
		Layout: state.bgLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, TextureView: depthView},
			{Binding: 1, TextureView: normalView},
			{Binding: 2, TextureView: sceneView},
			{Binding: 3, Sampler: state.pointSampler},
			{Binding: 4, Sampler: state.linearSampler},
			{Binding: 5, Buffer: state.paramsBuffer, Offset: 0, Size: uint64(unsafe.Sizeof(ssrParams{}))},
		},
	})
	if err != nil {
		return fmt.Errorf("ssr: bind group: %w", err)
	}
	state.bindGroup = bg
	return nil
}

func ssrExecute(state *ssrPassState, context *render.PassContext) error {
	if state.rawView == nil || state.bindGroup == nil {
		return nil
	}
	passEnc := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		Label: "ssr",
		ColorAttachments: []wgpu.RenderPassColorAttachment{{
			View:       state.rawView,
			LoadOp:     wgpu.LoadOpClear,
			StoreOp:    wgpu.StoreOpStore,
			ClearValue: wgpu.Color{},
		}},
	})
	passEnc.SetPipeline(state.pipeline)
	passEnc.SetBindGroup(0, state.bindGroup, nil)
	passEnc.Draw(3, 1, 0, 0)
	passEnc.End()
	passEnc.Release()
	return nil
}

func ssrInvalidate(state *ssrPassState) {
	if state.bindGroup != nil {
		state.bindGroup.Release()
		state.bindGroup = nil
	}
}

func ssrRelease(state *ssrPassState) {
	if state.bindGroup != nil {
		state.bindGroup.Release()
	}
	if state.rawView != nil {
		state.rawView.Release()
	}
	if state.rawTexture != nil {
		state.rawTexture.Release()
	}
	if state.paramsBuffer != nil {
		state.paramsBuffer.Release()
	}
	if state.linearSampler != nil {
		state.linearSampler.Release()
	}
	if state.pointSampler != nil {
		state.pointSampler.Release()
	}
	if state.pipeline != nil {
		state.pipeline.Release()
	}
	if state.bgLayout != nil {
		state.bgLayout.Release()
	}
}

func (state *ssrPassState) recreateRawTexture(device *wgpu.Device, width, height uint32) error {
	if state.rawView != nil {
		state.rawView.Release()
	}
	if state.rawTexture != nil {
		state.rawTexture.Release()
	}
	tex, err := device.CreateTexture(&wgpu.TextureDescriptor{
		Label:         "ssr raw",
		Size:          wgpu.Extent3D{Width: width, Height: height, DepthOrArrayLayers: 1},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     wgpu.TextureDimension2D,
		Format:        ssrFormat,
		Usage:         wgpu.TextureUsageRenderAttachment | wgpu.TextureUsageTextureBinding,
	})
	if err != nil {
		return fmt.Errorf("ssr: raw texture: %w", err)
	}
	view, err := tex.CreateView(nil)
	if err != nil {
		tex.Release()
		return fmt.Errorf("ssr: raw view: %w", err)
	}
	state.rawTexture = tex
	state.rawView = view
	return nil
}

type ssrBlurPassState struct {
	owner         *ssrPassState
	pipeline      *wgpu.RenderPipeline
	bgLayout      *wgpu.BindGroupLayout
	paramsBuffer  *wgpu.Buffer
	outputTexture *wgpu.Texture
	outputView    *wgpu.TextureView
	bindGroup     *wgpu.BindGroup
	currentWidth  uint32
	currentHeight uint32
}

func newSsrBlurState(device *wgpu.Device, owner *ssrPassState) (*ssrBlurPassState, error) {
	module, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "ssr blur shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: ssrBlurShader},
	})
	if err != nil {
		return nil, fmt.Errorf("ssr blur: shader: %w", err)
	}
	defer module.Release()

	bgLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "ssr blur bg layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageFragment, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeFloat, ViewDimension: wgpu.TextureViewDimension2D}},
			{Binding: 1, Visibility: wgpu.ShaderStageFragment, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeDepth, ViewDimension: wgpu.TextureViewDimension2D}},
			{Binding: 2, Visibility: wgpu.ShaderStageFragment, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeFloat, ViewDimension: wgpu.TextureViewDimension2D}},
			{Binding: 3, Visibility: wgpu.ShaderStageFragment, Sampler: wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeFiltering}},
			{Binding: 4, Visibility: wgpu.ShaderStageFragment, Sampler: wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeNonFiltering}},
			{Binding: 5, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("ssr blur: bg layout: %w", err)
	}

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "ssr blur pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bgLayout},
	})
	if err != nil {
		return nil, fmt.Errorf("ssr blur: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	pipeline, err := device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:       "ssr blur pipeline",
		Layout:      pipelineLayout,
		Vertex:      wgpu.VertexState{Module: module, EntryPoint: "vertex_main"},
		Primitive:   wgpu.PrimitiveState{Topology: wgpu.PrimitiveTopologyTriangleList, CullMode: wgpu.CullModeNone},
		Multisample: wgpu.MultisampleState{Count: 1, Mask: 0xFFFFFFFF},
		Fragment: &wgpu.FragmentState{
			Module:     module,
			EntryPoint: "fragment_main",
			Targets:    []wgpu.ColorTargetState{{Format: ssrFormat, WriteMask: wgpu.ColorWriteMaskAll}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("ssr blur: pipeline: %w", err)
	}

	paramsBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "ssr blur params",
		Size:  uint64(unsafe.Sizeof(ssrBlurParams{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("ssr blur: params: %w", err)
	}

	return &ssrBlurPassState{
		owner:        owner,
		pipeline:     pipeline,
		bgLayout:     bgLayout,
		paramsBuffer: paramsBuffer,
	}, nil
}

func ssrBlurPrepare(state *ssrBlurPassState, context *render.PassContext) error {
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

	params := ssrBlurParams{
		ScreenSize:      mgl32.Vec2{float32(width), float32(height)},
		DepthThreshold:  0.05,
		NormalThreshold: 16.0,
	}
	writeBuffer(context.Device, context.Queue, context.Encoder, state.paramsBuffer, 0, bytesOf(&params))

	if state.bindGroup == nil {
		if state.owner == nil || state.owner.rawView == nil {
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
		bg, err := context.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  "ssr blur bg",
			Layout: state.bgLayout,
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, TextureView: state.owner.rawView},
				{Binding: 1, TextureView: depthView},
				{Binding: 2, TextureView: normalView},
				{Binding: 3, Sampler: state.owner.linearSampler},
				{Binding: 4, Sampler: state.owner.pointSampler},
				{Binding: 5, Buffer: state.paramsBuffer, Offset: 0, Size: uint64(unsafe.Sizeof(ssrBlurParams{}))},
			},
		})
		if err != nil {
			return fmt.Errorf("ssr blur: bind group: %w", err)
		}
		state.bindGroup = bg
	}

	ecsSetSsrResource(context, state.outputView)
	return nil
}

func ssrBlurExecute(state *ssrBlurPassState, context *render.PassContext) error {
	if state.outputView == nil || state.bindGroup == nil {
		return nil
	}
	passEnc := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		Label: "ssr blur",
		ColorAttachments: []wgpu.RenderPassColorAttachment{{
			View:       state.outputView,
			LoadOp:     wgpu.LoadOpClear,
			StoreOp:    wgpu.StoreOpStore,
			ClearValue: wgpu.Color{},
		}},
	})
	passEnc.SetPipeline(state.pipeline)
	passEnc.SetBindGroup(0, state.bindGroup, nil)
	passEnc.Draw(3, 1, 0, 0)
	passEnc.End()
	passEnc.Release()
	return nil
}

func ssrBlurInvalidate(state *ssrBlurPassState) {
	if state.bindGroup != nil {
		state.bindGroup.Release()
		state.bindGroup = nil
	}
}

func ssrBlurRelease(state *ssrBlurPassState) {
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
	if state.pipeline != nil {
		state.pipeline.Release()
	}
	if state.bgLayout != nil {
		state.bgLayout.Release()
	}
}

func (state *ssrBlurPassState) recreateOutput(device *wgpu.Device, width, height uint32) error {
	if state.outputView != nil {
		state.outputView.Release()
	}
	if state.outputTexture != nil {
		state.outputTexture.Release()
	}
	tex, err := device.CreateTexture(&wgpu.TextureDescriptor{
		Label:         "ssr blurred",
		Size:          wgpu.Extent3D{Width: width, Height: height, DepthOrArrayLayers: 1},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     wgpu.TextureDimension2D,
		Format:        ssrFormat,
		Usage:         wgpu.TextureUsageRenderAttachment | wgpu.TextureUsageTextureBinding,
	})
	if err != nil {
		return fmt.Errorf("ssr blur: output texture: %w", err)
	}
	view, err := tex.CreateView(nil)
	if err != nil {
		tex.Release()
		return fmt.Errorf("ssr blur: output view: %w", err)
	}
	state.outputTexture = tex
	state.outputView = view
	return nil
}

func ecsSetSsrResource(context *render.PassContext, view *wgpu.TextureView) {
	if view == nil {
		return
	}
	if resource, ok := ecs.Resource[SsrResource](context.World); ok && resource != nil && resource.Result != nil {
		resource.Result.View = view
		return
	}
	ecs.SetResource(context.World, SsrResource{Result: &SsrResult{View: view}})
}
