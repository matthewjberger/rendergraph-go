package pass

import (
	_ "embed"
	"fmt"
	"math"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/go-gl/mathgl/mgl32"

	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
)

//go:embed ssao.wgsl
var ssaoShader string

//go:embed ssao_blur.wgsl
var ssaoBlurShader string

const ssaoKernelSize = 64
const ssaoNoiseSize = 4

type ssaoParams struct {
	Projection    mgl32.Mat4
	InvProjection mgl32.Mat4
	ScreenSize    mgl32.Vec2
	Radius        float32
	Bias          float32
	Intensity     float32
	Enabled       float32
	SampleCount   float32
	_             float32
}

type ssaoBlurParams struct {
	ScreenSize     mgl32.Vec2
	DepthThreshold float32
	NormalPower    float32
}

const ssaoFormat = wgpu.TextureFormatR8Unorm

type ssaoPassState struct {
	pipeline      *wgpu.RenderPipeline
	bgLayout      *wgpu.BindGroupLayout
	pointSampler  *wgpu.Sampler
	noiseSampler  *wgpu.Sampler
	paramsBuffer  *wgpu.Buffer
	kernelBuffer  *wgpu.Buffer
	noiseTexture  *wgpu.Texture
	noiseView     *wgpu.TextureView
	rawTexture    *wgpu.Texture
	rawView       *wgpu.TextureView
	bindGroup     *wgpu.BindGroup
	currentWidth  uint32
	currentHeight uint32
	aspectFn      func() float32
}

type SsaoResult struct {
	View *wgpu.TextureView
}

type SsaoResource struct {
	Result *SsaoResult
}

func AddSsaoPass(renderer *render.Renderer, aspect func() float32) (*render.Pass, *render.Pass, error) {
	state, err := newSsaoState(renderer.Device, aspect)
	if err != nil {
		return nil, nil, err
	}

	pass := &render.Pass{
		Name:                 "ssao",
		Reads:                []string{"depth", "view_normals"},
		Prepare:              func(c *render.PassContext) error { return ssaoPrepare(state, c) },
		Execute:              func(c *render.PassContext) error { return ssaoExecute(state, c) },
		InvalidateBindGroups: func() { ssaoInvalidate(state) },
		Release:              func() { ssaoRelease(state) },
	}
	if err := renderer.Graph.AddPass(pass, []render.SlotBinding{
		{Slot: "depth", ResourceID: renderer.DepthID},
		{Slot: "view_normals", ResourceID: renderer.ViewNormalsID},
	}); err != nil {
		return nil, nil, err
	}

	blurState, err := newSsaoBlurState(renderer.Device, state)
	if err != nil {
		return nil, nil, err
	}
	blurPass := &render.Pass{
		Name:                 "ssao_blur",
		Reads:                []string{"depth", "view_normals"},
		Prepare:              func(c *render.PassContext) error { return ssaoBlurPrepare(blurState, c) },
		Execute:              func(c *render.PassContext) error { return ssaoBlurExecute(blurState, c) },
		InvalidateBindGroups: func() { ssaoBlurInvalidate(blurState) },
		Release:              func() { ssaoBlurRelease(blurState) },
	}
	if err := renderer.Graph.AddPass(blurPass, []render.SlotBinding{
		{Slot: "depth", ResourceID: renderer.DepthID},
		{Slot: "view_normals", ResourceID: renderer.ViewNormalsID},
	}); err != nil {
		return nil, nil, err
	}

	return pass, blurPass, nil
}

func newSsaoState(device *wgpu.Device, aspect func() float32) (*ssaoPassState, error) {
	module, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "ssao shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: ssaoShader},
	})
	if err != nil {
		return nil, fmt.Errorf("ssao: shader: %w", err)
	}
	defer module.Release()

	bgLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "ssao bg layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageFragment, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeDepth, ViewDimension: wgpu.TextureViewDimension2D}},
			{Binding: 1, Visibility: wgpu.ShaderStageFragment, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeFloat, ViewDimension: wgpu.TextureViewDimension2D}},
			{Binding: 2, Visibility: wgpu.ShaderStageFragment, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeFloat, ViewDimension: wgpu.TextureViewDimension2D}},
			{Binding: 3, Visibility: wgpu.ShaderStageFragment, Sampler: wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeNonFiltering}},
			{Binding: 4, Visibility: wgpu.ShaderStageFragment, Sampler: wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeFiltering}},
			{Binding: 5, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform}},
			{Binding: 6, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("ssao: bg layout: %w", err)
	}

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "ssao pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bgLayout},
	})
	if err != nil {
		return nil, fmt.Errorf("ssao: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	pipeline, err := device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "ssao pipeline",
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
		return nil, fmt.Errorf("ssao: pipeline: %w", err)
	}

	pointSampler, err := device.CreateSampler(&wgpu.SamplerDescriptor{
		Label:         "ssao point sampler",
		AddressModeU:  wgpu.AddressModeClampToEdge,
		AddressModeV:  wgpu.AddressModeClampToEdge,
		AddressModeW:  wgpu.AddressModeClampToEdge,
		MagFilter:     wgpu.FilterModeNearest,
		MinFilter:     wgpu.FilterModeNearest,
		MipmapFilter:  wgpu.MipmapFilterModeNearest,
		MaxAnisotropy: 1,
	})
	if err != nil {
		return nil, fmt.Errorf("ssao: point sampler: %w", err)
	}
	noiseSampler, err := device.CreateSampler(&wgpu.SamplerDescriptor{
		Label:         "ssao noise sampler",
		AddressModeU:  wgpu.AddressModeRepeat,
		AddressModeV:  wgpu.AddressModeRepeat,
		AddressModeW:  wgpu.AddressModeRepeat,
		MagFilter:     wgpu.FilterModeNearest,
		MinFilter:     wgpu.FilterModeNearest,
		MipmapFilter:  wgpu.MipmapFilterModeNearest,
		MaxAnisotropy: 1,
	})
	if err != nil {
		return nil, fmt.Errorf("ssao: noise sampler: %w", err)
	}

	paramsBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "ssao params",
		Size:  uint64(unsafe.Sizeof(ssaoParams{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("ssao: params buffer: %w", err)
	}

	kernel := buildSsaoKernel()
	kernelBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "ssao kernel",
		Size:  uint64(len(kernel)) * uint64(unsafe.Sizeof(mgl32.Vec4{})),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("ssao: kernel buffer: %w", err)
	}
	kernelBytes := sliceBytes(kernel)
	device.GetQueue().WriteBuffer(kernelBuffer, 0, kernelBytes)

	noise := buildSsaoNoise()
	noiseTex, err := device.CreateTexture(&wgpu.TextureDescriptor{
		Label:         "ssao noise",
		Size:          wgpu.Extent3D{Width: ssaoNoiseSize, Height: ssaoNoiseSize, DepthOrArrayLayers: 1},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     wgpu.TextureDimension2D,
		Format:        wgpu.TextureFormatRGBA8Unorm,
		Usage:         wgpu.TextureUsageTextureBinding | wgpu.TextureUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("ssao: noise texture: %w", err)
	}
	noiseView, err := noiseTex.CreateView(nil)
	if err != nil {
		return nil, fmt.Errorf("ssao: noise view: %w", err)
	}
	device.GetQueue().WriteTexture(
		&wgpu.ImageCopyTexture{Texture: noiseTex, Aspect: wgpu.TextureAspectAll},
		noise,
		&wgpu.TextureDataLayout{BytesPerRow: ssaoNoiseSize * 4, RowsPerImage: ssaoNoiseSize},
		&wgpu.Extent3D{Width: ssaoNoiseSize, Height: ssaoNoiseSize, DepthOrArrayLayers: 1},
	)

	return &ssaoPassState{
		pipeline:     pipeline,
		bgLayout:     bgLayout,
		pointSampler: pointSampler,
		noiseSampler: noiseSampler,
		paramsBuffer: paramsBuffer,
		kernelBuffer: kernelBuffer,
		noiseTexture: noiseTex,
		noiseView:    noiseView,
		aspectFn:     aspect,
	}, nil
}

func ssaoPrepare(state *ssaoPassState, context *render.PassContext) error {
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

	settings := render.DefaultGraphics().Ssao
	if g, ok := ecs.Resource[render.Graphics](context.World); ok && g != nil {
		settings = g.Ssao
	}

	enabled := float32(0)
	if settings.Enabled {
		enabled = 1
	}

	view, projection := ssaoViewProj(context, state.aspectFn())
	_ = view

	invProjection := projection.Inv()
	params := ssaoParams{
		Projection:    projection,
		InvProjection: invProjection,
		ScreenSize:    mgl32.Vec2{float32(width), float32(height)},
		Radius:        settings.Radius,
		Bias:          settings.Bias,
		Intensity:     settings.Intensity,
		Enabled:       enabled,
		SampleCount:   float32(settings.SampleCount),
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
	bg, err := context.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "ssao bg",
		Layout: state.bgLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, TextureView: depthView},
			{Binding: 1, TextureView: normalView},
			{Binding: 2, TextureView: state.noiseView},
			{Binding: 3, Sampler: state.pointSampler},
			{Binding: 4, Sampler: state.noiseSampler},
			{Binding: 5, Buffer: state.paramsBuffer, Offset: 0, Size: uint64(unsafe.Sizeof(ssaoParams{}))},
			{Binding: 6, Buffer: state.kernelBuffer, Offset: 0, Size: uint64(ssaoKernelSize) * uint64(unsafe.Sizeof(mgl32.Vec4{}))},
		},
	})
	if err != nil {
		return fmt.Errorf("ssao: bind group: %w", err)
	}
	state.bindGroup = bg
	return nil
}

func ssaoExecute(state *ssaoPassState, context *render.PassContext) error {
	if state.rawView == nil {
		return nil
	}
	passEnc := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		Label: "ssao",
		ColorAttachments: []wgpu.RenderPassColorAttachment{{
			View:       state.rawView,
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

func ssaoInvalidate(state *ssaoPassState) {
	if state.bindGroup != nil {
		state.bindGroup.Release()
		state.bindGroup = nil
	}
}

func ssaoRelease(state *ssaoPassState) {
	if state.bindGroup != nil {
		state.bindGroup.Release()
	}
	if state.rawView != nil {
		state.rawView.Release()
	}
	if state.rawTexture != nil {
		state.rawTexture.Release()
	}
	if state.noiseView != nil {
		state.noiseView.Release()
	}
	if state.noiseTexture != nil {
		state.noiseTexture.Release()
	}
	if state.kernelBuffer != nil {
		state.kernelBuffer.Release()
	}
	if state.paramsBuffer != nil {
		state.paramsBuffer.Release()
	}
	if state.noiseSampler != nil {
		state.noiseSampler.Release()
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

func (state *ssaoPassState) recreateRawTexture(device *wgpu.Device, width, height uint32) error {
	if state.rawView != nil {
		state.rawView.Release()
	}
	if state.rawTexture != nil {
		state.rawTexture.Release()
	}
	tex, err := device.CreateTexture(&wgpu.TextureDescriptor{
		Label:         "ssao raw",
		Size:          wgpu.Extent3D{Width: width, Height: height, DepthOrArrayLayers: 1},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     wgpu.TextureDimension2D,
		Format:        ssaoFormat,
		Usage:         wgpu.TextureUsageRenderAttachment | wgpu.TextureUsageTextureBinding,
	})
	if err != nil {
		return fmt.Errorf("ssao: raw texture: %w", err)
	}
	view, err := tex.CreateView(nil)
	if err != nil {
		tex.Release()
		return fmt.Errorf("ssao: raw view: %w", err)
	}
	state.rawTexture = tex
	state.rawView = view
	return nil
}

func ssaoSize(context *render.PassContext, slot string) (uint32, uint32, error) {
	id, ok := context.Slots[slot]
	if !ok {
		return 0, 0, fmt.Errorf("ssao: slot %q not bound", slot)
	}
	handle := context.Resources.Handles[id]
	return handle.Width, handle.Height, nil
}

func ssaoViewProj(context *render.PassContext, aspect float32) (mgl32.Mat4, mgl32.Mat4) {
	camera, ok := ecs.Resource[render.Camera](context.World)
	if !ok || camera == nil {
		return mgl32.Ident4(), mgl32.Ident4()
	}
	view := render.CameraView(camera)
	proj := render.CameraProjection(camera, aspect)
	return view, proj
}

type nightshadeLCG uint32

func (r *nightshadeLCG) nextFloat() float32 {
	*r = nightshadeLCG((uint32(*r) * 1103515245) + 12345)
	return float32(*r) / float32(math.MaxUint32)
}

func buildSsaoKernel() []mgl32.Vec4 {
	rng := nightshadeLCG(12345)
	kernel := make([]mgl32.Vec4, ssaoKernelSize)
	for index := 0; index < ssaoKernelSize; index++ {
		x := rng.nextFloat()*2 - 1
		y := rng.nextFloat()*2 - 1
		z := rng.nextFloat()*0.9 + 0.1
		sample := mgl32.Vec3{x, y, z}.Normalize()
		t := float32(index) / float32(ssaoKernelSize)
		scale := 0.1 + t*t*0.9
		sample = sample.Mul(scale)
		kernel[index] = mgl32.Vec4{sample.X(), sample.Y(), sample.Z(), 0}
	}
	return kernel
}

func buildSsaoNoise() []byte {
	rng := nightshadeLCG(54321)
	noise := make([]byte, 0, ssaoNoiseSize*ssaoNoiseSize*4)
	for index := 0; index < ssaoNoiseSize*ssaoNoiseSize; index++ {
		x := rng.nextFloat()*2 - 1
		y := rng.nextFloat()*2 - 1
		noise = append(noise, byte((x*0.5+0.5)*255), byte((y*0.5+0.5)*255), 128, 255)
	}
	return noise
}
