package pass

import (
	_ "embed"
	"fmt"
	"math"
	"math/rand"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/go-gl/mathgl/mgl32"

	"indigo/ecs"
	"indigo/render"
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

// ssaoFormat is the output format of the SSAO + SSAO blur passes.
// R8Unorm is enough for a single-channel occlusion factor and saves
// bandwidth.
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

// SsaoSettings drives the SSAO pass each frame. Stored as an ECS
// resource so any system or future editor UI can tune it without
// touching pass internals.
type SsaoSettings struct {
	Enabled     bool
	Radius      float32
	Bias        float32
	Intensity   float32
	SampleCount int
}

// DefaultSsaoSettings matches the reference engine's defaults: a
// fairly tight radius with a moderate sample count.
func DefaultSsaoSettings() SsaoSettings {
	return SsaoSettings{
		Enabled:     true,
		Radius:      0.5,
		Bias:        0.025,
		Intensity:   1.5,
		SampleCount: 32,
	}
}

// SsaoResult exposes the blurred occlusion texture so postprocess
// can sample it.
type SsaoResult struct {
	View *wgpu.TextureView
}

// SsaoResource wraps the blurred-AO view for the postprocess pass to
// pick up by ECS resource lookup.
type SsaoResource struct {
	Result *SsaoResult
}

// AddSsaoPass registers a postprocess SSAO pass: depth + view_normal
// in, R8 occlusion out. The blurred output is exposed via SsaoResource
// so the tonemap can multiply it into the HDR scene color.
func AddSsaoPass(renderer *render.Renderer, aspect func() float32) (*render.Pass, *render.Pass, error) {
	state, err := newSsaoState(renderer.Device, aspect)
	if err != nil {
		return nil, nil, err
	}

	pass := &render.Pass{
		Name:    "ssao",
		Reads:   []string{"depth", "view_normals"},
		State:   state,
		Prepare: ssaoPrepare,
		Execute: ssaoExecute,
		Release: ssaoRelease,
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
		Name:    "ssao_blur",
		Reads:   []string{"depth", "view_normals"},
		State:   blurState,
		Prepare: ssaoBlurPrepare,
		Execute: ssaoBlurExecute,
		Release: ssaoBlurRelease,
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
	kernelBytes := unsafe.Slice((*byte)(unsafe.Pointer(&kernel[0])), len(kernel)*16)
	device.GetQueue().WriteBuffer(kernelBuffer, 0, kernelBytes)

	noise := buildSsaoNoise()
	noiseTex, err := device.CreateTexture(&wgpu.TextureDescriptor{
		Label:         "ssao noise",
		Size:          wgpu.Extent3D{Width: ssaoNoiseSize, Height: ssaoNoiseSize, DepthOrArrayLayers: 1},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     wgpu.TextureDimension2D,
		Format:        wgpu.TextureFormatRGBA16Float,
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
		unsafe.Slice((*byte)(unsafe.Pointer(&noise[0])), len(noise)*2),
		&wgpu.TextureDataLayout{BytesPerRow: ssaoNoiseSize * 8, RowsPerImage: ssaoNoiseSize},
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

func ssaoPrepare(s any, context *render.PassContext) error {
	state := s.(*ssaoPassState)
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

	settings := DefaultSsaoSettings()
	if resource, ok := ecs.Resource[SsaoSettings](context.World); ok {
		settings = *resource
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

func ssaoExecute(s any, context *render.PassContext) error {
	state := s.(*ssaoPassState)
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

func ssaoRelease(s any) {
	state := s.(*ssaoPassState)
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

// buildSsaoKernel returns a hemisphere-distributed set of unit-ish
// vectors weighted toward the surface plane, matching the reference
// engine's SSAO kernel.
func buildSsaoKernel() []mgl32.Vec4 {
	rng := rand.New(rand.NewSource(1))
	kernel := make([]mgl32.Vec4, ssaoKernelSize)
	for index := 0; index < ssaoKernelSize; index++ {
		sample := mgl32.Vec3{
			float32(rng.Float64()*2 - 1),
			float32(rng.Float64()*2 - 1),
			float32(rng.Float64()),
		}.Normalize()
		scale := float32(index) / float32(ssaoKernelSize)
		scale = lerpFloat(0.1, 1.0, scale*scale)
		sample = sample.Mul(float32(rng.Float64()) * scale)
		kernel[index] = mgl32.Vec4{sample.X(), sample.Y(), sample.Z(), 0}
	}
	return kernel
}

func lerpFloat(a, b, t float32) float32 {
	return a + (b-a)*t
}

// buildSsaoNoise returns a 4x4 random-rotation noise pattern, packed
// as Rgba16Float values that the shader expands to (-1, 1) in xy.
func buildSsaoNoise() []uint16 {
	rng := rand.New(rand.NewSource(2))
	noise := make([]uint16, ssaoNoiseSize*ssaoNoiseSize*4)
	for index := 0; index < ssaoNoiseSize*ssaoNoiseSize; index++ {
		x := float32(rng.Float64()*2 - 1)
		y := float32(rng.Float64()*2 - 1)
		noise[index*4+0] = float16FromFloat32(x*0.5 + 0.5)
		noise[index*4+1] = float16FromFloat32(y*0.5 + 0.5)
		noise[index*4+2] = float16FromFloat32(0.5)
		noise[index*4+3] = float16FromFloat32(1.0)
	}
	return noise
}

// float16FromFloat32 encodes a float32 as IEEE 754 binary16. Used to
// upload the SSAO noise pattern as Rgba16Float without pulling in a
// half-float dependency.
func float16FromFloat32(value float32) uint16 {
	bits := math.Float32bits(value)
	sign := uint16((bits >> 16) & 0x8000)
	mant := bits & 0x7fffff
	exp := int32((bits >> 23) & 0xff)
	if exp == 0xff {
		if mant != 0 {
			return sign | 0x7e00
		}
		return sign | 0x7c00
	}
	exp -= 127 - 15
	if exp >= 0x1f {
		return sign | 0x7c00
	}
	if exp <= 0 {
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint32(14 - exp)
		mant = mant >> shift
		return sign | uint16(mant)
	}
	return sign | uint16(exp<<10) | uint16(mant>>13)
}
