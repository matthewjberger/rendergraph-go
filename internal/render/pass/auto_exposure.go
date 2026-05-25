package pass

import (
	_ "embed"
	"fmt"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"

	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/window"
)

//go:embed auto_exposure.wgsl
var autoExposureShader string

// autoExposureBuffer keeps the CPU-written fields (AdaptationRate, DeltaTime)
// first so the per-frame param update is a single contiguous write. The
// remaining fields are maintained by the compute shader across frames, so the
// CPU must not overwrite them.
type autoExposureBuffer struct {
	AdaptationRate      float32
	DeltaTime           float32
	TargetLogLuminance  float32
	CurrentLogLuminance float32
	Primed              uint32
	Pad0                uint32
	Pad1                uint32
	Pad2                uint32
}

type AutoExposureSettings struct {
	Enabled         bool
	AdaptationRate  float32
	TargetLuminance float32
	MinScale        float32
	MaxScale        float32
}

func DefaultAutoExposureSettings() AutoExposureSettings {
	return AutoExposureSettings{
		Enabled:         true,
		AdaptationRate:  1.5,
		TargetLuminance: 0.5,
		MinScale:        0.25,
		MaxScale:        4.0,
	}
}

type AutoExposureResource struct {
	Buffer *wgpu.Buffer
}

type autoExposurePassState struct {
	pipeline *wgpu.ComputePipeline
	bgLayout *wgpu.BindGroupLayout
	buffer   *wgpu.Buffer
	bg       *wgpu.BindGroup
	lastView *wgpu.TextureView
}

func AddAutoExposurePass(renderer *render.Renderer) (*render.Pass, error) {
	state, err := newAutoExposureState(renderer.Device)
	if err != nil {
		return nil, err
	}
	pass := &render.Pass{
		Name:                 "auto_exposure",
		Reads:                []string{"scene_color"},
		Prepare:              func(c *render.PassContext) error { return autoExposurePrepare(state, c) },
		Execute:              func(c *render.PassContext) error { return autoExposureExecute(state, c) },
		InvalidateBindGroups: func() { autoExposureInvalidate(state) },
		Release:              func() { autoExposureRelease(state) },
	}
	if err := renderer.Graph.AddPass(pass, []render.SlotBinding{
		{Slot: "scene_color", ResourceID: renderer.SceneColorID},
	}); err != nil {
		return nil, err
	}
	return pass, nil
}

func newAutoExposureState(device *wgpu.Device) (*autoExposurePassState, error) {
	module, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "auto_exposure shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: autoExposureShader},
	})
	if err != nil {
		return nil, fmt.Errorf("auto_exposure: shader: %w", err)
	}
	defer module.Release()

	bgLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "auto_exposure bg layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeUnfilterableFloat, ViewDimension: wgpu.TextureViewDimension2D}},
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("auto_exposure: bg layout: %w", err)
	}

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "auto_exposure pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bgLayout},
	})
	if err != nil {
		return nil, fmt.Errorf("auto_exposure: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	pipeline, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Layout: pipelineLayout,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     module,
			EntryPoint: "main",
		},
	})
	if err != nil {
		return nil, fmt.Errorf("auto_exposure: pipeline: %w", err)
	}

	buffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "auto_exposure buffer",
		Size:  uint64(unsafe.Sizeof(autoExposureBuffer{})),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageUniform,
	})
	if err != nil {
		return nil, fmt.Errorf("auto_exposure: buffer: %w", err)
	}

	return &autoExposurePassState{
		pipeline: pipeline,
		bgLayout: bgLayout,
		buffer:   buffer,
	}, nil
}

func autoExposurePrepare(state *autoExposurePassState, context *render.PassContext) error {
	settings := DefaultAutoExposureSettings()
	if resource, ok := ecs.Resource[AutoExposureSettings](context.World); ok && resource != nil {
		settings = *resource
	}
	delta := float32(0.016)
	if win, ok := ecs.Resource[window.Window](context.World); ok && win != nil {
		delta = win.Timing.DeltaSeconds
	}
	header := autoExposureBuffer{
		AdaptationRate: settings.AdaptationRate,
		DeltaTime:      delta,
	}

	// AdaptationRate and DeltaTime are the first two fields, so this single
	// write updates exactly the CPU-controlled params while leaving the
	// shader-maintained luminance fields intact. Go through the staging helper
	// rather than queue.WriteBuffer: on wasm the latter builds a view over the
	// Go heap, which is invalidated once the heap grows (e.g. texture streaming),
	// crashing with "Number of bytes to write is too large".
	writeBuffer(context.Device, context.Queue, context.Encoder, state.buffer, 0, bytesOfN(&header, 8))

	sceneView, err := context.TextureView("scene_color")
	if err != nil {
		return err
	}
	if state.bg != nil && state.lastView == sceneView {
		ecsSetAutoExposureResource(context, state.buffer)
		return nil
	}
	if state.bg != nil {
		state.bg.Release()
	}
	bg, err := context.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "auto_exposure bg",
		Layout: state.bgLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, TextureView: sceneView},
			{Binding: 1, Buffer: state.buffer, Offset: 0, Size: uint64(unsafe.Sizeof(autoExposureBuffer{}))},
		},
	})
	if err != nil {
		return fmt.Errorf("auto_exposure: bind group: %w", err)
	}
	state.bg = bg
	state.lastView = sceneView
	ecsSetAutoExposureResource(context, state.buffer)
	return nil
}

func autoExposureExecute(state *autoExposurePassState, context *render.PassContext) error {
	if state.bg == nil {
		return nil
	}
	passEnc := context.Encoder.BeginComputePass(nil)
	passEnc.SetPipeline(state.pipeline)
	passEnc.SetBindGroup(0, state.bg, nil)
	passEnc.DispatchWorkgroups(1, 1, 1)
	passEnc.End()
	passEnc.Release()
	return nil
}

func autoExposureInvalidate(state *autoExposurePassState) {
	if state.bg != nil {
		state.bg.Release()
		state.bg = nil
	}
	state.lastView = nil
}

func autoExposureRelease(state *autoExposurePassState) {
	if state.bg != nil {
		state.bg.Release()
	}
	if state.buffer != nil {
		state.buffer.Release()
	}
	if state.pipeline != nil {
		state.pipeline.Release()
	}
	if state.bgLayout != nil {
		state.bgLayout.Release()
	}
}

func ecsSetAutoExposureResource(context *render.PassContext, buffer *wgpu.Buffer) {
	if buffer == nil {
		return
	}
	if resource, ok := ecs.Resource[AutoExposureResource](context.World); ok && resource != nil {
		resource.Buffer = buffer
		return
	}
	ecs.SetResource(context.World, AutoExposureResource{Buffer: buffer})
}
