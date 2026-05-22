package pass

import (
	_ "embed"
	"fmt"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"

	"indigo/ecs"
	"indigo/render"
	"indigo/window"
)

//go:embed fxaa.wgsl
var fxaaShader string

// fxaaParams matches the WGSL FxaaParams struct (texel + scalar
// settings). 16 bytes total, no padding issues.
type fxaaParams struct {
	TexelX          float32
	TexelY          float32
	Enabled         float32
	SubpixelQuality float32
}

type fxaaPassState struct {
	pipeline        *wgpu.RenderPipeline
	bindGroupLayout *wgpu.BindGroupLayout
	sampler         *wgpu.Sampler
	paramsBuffer    *wgpu.Buffer
	bindGroup       *wgpu.BindGroup
}

// NewFxaaPass builds the engine's FXAA antialiasing pass: sample
// "input" with a fullscreen blit, run edge-detect + subpixel-blend,
// write to "output". Mirrors nightshade's FxaaPass. The cached bind
// group is rebuilt whenever the input texture view changes via the
// render graph's [render.Pass.InvalidateBindGroups] hook, which fires on
// resize and any other resource-version bump.
func NewFxaaPass(device *wgpu.Device, surfaceFormat wgpu.TextureFormat) (*render.Pass, error) {
	state := &fxaaPassState{}

	bindGroupLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "fxaa bind group layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: wgpu.ShaderStageFragment,
				Texture: wgpu.TextureBindingLayout{
					SampleType:    wgpu.TextureSampleTypeFloat,
					ViewDimension: wgpu.TextureViewDimension2D,
					Multisampled:  false,
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
		},
	})
	if err != nil {
		return nil, fmt.Errorf("fxaa pass: bind group layout: %w", err)
	}
	state.bindGroupLayout = bindGroupLayout

	sampler, err := device.CreateSampler(&wgpu.SamplerDescriptor{
		Label:         "fxaa sampler",
		AddressModeU:  wgpu.AddressModeClampToEdge,
		AddressModeV:  wgpu.AddressModeClampToEdge,
		AddressModeW:  wgpu.AddressModeClampToEdge,
		MagFilter:     wgpu.FilterModeLinear,
		MinFilter:     wgpu.FilterModeLinear,
		MipmapFilter:  wgpu.MipmapFilterModeNearest,
		MaxAnisotropy: 1,
	})
	if err != nil {
		return nil, fmt.Errorf("fxaa pass: sampler: %w", err)
	}
	state.sampler = sampler

	paramsBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "fxaa params buffer",
		Size:  uint64(unsafe.Sizeof(fxaaParams{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("fxaa pass: params buffer: %w", err)
	}
	state.paramsBuffer = paramsBuffer

	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "fxaa shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: fxaaShader},
	})
	if err != nil {
		return nil, fmt.Errorf("fxaa pass: shader: %w", err)
	}
	defer shader.Release()

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "fxaa pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bindGroupLayout},
	})
	if err != nil {
		return nil, fmt.Errorf("fxaa pass: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	pipeline, err := device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "fxaa pipeline",
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
		return nil, fmt.Errorf("fxaa pass: pipeline: %w", err)
	}
	state.pipeline = pipeline

	return &render.Pass{
		Name:                 "fxaa",
		Reads:                []string{"input"},
		Writes:               []string{"output"},
		State:                state,
		Prepare:              fxaaPrepare,
		Execute:              fxaaExecute,
		InvalidateBindGroups: fxaaInvalidateBindGroups,
		Release:              fxaaRelease,
	}, nil
}

func fxaaPrepare(s any, context *render.PassContext) error {
	state := s.(*fxaaPassState)
	viewport := ecs.MustResource[window.Window](context.World).Viewport
	settings := ecs.MustResource[render.GraphicsSettings](context.World)
	width := float32(viewport.Width)
	height := float32(viewport.Height)
	if width <= 0 {
		width = 1
	}
	if height <= 0 {
		height = 1
	}
	enabled := float32(0)
	if settings.FxaaEnabled {
		enabled = 1
	}
	params := fxaaParams{
		TexelX:          1.0 / width,
		TexelY:          1.0 / height,
		Enabled:         enabled,
		SubpixelQuality: 0.75,
	}
	writeBuffer(context.Device, context.Queue, context.Encoder, state.paramsBuffer, 0, bytesOf(&params))
	return nil
}

func fxaaExecute(s any, context *render.PassContext) error {
	state := s.(*fxaaPassState)

	if state.bindGroup == nil {
		inputView, err := context.TextureView("input")
		if err != nil {
			return err
		}
		group, err := context.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  "fxaa bind group",
			Layout: state.bindGroupLayout,
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, TextureView: inputView},
				{Binding: 1, Sampler: state.sampler},
				{Binding: 2, Buffer: state.paramsBuffer, Offset: 0, Size: uint64(unsafe.Sizeof(fxaaParams{}))},
			},
		})
		if err != nil {
			return fmt.Errorf("fxaa pass: bind group: %w", err)
		}
		state.bindGroup = group
	}

	colorAttachment, err := context.ColorAttachment("output")
	if err != nil {
		return err
	}

	pass := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		Label:            "fxaa",
		ColorAttachments: []wgpu.RenderPassColorAttachment{colorAttachment},
	})
	pass.SetPipeline(state.pipeline)
	pass.SetBindGroup(0, state.bindGroup, nil)
	pass.Draw(3, 1, 0, 0)
	pass.End()
	pass.Release()
	return nil
}

func fxaaInvalidateBindGroups(s any) {
	state := s.(*fxaaPassState)
	if state.bindGroup != nil {
		state.bindGroup.Release()
		state.bindGroup = nil
	}
}

func fxaaRelease(s any) {
	state := s.(*fxaaPassState)
	if state.bindGroup != nil {
		state.bindGroup.Release()
	}
	if state.paramsBuffer != nil {
		state.paramsBuffer.Release()
	}
	if state.sampler != nil {
		state.sampler.Release()
	}
	if state.pipeline != nil {
		state.pipeline.Release()
	}
	if state.bindGroupLayout != nil {
		state.bindGroupLayout.Release()
	}
}
