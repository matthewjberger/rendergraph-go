package pass

import (
	_ "embed"
	"fmt"
	"github.com/matthewjberger/indigo/render"

	"github.com/cogentcore/webgpu/wgpu"
)

//go:embed present.wgsl
var presentShader string

type presentPassState struct {
	pipeline        *wgpu.RenderPipeline
	bindGroupLayout *wgpu.BindGroupLayout
	sampler         *wgpu.Sampler
	bindGroup       *wgpu.BindGroup
}

func NewPresentPass(device *wgpu.Device, surfaceFormat wgpu.TextureFormat) (*render.Pass, error) {
	state := &presentPassState{}

	bindGroupLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "present bind group layout",
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
				Sampler: wgpu.SamplerBindingLayout{
					Type: wgpu.SamplerBindingTypeFiltering,
				},
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("present pass: bind group layout: %w", err)
	}
	state.bindGroupLayout = bindGroupLayout

	sampler, err := device.CreateSampler(&wgpu.SamplerDescriptor{
		Label:         "present sampler",
		AddressModeU:  wgpu.AddressModeClampToEdge,
		AddressModeV:  wgpu.AddressModeClampToEdge,
		AddressModeW:  wgpu.AddressModeClampToEdge,
		MagFilter:     wgpu.FilterModeLinear,
		MinFilter:     wgpu.FilterModeLinear,
		MipmapFilter:  wgpu.MipmapFilterModeNearest,
		MaxAnisotropy: 1,
	})
	if err != nil {
		return nil, fmt.Errorf("present pass: sampler: %w", err)
	}
	state.sampler = sampler

	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "present shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: presentShader},
	})
	if err != nil {
		return nil, fmt.Errorf("present pass: shader: %w", err)
	}
	defer shader.Release()

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "present pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bindGroupLayout},
	})
	if err != nil {
		return nil, fmt.Errorf("present pass: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	pipeline, err := device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "present pipeline",
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
		return nil, fmt.Errorf("present pass: pipeline: %w", err)
	}
	state.pipeline = pipeline

	return &render.Pass{
		Name:                 "present",
		Reads:                []string{"input"},
		Writes:               []string{"output"},
		Execute:              func(c *render.PassContext) error { return presentExecute(state, c) },
		InvalidateBindGroups: func() { presentInvalidateBindGroups(state) },
		Release:              func() { presentRelease(state) },
	}, nil
}

func presentExecute(state *presentPassState, context *render.PassContext) error {

	if state.bindGroup == nil {
		inputView, err := context.TextureView("input")
		if err != nil {
			return err
		}
		group, err := context.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  "present bind group",
			Layout: state.bindGroupLayout,
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, TextureView: inputView},
				{Binding: 1, Sampler: state.sampler},
			},
		})
		if err != nil {
			return fmt.Errorf("present pass: bind group: %w", err)
		}
		state.bindGroup = group
	}

	colorAttachment, err := context.ColorAttachment("output")
	if err != nil {
		return err
	}

	pass := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		Label:            "present",
		ColorAttachments: []wgpu.RenderPassColorAttachment{colorAttachment},
	})
	pass.SetPipeline(state.pipeline)
	pass.SetBindGroup(0, state.bindGroup, nil)
	pass.Draw(3, 1, 0, 0)
	pass.End()
	pass.Release()
	return nil
}

func presentInvalidateBindGroups(state *presentPassState) {
	if state.bindGroup != nil {
		state.bindGroup.Release()
		state.bindGroup = nil
	}
}

func presentRelease(state *presentPassState) {
	if state.bindGroup != nil {
		state.bindGroup.Release()
	}
	if state.pipeline != nil {
		state.pipeline.Release()
	}
	if state.sampler != nil {
		state.sampler.Release()
	}
	if state.bindGroupLayout != nil {
		state.bindGroupLayout.Release()
	}
}
