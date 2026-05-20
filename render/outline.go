package render

import (
	_ "embed"
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"
)

//go:embed outline.wgsl
var outlineShader string

// OutlineColor is the default outline tint applied to selected
// entities. Adapted from nightshade's editor outline color.
var OutlineColor = [4]float32{1.0, 0.45, 0.0, 1.0}

// OutlineWidth is the default outline thickness, measured in pixels
// of the selection-mask texture.
const OutlineWidth float32 = 2.0

type outlineParams struct {
	OutlineColor  [4]float32
	OutlineWidth  float32
	TextureWidth  float32
	TextureHeight float32
	Pad           float32
}

type outlinePassState struct {
	pipeline        *wgpu.RenderPipeline
	bindGroupLayout *wgpu.BindGroupLayout
	sampler         *wgpu.Sampler
	paramsBuffer    *wgpu.Buffer
	bindGroup       *wgpu.BindGroup

	color [4]float32
	width float32

	aspectFn func() (uint32, uint32)
}

// NewOutlinePass builds the selection-outline post-process pass.
// Reads the selection_mask texture, performs an 8-neighbor max-pool
// dilation, and writes pixels along the dilated boundary into the
// scene_color attachment using alpha blending.
//
// Mirrors nightshade's editor outline shader.
func NewOutlinePass(device *wgpu.Device, surfaceFormat wgpu.TextureFormat, viewportSize func() (uint32, uint32)) (*Pass, error) {
	state := &outlinePassState{
		color:    OutlineColor,
		width:    OutlineWidth,
		aspectFn: viewportSize,
	}

	bindGroupLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "outline bind group layout",
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
		},
	})
	if err != nil {
		return nil, fmt.Errorf("outline: bind group layout: %w", err)
	}
	state.bindGroupLayout = bindGroupLayout

	sampler, err := device.CreateSampler(&wgpu.SamplerDescriptor{
		Label:         "outline mask sampler",
		AddressModeU:  wgpu.AddressModeClampToEdge,
		AddressModeV:  wgpu.AddressModeClampToEdge,
		AddressModeW:  wgpu.AddressModeClampToEdge,
		MagFilter:     wgpu.FilterModeLinear,
		MinFilter:     wgpu.FilterModeLinear,
		MipmapFilter:  wgpu.MipmapFilterModeNearest,
		MaxAnisotropy: 1,
	})
	if err != nil {
		return nil, fmt.Errorf("outline: sampler: %w", err)
	}
	state.sampler = sampler

	paramsBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "outline params",
		Size:  32,
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("outline: params buffer: %w", err)
	}
	state.paramsBuffer = paramsBuffer

	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "outline shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: outlineShader},
	})
	if err != nil {
		return nil, fmt.Errorf("outline: shader: %w", err)
	}
	defer shader.Release()

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "outline pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bindGroupLayout},
	})
	if err != nil {
		return nil, fmt.Errorf("outline: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	pipeline, err := device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "outline pipeline",
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
				Blend: &wgpu.BlendState{
					Color: wgpu.BlendComponent{
						SrcFactor: wgpu.BlendFactorSrcAlpha,
						DstFactor: wgpu.BlendFactorOneMinusSrcAlpha,
						Operation: wgpu.BlendOperationAdd,
					},
					Alpha: wgpu.BlendComponent{
						SrcFactor: wgpu.BlendFactorOne,
						DstFactor: wgpu.BlendFactorOneMinusSrcAlpha,
						Operation: wgpu.BlendOperationAdd,
					},
				},
			}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("outline: pipeline: %w", err)
	}
	state.pipeline = pipeline

	return &Pass{
		Name:                 "outline",
		Reads:                []string{"selection_mask"},
		Writes:               []string{"color"},
		State:                state,
		Prepare:              outlinePrepare,
		Execute:              outlineExecute,
		InvalidateBindGroups: outlineInvalidate,
		Release:              outlineRelease,
	}, nil
}

func outlinePrepare(s any, context *PassContext) error {
	state := s.(*outlinePassState)

	width, height := state.aspectFn()
	params := outlineParams{
		OutlineColor:  state.color,
		OutlineWidth:  state.width,
		TextureWidth:  float32(width),
		TextureHeight: float32(height),
	}
	writeBuffer(context.Device, context.Queue, context.Encoder, state.paramsBuffer, 0, bytesOf(&params))
	return nil
}

func outlineExecute(s any, context *PassContext) error {
	state := s.(*outlinePassState)

	if state.bindGroup == nil {
		maskView, err := context.TextureView("selection_mask")
		if err != nil {
			return err
		}
		group, err := context.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  "outline bind group",
			Layout: state.bindGroupLayout,
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, TextureView: maskView},
				{Binding: 1, Sampler: state.sampler},
				{Binding: 2, Buffer: state.paramsBuffer, Offset: 0, Size: 32},
			},
		})
		if err != nil {
			return fmt.Errorf("outline: bind group: %w", err)
		}
		state.bindGroup = group
	}

	colorAttachment, err := context.ColorAttachment("color")
	if err != nil {
		return err
	}
	pass := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		Label:            "outline",
		ColorAttachments: []wgpu.RenderPassColorAttachment{colorAttachment},
	})
	pass.SetPipeline(state.pipeline)
	pass.SetBindGroup(0, state.bindGroup, nil)
	pass.Draw(3, 1, 0, 0)
	pass.End()
	pass.Release()
	return nil
}

func outlineInvalidate(s any) {
	state := s.(*outlinePassState)
	if state.bindGroup != nil {
		state.bindGroup.Release()
		state.bindGroup = nil
	}
}

func outlineRelease(s any) {
	state := s.(*outlinePassState)
	if state.bindGroup != nil {
		state.bindGroup.Release()
	}
	if state.pipeline != nil {
		state.pipeline.Release()
	}
	if state.paramsBuffer != nil {
		state.paramsBuffer.Release()
	}
	if state.sampler != nil {
		state.sampler.Release()
	}
	if state.bindGroupLayout != nil {
		state.bindGroupLayout.Release()
	}
}
