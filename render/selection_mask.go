package render

import (
	_ "embed"
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"

	"indigo/ecs"
)

//go:embed selection_mask.wgsl
var selectionMaskShader string

// SelectionMaskFormat is the texture format the selection mask is
// rendered into. Single-channel 8-bit unorm: 1.0 for "this pixel is
// covered by a selected entity," 0.0 otherwise.
const SelectionMaskFormat = wgpu.TextureFormatR8Unorm

// selectionMaskMaxEntities caps how many distinct entity IDs the
// shader's selection_bits buffer can address. The buffer is sized
// for this many entities upfront so the bind group stays static.
// Grow it if the engine starts spawning more than this number of
// entities in a single world.
const selectionMaskMaxEntities = 4096

const selectionMaskWordCount = selectionMaskMaxEntities / 32

type selectionMaskUniforms struct {
	BitWordCount uint32
	Pad0         uint32
	Pad1         uint32
	Pad2         uint32
}

type selectionMaskPassState struct {
	pipeline        *wgpu.RenderPipeline
	bindGroupLayout *wgpu.BindGroupLayout
	bitsBuffer      *wgpu.Buffer
	uniformsBuffer  *wgpu.Buffer
	bindGroup       *wgpu.BindGroup

	bitsetScratch [selectionMaskWordCount]uint32
}

// NewSelectionMaskPass builds the selection-mask pass: a fullscreen
// postprocess that samples the entity_id texture per pixel, tests
// each pixel's entity ID against a CPU-uploaded bitset of selected
// entities, and writes 1.0 to the selection_mask target where the
// test passes.
//
// Mirrors nightshade's selection-mask shader. Reuses the entity_id
// texture the mesh pass already writes for picking, so the only
// per-frame upload is the small bitset.
func NewSelectionMaskPass(device *wgpu.Device) (*Pass, error) {
	state := &selectionMaskPassState{}

	bindGroupLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "selection_mask bind group layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: wgpu.ShaderStageFragment,
				Texture: wgpu.TextureBindingLayout{
					SampleType:    wgpu.TextureSampleTypeUint,
					ViewDimension: wgpu.TextureViewDimension2D,
				},
			},
			{
				Binding:    1,
				Visibility: wgpu.ShaderStageFragment,
				Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage},
			},
			{
				Binding:    2,
				Visibility: wgpu.ShaderStageFragment,
				Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform},
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("selection_mask: bind group layout: %w", err)
	}
	state.bindGroupLayout = bindGroupLayout

	bitsBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "selection_mask bits",
		Size:  uint64(selectionMaskWordCount * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("selection_mask: bits buffer: %w", err)
	}
	state.bitsBuffer = bitsBuffer

	uniformsBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "selection_mask uniforms",
		Size:  16,
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("selection_mask: uniforms buffer: %w", err)
	}
	state.uniformsBuffer = uniformsBuffer

	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "selection_mask shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: selectionMaskShader},
	})
	if err != nil {
		return nil, fmt.Errorf("selection_mask: shader: %w", err)
	}
	defer shader.Release()

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "selection_mask pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bindGroupLayout},
	})
	if err != nil {
		return nil, fmt.Errorf("selection_mask: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	pipeline, err := device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "selection_mask pipeline",
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
				Format:    SelectionMaskFormat,
				WriteMask: wgpu.ColorWriteMaskAll,
			}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("selection_mask: pipeline: %w", err)
	}
	state.pipeline = pipeline

	return &Pass{
		Name:                 "selection_mask",
		Reads:                []string{"entity_id"},
		Writes:               []string{"selection_mask"},
		State:                state,
		Prepare:              selectionMaskPrepare,
		Execute:              selectionMaskExecute,
		InvalidateBindGroups: selectionMaskInvalidate,
		Release:              selectionMaskRelease,
	}, nil
}

func selectionMaskPrepare(s any, context *PassContext) error {
	state := s.(*selectionMaskPassState)

	for i := range state.bitsetScratch {
		state.bitsetScratch[i] = 0
	}

	selectedMask := ecs.MaskOf[Selected](context.World)
	context.World.ForEach(selectedMask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		id := entity.ID
		word := id / 32
		bit := id & 31
		if int(word) >= len(state.bitsetScratch) {
			return
		}
		state.bitsetScratch[word] |= 1 << bit
	})

	writeBuffer(context.Device, context.Queue, context.Encoder, state.bitsBuffer, 0, bytesOf(&state.bitsetScratch))

	uniforms := selectionMaskUniforms{BitWordCount: uint32(selectionMaskWordCount)}
	writeBuffer(context.Device, context.Queue, context.Encoder, state.uniformsBuffer, 0, bytesOf(&uniforms))

	return nil
}

func selectionMaskExecute(s any, context *PassContext) error {
	state := s.(*selectionMaskPassState)

	if state.bindGroup == nil {
		entityIdView, err := context.TextureView("entity_id")
		if err != nil {
			return err
		}
		group, err := context.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  "selection_mask bind group",
			Layout: state.bindGroupLayout,
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, TextureView: entityIdView},
				{Binding: 1, Buffer: state.bitsBuffer, Offset: 0, Size: wgpu.WholeSize},
				{Binding: 2, Buffer: state.uniformsBuffer, Offset: 0, Size: 16},
			},
		})
		if err != nil {
			return fmt.Errorf("selection_mask: bind group: %w", err)
		}
		state.bindGroup = group
	}

	colorAttachment, err := context.ColorAttachment("selection_mask")
	if err != nil {
		return err
	}
	pass := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		Label:            "selection_mask",
		ColorAttachments: []wgpu.RenderPassColorAttachment{colorAttachment},
	})
	pass.SetPipeline(state.pipeline)
	pass.SetBindGroup(0, state.bindGroup, nil)
	pass.Draw(3, 1, 0, 0)
	pass.End()
	pass.Release()
	return nil
}

func selectionMaskInvalidate(s any) {
	state := s.(*selectionMaskPassState)
	if state.bindGroup != nil {
		state.bindGroup.Release()
		state.bindGroup = nil
	}
}

func selectionMaskRelease(s any) {
	state := s.(*selectionMaskPassState)
	if state.bindGroup != nil {
		state.bindGroup.Release()
	}
	if state.pipeline != nil {
		state.pipeline.Release()
	}
	if state.uniformsBuffer != nil {
		state.uniformsBuffer.Release()
	}
	if state.bitsBuffer != nil {
		state.bitsBuffer.Release()
	}
	if state.bindGroupLayout != nil {
		state.bindGroupLayout.Release()
	}
}
