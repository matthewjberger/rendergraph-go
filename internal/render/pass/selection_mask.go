package pass

import (
	_ "embed"
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"

	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/transform"
)

//go:embed selection_mask.wgsl
var selectionMaskShader string

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

func NewSelectionMaskPass(device *wgpu.Device) (*render.Pass, error) {
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
				Format:    render.SelectionMaskFormat,
				WriteMask: wgpu.ColorWriteMaskAll,
			}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("selection_mask: pipeline: %w", err)
	}
	state.pipeline = pipeline

	return &render.Pass{
		Name:                 "selection_mask",
		Reads:                []string{"entity_id"},
		Writes:               []string{"selection_mask"},
		Prepare:              func(c *render.PassContext) error { return selectionMaskPrepare(state, c) },
		Execute:              func(c *render.PassContext) error { return selectionMaskExecute(state, c) },
		InvalidateBindGroups: func() { selectionMaskInvalidate(state) },
		Release:              func() { selectionMaskRelease(state) },
	}, nil
}

func setSelectionBit(bits []uint32, id uint32) {
	word := id / 32
	bit := id & 31
	if int(word) >= len(bits) {
		return
	}
	bits[word] |= 1 << bit
}

func collectDescendants(world *ecs.World, root ecs.Entity) []ecs.Entity {
	stack := []ecs.Entity{root}
	var out []ecs.Entity
	for len(stack) > 0 {
		current := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		for _, child := range transform.QueryChildren(world, current) {
			out = append(out, child)
			stack = append(stack, child)
		}
	}
	return out
}

func selectionMaskPrepare(state *selectionMaskPassState, context *render.PassContext) error {

	for i := range state.bitsetScratch {
		state.bitsetScratch[i] = 0
	}

	selectedMask := ecs.MustMaskOf[render.Selected](context.World)
	context.World.ForEach(selectedMask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		setSelectionBit(state.bitsetScratch[:], entity.ID)
		for _, descendant := range collectDescendants(context.World, entity) {
			setSelectionBit(state.bitsetScratch[:], descendant.ID)
		}
	})

	writeBuffer(context.Device, context.Queue, context.Encoder, state.bitsBuffer, 0, bytesOf(&state.bitsetScratch))

	uniforms := selectionMaskUniforms{BitWordCount: uint32(selectionMaskWordCount)}
	writeBuffer(context.Device, context.Queue, context.Encoder, state.uniformsBuffer, 0, bytesOf(&uniforms))

	return nil
}

func selectionMaskExecute(state *selectionMaskPassState, context *render.PassContext) error {

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

func selectionMaskInvalidate(state *selectionMaskPassState) {
	if state.bindGroup != nil {
		state.bindGroup.Release()
		state.bindGroup = nil
	}
}

func selectionMaskRelease(state *selectionMaskPassState) {
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
