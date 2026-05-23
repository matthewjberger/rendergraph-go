package pass

import (
	_ "embed"
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"

	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/ui"
)

//go:embed ui_quad.wgsl
var uiQuadShader string

const uiQuadInstanceBytes = uint64(32)
const uiQuadMinCapacity uint32 = 64

type uiQuadInstance struct {
	Rect  [4]float32
	Color [4]float32
}

type uiQuadViewport struct {
	Width  float32
	Height float32
	Pad0   float32
	Pad1   float32
}

type uiQuadPassState struct {
	pipeline        *wgpu.RenderPipeline
	bindGroupLayout *wgpu.BindGroupLayout
	viewportBuffer  *wgpu.Buffer
	instanceBuffer  *wgpu.Buffer
	bindGroup       *wgpu.BindGroup
	capacity        uint32
	count           uint32
	scratch         []uiQuadInstance
	scratchZ        []int32
}

func visibleInClip(node *ui.Node) bool {
	if node.Clip.Width <= 0 || node.Clip.Height <= 0 {
		return true
	}
	r := node.Resolved
	c := node.Clip
	return r.X+r.Width > c.X && r.X < c.X+c.Width &&
		r.Y+r.Height > c.Y && r.Y < c.Y+c.Height
}

func sortInstancesByZ(instances []uiQuadInstance, z []int32) {
	if len(instances) != len(z) || len(instances) < 2 {
		return
	}
	indexed := make([]int, len(instances))
	for i := range indexed {
		indexed[i] = i
	}
	sortStableByZ(indexed, z)
	tmpInst := make([]uiQuadInstance, len(instances))
	tmpZ := make([]int32, len(z))
	for newPos, oldPos := range indexed {
		tmpInst[newPos] = instances[oldPos]
		tmpZ[newPos] = z[oldPos]
	}
	copy(instances, tmpInst)
	copy(z, tmpZ)
}

func sortStableByZ(indices []int, z []int32) {
	for i := 1; i < len(indices); i++ {
		j := i
		for j > 0 && z[indices[j-1]] > z[indices[j]] {
			indices[j-1], indices[j] = indices[j], indices[j-1]
			j--
		}
	}
}

func NewUiQuadPass(device *wgpu.Device, surfaceFormat wgpu.TextureFormat) (*render.Pass, error) {
	state := &uiQuadPassState{}

	bindGroupLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "ui_quad bind group layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: wgpu.ShaderStageVertex,
				Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform},
			},
			{
				Binding:    1,
				Visibility: wgpu.ShaderStageVertex,
				Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage},
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("ui_quad: bind group layout: %w", err)
	}
	state.bindGroupLayout = bindGroupLayout

	viewportBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "ui_quad viewport",
		Size:  16,
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("ui_quad: viewport buffer: %w", err)
	}
	state.viewportBuffer = viewportBuffer

	if err := ensureUiQuadCapacity(state, device, uiQuadMinCapacity); err != nil {
		return nil, err
	}

	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "ui_quad shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: uiQuadShader},
	})
	if err != nil {
		return nil, fmt.Errorf("ui_quad: shader: %w", err)
	}
	defer shader.Release()

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "ui_quad pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bindGroupLayout},
	})
	if err != nil {
		return nil, fmt.Errorf("ui_quad: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	pipeline, err := device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "ui_quad pipeline",
		Layout: pipelineLayout,
		Vertex: wgpu.VertexState{Module: shader, EntryPoint: "vertex_main"},
		Primitive: wgpu.PrimitiveState{
			Topology: wgpu.PrimitiveTopologyTriangleList,
			CullMode: wgpu.CullModeNone,
		},
		Multisample: wgpu.MultisampleState{
			Count: 1, Mask: 0xFFFFFFFF,
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
		return nil, fmt.Errorf("ui_quad: pipeline: %w", err)
	}
	state.pipeline = pipeline

	return &render.Pass{
		Name:    "ui_quad",
		Writes:  []string{"color"},
		Prepare: func(c *render.PassContext) error { return uiQuadPrepare(state, c) },
		Execute: func(c *render.PassContext) error { return uiQuadExecute(state, c) },
		Release: func() { uiQuadRelease(state) },
	}, nil
}

func uiQuadPrepare(state *uiQuadPassState, context *render.PassContext) error {
	state.count = 0

	if !ui.HasUI(context.World) {
		return nil
	}
	uiWorld := ecs.MustResource[ui.WorldRef](context.World).World

	state.scratch = state.scratch[:0]
	state.scratchZ = state.scratchZ[:0]
	nodeColorMask := ecs.MustMaskOf[ui.Node](uiWorld) | ecs.MustMaskOf[ui.Color](uiWorld)
	interactiveMask := ecs.MustMaskOf[ui.Interactive](uiWorld)
	uiWorld.ForEach(nodeColorMask, 0, func(entity ecs.Entity, table *ecs.Archetype, index int) {
		nodes, _ := ecs.Column[ui.Node](uiWorld, table)
		colors, _ := ecs.Column[ui.Color](uiWorld, table)
		node := &nodes[index]
		if !visibleInClip(node) {
			return
		}
		color := colors[index].RGBA
		if uiWorld.HasComponents(entity, interactiveMask) {
			interactive, _ := ecs.Get[ui.Interactive](uiWorld, entity)
			if interactive.Pressed {
				color = darken(color, 0.6)
			} else if interactive.Hovered {
				color = lighten(color, 1.15)
			}
		}
		state.scratch = append(state.scratch, uiQuadInstance{
			Rect:  [4]float32{node.Resolved.X, node.Resolved.Y, node.Resolved.Width, node.Resolved.Height},
			Color: color,
		})
		state.scratchZ = append(state.scratchZ, node.ZIndex)
	})

	sortInstancesByZ(state.scratch, state.scratchZ)
	state.count = uint32(len(state.scratch))
	if state.count == 0 {
		return nil
	}
	if err := ensureUiQuadCapacity(state, context.Device, state.count); err != nil {
		return err
	}

	viewport := uiQuadViewport{
		Width:  float32(ecs.MustResource[render.RendererResource](context.World).Renderer.Config.Width),
		Height: float32(ecs.MustResource[render.RendererResource](context.World).Renderer.Config.Height),
	}
	writeBuffer(context.Device, context.Queue, context.Encoder, state.viewportBuffer, 0, bytesOf(&viewport))

	instanceBytes := uint64(state.count) * uiQuadInstanceBytes
	writeBuffer(context.Device, context.Queue, context.Encoder, state.instanceBuffer, 0, instanceSlice(state.scratch)[:instanceBytes])

	return nil
}

func uiQuadExecute(state *uiQuadPassState, context *render.PassContext) error {
	if state.count == 0 {
		return nil
	}
	attachment, err := context.ColorAttachment("color")
	if err != nil {
		return err
	}
	pass := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		Label:            "ui_quad",
		ColorAttachments: []wgpu.RenderPassColorAttachment{attachment},
	})
	pass.SetPipeline(state.pipeline)
	pass.SetBindGroup(0, state.bindGroup, nil)
	pass.Draw(6, state.count, 0, 0)
	pass.End()
	pass.Release()
	return nil
}

func uiQuadRelease(state *uiQuadPassState) {
	if state.bindGroup != nil {
		state.bindGroup.Release()
	}
	if state.instanceBuffer != nil {
		state.instanceBuffer.Release()
	}
	if state.viewportBuffer != nil {
		state.viewportBuffer.Release()
	}
	if state.pipeline != nil {
		state.pipeline.Release()
	}
	if state.bindGroupLayout != nil {
		state.bindGroupLayout.Release()
	}
}

func ensureUiQuadCapacity(state *uiQuadPassState, device *wgpu.Device, required uint32) error {
	if state.capacity >= required && state.instanceBuffer != nil {
		return nil
	}
	newCapacity := state.capacity
	if newCapacity == 0 {
		newCapacity = uiQuadMinCapacity
	}
	for newCapacity < required {
		newCapacity *= 2
	}
	buffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "ui_quad instances",
		Size:  uint64(newCapacity) * uiQuadInstanceBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return fmt.Errorf("ui_quad: instance buffer: %w", err)
	}
	bindGroup, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "ui_quad bind group",
		Layout: state.bindGroupLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: state.viewportBuffer, Offset: 0, Size: 16},
			{Binding: 1, Buffer: buffer, Offset: 0, Size: wgpu.WholeSize},
		},
	})
	if err != nil {
		buffer.Release()
		return fmt.Errorf("ui_quad: bind group: %w", err)
	}
	if state.bindGroup != nil {
		state.bindGroup.Release()
	}
	if state.instanceBuffer != nil {
		state.instanceBuffer.Release()
	}
	state.instanceBuffer = buffer
	state.bindGroup = bindGroup
	state.capacity = newCapacity
	return nil
}

func instanceSlice(s []uiQuadInstance) []byte {
	if len(s) == 0 {
		return nil
	}
	return bytesOfN(&s[0], uint64(len(s))*uiQuadInstanceBytes)
}

func darken(c [4]float32, k float32) [4]float32 {
	return [4]float32{c[0] * k, c[1] * k, c[2] * k, c[3]}
}

func lighten(c [4]float32, k float32) [4]float32 {
	return [4]float32{
		clamp01(c[0] * k),
		clamp01(c[1] * k),
		clamp01(c[2] * k),
		c[3],
	}
}

func clamp01(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}
