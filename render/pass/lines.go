package pass

import (
	_ "embed"
	"fmt"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/go-gl/mathgl/mgl32"

	"indigo/ecs"
	"indigo/render"
	"indigo/render/asset"
	"indigo/transform"
)

//go:embed lines.wgsl
var linesShader string

// LineSegment is a single straight-line draw in world space.
// Position fields use vec4 with w == 1.0 to match WGSL std430 layout
// on the storage buffer side.
type LineSegment struct {
	Start [4]float32
	End   [4]float32
	Color [4]float32
}

// Lines is the resource apps push per-frame line segments into. The
// lines pass consumes the slice each frame and resets it for the
// next.
type Lines struct {
	Segments []LineSegment
}

type LinesResource struct{ Lines *Lines }

// AddSegment appends a world-space line from start to end with the
// given RGBA color.
func (l *Lines) AddSegment(start, end [3]float32, color [4]float32) {
	l.Segments = append(l.Segments, LineSegment{
		Start: [4]float32{start[0], start[1], start[2], 1},
		End:   [4]float32{end[0], end[1], end[2], 1},
		Color: color,
	})
}

// AddBox emits the 12 edges of an axis-aligned box transformed by a
// 4x4 model matrix. Used by the bounding-volume visualization but
// public so any system can draw an oriented box.
func (l *Lines) AddBox(bounds asset.BoundingVolume, model *transform.Mat4, color [4]float32) {
	corners := bounds.Corners()
	var worldCorners [8][3]float32
	for i, c := range corners {
		p := model.Mul4x1(mgl32.Vec4{c[0], c[1], c[2], 1})
		worldCorners[i] = [3]float32{p.X(), p.Y(), p.Z()}
	}
	for i := 0; i < len(asset.BoundingBoxEdges); i += 2 {
		a := worldCorners[asset.BoundingBoxEdges[i]]
		b := worldCorners[asset.BoundingBoxEdges[i+1]]
		l.AddSegment(a, b, color)
	}
}

const linesMinCapacity uint32 = 256
const lineSegmentBytes = uint64(48)

type linesPassState struct {
	pipeline        *wgpu.RenderPipeline
	bindGroupLayout *wgpu.BindGroupLayout
	viewProjBuffer  *wgpu.Buffer
	instanceBuffer  *wgpu.Buffer
	bindGroup       *wgpu.BindGroup
	capacity        uint32
	count           uint32
	aspectFn        func() float32
}

// NewLinesPass builds a screen-space-thin world-space line render
// pass. Lines come from the [Lines] resource on the engine world;
// the pass uploads them to a storage buffer each frame and draws
// them as a primitive-line-list.
func NewLinesPass(device *wgpu.Device, surfaceFormat wgpu.TextureFormat, aspect func() float32) (*render.Pass, error) {
	state := &linesPassState{aspectFn: aspect}

	layout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "lines bind group layout",
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
		return nil, fmt.Errorf("lines pass: bind group layout: %w", err)
	}
	state.bindGroupLayout = layout

	viewProj, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "lines view_proj buffer",
		Size:  uint64(unsafe.Sizeof(mgl32.Mat4{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("lines pass: view_proj buffer: %w", err)
	}
	state.viewProjBuffer = viewProj

	if err := ensureLinesCapacity(state, device, linesMinCapacity); err != nil {
		return nil, err
	}

	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "lines shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: linesShader},
	})
	if err != nil {
		return nil, fmt.Errorf("lines pass: shader: %w", err)
	}
	defer shader.Release()

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "lines pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{layout},
	})
	if err != nil {
		return nil, fmt.Errorf("lines pass: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	pipeline, err := device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "lines pipeline",
		Layout: pipelineLayout,
		Vertex: wgpu.VertexState{Module: shader, EntryPoint: "vertex_main"},
		Primitive: wgpu.PrimitiveState{
			Topology: wgpu.PrimitiveTopologyLineList,
			CullMode: wgpu.CullModeNone,
		},
		DepthStencil: &wgpu.DepthStencilState{
			Format:            render.DepthFormat,
			DepthWriteEnabled: false,
			DepthCompare:      wgpu.CompareFunctionLessEqual,
			StencilFront: wgpu.StencilFaceState{
				Compare:     wgpu.CompareFunctionAlways,
				FailOp:      wgpu.StencilOperationKeep,
				DepthFailOp: wgpu.StencilOperationKeep,
				PassOp:      wgpu.StencilOperationKeep,
			},
			StencilBack: wgpu.StencilFaceState{
				Compare:     wgpu.CompareFunctionAlways,
				FailOp:      wgpu.StencilOperationKeep,
				DepthFailOp: wgpu.StencilOperationKeep,
				PassOp:      wgpu.StencilOperationKeep,
			},
		},
		Multisample: wgpu.MultisampleState{Count: 1, Mask: 0xFFFFFFFF},
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
		return nil, fmt.Errorf("lines pass: pipeline: %w", err)
	}
	state.pipeline = pipeline

	return &render.Pass{
		Name:    "lines",
		Writes:  []string{"color", "depth"},
		State:   state,
		Prepare: linesPrepare,
		Execute: linesExecute,
		Release: linesRelease,
	}, nil
}

func linesPrepare(s any, context *render.PassContext) error {
	state := s.(*linesPassState)
	state.count = 0

	if !ecs.HasResource[LinesResource](context.World) {
		return nil
	}
	lines := ecs.MustResource[LinesResource](context.World).Lines
	if lines == nil || len(lines.Segments) == 0 {
		return nil
	}

	camera := ecs.MustResource[render.Camera](context.World)
	viewProj := render.CameraViewProjection(camera, state.aspectFn())
	writeBuffer(context.Device, context.Queue, context.Encoder, state.viewProjBuffer, 0, bytesOf(&viewProj))

	state.count = uint32(len(lines.Segments))
	if err := ensureLinesCapacity(state, context.Device, state.count); err != nil {
		return err
	}
	writeBuffer(context.Device, context.Queue, context.Encoder, state.instanceBuffer, 0, bytesOfN(&lines.Segments[0], uint64(state.count)*lineSegmentBytes))
	return nil
}

func linesExecute(s any, context *render.PassContext) error {
	state := s.(*linesPassState)
	if state.count == 0 {
		clearLines(context.World)
		return nil
	}
	color, err := context.ColorAttachment("color")
	if err != nil {
		return err
	}
	depth, err := context.DepthAttachment("depth")
	if err != nil {
		return err
	}
	pass := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		Label:                  "lines",
		ColorAttachments:       []wgpu.RenderPassColorAttachment{color},
		DepthStencilAttachment: &depth,
	})
	pass.SetPipeline(state.pipeline)
	pass.SetBindGroup(0, state.bindGroup, nil)
	pass.Draw(2, state.count, 0, 0)
	pass.End()
	pass.Release()
	clearLines(context.World)
	return nil
}

func clearLines(world *ecs.World) {
	if !ecs.HasResource[LinesResource](world) {
		return
	}
	lines := ecs.MustResource[LinesResource](world).Lines
	if lines == nil {
		return
	}
	lines.Segments = lines.Segments[:0]
}

func linesRelease(s any) {
	state := s.(*linesPassState)
	if state.bindGroup != nil {
		state.bindGroup.Release()
	}
	if state.instanceBuffer != nil {
		state.instanceBuffer.Release()
	}
	if state.viewProjBuffer != nil {
		state.viewProjBuffer.Release()
	}
	if state.pipeline != nil {
		state.pipeline.Release()
	}
	if state.bindGroupLayout != nil {
		state.bindGroupLayout.Release()
	}
}

func ensureLinesCapacity(state *linesPassState, device *wgpu.Device, required uint32) error {
	if state.capacity >= required && state.instanceBuffer != nil {
		return nil
	}
	newCapacity := state.capacity
	if newCapacity == 0 {
		newCapacity = linesMinCapacity
	}
	for newCapacity < required {
		newCapacity *= 2
	}
	buffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "lines instance buffer",
		Size:  uint64(newCapacity) * lineSegmentBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return fmt.Errorf("lines pass: instance buffer: %w", err)
	}
	bindGroup, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "lines bind group",
		Layout: state.bindGroupLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: state.viewProjBuffer, Offset: 0, Size: uint64(unsafe.Sizeof(mgl32.Mat4{}))},
			{Binding: 1, Buffer: buffer, Offset: 0, Size: wgpu.WholeSize},
		},
	})
	if err != nil {
		buffer.Release()
		return fmt.Errorf("lines pass: bind group: %w", err)
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

// AddLinesPass registers the line pass with renderer.Graph and binds
// it to write into the scene color attachment with the scene depth
// buffer, so world-space lines respect the mesh-pass depth.
func AddLinesPass(renderer *render.Renderer) (*render.Pass, error) {
	pass, err := NewLinesPass(renderer.Device, render.HdrFormat, renderer.AspectRatio)
	if err != nil {
		return nil, err
	}
	if err := renderer.Graph.AddPass(pass, []render.SlotBinding{
		{Slot: "color", ResourceID: renderer.SceneColorID},
		{Slot: "depth", ResourceID: renderer.DepthID},
	}); err != nil {
		return nil, err
	}
	return pass, nil
}
