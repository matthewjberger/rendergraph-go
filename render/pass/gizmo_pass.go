package pass

import (
	_ "embed"
	"fmt"
	"math"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/go-gl/mathgl/mgl32"

	"indigo/ecs"
	"indigo/render"
	"indigo/transform"
)

//go:embed gizmo.wgsl
var gizmoShader string

const gizmoLineInstanceBytes = uint64(48)
const gizmoMinCapacity uint32 = 64

type gizmoLineInstance struct {
	StartEnd [4]float32
	Color    [4]float32
	Extra    [4]float32
}

type gizmoPassState struct {
	pipeline        *wgpu.RenderPipeline
	bindGroupLayout *wgpu.BindGroupLayout
	viewportBuffer  *wgpu.Buffer
	instanceBuffer  *wgpu.Buffer
	bindGroup       *wgpu.BindGroup
	capacity        uint32
	count           uint32
	scratch         []gizmoLineInstance
	aspectFn        func() float32
}

// NewGizmoPass builds the 2D gizmo overlay pass: a screen-space
// line/triangle rasterizer driven entirely by per-frame CPU
// projection of the selected entity's world axes. No 3D geometry
// is involved; this is a UI-layer overlay.
func NewGizmoPass(device *wgpu.Device, surfaceFormat wgpu.TextureFormat, aspect func() float32) (*render.Pass, error) {
	state := &gizmoPassState{aspectFn: aspect}

	bindGroupLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "gizmo bind group layout",
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
		return nil, fmt.Errorf("gizmo pass: bind group layout: %w", err)
	}
	state.bindGroupLayout = bindGroupLayout

	viewportBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "gizmo viewport",
		Size:  16,
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("gizmo pass: viewport buffer: %w", err)
	}
	state.viewportBuffer = viewportBuffer

	if err := ensureGizmoCapacity(state, device, gizmoMinCapacity); err != nil {
		return nil, err
	}

	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "gizmo shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: gizmoShader},
	})
	if err != nil {
		return nil, fmt.Errorf("gizmo pass: shader: %w", err)
	}
	defer shader.Release()

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "gizmo pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bindGroupLayout},
	})
	if err != nil {
		return nil, fmt.Errorf("gizmo pass: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	pipeline, err := device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "gizmo pipeline",
		Layout: pipelineLayout,
		Vertex: wgpu.VertexState{Module: shader, EntryPoint: "vertex_main"},
		Primitive: wgpu.PrimitiveState{
			Topology: wgpu.PrimitiveTopologyTriangleList,
			CullMode: wgpu.CullModeNone,
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
		return nil, fmt.Errorf("gizmo pass: pipeline: %w", err)
	}
	state.pipeline = pipeline

	return &render.Pass{
		Name:    "gizmo",
		Writes:  []string{"color"},
		State:   state,
		Prepare: gizmoPrepare,
		Execute: gizmoExecute,
		Release: gizmoRelease,
	}, nil
}

func gizmoPrepare(s any, context *render.PassContext) error {
	state := s.(*gizmoPassState)
	state.count = 0
	state.scratch = state.scratch[:0]

	gizmoPtr, ok := lookupGizmosPtr(context.World)
	if !ok {
		return nil
	}
	gizmo := *gizmoPtr
	if gizmo == nil || gizmo.Mode == render.GizmoNone {
		return nil
	}

	target, ok := SelectedTarget(context.World)
	if !ok {
		return nil
	}
	global, ok := ecs.Get[transform.GlobalTransform](context.World, target)
	if !ok {
		return nil
	}

	camera := ecs.MustResource[render.Camera](context.World)
	renderer := ecs.MustResource[render.RendererResource](context.World).Renderer
	aspect := state.aspectFn()
	viewport := [2]float32{
		float32(renderer.Config.Width),
		float32(renderer.Config.Height),
	}
	viewProj := render.CameraViewProjection(camera, aspect)
	tanHalfFov := float32(math.Tan(float64(camera.FovYRadians) * 0.5))

	origin := mgl32.Vec3{
		global.Matrix[12], global.Matrix[13], global.Matrix[14],
	}
	worldLength := AxisWorldLengthForScreen(origin, viewProj, viewport, tanHalfFov, gizmoAxisScreenLength)
	if worldLength <= 0 {
		return nil
	}
	originScreen, ends, valid := GizmoScreenPositions(origin, viewProj, viewport, worldLength)

	if gizmo.Mode == render.GizmoRotate {
		for axis := 0; axis < 3; axis++ {
			color := AxisColor(uint8(axis))
			thickness := gizmoAxisThicknessPx
			if int(axis) == gizmo.HoverAxis || (gizmo.Dragging && int(gizmo.DragAxis) == axis) {
				color = AxisHoverColor
				thickness = gizmoAxisHoverThicknessPx
			}
			points, validPts := RingScreenPoints(origin, AxisDirection(uint8(axis)), worldLength, viewProj, viewport)
			for sample := 0; sample < RingSegmentCount; sample++ {
				if !validPts[sample] || !validPts[sample+1] {
					continue
				}
				state.scratch = append(state.scratch, gizmoLineInstance{
					StartEnd: [4]float32{points[sample].X(), points[sample].Y(), points[sample+1].X(), points[sample+1].Y()},
					Color:    color,
					Extra:    [4]float32{thickness, 0, 0, 0},
				})
			}
		}
	} else {
		for axis := 0; axis < 3; axis++ {
			if !valid[axis] {
				continue
			}
			color := AxisColor(uint8(axis))
			thickness := gizmoAxisThicknessPx
			if int(axis) == gizmo.HoverAxis || (gizmo.Dragging && int(gizmo.DragAxis) == axis) {
				color = AxisHoverColor
				thickness = gizmoAxisHoverThicknessPx
			}
			state.scratch = append(state.scratch, gizmoLineInstance{
				StartEnd: [4]float32{originScreen.X(), originScreen.Y(), ends[axis].X(), ends[axis].Y()},
				Color:    color,
				Extra:    [4]float32{thickness, 0, 0, 0},
			})

			tipColor := color
			tipThickness := thickness + 2
			switch gizmo.Mode {
			case render.GizmoScale:
				state.scratch = append(state.scratch, makeTipSquare(ends[axis], tipColor, tipThickness*2))
			case render.GizmoTranslate:
				state.scratch = append(state.scratch, makeTipArrow(originScreen, ends[axis], tipColor))
				_ = tipThickness
			}
		}
	}

	state.scratch = append(state.scratch, makeOriginMarker(originScreen, [4]float32{0.92, 0.94, 0.98, 1}, gizmoOriginRadiusPx*2)...)

	state.count = uint32(len(state.scratch))
	if state.count == 0 {
		return nil
	}
	if err := ensureGizmoCapacity(state, context.Device, state.count); err != nil {
		return err
	}

	view := struct {
		W float32
		H float32
		_ float32
		_ float32
	}{viewport[0], viewport[1], 0, 0}
	writeBuffer(context.Device, context.Queue, context.Encoder, state.viewportBuffer, 0, bytesOf(&view))

	bytes := uint64(state.count) * gizmoLineInstanceBytes
	writeBuffer(context.Device, context.Queue, context.Encoder, state.instanceBuffer, 0, bytesOfN(&state.scratch[0], bytes))
	return nil
}

func gizmoExecute(s any, context *render.PassContext) error {
	state := s.(*gizmoPassState)
	if state.count == 0 {
		return nil
	}
	attachment, err := context.ColorAttachment("color")
	if err != nil {
		return err
	}
	pass := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		Label:            "gizmo",
		ColorAttachments: []wgpu.RenderPassColorAttachment{attachment},
	})
	pass.SetPipeline(state.pipeline)
	pass.SetBindGroup(0, state.bindGroup, nil)
	pass.Draw(6, state.count, 0, 0)
	pass.End()
	pass.Release()
	return nil
}

func gizmoRelease(s any) {
	state := s.(*gizmoPassState)
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

func ensureGizmoCapacity(state *gizmoPassState, device *wgpu.Device, required uint32) error {
	if state.capacity >= required && state.instanceBuffer != nil {
		return nil
	}
	newCapacity := state.capacity
	if newCapacity == 0 {
		newCapacity = gizmoMinCapacity
	}
	for newCapacity < required {
		newCapacity *= 2
	}
	buffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "gizmo instances",
		Size:  uint64(newCapacity) * gizmoLineInstanceBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return fmt.Errorf("gizmo pass: instance buffer: %w", err)
	}
	bindGroup, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "gizmo bind group",
		Layout: state.bindGroupLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: state.viewportBuffer, Offset: 0, Size: 16},
			{Binding: 1, Buffer: buffer, Offset: 0, Size: wgpu.WholeSize},
		},
	})
	if err != nil {
		buffer.Release()
		return fmt.Errorf("gizmo pass: bind group: %w", err)
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

func makeTipSquare(at mgl32.Vec2, color [4]float32, side float32) gizmoLineInstance {
	half := side * 0.5
	return gizmoLineInstance{
		StartEnd: [4]float32{at.X() - half, at.Y(), at.X() + half, at.Y()},
		Color:    color,
		Extra:    [4]float32{side, 0, 0, 0},
	}
}

// makeTipArrow returns one instance configured with kind=1 (filled
// triangle): the WGSL vertex shader collapses three of the six
// vertices onto the tip, leaving a single triangle whose base sits
// at the axis line's endpoint and whose apex sits headLength
// further along the axis direction. The arrowhead therefore
// EXTENDS BEYOND the line, instead of overlapping its last several
// pixels.
func makeTipArrow(origin, end mgl32.Vec2, color [4]float32) gizmoLineInstance {
	const headLength float32 = 18
	const headBaseWidth float32 = 14

	dir := end.Sub(origin)
	dirLen := float32(math.Hypot(float64(dir.X()), float64(dir.Y())))
	if dirLen < 1e-3 {
		return gizmoLineInstance{}
	}
	unit := dir.Mul(1 / dirLen)
	tip := end.Add(unit.Mul(headLength))
	return gizmoLineInstance{
		StartEnd: [4]float32{end.X(), end.Y(), tip.X(), tip.Y()},
		Color:    color,
		Extra:    [4]float32{headBaseWidth, 1, 0, 0},
	}
}

func makeOriginMarker(center mgl32.Vec2, color [4]float32, size float32) []gizmoLineInstance {
	half := size * 0.5
	return []gizmoLineInstance{{
		StartEnd: [4]float32{center.X() - half, center.Y(), center.X() + half, center.Y()},
		Color:    color,
		Extra:    [4]float32{size, 0, 0, 0},
	}}
}

func lookupGizmosPtr(world *ecs.World) (**render.Gizmos, bool) {
	if !ecs.HasResource[*render.Gizmos](world) {
		return nil, false
	}
	return ecs.MustResource[*render.Gizmos](world), true
}
