package render

import (
	_ "embed"
	"fmt"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"

	"indigo/ecs"
)

//go:embed grid.wgsl
var gridShader string

// gridUniform mirrors the WGSL Uniform struct in grid.wgsl. Field
// order and padding are chosen so the Go struct lays out at the
// offsets WGSL expects for a uniform-address-space struct (vec3<f32>
// is align 16, size 12, leaving the next f32 to pack at offset 76).
// Total size is 96 bytes; the trailing _pad fields are zero-filled
// reserved space that matches the shader's `_pad0 / _pad1`.
type gridUniform struct {
	ViewProj       [16]float32
	CameraWorldPos [3]float32
	GridSize       float32
	GridMinPixels  float32
	GridCellSize   float32
	Pad0           float32
	Pad1           float32
}

type gridPassState struct {
	pipeline        *wgpu.RenderPipeline
	bindGroupLayout *wgpu.BindGroupLayout
	uniformBuffer   *wgpu.Buffer
	bindGroup       *wgpu.BindGroup
	aspectFn        func() float32
}

// NewGridPass builds the engine's ground-plane grid pass: a fullscreen
// procedurally-shaded quad rendered at y=-0.01 that follows the
// camera, fades with distance, and highlights world X and Z axes. The
// pass reads scene_color + depth, depth-tests against existing
// geometry (no depth write), and alpha-blends on top.
//
// Ported from nightshade's GridPass + grid.wgsl. nightshade uses
// reverse-Z, so its depth state is GreaterEqual / clear 0.0; we use
// standard-Z (Less / clear 1.0), so the grid uses LessEqual to draw
// where the depth buffer is at or beyond the grid's depth.
func NewGridPass(device *wgpu.Device, surfaceFormat wgpu.TextureFormat, aspect func() float32) (*Pass, error) {
	state := &gridPassState{aspectFn: aspect}

	bindGroupLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "grid bind group layout",
		Entries: []wgpu.BindGroupLayoutEntry{{
			Binding:    0,
			Visibility: wgpu.ShaderStageVertex | wgpu.ShaderStageFragment,
			Buffer: wgpu.BufferBindingLayout{
				Type: wgpu.BufferBindingTypeUniform,
			},
		}},
	})
	if err != nil {
		return nil, fmt.Errorf("grid pass: bind group layout: %w", err)
	}
	state.bindGroupLayout = bindGroupLayout

	uniformBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "grid uniform buffer",
		Size:  uint64(unsafe.Sizeof(gridUniform{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("grid pass: uniform buffer: %w", err)
	}
	state.uniformBuffer = uniformBuffer

	bindGroup, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "grid bind group",
		Layout: bindGroupLayout,
		Entries: []wgpu.BindGroupEntry{{
			Binding: 0,
			Buffer:  uniformBuffer,
			Offset:  0,
			Size:    uint64(unsafe.Sizeof(gridUniform{})),
		}},
	})
	if err != nil {
		return nil, fmt.Errorf("grid pass: bind group: %w", err)
	}
	state.bindGroup = bindGroup

	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "grid shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: gridShader},
	})
	if err != nil {
		return nil, fmt.Errorf("grid pass: shader: %w", err)
	}
	defer shader.Release()

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "grid pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bindGroupLayout},
	})
	if err != nil {
		return nil, fmt.Errorf("grid pass: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	pipeline, err := device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "grid pipeline",
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
		DepthStencil: &wgpu.DepthStencilState{
			Format:            DepthFormat,
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
				Blend:     &wgpu.BlendStateAlphaBlending,
				WriteMask: wgpu.ColorWriteMaskAll,
			}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("grid pass: pipeline: %w", err)
	}
	state.pipeline = pipeline

	return &Pass{
		Name:    "grid",
		Writes:  []string{"color", "depth"},
		State:   state,
		Prepare: gridPrepare,
		Execute: gridExecute,
		Release: gridRelease,
	}, nil
}

func gridPrepare(s any, context *PassContext) error {
	state := s.(*gridPassState)
	camera := ecs.Resource[Camera](context.World)
	aspect := state.aspectFn()

	viewProj := CameraViewProjection(camera, aspect)

	uniform := gridUniform{
		ViewProj: viewProj,
		CameraWorldPos: [3]float32{
			camera.Eye[0], camera.Eye[1], camera.Eye[2],
		},
		GridSize:      500.0,
		GridMinPixels: 2.0,
		GridCellSize:  0.025,
	}
	writeBuffer(context.Device, context.Queue, context.Encoder, state.uniformBuffer, 0, bytesOf(&uniform))
	return nil
}

func gridExecute(s any, context *PassContext) error {
	state := s.(*gridPassState)
	settings := ecs.Resource[GraphicsSettings](context.World)
	if !settings.ShowGrid {
		return nil
	}

	colorAttachment, err := context.ColorAttachment("color")
	if err != nil {
		return err
	}
	depthAttachment, err := context.DepthAttachment("depth")
	if err != nil {
		return err
	}

	pass := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		Label:                  "grid",
		ColorAttachments:       []wgpu.RenderPassColorAttachment{colorAttachment},
		DepthStencilAttachment: &depthAttachment,
	})
	pass.SetPipeline(state.pipeline)
	pass.SetBindGroup(0, state.bindGroup, nil)
	pass.Draw(6, 1, 0, 0)
	pass.End()
	pass.Release()
	return nil
}

func gridRelease(s any) {
	state := s.(*gridPassState)
	if state.bindGroup != nil {
		state.bindGroup.Release()
	}
	if state.uniformBuffer != nil {
		state.uniformBuffer.Release()
	}
	if state.pipeline != nil {
		state.pipeline.Release()
	}
	if state.bindGroupLayout != nil {
		state.bindGroupLayout.Release()
	}
}
