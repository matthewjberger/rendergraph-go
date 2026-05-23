package pass

import (
	_ "embed"
	"fmt"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"

	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
)

//go:embed grid.wgsl
var gridShader string

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

func NewGridPass(device *wgpu.Device, surfaceFormat wgpu.TextureFormat, aspect func() float32) (*render.Pass, error) {
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

	return &render.Pass{
		Name:    "grid",
		Writes:  []string{"color", "depth"},
		Prepare: func(c *render.PassContext) error { return gridPrepare(state, c) },
		Execute: func(c *render.PassContext) error { return gridExecute(state, c) },
		Release: func() { gridRelease(state) },
	}, nil
}

func gridPrepare(state *gridPassState, context *render.PassContext) error {
	camera := ecs.MustResource[render.Camera](context.World)
	aspect := state.aspectFn()

	viewProj := render.CameraViewProjection(camera, aspect)

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

func gridExecute(state *gridPassState, context *render.PassContext) error {
	settings := ecs.MustResource[render.Graphics](context.World)
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

func gridRelease(state *gridPassState) {
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
