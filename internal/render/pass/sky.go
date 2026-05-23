package pass

import (
	_ "embed"
	"fmt"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"

	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/window"
)

//go:embed sky.wgsl
var skyShader string

type skyUniform struct {
	Proj    [16]float32
	ProjInv [16]float32
	View    [16]float32
	CamPos  [4]float32
	Time    float32
	Pad0    float32
	Pad1    float32
	Pad2    float32
}

type skyPassState struct {
	pipeline        *wgpu.RenderPipeline
	bindGroupLayout *wgpu.BindGroupLayout
	uniformBuffer   *wgpu.Buffer
	bindGroup       *wgpu.BindGroup
	aspectFn        func() float32
}

func NewSkyPass(device *wgpu.Device, surfaceFormat wgpu.TextureFormat, aspect func() float32) (*render.Pass, error) {
	state := &skyPassState{aspectFn: aspect}

	bindGroupLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "sky bind group layout",
		Entries: []wgpu.BindGroupLayoutEntry{{
			Binding:    0,
			Visibility: wgpu.ShaderStageVertex | wgpu.ShaderStageFragment,
			Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform},
		}},
	})
	if err != nil {
		return nil, fmt.Errorf("sky pass: bind group layout: %w", err)
	}
	state.bindGroupLayout = bindGroupLayout

	uniformBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "sky uniform buffer",
		Size:  uint64(unsafe.Sizeof(skyUniform{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("sky pass: uniform buffer: %w", err)
	}
	state.uniformBuffer = uniformBuffer

	bindGroup, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "sky bind group",
		Layout: bindGroupLayout,
		Entries: []wgpu.BindGroupEntry{{
			Binding: 0,
			Buffer:  uniformBuffer,
			Offset:  0,
			Size:    uint64(unsafe.Sizeof(skyUniform{})),
		}},
	})
	if err != nil {
		return nil, fmt.Errorf("sky pass: bind group: %w", err)
	}
	state.bindGroup = bindGroup

	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "sky shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: skyShader},
	})
	if err != nil {
		return nil, fmt.Errorf("sky pass: shader: %w", err)
	}
	defer shader.Release()

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "sky pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bindGroupLayout},
	})
	if err != nil {
		return nil, fmt.Errorf("sky pass: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	pipeline, err := device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "sky pipeline",
		Layout: pipelineLayout,
		Vertex: wgpu.VertexState{
			Module:     shader,
			EntryPoint: "vs_sky",
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
			EntryPoint: "fs_sky",
			Targets: []wgpu.ColorTargetState{{
				Format:    surfaceFormat,
				WriteMask: wgpu.ColorWriteMaskAll,
			}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("sky pass: pipeline: %w", err)
	}
	state.pipeline = pipeline

	return &render.Pass{
		Name:    "sky",
		Writes:  []string{"color"},
		Prepare: func(c *render.PassContext) error { return skyPrepare(state, c) },
		Execute: func(c *render.PassContext) error { return skyExecute(state, c) },
		Release: func() { skyRelease(state) },
	}, nil
}

func skyPrepare(state *skyPassState, context *render.PassContext) error {
	camera := ecs.MustResource[render.Camera](context.World)
	aspect := state.aspectFn()
	proj := render.CameraProjection(camera, aspect)
	view := render.CameraView(camera)
	projInv := proj.Inv()

	uptime := ecs.MustResource[window.Window](context.World).Timing.UptimeSeconds
	uniform := skyUniform{
		Proj:    proj,
		ProjInv: projInv,
		View:    view,
		CamPos:  [4]float32{camera.Eye[0], camera.Eye[1], camera.Eye[2], 1.0},
		Time:    uptime,
	}
	writeBuffer(context.Device, context.Queue, context.Encoder, state.uniformBuffer, 0, bytesOf(&uniform))
	return nil
}

func skyExecute(state *skyPassState, context *render.PassContext) error {
	settings := ecs.MustResource[render.Graphics](context.World)

	colorAttachment, err := context.ColorAttachment("color")
	if err != nil {
		return err
	}

	pass := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		Label:            "sky",
		ColorAttachments: []wgpu.RenderPassColorAttachment{colorAttachment},
	})
	if settings.ShowSky {
		pass.SetPipeline(state.pipeline)
		pass.SetBindGroup(0, state.bindGroup, nil)
		pass.Draw(3, 1, 0, 0)
	}
	pass.End()
	pass.Release()
	return nil
}

func skyRelease(state *skyPassState) {
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
