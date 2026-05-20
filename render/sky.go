package render

import (
	_ "embed"
	"fmt"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"

	"indigo/ecs"
)

//go:embed sky.wgsl
var skyShader string

// skyUniform mirrors the WGSL Uniform struct in sky.wgsl. 224 bytes
// total: three mat4x4 (proj, proj_inv, view), one vec4 (cam_pos), one
// f32 (time), three f32 padding. The padding keeps the struct's tail
// at the 16-byte alignment WGSL requires.
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

// NewSkyPass builds the engine's procedural sky pass. Draws a
// fullscreen triangle that fills scene_color with a sky+ground
// gradient plus a small sun disk, reconstructing per-pixel world
// directions from the camera's inverse projection. Ported from
// nightshade's SkyPass + sky.wgsl, scaled down to the single procedural
// atmosphere mode (no HDR / nebula / day-night).
//
// The pass writes only "color" (no depth), so it should appear FIRST
// in the graph: as the first writer of scene_color it gets the
// clear-on-load op, and subsequent passes (mesh, grid) overwrite the
// pixels they cover.
func NewSkyPass(device *wgpu.Device, surfaceFormat wgpu.TextureFormat, aspect func() float32) (*Pass, error) {
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

	return &Pass{
		Name:    "sky",
		Writes:  []string{"color"},
		State:   state,
		Prepare: skyPrepare,
		Execute: skyExecute,
		Release: skyRelease,
	}, nil
}

func skyPrepare(s any, context *PassContext) error {
	state := s.(*skyPassState)
	camera := ecs.Resource[Camera](context.World)
	aspect := state.aspectFn()
	proj := CameraProjection(camera, aspect)
	view := CameraView(camera)
	projInv := proj.Inv()

	uniform := skyUniform{
		Proj:    proj,
		ProjInv: projInv,
		View:    view,
		CamPos:  [4]float32{camera.Eye[0], camera.Eye[1], camera.Eye[2], 1.0},
		Time:    0.0,
	}
	writeBuffer(context.Device, context.Queue, context.Encoder, state.uniformBuffer, 0, bytesOf(&uniform))
	return nil
}

func skyExecute(s any, context *PassContext) error {
	state := s.(*skyPassState)
	settings := ecs.Resource[GraphicsSettings](context.World)

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

func skyRelease(s any) {
	state := s.(*skyPassState)
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
