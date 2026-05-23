package pass

import (
	_ "embed"
	"fmt"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/go-gl/mathgl/mgl32"

	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/render/asset"
	"github.com/matthewjberger/indigo/transform"
)

//go:embed pick_proxy.wgsl
var pickProxyShader string

type pickProxyUniform struct {
	ViewProj mgl32.Mat4
	Model    mgl32.Mat4
	EntityID uint32
	Pad0     uint32
	Pad1     uint32
	Pad2     uint32
}

type pickProxyPassState struct {
	pipeline        *wgpu.RenderPipeline
	bindGroupLayout *wgpu.BindGroupLayout
	uniformBuffer   *wgpu.Buffer
	bindGroup       *wgpu.BindGroup
	aspectFn        func() float32
}

func AddPickProxyPass(renderer *render.Renderer) (*render.Pass, error) {
	state := &pickProxyPassState{aspectFn: renderer.AspectRatio}

	module, err := renderer.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "pick_proxy shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: pickProxyShader},
	})
	if err != nil {
		return nil, fmt.Errorf("pick_proxy: shader: %w", err)
	}
	defer module.Release()

	layout, err := renderer.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "pick_proxy layout",
		Entries: []wgpu.BindGroupLayoutEntry{{
			Binding:    0,
			Visibility: wgpu.ShaderStageVertex | wgpu.ShaderStageFragment,
			Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform},
		}},
	})
	if err != nil {
		return nil, fmt.Errorf("pick_proxy: layout: %w", err)
	}
	state.bindGroupLayout = layout

	pipelineLayout, err := renderer.Device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "pick_proxy pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{layout},
	})
	if err != nil {
		layout.Release()
		return nil, fmt.Errorf("pick_proxy: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	pipeline, err := renderer.Device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "pick_proxy pipeline",
		Layout: pipelineLayout,
		Vertex: wgpu.VertexState{
			Module:     module,
			EntryPoint: "vertex_main",
			Buffers: []wgpu.VertexBufferLayout{{
				ArrayStride: uint64(unsafe.Sizeof(asset.MeshVertex{})),
				StepMode:    wgpu.VertexStepModeVertex,
				Attributes: []wgpu.VertexAttribute{
					{Format: wgpu.VertexFormatFloat32x4, Offset: 0, ShaderLocation: 0},
				},
			}},
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
			StencilFront:      wgpu.StencilFaceState{Compare: wgpu.CompareFunctionAlways},
			StencilBack:       wgpu.StencilFaceState{Compare: wgpu.CompareFunctionAlways},
		},
		Multisample: wgpu.MultisampleState{Count: 1, Mask: 0xFFFFFFFF},
		Fragment: &wgpu.FragmentState{
			Module:     module,
			EntryPoint: "fragment_main",
			Targets: []wgpu.ColorTargetState{
				{Format: render.EntityIdFormat, WriteMask: wgpu.ColorWriteMaskRed},
			},
		},
	})
	if err != nil {
		layout.Release()
		return nil, fmt.Errorf("pick_proxy: pipeline: %w", err)
	}
	state.pipeline = pipeline

	uniformBuffer, err := renderer.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "pick_proxy uniform",
		Size:  uint64(unsafe.Sizeof(pickProxyUniform{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		layout.Release()
		pipeline.Release()
		return nil, fmt.Errorf("pick_proxy: uniform buffer: %w", err)
	}
	state.uniformBuffer = uniformBuffer

	bg, err := renderer.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "pick_proxy bg",
		Layout: layout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: uniformBuffer, Offset: 0, Size: uint64(unsafe.Sizeof(pickProxyUniform{}))},
		},
	})
	if err != nil {
		layout.Release()
		pipeline.Release()
		uniformBuffer.Release()
		return nil, fmt.Errorf("pick_proxy: bg: %w", err)
	}
	state.bindGroup = bg

	pass := &render.Pass{
		Name:    "pick_proxy",
		Reads:   []string{"depth"},
		Writes:  []string{"entity_id"},
		Execute: func(c *render.PassContext) error { return pickProxyExecute(state, c) },
		Release: func() { pickProxyRelease(state) },
	}
	if err := renderer.Graph.AddPass(pass, []render.SlotBinding{
		{Slot: "depth", ResourceID: renderer.DepthID},
		{Slot: "entity_id", ResourceID: renderer.EntityIdID},
	}); err != nil {
		return nil, err
	}
	return pass, nil
}

func pickProxyExecute(state *pickProxyPassState, context *render.PassContext) error {
	mask := ecs.MustMaskOf[render.PickProxy](context.World) |
		ecs.MustMaskOf[asset.RenderMesh](context.World) |
		ecs.MustMaskOf[transform.GlobalTransform](context.World)

	var proxies []ecs.Entity
	context.World.ForEach(mask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		proxies = append(proxies, entity)
	})
	if len(proxies) == 0 {
		return nil
	}

	assets := ecs.MustResource[asset.MeshAssetsResource](context.World).Assets
	camera, hasCamera := ecs.Resource[render.Camera](context.World)
	if !hasCamera || camera == nil {
		return nil
	}
	viewProj := render.CameraViewProjection(camera, state.aspectFn())

	entityIdAttachment, err := context.ColorAttachment("entity_id")
	if err != nil {
		return err
	}
	entityIdAttachment.LoadOp = wgpu.LoadOpLoad
	depthAttachment, err := context.DepthAttachment("depth")
	if err != nil {
		return err
	}
	depthAttachment.DepthLoadOp = wgpu.LoadOpLoad
	depthAttachment.DepthStoreOp = wgpu.StoreOpStore
	depthAttachment.StencilLoadOp = wgpu.LoadOpLoad
	depthAttachment.StencilStoreOp = wgpu.StoreOpStore

	for _, entity := range proxies {
		mesh, ok := ecs.Get[asset.RenderMesh](context.World, entity)
		if !ok {
			continue
		}
		global, ok := ecs.Get[transform.GlobalTransform](context.World, entity)
		if !ok {
			continue
		}
		entry, ok := assets.Lookup(mesh.Mesh)
		if !ok {
			continue
		}
		uniform := pickProxyUniform{
			ViewProj: viewProj,
			Model:    global.Matrix,
			EntityID: entity.ID,
		}
		writeBuffer(context.Device, context.Queue, context.Encoder, state.uniformBuffer, 0, bytesOf(&uniform))

		passEnc := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
			Label:                  "pick_proxy",
			ColorAttachments:       []wgpu.RenderPassColorAttachment{entityIdAttachment},
			DepthStencilAttachment: &depthAttachment,
		})
		passEnc.SetPipeline(state.pipeline)
		passEnc.SetBindGroup(0, state.bindGroup, nil)
		passEnc.SetVertexBuffer(0, entry.Vertices, 0, wgpu.WholeSize)
		passEnc.Draw(entry.VertexCount, 1, 0, 0)
		passEnc.End()
		passEnc.Release()
	}
	return nil
}

func pickProxyRelease(state *pickProxyPassState) {
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
