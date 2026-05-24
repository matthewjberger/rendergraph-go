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

//go:embed instanced_transform_compute.wgsl
var instancedTransformShader string

//go:embed instanced_mesh.wgsl
var instancedMeshShader string

type instancedComputeUniform struct {
	ParentTransform mgl32.Mat4
	InstanceCount   uint32
	OutputOffset    uint32
	Pad0            uint32
	Pad1            uint32
}

type instancedPerEntity struct {
	BaseColor [4]float32
	EntityID  uint32
	Pad0      uint32
	Pad1      uint32
	Pad2      uint32
}

type instancedRenderEntity struct {
	perEntityBuffer *wgpu.Buffer
	bindGroup       *wgpu.BindGroup
	generation      uint64
	worldBuffer     *wgpu.Buffer
}

type instancedMeshState struct {
	pipeline        *wgpu.RenderPipeline
	prepassPipeline *wgpu.RenderPipeline
	viewProjLayout  *wgpu.BindGroupLayout
	globalLayout    *wgpu.BindGroupLayout
	handleLayout    *wgpu.BindGroupLayout
	viewProjBuffer  *wgpu.Buffer
	viewProjGroup   *wgpu.BindGroup
	uniformBuffer   *wgpu.Buffer
	globalGroup     *wgpu.BindGroup
	aspectFn        func() float32

	perEntity map[ecs.Entity]*instancedRenderEntity
}

func AddInstancedMeshPass(renderer *render.Renderer) (*render.Pass, error) {
	state, err := newInstancedMeshState(renderer.Device, renderer.AspectRatio)
	if err != nil {
		return nil, err
	}
	pass := &render.Pass{
		Name:    "instanced_mesh",
		Writes:  []string{"color", "depth", "entity_id", "view_normals"},
		Prepare: func(c *render.PassContext) error { return instancedMeshPrepare(state, c) },
		Execute: func(c *render.PassContext) error { return instancedMeshExecute(state, c) },
		Release: func() { instancedMeshRelease(state) },
	}
	if err := renderer.Graph.AddPass(pass, []render.SlotBinding{
		{Slot: "color", ResourceID: renderer.SceneColorID},
		{Slot: "depth", ResourceID: renderer.DepthID},
		{Slot: "entity_id", ResourceID: renderer.EntityIdID},
		{Slot: "view_normals", ResourceID: renderer.ViewNormalsID},
	}); err != nil {
		return nil, err
	}
	return pass, nil
}

func newInstancedMeshState(device *wgpu.Device, aspect func() float32) (*instancedMeshState, error) {
	module, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "instanced_mesh shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: instancedMeshShader},
	})
	if err != nil {
		return nil, fmt.Errorf("instanced_mesh: shader: %w", err)
	}
	defer module.Release()

	viewProjLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "instanced_mesh view_proj layout",
		Entries: []wgpu.BindGroupLayoutEntry{{
			Binding: 0, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform},
		}},
	})
	if err != nil {
		return nil, fmt.Errorf("instanced_mesh: view_proj layout: %w", err)
	}
	globalLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "instanced_mesh global layout",
		Entries: []wgpu.BindGroupLayoutEntry{{
			Binding: 0, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform},
		}},
	})
	if err != nil {
		viewProjLayout.Release()
		return nil, fmt.Errorf("instanced_mesh: global layout: %w", err)
	}
	handleLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "instanced_mesh handle layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageVertex | wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform}},
		},
	})
	if err != nil {
		viewProjLayout.Release()
		globalLayout.Release()
		return nil, fmt.Errorf("instanced_mesh: handle layout: %w", err)
	}

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "instanced_mesh pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{viewProjLayout, globalLayout, handleLayout},
	})
	if err != nil {
		viewProjLayout.Release()
		globalLayout.Release()
		handleLayout.Release()
		return nil, fmt.Errorf("instanced_mesh: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	pipeline, err := device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "instanced_mesh pipeline",
		Layout: pipelineLayout,
		Vertex: wgpu.VertexState{
			Module:     module,
			EntryPoint: "vertex_main",
			Buffers: []wgpu.VertexBufferLayout{{
				ArrayStride: uint64(unsafe.Sizeof(asset.MeshVertex{})),
				StepMode:    wgpu.VertexStepModeVertex,
				Attributes: []wgpu.VertexAttribute{
					{Format: wgpu.VertexFormatFloat32x4, Offset: 0, ShaderLocation: 0},
					{Format: wgpu.VertexFormatFloat32x4, Offset: 16, ShaderLocation: 1},
					{Format: wgpu.VertexFormatFloat32x4, Offset: 32, ShaderLocation: 2},
					{Format: wgpu.VertexFormatFloat32x4, Offset: 48, ShaderLocation: 3},
					{Format: wgpu.VertexFormatFloat32x4, Offset: 64, ShaderLocation: 4},
				},
			}},
		},
		Primitive: wgpu.PrimitiveState{Topology: wgpu.PrimitiveTopologyTriangleList, FrontFace: wgpu.FrontFaceCCW, CullMode: wgpu.CullModeBack},
		DepthStencil: &wgpu.DepthStencilState{
			Format:            render.DepthFormat,
			DepthWriteEnabled: false,
			DepthCompare:      wgpu.CompareFunctionGreaterEqual,
			StencilFront:      wgpu.StencilFaceState{Compare: wgpu.CompareFunctionAlways},
			StencilBack:       wgpu.StencilFaceState{Compare: wgpu.CompareFunctionAlways},
		},
		Multisample: wgpu.MultisampleState{Count: 1, Mask: 0xFFFFFFFF},
		Fragment: &wgpu.FragmentState{
			Module:     module,
			EntryPoint: "fragment_main",
			Targets: []wgpu.ColorTargetState{
				{Format: render.HdrFormat, WriteMask: wgpu.ColorWriteMaskAll},
				{Format: render.EntityIdFormat, WriteMask: wgpu.ColorWriteMaskAll},
				{Format: render.HdrFormat, WriteMask: wgpu.ColorWriteMaskAll},
			},
		},
	})
	if err != nil {
		viewProjLayout.Release()
		globalLayout.Release()
		handleLayout.Release()
		return nil, fmt.Errorf("instanced_mesh: pipeline: %w", err)
	}

	prepassPipeline, err := device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "instanced_mesh depth prepass pipeline",
		Layout: pipelineLayout,
		Vertex: wgpu.VertexState{
			Module:     module,
			EntryPoint: "vs_prepass",
			Buffers: []wgpu.VertexBufferLayout{{
				ArrayStride: uint64(unsafe.Sizeof(asset.MeshVertex{})),
				StepMode:    wgpu.VertexStepModeVertex,
				Attributes: []wgpu.VertexAttribute{
					{Format: wgpu.VertexFormatFloat32x4, Offset: 0, ShaderLocation: 0},
				},
			}},
		},
		Primitive: wgpu.PrimitiveState{Topology: wgpu.PrimitiveTopologyTriangleList, FrontFace: wgpu.FrontFaceCCW, CullMode: wgpu.CullModeBack},
		DepthStencil: &wgpu.DepthStencilState{
			Format:            render.DepthFormat,
			DepthWriteEnabled: true,
			DepthCompare:      wgpu.CompareFunctionGreater,
			StencilFront:      wgpu.StencilFaceState{Compare: wgpu.CompareFunctionAlways},
			StencilBack:       wgpu.StencilFaceState{Compare: wgpu.CompareFunctionAlways},
		},
		Multisample: wgpu.MultisampleState{Count: 1, Mask: 0xFFFFFFFF},
	})
	if err != nil {
		pipeline.Release()
		viewProjLayout.Release()
		globalLayout.Release()
		handleLayout.Release()
		return nil, fmt.Errorf("instanced_mesh: depth prepass pipeline: %w", err)
	}

	viewProjBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "instanced_mesh view_proj",
		Size:  uint64(unsafe.Sizeof(mgl32.Mat4{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("instanced_mesh: view_proj buffer: %w", err)
	}
	viewProjGroup, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:   "instanced_mesh view_proj bg",
		Layout:  viewProjLayout,
		Entries: []wgpu.BindGroupEntry{{Binding: 0, Buffer: viewProjBuffer, Offset: 0, Size: uint64(unsafe.Sizeof(mgl32.Mat4{}))}},
	})
	if err != nil {
		return nil, fmt.Errorf("instanced_mesh: view_proj bg: %w", err)
	}
	uniformBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "instanced_mesh light uniform",
		Size:  uint64(unsafe.Sizeof(skinnedUniforms{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("instanced_mesh: light buffer: %w", err)
	}

	return &instancedMeshState{
		pipeline:        pipeline,
		prepassPipeline: prepassPipeline,
		viewProjLayout:  viewProjLayout,
		globalLayout:    globalLayout,
		handleLayout:    handleLayout,
		viewProjBuffer:  viewProjBuffer,
		viewProjGroup:   viewProjGroup,
		uniformBuffer:   uniformBuffer,
		aspectFn:        aspect,
		perEntity:       make(map[ecs.Entity]*instancedRenderEntity),
	}, nil
}

func instancedMeshPrepare(state *instancedMeshState, context *render.PassContext) error {

	camera, hasCamera := ecs.Resource[render.Camera](context.World)
	if hasCamera && camera != nil {
		viewProj := render.CameraViewProjection(camera, state.aspectFn())
		writeBuffer(context.Device, context.Queue, context.Encoder, state.viewProjBuffer, 0, bytesOf(&viewProj))
	}

	light := defaultSkinnedUniforms()
	applyDirectionalToSkinned(context.World, &light)
	writeBuffer(context.Device, context.Queue, context.Encoder, state.uniformBuffer, 0, bytesOf(&light))

	if state.globalGroup == nil {
		bg, err := context.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:   "instanced_mesh global bg",
			Layout:  state.globalLayout,
			Entries: []wgpu.BindGroupEntry{{Binding: 0, Buffer: state.uniformBuffer, Offset: 0, Size: uint64(unsafe.Sizeof(skinnedUniforms{}))}},
		})
		if err != nil {
			return fmt.Errorf("instanced_mesh: global bg: %w", err)
		}
		state.globalGroup = bg
	}

	computeRes, ok := ecs.Resource[InstancedComputeResource](context.World)
	if !ok || computeRes == nil || computeRes.Compute == nil {
		return nil
	}
	compute := computeRes.Compute

	mask := ecs.MustMaskOf[asset.InstancedMesh](context.World) |
		ecs.MustMaskOf[transform.GlobalTransform](context.World)
	context.World.ForEach(mask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		worldBuffer := compute.WorldBuffer(entity)
		if worldBuffer == nil || compute.InstanceCount(entity) == 0 {
			return
		}
		es := state.perEntity[entity]
		if es == nil {
			es = &instancedRenderEntity{}
			state.perEntity[entity] = es
		}
		if es.perEntityBuffer == nil {
			buf, err := context.Device.CreateBuffer(&wgpu.BufferDescriptor{
				Label: "instanced per-entity",
				Size:  uint64(unsafe.Sizeof(instancedPerEntity{})),
				Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
			})
			if err != nil {
				delete(state.perEntity, entity)
				return
			}
			es.perEntityBuffer = buf
		}
		per := instancedPerEntity{BaseColor: [4]float32{1, 1, 1, 1}, EntityID: entity.ID}
		if material, ok := ecs.Get[asset.Material](context.World, entity); ok && material != nil {
			per.BaseColor = material.BaseColor
		}
		writeBuffer(context.Device, context.Queue, context.Encoder, es.perEntityBuffer, 0, bytesOf(&per))

		generation := compute.Generation(entity)
		if es.bindGroup != nil && (es.generation != generation || es.worldBuffer != worldBuffer) {
			es.bindGroup.Release()
			es.bindGroup = nil
		}
		if es.bindGroup == nil {
			bg, err := context.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
				Label:  "instanced render bg",
				Layout: state.handleLayout,
				Entries: []wgpu.BindGroupEntry{
					{Binding: 0, Buffer: worldBuffer, Offset: 0, Size: wgpu.WholeSize},
					{Binding: 1, Buffer: es.perEntityBuffer, Offset: 0, Size: uint64(unsafe.Sizeof(instancedPerEntity{}))},
				},
			})
			if err != nil {
				return
			}
			es.bindGroup = bg
			es.generation = generation
			es.worldBuffer = worldBuffer
		}
	})
	return nil
}

func instancedMeshExecute(state *instancedMeshState, context *render.PassContext) error {
	computeRes, ok := ecs.Resource[InstancedComputeResource](context.World)
	if !ok || computeRes == nil || computeRes.Compute == nil {
		return nil
	}
	compute := computeRes.Compute
	if len(state.perEntity) == 0 {
		return nil
	}
	assets := ecs.MustResource[asset.MeshAssetsResource](context.World).Assets

	colorAttachment, err := context.ColorAttachment("color")
	if err != nil {
		return err
	}
	colorAttachment.LoadOp = wgpu.LoadOpLoad
	entityIdAttachment, err := context.ColorAttachment("entity_id")
	if err != nil {
		return err
	}
	entityIdAttachment.LoadOp = wgpu.LoadOpLoad
	viewNormalsAttachment, err := context.ColorAttachment("view_normals")
	if err != nil {
		return err
	}
	viewNormalsAttachment.LoadOp = wgpu.LoadOpLoad
	depthAttachment, err := context.DepthAttachment("depth")
	if err != nil {
		return err
	}
	depthAttachment.DepthLoadOp = wgpu.LoadOpLoad

	prepassDepth, err := context.DepthAttachment("depth")
	if err != nil {
		return err
	}
	prepassDepth.DepthLoadOp = wgpu.LoadOpLoad
	prepassDepth.DepthStoreOp = wgpu.StoreOpStore

	mask := ecs.MustMaskOf[asset.InstancedMesh](context.World) |
		ecs.MustMaskOf[transform.GlobalTransform](context.World)
	drawInstances := func(enc *wgpu.RenderPassEncoder) {
		context.World.ForEach(mask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
			inst, _ := ecs.Get[asset.InstancedMesh](context.World, entity)
			if inst == nil {
				return
			}
			es := state.perEntity[entity]
			if es == nil || es.bindGroup == nil {
				return
			}
			count := compute.InstanceCount(entity)
			if count == 0 {
				return
			}
			entry, ok := assets.Lookup(inst.Mesh)
			if !ok {
				return
			}
			enc.SetBindGroup(2, es.bindGroup, nil)
			enc.SetVertexBuffer(0, entry.Vertices, 0, wgpu.WholeSize)
			enc.Draw(entry.VertexCount, count, 0, 0)
		})
	}

	prepass := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		Label:                  "instanced_mesh depth prepass",
		DepthStencilAttachment: &prepassDepth,
	})
	prepass.SetPipeline(state.prepassPipeline)
	prepass.SetBindGroup(0, state.viewProjGroup, nil)
	prepass.SetBindGroup(1, state.globalGroup, nil)
	drawInstances(prepass)
	prepass.End()
	prepass.Release()

	passEnc := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		Label:                  "instanced_mesh",
		ColorAttachments:       []wgpu.RenderPassColorAttachment{colorAttachment, entityIdAttachment, viewNormalsAttachment},
		DepthStencilAttachment: &depthAttachment,
	})
	passEnc.SetPipeline(state.pipeline)
	passEnc.SetBindGroup(0, state.viewProjGroup, nil)
	passEnc.SetBindGroup(1, state.globalGroup, nil)
	drawInstances(passEnc)
	passEnc.End()
	passEnc.Release()
	return nil
}

func instancedMeshRelease(state *instancedMeshState) {
	for _, es := range state.perEntity {
		if es.perEntityBuffer != nil {
			es.perEntityBuffer.Release()
		}
		if es.bindGroup != nil {
			es.bindGroup.Release()
		}
	}
	state.perEntity = nil
	if state.globalGroup != nil {
		state.globalGroup.Release()
	}
	if state.viewProjGroup != nil {
		state.viewProjGroup.Release()
	}
	if state.uniformBuffer != nil {
		state.uniformBuffer.Release()
	}
	if state.viewProjBuffer != nil {
		state.viewProjBuffer.Release()
	}
	if state.prepassPipeline != nil {
		state.prepassPipeline.Release()
	}
	if state.pipeline != nil {
		state.pipeline.Release()
	}
	if state.handleLayout != nil {
		state.handleLayout.Release()
	}
	if state.globalLayout != nil {
		state.globalLayout.Release()
	}
	if state.viewProjLayout != nil {
		state.viewProjLayout.Release()
	}
}
