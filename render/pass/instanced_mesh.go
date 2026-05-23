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

type instancedEntityState struct {
	capacity         int
	uploadedCount    int
	localBuffer      *wgpu.Buffer
	worldBuffer      *wgpu.Buffer
	computeUniform   *wgpu.Buffer
	computeBindGroup *wgpu.BindGroup
	perEntityBuffer  *wgpu.Buffer
	renderBindGroup  *wgpu.BindGroup
	instanceCount    uint32
}

type instancedMeshState struct {
	computePipeline *wgpu.ComputePipeline
	computeLayout   *wgpu.BindGroupLayout

	pipeline       *wgpu.RenderPipeline
	viewProjLayout *wgpu.BindGroupLayout
	globalLayout   *wgpu.BindGroupLayout
	handleLayout   *wgpu.BindGroupLayout
	viewProjBuffer *wgpu.Buffer
	viewProjGroup  *wgpu.BindGroup
	uniformBuffer  *wgpu.Buffer
	globalGroup    *wgpu.BindGroup
	aspectFn       func() float32

	perEntity map[ecs.Entity]*instancedEntityState
}

// AddInstancedMeshPass renders InstancedMesh entities. A compute
// dispatch multiplies the entity's parent GlobalTransform by each
// instance's local matrix to fill a world-matrix buffer, then a
// single instanced draw renders every instance reading that buffer
// by instance index. Writes scene_color + entity_id + view_normals
// like the other geometry passes.
func AddInstancedMeshPass(renderer *render.Renderer) (*render.Pass, error) {
	state, err := newInstancedMeshState(renderer.Device, renderer.AspectRatio)
	if err != nil {
		return nil, err
	}
	pass := &render.Pass{
		Name:    "instanced_mesh",
		Writes:  []string{"color", "depth", "entity_id", "view_normals"},
		State:   state,
		Prepare: instancedMeshPrepare,
		Execute: instancedMeshExecute,
		Release: instancedMeshRelease,
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
	computeModule, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "instanced_transform shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: instancedTransformShader},
	})
	if err != nil {
		return nil, fmt.Errorf("instanced_mesh: compute shader: %w", err)
	}
	defer computeModule.Release()

	computeLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "instanced_transform layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("instanced_mesh: compute layout: %w", err)
	}
	computePipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "instanced_transform pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{computeLayout},
	})
	if err != nil {
		computeLayout.Release()
		return nil, fmt.Errorf("instanced_mesh: compute pipeline layout: %w", err)
	}
	defer computePipelineLayout.Release()
	computePipeline, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Layout:  computePipelineLayout,
		Compute: wgpu.ProgrammableStageDescriptor{Module: computeModule, EntryPoint: "main"},
	})
	if err != nil {
		computeLayout.Release()
		return nil, fmt.Errorf("instanced_mesh: compute pipeline: %w", err)
	}

	module, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "instanced_mesh shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: instancedMeshShader},
	})
	if err != nil {
		computePipeline.Release()
		computeLayout.Release()
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
		computePipeline.Release()
		computeLayout.Release()
		return nil, fmt.Errorf("instanced_mesh: view_proj layout: %w", err)
	}
	globalLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "instanced_mesh global layout",
		Entries: []wgpu.BindGroupLayoutEntry{{
			Binding: 0, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform},
		}},
	})
	if err != nil {
		computePipeline.Release()
		computeLayout.Release()
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
		computePipeline.Release()
		computeLayout.Release()
		viewProjLayout.Release()
		globalLayout.Release()
		return nil, fmt.Errorf("instanced_mesh: handle layout: %w", err)
	}

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "instanced_mesh pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{viewProjLayout, globalLayout, handleLayout},
	})
	if err != nil {
		computePipeline.Release()
		computeLayout.Release()
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
			DepthWriteEnabled: true,
			DepthCompare:      wgpu.CompareFunctionLess,
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
		computePipeline.Release()
		computeLayout.Release()
		viewProjLayout.Release()
		globalLayout.Release()
		handleLayout.Release()
		return nil, fmt.Errorf("instanced_mesh: pipeline: %w", err)
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
		computePipeline: computePipeline,
		computeLayout:   computeLayout,
		pipeline:        pipeline,
		viewProjLayout:  viewProjLayout,
		globalLayout:    globalLayout,
		handleLayout:    handleLayout,
		viewProjBuffer:  viewProjBuffer,
		viewProjGroup:   viewProjGroup,
		uniformBuffer:   uniformBuffer,
		aspectFn:        aspect,
		perEntity:       make(map[ecs.Entity]*instancedEntityState),
	}, nil
}

func instancedMeshPrepare(s any, context *render.PassContext) error {
	state := s.(*instancedMeshState)

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

	mask := ecs.MustMaskOf[asset.InstancedMesh](context.World) |
		ecs.MustMaskOf[transform.GlobalTransform](context.World)
	context.World.ForEach(mask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		inst, _ := ecs.Get[asset.InstancedMesh](context.World, entity)
		if inst == nil || len(inst.Instances) == 0 {
			return
		}
		global, ok := ecs.Get[transform.GlobalTransform](context.World, entity)
		if !ok {
			return
		}
		es := state.perEntity[entity]
		if es == nil {
			es = &instancedEntityState{}
			state.perEntity[entity] = es
		}
		count := len(inst.Instances)
		matBytes := int(unsafe.Sizeof(mgl32.Mat4{}))
		if es.capacity < count {
			if es.localBuffer != nil {
				es.localBuffer.Release()
			}
			if es.worldBuffer != nil {
				es.worldBuffer.Release()
			}
			localBuf, err := context.Device.CreateBuffer(&wgpu.BufferDescriptor{
				Label: "instanced local matrices",
				Size:  uint64(count) * uint64(matBytes),
				Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
			})
			if err != nil {
				delete(state.perEntity, entity)
				return
			}
			worldBuf, err := context.Device.CreateBuffer(&wgpu.BufferDescriptor{
				Label: "instanced world matrices",
				Size:  uint64(count) * uint64(matBytes),
				Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
			})
			if err != nil {
				localBuf.Release()
				delete(state.perEntity, entity)
				return
			}
			es.localBuffer = localBuf
			es.worldBuffer = worldBuf
			es.capacity = count
			es.uploadedCount = 0
			if es.computeBindGroup != nil {
				es.computeBindGroup.Release()
				es.computeBindGroup = nil
			}
			if es.renderBindGroup != nil {
				es.renderBindGroup.Release()
				es.renderBindGroup = nil
			}
		}
		if es.computeUniform == nil {
			buf, err := context.Device.CreateBuffer(&wgpu.BufferDescriptor{
				Label: "instanced compute uniform",
				Size:  uint64(unsafe.Sizeof(instancedComputeUniform{})),
				Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
			})
			if err != nil {
				return
			}
			es.computeUniform = buf
		}
		if es.perEntityBuffer == nil {
			buf, err := context.Device.CreateBuffer(&wgpu.BufferDescriptor{
				Label: "instanced per-entity",
				Size:  uint64(unsafe.Sizeof(instancedPerEntity{})),
				Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
			})
			if err != nil {
				return
			}
			es.perEntityBuffer = buf
		}

		// Local matrices are static unless the instance list grows;
		// upload once per (re)allocation.
		if es.uploadedCount != count {
			writeBuffer(context.Device, context.Queue, context.Encoder, es.localBuffer, 0, unsafe.Slice((*byte)(unsafe.Pointer(&inst.Instances[0])), count*matBytes))
			es.uploadedCount = count
		}

		cu := instancedComputeUniform{ParentTransform: global.Matrix, InstanceCount: uint32(count)}
		writeBuffer(context.Device, context.Queue, context.Encoder, es.computeUniform, 0, bytesOf(&cu))

		per := instancedPerEntity{BaseColor: [4]float32{1, 1, 1, 1}, EntityID: entity.ID}
		if material, ok := ecs.Get[asset.Material](context.World, entity); ok && material != nil {
			per.BaseColor = material.BaseColor
		}
		writeBuffer(context.Device, context.Queue, context.Encoder, es.perEntityBuffer, 0, bytesOf(&per))

		if es.computeBindGroup == nil {
			bg, err := context.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
				Label:  "instanced compute bg",
				Layout: state.computeLayout,
				Entries: []wgpu.BindGroupEntry{
					{Binding: 0, Buffer: es.localBuffer, Offset: 0, Size: wgpu.WholeSize},
					{Binding: 1, Buffer: es.worldBuffer, Offset: 0, Size: wgpu.WholeSize},
					{Binding: 2, Buffer: es.computeUniform, Offset: 0, Size: uint64(unsafe.Sizeof(instancedComputeUniform{}))},
				},
			})
			if err != nil {
				return
			}
			es.computeBindGroup = bg
		}
		if es.renderBindGroup == nil {
			bg, err := context.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
				Label:  "instanced render bg",
				Layout: state.handleLayout,
				Entries: []wgpu.BindGroupEntry{
					{Binding: 0, Buffer: es.worldBuffer, Offset: 0, Size: wgpu.WholeSize},
					{Binding: 1, Buffer: es.perEntityBuffer, Offset: 0, Size: uint64(unsafe.Sizeof(instancedPerEntity{}))},
				},
			})
			if err != nil {
				return
			}
			es.renderBindGroup = bg
		}
		es.instanceCount = uint32(count)
	})
	return nil
}

func instancedMeshExecute(s any, context *render.PassContext) error {
	state := s.(*instancedMeshState)
	if len(state.perEntity) == 0 {
		return nil
	}
	assets := ecs.MustResource[asset.MeshAssetsResource](context.World).Assets

	// Compute world matrices for every instanced entity before the
	// render pass opens.
	computePass := context.Encoder.BeginComputePass(&wgpu.ComputePassDescriptor{})
	computePass.SetPipeline(state.computePipeline)
	for _, es := range state.perEntity {
		if es.computeBindGroup == nil || es.instanceCount == 0 {
			continue
		}
		computePass.SetBindGroup(0, es.computeBindGroup, nil)
		computePass.DispatchWorkgroups((es.instanceCount+63)/64, 1, 1)
	}
	computePass.End()
	computePass.Release()

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

	passEnc := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		Label:                  "instanced_mesh",
		ColorAttachments:       []wgpu.RenderPassColorAttachment{colorAttachment, entityIdAttachment, viewNormalsAttachment},
		DepthStencilAttachment: &depthAttachment,
	})
	passEnc.SetPipeline(state.pipeline)
	passEnc.SetBindGroup(0, state.viewProjGroup, nil)
	passEnc.SetBindGroup(1, state.globalGroup, nil)

	mask := ecs.MustMaskOf[asset.InstancedMesh](context.World) |
		ecs.MustMaskOf[transform.GlobalTransform](context.World)
	context.World.ForEach(mask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		inst, _ := ecs.Get[asset.InstancedMesh](context.World, entity)
		if inst == nil {
			return
		}
		es := state.perEntity[entity]
		if es == nil || es.renderBindGroup == nil || es.instanceCount == 0 {
			return
		}
		entry, ok := assets.Lookup(inst.Mesh)
		if !ok {
			return
		}
		passEnc.SetBindGroup(2, es.renderBindGroup, nil)
		passEnc.SetVertexBuffer(0, entry.Vertices, 0, wgpu.WholeSize)
		passEnc.Draw(entry.VertexCount, es.instanceCount, 0, 0)
	})
	passEnc.End()
	passEnc.Release()
	return nil
}

func instancedMeshRelease(s any) {
	state := s.(*instancedMeshState)
	for _, es := range state.perEntity {
		for _, b := range []*wgpu.Buffer{es.localBuffer, es.worldBuffer, es.computeUniform, es.perEntityBuffer} {
			if b != nil {
				b.Release()
			}
		}
		if es.computeBindGroup != nil {
			es.computeBindGroup.Release()
		}
		if es.renderBindGroup != nil {
			es.renderBindGroup.Release()
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
	if state.pipeline != nil {
		state.pipeline.Release()
	}
	if state.computePipeline != nil {
		state.computePipeline.Release()
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
	if state.computeLayout != nil {
		state.computeLayout.Release()
	}
}
