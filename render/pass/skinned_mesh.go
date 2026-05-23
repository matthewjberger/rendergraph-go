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

//go:embed skinned_mesh.wgsl
var skinnedMeshShader string

type skinnedUniforms struct {
	LightDirection mgl32.Vec4
	LightColor     mgl32.Vec4
	AmbientColor   mgl32.Vec4
}

type skinnedHandleBuffers struct {
	entityIdBuffer *wgpu.Buffer
	entityCapacity uint32
	bindGroup      *wgpu.BindGroup
}

type skinnedMeshPassState struct {
	pipeline       *wgpu.RenderPipeline
	viewProjLayout *wgpu.BindGroupLayout
	globalLayout   *wgpu.BindGroupLayout
	handleLayout   *wgpu.BindGroupLayout
	viewProjBuffer *wgpu.Buffer
	viewProjGroup  *wgpu.BindGroup
	uniformBuffer  *wgpu.Buffer
	globalGroup    *wgpu.BindGroup
	aspectFn       func() float32

	// perEntity caches the per-handle bind group keyed by the
	// rendered skinned entity. A skinned mesh has one Skin per
	// entity (no instancing yet), so the cache is entity-scoped.
	perEntity map[ecs.Entity]*skinnedHandleBuffers
}

// AddSkinnedMeshPass registers the skinned mesh render pass. It
// runs after the static mesh pass and writes the same scene_color
// + entity_id + view_normals attachments so the postprocess and
// G-buffer-consuming passes work uniformly across static and
// skinned geometry.
func AddSkinnedMeshPass(renderer *render.Renderer) (*render.Pass, error) {
	state, err := newSkinnedMeshState(renderer.Device, renderer.AspectRatio)
	if err != nil {
		return nil, err
	}
	pass := &render.Pass{
		Name:    "skinned_mesh",
		Writes:  []string{"color", "depth", "entity_id", "view_normals"},
		State:   state,
		Prepare: skinnedMeshPrepare,
		Execute: skinnedMeshExecute,
		Release: skinnedMeshRelease,
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

func newSkinnedMeshState(device *wgpu.Device, aspect func() float32) (*skinnedMeshPassState, error) {
	module, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "skinned_mesh shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: skinnedMeshShader},
	})
	if err != nil {
		return nil, fmt.Errorf("skinned_mesh: shader: %w", err)
	}
	defer module.Release()

	viewProjLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "skinned_mesh view_proj layout",
		Entries: []wgpu.BindGroupLayoutEntry{{
			Binding:    0,
			Visibility: wgpu.ShaderStageVertex,
			Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform},
		}},
	})
	if err != nil {
		return nil, fmt.Errorf("skinned_mesh: view_proj layout: %w", err)
	}
	globalLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "skinned_mesh global layout",
		Entries: []wgpu.BindGroupLayoutEntry{{
			Binding:    0,
			Visibility: wgpu.ShaderStageFragment,
			Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform},
		}},
	})
	if err != nil {
		return nil, fmt.Errorf("skinned_mesh: global layout: %w", err)
	}
	handleLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "skinned_mesh handle layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("skinned_mesh: handle layout: %w", err)
	}

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "skinned_mesh pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{viewProjLayout, globalLayout, handleLayout},
	})
	if err != nil {
		return nil, fmt.Errorf("skinned_mesh: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	pipeline, err := device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "skinned_mesh pipeline",
		Layout: pipelineLayout,
		Vertex: wgpu.VertexState{
			Module:     module,
			EntryPoint: "vertex_main",
			Buffers: []wgpu.VertexBufferLayout{{
				ArrayStride: uint64(unsafe.Sizeof(asset.SkinnedMeshVertex{})),
				StepMode:    wgpu.VertexStepModeVertex,
				Attributes: []wgpu.VertexAttribute{
					{Format: wgpu.VertexFormatFloat32x4, Offset: 0, ShaderLocation: 0},
					{Format: wgpu.VertexFormatFloat32x4, Offset: 16, ShaderLocation: 1},
					{Format: wgpu.VertexFormatFloat32x4, Offset: 32, ShaderLocation: 2},
					{Format: wgpu.VertexFormatFloat32x4, Offset: 48, ShaderLocation: 3},
					{Format: wgpu.VertexFormatFloat32x4, Offset: 64, ShaderLocation: 4},
					{Format: wgpu.VertexFormatUint32x4, Offset: 80, ShaderLocation: 5},
					{Format: wgpu.VertexFormatFloat32x4, Offset: 96, ShaderLocation: 6},
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
			DepthWriteEnabled: true,
			DepthCompare:      wgpu.CompareFunctionLess,
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
		return nil, fmt.Errorf("skinned_mesh: pipeline: %w", err)
	}

	viewProjBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "skinned_mesh view_proj",
		Size:  uint64(unsafe.Sizeof(mgl32.Mat4{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("skinned_mesh: view_proj buffer: %w", err)
	}
	viewProjGroup, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:   "skinned_mesh view_proj bg",
		Layout:  viewProjLayout,
		Entries: []wgpu.BindGroupEntry{{Binding: 0, Buffer: viewProjBuffer, Offset: 0, Size: uint64(unsafe.Sizeof(mgl32.Mat4{}))}},
	})
	if err != nil {
		return nil, fmt.Errorf("skinned_mesh: view_proj bg: %w", err)
	}

	uniformBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "skinned_mesh uniforms",
		Size:  uint64(unsafe.Sizeof(skinnedUniforms{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("skinned_mesh: uniforms buffer: %w", err)
	}
	globalGroup, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:   "skinned_mesh global bg",
		Layout:  globalLayout,
		Entries: []wgpu.BindGroupEntry{{Binding: 0, Buffer: uniformBuffer, Offset: 0, Size: uint64(unsafe.Sizeof(skinnedUniforms{}))}},
	})
	if err != nil {
		return nil, fmt.Errorf("skinned_mesh: global bg: %w", err)
	}

	return &skinnedMeshPassState{
		pipeline:       pipeline,
		viewProjLayout: viewProjLayout,
		globalLayout:   globalLayout,
		handleLayout:   handleLayout,
		viewProjBuffer: viewProjBuffer,
		viewProjGroup:  viewProjGroup,
		uniformBuffer:  uniformBuffer,
		globalGroup:    globalGroup,
		aspectFn:       aspect,
		perEntity:      make(map[ecs.Entity]*skinnedHandleBuffers),
	}, nil
}

func skinnedMeshPrepare(s any, context *render.PassContext) error {
	state := s.(*skinnedMeshPassState)

	camera, hasCamera := ecs.Resource[render.Camera](context.World)
	if hasCamera && camera != nil {
		viewProj := render.CameraViewProjection(camera, state.aspectFn())
		writeBuffer(context.Device, context.Queue, context.Encoder, state.viewProjBuffer, 0, bytesOf(&viewProj))
	}

	uniforms := defaultSkinnedUniforms()
	lightMask := ecs.MustMaskOf[render.Light](context.World)
	context.World.ForEach(lightMask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		light, ok := ecs.Get[render.Light](context.World, entity)
		if !ok || light.Type != render.LightTypeDirectional {
			return
		}
		intensity := light.Intensity
		if intensity <= 0 {
			intensity = 1.0
		}
		color := mgl32.Vec3{light.Color[0] * intensity, light.Color[1] * intensity, light.Color[2] * intensity}
		uniforms.LightColor = mgl32.Vec4{color[0], color[1], color[2], 1}
		if global, ok := ecs.Get[transform.GlobalTransform](context.World, entity); ok {
			forward := mgl32.Vec3{-global.Matrix[8], -global.Matrix[9], -global.Matrix[10]}
			if forward.Len() > 1e-4 {
				forward = forward.Normalize()
				uniforms.LightDirection = mgl32.Vec4{forward[0], forward[1], forward[2], 0}
			}
		}
	})
	writeBuffer(context.Device, context.Queue, context.Encoder, state.uniformBuffer, 0, bytesOf(&uniforms))

	skinnedMask := ecs.MustMaskOf[asset.SkinnedMesh](context.World) |
		ecs.MustMaskOf[transform.GlobalTransform](context.World)
	context.World.ForEach(skinnedMask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		skinned, _ := ecs.Get[asset.SkinnedMesh](context.World, entity)
		if skinned == nil || skinned.Skin == nil {
			return
		}
		buffers := state.perEntity[entity]
		if buffers == nil {
			buffers = &skinnedHandleBuffers{}
			state.perEntity[entity] = buffers
		}
		if err := ensureSkinnedBuffers(buffers, context.Device, 1); err != nil {
			delete(state.perEntity, entity)
			return
		}
		id := uint32(entity.ID)
		writeBuffer(context.Device, context.Queue, context.Encoder, buffers.entityIdBuffer, 0, bytesOf(&id))
		if buffers.bindGroup == nil {
			bg, err := context.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
				Label:  "skinned_mesh handle bg",
				Layout: state.handleLayout,
				Entries: []wgpu.BindGroupEntry{
					{Binding: 0, Buffer: skinned.Skin.JointMatrixBuffer, Offset: 0, Size: wgpu.WholeSize},
					{Binding: 1, Buffer: buffers.entityIdBuffer, Offset: 0, Size: wgpu.WholeSize},
				},
			})
			if err != nil {
				return
			}
			buffers.bindGroup = bg
		}
	})
	return nil
}

func skinnedMeshExecute(s any, context *render.PassContext) error {
	state := s.(*skinnedMeshPassState)
	skinnedMask := ecs.MustMaskOf[asset.SkinnedMesh](context.World) |
		ecs.MustMaskOf[transform.GlobalTransform](context.World)
	hasAny := false
	context.World.ForEach(skinnedMask, 0, func(_ ecs.Entity, _ *ecs.Archetype, _ int) {
		hasAny = true
	})
	if !hasAny {
		return nil
	}

	colorAttachment, err := context.ColorAttachment("color")
	if err != nil {
		return err
	}
	entityIdAttachment, err := context.ColorAttachment("entity_id")
	if err != nil {
		return err
	}
	viewNormalsAttachment, err := context.ColorAttachment("view_normals")
	if err != nil {
		return err
	}
	depthAttachment, err := context.DepthAttachment("depth")
	if err != nil {
		return err
	}

	assets := ecs.MustResource[asset.SkinnedMeshAssetsResource](context.World).Assets

	passEnc := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		Label:                  "skinned_mesh",
		ColorAttachments:       []wgpu.RenderPassColorAttachment{colorAttachment, entityIdAttachment, viewNormalsAttachment},
		DepthStencilAttachment: &depthAttachment,
	})
	passEnc.SetPipeline(state.pipeline)
	passEnc.SetBindGroup(0, state.viewProjGroup, nil)
	passEnc.SetBindGroup(1, state.globalGroup, nil)

	context.World.ForEach(skinnedMask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		skinned, _ := ecs.Get[asset.SkinnedMesh](context.World, entity)
		if skinned == nil {
			return
		}
		buffers := state.perEntity[entity]
		if buffers == nil || buffers.bindGroup == nil {
			return
		}
		entry, ok := assets.Lookup(skinned.Mesh)
		if !ok || entry == nil {
			return
		}
		passEnc.SetBindGroup(2, buffers.bindGroup, nil)
		passEnc.SetVertexBuffer(0, entry.Vertices, 0, wgpu.WholeSize)
		passEnc.Draw(entry.VertexCount, 1, 0, 0)
	})
	passEnc.End()
	passEnc.Release()
	return nil
}

func skinnedMeshRelease(s any) {
	state := s.(*skinnedMeshPassState)
	for entity, buffers := range state.perEntity {
		if buffers.bindGroup != nil {
			buffers.bindGroup.Release()
		}
		if buffers.entityIdBuffer != nil {
			buffers.entityIdBuffer.Release()
		}
		delete(state.perEntity, entity)
	}
	if state.globalGroup != nil {
		state.globalGroup.Release()
	}
	if state.uniformBuffer != nil {
		state.uniformBuffer.Release()
	}
	if state.viewProjGroup != nil {
		state.viewProjGroup.Release()
	}
	if state.viewProjBuffer != nil {
		state.viewProjBuffer.Release()
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

func ensureSkinnedBuffers(buffers *skinnedHandleBuffers, device *wgpu.Device, count uint32) error {
	if buffers.entityCapacity < count {
		if buffers.entityIdBuffer != nil {
			buffers.entityIdBuffer.Release()
		}
		// Storage buffers need a minimum 16-byte allocation to be
		// valid bind targets on every backend, even when the data
		// is a single u32.
		size := uint64(count) * 4
		if size < 16 {
			size = 16
		}
		buf, err := device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: "skinned_mesh entity_id buffer",
			Size:  size,
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
		})
		if err != nil {
			return err
		}
		buffers.entityIdBuffer = buf
		buffers.entityCapacity = count
		buffers.bindGroup = nil
	}
	return nil
}

func defaultSkinnedUniforms() skinnedUniforms {
	return skinnedUniforms{
		LightDirection: mgl32.Vec4{0, -1, 0, 0},
		LightColor:     mgl32.Vec4{1, 1, 1, 1},
		AmbientColor:   mgl32.Vec4{0.15, 0.18, 0.22, 1},
	}
}

// PrepareSkinMatrices walks every SkinnedMesh entity, builds the
// joint matrix array from each joint's GlobalTransform and inverse
// bind matrix, and uploads it to the skin's storage buffer.
func PrepareSkinMatrices(world *ecs.World) {
	if !ecs.HasResource[render.RendererResource](world) {
		return
	}
	renderer := ecs.MustResource[render.RendererResource](world).Renderer
	queue := renderer.Queue
	mask := ecs.MustMaskOf[asset.SkinnedMesh](world)
	world.ForEach(mask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		skinned, _ := ecs.Get[asset.SkinnedMesh](world, entity)
		if skinned == nil || skinned.Skin == nil {
			return
		}
		skin := skinned.Skin
		if len(skin.Joints) != len(skin.InverseBindMatrices) || len(skin.Joints) != len(skin.JointMatrices) {
			return
		}
		for jointIndex, jointEntity := range skin.Joints {
			global, ok := ecs.Get[transform.GlobalTransform](world, jointEntity)
			if !ok {
				skin.JointMatrices[jointIndex] = skin.InverseBindMatrices[jointIndex]
				continue
			}
			skin.JointMatrices[jointIndex] = global.Matrix.Mul4(skin.InverseBindMatrices[jointIndex])
		}
		if len(skin.JointMatrices) == 0 {
			return
		}
		bytesPerMatrix := int(unsafe.Sizeof(mgl32.Mat4{}))
		data := unsafe.Slice((*byte)(unsafe.Pointer(&skin.JointMatrices[0])), len(skin.JointMatrices)*bytesPerMatrix)
		queue.WriteBuffer(skin.JointMatrixBuffer, 0, data)
	})
}
