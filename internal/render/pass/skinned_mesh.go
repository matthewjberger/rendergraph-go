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

//go:embed skinned_mesh.wgsl
var skinnedMeshShader string

type skinnedUniforms struct {
	LightDirection mgl32.Vec4
	LightColor     mgl32.Vec4
	AmbientColor   mgl32.Vec4
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

	handleBindGroup *wgpu.BindGroup
	jointGen        uint64
	instancesGen    uint64
	jointBuffer     *wgpu.Buffer
	instancesBuffer *wgpu.Buffer
}

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
		Entries: []wgpu.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: wgpu.ShaderStageFragment,
				Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform},
			},
			{
				Binding:    1,
				Visibility: wgpu.ShaderStageFragment,
				Texture: wgpu.TextureBindingLayout{
					SampleType:    wgpu.TextureSampleTypeFloat,
					ViewDimension: wgpu.TextureViewDimension2DArray,
				},
			},
			{
				Binding:    2,
				Visibility: wgpu.ShaderStageFragment,
				Sampler:    wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeFiltering},
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("skinned_mesh: global layout: %w", err)
	}
	handleLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "skinned_mesh handle layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageVertex | wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
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

	return &skinnedMeshPassState{
		pipeline:       pipeline,
		viewProjLayout: viewProjLayout,
		globalLayout:   globalLayout,
		handleLayout:   handleLayout,
		viewProjBuffer: viewProjBuffer,
		viewProjGroup:  viewProjGroup,
		uniformBuffer:  uniformBuffer,
		aspectFn:       aspect,
	}, nil
}

func ensureSkinnedHandleBindGroup(device *wgpu.Device, layout *wgpu.BindGroupLayout, cached **wgpu.BindGroup, jointGen, instGen *uint64, jointBuf, instBuf **wgpu.Buffer, sc *SkinningCompute) *wgpu.BindGroup {
	joint := sc.JointMatrixBuffer()
	instances := sc.InstancesBuffer()
	if joint == nil || instances == nil {
		return nil
	}
	jg := sc.Generation()
	ig := sc.InstancesGeneration()
	if *cached != nil && (*jointGen != jg || *instGen != ig || *jointBuf != joint || *instBuf != instances) {
		(*cached).Release()
		*cached = nil
	}
	if *cached == nil {
		bg, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  "skinned handle bg",
			Layout: layout,
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: joint, Offset: 0, Size: wgpu.WholeSize},
				{Binding: 1, Buffer: instances, Offset: 0, Size: wgpu.WholeSize},
			},
		})
		if err != nil {
			return nil
		}
		*cached = bg
		*jointGen = jg
		*instGen = ig
		*jointBuf = joint
		*instBuf = instances
	}
	return *cached
}

func skinnedMeshPrepare(s any, context *render.PassContext) error {
	state := s.(*skinnedMeshPassState)

	camera, hasCamera := ecs.Resource[render.Camera](context.World)
	if hasCamera && camera != nil {
		viewProj := render.CameraViewProjection(camera, state.aspectFn())
		writeBuffer(context.Device, context.Queue, context.Encoder, state.viewProjBuffer, 0, bytesOf(&viewProj))
	}

	uniforms := defaultSkinnedUniforms()
	applyDirectionalToSkinned(context.World, &uniforms)
	writeBuffer(context.Device, context.Queue, context.Encoder, state.uniformBuffer, 0, bytesOf(&uniforms))

	if state.globalGroup == nil {
		arraysResource, ok := ecs.Resource[asset.MaterialTextureArraysResource](context.World)
		if !ok || arraysResource == nil || arraysResource.Arrays == nil {
			return fmt.Errorf("skinned_mesh: MaterialTextureArraysResource missing")
		}
		arrays := arraysResource.Arrays
		bg, err := context.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  "skinned_mesh global bg",
			Layout: state.globalLayout,
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: state.uniformBuffer, Offset: 0, Size: uint64(unsafe.Sizeof(skinnedUniforms{}))},
				{Binding: 1, TextureView: arrays.SRGBView},
				{Binding: 2, Sampler: arrays.Sampler},
			},
		})
		if err != nil {
			return fmt.Errorf("skinned_mesh: global bg: %w", err)
		}
		state.globalGroup = bg
	}

	skinningRes, ok := ecs.Resource[SkinningComputeResource](context.World)
	if !ok || skinningRes == nil || skinningRes.Compute == nil {
		return nil
	}
	ensureSkinnedHandleBindGroup(context.Device, state.handleLayout, &state.handleBindGroup,
		&state.jointGen, &state.instancesGen, &state.jointBuffer, &state.instancesBuffer, skinningRes.Compute)
	return nil
}

func skinnedMeshExecute(s any, context *render.PassContext) error {
	state := s.(*skinnedMeshPassState)
	if state.handleBindGroup == nil {
		return nil
	}
	skinningRes, ok := ecs.Resource[SkinningComputeResource](context.World)
	if !ok || skinningRes == nil || skinningRes.Compute == nil {
		return nil
	}
	groups := skinningRes.Compute.OpaqueGroups()
	if len(groups) == 0 {
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
	passEnc.SetBindGroup(2, state.handleBindGroup, nil)

	for _, group := range groups {
		entry, ok := assets.Lookup(group.Mesh)
		if !ok || entry == nil {
			continue
		}
		passEnc.SetVertexBuffer(0, entry.Vertices, 0, wgpu.WholeSize)
		passEnc.Draw(entry.VertexCount, group.Count, 0, group.FirstInstance)
	}
	passEnc.End()
	passEnc.Release()
	return nil
}

func skinnedMeshRelease(s any) {
	state := s.(*skinnedMeshPassState)
	if state.handleBindGroup != nil {
		state.handleBindGroup.Release()
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

func applyDirectionalToSkinned(world *ecs.World, uniforms *skinnedUniforms) {
	lightMask := ecs.MustMaskOf[render.Light](world)
	world.ForEach(lightMask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		light, ok := ecs.Get[render.Light](world, entity)
		if !ok || light.Type != render.LightTypeDirectional {
			return
		}
		intensity := light.Intensity
		if intensity <= 0 {
			intensity = 1.0
		}
		uniforms.LightColor = mgl32.Vec4{light.Color[0] * intensity, light.Color[1] * intensity, light.Color[2] * intensity, 1}
		if global, ok := ecs.Get[transform.GlobalTransform](world, entity); ok {
			forward := mgl32.Vec3{-global.Matrix[8], -global.Matrix[9], -global.Matrix[10]}
			if forward.Len() > 1e-4 {
				forward = forward.Normalize()
				uniforms.LightDirection = mgl32.Vec4{forward[0], forward[1], forward[2], 0}
			}
		}
	})
}

func defaultSkinnedUniforms() skinnedUniforms {
	return skinnedUniforms{
		LightDirection: mgl32.Vec4{0, -1, 0, 0},
		LightColor:     mgl32.Vec4{1, 1, 1, 1},
		AmbientColor:   mgl32.Vec4{0.15, 0.18, 0.22, 1},
	}
}
