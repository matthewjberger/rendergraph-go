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

// skinnedUniforms is retained for the instanced mesh pass, which still uses the
// lightweight directional-light model.
type skinnedUniforms struct {
	LightDirection mgl32.Vec4
	LightColor     mgl32.Vec4
	AmbientColor   mgl32.Vec4
}

// skinnedMeshPassState renders opaque and masked skinned meshes with the same
// full PBR path as the static mesh pass. It reuses that pass's view-proj, global
// (lights/shadows/material registry/texture arrays) and IBL bind groups; only
// group 2 (joint matrices + per-instance data) is skinned-specific.
type skinnedMeshPassState struct {
	pipeline     *wgpu.RenderPipeline
	handleLayout *wgpu.BindGroupLayout

	handleBindGroup *wgpu.BindGroup
	jointGen        uint64
	instancesGen    uint64
	jointBuffer     *wgpu.Buffer
	instancesBuffer *wgpu.Buffer
}

func AddSkinnedMeshPass(renderer *render.Renderer) (*render.Pass, error) {
	state, err := newSkinnedMeshState(renderer.Device)
	if err != nil {
		return nil, err
	}
	pass := &render.Pass{
		Name:    "skinned_mesh",
		Writes:  []string{"color", "depth", "entity_id", "view_normals"},
		Prepare: func(c *render.PassContext) error { return skinnedMeshPrepare(state, c) },
		Execute: func(c *render.PassContext) error { return skinnedMeshExecute(state, c) },
		Release: func() { skinnedMeshRelease(state) },
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

func newSkinnedMeshState(device *wgpu.Device) (*skinnedMeshPassState, error) {
	meshState, ok := findMeshPassState(nil)
	if !ok {
		return nil, fmt.Errorf("skinned_mesh: mesh pass must be created before the skinned mesh pass")
	}

	module, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "skinned_mesh shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: skinnedMeshShader},
	})
	if err != nil {
		return nil, fmt.Errorf("skinned_mesh: shader: %w", err)
	}
	defer module.Release()

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
		Label: "skinned_mesh pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{
			meshState.viewProjLayout,
			meshState.globalBgLayout,
			handleLayout,
			meshState.iblBgLayout,
		},
	})
	if err != nil {
		handleLayout.Release()
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
			DepthCompare:      wgpu.CompareFunctionGreater,
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
		handleLayout.Release()
		return nil, fmt.Errorf("skinned_mesh: pipeline: %w", err)
	}

	return &skinnedMeshPassState{
		pipeline:     pipeline,
		handleLayout: handleLayout,
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

func skinnedMeshPrepare(state *skinnedMeshPassState, context *render.PassContext) error {
	skinningRes, ok := ecs.Resource[SkinningComputeResource](context.World)
	if !ok || skinningRes == nil || skinningRes.Compute == nil {
		return nil
	}
	ensureSkinnedHandleBindGroup(context.Device, state.handleLayout, &state.handleBindGroup,
		&state.jointGen, &state.instancesGen, &state.jointBuffer, &state.instancesBuffer, skinningRes.Compute)
	return nil
}

func skinnedMeshExecute(state *skinnedMeshPassState, context *render.PassContext) error {
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
	meshState, ok := findMeshPassState(context.World)
	if !ok || meshState.globalBindGroup == nil || meshState.iblBindGroup == nil || meshState.viewProjBindGroup == nil {
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
	passEnc.SetBindGroup(0, meshState.viewProjBindGroup, nil)
	passEnc.SetBindGroup(1, meshState.globalBindGroup, nil)
	passEnc.SetBindGroup(2, state.handleBindGroup, nil)
	passEnc.SetBindGroup(3, meshState.iblBindGroup, nil)

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

func skinnedMeshRelease(state *skinnedMeshPassState) {
	if state.handleBindGroup != nil {
		state.handleBindGroup.Release()
	}
	if state.pipeline != nil {
		state.pipeline.Release()
	}
	if state.handleLayout != nil {
		state.handleLayout.Release()
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
