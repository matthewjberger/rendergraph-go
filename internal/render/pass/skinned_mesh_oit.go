package pass

import (
	"fmt"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"

	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/render/asset"
)

// skinnedMeshOitState renders blend and transmissive skinned meshes through the
// same full PBR + screen-space transmission path as the static OIT pass. It
// reuses the forward mesh pass's view-proj, global (lights/shadows/material
// registry/texture arrays) and IBL bind groups verbatim; only group 2 is
// skinned-specific (joint matrices + per-instance data that indexes the shared
// material registry).
type skinnedMeshOitState struct {
	pipeline        *wgpu.RenderPipeline
	prepassPipeline *wgpu.RenderPipeline
	handleLayout    *wgpu.BindGroupLayout

	handleBindGroup *wgpu.BindGroup
	jointGen        uint64
	instancesGen    uint64
	jointBuffer     *wgpu.Buffer
	instancesBuffer *wgpu.Buffer
	morphBuffer     *wgpu.Buffer
}

func AddSkinnedMeshOitPass(renderer *render.Renderer) (*render.Pass, error) {
	state, err := newSkinnedMeshOitState(renderer.Device)
	if err != nil {
		return nil, err
	}
	pass := &render.Pass{
		Name:    "skinned_mesh_oit",
		Reads:   []string{"depth"},
		Writes:  []string{"oit_accum", "oit_reveal", "entity_id"},
		Prepare: func(c *render.PassContext) error { return skinnedMeshOitPrepare(state, c) },
		Execute: func(c *render.PassContext) error { return skinnedMeshOitExecute(state, c) },
		Release: func() { skinnedMeshOitRelease(state) },
	}
	if err := renderer.Graph.AddPass(pass, []render.SlotBinding{
		{Slot: "oit_accum", ResourceID: renderer.OitAccumID},
		{Slot: "oit_reveal", ResourceID: renderer.OitRevealID},
		{Slot: "entity_id", ResourceID: renderer.EntityIdID},
		{Slot: "depth", ResourceID: renderer.DepthID},
	}); err != nil {
		return nil, err
	}
	return pass, nil
}

func newSkinnedMeshOitState(device *wgpu.Device) (*skinnedMeshOitState, error) {
	meshState, ok := findMeshPassState(nil)
	if !ok {
		return nil, fmt.Errorf("skinned_mesh_oit: mesh pass must be created before the skinned OIT pass")
	}

	module, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "skinned_mesh_oit shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: skinnedMeshOitShader},
	})
	if err != nil {
		return nil, fmt.Errorf("skinned_mesh_oit: shader: %w", err)
	}
	defer module.Release()

	handleLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "skinned_mesh_oit handle layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 2, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("skinned_mesh_oit: handle layout: %w", err)
	}

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label: "skinned_mesh_oit pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{
			meshState.viewProjLayout,
			meshState.globalBgLayout,
			handleLayout,
			meshState.iblBgLayout,
		},
	})
	if err != nil {
		handleLayout.Release()
		return nil, fmt.Errorf("skinned_mesh_oit: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	vertexBuffers := []wgpu.VertexBufferLayout{{
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
	}}

	accumBlend := wgpu.BlendState{
		Color: wgpu.BlendComponent{SrcFactor: wgpu.BlendFactorOne, DstFactor: wgpu.BlendFactorOne, Operation: wgpu.BlendOperationAdd},
		Alpha: wgpu.BlendComponent{SrcFactor: wgpu.BlendFactorOne, DstFactor: wgpu.BlendFactorOne, Operation: wgpu.BlendOperationAdd},
	}
	revealBlend := wgpu.BlendState{
		Color: wgpu.BlendComponent{SrcFactor: wgpu.BlendFactorZero, DstFactor: wgpu.BlendFactorOneMinusSrc, Operation: wgpu.BlendOperationAdd},
		Alpha: wgpu.BlendComponent{SrcFactor: wgpu.BlendFactorZero, DstFactor: wgpu.BlendFactorOneMinusSrc, Operation: wgpu.BlendOperationAdd},
	}

	pipeline, err := device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "skinned_mesh_oit pipeline",
		Layout: pipelineLayout,
		Vertex: wgpu.VertexState{
			Module:     module,
			EntryPoint: "vertex_main",
			Buffers:    vertexBuffers,
		},
		Primitive: wgpu.PrimitiveState{
			Topology:  wgpu.PrimitiveTopologyTriangleList,
			FrontFace: wgpu.FrontFaceCCW,
			CullMode:  wgpu.CullModeNone,
		},
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
				{Format: render.HdrFormat, Blend: &accumBlend, WriteMask: wgpu.ColorWriteMaskAll},
				{Format: wgpu.TextureFormatR8Unorm, Blend: &revealBlend, WriteMask: wgpu.ColorWriteMaskRed},
				{Format: render.EntityIdFormat, WriteMask: wgpu.ColorWriteMaskRed},
			},
		},
	})
	if err != nil {
		handleLayout.Release()
		return nil, fmt.Errorf("skinned_mesh_oit: pipeline: %w", err)
	}

	prepassPipeline, err := device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "skinned_mesh_oit blend-opaque prepass pipeline",
		Layout: pipelineLayout,
		Vertex: wgpu.VertexState{
			Module:     module,
			EntryPoint: "vertex_main",
			Buffers:    vertexBuffers,
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
			StencilFront:      wgpu.StencilFaceState{Compare: wgpu.CompareFunctionAlways},
			StencilBack:       wgpu.StencilFaceState{Compare: wgpu.CompareFunctionAlways},
		},
		Multisample: wgpu.MultisampleState{Count: 1, Mask: 0xFFFFFFFF},
		Fragment: &wgpu.FragmentState{
			Module:     module,
			EntryPoint: "fs_blend_opaque_prepass",
		},
	})
	if err != nil {
		pipeline.Release()
		handleLayout.Release()
		return nil, fmt.Errorf("skinned_mesh_oit: blend-opaque prepass pipeline: %w", err)
	}

	return &skinnedMeshOitState{
		pipeline:        pipeline,
		prepassPipeline: prepassPipeline,
		handleLayout:    handleLayout,
	}, nil
}

func skinnedMeshOitPrepare(state *skinnedMeshOitState, context *render.PassContext) error {
	skinningRes, ok := ecs.Resource[SkinningComputeResource](context.World)
	if !ok || skinningRes == nil || skinningRes.Compute == nil {
		return nil
	}
	ensureSkinnedHandleBindGroup(context.Device, state.handleLayout, &state.handleBindGroup,
		&state.jointGen, &state.instancesGen, &state.jointBuffer, &state.instancesBuffer, &state.morphBuffer, skinningRes.Compute, skinnedMorphBuffer(context.World, context.Device))
	return nil
}

func skinnedMeshOitExecute(state *skinnedMeshOitState, context *render.PassContext) error {
	if state.handleBindGroup == nil {
		return nil
	}
	skinningRes, ok := ecs.Resource[SkinningComputeResource](context.World)
	if !ok || skinningRes == nil || skinningRes.Compute == nil {
		return nil
	}
	groups := skinningRes.Compute.BlendGroups()
	if len(groups) == 0 {
		return nil
	}
	meshState, ok := findMeshPassState(context.World)
	if !ok || meshState.globalBindGroup == nil || meshState.iblBindGroup == nil || meshState.viewProjBindGroup == nil {
		return nil
	}
	assets := ecs.MustResource[asset.SkinnedMeshAssetsResource](context.World).Assets

	accum, err := context.ColorAttachment("oit_accum")
	if err != nil {
		return err
	}
	accum.LoadOp = wgpu.LoadOpLoad
	reveal, err := context.ColorAttachment("oit_reveal")
	if err != nil {
		return err
	}
	reveal.LoadOp = wgpu.LoadOpLoad
	entityID, err := context.ColorAttachment("entity_id")
	if err != nil {
		return err
	}
	entityID.LoadOp = wgpu.LoadOpLoad
	depth, err := context.DepthAttachment("depth")
	if err != nil {
		return err
	}
	depth.DepthLoadOp = wgpu.LoadOpLoad
	depth.DepthStoreOp = wgpu.StoreOpStore
	depth.StencilLoadOp = wgpu.LoadOpLoad
	depth.StencilStoreOp = wgpu.StoreOpStore

	prepassDepth, err := context.DepthAttachment("depth")
	if err != nil {
		return err
	}
	prepassDepth.DepthLoadOp = wgpu.LoadOpLoad
	prepassDepth.DepthStoreOp = wgpu.StoreOpStore
	prepassDepth.StencilLoadOp = wgpu.LoadOpLoad
	prepassDepth.StencilStoreOp = wgpu.StoreOpStore

	prepass := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		Label:                  "skinned_mesh_oit blend-opaque prepass",
		DepthStencilAttachment: &prepassDepth,
	})
	prepass.SetPipeline(state.prepassPipeline)
	prepass.SetBindGroup(0, meshState.viewProjBindGroup, nil)
	prepass.SetBindGroup(1, meshState.globalBindGroup, nil)
	prepass.SetBindGroup(2, state.handleBindGroup, nil)
	prepass.SetBindGroup(3, meshState.iblBindGroup, nil)
	for _, group := range groups {
		entry, ok := assets.Lookup(group.Mesh)
		if !ok || entry == nil {
			continue
		}
		prepass.SetVertexBuffer(0, entry.Vertices, 0, wgpu.WholeSize)
		prepass.Draw(entry.VertexCount, group.Count, 0, group.FirstInstance)
	}
	prepass.End()
	prepass.Release()

	passEnc := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		Label:                  "skinned_mesh_oit",
		ColorAttachments:       []wgpu.RenderPassColorAttachment{accum, reveal, entityID},
		DepthStencilAttachment: &depth,
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

func skinnedMeshOitRelease(state *skinnedMeshOitState) {
	if state.handleBindGroup != nil {
		state.handleBindGroup.Release()
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
}
