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
)

//go:embed skinned_mesh_oit.wgsl
var skinnedMeshOitShader string

type skinnedOitViewProj struct {
	ViewProj mgl32.Mat4
	ZScale   float32
	Pad0     float32
	Pad1     float32
	Pad2     float32
}

type skinnedMeshOitState struct {
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

func AddSkinnedMeshOitPass(renderer *render.Renderer) (*render.Pass, error) {
	state, err := newSkinnedMeshOitState(renderer.Device, renderer.AspectRatio)
	if err != nil {
		return nil, err
	}
	pass := &render.Pass{
		Name:    "skinned_mesh_oit",
		Reads:   []string{"depth"},
		Writes:  []string{"oit_accum", "oit_reveal", "entity_id"},
		State:   state,
		Prepare: skinnedMeshOitPrepare,
		Execute: skinnedMeshOitExecute,
		Release: skinnedMeshOitRelease,
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

func newSkinnedMeshOitState(device *wgpu.Device, aspect func() float32) (*skinnedMeshOitState, error) {
	module, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "skinned_mesh_oit shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: skinnedMeshOitShader},
	})
	if err != nil {
		return nil, fmt.Errorf("skinned_mesh_oit: shader: %w", err)
	}
	defer module.Release()

	viewProjLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "skinned_mesh_oit view_proj layout",
		Entries: []wgpu.BindGroupLayoutEntry{{
			Binding:    0,
			Visibility: wgpu.ShaderStageVertex | wgpu.ShaderStageFragment,
			Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform},
		}},
	})
	if err != nil {
		return nil, fmt.Errorf("skinned_mesh_oit: view_proj layout: %w", err)
	}
	globalLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "skinned_mesh_oit global layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform}},
			{Binding: 1, Visibility: wgpu.ShaderStageFragment, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeFloat, ViewDimension: wgpu.TextureViewDimension2DArray}},
			{Binding: 2, Visibility: wgpu.ShaderStageFragment, Sampler: wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeFiltering}},
		},
	})
	if err != nil {
		viewProjLayout.Release()
		return nil, fmt.Errorf("skinned_mesh_oit: global layout: %w", err)
	}
	handleLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "skinned_mesh_oit handle layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageVertex | wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
		},
	})
	if err != nil {
		viewProjLayout.Release()
		globalLayout.Release()
		return nil, fmt.Errorf("skinned_mesh_oit: handle layout: %w", err)
	}

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "skinned_mesh_oit pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{viewProjLayout, globalLayout, handleLayout},
	})
	if err != nil {
		viewProjLayout.Release()
		globalLayout.Release()
		handleLayout.Release()
		return nil, fmt.Errorf("skinned_mesh_oit: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

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
				{Format: render.HdrFormat, Blend: &accumBlend, WriteMask: wgpu.ColorWriteMaskAll},
				{Format: wgpu.TextureFormatR8Unorm, Blend: &revealBlend, WriteMask: wgpu.ColorWriteMaskRed},
				{Format: render.EntityIdFormat, WriteMask: wgpu.ColorWriteMaskRed},
			},
		},
	})
	if err != nil {
		viewProjLayout.Release()
		globalLayout.Release()
		handleLayout.Release()
		return nil, fmt.Errorf("skinned_mesh_oit: pipeline: %w", err)
	}

	viewProjBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "skinned_mesh_oit view_proj",
		Size:  uint64(unsafe.Sizeof(skinnedOitViewProj{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		pipeline.Release()
		viewProjLayout.Release()
		globalLayout.Release()
		handleLayout.Release()
		return nil, fmt.Errorf("skinned_mesh_oit: view_proj buffer: %w", err)
	}
	viewProjGroup, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "skinned_mesh_oit view_proj bg",
		Layout: viewProjLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: viewProjBuffer, Offset: 0, Size: uint64(unsafe.Sizeof(skinnedOitViewProj{}))},
		},
	})
	if err != nil {
		viewProjBuffer.Release()
		pipeline.Release()
		viewProjLayout.Release()
		globalLayout.Release()
		handleLayout.Release()
		return nil, fmt.Errorf("skinned_mesh_oit: view_proj bg: %w", err)
	}

	uniformBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "skinned_mesh_oit light uniform",
		Size:  uint64(unsafe.Sizeof(skinnedUniforms{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		viewProjGroup.Release()
		viewProjBuffer.Release()
		pipeline.Release()
		viewProjLayout.Release()
		globalLayout.Release()
		handleLayout.Release()
		return nil, fmt.Errorf("skinned_mesh_oit: light buffer: %w", err)
	}

	return &skinnedMeshOitState{
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

func skinnedMeshOitPrepare(s any, context *render.PassContext) error {
	state := s.(*skinnedMeshOitState)

	camera, hasCamera := ecs.Resource[render.Camera](context.World)
	viewProj := mgl32.Ident4()
	zFar := float32(1000)
	if hasCamera && camera != nil {
		viewProj = render.CameraViewProjection(camera, state.aspectFn())
		zFar = camera.Far
	}
	vp := skinnedOitViewProj{ViewProj: viewProj, ZScale: zFar * 0.2}
	writeBuffer(context.Device, context.Queue, context.Encoder, state.viewProjBuffer, 0, bytesOf(&vp))

	light := defaultSkinnedUniforms()
	applyDirectionalToSkinned(context.World, &light)
	writeBuffer(context.Device, context.Queue, context.Encoder, state.uniformBuffer, 0, bytesOf(&light))

	if state.globalGroup == nil {
		arraysResource, ok := ecs.Resource[asset.MaterialTextureArraysResource](context.World)
		if !ok || arraysResource == nil || arraysResource.Arrays == nil {
			return fmt.Errorf("skinned_mesh_oit: MaterialTextureArraysResource missing")
		}
		arrays := arraysResource.Arrays
		bg, err := context.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  "skinned_mesh_oit global bg",
			Layout: state.globalLayout,
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: state.uniformBuffer, Offset: 0, Size: uint64(unsafe.Sizeof(skinnedUniforms{}))},
				{Binding: 1, TextureView: arrays.SRGBView},
				{Binding: 2, Sampler: arrays.Sampler},
			},
		})
		if err != nil {
			return fmt.Errorf("skinned_mesh_oit: global bg: %w", err)
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

func skinnedMeshOitExecute(s any, context *render.PassContext) error {
	state := s.(*skinnedMeshOitState)
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

	passEnc := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		Label:                  "skinned_mesh_oit",
		ColorAttachments:       []wgpu.RenderPassColorAttachment{accum, reveal, entityID},
		DepthStencilAttachment: &depth,
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

func skinnedMeshOitRelease(s any) {
	state := s.(*skinnedMeshOitState)
	if state.handleBindGroup != nil {
		state.handleBindGroup.Release()
	}
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
