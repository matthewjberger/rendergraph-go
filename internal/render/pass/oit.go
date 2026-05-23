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

//go:embed mesh_oit.wgsl
var meshOitShader string

//go:embed oit_composite.wgsl
var oitCompositeShader string

type oitViewProjUniform struct {
	ViewProj       mgl32.Mat4
	CameraPosition mgl32.Vec4
	CameraZFar     float32
	OitZScale      float32
	Pad0           float32
	Pad1           float32
}

type oitDirectionalUniform struct {
	Direction mgl32.Vec4
	Color     mgl32.Vec4
	Ambient   mgl32.Vec4
}

type oitMeshPassState struct {
	pipeline       *wgpu.RenderPipeline
	globalLayout   *wgpu.BindGroupLayout
	globalGroup    *wgpu.BindGroup
	viewProjBuffer *wgpu.Buffer
	directionalBuf *wgpu.Buffer
	aspectFn       func() float32
}

func AddOitMeshPass(renderer *render.Renderer) (*render.Pass, error) {
	state, err := newOitMeshState(renderer.Device, renderer.AspectRatio)
	if err != nil {
		return nil, err
	}
	pass := &render.Pass{
		Name:                 "oit_mesh",
		Reads:                []string{"depth"},
		Writes:               []string{"oit_accum", "oit_reveal", "entity_id"},
		Prepare:              func(c *render.PassContext) error { return oitMeshPrepare(state, c) },
		Execute:              func(c *render.PassContext) error { return oitMeshExecute(state, c) },
		InvalidateBindGroups: func() { oitMeshInvalidate(state) },
		Release:              func() { oitMeshRelease(state) },
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

func newOitMeshState(device *wgpu.Device, aspect func() float32) (*oitMeshPassState, error) {
	module, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "mesh_oit shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: meshOitShader},
	})
	if err != nil {
		return nil, fmt.Errorf("oit_mesh: shader: %w", err)
	}
	defer module.Release()

	globalLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "oit_mesh global layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageVertex | wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform}},
			{Binding: 1, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform}},
			{Binding: 2, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 3, Visibility: wgpu.ShaderStageFragment, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeFloat, ViewDimension: wgpu.TextureViewDimension2DArray}},
			{Binding: 4, Visibility: wgpu.ShaderStageFragment, Sampler: wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeFiltering}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("oit_mesh: global layout: %w", err)
	}

	handleLayout, err := createHandleBgLayout(device)
	if err != nil {
		globalLayout.Release()
		return nil, fmt.Errorf("oit_mesh: handle layout: %w", err)
	}
	defer handleLayout.Release()

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "oit_mesh pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{globalLayout, handleLayout},
	})
	if err != nil {
		globalLayout.Release()
		return nil, fmt.Errorf("oit_mesh: pipeline layout: %w", err)
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
		Label:  "oit_mesh pipeline",
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
		Primitive: wgpu.PrimitiveState{
			Topology:  wgpu.PrimitiveTopologyTriangleList,
			FrontFace: wgpu.FrontFaceCCW,
			CullMode:  wgpu.CullModeNone,
		},
		DepthStencil: &wgpu.DepthStencilState{
			Format:            render.DepthFormat,
			DepthWriteEnabled: false,

			DepthCompare: wgpu.CompareFunctionLessEqual,
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
				{Format: render.HdrFormat, Blend: &accumBlend, WriteMask: wgpu.ColorWriteMaskAll},
				{Format: wgpu.TextureFormatR8Unorm, Blend: &revealBlend, WriteMask: wgpu.ColorWriteMaskRed},

				{Format: render.EntityIdFormat, WriteMask: wgpu.ColorWriteMaskRed},
			},
		},
	})
	if err != nil {
		globalLayout.Release()
		return nil, fmt.Errorf("oit_mesh: pipeline: %w", err)
	}

	viewProjBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "oit_mesh view_proj",
		Size:  uint64(unsafe.Sizeof(oitViewProjUniform{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		pipeline.Release()
		globalLayout.Release()
		return nil, fmt.Errorf("oit_mesh: view_proj buffer: %w", err)
	}
	directionalBuf, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "oit_mesh directional",
		Size:  uint64(unsafe.Sizeof(oitDirectionalUniform{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		viewProjBuffer.Release()
		pipeline.Release()
		globalLayout.Release()
		return nil, fmt.Errorf("oit_mesh: directional buffer: %w", err)
	}

	return &oitMeshPassState{
		pipeline:       pipeline,
		globalLayout:   globalLayout,
		viewProjBuffer: viewProjBuffer,
		directionalBuf: directionalBuf,
		aspectFn:       aspect,
	}, nil
}

func oitMeshPrepare(state *oitMeshPassState, context *render.PassContext) error {

	camera, hasCamera := ecs.Resource[render.Camera](context.World)
	viewProj := mgl32.Ident4()
	cameraPos := mgl32.Vec4{0, 0, 0, 1}
	zFar := float32(1000.0)
	if hasCamera && camera != nil {
		viewProj = render.CameraViewProjection(camera, state.aspectFn())
		cameraPos = mgl32.Vec4{camera.Eye[0], camera.Eye[1], camera.Eye[2], 1}
		zFar = camera.Far
	}
	uniform := oitViewProjUniform{
		ViewProj:       viewProj,
		CameraPosition: cameraPos,
		CameraZFar:     zFar,
		OitZScale:      zFar * 0.2,
	}
	writeBuffer(context.Device, context.Queue, context.Encoder, state.viewProjBuffer, 0, bytesOf(&uniform))

	directional := oitDirectionalUniform{
		Direction: mgl32.Vec4{0, -1, 0, 0},
		Color:     mgl32.Vec4{1, 1, 1, 1},
		Ambient:   mgl32.Vec4{0.18, 0.20, 0.25, 1},
	}
	lightMask := ecs.MustMaskOf[render.Light](context.World)
	context.World.ForEach(lightMask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		light, ok := ecs.Get[render.Light](context.World, entity)
		if !ok || light.Type != render.LightTypeDirectional {
			return
		}
		intensity := light.Intensity
		if intensity <= 0 {
			intensity = 1
		}
		directional.Color = mgl32.Vec4{
			light.Color[0] * intensity,
			light.Color[1] * intensity,
			light.Color[2] * intensity,
			1,
		}
		if global, ok := ecs.Get[transform.GlobalTransform](context.World, entity); ok {
			forward := mgl32.Vec3{-global.Matrix[8], -global.Matrix[9], -global.Matrix[10]}
			if forward.Len() > 1e-4 {
				forward = forward.Normalize()
				directional.Direction = mgl32.Vec4{forward[0], forward[1], forward[2], 0}
			}
		}
	})
	writeBuffer(context.Device, context.Queue, context.Encoder, state.directionalBuf, 0, bytesOf(&directional))

	if state.globalGroup == nil {
		arraysResource, ok := ecs.Resource[asset.MaterialTextureArraysResource](context.World)
		if !ok || arraysResource == nil || arraysResource.Arrays == nil {
			return fmt.Errorf("oit_mesh: MaterialTextureArraysResource missing")
		}
		registryResource, ok := ecs.Resource[asset.MaterialRegistryResource](context.World)
		if !ok || registryResource == nil || registryResource.Registry == nil {
			return fmt.Errorf("oit_mesh: MaterialRegistryResource missing")
		}
		arrays := arraysResource.Arrays
		bg, err := context.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  "oit_mesh global bg",
			Layout: state.globalLayout,
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: state.viewProjBuffer, Offset: 0, Size: uint64(unsafe.Sizeof(oitViewProjUniform{}))},
				{Binding: 1, Buffer: state.directionalBuf, Offset: 0, Size: uint64(unsafe.Sizeof(oitDirectionalUniform{}))},
				{Binding: 2, Buffer: registryResource.Registry.Buffer(), Offset: 0, Size: wgpu.WholeSize},
				{Binding: 3, TextureView: arrays.SRGBView},
				{Binding: 4, Sampler: arrays.Sampler},
			},
		})
		if err != nil {
			return fmt.Errorf("oit_mesh: global bg: %w", err)
		}
		state.globalGroup = bg
	}
	return nil
}

func oitMeshExecute(state *oitMeshPassState, context *render.PassContext) error {
	meshState, ok := findMeshPassState(context.World)
	if !ok || len(meshState.sortedHandles) == 0 {
		return nil
	}
	assets := ecs.MustResource[asset.MeshAssetsResource](context.World).Assets

	accum, err := context.ColorAttachment("oit_accum")
	if err != nil {
		return err
	}
	reveal, err := context.ColorAttachment("oit_reveal")
	if err != nil {
		return err
	}
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
		Label:                  "oit_mesh",
		ColorAttachments:       []wgpu.RenderPassColorAttachment{accum, reveal, entityID},
		DepthStencilAttachment: &depth,
	})
	passEnc.SetPipeline(state.pipeline)
	passEnc.SetBindGroup(0, state.globalGroup, nil)
	for _, handle := range meshState.sortedHandles {
		bucket := meshState.perHandle[handle]
		if bucket == nil || bucket.bindGroup == nil {
			continue
		}
		entry, ok := assets.Lookup(handle)
		if !ok {
			continue
		}
		passEnc.SetBindGroup(1, bucket.bindGroup, nil)
		passEnc.SetVertexBuffer(0, entry.Vertices, 0, wgpu.WholeSize)
		passEnc.Draw(entry.VertexCount, uint32(len(bucket.slotEntity)), 0, 0)
	}
	passEnc.End()
	passEnc.Release()
	return nil
}

func oitMeshInvalidate(state *oitMeshPassState) {
	if state.globalGroup != nil {
		state.globalGroup.Release()
		state.globalGroup = nil
	}
}

func oitMeshRelease(state *oitMeshPassState) {
	if state.globalGroup != nil {
		state.globalGroup.Release()
	}
	if state.directionalBuf != nil {
		state.directionalBuf.Release()
	}
	if state.viewProjBuffer != nil {
		state.viewProjBuffer.Release()
	}
	if state.pipeline != nil {
		state.pipeline.Release()
	}
	if state.globalLayout != nil {
		state.globalLayout.Release()
	}
}

type oitCompositePassState struct {
	pipeline        *wgpu.RenderPipeline
	bindGroupLayout *wgpu.BindGroupLayout
	sampler         *wgpu.Sampler
	bindGroup       *wgpu.BindGroup
	lastAccumView   *wgpu.TextureView
	lastRevealView  *wgpu.TextureView
}

func AddOitCompositePass(renderer *render.Renderer) (*render.Pass, error) {
	state, err := newOitCompositeState(renderer.Device)
	if err != nil {
		return nil, err
	}
	pass := &render.Pass{
		Name:                 "oit_composite",
		Reads:                []string{"oit_accum", "oit_reveal"},
		Writes:               []string{"scene_color"},
		Prepare:              func(c *render.PassContext) error { return oitCompositePrepare(state, c) },
		Execute:              func(c *render.PassContext) error { return oitCompositeExecute(state, c) },
		InvalidateBindGroups: func() { oitCompositeInvalidate(state) },
		Release:              func() { oitCompositeRelease(state) },
	}
	if err := renderer.Graph.AddPass(pass, []render.SlotBinding{
		{Slot: "oit_accum", ResourceID: renderer.OitAccumID},
		{Slot: "oit_reveal", ResourceID: renderer.OitRevealID},
		{Slot: "scene_color", ResourceID: renderer.SceneColorID},
	}); err != nil {
		return nil, err
	}
	return pass, nil
}

func newOitCompositeState(device *wgpu.Device) (*oitCompositePassState, error) {
	module, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "oit_composite shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: oitCompositeShader},
	})
	if err != nil {
		return nil, fmt.Errorf("oit_composite: shader: %w", err)
	}
	defer module.Release()

	layout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "oit_composite layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageFragment, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeFloat, ViewDimension: wgpu.TextureViewDimension2D}},
			{Binding: 1, Visibility: wgpu.ShaderStageFragment, Sampler: wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeFiltering}},
			{Binding: 2, Visibility: wgpu.ShaderStageFragment, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeFloat, ViewDimension: wgpu.TextureViewDimension2D}},
			{Binding: 3, Visibility: wgpu.ShaderStageFragment, Sampler: wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeFiltering}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("oit_composite: bind group layout: %w", err)
	}

	sampler, err := device.CreateSampler(&wgpu.SamplerDescriptor{
		Label:         "oit_composite sampler",
		AddressModeU:  wgpu.AddressModeClampToEdge,
		AddressModeV:  wgpu.AddressModeClampToEdge,
		AddressModeW:  wgpu.AddressModeClampToEdge,
		MagFilter:     wgpu.FilterModeLinear,
		MinFilter:     wgpu.FilterModeLinear,
		MipmapFilter:  wgpu.MipmapFilterModeNearest,
		MaxAnisotropy: 1,
	})
	if err != nil {
		layout.Release()
		return nil, fmt.Errorf("oit_composite: sampler: %w", err)
	}

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "oit_composite pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{layout},
	})
	if err != nil {
		sampler.Release()
		layout.Release()
		return nil, fmt.Errorf("oit_composite: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	blend := wgpu.BlendState{
		Color: wgpu.BlendComponent{SrcFactor: wgpu.BlendFactorSrcAlpha, DstFactor: wgpu.BlendFactorOneMinusSrcAlpha, Operation: wgpu.BlendOperationAdd},
		Alpha: wgpu.BlendComponent{SrcFactor: wgpu.BlendFactorOne, DstFactor: wgpu.BlendFactorOneMinusSrcAlpha, Operation: wgpu.BlendOperationAdd},
	}
	pipeline, err := device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "oit_composite pipeline",
		Layout: pipelineLayout,
		Vertex: wgpu.VertexState{Module: module, EntryPoint: "vertex_main"},
		Primitive: wgpu.PrimitiveState{
			Topology:  wgpu.PrimitiveTopologyTriangleList,
			FrontFace: wgpu.FrontFaceCCW,
			CullMode:  wgpu.CullModeNone,
		},
		Multisample: wgpu.MultisampleState{Count: 1, Mask: 0xFFFFFFFF},
		Fragment: &wgpu.FragmentState{
			Module:     module,
			EntryPoint: "fragment_main",
			Targets: []wgpu.ColorTargetState{{
				Format:    render.HdrFormat,
				Blend:     &blend,
				WriteMask: wgpu.ColorWriteMaskAll,
			}},
		},
	})
	if err != nil {
		sampler.Release()
		layout.Release()
		return nil, fmt.Errorf("oit_composite: pipeline: %w", err)
	}

	return &oitCompositePassState{
		pipeline:        pipeline,
		bindGroupLayout: layout,
		sampler:         sampler,
	}, nil
}

func oitCompositePrepare(state *oitCompositePassState, context *render.PassContext) error {
	accumView, err := context.TextureView("oit_accum")
	if err != nil {
		return err
	}
	revealView, err := context.TextureView("oit_reveal")
	if err != nil {
		return err
	}
	if state.bindGroup != nil && state.lastAccumView == accumView && state.lastRevealView == revealView {
		return nil
	}
	if state.bindGroup != nil {
		state.bindGroup.Release()
	}
	bg, err := context.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "oit_composite bg",
		Layout: state.bindGroupLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, TextureView: accumView},
			{Binding: 1, Sampler: state.sampler},
			{Binding: 2, TextureView: revealView},
			{Binding: 3, Sampler: state.sampler},
		},
	})
	if err != nil {
		return fmt.Errorf("oit_composite: bind group: %w", err)
	}
	state.bindGroup = bg
	state.lastAccumView = accumView
	state.lastRevealView = revealView
	return nil
}

func oitCompositeExecute(state *oitCompositePassState, context *render.PassContext) error {
	color, err := context.ColorAttachment("scene_color")
	if err != nil {
		return err
	}
	color.LoadOp = wgpu.LoadOpLoad
	passEnc := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		Label:            "oit_composite",
		ColorAttachments: []wgpu.RenderPassColorAttachment{color},
	})
	passEnc.SetPipeline(state.pipeline)
	passEnc.SetBindGroup(0, state.bindGroup, nil)
	passEnc.Draw(3, 1, 0, 0)
	passEnc.End()
	passEnc.Release()
	return nil
}

func oitCompositeInvalidate(state *oitCompositePassState) {
	if state.bindGroup != nil {
		state.bindGroup.Release()
		state.bindGroup = nil
	}
	state.lastAccumView = nil
	state.lastRevealView = nil
}

func oitCompositeRelease(state *oitCompositePassState) {
	if state.bindGroup != nil {
		state.bindGroup.Release()
	}
	if state.sampler != nil {
		state.sampler.Release()
	}
	if state.bindGroupLayout != nil {
		state.bindGroupLayout.Release()
	}
	if state.pipeline != nil {
		state.pipeline.Release()
	}
}
