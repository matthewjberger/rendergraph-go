package pass

import (
	_ "embed"
	"fmt"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"

	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/render/asset"
)

//go:embed mesh_oit.wgsl
var meshOitShader string

//go:embed oit_composite.wgsl
var oitCompositeShader string

// oitMeshPassState renders blend and transmissive meshes through weighted
// blended OIT. It reuses the forward mesh pass's bind group layouts and bind
// groups verbatim so the lighting (clustered lights, shadows, IBL, screen-space
// transmission) is identical to the opaque pass.
type oitMeshPassState struct {
	pipeline        *wgpu.RenderPipeline
	prepassPipeline *wgpu.RenderPipeline
}

func AddOitMeshPass(renderer *render.Renderer) (*render.Pass, error) {
	state, err := newOitMeshState(renderer.Device)
	if err != nil {
		return nil, err
	}
	pass := &render.Pass{
		Name:    "oit_mesh",
		Reads:   []string{"depth"},
		Writes:  []string{"oit_accum", "oit_reveal", "entity_id"},
		Execute: func(c *render.PassContext) error { return oitMeshExecute(state, c) },
		Release: func() { oitMeshRelease(state) },
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

func newOitMeshState(device *wgpu.Device) (*oitMeshPassState, error) {
	meshState, ok := findMeshPassState(nil)
	if !ok {
		return nil, fmt.Errorf("oit_mesh: mesh pass must be created before the OIT mesh pass")
	}

	module, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "mesh_oit shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: meshOitShader},
	})
	if err != nil {
		return nil, fmt.Errorf("oit_mesh: shader: %w", err)
	}
	defer module.Release()

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label: "oit_mesh pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{
			meshState.viewProjLayout,
			meshState.globalBgLayout,
			meshState.handleBgLayout,
			meshState.iblBgLayout,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("oit_mesh: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	vertexBuffers := []wgpu.VertexBufferLayout{{
		ArrayStride: uint64(unsafe.Sizeof(asset.MeshVertex{})),
		StepMode:    wgpu.VertexStepModeVertex,
		Attributes: []wgpu.VertexAttribute{
			{Format: wgpu.VertexFormatFloat32x4, Offset: 0, ShaderLocation: 0},
			{Format: wgpu.VertexFormatFloat32x4, Offset: 16, ShaderLocation: 1},
			{Format: wgpu.VertexFormatFloat32x4, Offset: 32, ShaderLocation: 2},
			{Format: wgpu.VertexFormatFloat32x4, Offset: 48, ShaderLocation: 3},
			{Format: wgpu.VertexFormatFloat32x4, Offset: 64, ShaderLocation: 4},
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
		Label:  "oit_mesh pipeline",
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
		return nil, fmt.Errorf("oit_mesh: pipeline: %w", err)
	}

	prepassPipeline, err := device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "oit_mesh blend-opaque prepass pipeline",
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
		return nil, fmt.Errorf("oit_mesh: blend-opaque prepass pipeline: %w", err)
	}

	return &oitMeshPassState{
		pipeline:        pipeline,
		prepassPipeline: prepassPipeline,
	}, nil
}

func oitMeshExecute(state *oitMeshPassState, context *render.PassContext) error {
	meshState, ok := findMeshPassState(context.World)
	if !ok {
		return nil
	}
	hasGeometry := len(meshState.sortedHandles) > 0 &&
		meshState.globalBindGroup != nil &&
		meshState.iblBindGroup != nil &&
		meshState.viewProjBindGroup != nil
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

	if hasGeometry {
		prepassDepth, err := context.DepthAttachment("depth")
		if err != nil {
			return err
		}
		prepassDepth.DepthLoadOp = wgpu.LoadOpLoad
		prepassDepth.DepthStoreOp = wgpu.StoreOpStore
		prepassDepth.StencilLoadOp = wgpu.LoadOpLoad
		prepassDepth.StencilStoreOp = wgpu.StoreOpStore

		prepass := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
			Label:                  "oit_mesh blend-opaque prepass",
			DepthStencilAttachment: &prepassDepth,
		})
		prepass.SetPipeline(state.prepassPipeline)
		prepass.SetBindGroup(0, meshState.viewProjBindGroup, nil)
		prepass.SetBindGroup(1, meshState.globalBindGroup, nil)
		prepass.SetBindGroup(3, meshState.iblBindGroup, nil)
		for _, handle := range meshState.sortedHandles {
			bucket := meshState.perHandle[handle]
			if bucket == nil || bucket.bindGroup == nil {
				continue
			}
			entry, ok := assets.Lookup(handle)
			if !ok {
				continue
			}
			prepass.SetBindGroup(2, bucket.bindGroup, nil)
			prepass.SetVertexBuffer(0, entry.Vertices, 0, wgpu.WholeSize)
			prepass.Draw(entry.VertexCount, uint32(len(bucket.slotEntity)), 0, 0)
		}
		prepass.End()
		prepass.Release()
	}

	passEnc := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		Label:                  "oit_mesh",
		ColorAttachments:       []wgpu.RenderPassColorAttachment{accum, reveal, entityID},
		DepthStencilAttachment: &depth,
	})
	if hasGeometry {
		passEnc.SetPipeline(state.pipeline)
		passEnc.SetBindGroup(0, meshState.viewProjBindGroup, nil)
		passEnc.SetBindGroup(1, meshState.globalBindGroup, nil)
		passEnc.SetBindGroup(3, meshState.iblBindGroup, nil)
		for _, handle := range meshState.sortedHandles {
			bucket := meshState.perHandle[handle]
			if bucket == nil || bucket.bindGroup == nil {
				continue
			}
			entry, ok := assets.Lookup(handle)
			if !ok {
				continue
			}
			passEnc.SetBindGroup(2, bucket.bindGroup, nil)
			passEnc.SetVertexBuffer(0, entry.Vertices, 0, wgpu.WholeSize)
			passEnc.Draw(entry.VertexCount, uint32(len(bucket.slotEntity)), 0, 0)
		}
	}
	passEnc.End()
	passEnc.Release()
	return nil
}

func oitMeshRelease(state *oitMeshPassState) {
	if state.prepassPipeline != nil {
		state.prepassPipeline.Release()
	}
	if state.pipeline != nil {
		state.pipeline.Release()
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
