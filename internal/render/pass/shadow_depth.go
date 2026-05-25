package pass

import (
	_ "embed"
	"fmt"
	"math"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/go-gl/mathgl/mgl32"

	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/render/asset"
	"github.com/matthewjberger/indigo/transform"
	"github.com/matthewjberger/indigo/window"
)

//go:embed shadow_depth.wgsl
var shadowDepthShader string

//go:embed skinned_shadow_depth.wgsl
var skinnedShadowDepthShader string

const ShadowMapSize uint32 = 2048

const ShadowMapFormat = wgpu.TextureFormatDepth32Float

const NumShadowCascades = 4

var CascadeSplitDistances = [NumShadowCascades]float32{10.0, 40.0, 150.0, 500.0}

type Shadow struct {
	Texture       *wgpu.Texture
	ArrayView     *wgpu.TextureView
	CascadeViews  [NumShadowCascades]*wgpu.TextureView
	Sampler       *wgpu.Sampler
	RawSampler    *wgpu.Sampler
	CascadeVPs    [NumShadowCascades]*wgpu.Buffer
	UniformBuffer *wgpu.Buffer
	LightViewVPs  [NumShadowCascades]mgl32.Mat4
}

type ShadowResource struct {
	Shadow *Shadow
}

type shadowCascadeUniform struct {
	LightViewProj mgl32.Mat4
}

type shadowMeshUniform struct {
	LightViewProjections [NumShadowCascades]mgl32.Mat4
	CascadeSplits        mgl32.Vec4
	LightDirection       mgl32.Vec4
	CascadeTexelWorld    mgl32.Vec4
	LightSize            float32
	Pad0                 float32
	Pad1                 float32
	Pad2                 float32
}

type shadowDepthPassState struct {
	pipeline              *wgpu.RenderPipeline
	viewProjLayout        *wgpu.BindGroupLayout
	handleBgLayout        *wgpu.BindGroupLayout
	cascadeBgs            [NumShadowCascades]*wgpu.BindGroup
	skinnedPipeline       *wgpu.RenderPipeline
	skinnedHandleBgLayout *wgpu.BindGroupLayout
	skinnedJointBindCache map[ecs.Entity]*shadowSkinnedEntry
}

type shadowSkinnedEntry struct {
	offsetBuffer *wgpu.Buffer
	bindGroup    *wgpu.BindGroup
	jointGen     uint64
	jointBuffer  *wgpu.Buffer
	morphBuffer  *wgpu.Buffer
}

// skinnedShadowParams mirrors the WGSL uniform: joint offset plus the morph
// fields needed to deform skinned shadow casters.
type skinnedShadowParams struct {
	JointOffset             uint32
	MorphTargetCount        uint32
	MorphDisplacementOffset uint32
	MorphVertexCount        uint32
	MorphWeights            [8]float32
}

func NewShadow(device *wgpu.Device) (*Shadow, error) {
	shadow := &Shadow{}
	tex, err := device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "shadow cascades",
		Size: wgpu.Extent3D{
			Width:              ShadowMapSize,
			Height:             ShadowMapSize,
			DepthOrArrayLayers: NumShadowCascades,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     wgpu.TextureDimension2D,
		Format:        ShadowMapFormat,
		Usage:         wgpu.TextureUsageRenderAttachment | wgpu.TextureUsageTextureBinding,
	})
	if err != nil {
		return nil, fmt.Errorf("shadow: cascade texture: %w", err)
	}
	shadow.Texture = tex

	arrayView, err := tex.CreateView(&wgpu.TextureViewDescriptor{
		Label:           "shadow cascade array view",
		Dimension:       wgpu.TextureViewDimension2DArray,
		BaseMipLevel:    0,
		MipLevelCount:   1,
		BaseArrayLayer:  0,
		ArrayLayerCount: NumShadowCascades,
		Aspect:          wgpu.TextureAspectAll,
	})
	if err != nil {
		shadow.Release()
		return nil, fmt.Errorf("shadow: array view: %w", err)
	}
	shadow.ArrayView = arrayView

	for index := 0; index < NumShadowCascades; index++ {
		view, err := tex.CreateView(&wgpu.TextureViewDescriptor{
			Label:           "shadow cascade slice",
			Dimension:       wgpu.TextureViewDimension2D,
			BaseMipLevel:    0,
			MipLevelCount:   1,
			BaseArrayLayer:  uint32(index),
			ArrayLayerCount: 1,
			Aspect:          wgpu.TextureAspectAll,
		})
		if err != nil {
			shadow.Release()
			return nil, fmt.Errorf("shadow: cascade %d view: %w", index, err)
		}
		shadow.CascadeViews[index] = view
	}

	sampler, err := device.CreateSampler(&wgpu.SamplerDescriptor{
		Label:         "shadow sampler",
		AddressModeU:  wgpu.AddressModeClampToEdge,
		AddressModeV:  wgpu.AddressModeClampToEdge,
		AddressModeW:  wgpu.AddressModeClampToEdge,
		MagFilter:     wgpu.FilterModeLinear,
		MinFilter:     wgpu.FilterModeLinear,
		MipmapFilter:  wgpu.MipmapFilterModeNearest,
		MaxAnisotropy: 1,
		Compare:       wgpu.CompareFunctionLessEqual,
	})
	if err != nil {
		shadow.Release()
		return nil, fmt.Errorf("shadow: sampler: %w", err)
	}
	rawSampler, err := device.CreateSampler(&wgpu.SamplerDescriptor{
		Label:         "shadow raw sampler",
		AddressModeU:  wgpu.AddressModeClampToEdge,
		AddressModeV:  wgpu.AddressModeClampToEdge,
		AddressModeW:  wgpu.AddressModeClampToEdge,
		MagFilter:     wgpu.FilterModeNearest,
		MinFilter:     wgpu.FilterModeNearest,
		MipmapFilter:  wgpu.MipmapFilterModeNearest,
		MaxAnisotropy: 1,
	})
	if err != nil {
		shadow.Release()
		return nil, fmt.Errorf("shadow: raw sampler: %w", err)
	}
	shadow.RawSampler = rawSampler
	shadow.Sampler = sampler

	for index := 0; index < NumShadowCascades; index++ {
		buffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: "shadow cascade light_vp",
			Size:  uint64(unsafe.Sizeof(shadowCascadeUniform{})),
			Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
		})
		if err != nil {
			shadow.Release()
			return nil, fmt.Errorf("shadow: cascade %d uniform: %w", index, err)
		}
		shadow.CascadeVPs[index] = buffer
		shadow.LightViewVPs[index] = mgl32.Ident4()
	}

	meshUniform, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "shadow mesh uniform",
		Size:  uint64(unsafe.Sizeof(shadowMeshUniform{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		shadow.Release()
		return nil, fmt.Errorf("shadow: mesh uniform: %w", err)
	}
	shadow.UniformBuffer = meshUniform

	return shadow, nil
}

func (s *Shadow) Release() {
	if s.UniformBuffer != nil {
		s.UniformBuffer.Release()
		s.UniformBuffer = nil
	}
	for index := range s.CascadeVPs {
		if s.CascadeVPs[index] != nil {
			s.CascadeVPs[index].Release()
			s.CascadeVPs[index] = nil
		}
	}
	if s.Sampler != nil {
		s.Sampler.Release()
		s.Sampler = nil
	}
	if s.RawSampler != nil {
		s.RawSampler.Release()
		s.RawSampler = nil
	}
	for index := range s.CascadeViews {
		if s.CascadeViews[index] != nil {
			s.CascadeViews[index].Release()
			s.CascadeViews[index] = nil
		}
	}
	if s.ArrayView != nil {
		s.ArrayView.Release()
		s.ArrayView = nil
	}
	if s.Texture != nil {
		s.Texture.Release()
		s.Texture = nil
	}
}

func NewShadowDepthPass(device *wgpu.Device, shadow *Shadow) (*render.Pass, error) {
	state := &shadowDepthPassState{}

	viewProjLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "shadow view_proj bind group layout",
		Entries: []wgpu.BindGroupLayoutEntry{{
			Binding:    0,
			Visibility: wgpu.ShaderStageVertex,
			Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform},
		}},
	})
	if err != nil {
		return nil, fmt.Errorf("shadow_depth: view_proj layout: %w", err)
	}
	state.viewProjLayout = viewProjLayout

	handleBgLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "shadow per-handle bind group layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 2, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("shadow_depth: handle layout: %w", err)
	}
	state.handleBgLayout = handleBgLayout

	for index := 0; index < NumShadowCascades; index++ {
		bg, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  "shadow cascade view_proj bind group",
			Layout: viewProjLayout,
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: shadow.CascadeVPs[index], Offset: 0, Size: uint64(unsafe.Sizeof(shadowCascadeUniform{}))},
			},
		})
		if err != nil {
			return nil, fmt.Errorf("shadow_depth: cascade %d view_proj bind group: %w", index, err)
		}
		state.cascadeBgs[index] = bg
	}

	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "shadow_depth shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shadowDepthShader},
	})
	if err != nil {
		return nil, fmt.Errorf("shadow_depth: shader: %w", err)
	}
	defer shader.Release()

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "shadow_depth pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{viewProjLayout, handleBgLayout},
	})
	if err != nil {
		return nil, fmt.Errorf("shadow_depth: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	pipeline, err := device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "shadow_depth pipeline",
		Layout: pipelineLayout,
		Vertex: wgpu.VertexState{
			Module:     shader,
			EntryPoint: "vertex_main",
			Buffers: []wgpu.VertexBufferLayout{{
				ArrayStride: uint64(unsafe.Sizeof(asset.MeshVertex{})),
				StepMode:    wgpu.VertexStepModeVertex,
				Attributes: []wgpu.VertexAttribute{
					{Format: wgpu.VertexFormatFloat32x4, Offset: 0, ShaderLocation: 0},
				},
			}},
		},
		Fragment: &wgpu.FragmentState{
			Module:     shader,
			EntryPoint: "fragment_main",
		},
		Primitive: wgpu.PrimitiveState{
			Topology:  wgpu.PrimitiveTopologyTriangleList,
			FrontFace: wgpu.FrontFaceCCW,
			CullMode:  wgpu.CullModeNone,
		},
		DepthStencil: &wgpu.DepthStencilState{
			Format:            ShadowMapFormat,
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
		Multisample: wgpu.MultisampleState{
			Count:                  1,
			Mask:                   0xFFFFFFFF,
			AlphaToCoverageEnabled: false,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("shadow_depth: pipeline: %w", err)
	}
	state.pipeline = pipeline

	skinnedHandleBgLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "shadow per-handle skinned bind group layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform}},
			{Binding: 2, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("shadow_depth: skinned handle layout: %w", err)
	}
	state.skinnedHandleBgLayout = skinnedHandleBgLayout
	state.skinnedJointBindCache = make(map[ecs.Entity]*shadowSkinnedEntry)

	skinnedShader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "skinned_shadow_depth shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: skinnedShadowDepthShader},
	})
	if err != nil {
		return nil, fmt.Errorf("skinned_shadow_depth: shader: %w", err)
	}
	defer skinnedShader.Release()

	skinnedPipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "skinned_shadow_depth pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{viewProjLayout, skinnedHandleBgLayout},
	})
	if err != nil {
		return nil, fmt.Errorf("skinned_shadow_depth: pipeline layout: %w", err)
	}
	defer skinnedPipelineLayout.Release()

	skinnedPipeline, err := device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "skinned_shadow_depth pipeline",
		Layout: skinnedPipelineLayout,
		Vertex: wgpu.VertexState{
			Module:     skinnedShader,
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
			CullMode:  wgpu.CullModeBack,
		},
		DepthStencil: &wgpu.DepthStencilState{
			Format:            ShadowMapFormat,
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
	})
	if err != nil {
		return nil, fmt.Errorf("skinned_shadow_depth: pipeline: %w", err)
	}
	state.skinnedPipeline = skinnedPipeline

	return &render.Pass{
		Name:    "shadow_depth",
		Prepare: func(c *render.PassContext) error { return shadowDepthPrepare(state, c) },
		Execute: func(c *render.PassContext) error { return shadowDepthExecute(state, c) },
		Release: func() { shadowDepthRelease(state) },
	}, nil
}

func shadowDepthPrepare(s any, context *render.PassContext) error {
	_ = s

	lightDir := mgl32.Vec3{-0.3, -1.0, -0.4}.Normalize()
	lightSize := float32(1.0)
	lightMask := ecs.MustMaskOf[render.Light](context.World)
	context.World.ForEach(lightMask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		light, ok := ecs.Get[render.Light](context.World, entity)
		if !ok || light.Type != render.LightTypeDirectional {
			return
		}
		if light.LightSize > 0 {
			lightSize = light.LightSize
		}
		if global, ok := ecs.Get[transform.GlobalTransform](context.World, entity); ok {
			forward := mgl32.Vec3{-global.Matrix[8], -global.Matrix[9], -global.Matrix[10]}
			if forward.Len() > 1e-4 {
				lightDir = forward.Normalize()
			}
		}
	})

	camera, hasCamera := ecs.Resource[render.Camera](context.World)
	aspect := float32(1.0)
	if resource, hasViewport := ecs.Resource[window.Window](context.World); hasViewport {
		if resource.Viewport.Height > 0 {
			aspect = float32(resource.Viewport.Width) / float32(resource.Viewport.Height)
		}
	}

	cameraNear := float32(0.1)
	if hasCamera && camera != nil {
		cameraNear = camera.Near
	}
	splits := scaledCascadeSplits(cameraNear)

	shadow := ecs.MustResource[ShadowResource](context.World).Shadow

	var meshUniform shadowMeshUniform
	for index := 0; index < NumShadowCascades; index++ {
		cascadeNear := cameraNear
		if index > 0 {
			cascadeNear = splits[index-1]
		}
		cascadeFar := splits[index]
		corners := cameraFrustumCornersWorldRange(camera, hasCamera, aspect, cascadeNear, cascadeFar)
		lightVP, texelWorld := fitLightFrustum(corners, lightDir)
		shadow.LightViewVPs[index] = lightVP
		meshUniform.LightViewProjections[index] = lightVP
		meshUniform.CascadeTexelWorld[index] = texelWorld

		cascadeUniform := shadowCascadeUniform{LightViewProj: lightVP}
		writeBuffer(context.Device, context.Queue, context.Encoder, shadow.CascadeVPs[index], 0, bytesOf(&cascadeUniform))
	}
	meshUniform.CascadeSplits = mgl32.Vec4{splits[0], splits[1], splits[2], splits[3]}
	meshUniform.LightDirection = mgl32.Vec4{lightDir.X(), lightDir.Y(), lightDir.Z(), 0}
	meshUniform.LightSize = lightSize
	writeBuffer(context.Device, context.Queue, context.Encoder, shadow.UniformBuffer, 0, bytesOf(&meshUniform))
	return nil
}

func scaledCascadeSplits(_ float32) [NumShadowCascades]float32 {
	return CascadeSplitDistances
}

func fitLightFrustum(corners [8]mgl32.Vec3, lightDir mgl32.Vec3) (mgl32.Mat4, float32) {
	center := mgl32.Vec3{0, 0, 0}
	for _, corner := range corners {
		center = center.Add(corner)
	}
	center = center.Mul(1.0 / 8.0)
	maxRadius := float32(0)
	for _, corner := range corners {
		d := corner.Sub(center).Len()
		if d > maxRadius {
			maxRadius = d
		}
	}
	if maxRadius < 1.0 {
		maxRadius = 1.0
	}
	up := mgl32.Vec3{0, 1, 0}
	if mgl32.Abs(lightDir.Y()) > 0.99 {
		up = mgl32.Vec3{1, 0, 0}
	}
	eye := center.Sub(lightDir.Mul(maxRadius * 4.0))
	lightView := mgl32.LookAtV(eye, center, up)
	minX, maxX := float32(math.MaxFloat32), float32(-math.MaxFloat32)
	minY, maxY := minX, maxX
	minZ, maxZ := minX, maxX
	for _, corner := range corners {
		light4 := lightView.Mul4x1(mgl32.Vec4{corner.X(), corner.Y(), corner.Z(), 1.0})
		if light4.X() < minX {
			minX = light4.X()
		}
		if light4.X() > maxX {
			maxX = light4.X()
		}
		if light4.Y() < minY {
			minY = light4.Y()
		}
		if light4.Y() > maxY {
			maxY = light4.Y()
		}
		if light4.Z() < minZ {
			minZ = light4.Z()
		}
		if light4.Z() > maxZ {
			maxZ = light4.Z()
		}
	}
	pad := (maxX - minX)
	if (maxY - minY) > pad {
		pad = maxY - minY
	}
	pad *= 0.1
	minX -= pad
	maxX += pad
	minY -= pad
	maxY += pad
	extent := maxX - minX
	if (maxY - minY) > extent {
		extent = maxY - minY
	}
	zPad := extent * 0.5
	if zPad < 10.0 {
		zPad = 10.0
	}
	minZ -= zPad
	maxZ += zPad
	lightProj := orthoZO(minX, maxX, minY, maxY, -maxZ, -minZ)
	largest := maxX - minX
	if (maxY - minY) > largest {
		largest = maxY - minY
	}
	texelWorld := largest / float32(ShadowMapSize)
	return lightProj.Mul4(lightView), texelWorld
}

func shadowDepthExecute(state *shadowDepthPassState, context *render.PassContext) error {
	shadow := ecs.MustResource[ShadowResource](context.World).Shadow

	meshState, ok := findMeshPassState(context.World)
	if !ok {
		return nil
	}
	assets := ecs.MustResource[asset.MeshAssetsResource](context.World).Assets

	skinningRes, hasSkinning := ecs.Resource[SkinningComputeResource](context.World)
	var skinnedAssets *asset.SkinnedMeshAssets
	skinnedReady := false
	if skinnedAssetsResource, ok := ecs.Resource[asset.SkinnedMeshAssetsResource](context.World); ok && skinnedAssetsResource != nil && skinnedAssetsResource.Assets != nil && hasSkinning && skinningRes != nil && skinningRes.Compute != nil && skinningRes.Compute.JointMatrixBuffer() != nil {
		skinnedAssets = skinnedAssetsResource.Assets
		skinning := skinningRes.Compute
		jointBuffer := skinning.JointMatrixBuffer()
		generation := skinning.Generation()
		skinnedReady = true
		skinnedMask := ecs.MustMaskOf[asset.SkinnedMesh](context.World)
		context.World.ForEach(skinnedMask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
			skinned, _ := ecs.Get[asset.SkinnedMesh](context.World, entity)
			if skinned == nil || skinned.Skin == nil {
				return
			}
			entry := state.skinnedJointBindCache[entity]
			if entry == nil {
				offsetBuf, err := context.Device.CreateBuffer(&wgpu.BufferDescriptor{
					Label: "shadow skinned joint offset",
					Size:  uint64(unsafe.Sizeof(skinnedShadowParams{})),
					Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
				})
				if err != nil {
					return
				}
				entry = &shadowSkinnedEntry{offsetBuffer: offsetBuf}
				state.skinnedJointBindCache[entity] = entry
			}
			params := skinnedShadowParams{JointOffset: skinning.JointOffset(entity)}
			morphBuffer := skinnedMorphBuffer(context.World, context.Device)
			if skinnedAssets != nil {
				offset, targetCount := skinnedAssets.MorphInfo(skinned.Mesh)
				if targetCount > 0 {
					params.MorphTargetCount = targetCount
					params.MorphDisplacementOffset = offset
					if meshEntry, ok := skinnedAssets.Lookup(skinned.Mesh); ok {
						params.MorphVertexCount = meshEntry.VertexCount
					}
					if weights, ok := ecs.Get[asset.MorphWeights](context.World, entity); ok && weights != nil {
						for i := 0; i < len(weights.Weights) && i < 8; i++ {
							params.MorphWeights[i] = weights.Weights[i]
						}
					}
				}
			}
			writeBuffer(context.Device, context.Queue, context.Encoder, entry.offsetBuffer, 0, bytesOf(&params))
			if entry.bindGroup != nil && (entry.jointGen != generation || entry.jointBuffer != jointBuffer || entry.morphBuffer != morphBuffer) {
				entry.bindGroup.Release()
				entry.bindGroup = nil
			}
			if entry.bindGroup == nil && morphBuffer != nil {
				bg, err := context.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
					Label:  "shadow skinned handle bg",
					Layout: state.skinnedHandleBgLayout,
					Entries: []wgpu.BindGroupEntry{
						{Binding: 0, Buffer: jointBuffer, Offset: 0, Size: wgpu.WholeSize},
						{Binding: 1, Buffer: entry.offsetBuffer, Offset: 0, Size: uint64(unsafe.Sizeof(skinnedShadowParams{}))},
						{Binding: 2, Buffer: morphBuffer, Offset: 0, Size: wgpu.WholeSize},
					},
				})
				if err != nil {
					return
				}
				entry.bindGroup = bg
				entry.jointGen = generation
				entry.jointBuffer = jointBuffer
				entry.morphBuffer = morphBuffer
			}
		})
	}

	for cascade := 0; cascade < NumShadowCascades; cascade++ {
		depthAttachment := wgpu.RenderPassDepthStencilAttachment{
			View:              shadow.CascadeViews[cascade],
			DepthLoadOp:       wgpu.LoadOpClear,
			DepthStoreOp:      wgpu.StoreOpStore,
			DepthClearValue:   1.0,
			DepthReadOnly:     false,
			StencilLoadOp:     wgpu.LoadOpClear,
			StencilStoreOp:    wgpu.StoreOpStore,
			StencilClearValue: 0,
			StencilReadOnly:   true,
		}
		pass := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
			Label:                  "shadow_depth cascade",
			DepthStencilAttachment: &depthAttachment,
		})
		pass.SetPipeline(state.pipeline)
		pass.SetBindGroup(0, state.cascadeBgs[cascade], nil)
		for _, handle := range meshState.sortedHandles {
			bucket := meshState.perHandle[handle]
			entry, ok := assets.Lookup(handle)
			if !ok {
				continue
			}
			shadowBg, err := ensureShadowHandleBindGroup(bucket, context.Device, state.handleBgLayout, shadowMorphBuffer(context.World, assets, handle))
			if err != nil {
				pass.End()
				pass.Release()
				return err
			}
			pass.SetBindGroup(1, shadowBg, nil)
			pass.SetVertexBuffer(0, entry.Vertices, 0, wgpu.WholeSize)
			pass.Draw(entry.VertexCount, uint32(len(bucket.slotEntity)), 0, 0)
		}

		if skinnedReady {
			pass.SetPipeline(state.skinnedPipeline)
			pass.SetBindGroup(0, state.cascadeBgs[cascade], nil)
			skinnedMask := ecs.MustMaskOf[asset.SkinnedMesh](context.World)
			context.World.ForEach(skinnedMask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
				skinned, _ := ecs.Get[asset.SkinnedMesh](context.World, entity)
				if skinned == nil || skinned.Skin == nil {
					return
				}
				meshEntry, ok := skinnedAssets.Lookup(skinned.Mesh)
				if !ok || meshEntry == nil {
					return
				}
				entry := state.skinnedJointBindCache[entity]
				if entry == nil || entry.bindGroup == nil {
					return
				}
				pass.SetBindGroup(1, entry.bindGroup, nil)
				pass.SetVertexBuffer(0, meshEntry.Vertices, 0, wgpu.WholeSize)
				pass.Draw(meshEntry.VertexCount, 1, 0, 0)
			})
		}

		pass.End()
		pass.Release()
	}
	return nil
}

func shadowDepthRelease(state *shadowDepthPassState) {
	for index := range state.cascadeBgs {
		if state.cascadeBgs[index] != nil {
			state.cascadeBgs[index].Release()
			state.cascadeBgs[index] = nil
		}
	}
	if state.pipeline != nil {
		state.pipeline.Release()
	}
	if state.skinnedPipeline != nil {
		state.skinnedPipeline.Release()
	}
	for _, entry := range state.skinnedJointBindCache {
		if entry == nil {
			continue
		}
		if entry.bindGroup != nil {
			entry.bindGroup.Release()
		}
		if entry.offsetBuffer != nil {
			entry.offsetBuffer.Release()
		}
	}
	state.skinnedJointBindCache = nil
	if state.handleBgLayout != nil {
		state.handleBgLayout.Release()
	}
	if state.skinnedHandleBgLayout != nil {
		state.skinnedHandleBgLayout.Release()
	}
	if state.viewProjLayout != nil {
		state.viewProjLayout.Release()
	}
}

func ensureShadowHandleBindGroup(bucket *handleInstances, device *wgpu.Device, layout *wgpu.BindGroupLayout, morphDisplacement *wgpu.Buffer) (*wgpu.BindGroup, error) {
	if bucket.shadowBindGroup != nil {
		return bucket.shadowBindGroup, nil
	}
	if bucket.morphInstanceBuffer == nil || morphDisplacement == nil {
		return nil, nil
	}
	bg, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "shadow per-handle bind group",
		Layout: layout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: bucket.modelBuffer, Offset: 0, Size: wgpu.WholeSize},
			{Binding: 1, Buffer: morphDisplacement, Offset: 0, Size: wgpu.WholeSize},
			{Binding: 2, Buffer: bucket.morphInstanceBuffer, Offset: 0, Size: wgpu.WholeSize},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("shadow_depth: per-handle bind group: %w", err)
	}
	bucket.shadowBindGroup = bg
	return bg, nil
}

// shadowMorphBuffer returns the morph displacement buffer for a static mesh, or
// the mesh pass's dummy when the mesh has none.
func shadowMorphBuffer(world *ecs.World, assets *asset.MeshAssets, handle asset.MeshHandle) *wgpu.Buffer {
	if buffer, _ := assets.MorphInfo(handle); buffer != nil {
		return buffer
	}
	if state, ok := findMeshPassState(world); ok {
		return state.dummyMorphBuffer
	}
	return nil
}

func AddShadowDepthPass(renderer *render.Renderer, shadow *Shadow) (*render.Pass, error) {
	pass, err := NewShadowDepthPass(renderer.Device, shadow)
	if err != nil {
		return nil, err
	}
	if err := renderer.Graph.AddPass(pass, nil); err != nil {
		return nil, err
	}
	return pass, nil
}

func findMeshPassState(_ *ecs.World) (*meshPassState, bool) {
	state, ok := sharedMeshPassState.Load().(*meshPassState)
	return state, ok && state != nil
}

func orthoZO(left, right, bottom, top, near, far float32) mgl32.Mat4 {
	width := right - left
	height := top - bottom
	depth := far - near
	return mgl32.Mat4{
		2.0 / width, 0, 0, 0,
		0, 2.0 / height, 0, 0,
		0, 0, -1.0 / depth, 0,
		-(right + left) / width, -(top + bottom) / height, -near / depth, 1,
	}
}

func cameraFrustumCornersWorldRange(camera *render.Camera, hasCamera bool, aspect, near, far float32) [8]mgl32.Vec3 {
	if !hasCamera || camera == nil {
		var fallback [8]mgl32.Vec3
		extent := far
		if extent < 30.0 {
			extent = 30.0
		}
		index := 0
		for _, sx := range []float32{-1, 1} {
			for _, sy := range []float32{-1, 1} {
				for _, sz := range []float32{-1, 1} {
					fallback[index] = mgl32.Vec3{sx * extent, sy * extent, sz * extent}
					index++
				}
			}
		}
		return fallback
	}
	fov := camera.FovYRadians
	tanHalf := float32(math.Tan(float64(fov) * 0.5))
	nearHeight := near * tanHalf
	nearWidth := nearHeight * aspect
	farHeight := far * tanHalf
	farWidth := farHeight * aspect
	viewSpace := [8]mgl32.Vec3{
		{-nearWidth, -nearHeight, -near},
		{nearWidth, -nearHeight, -near},
		{nearWidth, nearHeight, -near},
		{-nearWidth, nearHeight, -near},
		{-farWidth, -farHeight, -far},
		{farWidth, -farHeight, -far},
		{farWidth, farHeight, -far},
		{-farWidth, farHeight, -far},
	}
	view := render.CameraView(camera)
	inv := view.Inv()
	var corners [8]mgl32.Vec3
	for index, corner := range viewSpace {
		world := inv.Mul4x1(mgl32.Vec4{corner.X(), corner.Y(), corner.Z(), 1.0})
		corners[index] = mgl32.Vec3{world.X(), world.Y(), world.Z()}
	}
	return corners
}
