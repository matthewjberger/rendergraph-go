package pass

import (
	"fmt"
	"math"
	"sort"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/go-gl/mathgl/mgl32"

	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/render/asset"
	"github.com/matthewjberger/indigo/transform"
)

const MaxSpotShadows = 4

const SpotShadowAtlasSize uint32 = 2048

const SpotShadowSlotsPerRow uint32 = 2

const SpotShadowSlotSize uint32 = SpotShadowAtlasSize / SpotShadowSlotsPerRow

type SpotShadow struct {
	Texture       *wgpu.Texture
	AtlasView     *wgpu.TextureView
	Sampler       *wgpu.Sampler
	SlotBuffers   [MaxSpotShadows]*wgpu.Buffer
	UniformBuffer *wgpu.Buffer
	ActiveCount   uint32
	SlotEntities  [MaxSpotShadows]ecs.Entity
}

func (shadow *SpotShadow) slotEntity(index uint32) ecs.Entity {
	if index >= uint32(len(shadow.SlotEntities)) {
		return ecs.Entity{}
	}
	return shadow.SlotEntities[index]
}

type SpotShadowResource struct {
	Shadow *SpotShadow
}

type spotShadowSlotUniform struct {
	LightViewProj mgl32.Mat4
}

type spotShadowEntry struct {
	ViewProj    mgl32.Mat4
	AtlasOffset mgl32.Vec2
	AtlasScale  mgl32.Vec2
	Bias        float32
	Active      uint32
	LightSize   float32
	Pad1        float32
}

type spotShadowMeshUniform struct {
	Entries [MaxSpotShadows]spotShadowEntry
	Count   uint32
	Pad0    uint32
	Pad1    uint32
	Pad2    uint32
}

func NewSpotShadow(device *wgpu.Device) (*SpotShadow, error) {
	shadow := &SpotShadow{}
	tex, err := device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "spot shadow atlas",
		Size: wgpu.Extent3D{
			Width:              SpotShadowAtlasSize,
			Height:             SpotShadowAtlasSize,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     wgpu.TextureDimension2D,
		Format:        ShadowMapFormat,
		Usage:         wgpu.TextureUsageRenderAttachment | wgpu.TextureUsageTextureBinding,
	})
	if err != nil {
		return nil, fmt.Errorf("spot shadow: atlas: %w", err)
	}
	shadow.Texture = tex

	atlasView, err := tex.CreateView(&wgpu.TextureViewDescriptor{
		Label:           "spot shadow atlas view",
		Format:          ShadowMapFormat,
		Dimension:       wgpu.TextureViewDimension2D,
		Aspect:          wgpu.TextureAspectDepthOnly,
		BaseMipLevel:    0,
		MipLevelCount:   1,
		BaseArrayLayer:  0,
		ArrayLayerCount: 1,
	})
	if err != nil {
		return nil, fmt.Errorf("spot shadow: atlas view: %w", err)
	}
	shadow.AtlasView = atlasView

	sampler, err := device.CreateSampler(&wgpu.SamplerDescriptor{
		Label:         "spot shadow sampler",
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
		return nil, fmt.Errorf("spot shadow: sampler: %w", err)
	}
	shadow.Sampler = sampler

	for index := 0; index < MaxSpotShadows; index++ {
		buf, err := device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: "spot shadow slot vp",
			Size:  uint64(unsafe.Sizeof(spotShadowSlotUniform{})),
			Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
		})
		if err != nil {
			return nil, fmt.Errorf("spot shadow: slot vp %d: %w", index, err)
		}
		shadow.SlotBuffers[index] = buf
	}

	uniformBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "spot shadow mesh uniform",
		Size:  uint64(unsafe.Sizeof(spotShadowMeshUniform{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("spot shadow: mesh uniform: %w", err)
	}
	shadow.UniformBuffer = uniformBuffer

	return shadow, nil
}

type spotShadowCandidate struct {
	Entity     ecs.Entity
	Light      render.Light
	Global     transform.GlobalTransform
	DistanceSq float32
}

func AddSpotShadowPass(renderer *render.Renderer, shadow *SpotShadow) (*render.Pass, error) {
	shaderModule, err := renderer.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "spot shadow shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shadowDepthShader},
	})
	if err != nil {
		return nil, fmt.Errorf("spot shadow: shader: %w", err)
	}
	defer shaderModule.Release()

	viewProjLayout, err := renderer.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "spot shadow vp bg layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: wgpu.ShaderStageVertex,
				Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform},
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("spot shadow: vp layout: %w", err)
	}

	handleBgLayout, err := renderer.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "spot shadow handle bg layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: wgpu.ShaderStageVertex,
				Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage},
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("spot shadow: handle layout: %w", err)
	}

	pipelineLayout, err := renderer.Device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "spot shadow pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{viewProjLayout, handleBgLayout},
	})
	if err != nil {
		return nil, fmt.Errorf("spot shadow: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	pipeline, err := renderer.Device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "spot shadow pipeline",
		Layout: pipelineLayout,
		Vertex: wgpu.VertexState{
			Module:     shaderModule,
			EntryPoint: "vertex_main",
			Buffers: []wgpu.VertexBufferLayout{{
				ArrayStride: uint64(unsafe.Sizeof(asset.MeshVertex{})),
				StepMode:    wgpu.VertexStepModeVertex,
				Attributes: []wgpu.VertexAttribute{{
					Format:         wgpu.VertexFormatFloat32x3,
					Offset:         0,
					ShaderLocation: 0,
				}},
			}},
		},
		Primitive: wgpu.PrimitiveState{
			Topology:  wgpu.PrimitiveTopologyTriangleList,
			CullMode:  wgpu.CullModeBack,
			FrontFace: wgpu.FrontFaceCCW,
		},
		DepthStencil: &wgpu.DepthStencilState{
			DepthWriteEnabled: true,
			DepthCompare:      wgpu.CompareFunctionLess,
			Format:            ShadowMapFormat,
			StencilFront:      wgpu.StencilFaceState{Compare: wgpu.CompareFunctionAlways},
			StencilBack:       wgpu.StencilFaceState{Compare: wgpu.CompareFunctionAlways},
		},
		Multisample: wgpu.MultisampleState{Count: 1, Mask: 0xFFFFFFFF},
	})
	if err != nil {
		return nil, fmt.Errorf("spot shadow: pipeline: %w", err)
	}

	slotBgs := [MaxSpotShadows]*wgpu.BindGroup{}
	for index := 0; index < MaxSpotShadows; index++ {
		bg, err := renderer.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  "spot shadow slot bg",
			Layout: viewProjLayout,
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: shadow.SlotBuffers[index], Offset: 0, Size: uint64(unsafe.Sizeof(spotShadowSlotUniform{}))},
			},
		})
		if err != nil {
			return nil, fmt.Errorf("spot shadow: slot bg %d: %w", index, err)
		}
		slotBgs[index] = bg
	}

	state := &spotShadowPassState{
		shadow:         shadow,
		pipeline:       pipeline,
		viewProjLayout: viewProjLayout,
		handleBgLayout: handleBgLayout,
		slotBgs:        slotBgs,
	}

	p := &render.Pass{
		Name:    "spot_shadow",
		Reads:   nil,
		Writes:  nil,
		State:   state,
		Prepare: spotShadowPrepare,
		Execute: spotShadowExecute,
		Release: spotShadowRelease,
	}
	if err := renderer.Graph.AddPass(p, nil); err != nil {
		return nil, err
	}
	return p, nil
}

type spotShadowPassState struct {
	shadow         *SpotShadow
	pipeline       *wgpu.RenderPipeline
	viewProjLayout *wgpu.BindGroupLayout
	handleBgLayout *wgpu.BindGroupLayout
	slotBgs        [MaxSpotShadows]*wgpu.BindGroup
}

func spotShadowPrepare(s any, context *render.PassContext) error {
	state := s.(*spotShadowPassState)
	shadow := state.shadow
	shadow.ActiveCount = 0
	for index := range shadow.SlotEntities {
		shadow.SlotEntities[index] = ecs.Entity{}
	}

	camPos := mgl32.Vec3{}
	if camera, ok := ecs.Resource[render.Camera](context.World); ok && camera != nil {
		camPos = mgl32.Vec3{camera.Eye[0], camera.Eye[1], camera.Eye[2]}
	}

	candidates := make([]spotShadowCandidate, 0, 8)
	mask := ecs.MustMaskOf[render.Light](context.World) | ecs.MustMaskOf[transform.GlobalTransform](context.World)
	context.World.ForEach(mask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		light, _ := ecs.Get[render.Light](context.World, entity)
		if light.Type != render.LightTypeSpot || !light.CastShadows {
			return
		}
		global, _ := ecs.Get[transform.GlobalTransform](context.World, entity)
		position := mgl32.Vec3{global.Matrix[12], global.Matrix[13], global.Matrix[14]}
		distance := position.Sub(camPos)
		candidates = append(candidates, spotShadowCandidate{
			Entity:     entity,
			Light:      *light,
			Global:     *global,
			DistanceSq: distance.LenSqr(),
		})
	})

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].DistanceSq < candidates[j].DistanceSq
	})

	if len(candidates) > MaxSpotShadows {
		candidates = candidates[:MaxSpotShadows]
	}

	var meshUniform spotShadowMeshUniform
	for index, candidate := range candidates {
		light := candidate.Light
		global := candidate.Global
		position := mgl32.Vec3{global.Matrix[12], global.Matrix[13], global.Matrix[14]}
		forward := mgl32.Vec3{-global.Matrix[8], -global.Matrix[9], -global.Matrix[10]}.Normalize()
		up := mgl32.Vec3{0, 1, 0}
		if math.Abs(float64(forward.Y())) > 0.99 {
			up = mgl32.Vec3{1, 0, 0}
		}
		target := position.Add(forward)
		view := mgl32.LookAtV(position, target, up)
		outer := light.OuterConeAngle
		if outer < 0.01 {
			outer = 0.01
		}
		far := light.Range
		if far < 1.0 {
			far = 1.0
		}
		proj := spotPerspectiveZO(outer*2, 1.0, 0.1, far)
		viewProj := proj.Mul4(view)

		row := uint32(index) / SpotShadowSlotsPerRow
		col := uint32(index) % SpotShadowSlotsPerRow
		atlasOffX := float32(col) / float32(SpotShadowSlotsPerRow)
		atlasOffY := float32(row) / float32(SpotShadowSlotsPerRow)
		atlasScale := float32(1) / float32(SpotShadowSlotsPerRow)

		atlasBiasX := 2*atlasOffX - 1 + atlasScale
		atlasBiasY := 1 - 2*atlasOffY - atlasScale
		atlasTransform := mgl32.Mat4{
			atlasScale, 0, 0, 0,
			0, atlasScale, 0, 0,
			0, 0, 1, 0,
			atlasBiasX, atlasBiasY, 0, 1,
		}
		slotViewProj := atlasTransform.Mul4(viewProj)

		slotUniform := spotShadowSlotUniform{LightViewProj: slotViewProj}
		writeBuffer(context.Device, context.Queue, context.Encoder, shadow.SlotBuffers[index], 0, bytesOf(&slotUniform))

		lightSize := light.LightSize
		if lightSize <= 0 {
			lightSize = 1.0
		}
		meshUniform.Entries[index] = spotShadowEntry{
			ViewProj:    viewProj,
			AtlasOffset: mgl32.Vec2{atlasOffX, atlasOffY},
			AtlasScale:  mgl32.Vec2{atlasScale, atlasScale},
			Bias:        light.ShadowBias,
			Active:      1,
			LightSize:   lightSize,
		}
		shadow.SlotEntities[index] = candidate.Entity
	}
	meshUniform.Count = uint32(len(candidates))
	shadow.ActiveCount = meshUniform.Count
	writeBuffer(context.Device, context.Queue, context.Encoder, shadow.UniformBuffer, 0, bytesOf(&meshUniform))
	return nil
}

func drawNonLightInstances(passEnc *wgpu.RenderPassEncoder, bucket *handleInstances, vertexCount uint32, lightMask ecs.Mask, world *ecs.World) {
	start := uint32(0)
	for slot, entity := range bucket.slotEntity {
		if world.HasComponents(entity, lightMask) {
			if uint32(slot) > start {
				passEnc.Draw(vertexCount, uint32(slot)-start, 0, start)
			}
			start = uint32(slot) + 1
		}
	}
	if uint32(len(bucket.slotEntity)) > start {
		passEnc.Draw(vertexCount, uint32(len(bucket.slotEntity))-start, 0, start)
	}
}

func spotShadowRelease(s any) {
	state := s.(*spotShadowPassState)
	for index := range state.slotBgs {
		if state.slotBgs[index] != nil {
			state.slotBgs[index].Release()
		}
	}
	if state.pipeline != nil {
		state.pipeline.Release()
	}
	if state.viewProjLayout != nil {
		state.viewProjLayout.Release()
	}
	if state.handleBgLayout != nil {
		state.handleBgLayout.Release()
	}
}

func spotPerspectiveZO(fovYRadians, aspect, near, far float32) mgl32.Mat4 {
	f := float32(1.0 / math.Tan(float64(fovYRadians)/2.0))
	return mgl32.Mat4{
		f / aspect, 0, 0, 0,
		0, f, 0, 0,
		0, 0, far / (near - far), -1,
		0, 0, (far * near) / (near - far), 0,
	}
}
