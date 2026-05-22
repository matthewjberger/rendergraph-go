package pass

import (
	"fmt"
	"math"
	"sort"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/go-gl/mathgl/mgl32"

	"indigo/ecs"
	"indigo/render"
	"indigo/render/asset"
	"indigo/transform"
)

// MaxSpotShadows caps the spotlights that can write into the atlas
// per frame. The reference engine packs by shelf; we use a fixed
// 2x2 grid (1024x1024 per slot in a 2048 atlas).
const MaxSpotShadows = 4

// SpotShadowAtlasSize is the side length of the single depth atlas
// shared by every shadow-casting spot light this frame.
const SpotShadowAtlasSize uint32 = 2048

// SpotShadowSlotsPerRow controls the atlas layout. 2 -> 2x2 grid.
const SpotShadowSlotsPerRow uint32 = 2

// SpotShadowSlotSize is the side length of one atlas slot.
const SpotShadowSlotSize uint32 = SpotShadowAtlasSize / SpotShadowSlotsPerRow

// SpotShadow owns the spotlight shadow atlas + per-slot uniform
// buffers + the mesh-side uniform array the mesh shader samples.
type SpotShadow struct {
	Texture       *wgpu.Texture
	AtlasView     *wgpu.TextureView
	Sampler       *wgpu.Sampler
	SlotBuffers   [MaxSpotShadows]*wgpu.Buffer
	UniformBuffer *wgpu.Buffer
	ActiveCount   uint32
	SlotEntities  [MaxSpotShadows]ecs.Entity
}

// slotEntity returns the ECS entity assigned to a given atlas slot
// this frame. Used by the mesh pass to thread shadow_index through
// each light's GPU record.
func (shadow *SpotShadow) slotEntity(index uint32) ecs.Entity {
	if index >= uint32(len(shadow.SlotEntities)) {
		return ecs.Entity{}
	}
	return shadow.SlotEntities[index]
}

// SpotShadowResource wraps SpotShadow for engine-world storage.
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
	Pad0        float32
	Pad1        float32
}

type spotShadowMeshUniform struct {
	Entries [MaxSpotShadows]spotShadowEntry
	Count   uint32
	Pad0    uint32
	Pad1    uint32
	Pad2    uint32
}

// NewSpotShadow builds the depth atlas + sampler + per-slot
// view-projection uniform buffers + the mesh-side uniform buffer.
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

// AddSpotShadowPass registers a pass that gathers spotlight
// candidates, computes per-light view-projection matrices, packs
// them into atlas slots, and renders each slot's geometry to the
// atlas with a viewport/scissor restricted to that slot.
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

		slotUniform := spotShadowSlotUniform{LightViewProj: viewProj}
		writeBuffer(context.Device, context.Queue, context.Encoder, shadow.SlotBuffers[index], 0, bytesOf(&slotUniform))

		meshUniform.Entries[index] = spotShadowEntry{
			ViewProj:    viewProj,
			AtlasOffset: mgl32.Vec2{atlasOffX, atlasOffY},
			AtlasScale:  mgl32.Vec2{atlasScale, atlasScale},
			Bias:        light.ShadowBias,
			Active:      1,
		}
		shadow.SlotEntities[index] = candidate.Entity
	}
	meshUniform.Count = uint32(len(candidates))
	shadow.ActiveCount = meshUniform.Count
	writeBuffer(context.Device, context.Queue, context.Encoder, shadow.UniformBuffer, 0, bytesOf(&meshUniform))
	return nil
}

func spotShadowExecute(s any, context *render.PassContext) error {
	state := s.(*spotShadowPassState)
	shadow := state.shadow
	if shadow.ActiveCount == 0 {
		return nil
	}

	meshState, ok := findMeshPassState(context.World)
	if !ok {
		return nil
	}
	assets := ecs.MustResource[asset.MeshAssetsResource](context.World).Assets
	lightMask := ecs.MustMaskOf[render.Light](context.World)

	for index := uint32(0); index < shadow.ActiveCount; index++ {
		slotX := (index % SpotShadowSlotsPerRow) * SpotShadowSlotSize
		slotY := (index / SpotShadowSlotsPerRow) * SpotShadowSlotSize
		var loadOp wgpu.LoadOp
		if index == 0 {
			loadOp = wgpu.LoadOpClear
		} else {
			loadOp = wgpu.LoadOpLoad
		}
		passEnc := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
			Label: "spot shadow slot",
			DepthStencilAttachment: &wgpu.RenderPassDepthStencilAttachment{
				View:            shadow.AtlasView,
				DepthLoadOp:     loadOp,
				DepthStoreOp:    wgpu.StoreOpStore,
				DepthClearValue: 1.0,
			},
		})
		passEnc.SetPipeline(state.pipeline)
		passEnc.SetViewport(float32(slotX), float32(slotY), float32(SpotShadowSlotSize), float32(SpotShadowSlotSize), 0, 1)
		passEnc.SetScissorRect(slotX, slotY, SpotShadowSlotSize, SpotShadowSlotSize)
		passEnc.SetBindGroup(0, state.slotBgs[index], nil)
		for _, handle := range meshState.sortedHandles {
			bucket := meshState.perHandle[handle]
			entry, ok := assets.Lookup(handle)
			if !ok {
				continue
			}
			shadowBg, err := ensureShadowHandleBindGroup(bucket, context.Device, state.handleBgLayout)
			if err != nil {
				passEnc.End()
				passEnc.Release()
				return err
			}
			passEnc.SetBindGroup(1, shadowBg, nil)
			passEnc.SetVertexBuffer(0, entry.Vertices, 0, wgpu.WholeSize)
			drawNonLightInstances(passEnc, bucket, entry.VertexCount, lightMask, context.World)
		}
		passEnc.End()
		passEnc.Release()
	}
	return nil
}

// drawNonLightInstances issues per-instance draws over a bucket,
// skipping any instance whose ECS entity has a Light component.
// Without this, the spotlight's own orb mesh (sitting at the
// shadow camera's near plane) projects as a giant occluder and
// shadows everything below the light.
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

// spotPerspectiveZO is a right-handed perspective with [0,1] clip
// depth, matching WebGPU's NDC. Mirror of mgl32.Perspective but
// with z mapped to [0, 1] instead of [-1, 1].
func spotPerspectiveZO(fovYRadians, aspect, near, far float32) mgl32.Mat4 {
	f := float32(1.0 / math.Tan(float64(fovYRadians)/2.0))
	return mgl32.Mat4{
		f / aspect, 0, 0, 0,
		0, f, 0, 0,
		0, 0, far / (near - far), -1,
		0, 0, (far * near) / (near - far), 0,
	}
}

// float32SliceBytes reinterprets a float32 slice as a byte slice
// for buffer writes without copying.
func float32SliceBytes(values []float32) []byte {
	if len(values) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), len(values)*4)
}
