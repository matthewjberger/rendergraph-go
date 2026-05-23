package pass

import (
	_ "embed"
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

//go:embed point_shadow_depth.wgsl
var pointShadowDepthShader string

const MaxPointShadows = 4

const PointShadowFaceSize uint32 = 512

const PointShadowFaces = 6

var pointShadowFaceAxes = [PointShadowFaces][2]mgl32.Vec3{
	{{1, 0, 0}, {0, -1, 0}},
	{{-1, 0, 0}, {0, -1, 0}},
	{{0, 1, 0}, {0, 0, 1}},
	{{0, -1, 0}, {0, 0, -1}},
	{{0, 0, 1}, {0, -1, 0}},
	{{0, 0, -1}, {0, -1, 0}},
}

type PointShadow struct {
	Texture       *wgpu.Texture
	CubeArrayView *wgpu.TextureView
	FaceViews     [MaxPointShadows * PointShadowFaces]*wgpu.TextureView
	DepthTexture  *wgpu.Texture
	DepthView     *wgpu.TextureView
	Sampler       *wgpu.Sampler
	FaceUniform   *wgpu.Buffer
	DataBuffer    *wgpu.Buffer

	ActiveCount  uint32
	SlotEntities [MaxPointShadows]ecs.Entity
}

func (shadow *PointShadow) slotEntity(index uint32) ecs.Entity {
	if index >= uint32(len(shadow.SlotEntities)) {
		return ecs.Entity{}
	}
	return shadow.SlotEntities[index]
}

type PointShadowResource struct {
	Shadow *PointShadow
}

type pointShadowFaceUniform struct {
	ViewProjection mgl32.Mat4
	LightPosition  mgl32.Vec3
	LightRange     float32
}

type pointShadowEntry struct {
	Position  mgl32.Vec3
	Range     float32
	Bias      float32
	LightSize float32
	Pad1      float32
	Pad2      float32
}

type pointShadowMeshUniform struct {
	Entries [MaxPointShadows]pointShadowEntry
	Count   uint32
	Pad0    uint32
	Pad1    uint32
	Pad2    uint32
}

func NewPointShadow(device *wgpu.Device) (*PointShadow, error) {
	shadow := &PointShadow{}
	tex, err := device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "point shadow cube array",
		Size: wgpu.Extent3D{
			Width:              PointShadowFaceSize,
			Height:             PointShadowFaceSize,
			DepthOrArrayLayers: MaxPointShadows * PointShadowFaces,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     wgpu.TextureDimension2D,
		Format:        wgpu.TextureFormatR32Float,
		Usage:         wgpu.TextureUsageRenderAttachment | wgpu.TextureUsageTextureBinding,
	})
	if err != nil {
		return nil, fmt.Errorf("point shadow: texture: %w", err)
	}
	shadow.Texture = tex

	cubeArrayView, err := tex.CreateView(&wgpu.TextureViewDescriptor{
		Label:           "point shadow cube array view",
		Format:          wgpu.TextureFormatR32Float,
		Dimension:       wgpu.TextureViewDimensionCubeArray,
		Aspect:          wgpu.TextureAspectAll,
		BaseMipLevel:    0,
		MipLevelCount:   1,
		BaseArrayLayer:  0,
		ArrayLayerCount: MaxPointShadows * PointShadowFaces,
	})
	if err != nil {
		return nil, fmt.Errorf("point shadow: cube array view: %w", err)
	}
	shadow.CubeArrayView = cubeArrayView

	for index := 0; index < MaxPointShadows*PointShadowFaces; index++ {
		view, err := tex.CreateView(&wgpu.TextureViewDescriptor{
			Label:           "point shadow face view",
			Format:          wgpu.TextureFormatR32Float,
			Dimension:       wgpu.TextureViewDimension2D,
			Aspect:          wgpu.TextureAspectAll,
			BaseMipLevel:    0,
			MipLevelCount:   1,
			BaseArrayLayer:  uint32(index),
			ArrayLayerCount: 1,
		})
		if err != nil {
			return nil, fmt.Errorf("point shadow: face view %d: %w", index, err)
		}
		shadow.FaceViews[index] = view
	}

	depthTex, err := device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "point shadow depth",
		Size: wgpu.Extent3D{
			Width:              PointShadowFaceSize,
			Height:             PointShadowFaceSize,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     wgpu.TextureDimension2D,
		Format:        wgpu.TextureFormatDepth32Float,
		Usage:         wgpu.TextureUsageRenderAttachment,
	})
	if err != nil {
		return nil, fmt.Errorf("point shadow: depth texture: %w", err)
	}
	shadow.DepthTexture = depthTex
	depthView, err := depthTex.CreateView(nil)
	if err != nil {
		return nil, fmt.Errorf("point shadow: depth view: %w", err)
	}
	shadow.DepthView = depthView

	sampler, err := device.CreateSampler(&wgpu.SamplerDescriptor{
		Label:         "point shadow sampler",
		AddressModeU:  wgpu.AddressModeClampToEdge,
		AddressModeV:  wgpu.AddressModeClampToEdge,
		AddressModeW:  wgpu.AddressModeClampToEdge,
		MagFilter:     wgpu.FilterModeNearest,
		MinFilter:     wgpu.FilterModeNearest,
		MipmapFilter:  wgpu.MipmapFilterModeNearest,
		MaxAnisotropy: 1,
	})
	if err != nil {
		return nil, fmt.Errorf("point shadow: sampler: %w", err)
	}
	shadow.Sampler = sampler

	faceUniform, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "point shadow face uniforms",
		Size:  uint64(MaxPointShadows*PointShadowFaces) * pointShadowFaceUniformStride,
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("point shadow: face uniform: %w", err)
	}
	shadow.FaceUniform = faceUniform

	dataBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "point shadow data",
		Size:  uint64(unsafe.Sizeof(pointShadowMeshUniform{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("point shadow: data buffer: %w", err)
	}
	shadow.DataBuffer = dataBuffer

	return shadow, nil
}

const pointShadowFaceUniformStride uint64 = 256

type pointShadowCandidate struct {
	Entity     ecs.Entity
	Light      render.Light
	Global     transform.GlobalTransform
	DistanceSq float32
}

type pointShadowPassState struct {
	shadow        *PointShadow
	pipeline      *wgpu.RenderPipeline
	uniformLayout *wgpu.BindGroupLayout
	handleLayout  *wgpu.BindGroupLayout
	faceBgs       [MaxPointShadows * PointShadowFaces]*wgpu.BindGroup
}

func AddPointShadowPass(renderer *render.Renderer, shadow *PointShadow) (*render.Pass, error) {
	shaderModule, err := renderer.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "point shadow shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: pointShadowDepthShader},
	})
	if err != nil {
		return nil, fmt.Errorf("point shadow: shader: %w", err)
	}
	defer shaderModule.Release()

	uniformLayout, err := renderer.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "point shadow uniform layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: wgpu.ShaderStageVertex | wgpu.ShaderStageFragment,
				Buffer: wgpu.BufferBindingLayout{
					Type:           wgpu.BufferBindingTypeUniform,
					MinBindingSize: uint64(unsafe.Sizeof(pointShadowFaceUniform{})),
				},
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("point shadow: uniform layout: %w", err)
	}

	handleLayout, err := renderer.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "point shadow handle layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: wgpu.ShaderStageVertex,
				Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage},
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("point shadow: handle layout: %w", err)
	}

	pipelineLayout, err := renderer.Device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "point shadow pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{uniformLayout, handleLayout},
	})
	if err != nil {
		return nil, fmt.Errorf("point shadow: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	pipeline, err := renderer.Device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "point shadow pipeline",
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
			Topology: wgpu.PrimitiveTopologyTriangleList,
			CullMode: wgpu.CullModeBack,

			FrontFace: wgpu.FrontFaceCW,
		},
		DepthStencil: &wgpu.DepthStencilState{
			Format:            wgpu.TextureFormatDepth32Float,
			DepthWriteEnabled: true,
			DepthCompare:      wgpu.CompareFunctionLess,
			StencilFront:      wgpu.StencilFaceState{Compare: wgpu.CompareFunctionAlways},
			StencilBack:       wgpu.StencilFaceState{Compare: wgpu.CompareFunctionAlways},
		},
		Multisample: wgpu.MultisampleState{Count: 1, Mask: 0xFFFFFFFF},
		Fragment: &wgpu.FragmentState{
			Module:     shaderModule,
			EntryPoint: "fragment_main",
			Targets: []wgpu.ColorTargetState{{
				Format:    wgpu.TextureFormatR32Float,
				WriteMask: wgpu.ColorWriteMaskAll,
			}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("point shadow: pipeline: %w", err)
	}

	state := &pointShadowPassState{
		shadow:        shadow,
		pipeline:      pipeline,
		uniformLayout: uniformLayout,
		handleLayout:  handleLayout,
	}

	for face := 0; face < MaxPointShadows*PointShadowFaces; face++ {
		bg, err := renderer.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  "point shadow face bg",
			Layout: uniformLayout,
			Entries: []wgpu.BindGroupEntry{
				{
					Binding: 0,
					Buffer:  shadow.FaceUniform,
					Offset:  uint64(face) * pointShadowFaceUniformStride,
					Size:    uint64(unsafe.Sizeof(pointShadowFaceUniform{})),
				},
			},
		})
		if err != nil {
			return nil, fmt.Errorf("point shadow: face bg %d: %w", face, err)
		}
		state.faceBgs[face] = bg
	}

	p := &render.Pass{
		Name:    "point_shadow",
		State:   state,
		Prepare: pointShadowPrepare,
		Execute: pointShadowExecute,
		Release: pointShadowRelease,
	}
	if err := renderer.Graph.AddPass(p, nil); err != nil {
		return nil, err
	}
	return p, nil
}

func pointShadowPrepare(s any, context *render.PassContext) error {
	state := s.(*pointShadowPassState)
	shadow := state.shadow
	shadow.ActiveCount = 0
	for index := range shadow.SlotEntities {
		shadow.SlotEntities[index] = ecs.Entity{}
	}

	camPos := mgl32.Vec3{}
	if camera, ok := ecs.Resource[render.Camera](context.World); ok && camera != nil {
		camPos = mgl32.Vec3{camera.Eye[0], camera.Eye[1], camera.Eye[2]}
	}

	candidates := make([]pointShadowCandidate, 0, 8)
	mask := ecs.MustMaskOf[render.Light](context.World) | ecs.MustMaskOf[transform.GlobalTransform](context.World)
	context.World.ForEach(mask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		light, _ := ecs.Get[render.Light](context.World, entity)
		if light.Type != render.LightTypePoint || !light.CastShadows {
			return
		}
		global, _ := ecs.Get[transform.GlobalTransform](context.World, entity)
		position := mgl32.Vec3{global.Matrix[12], global.Matrix[13], global.Matrix[14]}
		distance := position.Sub(camPos)
		candidates = append(candidates, pointShadowCandidate{
			Entity:     entity,
			Light:      *light,
			Global:     *global,
			DistanceSq: distance.LenSqr(),
		})
	})

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].DistanceSq < candidates[j].DistanceSq
	})

	if len(candidates) > MaxPointShadows {
		candidates = candidates[:MaxPointShadows]
	}

	var meshUniform pointShadowMeshUniform
	for index, candidate := range candidates {
		light := candidate.Light
		global := candidate.Global
		position := mgl32.Vec3{global.Matrix[12], global.Matrix[13], global.Matrix[14]}
		rangeValue := light.Range
		if rangeValue < 0.5 {
			rangeValue = 0.5
		}
		baseProj := spotPerspectiveZO(float32(math.Pi/2), 1.0, 0.1, rangeValue)

		yFlip := mgl32.Mat4{
			1, 0, 0, 0,
			0, -1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		}
		proj := yFlip.Mul4(baseProj)
		for face := 0; face < PointShadowFaces; face++ {
			axes := pointShadowFaceAxes[face]
			forward := axes[0]
			up := axes[1]
			target := position.Add(forward)
			view := mgl32.LookAtV(position, target, up)
			viewProj := proj.Mul4(view)

			uniform := pointShadowFaceUniform{
				ViewProjection: viewProj,
				LightPosition:  position,
				LightRange:     rangeValue,
			}
			offset := uint64(index*PointShadowFaces+face) * pointShadowFaceUniformStride
			writeBuffer(context.Device, context.Queue, context.Encoder, shadow.FaceUniform, offset, bytesOf(&uniform))
		}

		lightSize := light.LightSize
		if lightSize <= 0 {
			lightSize = 1.0
		}
		meshUniform.Entries[index] = pointShadowEntry{
			Position:  position,
			Range:     rangeValue,
			Bias:      light.ShadowBias,
			LightSize: lightSize,
		}
		shadow.SlotEntities[index] = candidate.Entity
	}

	meshUniform.Count = uint32(len(candidates))
	shadow.ActiveCount = meshUniform.Count
	writeBuffer(context.Device, context.Queue, context.Encoder, shadow.DataBuffer, 0, bytesOf(&meshUniform))
	return nil
}

func pointShadowExecute(s any, context *render.PassContext) error {
	state := s.(*pointShadowPassState)
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

	for slot := uint32(0); slot < shadow.ActiveCount; slot++ {
		for face := 0; face < PointShadowFaces; face++ {
			layer := slot*PointShadowFaces + uint32(face)
			passEnc := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
				Label: "point shadow face",
				ColorAttachments: []wgpu.RenderPassColorAttachment{{
					View:       shadow.FaceViews[layer],
					LoadOp:     wgpu.LoadOpClear,
					StoreOp:    wgpu.StoreOpStore,
					ClearValue: wgpu.Color{R: 1.0, G: 1.0, B: 1.0, A: 1.0},
				}},
				DepthStencilAttachment: &wgpu.RenderPassDepthStencilAttachment{
					View:            shadow.DepthView,
					DepthLoadOp:     wgpu.LoadOpClear,
					DepthStoreOp:    wgpu.StoreOpStore,
					DepthClearValue: 1.0,
				},
			})
			passEnc.SetPipeline(state.pipeline)
			passEnc.SetBindGroup(0, state.faceBgs[layer], nil)
			for _, handle := range meshState.sortedHandles {
				bucket := meshState.perHandle[handle]
				entry, ok := assets.Lookup(handle)
				if !ok {
					continue
				}
				shadowBg, err := ensureShadowHandleBindGroup(bucket, context.Device, state.handleLayout)
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
	}
	return nil
}

func pointShadowRelease(s any) {
	state := s.(*pointShadowPassState)
	for index := range state.faceBgs {
		if state.faceBgs[index] != nil {
			state.faceBgs[index].Release()
		}
	}
	if state.pipeline != nil {
		state.pipeline.Release()
	}
	if state.uniformLayout != nil {
		state.uniformLayout.Release()
	}
	if state.handleLayout != nil {
		state.handleLayout.Release()
	}
}
