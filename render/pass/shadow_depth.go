package pass

import (
	_ "embed"
	"fmt"
	"math"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/go-gl/mathgl/mgl32"

	"indigo/ecs"
	"indigo/render"
	"indigo/render/asset"
	"indigo/transform"
	"indigo/window"
)

//go:embed shadow_depth.wgsl
var shadowDepthShader string

// ShadowMapSize is the side length of the directional shadow map.
// 2048 trades VRAM for sharper edges at typical scene distances.
const ShadowMapSize uint32 = 4096

// ShadowMapFormat is depth32float to match the engine's main depth.
const ShadowMapFormat = wgpu.TextureFormatDepth32Float

// Shadow owns the GPU-side directional shadow map: a depth texture
// the mesh pass samples + the light view-projection used to render
// it. One light slot today (the first directional light); cascades
// + multi-light atlases follow the same shape.
type Shadow struct {
	Texture       *wgpu.Texture
	View          *wgpu.TextureView
	Sampler       *wgpu.Sampler
	LightVPBuffer *wgpu.Buffer
	LightViewVP   mgl32.Mat4
}

// ShadowResource wraps Shadow as an engine-world resource.
type ShadowResource struct {
	Shadow *Shadow
}

type shadowUniform struct {
	LightViewProj mgl32.Mat4
}

type shadowDepthPassState struct {
	pipeline       *wgpu.RenderPipeline
	viewProjLayout *wgpu.BindGroupLayout
	handleBgLayout *wgpu.BindGroupLayout
	viewProjBg     *wgpu.BindGroup
}

// NewShadow allocates the depth texture + sampler + view. Stored
// as an engine-world resource so the mesh pass can fetch it for
// the lit fragment computation.
func NewShadow(device *wgpu.Device) (*Shadow, error) {
	tex, err := device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "shadow depth",
		Size: wgpu.Extent3D{
			Width:              ShadowMapSize,
			Height:             ShadowMapSize,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     wgpu.TextureDimension2D,
		Format:        ShadowMapFormat,
		Usage:         wgpu.TextureUsageRenderAttachment | wgpu.TextureUsageTextureBinding,
	})
	if err != nil {
		return nil, fmt.Errorf("shadow: depth texture: %w", err)
	}
	view, err := tex.CreateView(nil)
	if err != nil {
		tex.Release()
		return nil, fmt.Errorf("shadow: depth view: %w", err)
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
		tex.Release()
		view.Release()
		return nil, fmt.Errorf("shadow: sampler: %w", err)
	}
	uniformBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "shadow light_vp uniform",
		Size:  uint64(unsafe.Sizeof(shadowUniform{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		tex.Release()
		view.Release()
		sampler.Release()
		return nil, fmt.Errorf("shadow: uniform buffer: %w", err)
	}
	return &Shadow{
		Texture:       tex,
		View:          view,
		Sampler:       sampler,
		LightVPBuffer: uniformBuffer,
		LightViewVP:   mgl32.Ident4(),
	}, nil
}

// Release frees the shadow map's GPU resources.
func (s *Shadow) Release() {
	if s.LightVPBuffer != nil {
		s.LightVPBuffer.Release()
	}
	if s.Sampler != nil {
		s.Sampler.Release()
	}
	if s.View != nil {
		s.View.Release()
	}
	if s.Texture != nil {
		s.Texture.Release()
	}
}

// NewShadowDepthPass builds the depth-only render pass that draws
// every RenderMesh entity into the shadow texture from the
// directional sun's point of view. Mesh-pass per-handle bind
// groups are reused; only the model storage buffer at binding 0
// is read by the shadow shader (positions only, no materials).
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
		Entries: []wgpu.BindGroupLayoutEntry{{
			Binding:    0,
			Visibility: wgpu.ShaderStageVertex,
			Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage},
		}},
	})
	if err != nil {
		return nil, fmt.Errorf("shadow_depth: handle layout: %w", err)
	}
	state.handleBgLayout = handleBgLayout

	viewProjBg, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "shadow view_proj bind group",
		Layout: viewProjLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: shadow.LightVPBuffer, Offset: 0, Size: uint64(unsafe.Sizeof(shadowUniform{}))},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("shadow_depth: view_proj bind group: %w", err)
	}
	state.viewProjBg = viewProjBg

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

	return &render.Pass{
		Name:    "shadow_depth",
		State:   state,
		Prepare: shadowDepthPrepare,
		Execute: shadowDepthExecute,
		Release: shadowDepthRelease,
	}, nil
}

// shadowDepthPrepare computes the directional light's view +
// ortho projection. The frustum encloses the active camera's
// visible frustum corners in light space so the shadow texture
// is sized to what the camera sees.
func shadowDepthPrepare(s any, context *render.PassContext) error {
	_ = s

	lightDir := mgl32.Vec3{-0.3, -1.0, -0.4}.Normalize()
	lightMask := ecs.MustMaskOf[render.Light](context.World)
	context.World.ForEach(lightMask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		light, ok := ecs.Get[render.Light](context.World, entity)
		if !ok || light.Type != render.LightTypeDirectional {
			return
		}
		if global, ok := ecs.Get[transform.GlobalTransform](context.World, entity); ok {
			forward := mgl32.Vec3{-global.Matrix[8], -global.Matrix[9], -global.Matrix[10]}
			if forward.Len() > 1e-4 {
				lightDir = forward.Normalize()
			}
		}
	})

	camera, ok := ecs.Resource[render.Camera](context.World)
	aspect := float32(1.0)
	if resource, hasViewport := ecs.Resource[window.Window](context.World); hasViewport {
		if resource.Viewport.Height > 0 {
			aspect = float32(resource.Viewport.Width) / float32(resource.Viewport.Height)
		}
	}
	corners := cameraFrustumCornersWorld(camera, ok, aspect)
	center := mgl32.Vec3{0, 0, 0}
	for _, corner := range corners {
		center = center.Add(corner)
	}
	center = center.Mul(1.0 / float32(len(corners)))
	maxRadius := float32(0)
	for _, corner := range corners {
		d := corner.Sub(center).Len()
		if d > maxRadius {
			maxRadius = d
		}
	}
	if maxRadius < 1.0 {
		maxRadius = 30.0
	}

	up := mgl32.Vec3{0, 1, 0}
	if mgl32.Abs(lightDir.Y()) > 0.99 {
		up = mgl32.Vec3{1, 0, 0}
	}
	eye := center.Sub(lightDir.Mul(maxRadius * 4.0))
	lightView := mgl32.LookAtV(eye, center, up)

	minX := float32(math.MaxFloat32)
	maxX := float32(-math.MaxFloat32)
	minY := minX
	maxY := maxX
	minZ := minX
	maxZ := maxX
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
	zPad := float32(50.0)
	minZ -= zPad
	maxZ += zPad
	lightProj := orthoZO(minX, maxX, minY, maxY, -maxZ, -minZ)
	lightVP := lightProj.Mul4(lightView)

	shadow := ecs.MustResource[ShadowResource](context.World).Shadow
	uniform := shadowUniform{LightViewProj: lightVP}
	writeBuffer(context.Device, context.Queue, context.Encoder, shadow.LightVPBuffer, 0, bytesOf(&uniform))
	shadow.LightViewVP = lightVP
	return nil
}

// shadowDepthExecute draws every RenderMesh bucket the mesh pass
// already knows about into the shadow depth texture. Reads the
// mesh pass's per-handle model storage buffer through binding 0
// of the shadow's handle bind group, which is layout-compatible
// with the mesh pass's per-handle group.
func shadowDepthExecute(s any, context *render.PassContext) error {
	state := s.(*shadowDepthPassState)
	shadow := ecs.MustResource[ShadowResource](context.World).Shadow

	depthAttachment := wgpu.RenderPassDepthStencilAttachment{
		View:              shadow.View,
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
		Label:                  "shadow_depth",
		DepthStencilAttachment: &depthAttachment,
	})
	pass.SetPipeline(state.pipeline)
	pass.SetBindGroup(0, state.viewProjBg, nil)

	meshState, ok := findMeshPassState(context.World)
	if !ok {
		pass.End()
		pass.Release()
		return nil
	}
	assets := ecs.MustResource[asset.MeshAssetsResource](context.World).Assets

	for _, handle := range meshState.sortedHandles {
		bucket := meshState.perHandle[handle]
		entry, ok := assets.Lookup(handle)
		if !ok {
			continue
		}
		shadowBg, err := ensureShadowHandleBindGroup(bucket, context.Device, state.handleBgLayout)
		if err != nil {
			pass.End()
			pass.Release()
			return err
		}
		pass.SetBindGroup(1, shadowBg, nil)
		pass.SetVertexBuffer(0, entry.Vertices, 0, wgpu.WholeSize)
		pass.Draw(entry.VertexCount, uint32(len(bucket.slotEntity)), 0, 0)
	}

	pass.End()
	pass.Release()
	return nil
}

func shadowDepthRelease(s any) {
	state := s.(*shadowDepthPassState)
	if state.viewProjBg != nil {
		state.viewProjBg.Release()
	}
	if state.pipeline != nil {
		state.pipeline.Release()
	}
	if state.handleBgLayout != nil {
		state.handleBgLayout.Release()
	}
	if state.viewProjLayout != nil {
		state.viewProjLayout.Release()
	}
}

// ensureShadowHandleBindGroup lazily creates a one-binding bind
// group over the bucket's modelBuffer. Cached on the bucket so
// subsequent frames reuse it; invalidated when the bucket's
// buffers are reallocated by ensureHandleCapacity (which clears
// the cached group via the meshPass invalidate hook).
func ensureShadowHandleBindGroup(bucket *handleInstances, device *wgpu.Device, layout *wgpu.BindGroupLayout) (*wgpu.BindGroup, error) {
	if bucket.shadowBindGroup != nil {
		return bucket.shadowBindGroup, nil
	}
	bg, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "shadow per-handle bind group",
		Layout: layout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: bucket.modelBuffer, Offset: 0, Size: wgpu.WholeSize},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("shadow_depth: per-handle bind group: %w", err)
	}
	bucket.shadowBindGroup = bg
	return bg, nil
}

// AddShadowDepthPass registers the shadow depth pass with the
// renderer's graph. It has no graph slots (its output is the
// engine-world ShadowResource's texture); placed before the mesh
// pass so the mesh shader can sample the populated depth texture.
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

// findMeshPassState fishes the mesh pass's state out of the render
// graph by name. Used by the shadow pass to share per-handle
// buckets without duplicating the mesh/entity bookkeeping.
func findMeshPassState(_ *ecs.World) (*meshPassState, bool) {
	state, ok := sharedMeshPassState.Load().(*meshPassState)
	return state, ok && state != nil
}

// orthoZO returns a zero-to-one depth orthographic projection
// suitable for WebGPU's clip space. mgl32.Ortho produces the
// OpenGL convention with z in [-1, 1] which gets half-clipped
// when used as a shadow projection target.
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

// cameraFrustumCornersWorld returns the 8 world-space corners of
// the camera's view frustum (near + far rect corners), or a
// scene-centered fallback box when the camera resource is missing
// or has zero extent. Used by the shadow pass to fit the
// directional light's ortho projection around the visible scene.
func cameraFrustumCornersWorld(camera *render.Camera, hasCamera bool, aspect float32) [8]mgl32.Vec3 {
	if !hasCamera || camera == nil {
		var fallback [8]mgl32.Vec3
		extent := float32(30.0)
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
	near := camera.Near
	far := camera.Far
	if far > near+40.0 {
		far = near + 40.0
	}
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
