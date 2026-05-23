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
	"github.com/matthewjberger/indigo/window"
)

func meshPrepare(s any, context *render.PassContext) error {
	state := s.(*meshPassState)

	camera := ecs.MustResource[render.Camera](context.World)
	aspect := state.aspectFn()
	viewProjection := render.CameraViewProjection(camera, aspect)
	writeBuffer(context.Device, context.Queue, context.Encoder, state.viewProjBuffer, 0, bytesOf(&viewProjection))

	view := render.CameraView(camera)
	writeBuffer(context.Device, context.Queue, context.Encoder, state.clusters.viewMatrix, 0, bytesOf(&view))

	state.lightScratch = extractLights(context.World, state.lightScratch[:0])
	uploadLights(context.Device, context.Queue, context.Encoder, state.clusters.lights, state.lightScratch)
	numDirectional, numLocal := splitDirectionalAndLocal(state.lightScratch)

	uniforms := buildClusterUniforms(camera, aspect, context, numDirectional, numLocal)
	state.clusterUniformsScratch = uniforms
	writeBuffer(context.Device, context.Queue, context.Encoder, state.clusters.clusterUniforms, 0, bytesOf(&uniforms))

	for _, event := range ecs.DrainEvents[ecs.EntityDespawned](context.World) {
		releaseEntitySlot(state, context, event.Entity)
	}

	globalMask := ecs.MustMaskOf[transform.GlobalTransform](context.World)
	renderMeshMask := ecs.MustMaskOf[asset.RenderMesh](context.World)

	ecs.IterChanged1[asset.RenderMesh](
		context.World,
		globalMask,
		0,
		func(entity ecs.Entity, mesh *asset.RenderMesh) {
			if existing, already := state.entityHandle[entity]; already {
				if existing == mesh.Mesh {
					return
				}
				releaseEntitySlot(state, context, entity)
			}
			global, ok := ecs.Get[transform.GlobalTransform](context.World, entity)
			if !ok {
				return
			}
			handle := mesh.Mesh
			bucket, ok := state.perHandle[handle]
			if !ok {
				bucket = &handleInstances{
					entityToSlot: make(map[ecs.Entity]uint32, 16),
				}
				state.perHandle[handle] = bucket
			}
			slot := uint32(len(bucket.slotEntity))
			bucket.entityToSlot[entity] = slot
			bucket.slotEntity = append(bucket.slotEntity, entity)
			state.entityHandle[entity] = handle
			if err := ensureHandleCapacity(bucket, context.Device, state.handleBgLayout); err != nil {
				return
			}
			writeBuffer(context.Device, context.Queue, context.Encoder, bucket.modelBuffer, uint64(slot)*matrixSize, bytesOf(&global.Matrix))

			registry := ecs.MustResource[asset.MaterialRegistryResource](context.World).Registry
			materialID, isNew := registry.AssignID(entity)
			if isNew {
				if _, err := registry.EnsureCapacity(registry.Count()); err != nil {
					return
				}
				material := asset.DefaultMaterial().ToGPU()
				if mat, ok := ecs.Get[asset.Material](context.World, entity); ok {
					material = mat.ToGPU()
				}
				registry.Write(context.Queue, materialID, material)
			}
			writeBuffer(context.Device, context.Queue, context.Encoder, bucket.materialIndexBuffer, uint64(slot)*4, bytesOf(&materialID))

			entityID := entity.ID
			writeBuffer(context.Device, context.Queue, context.Encoder, bucket.entityIdBuffer, uint64(slot)*4, bytesOf(&entityID))
		},
	)

	ecs.IterChanged1[transform.GlobalTransform](
		context.World,
		renderMeshMask,
		0,
		func(entity ecs.Entity, global *transform.GlobalTransform) {
			mesh, ok := ecs.Get[asset.RenderMesh](context.World, entity)
			if !ok {
				return
			}
			bucket, ok := state.perHandle[mesh.Mesh]
			if !ok {
				return
			}
			slot, ok := bucket.entityToSlot[entity]
			if !ok {
				return
			}
			writeBuffer(context.Device, context.Queue, context.Encoder, bucket.modelBuffer, uint64(slot)*matrixSize, bytesOf(&global.Matrix))
		},
	)

	ecs.IterChanged1[asset.Material](
		context.World,
		renderMeshMask,
		0,
		func(entity ecs.Entity, material *asset.Material) {
			registry := ecs.MustResource[asset.MaterialRegistryResource](context.World).Registry
			materialID, ok := registry.IDFor(entity)
			if !ok {
				return
			}
			data := material.ToGPU()
			registry.Write(context.Queue, materialID, data)
		},
	)

	state.sortedHandles = state.sortedHandles[:0]
	for handle, bucket := range state.perHandle {
		if len(bucket.slotEntity) == 0 {
			continue
		}
		state.sortedHandles = append(state.sortedHandles, handle)
	}
	sort.Slice(state.sortedHandles, func(i, j int) bool {
		return state.sortedHandles[i] < state.sortedHandles[j]
	})

	assets := ecs.MustResource[asset.MeshAssetsResource](context.World).Assets
	camera, hasCamera := ecs.Resource[render.Camera](context.World)
	aspect = state.aspectFn()
	var viewProj mgl32.Mat4
	if hasCamera && camera != nil {
		viewProj = render.CameraViewProjection(camera, aspect)
	} else {
		viewProj = mgl32.Ident4()
	}
	screenW, screenH := float32(1920), float32(1080)
	if win, ok := ecs.Resource[window.Window](context.World); ok && win != nil {
		if win.Viewport.Width > 0 {
			screenW = float32(win.Viewport.Width)
		}
		if win.Viewport.Height > 0 {
			screenH = float32(win.Viewport.Height)
		}
	}
	cullSettings := render.DefaultGraphics().Cull
	if g, ok := ecs.Resource[render.Graphics](context.World); ok && g != nil {
		cullSettings = g.Cull
	}
	enabled := uint32(0)
	if cullSettings.Enabled {
		enabled = 1
	}
	fovY := float32(1.0)
	if hasCamera && camera != nil {
		fovY = camera.FovYRadians
	}
	projScaleY := 1.0 / float32(math.Tan(float64(fovY)*0.5))
	cullUniform := CullingUniforms{
		FrustumPlanes:      ExtractFrustumPlanes(viewProj),
		ViewProjection:     viewProj,
		ScreenSize:         mgl32.Vec2{screenW, screenH},
		MinScreenPixelSize: cullSettings.MinScreenPixelSize,
		ProjectionScaleY:   projScaleY,
		Enabled:            enabled,
	}
	writeBuffer(context.Device, context.Queue, context.Encoder, state.meshCulling.frameBuffer, 0, bytesOf(&cullUniform))

	for _, handle := range state.sortedHandles {
		bucket := state.perHandle[handle]
		entry, ok := assets.Lookup(handle)
		if !ok {
			continue
		}
		if err := ensureCullBindings(bucket, context.Device, state.meshCulling); err != nil {
			return err
		}
		bcenter := entry.Bounds.Center()
		bradius := entry.Bounds.Radius()
		params := bucketCullParams{
			BoundsCenter: mgl32.Vec3{bcenter[0], bcenter[1], bcenter[2]},
			BoundsRadius: bradius,
			ObjectCount:  uint32(len(bucket.slotEntity)),
		}
		writeBuffer(context.Device, context.Queue, context.Encoder, bucket.cullParamsBuffer, 0, bytesOf(&params))
		indirectTemplate := drawIndirectCommand{
			VertexCount:   entry.VertexCount,
			InstanceCount: 0,
			FirstVertex:   0,
			FirstInstance: 0,
		}
		writeBuffer(context.Device, context.Queue, context.Encoder, bucket.indirectBuffer, 0, bytesOf(&indirectTemplate))

		count := uint32(len(bucket.slotEntity))
		if count > 0 {
			identity := state.identityScratch[:0]
			for index := uint32(0); index < count; index = index + 1 {
				identity = append(identity, index)
			}
			state.identityScratch = identity
			writeBuffer(context.Device, context.Queue, context.Encoder, bucket.visibleIndicesBuffer, 0, bytesOfN(&identity[0], uint64(count)*4))
		}
	}

	dispatchCullPasses(state, context)
	return nil
}

func ensureCullBindings(bucket *handleInstances, device *wgpu.Device, culling *meshCullingPipeline) error {
	if bucket.indirectBuffer == nil || culling == nil {
		return nil
	}
	if bucket.cullParamsBuffer == nil {
		buf, err := device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: "mesh cull params",
			Size:  uint64(unsafe.Sizeof(bucketCullParams{})),
			Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
		})
		if err != nil {
			return fmt.Errorf("mesh pass: cull params: %w", err)
		}
		bucket.cullParamsBuffer = buf
	}
	if bucket.cullBindGroup == nil {
		bg, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  "mesh cull bind group",
			Layout: culling.bucketLayout,
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: bucket.cullParamsBuffer, Offset: 0, Size: uint64(unsafe.Sizeof(bucketCullParams{}))},
				{Binding: 1, Buffer: bucket.modelBuffer, Offset: 0, Size: wgpu.WholeSize},
				{Binding: 2, Buffer: bucket.indirectBuffer, Offset: 0, Size: drawIndirectCommandSize},
				{Binding: 3, Buffer: bucket.visibleIndicesBuffer, Offset: 0, Size: wgpu.WholeSize},
			},
		})
		if err != nil {
			return fmt.Errorf("mesh pass: cull bind group: %w", err)
		}
		bucket.cullBindGroup = bg
	}
	return nil
}

func meshExecute(s any, context *render.PassContext) error {
	state := s.(*meshPassState)

	dispatchClusterPasses(state, context)

	hasGeometry := len(state.sortedHandles) > 0
	assets := ecs.MustResource[asset.MeshAssetsResource](context.World).Assets

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

	if hasGeometry && meshRunPrepass && state.depthPrepass != nil {
		prepassDepth := depthAttachment
		prepass := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
			Label:                  "mesh depth prepass",
			ColorAttachments:       []wgpu.RenderPassColorAttachment{},
			DepthStencilAttachment: &prepassDepth,
		})
		prepass.SetPipeline(state.depthPrepass.pipeline)
		prepass.SetBindGroup(0, state.viewProjBindGroup, nil)
		prepass.SetBindGroup(2, state.depthPrepass.matsBg, nil)
		for _, handle := range state.sortedHandles {
			bucket := state.perHandle[handle]
			entry, ok := assets.Lookup(handle)
			if !ok {
				continue
			}
			prepass.SetBindGroup(1, bucket.bindGroup, nil)
			prepass.SetVertexBuffer(0, entry.Vertices, 0, wgpu.WholeSize)
			drawHandle(prepass, bucket, entry.VertexCount, uint32(len(bucket.slotEntity)))
		}
		prepass.End()
		prepass.Release()
		depthAttachment.DepthLoadOp = wgpu.LoadOpLoad
	}

	pass := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		Label:                  "mesh",
		ColorAttachments:       []wgpu.RenderPassColorAttachment{colorAttachment, entityIdAttachment, viewNormalsAttachment},
		DepthStencilAttachment: &depthAttachment,
	})
	if hasGeometry {
		pass.SetPipeline(state.pipeline)
		pass.SetBindGroup(0, state.viewProjBindGroup, nil)
		pass.SetBindGroup(1, state.globalBindGroup, nil)
		pass.SetBindGroup(3, state.iblBindGroup, nil)
		for _, handle := range state.sortedHandles {
			bucket := state.perHandle[handle]
			entry, ok := assets.Lookup(handle)
			if !ok {
				continue
			}
			pass.SetBindGroup(2, bucket.bindGroup, nil)
			pass.SetVertexBuffer(0, entry.Vertices, 0, wgpu.WholeSize)
			drawHandle(pass, bucket, entry.VertexCount, uint32(len(bucket.slotEntity)))
		}
	}
	pass.End()
	pass.Release()
	return nil
}

func meshRelease(s any) {
	state := s.(*meshPassState)
	for _, h := range state.perHandle {
		releaseHandleInstances(h)
	}
	if state.iblBindGroup != nil {
		state.iblBindGroup.Release()
	}
	if state.iblBgLayout != nil {
		state.iblBgLayout.Release()
	}
	if state.globalBindGroup != nil {
		state.globalBindGroup.Release()
	}
	if state.clusters != nil {
		state.clusters.release()
	}
	if state.meshCulling != nil {
		state.meshCulling.release()
	}
	sharedMeshPassState.Store((*meshPassState)(nil))
	if state.viewProjBindGroup != nil {
		state.viewProjBindGroup.Release()
	}
	if state.viewProjBuffer != nil {
		state.viewProjBuffer.Release()
	}
	if state.handleBgLayout != nil {
		state.handleBgLayout.Release()
	}
	if state.globalBgLayout != nil {
		state.globalBgLayout.Release()
	}
	if state.viewProjLayout != nil {
		state.viewProjLayout.Release()
	}
	if state.pipeline != nil {
		state.pipeline.Release()
	}
	if state.depthPrepass != nil {
		state.depthPrepass.release()
	}
}

func extractLights(world *ecs.World, scratch []LightGPU) []LightGPU {
	out := scratch
	out = out[:0]
	var spotShadowEntities [MaxSpotShadows]ecs.Entity
	var spotShadowCount uint32
	if resource, ok := ecs.Resource[SpotShadowResource](world); ok && resource.Shadow != nil {
		spotShadowCount = resource.Shadow.ActiveCount
		for index := uint32(0); index < spotShadowCount; index++ {
			spotShadowEntities[index] = resource.Shadow.slotEntity(index)
		}
	}
	var pointShadowEntities [MaxPointShadows]ecs.Entity
	var pointShadowCount uint32
	if resource, ok := ecs.Resource[PointShadowResource](world); ok && resource.Shadow != nil {
		pointShadowCount = resource.Shadow.ActiveCount
		for index := uint32(0); index < pointShadowCount; index++ {
			pointShadowEntities[index] = resource.Shadow.slotEntity(index)
		}
	}
	lightMask := ecs.MustMaskOf[render.Light](world)
	globalMask := ecs.MustMaskOf[transform.GlobalTransform](world)
	world.ForEach(lightMask|globalMask, 0, func(entity ecs.Entity, table *ecs.Archetype, index int) {
		if uint32(len(out)) >= MaxLightsBuffer {
			return
		}
		lights, _ := ecs.Column[render.Light](world, table)
		globals, _ := ecs.Column[transform.GlobalTransform](world, table)
		light := &lights[index]
		matrix := globals[index].Matrix
		shadowIndex := int32(-1)
		if light.Type == render.LightTypeSpot {
			for slot := uint32(0); slot < spotShadowCount; slot++ {
				if spotShadowEntities[slot] == entity {
					shadowIndex = int32(slot)
					break
				}
			}
		}
		if light.Type == render.LightTypePoint {
			for slot := uint32(0); slot < pointShadowCount; slot++ {
				if pointShadowEntities[slot] == entity {
					shadowIndex = int32(slot)
					break
				}
			}
		}
		out = append(out, LightGPU{
			Position:    [4]float32{matrix[12], matrix[13], matrix[14], 1.0},
			Direction:   [4]float32{-matrix[8], -matrix[9], -matrix[10], 0.0},
			Color:       [4]float32{light.Color[0] * light.Intensity, light.Color[1] * light.Intensity, light.Color[2] * light.Intensity, 1.0},
			LightType:   uint32(light.Type),
			Range:       light.Range,
			InnerCone:   cosOrZero(light.InnerConeAngle),
			OuterCone:   cosOrZero(light.OuterConeAngle),
			ShadowIndex: shadowIndex,
			LightSize:   0,
			CookieLayer: 0xFFFFFFFF,
			Padding:     0,
		})
	})
	sortDirectionalFirst(out)
	return out
}

func cosOrZero(angle float32) float32 {
	if angle <= 0 {
		return 0
	}
	return float32(math.Cos(float64(angle)))
}

func sortDirectionalFirst(lights []LightGPU) {
	left := 0
	for i := range lights {
		if lights[i].LightType == uint32(render.LightTypeDirectional) {
			if i != left {
				lights[left], lights[i] = lights[i], lights[left]
			}
			left++
		}
	}
}

func splitDirectionalAndLocal(lights []LightGPU) (directional, local uint32) {
	for _, l := range lights {
		if l.LightType == uint32(render.LightTypeDirectional) {
			directional++
		}
	}
	local = uint32(len(lights)) - directional
	return
}

func uploadLights(device *wgpu.Device, queue *wgpu.Queue, encoder *wgpu.CommandEncoder, buffer *wgpu.Buffer, lights []LightGPU) {
	if len(lights) == 0 {
		return
	}
	writeBuffer(device, queue, encoder, buffer, 0, lightsAsBytes(lights))
}

func lightsAsBytes(lights []LightGPU) []byte {
	if len(lights) == 0 {
		return nil
	}
	return sliceBytes(lights)
}

func buildClusterUniforms(camera *render.Camera, aspect float32, context *render.PassContext, numDirectional, numLocal uint32) ClusterUniforms {
	proj := render.CameraProjection(camera, aspect)
	invProj := proj.Inv()
	w, h := framebufferSize(context)
	screenW := float32(w)
	screenH := float32(h)
	if screenW <= 0 {
		screenW = 1
	}
	if screenH <= 0 {
		screenH = 1
	}
	tileX := screenW / float32(ClusterGridX)
	tileY := screenH / float32(ClusterGridY)
	zNear := camera.Near
	zFar := camera.Far
	if zNear <= 0 {
		zNear = 0.1
	}
	if zFar <= zNear {
		zFar = zNear + 1.0
	}
	var u ClusterUniforms
	copy(u.InverseProjection[:], invProj[:])
	u.ScreenSize = [2]float32{screenW, screenH}
	u.ZNear = zNear
	u.ZFar = zFar
	u.ClusterCount = [4]uint32{ClusterGridX, ClusterGridY, ClusterGridZ, 0}
	u.TileSize = [2]float32{tileX, tileY}
	u.NumLights = numDirectional + numLocal
	u.NumDirectionalLights = numDirectional
	u.CameraPosition = [4]float32{camera.Eye[0], camera.Eye[1], camera.Eye[2], 1}
	return u
}

func framebufferSize(context *render.PassContext) (uint32, uint32) {
	renderer := ecs.MustResource[render.RendererResource](context.World).Renderer
	return renderer.Config.Width, renderer.Config.Height
}

func dispatchClusterPasses(state *meshPassState, context *render.PassContext) {
	dispatchX := (ClusterGridX + 7) / 8
	dispatchY := (ClusterGridY + 7) / 8
	dispatchZ := ClusterGridZ

	boundsPass := context.Encoder.BeginComputePass(&wgpu.ComputePassDescriptor{})
	boundsPass.SetPipeline(state.clusters.boundsPipeline)
	boundsPass.SetBindGroup(0, state.clusters.boundsBindGroup, nil)
	boundsPass.DispatchWorkgroups(dispatchX, dispatchY, dispatchZ)
	boundsPass.End()
	boundsPass.Release()

	lightGridBytes := LightGridSize * uint64(TotalClusters)
	context.Encoder.CopyBufferToBuffer(state.clusters.lightGridReset, 0, state.clusters.lightGrid, 0, lightGridBytes)

	assignPass := context.Encoder.BeginComputePass(&wgpu.ComputePassDescriptor{})
	assignPass.SetPipeline(state.clusters.assignPipeline)
	assignPass.SetBindGroup(0, state.clusters.assignBindGroup, nil)
	assignPass.DispatchWorkgroups(dispatchX, dispatchY, dispatchZ)
	assignPass.End()
	assignPass.Release()
}
