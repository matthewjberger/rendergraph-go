package pass

import (
	"math"
	"sort"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"

	"indigo/ecs"
	"indigo/render"
	"indigo/render/asset"
	"indigo/transform"
)

// meshPrepare runs every frame:
//  1. Write camera view * projection into the uniform.
//  2. Drain EntityDespawned events to free per-handle slots.
//  3. Allocate slots for entities whose RenderMesh changed.
//  4. Sparse-upload only the entities whose GlobalTransform or
//     Material was stamped this tick.
//  5. Rebuild the sorted handle list so the draw order is stable.
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

	return nil
}

// meshExecute runs the two cluster compute dispatches and then
// the actual mesh draw inside a single render pass. Compute
// dispatches happen before BeginRenderPass because a compute pass
// and a render pass can't share an encoder.
func meshExecute(s any, context *render.PassContext) error {
	state := s.(*meshPassState)

	dispatchClusterPasses(state, context)

	if len(state.sortedHandles) == 0 {
		return nil
	}

	assets := ecs.MustResource[asset.MeshAssetsResource](context.World).Assets

	colorAttachment, err := context.ColorAttachment("color")
	if err != nil {
		return err
	}
	entityIdAttachment, err := context.ColorAttachment("entity_id")
	if err != nil {
		return err
	}
	depthAttachment, err := context.DepthAttachment("depth")
	if err != nil {
		return err
	}

	pass := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		Label:                  "mesh",
		ColorAttachments:       []wgpu.RenderPassColorAttachment{colorAttachment, entityIdAttachment},
		DepthStencilAttachment: &depthAttachment,
	})
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
		pass.Draw(entry.VertexCount, uint32(len(bucket.slotEntity)), 0, 0)
	}

	pass.End()
	pass.Release()
	return nil
}

// meshRelease tears down every GPU resource the mesh pass owns
// when the renderer shuts down.
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
}

// extractLights walks the engine world for entities with both
// [render.Light] and [transform.GlobalTransform], packs them into
// the [LightGPU] layout the cluster compute + mesh shaders share,
// and returns directional lights first followed by local
// (point/spot) lights. The cluster_light_assign pass culls only
// the local-light suffix; directional lights are iterated by
// every fragment regardless of position.
//
// Color is premultiplied by intensity so the shader doesn't have
// to multiply at sample time.
func extractLights(world *ecs.World, scratch []LightGPU) []LightGPU {
	out := scratch
	out = out[:0]
	lightMask := ecs.MustMaskOf[render.Light](world)
	globalMask := ecs.MustMaskOf[transform.GlobalTransform](world)
	world.ForEach(lightMask|globalMask, 0, func(_ ecs.Entity, table *ecs.Archetype, index int) {
		if uint32(len(out)) >= MaxLightsBuffer {
			return
		}
		lights, _ := ecs.Column[render.Light](world, table)
		globals, _ := ecs.Column[transform.GlobalTransform](world, table)
		light := &lights[index]
		matrix := globals[index].Matrix
		out = append(out, LightGPU{
			Position:    [4]float32{matrix[12], matrix[13], matrix[14], 1.0},
			Direction:   [4]float32{-matrix[8], -matrix[9], -matrix[10], 0.0},
			Color:       [4]float32{light.Color[0] * light.Intensity, light.Color[1] * light.Intensity, light.Color[2] * light.Intensity, 1.0},
			LightType:   uint32(light.Type),
			Range:       light.Range,
			InnerCone:   cosOrZero(light.InnerConeAngle),
			OuterCone:   cosOrZero(light.OuterConeAngle),
			ShadowIndex: -1,
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

// sortDirectionalFirst stable-partitions the slice so every
// directional light precedes every point/spot light. The cluster
// shader assumes this layout (num_directional_lights is the
// boundary index into the same buffer).
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
	return unsafe.Slice((*byte)(unsafe.Pointer(&lights[0])), int(LightGPUSize)*len(lights))
}

// buildClusterUniforms snapshots the camera into the WGSL
// ClusterUniforms layout. The compute and fragment shaders read
// this every frame to size tiles and clamp depth slices.
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

// framebufferSize returns the renderer's current swapchain
// dimensions so the cluster grid tracks viewport resizes.
func framebufferSize(context *render.PassContext) (uint32, uint32) {
	renderer := ecs.MustResource[render.RendererResource](context.World).Renderer
	return renderer.Config.Width, renderer.Config.Height
}

// dispatchClusterPasses runs the two compute pipelines that
// rebuild the cluster grid and assign lights to each cluster.
// Called from [meshExecute] BEFORE BeginRenderPass — compute
// dispatches can't share an encoder pass with a render pass.
//
//  1. cluster_bounds: writes per-cluster view-space AABBs.
//     Re-dispatched every frame for simplicity; could be gated
//     on a "camera changed" flag later, but the cost of running
//     8x8x24 = 1536 invocations is trivial.
//  2. copy lightGridReset -> lightGrid (zeros every count).
//  3. cluster_light_assign: per cluster, tests every local light
//     against the AABB and writes intersecting indices to
//     light_indices + light_grid[cluster].count.
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
