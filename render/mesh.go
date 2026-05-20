package render

import (
	_ "embed"
	"fmt"
	"sort"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/go-gl/mathgl/mgl32"

	"rendergraph-go/ecs"
	"rendergraph-go/transform"
)

//go:embed mesh.wgsl
var meshShader string

// handleInstances owns the GPU and CPU bookkeeping for one mesh
// handle in the mesh pass. Each handle gets its own storage buffer
// and bind group so draws are flat `Draw(vertexCount, count, 0, 0)`
// against contiguous instances.
//
// entityToSlot + slotEntity form a bidirectional map: an entity stays
// at the same slot for its whole lifetime in this handle, so the
// mesh pass can sparse-update the GPU buffer at known offsets.
type handleInstances struct {
	buffer    *wgpu.Buffer
	bindGroup *wgpu.BindGroup
	capacity  uint32

	entityToSlot map[ecs.Entity]uint32
	slotEntity   []ecs.Entity
}

func newHandleInstances() *handleInstances {
	return &handleInstances{
		entityToSlot: make(map[ecs.Entity]uint32, 16),
	}
}

func (h *handleInstances) count() uint32 { return uint32(len(h.slotEntity)) }

func (h *handleInstances) release() {
	if h.bindGroup != nil {
		h.bindGroup.Release()
		h.bindGroup = nil
	}
	if h.buffer != nil {
		h.buffer.Release()
		h.buffer = nil
	}
}

// meshPassState is the long-lived state the mesh pass keeps. Every
// scratch slice is here and reset between frames.
//
// entityHandle is a top-level reverse index from entity to the
// MeshHandle whose slot table holds it. Despawn events arrive carrying
// only an Entity; this map turns that into O(1) lookup to the right
// bucket so the dead-entity sweep is O(despawned), not O(slots).
type meshPassState struct {
	pipeline       *wgpu.RenderPipeline
	modelBgLayout  *wgpu.BindGroupLayout
	viewProjLayout *wgpu.BindGroupLayout

	viewProjBuffer    *wgpu.Buffer
	viewProjBindGroup *wgpu.BindGroup

	perHandle     map[MeshHandle]*handleInstances
	entityHandle  map[ecs.Entity]MeshHandle
	sortedHandles []MeshHandle

	aspectFn func() float32
}

// NewMeshPass builds the engine's instanced mesh pass.
//
// View × projection lives in a small uniform updated every frame; per-
// entity model matrices live in per-handle storage buffers, sparse-
// updated via [ecs.IterChanged] on [transform.GlobalTransform]. Each
// entity gets a stable slot in its handle's buffer so the GPU side
// only writes the matrices that changed this frame.
func NewMeshPass(device *wgpu.Device, surfaceFormat wgpu.TextureFormat, aspect func() float32) (*Pass, error) {
	state := &meshPassState{
		perHandle:    make(map[MeshHandle]*handleInstances, 4),
		entityHandle: make(map[ecs.Entity]MeshHandle, 64),
		aspectFn:     aspect,
	}

	viewProjLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "mesh view_proj bind group layout",
		Entries: []wgpu.BindGroupLayoutEntry{{
			Binding:    0,
			Visibility: wgpu.ShaderStageVertex,
			Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform},
		}},
	})
	if err != nil {
		return nil, fmt.Errorf("mesh pass: view_proj bind group layout: %w", err)
	}
	state.viewProjLayout = viewProjLayout

	modelBgLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "mesh model bind group layout",
		Entries: []wgpu.BindGroupLayoutEntry{{
			Binding:    0,
			Visibility: wgpu.ShaderStageVertex,
			Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage},
		}},
	})
	if err != nil {
		return nil, fmt.Errorf("mesh pass: model bind group layout: %w", err)
	}
	state.modelBgLayout = modelBgLayout

	viewProjBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "mesh view_proj buffer",
		Size:  uint64(unsafe.Sizeof(mgl32.Mat4{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("mesh pass: view_proj buffer: %w", err)
	}
	state.viewProjBuffer = viewProjBuffer

	viewProjBindGroup, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "mesh view_proj bind group",
		Layout: viewProjLayout,
		Entries: []wgpu.BindGroupEntry{{
			Binding: 0,
			Buffer:  viewProjBuffer,
			Offset:  0,
			Size:    uint64(unsafe.Sizeof(mgl32.Mat4{})),
		}},
	})
	if err != nil {
		return nil, fmt.Errorf("mesh pass: view_proj bind group: %w", err)
	}
	state.viewProjBindGroup = viewProjBindGroup

	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "mesh shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: meshShader},
	})
	if err != nil {
		return nil, fmt.Errorf("mesh pass: shader: %w", err)
	}
	defer shader.Release()

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "mesh pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{viewProjLayout, modelBgLayout},
	})
	if err != nil {
		return nil, fmt.Errorf("mesh pass: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	pipeline, err := device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "mesh pipeline",
		Layout: pipelineLayout,
		Vertex: wgpu.VertexState{
			Module:     shader,
			EntryPoint: "vertex_main",
			Buffers: []wgpu.VertexBufferLayout{{
				ArrayStride: uint64(unsafe.Sizeof(MeshVertex{})),
				StepMode:    wgpu.VertexStepModeVertex,
				Attributes: []wgpu.VertexAttribute{
					{Format: wgpu.VertexFormatFloat32x4, Offset: 0, ShaderLocation: 0},
					{Format: wgpu.VertexFormatFloat32x4, Offset: 16, ShaderLocation: 1},
				},
			}},
		},
		Primitive: wgpu.PrimitiveState{
			Topology:  wgpu.PrimitiveTopologyTriangleList,
			FrontFace: wgpu.FrontFaceCCW,
			CullMode:  wgpu.CullModeNone,
		},
		DepthStencil: &wgpu.DepthStencilState{
			Format:            DepthFormat,
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
		Fragment: &wgpu.FragmentState{
			Module:     shader,
			EntryPoint: "fragment_main",
			Targets: []wgpu.ColorTargetState{{
				Format:    surfaceFormat,
				WriteMask: wgpu.ColorWriteMaskAll,
			}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("mesh pass: pipeline: %w", err)
	}
	state.pipeline = pipeline

	return &Pass{
		Name:    "mesh",
		Writes:  []string{"color", "depth"},
		State:   state,
		Prepare: meshPrepare,
		Execute: meshExecute,
		Release: meshRelease,
	}, nil
}

// meshPrepare runs every frame:
//  1. Write the camera's view × projection into the uniform.
//  2. Free slots whose entities were despawned (driven by the world's
//     [ecs.EntityDespawned] events; events stay readable for two frames,
//     so the pass need only run at least once per pair of frames to
//     avoid stale slots).
//  3. Allocate slots for newly-spawned entities (full upload at slot).
//  4. Sparse-upload only the entities whose GlobalTransform was
//     stamped this tick.
//  5. Rebuild the sorted handle list so the draw order is stable.
func meshPrepare(s any, context *PassContext) error {
	state := s.(*meshPassState)

	camera := ecs.Resource[Camera](context.World)
	viewProjection := camera.ViewProjection(state.aspectFn())
	context.Queue.WriteBuffer(state.viewProjBuffer, 0, bytesOf(&viewProjection))

	for _, event := range ecs.ReadEvents[ecs.EntityDespawned](context.World) {
		releaseEntitySlot(state, context, event.Entity)
	}

	renderMeshMask := ecs.MaskOf[RenderMesh](context.World)
	globalMask := ecs.MaskOf[transform.GlobalTransform](context.World)
	visibleMask := renderMeshMask | globalMask

	context.World.ForEach(visibleMask, 0, func(entity ecs.Entity, table *ecs.Archetype, index int) {
		meshes, _ := ecs.Column[RenderMesh](context.World, table)
		globals, _ := ecs.Column[transform.GlobalTransform](context.World, table)
		handle := meshes[index].Mesh

		bucket, ok := state.perHandle[handle]
		if !ok {
			bucket = newHandleInstances()
			state.perHandle[handle] = bucket
		}

		if _, already := bucket.entityToSlot[entity]; already {
			return
		}

		slot := uint32(len(bucket.slotEntity))
		bucket.entityToSlot[entity] = slot
		bucket.slotEntity = append(bucket.slotEntity, entity)
		state.entityHandle[entity] = handle
		if err := ensureHandleCapacity(bucket, context.Device, state.modelBgLayout); err != nil {
			return
		}
		matrix := globals[index].Matrix
		context.Queue.WriteBuffer(bucket.buffer, uint64(slot)*matrixSize, bytesOf(&matrix))
	})

	ecs.IterChanged1[transform.GlobalTransform](
		context.World,
		renderMeshMask,
		0,
		func(entity ecs.Entity, global *transform.GlobalTransform) {
			mesh, ok := ecs.Get[RenderMesh](context.World, entity)
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
			context.Queue.WriteBuffer(bucket.buffer, uint64(slot)*matrixSize, bytesOf(&global.Matrix))
		},
	)

	state.sortedHandles = state.sortedHandles[:0]
	for handle, bucket := range state.perHandle {
		if bucket.count() == 0 {
			continue
		}
		state.sortedHandles = append(state.sortedHandles, handle)
	}
	sort.Slice(state.sortedHandles, func(i, j int) bool {
		return state.sortedHandles[i] < state.sortedHandles[j]
	})

	return nil
}

func meshExecute(s any, context *PassContext) error {
	state := s.(*meshPassState)

	if len(state.sortedHandles) == 0 {
		return nil
	}

	assets := ecs.Resource[MeshAssetsResource](context.World).Assets

	colorAttachment, err := context.ColorAttachment("color")
	if err != nil {
		return err
	}
	depthAttachment, err := context.DepthAttachment("depth")
	if err != nil {
		return err
	}

	pass := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		Label:                  "mesh",
		ColorAttachments:       []wgpu.RenderPassColorAttachment{colorAttachment},
		DepthStencilAttachment: &depthAttachment,
	})
	pass.SetPipeline(state.pipeline)
	pass.SetBindGroup(0, state.viewProjBindGroup, nil)

	for _, handle := range state.sortedHandles {
		bucket := state.perHandle[handle]
		entry, ok := assets.Lookup(handle)
		if !ok {
			continue
		}
		pass.SetBindGroup(1, bucket.bindGroup, nil)
		pass.SetVertexBuffer(0, entry.Vertices, 0, wgpu.WholeSize)
		pass.Draw(entry.VertexCount, bucket.count(), 0, 0)
	}

	pass.End()
	pass.Release()
	return nil
}

func meshRelease(s any) {
	state := s.(*meshPassState)
	for _, h := range state.perHandle {
		h.release()
	}
	if state.viewProjBindGroup != nil {
		state.viewProjBindGroup.Release()
	}
	if state.viewProjBuffer != nil {
		state.viewProjBuffer.Release()
	}
	if state.modelBgLayout != nil {
		state.modelBgLayout.Release()
	}
	if state.viewProjLayout != nil {
		state.viewProjLayout.Release()
	}
	if state.pipeline != nil {
		state.pipeline.Release()
	}
}

// releaseEntitySlot is the despawn-event handler. Looks up the
// entity's handle via the top-level reverse index, swap-removes it
// from that handle's slot table, and rewrites the moved tail entity's
// GPU matrix at its new slot offset so subsequent draws don't read
// stale data from the freed position.
func releaseEntitySlot(state *meshPassState, context *PassContext, entity ecs.Entity) {
	handle, ok := state.entityHandle[entity]
	if !ok {
		return
	}
	delete(state.entityHandle, entity)
	bucket, ok := state.perHandle[handle]
	if !ok {
		return
	}
	slot, ok := bucket.entityToSlot[entity]
	if !ok {
		return
	}
	last := uint32(len(bucket.slotEntity) - 1)
	if slot != last {
		moved := bucket.slotEntity[last]
		bucket.slotEntity[slot] = moved
		bucket.entityToSlot[moved] = slot
		if global, ok := ecs.Get[transform.GlobalTransform](context.World, moved); ok {
			context.Queue.WriteBuffer(bucket.buffer, uint64(slot)*matrixSize, bytesOf(&global.Matrix))
		}
	}
	bucket.slotEntity = bucket.slotEntity[:last]
	delete(bucket.entityToSlot, entity)
}

// matrixSize is the byte size of a single mat4. Used for the GPU
// offset arithmetic in sparse uploads.
const matrixSize uint64 = uint64(unsafe.Sizeof(mgl32.Mat4{}))

// minHandleCapacity is the starting capacity for a handle's instance
// buffer. The buffer doubles on growth.
const minHandleCapacity uint32 = 64

// ensureHandleCapacity grows the handle's storage buffer (and rebuilds
// its bind group) if the current slot count exceeded the buffer's
// capacity. Existing matrix uploads are preserved — when the buffer
// grows, the next frame's IterChanged pass refreshes every entity that
// actually changed, and slots that didn't change need their content
// reuploaded.
func ensureHandleCapacity(h *handleInstances, device *wgpu.Device, layout *wgpu.BindGroupLayout) error {
	required := h.count()
	if h.capacity >= required && h.buffer != nil {
		return nil
	}
	newCapacity := h.capacity
	if newCapacity == 0 {
		newCapacity = minHandleCapacity
	}
	for newCapacity < required {
		newCapacity *= 2
	}
	if newCapacity < minHandleCapacity {
		newCapacity = minHandleCapacity
	}

	buffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "mesh model buffer",
		Size:  uint64(newCapacity) * matrixSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return fmt.Errorf("mesh pass: model buffer: %w", err)
	}
	bindGroup, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "mesh model bind group",
		Layout: layout,
		Entries: []wgpu.BindGroupEntry{{
			Binding: 0,
			Buffer:  buffer,
			Offset:  0,
			Size:    wgpu.WholeSize,
		}},
	})
	if err != nil {
		buffer.Release()
		return fmt.Errorf("mesh pass: model bind group: %w", err)
	}
	if h.bindGroup != nil {
		h.bindGroup.Release()
	}
	if h.buffer != nil {
		h.buffer.Release()
	}
	h.buffer = buffer
	h.bindGroup = bindGroup
	h.capacity = newCapacity
	return nil
}
