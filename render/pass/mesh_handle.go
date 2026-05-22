package pass

import (
	"fmt"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/go-gl/mathgl/mgl32"

	"indigo/ecs"
	"indigo/render"
	"indigo/render/asset"
	"indigo/transform"
)

// handleInstances owns the GPU and CPU bookkeeping for one mesh
// handle in the mesh pass. Each handle holds three storage buffers
// indexed by per-instance slot: world matrices, MaterialGPU
// entries, and entity IDs. The same slot stays with an entity for
// its whole lifetime so sparse uploads can write to known offsets.
type handleInstances struct {
	modelBuffer    *wgpu.Buffer
	materialBuffer *wgpu.Buffer
	entityIdBuffer *wgpu.Buffer
	bindGroup      *wgpu.BindGroup
	capacity       uint32

	entityToSlot map[ecs.Entity]uint32
	slotEntity   []ecs.Entity
}

func releaseHandleInstances(h *handleInstances) {
	if h.bindGroup != nil {
		h.bindGroup.Release()
		h.bindGroup = nil
	}
	if h.modelBuffer != nil {
		h.modelBuffer.Release()
		h.modelBuffer = nil
	}
	if h.materialBuffer != nil {
		h.materialBuffer.Release()
		h.materialBuffer = nil
	}
	if h.entityIdBuffer != nil {
		h.entityIdBuffer.Release()
		h.entityIdBuffer = nil
	}
}

// matrixSize is the byte size of a single mat4. Used for offset
// arithmetic in sparse uploads.
const matrixSize uint64 = uint64(unsafe.Sizeof(mgl32.Mat4{}))

// minHandleCapacity is the starting capacity for a handle's
// instance buffers. The buffers double on growth.
const minHandleCapacity uint32 = 64

// ensureHandleCapacity grows the handle's three storage buffers
// and rebuilds its bind group when the slot count exceeds the
// current capacity. Existing contents aren't preserved on grow —
// subsequent IterChanged passes refresh whatever changed this
// frame and the slot-stable layout ensures other entries get
// reuploaded next time their components stamp.
func ensureHandleCapacity(h *handleInstances, device *wgpu.Device, layout *wgpu.BindGroupLayout) error {
	required := uint32(len(h.slotEntity))
	if h.capacity >= required && h.modelBuffer != nil && h.materialBuffer != nil && h.entityIdBuffer != nil {
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

	modelBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "mesh model buffer",
		Size:  uint64(newCapacity) * matrixSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return fmt.Errorf("mesh pass: model buffer: %w", err)
	}
	materialBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "mesh material buffer",
		Size:  uint64(newCapacity) * asset.MaterialGPUSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		modelBuffer.Release()
		return fmt.Errorf("mesh pass: material buffer: %w", err)
	}
	entityIdBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "mesh entity_id buffer",
		Size:  uint64(newCapacity) * 4,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		modelBuffer.Release()
		materialBuffer.Release()
		return fmt.Errorf("mesh pass: entity_id buffer: %w", err)
	}
	bindGroup, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "mesh per-handle bind group",
		Layout: layout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: modelBuffer, Offset: 0, Size: wgpu.WholeSize},
			{Binding: 1, Buffer: materialBuffer, Offset: 0, Size: wgpu.WholeSize},
			{Binding: 2, Buffer: entityIdBuffer, Offset: 0, Size: wgpu.WholeSize},
		},
	})
	if err != nil {
		modelBuffer.Release()
		materialBuffer.Release()
		entityIdBuffer.Release()
		return fmt.Errorf("mesh pass: per-handle bind group: %w", err)
	}
	if h.bindGroup != nil {
		h.bindGroup.Release()
	}
	if h.modelBuffer != nil {
		h.modelBuffer.Release()
	}
	if h.materialBuffer != nil {
		h.materialBuffer.Release()
	}
	if h.entityIdBuffer != nil {
		h.entityIdBuffer.Release()
	}
	h.modelBuffer = modelBuffer
	h.materialBuffer = materialBuffer
	h.entityIdBuffer = entityIdBuffer
	h.bindGroup = bindGroup
	h.capacity = newCapacity
	return nil
}

// releaseEntitySlot is the despawn handler: swap-remove the
// entity's slot in its handle, then rewrite the moved tail
// entity's data at its new slot so subsequent draws don't read
// stale matrices / materials.
func releaseEntitySlot(state *meshPassState, context *render.PassContext, entity ecs.Entity) {
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
			writeBuffer(context.Device, context.Queue, context.Encoder, bucket.modelBuffer, uint64(slot)*matrixSize, bytesOf(&global.Matrix))
		}
		material := asset.DefaultMaterial().ToGPU()
		if mat, ok := ecs.Get[asset.Material](context.World, moved); ok {
			material = mat.ToGPU()
		}
		writeBuffer(context.Device, context.Queue, context.Encoder, bucket.materialBuffer, uint64(slot)*asset.MaterialGPUSize, bytesOf(&material))

		movedID := moved.ID
		writeBuffer(context.Device, context.Queue, context.Encoder, bucket.entityIdBuffer, uint64(slot)*4, bytesOf(&movedID))
	}
	bucket.slotEntity = bucket.slotEntity[:last]
	delete(bucket.entityToSlot, entity)
}
