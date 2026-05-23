package pass

import (
	"fmt"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/go-gl/mathgl/mgl32"

	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/render/asset"
	"github.com/matthewjberger/indigo/transform"
)

type handleInstances struct {
	modelBuffer          *wgpu.Buffer
	materialIndexBuffer  *wgpu.Buffer
	entityIdBuffer       *wgpu.Buffer
	visibleIndicesBuffer *wgpu.Buffer
	indirectBuffer       *wgpu.Buffer
	cullParamsBuffer     *wgpu.Buffer
	cullBindGroup        *wgpu.BindGroup
	bindGroup            *wgpu.BindGroup
	shadowBindGroup      *wgpu.BindGroup
	capacity             uint32

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
	if h.materialIndexBuffer != nil {
		h.materialIndexBuffer.Release()
		h.materialIndexBuffer = nil
	}
	if h.entityIdBuffer != nil {
		h.entityIdBuffer.Release()
		h.entityIdBuffer = nil
	}
	if h.indirectBuffer != nil {
		h.indirectBuffer.Release()
		h.indirectBuffer = nil
	}
	if h.visibleIndicesBuffer != nil {
		h.visibleIndicesBuffer.Release()
		h.visibleIndicesBuffer = nil
	}
	if h.cullBindGroup != nil {
		h.cullBindGroup.Release()
		h.cullBindGroup = nil
	}
	if h.cullParamsBuffer != nil {
		h.cullParamsBuffer.Release()
		h.cullParamsBuffer = nil
	}
	if h.shadowBindGroup != nil {
		h.shadowBindGroup.Release()
		h.shadowBindGroup = nil
	}
}

type drawIndirectCommand struct {
	VertexCount   uint32
	InstanceCount uint32
	FirstVertex   uint32
	FirstInstance uint32
}

const drawIndirectCommandSize = uint64(unsafe.Sizeof(drawIndirectCommand{}))

const matrixSize uint64 = uint64(unsafe.Sizeof(mgl32.Mat4{}))

const minHandleCapacity uint32 = 64

func ensureHandleCapacity(h *handleInstances, device *wgpu.Device, layout *wgpu.BindGroupLayout) error {
	required := uint32(len(h.slotEntity))
	if h.capacity >= required && h.modelBuffer != nil && h.materialIndexBuffer != nil && h.entityIdBuffer != nil && h.visibleIndicesBuffer != nil {
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
	materialIndexBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "mesh material_index buffer",
		Size:  uint64(newCapacity) * 4,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		modelBuffer.Release()
		return fmt.Errorf("mesh pass: material_index buffer: %w", err)
	}
	entityIdBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "mesh entity_id buffer",
		Size:  uint64(newCapacity) * 4,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		modelBuffer.Release()
		materialIndexBuffer.Release()
		return fmt.Errorf("mesh pass: entity_id buffer: %w", err)
	}
	visibleIndicesBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "mesh visible_indices buffer",
		Size:  uint64(newCapacity) * 4,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		modelBuffer.Release()
		materialIndexBuffer.Release()
		entityIdBuffer.Release()
		return fmt.Errorf("mesh pass: visible_indices buffer: %w", err)
	}
	bindGroup, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "mesh per-handle bind group",
		Layout: layout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: modelBuffer, Offset: 0, Size: wgpu.WholeSize},
			{Binding: 1, Buffer: materialIndexBuffer, Offset: 0, Size: wgpu.WholeSize},
			{Binding: 2, Buffer: entityIdBuffer, Offset: 0, Size: wgpu.WholeSize},
			{Binding: 3, Buffer: visibleIndicesBuffer, Offset: 0, Size: wgpu.WholeSize},
		},
	})
	if err != nil {
		modelBuffer.Release()
		materialIndexBuffer.Release()
		entityIdBuffer.Release()
		visibleIndicesBuffer.Release()
		return fmt.Errorf("mesh pass: per-handle bind group: %w", err)
	}
	if h.bindGroup != nil {
		h.bindGroup.Release()
	}
	if h.modelBuffer != nil {
		h.modelBuffer.Release()
	}
	if h.materialIndexBuffer != nil {
		h.materialIndexBuffer.Release()
	}
	if h.entityIdBuffer != nil {
		h.entityIdBuffer.Release()
	}
	if h.visibleIndicesBuffer != nil {
		h.visibleIndicesBuffer.Release()
	}
	if h.shadowBindGroup != nil {
		h.shadowBindGroup.Release()
		h.shadowBindGroup = nil
	}
	if h.cullBindGroup != nil {
		h.cullBindGroup.Release()
		h.cullBindGroup = nil
	}
	h.modelBuffer = modelBuffer
	h.materialIndexBuffer = materialIndexBuffer
	h.entityIdBuffer = entityIdBuffer
	h.visibleIndicesBuffer = visibleIndicesBuffer
	h.bindGroup = bindGroup
	h.capacity = newCapacity

	if h.indirectBuffer == nil {
		indirectBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: "mesh indirect buffer",
			Size:  drawIndirectCommandSize,
			Usage: wgpu.BufferUsageIndirect | wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
		})
		if err != nil {
			return fmt.Errorf("mesh pass: indirect buffer: %w", err)
		}
		h.indirectBuffer = indirectBuffer
	}
	return nil
}

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
		registry := ecs.MustResource[asset.MaterialRegistryResource](context.World).Registry
		materialID, _ := registry.IDFor(moved)
		writeBuffer(context.Device, context.Queue, context.Encoder, bucket.materialIndexBuffer, uint64(slot)*4, bytesOf(&materialID))

		movedID := moved.ID
		writeBuffer(context.Device, context.Queue, context.Encoder, bucket.entityIdBuffer, uint64(slot)*4, bytesOf(&movedID))
	}
	bucket.slotEntity = bucket.slotEntity[:last]
	delete(bucket.entityToSlot, entity)
}
