package asset

import (
	"fmt"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"

	"indigo/ecs"
)

// MaterialRegistry owns the engine-wide GPU buffer of MaterialGPU
// entries. Instead of every mesh handle carrying its own per-slot
// material storage, every material in the engine lives in this
// single buffer indexed by a u32 id. Per-instance mesh data carries
// the id; the mesh shader does materials[material_id[instance]].
//
// This is the foundation for material-sort batching and for sharing
// one Material across many entities without duplicating GPU bytes.
type MaterialRegistry struct {
	device   *wgpu.Device
	buffer   *wgpu.Buffer
	capacity uint32
	count    uint32

	entityToID map[ecs.Entity]uint32
	idToEntity []ecs.Entity
}

// MaterialRegistryResource wraps the registry as a typed ECS
// resource so passes can look it up via [ecs.MustResource].
type MaterialRegistryResource struct {
	Registry *MaterialRegistry
}

// minMaterialCapacity is the initial entry count; the buffer doubles
// on growth and never shrinks.
const minMaterialCapacity uint32 = 256

// NewMaterialRegistry allocates the initial GPU buffer.
func NewMaterialRegistry(device *wgpu.Device) (*MaterialRegistry, error) {
	registry := &MaterialRegistry{
		device:     device,
		capacity:   minMaterialCapacity,
		entityToID: make(map[ecs.Entity]uint32),
	}
	buffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "global material registry",
		Size:  uint64(registry.capacity) * MaterialGPUSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return nil, fmt.Errorf("material registry: buffer: %w", err)
	}
	registry.buffer = buffer
	return registry, nil
}

// Buffer returns the underlying storage buffer for binding.
func (r *MaterialRegistry) Buffer() *wgpu.Buffer { return r.buffer }

// AssignID returns the existing id for the entity or allocates one.
// Triggers GPU buffer growth + bind-group invalidation in the caller
// when capacity is exceeded.
func (r *MaterialRegistry) AssignID(entity ecs.Entity) (uint32, bool) {
	if id, ok := r.entityToID[entity]; ok {
		return id, false
	}
	id := r.count
	r.entityToID[entity] = id
	r.idToEntity = append(r.idToEntity, entity)
	r.count++
	return id, true
}

// Release releases the entity's id so future entities can reuse the
// slot via swap-remove. The buffer entry stays until the next entity
// claims that id (next AssignID).
func (r *MaterialRegistry) Release(entity ecs.Entity) {
	id, ok := r.entityToID[entity]
	if !ok {
		return
	}
	delete(r.entityToID, entity)
	last := r.count - 1
	if id != last {
		moved := r.idToEntity[last]
		r.idToEntity[id] = moved
		r.entityToID[moved] = id
	}
	r.idToEntity = r.idToEntity[:last]
	r.count = last
}

// IDFor returns the assigned id for the entity, or 0/false if it has
// no entry. Callers that need an id and don't have one must use
// [AssignID].
func (r *MaterialRegistry) IDFor(entity ecs.Entity) (uint32, bool) {
	id, ok := r.entityToID[entity]
	return id, ok
}

// EntityFor is the inverse of IDFor, used by debug tooling.
func (r *MaterialRegistry) EntityFor(id uint32) (ecs.Entity, bool) {
	if id >= r.count {
		return ecs.Entity{}, false
	}
	return r.idToEntity[id], true
}

// Count is the number of registered entries; the buffer capacity is
// always >= Count.
func (r *MaterialRegistry) Count() uint32 { return r.count }

// Capacity is the current buffer capacity in entries.
func (r *MaterialRegistry) Capacity() uint32 { return r.capacity }

// EnsureCapacity grows the GPU buffer to fit required entries.
// Returns true if the buffer was reallocated and any cached bind
// groups must invalidate.
func (r *MaterialRegistry) EnsureCapacity(required uint32) (bool, error) {
	if r.capacity >= required {
		return false, nil
	}
	newCapacity := r.capacity
	if newCapacity == 0 {
		newCapacity = minMaterialCapacity
	}
	for newCapacity < required {
		newCapacity *= 2
	}
	newBuffer, err := r.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "global material registry",
		Size:  uint64(newCapacity) * MaterialGPUSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return false, fmt.Errorf("material registry: grow: %w", err)
	}
	if r.buffer != nil {
		r.buffer.Release()
	}
	r.buffer = newBuffer
	r.capacity = newCapacity
	return true, nil
}

// Write uploads gpu into slot id of the registry buffer.
func (r *MaterialRegistry) Write(queue *wgpu.Queue, id uint32, gpu MaterialGPU) {
	queue.WriteBuffer(r.buffer, uint64(id)*MaterialGPUSize, unsafe.Slice((*byte)(unsafe.Pointer(&gpu)), MaterialGPUSize))
}

// Release frees the registry's GPU resources.
func (r *MaterialRegistry) ReleaseResources() {
	if r.buffer != nil {
		r.buffer.Release()
		r.buffer = nil
	}
}
