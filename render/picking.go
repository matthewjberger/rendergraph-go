package render

import (
	"errors"
	"fmt"
	"sync"

	"github.com/cogentcore/webgpu/wgpu"

	"indigo/ecs"
)

// PickRequest is what callers fill in to ask "which entity is under
// these screen pixels?" Screen coordinates are integer pixel offsets
// from the top-left of the surface; matches the coordinate space
// [Input.MousePosition] reports.
type PickRequest struct {
	X uint32
	Y uint32
}

// PickResult is what the picking system publishes once the GPU has
// finished writing the entity_id buffer and the readback has been
// mapped back to CPU memory. EntityID is the [ecs.Entity.ID] under
// the cursor, or 0 if nothing was. Request echoes the originating
// [PickRequest] so callers can match up multiple queued picks.
type PickResult struct {
	EntityID uint32
	Request  PickRequest
}

// Picking is the engine-side state shared between the picking pass
// and the readback system. Apps install it as a resource on the
// engine world, set Pending to enqueue a pick, and read Result on a
// later frame.
//
// The readback is async: on the frame a pick is queued, the picking
// pass records a 1-pixel CopyTextureToBuffer; once the encoder is
// submitted, [ProcessPickingReadback] issues a single MapAsync; the
// result becomes available the moment the GPU finishes the work.
// On native that's typically the same frame (device.Poll drives the
// callback). On wasm the browser's event loop processes the promise
// when it can, so the result usually arrives one frame later.
type Picking struct {
	Pending *PickRequest
	Result  *PickResult

	// In-flight state used by the readback system. Apps don't read
	// these.
	staging     *wgpu.Buffer
	requested   PickRequest
	hasInFlight bool
	mapping     bool
	mu          sync.Mutex
	mapped      bool
	mapErr      error
}

// NewPicking returns an empty Picking resource. Install it on the
// engine world before adding the picking pass.
func NewPicking() *Picking {
	return &Picking{}
}

// QueuePick records a pick request to fire next render. If a request
// is already pending or in flight, QueuePick is a no-op so callers
// can drive it from a per-frame "left mouse down" check without
// worrying about overlap.
func QueuePick(p *Picking, x, y uint32) {
	if p.Pending != nil || p.hasInFlight {
		return
	}
	p.Pending = &PickRequest{X: x, Y: y}
}

// pickingPassState owns the staging buffer used to copy a 1-pixel
// region of the entity_id texture back to CPU memory. The pass reads
// the entity_id resource; if a request is pending, it records a
// CopyTextureToBuffer for that pixel.
type pickingPassState struct {
	stagingBuffer *wgpu.Buffer
	picking       *Picking
}

// stagingRowStride is the wgpu-mandated row stride for buffers used
// in CopyTextureToBuffer. Must be a multiple of 256 bytes even when
// the actual data is a single 4-byte u32.
const stagingRowStride = uint64(256)

// NewPickingPass builds the picking pass. It reads the entity_id
// resource (so the graph orders it after the mesh pass) and produces
// no output of its own; the side effect is a recorded
// CopyTextureToBuffer when there's a pending pick.
//
// The pass needs access to the [Picking] resource from the engine
// world, which it gets from PassContext during Prepare.
func NewPickingPass(device *wgpu.Device) (*Pass, error) {
	stagingBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "picking staging buffer",
		Size:  stagingRowStride,
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("picking pass: staging buffer: %w", err)
	}

	state := &pickingPassState{stagingBuffer: stagingBuffer}

	return &Pass{
		Name:    "picking",
		Reads:   []string{"entity_id"},
		State:   state,
		Prepare: pickingPrepare,
		Release: pickingRelease,
	}, nil
}

func pickingPrepare(s any, context *PassContext) error {
	state := s.(*pickingPassState)
	if state.picking == nil {
		if !ecs.HasResource[*Picking](context.World) {
			return nil
		}
		state.picking = *ecs.Resource[*Picking](context.World)
	}

	picking := state.picking
	if picking == nil || picking.Pending == nil {
		return nil
	}
	if picking.hasInFlight {
		return nil
	}

	req := *picking.Pending
	picking.Pending = nil

	id, ok := context.Slots["entity_id"]
	if !ok {
		return errors.New("picking pass: entity_id slot not bound")
	}
	handle := context.Resources.Handle(id)
	if handle == nil || handle.Texture == nil {
		return nil
	}
	if req.X >= handle.Width || req.Y >= handle.Height {
		return nil
	}

	source := wgpu.ImageCopyTexture{
		Texture:  handle.Texture,
		MipLevel: 0,
		Origin:   wgpu.Origin3D{X: req.X, Y: req.Y, Z: 0},
		Aspect:   wgpu.TextureAspectAll,
	}
	destination := wgpu.ImageCopyBuffer{
		Buffer: state.stagingBuffer,
		Layout: wgpu.TextureDataLayout{
			Offset:       0,
			BytesPerRow:  uint32(stagingRowStride),
			RowsPerImage: 1,
		},
	}
	extent := wgpu.Extent3D{Width: 1, Height: 1, DepthOrArrayLayers: 1}
	context.Encoder.CopyTextureToBuffer(&source, &destination, &extent)

	picking.requested = req
	picking.hasInFlight = true
	picking.mapped = false
	picking.mapErr = nil
	picking.staging = state.stagingBuffer

	return nil
}

func pickingRelease(s any) {
	state := s.(*pickingPassState)
	if state.stagingBuffer != nil {
		state.stagingBuffer.Release()
	}
}

// resetPicking clears the in-flight bookkeeping after a pick has
// resolved (or failed). Shared by the native and js readback paths.
func resetPicking(p *Picking) {
	p.hasInFlight = false
	p.mapping = false
	p.staging = nil
	p.mu.Lock()
	p.mapped = false
	p.mapErr = nil
	p.mu.Unlock()
}
