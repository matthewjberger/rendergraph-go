package pass

import (
	"errors"
	"fmt"
	"sync"

	"github.com/cogentcore/webgpu/wgpu"

	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
)

type PickRequest struct {
	X uint32
	Y uint32
}

type PickResult struct {
	EntityID uint32
	Request  PickRequest
}

type Picking struct {
	Pending *PickRequest
	Result  *PickResult

	staging     *wgpu.Buffer
	requested   PickRequest
	hasInFlight bool
	mapping     bool
	mu          sync.Mutex
	mapped      bool
	mapErr      error
}

func NewPicking() *Picking {
	return &Picking{}
}

func QueuePick(p *Picking, x, y uint32) {
	if p.Pending != nil || p.hasInFlight {
		return
	}
	p.Pending = &PickRequest{X: x, Y: y}
}

type pickingPassState struct {
	stagingBuffer *wgpu.Buffer
	picking       *Picking
}

const stagingRowStride = uint64(256)

func NewPickingPass(device *wgpu.Device) (*render.Pass, error) {
	stagingBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "picking staging buffer",
		Size:  stagingRowStride,
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("picking pass: staging buffer: %w", err)
	}

	state := &pickingPassState{stagingBuffer: stagingBuffer}

	return &render.Pass{
		Name:    "picking",
		Reads:   []string{"entity_id"},
		State:   state,
		Prepare: pickingPrepare,
		Release: pickingRelease,
	}, nil
}

func pickingPrepare(s any, context *render.PassContext) error {
	state := s.(*pickingPassState)
	if state.picking == nil {
		if !ecs.HasResource[*Picking](context.World) {
			return nil
		}
		state.picking = *ecs.MustResource[*Picking](context.World)
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

func resetPicking(p *Picking) {
	p.hasInFlight = false
	p.mapping = false
	p.staging = nil
	p.mu.Lock()
	p.mapped = false
	p.mapErr = nil
	p.mu.Unlock()
}
