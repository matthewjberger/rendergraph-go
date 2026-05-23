package render

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"

	"github.com/matthewjberger/indigo/ecs"
)

type PassContext struct {
	Device  *wgpu.Device
	Queue   *wgpu.Queue
	Encoder *wgpu.CommandEncoder

	World *ecs.World

	Resources *Resources
	Slots     map[string]ResourceID

	PassIndex int

	ClearOps map[ClearKey]struct{}
}

type ClearKey struct {
	PassIndex  int
	ResourceID ResourceID
}

func (c *PassContext) Slot(slot string) (ResourceID, bool) {
	id, ok := c.Slots[slot]
	return id, ok
}

func (c *PassContext) Resource(slot string) (*TextureHandle, error) {
	id, ok := c.Slots[slot]
	if !ok {
		return nil, fmt.Errorf("render: pass slot %q: %w", slot, ErrSlotNotBound)
	}
	return c.Resources.Handle(id), nil
}

func (c *PassContext) TextureView(slot string) (*wgpu.TextureView, error) {
	handle, err := c.Resource(slot)
	if err != nil {
		return nil, err
	}
	if handle.View == nil {
		return nil, fmt.Errorf("render: slot %q: %w", slot, ErrSlotNoView)
	}
	return handle.View, nil
}

func (c *PassContext) ColorAttachment(slot string) (wgpu.RenderPassColorAttachment, error) {
	id, ok := c.Slots[slot]
	if !ok {
		return wgpu.RenderPassColorAttachment{}, fmt.Errorf("render: pass slot %q: %w", slot, ErrSlotNotBound)
	}
	handle := c.Resources.Handle(id)
	descriptor := c.Resources.Descriptor(id)
	if handle.View == nil {
		return wgpu.RenderPassColorAttachment{}, fmt.Errorf("render: color slot %q (%s): %w", slot, descriptor.Name, ErrSlotNoView)
	}

	loadOp := wgpu.LoadOpLoad
	clear := wgpu.Color{}
	_, firstUse := c.ClearOps[ClearKey{PassIndex: c.PassIndex, ResourceID: id}]
	if firstUse && descriptor.ClearColor != nil {
		loadOp = wgpu.LoadOpClear
		clear = *descriptor.ClearColor
	}
	return wgpu.RenderPassColorAttachment{
		View:       handle.View,
		LoadOp:     loadOp,
		StoreOp:    wgpu.StoreOpStore,
		ClearValue: clear,
	}, nil
}

func (c *PassContext) DepthAttachment(slot string) (wgpu.RenderPassDepthStencilAttachment, error) {
	id, ok := c.Slots[slot]
	if !ok {
		return wgpu.RenderPassDepthStencilAttachment{}, fmt.Errorf("render: pass slot %q: %w", slot, ErrSlotNotBound)
	}
	handle := c.Resources.Handle(id)
	descriptor := c.Resources.Descriptor(id)
	if handle.View == nil {
		return wgpu.RenderPassDepthStencilAttachment{}, fmt.Errorf("render: depth slot %q (%s): %w", slot, descriptor.Name, ErrSlotNoView)
	}

	loadOp := wgpu.LoadOpLoad
	clear := float32(0)
	_, firstUse := c.ClearOps[ClearKey{PassIndex: c.PassIndex, ResourceID: id}]
	if firstUse && descriptor.ClearDepth != nil {
		loadOp = wgpu.LoadOpClear
		clear = *descriptor.ClearDepth
	}
	return wgpu.RenderPassDepthStencilAttachment{
		View:            handle.View,
		DepthLoadOp:     loadOp,
		DepthStoreOp:    wgpu.StoreOpStore,
		DepthClearValue: clear,
	}, nil
}

type Pass struct {
	Name   string
	Reads  []string
	Writes []string

	Prepare func(context *PassContext) error

	Execute func(context *PassContext) error

	InvalidateBindGroups func()

	Release func()
}
