package render

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"

	"rendergraph-go/ecs"
)

// PassContext is what a pass's Prepare and Execute functions receive each
// frame. It mirrors nightshade's PassExecutionContext, including the
// typed handle to the ECS world (`configs: &C` in the Rust). Every
// pointer is borrowed for the duration of the call and must not be
// retained.
//
// The slot lookup uses the pass-local name (e.g. "color"), not the graph-
// global resource name. The graph wires those at compile time.
type PassContext struct {
	Device  *wgpu.Device
	Queue   *wgpu.Queue
	Encoder *wgpu.CommandEncoder

	// World is the engine's ECS world. Passes extract per-frame data
	// (camera, lights, input) from it inside Prepare and upload as
	// uniforms. Matches the role of `configs: &World` in nightshade's
	// PassExecutionContext.
	World *ecs.World

	Resources *Resources
	Slots     map[string]ResourceID

	// PassIndex is the position of this pass in the execution order. Used
	// by the graph to decide whether a slot's first use is a clear.
	PassIndex int

	// ClearOps tracks the (pass, resource) pairs that should clear-on-load.
	// Indexed by ClearKey so a pass can ask "should I clear this attachment".
	ClearOps map[ClearKey]struct{}
}

// ClearKey identifies a single attachment use within the graph. A pass
// clears a resource only when (PassIndex, ResourceID) is the first writer
// of that resource and the descriptor declared a clear color/depth.
type ClearKey struct {
	PassIndex  int
	ResourceID ResourceID
}

// Slot returns the resource id bound to slot, or false if the slot is
// not bound on this pass.
func (c *PassContext) Slot(slot string) (ResourceID, bool) {
	id, ok := c.Slots[slot]
	return id, ok
}

// Resource returns the runtime handle for the resource bound to slot.
func (c *PassContext) Resource(slot string) (*TextureHandle, error) {
	id, ok := c.Slots[slot]
	if !ok {
		return nil, fmt.Errorf("render: pass slot %q not bound", slot)
	}
	return c.Resources.Handle(id), nil
}

// TextureView returns the texture view for the resource bound to slot.
// Use it from a pass that samples the slot as an input (e.g. a blit
// reading scene_color).
func (c *PassContext) TextureView(slot string) (*wgpu.TextureView, error) {
	handle, err := c.Resource(slot)
	if err != nil {
		return nil, err
	}
	if handle.View == nil {
		return nil, fmt.Errorf("render: slot %q has no view", slot)
	}
	return handle.View, nil
}

// ColorAttachment returns a fully-prepared
// [wgpu.RenderPassColorAttachment] for slot, ready to drop into a
// render-pass descriptor. The load op is Clear iff this pass is the
// first writer of the resource and the descriptor declared a clear
// color, else Load. ClearValue is always populated (wgpu ignores it
// when the load op is Load).
func (c *PassContext) ColorAttachment(slot string) (wgpu.RenderPassColorAttachment, error) {
	id, ok := c.Slots[slot]
	if !ok {
		return wgpu.RenderPassColorAttachment{}, fmt.Errorf("render: pass slot %q not bound", slot)
	}
	handle := c.Resources.Handle(id)
	descriptor := c.Resources.Descriptor(id)
	if handle.View == nil {
		return wgpu.RenderPassColorAttachment{}, fmt.Errorf("render: color slot %q (%s) has no view", slot, descriptor.Name)
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

// DepthAttachment returns a fully-prepared
// [wgpu.RenderPassDepthStencilAttachment] for slot. Same first-write
// clear rule as [PassContext.ColorAttachment]. Caller takes the
// address to drop it into a render-pass descriptor's
// DepthStencilAttachment field.
func (c *PassContext) DepthAttachment(slot string) (wgpu.RenderPassDepthStencilAttachment, error) {
	id, ok := c.Slots[slot]
	if !ok {
		return wgpu.RenderPassDepthStencilAttachment{}, fmt.Errorf("render: pass slot %q not bound", slot)
	}
	handle := c.Resources.Handle(id)
	descriptor := c.Resources.Descriptor(id)
	if handle.View == nil {
		return wgpu.RenderPassDepthStencilAttachment{}, fmt.Errorf("render: depth slot %q (%s) has no view", slot, descriptor.Name)
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

// Pass is the data-oriented analogue of nightshade's PassNode trait.
// Instead of a trait with virtual dispatch, a pass is a struct of fields
// the graph reads. State lives in [State] (any-typed so the graph stays
// generic); per-pass logic is function values the graph calls.
//
// Reads and Writes declare slot names that must be wired to resources
// before the graph compiles.
type Pass struct {
	Name   string
	Reads  []string
	Writes []string

	// State holds whatever the pass needs across frames (pipeline,
	// vertex buffer, uniform binding). The graph never touches it.
	State any

	// Prepare runs before Execute every frame. Use it to upload uniforms
	// extracted from the ECS world. nil means "skip".
	Prepare func(state any, context *PassContext) error

	// Execute records GPU commands for the frame. nil means "no-op".
	Execute func(state any, context *PassContext) error

	// InvalidateBindGroups is called by the graph before Prepare when any
	// resource bound to one of the pass's slots has changed version since
	// the previous frame (its handle was replaced; either external view
	// refresh or transient reallocation). Mirrors nightshade's
	// `PassNode::invalidate_bind_groups`. nil means "no caching to drop".
	InvalidateBindGroups func(state any)

	// Release frees GPU objects held in State. nil means "nothing to release".
	Release func(state any)
}
