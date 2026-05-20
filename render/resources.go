package render

import "github.com/cogentcore/webgpu/wgpu"

// ResourceID is a stable handle into [Resources]. Issued by [Resources.Register].
// Zero is a valid id; the ID is just an index into the descriptor slice.
type ResourceID uint32

// ResourceKind enumerates the kinds of GPU resources a render graph can own.
// It mirrors the union of ResourceType in nightshade's rendergraph, pared to
// the minimum needed for the engine today: color + depth, external (the
// caller supplies a view each frame) or transient (the graph allocates).
type ResourceKind uint8

const (
	ResourceKindExternalColor ResourceKind = iota
	ResourceKindTransientColor
	ResourceKindExternalDepth
	ResourceKindTransientDepth
)

// TextureDescriptor describes a transient texture the graph will allocate
// each frame (or reallocate on resize).
type TextureDescriptor struct {
	Format wgpu.TextureFormat
	Width  uint32
	Height uint32
	Usage  wgpu.TextureUsage
}

// ResourceDescriptor is the declarative half of a render graph resource:
// what it is and how it behaves. The matching handle lives in [Resources.Handles].
type ResourceDescriptor struct {
	Name string
	Kind ResourceKind

	// Texture is populated for transient color/depth.
	Texture TextureDescriptor

	// ClearColor is set on color resources whose first writer should clear.
	ClearColor *wgpu.Color
	// ClearDepth is set on depth resources whose first writer should clear.
	ClearDepth *float32
}

// TextureHandle is the runtime side of a texture resource. For external
// resources the caller refreshes [View] each frame via
// [Resources.SetExternalTexture]; for transients the graph fills them in
// during execute.
type TextureHandle struct {
	Texture *wgpu.Texture
	View    *wgpu.TextureView
	Width   uint32
	Height  uint32
	// Owned is true when the graph allocated this texture and must release
	// it. False for external resources whose lifetime belongs to the caller.
	Owned bool
}

// Resources owns the descriptors, runtime handles, and versions for every
// graph resource.
//
// Versions are the dirty-detection mechanism. Each resource has a u64
// stamp that increments every time the underlying handle changes
// (external view replacement each frame, or transient reallocation on
// resize). Passes record the version they cached a bind group against; if
// the version moves they invalidate. This mirrors the
// `versions: HashMap<ResourceId, u64>` table in nightshade's
// rendergraph::resources.
type Resources struct {
	Descriptors []ResourceDescriptor
	Handles     []TextureHandle
	Versions    []uint64
}

// Register adds a descriptor and returns its id. The matching handle slot
// is zero-valued until [SetExternalTexture] or graph execution fills it.
func (r *Resources) Register(descriptor ResourceDescriptor) ResourceID {
	id := ResourceID(len(r.Descriptors))
	r.Descriptors = append(r.Descriptors, descriptor)
	r.Handles = append(r.Handles, TextureHandle{})
	r.Versions = append(r.Versions, 0)
	return id
}

// SetExternalTexture refreshes the view backing an external resource and
// bumps the resource's version. Call this each frame for the swapchain
// before [Graph.Execute].
func (r *Resources) SetExternalTexture(id ResourceID, view *wgpu.TextureView, width, height uint32) {
	r.Handles[id] = TextureHandle{View: view, Width: width, Height: height, Owned: false}
	r.Versions[id]++
}

// Descriptor returns the descriptor for id. Panics on out-of-range ids;
// the render graph never hands out ids it did not register.
func (r *Resources) Descriptor(id ResourceID) *ResourceDescriptor {
	return &r.Descriptors[id]
}

// Handle returns the live handle for id.
func (r *Resources) Handle(id ResourceID) *TextureHandle {
	return &r.Handles[id]
}

// Version returns the current version stamp for id. Use this from a pass
// to decide whether a cached bind group is still valid.
func (r *Resources) Version(id ResourceID) uint64 {
	return r.Versions[id]
}

// ReleaseOwned releases every transient texture the graph allocated and
// zeroes the handle. Call this when tearing the graph down or before
// reallocating after a resize.
func (r *Resources) ReleaseOwned() {
	for index := range r.Handles {
		handle := &r.Handles[index]
		if !handle.Owned {
			continue
		}
		if handle.View != nil {
			handle.View.Release()
		}
		if handle.Texture != nil {
			handle.Texture.Release()
		}
		*handle = TextureHandle{}
	}
}
