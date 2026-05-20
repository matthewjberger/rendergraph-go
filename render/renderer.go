// Package render owns the WGPU device, surface, and render graph for the
// engine. The data-oriented analogue of nightshade's WgpuRenderer: one
// value lives as a resource on the ECS world and is consulted by passes
// and by the main loop.
//
// The graph design is the nightshade fundamentals (declarative resources,
// passes that read/write named slots, a compile step that decides
// clear-vs-load, version stamps that drive bind-group invalidation, the
// ECS world threaded through PassContext) ported to Go. The default graph
// is wired in the same shape as nightshade's: passes write color into a
// transient scene_color target; a final present pass blits scene_color to
// the external swapchain. Future passes (bloom, SSAO, OIT) chain between
// scene_color and the present pass without changes here.
package render

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"

	"indigo/ecs"
)

// DepthFormat is the depth target the renderer creates. Chosen to match
// the nightshade engine so future passes can be ported as-is.
const DepthFormat = wgpu.TextureFormatDepth32Float

// Renderer owns the long-lived WGPU surface/device/queue plus the
// render graph and the ids of the graph's standard resources
// (swapchain, scene_color, depth). Mesh registries and primitive
// handles live as separate engine-world resources (see
// [MeshAssets] / [Primitives]). The renderer is stored as a resource
// on the ECS world and is not safe for concurrent use.
type Renderer struct {
	Surface       *wgpu.Surface
	Adapter       *wgpu.Adapter
	Device        *wgpu.Device
	Queue         *wgpu.Queue
	Config        *wgpu.SurfaceConfiguration
	SurfaceFormat wgpu.TextureFormat

	Graph        *Graph
	SwapchainID  ResourceID
	SceneColorID ResourceID
	DepthID      ResourceID
}

// NewRenderer acquires an adapter and device from the instance, configures
// the surface, and builds the render graph with the swapchain, scene_color,
// and depth resources registered.
func NewRenderer(instance *wgpu.Instance, surface *wgpu.Surface, width, height uint32) (*Renderer, error) {
	renderer := &Renderer{Surface: surface}

	adapter, err := instance.RequestAdapter(&wgpu.RequestAdapterOptions{
		CompatibleSurface: surface,
	})
	if err != nil {
		return nil, fmt.Errorf("render: request adapter: %w", err)
	}
	renderer.Adapter = adapter

	device, err := adapter.RequestDevice(nil)
	if err != nil {
		adapter.Release()
		return nil, fmt.Errorf("render: request device: %w", err)
	}
	renderer.Device = device
	renderer.Queue = device.GetQueue()

	caps := surface.GetCapabilities(adapter)
	renderer.SurfaceFormat = caps.Formats[0]
	for _, format := range caps.Formats {
		if !isSrgb(format) {
			renderer.SurfaceFormat = format
			break
		}
	}

	renderer.Config = &wgpu.SurfaceConfiguration{
		Usage:       wgpu.TextureUsageRenderAttachment,
		Format:      renderer.SurfaceFormat,
		Width:       width,
		Height:      height,
		PresentMode: caps.PresentModes[0],
		AlphaMode:   caps.AlphaModes[0],
	}
	surface.Configure(adapter, device, renderer.Config)

	renderer.Graph = defaultGraph(renderer.SurfaceFormat, width, height)
	renderer.SwapchainID = renderer.Graph.ResourceByName("swapchain")
	renderer.SceneColorID = renderer.Graph.ResourceByName("scene_color")
	renderer.DepthID = renderer.Graph.ResourceByName("depth")

	return renderer, nil
}

// defaultGraph returns a graph with the engine's standard resources
// registered: a transient scene_color (color passes render here), a
// transient depth, and an external swapchain (the present pass blits
// scene_color into it). It does not register any passes; the application
// adds those in its configure-render-graph hook.
//
// scene_color's format matches the surface so the final present pass can
// blit without a tonemap. Once the engine grows HDR + tonemapping this
// should switch to wgpu.TextureFormatRGBA16Float, matching nightshade.
func defaultGraph(surfaceFormat wgpu.TextureFormat, width, height uint32) *Graph {
	graph := NewGraph()
	clearColor := wgpu.Color{R: 0.19, G: 0.24, B: 0.42, A: 1.0}
	clearDepth := float32(1.0)
	graph.AddColorTexture(ResourceDescriptor{
		Name: "scene_color",
		Kind: ResourceKindTransientColor,
		Texture: TextureDescriptor{
			Format: surfaceFormat,
			Width:  width,
			Height: height,
			Usage:  wgpu.TextureUsageRenderAttachment | wgpu.TextureUsageTextureBinding | wgpu.TextureUsageCopySrc,
		},
		ClearColor: &clearColor,
	})
	graph.AddDepthTexture(ResourceDescriptor{
		Name: "depth",
		Kind: ResourceKindTransientDepth,
		Texture: TextureDescriptor{
			Format: DepthFormat,
			Width:  width,
			Height: height,
			Usage:  wgpu.TextureUsageRenderAttachment | wgpu.TextureUsageTextureBinding,
		},
		ClearDepth: &clearDepth,
	})
	graph.AddColorTexture(ResourceDescriptor{
		Name: "swapchain",
		Kind: ResourceKindExternalColor,
	})
	return graph
}

// AspectRatio returns the surface aspect ratio, clamped to avoid divide-
// by-zero during minimization.
func (r *Renderer) AspectRatio() float32 {
	height := r.Config.Height
	if height < 1 {
		height = 1
	}
	return float32(r.Config.Width) / float32(height)
}

// Resize reconfigures the surface and reallocates transients at the new
// dimensions.
func (r *Renderer) Resize(width, height uint32) error {
	r.Config.Width = width
	r.Config.Height = height
	r.Surface.Configure(r.Adapter, r.Device, r.Config)
	return r.Graph.ResizeTransients(r.Device, width, height)
}

// Reconfigure re-applies the current surface configuration without
// rebuilding transients.
func (r *Renderer) Reconfigure() {
	r.Surface.Configure(r.Adapter, r.Device, r.Config)
}

// RenderFrame acquires the next surface texture, wires it into the
// "swapchain" resource, runs the graph against the world, and
// presents. Shaped like a system (free function over the renderer +
// world) so the main loop's call site reads consistently with the
// scheduler.
func RenderFrame(r *Renderer, world *ecs.World) error {
	surfaceTexture, err := r.Surface.GetCurrentTexture()
	if err != nil {
		return wrapSurfaceErr(err)
	}

	view, err := surfaceTexture.CreateView(nil)
	if err != nil {
		return err
	}
	defer view.Release()

	r.Graph.Resources.SetExternalTexture(r.SwapchainID, view, r.Config.Width, r.Config.Height)

	encoder, err := r.Device.CreateCommandEncoder(&wgpu.CommandEncoderDescriptor{Label: "frame"})
	if err != nil {
		return err
	}
	defer encoder.Release()

	if err := r.Graph.Execute(r.Device, r.Queue, world, encoder); err != nil {
		return err
	}

	cmd, err := encoder.Finish(nil)
	if err != nil {
		return err
	}
	defer cmd.Release()

	r.Queue.Submit(cmd)
	r.Surface.Present()
	return nil
}

// RendererResource is the type wrapper applications use to put a
// *Renderer on an ECS world via [ecs.SetResource]. freecs-go keys
// resources by Go type, so a named wrapper keeps the renderer
// distinct from any other pointer-typed resource the application
// might add.
type RendererResource struct {
	Renderer *Renderer
}

// Release frees every WGPU object owned by the renderer. Mesh
// registries owned by the engine world ([MeshAssets] resource) are
// released independently.
func (r *Renderer) Release() {
	if r.Graph != nil {
		r.Graph.Release()
	}
	if r.Queue != nil {
		r.Queue.Release()
	}
	if r.Device != nil {
		r.Device.Release()
	}
	if r.Adapter != nil {
		r.Adapter.Release()
	}
	if r.Surface != nil {
		r.Surface.Release()
	}
}

func isSrgb(f wgpu.TextureFormat) bool {
	switch f {
	case wgpu.TextureFormatRGBA8UnormSrgb,
		wgpu.TextureFormatBGRA8UnormSrgb:
		return true
	}
	return false
}
