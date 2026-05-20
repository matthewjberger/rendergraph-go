package render

import "github.com/cogentcore/webgpu/wgpu"

// AddSkyPass builds the standard sky pass and registers it with the
// renderer's graph as a writer of scene_color. Returns the pass for
// callers that want a reference; the graph owns its release.
func AddSkyPass(renderer *Renderer) (*Pass, error) {
	pass, err := NewSkyPass(renderer.Device, renderer.SurfaceFormat, renderer.AspectRatio)
	if err != nil {
		return nil, err
	}
	if err := renderer.Graph.AddPass(pass, []SlotBinding{
		{Slot: "color", ResourceID: renderer.SceneColorID},
	}); err != nil {
		return nil, err
	}
	return pass, nil
}

// AddMeshPass builds the standard mesh pass and registers it as a
// writer of scene_color and depth.
func AddMeshPass(renderer *Renderer) (*Pass, error) {
	pass, err := NewMeshPass(renderer.Device, renderer.SurfaceFormat, renderer.AspectRatio)
	if err != nil {
		return nil, err
	}
	if err := renderer.Graph.AddPass(pass, []SlotBinding{
		{Slot: "color", ResourceID: renderer.SceneColorID},
		{Slot: "depth", ResourceID: renderer.DepthID},
	}); err != nil {
		return nil, err
	}
	return pass, nil
}

// AddGridPass builds the standard ground-grid pass. Reads scene_color
// + depth (loads, no clear) so it composites over whatever the mesh
// pass already wrote.
func AddGridPass(renderer *Renderer) (*Pass, error) {
	pass, err := NewGridPass(renderer.Device, renderer.SurfaceFormat, renderer.AspectRatio)
	if err != nil {
		return nil, err
	}
	if err := renderer.Graph.AddPass(pass, []SlotBinding{
		{Slot: "color", ResourceID: renderer.SceneColorID},
		{Slot: "depth", ResourceID: renderer.DepthID},
	}); err != nil {
		return nil, err
	}
	return pass, nil
}

// AddFxaaPass registers a transient fxaa_output color texture sized
// to the current renderer config, builds the FXAA pass to sample
// scene_color into it, and returns the new transient's ResourceID
// so the caller can wire the next stage (typically [AddPresentPass]).
func AddFxaaPass(renderer *Renderer) (*Pass, ResourceID, error) {
	fxaaOutputID := renderer.Graph.AddColorTexture(ResourceDescriptor{
		Name: "fxaa_output",
		Kind: ResourceKindTransientColor,
		Texture: TextureDescriptor{
			Format: renderer.SurfaceFormat,
			Width:  renderer.Config.Width,
			Height: renderer.Config.Height,
			Usage:  wgpu.TextureUsageRenderAttachment | wgpu.TextureUsageTextureBinding,
		},
	})
	pass, err := NewFxaaPass(renderer.Device, renderer.SurfaceFormat)
	if err != nil {
		return nil, 0, err
	}
	if err := renderer.Graph.AddPass(pass, []SlotBinding{
		{Slot: "input", ResourceID: renderer.SceneColorID},
		{Slot: "output", ResourceID: fxaaOutputID},
	}); err != nil {
		return nil, 0, err
	}
	return pass, fxaaOutputID, nil
}

// AddPresentPass builds the final present pass: samples inputID and
// blits to the swapchain.
func AddPresentPass(renderer *Renderer, inputID ResourceID) (*Pass, error) {
	pass, err := NewPresentPass(renderer.Device, renderer.SurfaceFormat)
	if err != nil {
		return nil, err
	}
	if err := renderer.Graph.AddPass(pass, []SlotBinding{
		{Slot: "input", ResourceID: inputID},
		{Slot: "output", ResourceID: renderer.SwapchainID},
	}); err != nil {
		return nil, err
	}
	return pass, nil
}
