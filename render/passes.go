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
// writer of scene_color, depth, and the entity_id pick target.
func AddMeshPass(renderer *Renderer) (*Pass, error) {
	pass, err := NewMeshPass(renderer.Device, renderer.SurfaceFormat, renderer.AspectRatio)
	if err != nil {
		return nil, err
	}
	if err := renderer.Graph.AddPass(pass, []SlotBinding{
		{Slot: "color", ResourceID: renderer.SceneColorID},
		{Slot: "depth", ResourceID: renderer.DepthID},
		{Slot: "entity_id", ResourceID: renderer.EntityIdID},
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

// AddPickingPass builds the picking pass and wires it to read the
// entity_id texture the mesh pass writes. The pass only does work
// when a pick request is queued through the [Picking] resource.
func AddPickingPass(renderer *Renderer) (*Pass, error) {
	pass, err := NewPickingPass(renderer.Device)
	if err != nil {
		return nil, err
	}
	if err := renderer.Graph.AddPass(pass, []SlotBinding{
		{Slot: "entity_id", ResourceID: renderer.EntityIdID},
	}); err != nil {
		return nil, err
	}
	return pass, nil
}

// AddSelectionMaskPass builds the selection-mask pass. Reads the
// entity_id texture, writes the selection_mask transient.
func AddSelectionMaskPass(renderer *Renderer) (*Pass, error) {
	pass, err := NewSelectionMaskPass(renderer.Device)
	if err != nil {
		return nil, err
	}
	if err := renderer.Graph.AddPass(pass, []SlotBinding{
		{Slot: "entity_id", ResourceID: renderer.EntityIdID},
		{Slot: "selection_mask", ResourceID: renderer.SelectionMaskID},
	}); err != nil {
		return nil, err
	}
	return pass, nil
}

// AddOutlinePass builds the editor outline post-process. Reads
// selection_mask, alpha-blends an outline color into scene_color
// along the dilated mask boundary.
func AddOutlinePass(renderer *Renderer) (*Pass, error) {
	pass, err := NewOutlinePass(renderer.Device, renderer.SurfaceFormat, func() (uint32, uint32) {
		return renderer.Config.Width, renderer.Config.Height
	})
	if err != nil {
		return nil, err
	}
	if err := renderer.Graph.AddPass(pass, []SlotBinding{
		{Slot: "selection_mask", ResourceID: renderer.SelectionMaskID},
		{Slot: "color", ResourceID: renderer.SceneColorID},
	}); err != nil {
		return nil, err
	}
	return pass, nil
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
