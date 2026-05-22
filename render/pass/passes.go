package pass

import (
	"github.com/cogentcore/webgpu/wgpu"

	"indigo/render"
	"indigo/render/asset"
)

// AddSkyPass builds the standard sky pass and registers it with the
// renderer's graph as a writer of scene_color. Returns the pass for
// callers that want a reference; the graph owns its release.
func AddSkyPass(renderer *render.Renderer) (*render.Pass, error) {
	pass, err := NewSkyPass(renderer.Device, render.HdrFormat, renderer.AspectRatio)
	if err != nil {
		return nil, err
	}
	if err := renderer.Graph.AddPass(pass, []render.SlotBinding{
		{Slot: "color", ResourceID: renderer.SceneColorID},
	}); err != nil {
		return nil, err
	}
	return pass, nil
}

// AddMeshPass builds the standard mesh pass and registers it as a
// writer of scene_color, depth, and the entity_id pick target.
// arrays is the engine-world's [asset.MaterialTextureArrays] (PBR
// texture source); ibl is the [IBL] bundle the fragment shader
// samples for ambient diffuse + specular.
func AddMeshPass(renderer *render.Renderer, arrays *asset.MaterialTextureArrays, ibl *IBL) (*render.Pass, error) {
	pass, err := NewMeshPass(renderer.Device, render.HdrFormat, renderer.AspectRatio, arrays, ibl)
	if err != nil {
		return nil, err
	}
	if err := renderer.Graph.AddPass(pass, []render.SlotBinding{
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
func AddGridPass(renderer *render.Renderer) (*render.Pass, error) {
	pass, err := NewGridPass(renderer.Device, render.HdrFormat, renderer.AspectRatio)
	if err != nil {
		return nil, err
	}
	if err := renderer.Graph.AddPass(pass, []render.SlotBinding{
		{Slot: "color", ResourceID: renderer.SceneColorID},
		{Slot: "depth", ResourceID: renderer.DepthID},
	}); err != nil {
		return nil, err
	}
	return pass, nil
}

// AddFxaaPass registers a transient fxaa_output color texture sized
// to the current renderer config, builds the FXAA pass to sample
// scene_color into it, and returns the new transient's render.ResourceID
// so the caller can wire the next stage (typically [AddPresentPass]).
func AddFxaaPass(renderer *render.Renderer) (*render.Pass, render.ResourceID, error) {
	fxaaOutputID := renderer.Graph.AddColorTexture(render.ResourceDescriptor{
		Name: "fxaa_output",
		Kind: render.ResourceKindTransientColor,
		Texture: render.TextureDescriptor{
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
	if err := renderer.Graph.AddPass(pass, []render.SlotBinding{
		{Slot: "input", ResourceID: renderer.LdrColorID},
		{Slot: "output", ResourceID: fxaaOutputID},
	}); err != nil {
		return nil, 0, err
	}
	return pass, fxaaOutputID, nil
}

// AddPostProcessPass builds the HDR -> LDR tonemap pass. Reads
// scene_color (HDR) and writes ldr_color (surface format). Must
// run after every 3D pass that writes scene_color (sky / mesh /
// grid / lines / outline) and before fxaa + UI passes.
func AddPostProcessPass(renderer *render.Renderer) (*render.Pass, error) {
	pass, err := NewPostProcessPass(renderer.Device, renderer.SurfaceFormat)
	if err != nil {
		return nil, err
	}
	if err := renderer.Graph.AddPass(pass, []render.SlotBinding{
		{Slot: "input", ResourceID: renderer.SceneColorID},
		{Slot: "output", ResourceID: renderer.LdrColorID},
	}); err != nil {
		return nil, err
	}
	return pass, nil
}

// AddPickingPass builds the picking pass and wires it to read the
// entity_id texture the mesh pass writes. The pass only does work
// when a pick request is queued through the [Picking] resource.
func AddPickingPass(renderer *render.Renderer) (*render.Pass, error) {
	pass, err := NewPickingPass(renderer.Device)
	if err != nil {
		return nil, err
	}
	if err := renderer.Graph.AddPass(pass, []render.SlotBinding{
		{Slot: "entity_id", ResourceID: renderer.EntityIdID},
	}); err != nil {
		return nil, err
	}
	return pass, nil
}

// AddUiQuadPass builds the UI colored-rectangle pass and binds it
// to write into colorID (typically fxaa_output so the UI lands on
// top of the antialiased scene before present).
func AddUiQuadPass(renderer *render.Renderer, colorID render.ResourceID) (*render.Pass, error) {
	pass, err := NewUiQuadPass(renderer.Device, renderer.SurfaceFormat)
	if err != nil {
		return nil, err
	}
	if err := renderer.Graph.AddPass(pass, []render.SlotBinding{
		{Slot: "color", ResourceID: colorID},
	}); err != nil {
		return nil, err
	}
	return pass, nil
}

// AddUiTextPass builds the UI text pass and binds it to write into
// colorID (typically the same buffer the UI quad pass writes to,
// inserted after it in the graph so glyphs draw on top of any
// background panels).
func AddUiTextPass(renderer *render.Renderer, colorID render.ResourceID) (*render.Pass, error) {
	pass, err := NewUiTextPass(renderer.Device, renderer.Queue, renderer.SurfaceFormat)
	if err != nil {
		return nil, err
	}
	if err := renderer.Graph.AddPass(pass, []render.SlotBinding{
		{Slot: "color", ResourceID: colorID},
	}); err != nil {
		return nil, err
	}
	return pass, nil
}

// AddSelectionMaskPass builds the selection-mask pass. Reads the
// entity_id texture, writes the selection_mask transient.
func AddSelectionMaskPass(renderer *render.Renderer) (*render.Pass, error) {
	pass, err := NewSelectionMaskPass(renderer.Device)
	if err != nil {
		return nil, err
	}
	if err := renderer.Graph.AddPass(pass, []render.SlotBinding{
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
func AddOutlinePass(renderer *render.Renderer) (*render.Pass, error) {
	pass, err := NewOutlinePass(renderer.Device, render.HdrFormat, func() (uint32, uint32) {
		return renderer.Config.Width, renderer.Config.Height
	})
	if err != nil {
		return nil, err
	}
	if err := renderer.Graph.AddPass(pass, []render.SlotBinding{
		{Slot: "selection_mask", ResourceID: renderer.SelectionMaskID},
		{Slot: "color", ResourceID: renderer.SceneColorID},
	}); err != nil {
		return nil, err
	}
	return pass, nil
}

// AddPresentPass builds the final present pass: samples inputID and
// blits to the swapchain.
func AddPresentPass(renderer *render.Renderer, inputID render.ResourceID) (*render.Pass, error) {
	pass, err := NewPresentPass(renderer.Device, renderer.SurfaceFormat)
	if err != nil {
		return nil, err
	}
	if err := renderer.Graph.AddPass(pass, []render.SlotBinding{
		{Slot: "input", ResourceID: inputID},
		{Slot: "output", ResourceID: renderer.SwapchainID},
	}); err != nil {
		return nil, err
	}
	return pass, nil
}
