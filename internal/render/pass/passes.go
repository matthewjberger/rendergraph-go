package pass

import (
	"github.com/cogentcore/webgpu/wgpu"

	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/render/asset"
)

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

func AddMeshPass(renderer *render.Renderer, arrays *asset.MaterialTextureArrays, registry *asset.MaterialRegistry, ibl *IBL, shadow *Shadow, spotShadow *SpotShadow, pointShadow *PointShadow) (*render.Pass, error) {
	pass, err := NewMeshPass(renderer.Device, render.HdrFormat, renderer.AspectRatio, arrays, registry, ibl, shadow, spotShadow, pointShadow)
	if err != nil {
		return nil, err
	}
	if err := renderer.Graph.AddPass(pass, []render.SlotBinding{
		{Slot: "color", ResourceID: renderer.SceneColorID},
		{Slot: "depth", ResourceID: renderer.DepthID},
		{Slot: "entity_id", ResourceID: renderer.EntityIdID},
		{Slot: "view_normals", ResourceID: renderer.ViewNormalsID},
	}); err != nil {
		return nil, err
	}
	return pass, nil
}

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

func AddPostProcessPass(renderer *render.Renderer, bloomMipView func() *wgpu.TextureView) (*render.Pass, error) {
	pass, err := NewPostProcessPass(renderer.Device, renderer.SurfaceFormat, bloomMipView)
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
