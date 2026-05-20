package render

import "rendergraph-go/ecs"

// GraphicsSettings is the engine's runtime toggle resource. Each pass
// reads the relevant bool in Prepare/Execute and skips its work when
// disabled. Ported from nightshade's `world.resources.graphics.*`
// fields, scaled down to the toggles our current passes care about.
type GraphicsSettings struct {
	ShowSky     bool
	ShowGrid    bool
	FxaaEnabled bool
}

// DefaultGraphicsSettings returns settings with everything enabled.
func DefaultGraphicsSettings() GraphicsSettings {
	return GraphicsSettings{
		ShowSky:     true,
		ShowGrid:    true,
		FxaaEnabled: true,
	}
}

// UpdateGraphicsToggles is the engine system that flips graphics
// settings based on this frame's keyboard input. Press G to toggle the
// grid, S to toggle the sky, F to toggle FXAA. Reads the keys-just-
// down slice from [Input] and clears nothing; [Input.BeginFrame]
// resets the just-pressed slice each frame.
func UpdateGraphicsToggles(world *ecs.World) {
	input := ecs.Resource[Input](world)
	settings := ecs.Resource[GraphicsSettings](world)
	for _, key := range input.KeysJustDown {
		switch key {
		case 'G':
			settings.ShowGrid = !settings.ShowGrid
		case 'S':
			settings.ShowSky = !settings.ShowSky
		case 'F':
			settings.FxaaEnabled = !settings.FxaaEnabled
		}
	}
}
