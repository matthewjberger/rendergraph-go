package app

import (
	"fmt"

	"indigo/ecs"
	"indigo/render"
	"indigo/render/asset"
	"indigo/render/pass"
	"indigo/transform"
	"indigo/ui"
	"indigo/window"
)

// EngineRef is the game-world resource pointing at the engine world.
// Game systems read this to look up the engine world they should
// sync into via the [SyncEngine...] helpers.
type EngineRef struct{ World *ecs.World }

// NewEngineWorld builds the engine-side ECS world every app shares:
// the standard transform + RenderMesh + Light component types and
// the resources the standard engine systems read (window, renderer,
// mesh assets + primitives, input, graphics settings, propagation
// state).
//
// Apps may register additional engine-side components and resources
// after this returns. The order of any later [ecs.Register] calls
// determines the bit positions for those types in this world; this
// helper does not stabilize bit positions across apps.
func NewEngineWorld(renderer *render.Renderer) (*ecs.World, error) {
	engine := ecs.New()
	ecs.Register[transform.LocalTransform](engine)
	ecs.Register[transform.GlobalTransform](engine)
	ecs.Register[transform.Parent](engine)
	ecs.Register[transform.LocalTransformDirty](engine)
	ecs.Register[transform.IgnoreParentScale](engine)
	ecs.Register[transform.GroupRoot](engine)
	ecs.Register[asset.RenderMesh](engine)
	ecs.Register[asset.Material](engine)
	ecs.Register[asset.AnimationPlayer](engine)
	ecs.Register[render.Light](engine)
	ecs.Register[render.CameraMarker](engine)
	ecs.Register[render.Selected](engine)
	ecs.Register[Name](engine)

	meshAssets := asset.NewMeshAssets()
	primitives, err := asset.RegisterPrimitives(renderer.Device, meshAssets)
	if err != nil {
		return nil, fmt.Errorf("app: register primitives: %w", err)
	}

	textureCache := asset.NewTextureCache()
	if _, err := textureCache.EnsureWhite(renderer.Device, renderer.Queue); err != nil {
		return nil, fmt.Errorf("app: register white texture: %w", err)
	}

	materialArrays, err := asset.NewMaterialTextureArrays(renderer.Device)
	if err != nil {
		return nil, fmt.Errorf("app: material texture arrays: %w", err)
	}

	materialRegistry, err := asset.NewMaterialRegistry(renderer.Device)
	if err != nil {
		return nil, fmt.Errorf("app: material registry: %w", err)
	}

	ibl, err := pass.NewIBL(renderer.Device, renderer.Queue)
	if err != nil {
		return nil, fmt.Errorf("app: ibl: %w", err)
	}

	shadow, err := pass.NewShadow(renderer.Device)
	if err != nil {
		return nil, fmt.Errorf("app: shadow: %w", err)
	}

	spotShadow, err := pass.NewSpotShadow(renderer.Device)
	if err != nil {
		return nil, fmt.Errorf("app: spot shadow: %w", err)
	}

	pointShadow, err := pass.NewPointShadow(renderer.Device)
	if err != nil {
		return nil, fmt.Errorf("app: point shadow: %w", err)
	}

	lines := &pass.Lines{}

	ecs.SetResource(engine, window.Window{
		Viewport: window.ViewportSize{
			Width:  renderer.Config.Width,
			Height: renderer.Config.Height,
		},
	})
	ecs.SetResource(engine, render.RendererResource{Renderer: renderer})
	ecs.SetResource(engine, asset.MeshAssetsResource{Assets: meshAssets})
	ecs.SetResource(engine, asset.TextureCacheResource{Cache: textureCache})
	ecs.SetResource(engine, asset.MaterialTextureArraysResource{Arrays: materialArrays})
	ecs.SetResource(engine, asset.MaterialRegistryResource{Registry: materialRegistry})
	ecs.SetResource(engine, pass.IBLResource{IBL: ibl})
	ecs.SetResource(engine, pass.ShadowResource{Shadow: shadow})
	ecs.SetResource(engine, pass.SpotShadowResource{Shadow: spotShadow})
	ecs.SetResource(engine, pass.PointShadowResource{Shadow: pointShadow})
	ecs.SetResource(engine, pass.LinesResource{Lines: lines})
	ecs.SetResource(engine, primitives)
	ecs.SetResource(engine, render.NewInput())
	ecs.SetResource(engine, render.DefaultGraphicsSettings())
	ecs.SetResource(engine, render.DefaultPostProcessSettings())
	ecs.SetResource(engine, transform.NewTransformState())
	ecs.SetResource(engine, ecs.EcsCommandQueueResource{})
	ecs.SetResource(engine, render.RenderCommandQueueResource{})
	return engine, nil
}

// Worlds bundles the engine, game, and UI worlds plus their
// schedules. Use [NewWorlds] to construct one with cross-world
// bridges (EngineRef on the game world, ui.WorldRef on the engine
// world) pre-installed.
type Worlds struct {
	Engine *ecs.World
	Game   *ecs.World
	UI     *ecs.World

	EngineSchedule *ecs.Schedule
	GameSchedule   *ecs.Schedule
	UISchedule     *ecs.Schedule
}

// NewWorlds returns a fully-wired Worlds: engine world via
// [NewEngineWorld], a game world with EngineRef installed, a UI
// world with ui.WorldRef installed on the engine world, and empty
// schedules ready for the app to populate. Eliminates the
// hand-wiring of cross-world resource bridges that apps used to do
// in their own setup.
func NewWorlds(renderer *render.Renderer) (Worlds, error) {
	engine, err := NewEngineWorld(renderer)
	if err != nil {
		return Worlds{}, err
	}

	game := ecs.New()
	ecs.Register[EngineEntity](game)
	ecs.SetResource(game, window.Window{})
	ecs.SetResource(game, EngineRef{World: engine})

	uiWorld := ui.NewWorld(renderer.Config.Width, renderer.Config.Height)
	ecs.SetResource(engine, ui.WorldRef{World: uiWorld})

	return Worlds{
		Engine:         engine,
		Game:           game,
		UI:             uiWorld,
		EngineSchedule: ecs.NewSchedule(),
		GameSchedule:   ecs.NewSchedule(),
		UISchedule:     ui.NewSchedule(),
	}, nil
}

// TickFrame runs the per-frame pre-render work: advance the window
// timing on every present world, run game then engine then ui
// schedules, run the app's optional [App] lifecycle hooks, apply any
// deferred ECS commands.
//
// World.Step is intentionally deferred to [PostFrame] so the renderer
// (which runs after this returns) can still see the current frame's
// change-detection stamps.
func TickFrame(worlds Worlds, hooks *App, delta float32) {
	window.Advance(&ecs.MustResource[window.Window](worlds.Engine).Timing, delta)
	window.Advance(&ecs.MustResource[window.Window](worlds.Game).Timing, delta)
	if worlds.UI != nil {
		window.Advance(&ecs.MustResource[window.Window](worlds.UI).Timing, delta)
	}

	if worlds.UI != nil && worlds.UISchedule != nil {
		worlds.UISchedule.Run(worlds.UI)
	}
	worlds.GameSchedule.Run(worlds.Game)
	worlds.EngineSchedule.Run(worlds.Engine)

	if hooks != nil {
		if hooks.RunSystems != nil {
			hooks.RunSystems(worlds.Engine)
		}
		if hooks.PreRender != nil {
			hooks.PreRender(worlds.Engine)
		}
	}

	ecs.ProcessEcsCommands(worlds.Engine)
	worlds.Engine.ApplyCommands()
	worlds.Game.ApplyCommands()
	if worlds.UI != nil {
		worlds.UI.ApplyCommands()
	}
}

// PostFrame finalizes the frame after the renderer has run: advance
// each world's tick (rolls event queues, ages change-detection
// watermarks) and clear the per-frame input deltas.
func PostFrame(worlds Worlds) {
	worlds.Engine.Step()
	worlds.Game.Step()
	if worlds.UI != nil {
		worlds.UI.Step()
	}
	render.InputBeginFrame(ecs.MustResource[render.Input](worlds.Engine))
}
