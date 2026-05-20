package app

import (
	"fmt"

	"indigo/ecs"
	"indigo/render"
	"indigo/transform"
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
	ecs.Register[render.RenderMesh](engine)
	ecs.Register[render.Light](engine)

	meshAssets := render.NewMeshAssets()
	primitives, err := render.RegisterPrimitives(renderer.Device, meshAssets)
	if err != nil {
		return nil, fmt.Errorf("app: register primitives: %w", err)
	}

	ecs.SetResource(engine, window.Window{
		Viewport: window.ViewportSize{
			Width:  renderer.Config.Width,
			Height: renderer.Config.Height,
		},
	})
	ecs.SetResource(engine, render.RendererResource{Renderer: renderer})
	ecs.SetResource(engine, render.MeshAssetsResource{Assets: meshAssets})
	ecs.SetResource(engine, primitives)
	ecs.SetResource(engine, render.NewInput())
	ecs.SetResource(engine, render.DefaultGraphicsSettings())
	ecs.SetResource(engine, transform.NewPropagationState())
	return engine, nil
}

// Worlds bundles an engine world, a game world, and their schedules.
// [TickFrame] and [PostFrame] walk both worlds in the order the engine
// expects.
type Worlds struct {
	Engine         *ecs.World
	Game           *ecs.World
	EngineSchedule *ecs.Schedule
	GameSchedule   *ecs.Schedule
}

// TickFrame runs the per-frame pre-render work: advance the window
// timing on both worlds, run the game schedule (game systems write
// back to the engine through the sync helpers), run the engine
// schedule, run the app's optional [App] lifecycle hooks, then apply
// any deferred ECS commands.
//
// World.Step is intentionally deferred to [PostFrame] so the renderer
// (which runs after this returns) can still see the current frame's
// change-detection stamps.
func TickFrame(worlds Worlds, hooks *App, delta float32) {
	window.Advance(&ecs.Resource[window.Window](worlds.Engine).Timing, delta)
	window.Advance(&ecs.Resource[window.Window](worlds.Game).Timing, delta)

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

	worlds.Engine.ApplyCommands()
	worlds.Game.ApplyCommands()
}

// PostFrame finalizes the frame after the renderer has run: advance
// each world's tick (rolls event queues, ages change-detection
// watermarks) and clear the per-frame input deltas.
func PostFrame(worlds Worlds) {
	worlds.Engine.Step()
	worlds.Game.Step()
	render.InputBeginFrame(ecs.Resource[render.Input](worlds.Engine))
}
