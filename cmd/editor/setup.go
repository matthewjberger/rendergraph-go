// Shared bootstrap used by both the desktop (!js) and wasm (js)
// entry points. Builds an engine world via [app.NewEngineWorld] plus
// a game world that holds Spinner state, wires schedules + the
// render graph, then spawns the demo scene.
package main

import (
	"log"
	"math"
	"os"

	"github.com/cogentcore/webgpu/wgpu"

	"indigo/app"
	"indigo/ecs"
	"indigo/render"
	"indigo/transform"
	"indigo/window"
)

func setupLogging() {
	switch os.Getenv("WGPU_LOG_LEVEL") {
	case "OFF":
		wgpu.SetLogLevel(wgpu.LogLevelOff)
	case "ERROR":
		wgpu.SetLogLevel(wgpu.LogLevelError)
	case "WARN":
		wgpu.SetLogLevel(wgpu.LogLevelWarn)
	case "INFO":
		wgpu.SetLogLevel(wgpu.LogLevelInfo)
	case "DEBUG":
		wgpu.SetLogLevel(wgpu.LogLevelDebug)
	case "TRACE":
		wgpu.SetLogLevel(wgpu.LogLevelTrace)
	}
}

// buildWorlds creates the engine + game worlds, installs editor-
// specific resources on top of the engine defaults, builds two
// schedules, compiles the render graph, and spawns the demo scene.
// Returns [app.Worlds] plus the editor [app.App] (its lifecycle hooks
// drive the render-graph wiring).
func buildWorlds(renderer *render.Renderer) (app.Worlds, *app.App) {
	engine, err := app.NewEngineWorld(renderer)
	if err != nil {
		log.Fatal(err)
	}
	ecs.SetResource(engine, render.DefaultCamera())
	ecs.SetResource(engine, render.DefaultPanOrbitController())
	ecs.SetResource(engine, render.NewPicking())

	game := ecs.New()
	ecs.Register[Spinner](game)
	ecs.Register[app.EngineEntity](game)
	ecs.SetResource(game, window.Window{})
	ecs.SetResource(game, app.EngineRef{World: engine})

	engineSchedule := ecs.NewSchedule()
	engineSchedule.Push("graphics_toggles", render.UpdateGraphicsToggles)
	engineSchedule.Push("pan_orbit_camera", render.UpdatePanOrbitCamera)
	engineSchedule.Push("transform_propagation", transform.UpdateGlobalTransforms)

	gameSchedule := ecs.NewSchedule()
	gameSchedule.Push("spinner", advanceSpinners)

	worlds := app.Worlds{
		Engine:         engine,
		Game:           game,
		EngineSchedule: engineSchedule,
		GameSchedule:   gameSchedule,
	}

	demo := editorApp()
	if demo.Initialize != nil {
		demo.Initialize(engine)
	}
	if demo.ConfigureRenderGraph != nil {
		demo.ConfigureRenderGraph(engine, renderer)
	}
	primitives := ecs.Resource[render.Primitives](engine)
	initializeWorldEntities(worlds, []render.MeshHandle{
		primitives.UnitTriangle,
		primitives.UnitQuad,
		primitives.UnitCube,
	})
	if err := renderer.Graph.Compile(renderer.Device); err != nil {
		log.Fatal(err)
	}
	return worlds, demo
}

// editorApp wires the editor's render pipeline: sky -> mesh -> grid
// -> fxaa -> present. Each [render.Add*Pass] helper builds the
// underlying [render.Pass] and registers it with the graph.
func editorApp() *app.App {
	return &app.App{
		ConfigureRenderGraph: func(world *ecs.World, renderer *render.Renderer) {
			if _, err := render.AddSkyPass(renderer); err != nil {
				log.Fatal(err)
			}
			if _, err := render.AddMeshPass(renderer); err != nil {
				log.Fatal(err)
			}
			if _, err := render.AddPickingPass(renderer); err != nil {
				log.Fatal(err)
			}
			if _, err := render.AddSelectionMaskPass(renderer); err != nil {
				log.Fatal(err)
			}
			if _, err := render.AddGridPass(renderer); err != nil {
				log.Fatal(err)
			}
			if _, err := render.AddOutlinePass(renderer); err != nil {
				log.Fatal(err)
			}
			_, fxaaOutputID, err := render.AddFxaaPass(renderer)
			if err != nil {
				log.Fatal(err)
			}
			if _, err := render.AddPresentPass(renderer, fxaaOutputID); err != nil {
				log.Fatal(err)
			}
		},
	}
}

// initializeWorldEntities spawns a 5x5 grid of engine entities
// cycling through the supplied mesh handles for visual variety, plus
// a matching game entity per engine entity carrying its Spinner state
// and EngineEntity bridge. Also spawns a sun-like directional light
// pointing straight down.
func initializeWorldEntities(worlds app.Worlds, meshes []render.MeshHandle) {
	const (
		gridExtent  = 2
		gridSpacing = 1.5
	)

	lightMask := ecs.MaskOf[transform.LocalTransform](worlds.Engine) |
		ecs.MaskOf[transform.GlobalTransform](worlds.Engine) |
		ecs.MaskOf[transform.LocalTransformDirty](worlds.Engine) |
		ecs.MaskOf[render.Light](worlds.Engine)

	sun := worlds.Engine.Spawn(lightMask)
	sunTransform := transform.IdentityLocalTransform()
	sunTransform.Rotation = transform.QuatFromAxisAngle(-float32(math.Pi/2), transform.Vec3{1, 0, 0})
	ecs.Set(worlds.Engine, sun, sunTransform)
	ecs.Set(worlds.Engine, sun, transform.IdentityGlobalTransform())
	ecs.Set(worlds.Engine, sun, render.Light{
		Type:      render.LightTypeDirectional,
		Color:     transform.Vec3{1.0, 0.95, 0.85},
		Intensity: 1.4,
	})

	engineMask := ecs.MaskOf[transform.LocalTransform](worlds.Engine) |
		ecs.MaskOf[transform.GlobalTransform](worlds.Engine) |
		ecs.MaskOf[transform.LocalTransformDirty](worlds.Engine) |
		ecs.MaskOf[render.RenderMesh](worlds.Engine)

	gameMask := ecs.MaskOf[Spinner](worlds.Game) |
		ecs.MaskOf[app.EngineEntity](worlds.Game)

	index := 0
	for x := -gridExtent; x <= gridExtent; x++ {
		for z := -gridExtent; z <= gridExtent; z++ {
			engineEntity := worlds.Engine.Spawn(engineMask)
			local := transform.FromTranslation(transform.Vec3{
				float32(x) * gridSpacing,
				0,
				float32(z) * gridSpacing,
			})
			local.Rotation = transform.QuatFromAxisAngle(
				float32(index)*0.4,
				transform.Vec3{0, 1, 0},
			)
			ecs.Set(worlds.Engine, engineEntity, local)
			ecs.Set(worlds.Engine, engineEntity, transform.IdentityGlobalTransform())
			ecs.Set(worlds.Engine, engineEntity, render.RenderMesh{Mesh: meshes[index%len(meshes)]})

			gameEntity := worlds.Game.Spawn(gameMask)
			ecs.Set(worlds.Game, gameEntity, Spinner{
				Axis:     transform.Vec3{0, 1, 0},
				Speed:    float32(math.Pi / 2),
				Rotation: local.Rotation,
			})
			ecs.Set(worlds.Game, gameEntity, app.EngineEntity{Entity: engineEntity})

			index++
		}
	}
}

// advanceSpinners is the game-side system. Walks the (Spinner |
// EngineEntity) archetype with [ecs.World.ForEach] for table-level
// column access, accumulates per-entity rotation, and writes the
// result to the engine world through [app.SyncEngineRotation]. Game
// state is the source of truth; reaching across into the engine world
// goes through the named sync API.
func advanceSpinners(game *ecs.World) {
	delta := ecs.Resource[window.Window](game).Timing.DeltaSeconds
	engineRef := ecs.Resource[app.EngineRef](game)

	spinnerMask := ecs.MaskOf[Spinner](game) | ecs.MaskOf[app.EngineEntity](game)

	game.ForEach(spinnerMask, 0, func(_ ecs.Entity, table *ecs.Archetype, index int) {
		spinners, _ := ecs.Column[Spinner](game, table)
		links, _ := ecs.Column[app.EngineEntity](game, table)

		s := &spinners[index]
		step := transform.QuatFromAxisAngle(s.Speed*delta, s.Axis)
		s.Rotation = step.Mul(s.Rotation).Normalize()

		app.SyncEngineRotation(engineRef.World, links[index], s.Rotation)
	})
}
