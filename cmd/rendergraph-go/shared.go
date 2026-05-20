// Shared bootstrap used by both the desktop (!js) and wasm (js)
// entry points. The engine world holds rendering data (transforms,
// RenderMesh, camera, input); the game world holds game state
// (spinner) plus an EngineEntity reference that links each game
// entity to its engine counterpart. Two [ecs.Schedule]s order
// systems on each world.
package main

import (
	"log"
	"math"
	"os"

	"github.com/cogentcore/webgpu/wgpu"

	"rendergraph-go/app"
	"rendergraph-go/ecs"
	"rendergraph-go/render"
	"rendergraph-go/transform"
	"rendergraph-go/window"
)

// EngineRef is the game-world resource that points at the engine
// world. Game systems read this to find the engine world they should
// sync into via the [app.SyncEngine...] helpers.
type EngineRef struct{ World *ecs.World }

// Worlds bundles the engine + game worlds and their schedules so
// platform main loops can pass them around as a unit.
type Worlds struct {
	Engine         *ecs.World
	Game           *ecs.World
	EngineSchedule *ecs.Schedule
	GameSchedule   *ecs.Schedule
}

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

// buildWorlds creates the engine + game worlds, registers components,
// installs typed resources, builds two schedules (one per world), and
// runs the app lifecycle hooks. The game world has an [EngineRef] so
// game systems can reach the engine world through the sync API.
func buildWorlds(renderer *render.Renderer) (Worlds, *app.App) {
	engine := ecs.New()
	ecs.Register[transform.LocalTransform](engine)
	ecs.Register[transform.GlobalTransform](engine)
	ecs.Register[transform.Parent](engine)
	ecs.Register[transform.LocalTransformDirty](engine)
	ecs.Register[render.RenderMesh](engine)

	ecs.SetResource(engine, window.Window{
		Viewport: window.ViewportSize{
			Width:  renderer.Config.Width,
			Height: renderer.Config.Height,
		},
	})
	ecs.SetResource(engine, render.RendererResource{Renderer: renderer})
	ecs.SetResource(engine, render.DefaultCamera())
	ecs.SetResource(engine, render.MeshAssetsResource{Assets: renderer.Meshes})
	ecs.SetResource(engine, render.Input{})
	ecs.SetResource(engine, render.DefaultPanOrbitController())
	ecs.SetResource(engine, render.DefaultGraphicsSettings())
	ecs.SetResource(engine, transform.NewPropagationState())

	game := ecs.New()
	ecs.Register[Spinner](game)
	ecs.Register[app.EngineEntity](game)
	ecs.SetResource(game, window.Window{})
	ecs.SetResource(game, EngineRef{World: engine})

	engineSchedule := ecs.NewSchedule()
	engineSchedule.Push("graphics_toggles", render.UpdateGraphicsToggles)
	engineSchedule.Push("pan_orbit_camera", render.UpdatePanOrbitCamera)
	engineSchedule.Push("transform_propagation", transform.UpdateGlobalTransforms)

	gameSchedule := ecs.NewSchedule()
	gameSchedule.Push("spinner", advanceSpinners)

	worlds := Worlds{
		Engine:         engine,
		Game:           game,
		EngineSchedule: engineSchedule,
		GameSchedule:   gameSchedule,
	}

	demo := spinnerDemo()
	if demo.Initialize != nil {
		demo.Initialize(engine)
	}
	if demo.ConfigureRenderGraph != nil {
		demo.ConfigureRenderGraph(engine, renderer)
	}
	initializeWorldEntities(worlds, []render.MeshHandle{
		renderer.UnitTriangle,
		renderer.UnitQuad,
		renderer.UnitCube,
	})
	if err := renderer.Graph.Compile(renderer.Device); err != nil {
		log.Fatal(err)
	}
	return worlds, demo
}

// tickFrame advances both worlds by delta. Updates window timing,
// runs the game schedule (writes to engine via the sync API), runs
// the engine schedule (pan-orbit camera then transform propagation),
// runs the demo's optional hooks, advances command + event queues on
// both worlds, then clears per-frame input deltas.
func tickFrame(worlds Worlds, demo *app.App, delta float32) {
	window.Advance(&ecs.Resource[window.Window](worlds.Engine).Timing, delta)
	window.Advance(&ecs.Resource[window.Window](worlds.Game).Timing, delta)

	worlds.GameSchedule.Run(worlds.Game)
	worlds.EngineSchedule.Run(worlds.Engine)

	if demo.RunSystems != nil {
		demo.RunSystems(worlds.Engine)
	}
	if demo.PreRender != nil {
		demo.PreRender(worlds.Engine)
	}

	worlds.Engine.ApplyCommands()
	worlds.Engine.Step()
	worlds.Game.ApplyCommands()
	worlds.Game.Step()

	ecs.Resource[render.Input](worlds.Engine).BeginFrame()
}

// spinnerDemo wires the render pipeline: mesh pass writing
// scene_color + depth, present pass blitting scene_color to the
// swapchain.
func spinnerDemo() *app.App {
	return &app.App{
		ConfigureRenderGraph: func(world *ecs.World, renderer *render.Renderer) {
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

			sky, err := render.NewSkyPass(renderer.Device, renderer.SurfaceFormat, renderer.AspectRatio)
			if err != nil {
				log.Fatal(err)
			}
			if err := renderer.Graph.AddPass(sky, []render.SlotBinding{
				{Slot: "color", ResourceID: renderer.SceneColorID},
			}); err != nil {
				log.Fatal(err)
			}

			mesh, err := render.NewMeshPass(renderer.Device, renderer.SurfaceFormat, renderer.AspectRatio)
			if err != nil {
				log.Fatal(err)
			}
			if err := renderer.Graph.AddPass(mesh, []render.SlotBinding{
				{Slot: "color", ResourceID: renderer.SceneColorID},
				{Slot: "depth", ResourceID: renderer.DepthID},
			}); err != nil {
				log.Fatal(err)
			}

			grid, err := render.NewGridPass(renderer.Device, renderer.SurfaceFormat, renderer.AspectRatio)
			if err != nil {
				log.Fatal(err)
			}
			if err := renderer.Graph.AddPass(grid, []render.SlotBinding{
				{Slot: "color", ResourceID: renderer.SceneColorID},
				{Slot: "depth", ResourceID: renderer.DepthID},
			}); err != nil {
				log.Fatal(err)
			}

			fxaa, err := render.NewFxaaPass(renderer.Device, renderer.SurfaceFormat)
			if err != nil {
				log.Fatal(err)
			}
			if err := renderer.Graph.AddPass(fxaa, []render.SlotBinding{
				{Slot: "input", ResourceID: renderer.SceneColorID},
				{Slot: "output", ResourceID: fxaaOutputID},
			}); err != nil {
				log.Fatal(err)
			}

			present, err := render.NewPresentPass(renderer.Device, renderer.SurfaceFormat)
			if err != nil {
				log.Fatal(err)
			}
			if err := renderer.Graph.AddPass(present, []render.SlotBinding{
				{Slot: "input", ResourceID: fxaaOutputID},
				{Slot: "output", ResourceID: renderer.SwapchainID},
			}); err != nil {
				log.Fatal(err)
			}
		},
	}
}

// initializeWorldEntities spawns a 5x5 grid of engine entities
// cycling through the supplied mesh handles for visual variety, plus
// a matching game entity per engine entity carrying its Spinner state
// and EngineEntity bridge.
func initializeWorldEntities(worlds Worlds, meshes []render.MeshHandle) {
	const (
		gridExtent  = 2
		gridSpacing = 1.5
	)

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
	engineRef := ecs.Resource[EngineRef](game)

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
