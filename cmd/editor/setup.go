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
	"indigo/ui"
	"indigo/window"
)

type HudHandles struct {
	StatusLabel     ecs.Entity
	ClearButton     ecs.Entity
	TranslateButton ecs.Entity
	RotateButton    ecs.Entity
	ScaleButton     ecs.Entity
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

func buildWorlds(renderer *render.Renderer) (app.Worlds, *app.App) {
	engine, err := app.NewEngineWorld(renderer)
	if err != nil {
		log.Fatal(err)
	}
	ecs.SetResource(engine, render.DefaultCamera())
	ecs.SetResource(engine, render.DefaultPanOrbitController())
	ecs.SetResource(engine, render.NewPicking())
	ecs.SetResource(engine, render.NewGizmos())

	game := ecs.New()
	ecs.Register[Spinner](game)
	ecs.Register[app.EngineEntity](game)
	ecs.SetResource(game, window.Window{})
	ecs.SetResource(game, app.EngineRef{World: engine})

	uiWorld := ui.NewWorld(renderer.Config.Width, renderer.Config.Height)
	ecs.SetResource(engine, ui.WorldRef{World: uiWorld})
	hud := buildHud(uiWorld)
	ecs.SetResource(engine, hud)

	engineSchedule := ecs.NewSchedule()
	engineSchedule.Push("graphics_toggles", render.UpdateGraphicsToggles)
	engineSchedule.Push("gizmos", render.UpdateGizmos)
	engineSchedule.Push("pan_orbit_camera", render.UpdatePanOrbitCamera)
	engineSchedule.Push("transform_propagation", transform.UpdateGlobalTransforms)

	// Editor leaves cubes stationary; manipulation happens via
	// gizmos rather than the spinner system. advanceSpinners is
	// kept around for reference but not scheduled.
	gameSchedule := ecs.NewSchedule()
	_ = advanceSpinners

	worlds := app.Worlds{
		Engine:         engine,
		Game:           game,
		UI:             uiWorld,
		EngineSchedule: engineSchedule,
		GameSchedule:   gameSchedule,
		UISchedule:     ui.NewSchedule(),
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
			if _, err := render.AddGizmoPass(renderer, fxaaOutputID); err != nil {
				log.Fatal(err)
			}
			if _, err := render.AddUiQuadPass(renderer, fxaaOutputID); err != nil {
				log.Fatal(err)
			}
			if _, err := render.AddUiTextPass(renderer, fxaaOutputID); err != nil {
				log.Fatal(err)
			}
			if _, err := render.AddPresentPass(renderer, fxaaOutputID); err != nil {
				log.Fatal(err)
			}
		},
	}
}

func buildHud(world *ecs.World) HudHandles {
	b := ui.NewBuilder(world)

	topBar := b.Node(ui.Node{
		X:       0,
		Y:       12,
		Width:   468,
		Height:  44,
		Anchor:  ui.AnchorTopCenter,
		Layout:  ui.LayoutRow,
		Padding: 6,
		Spacing: 6,
	}).Color(ui.Color{RGBA: [4]float32{0.08, 0.09, 0.12, 0.9}}).Entity()
	b.Push(topBar)
	translate := buildModeButton(b, "TRANSLATE")
	rotate := buildModeButton(b, "ROTATE")
	scale := buildModeButton(b, "SCALE")
	b.Pop()

	panel := b.Node(ui.Node{
		X:       16,
		Y:       16,
		Width:   220,
		Height:  76,
		Anchor:  ui.AnchorBottomLeft,
		Layout:  ui.LayoutColumn,
		Padding: 8,
		Spacing: 6,
	}).Color(ui.Color{RGBA: [4]float32{0.08, 0.09, 0.12, 0.85}}).Entity()

	b.Push(panel)
	label := b.Node(ui.Node{Width: 200, Height: 16}).Text(ui.Text{
		Content: "Pick a cube",
		Color:   [4]float32{0.95, 0.96, 0.98, 1},
		Scale:   1.6,
	}).Entity()

	clear := b.Node(ui.Node{Width: 96, Height: 24}).
		Color(ui.Color{RGBA: [4]float32{0.18, 0.32, 0.55, 1}}).
		Interactive().
		Text(ui.Text{
			Content: "Clear",
			Color:   [4]float32{0.98, 0.98, 1, 1},
			Scale:   1.4,
		}).Entity()
	b.Pop()

	return HudHandles{
		StatusLabel:     label,
		ClearButton:     clear,
		TranslateButton: translate,
		RotateButton:    rotate,
		ScaleButton:     scale,
	}
}

func buildModeButton(b *ui.Builder, text string) ecs.Entity {
	return b.Node(ui.Node{Width: 144, Height: 32}).
		Color(ui.Color{RGBA: [4]float32{0.18, 0.21, 0.28, 1}}).
		Interactive().
		Text(ui.Text{
			Content: text,
			Color:   [4]float32{0.95, 0.96, 0.98, 1},
			Scale:   1.6,
		}).Entity()
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
