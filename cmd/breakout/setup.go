// Setup wires the breakout app: engine + game worlds, the
// breakout-specific camera and lighting, the render-graph
// configuration (mesh -> fxaa -> present), and the brick wall.
package main

import (
	"log"
	"os"

	"github.com/cogentcore/webgpu/wgpu"

	"indigo/app"
	"indigo/ecs"
	"indigo/render"
	"indigo/transform"
	"indigo/window"
)

// EngineRef on the game world points at the engine world. Game
// systems read this to call [app.SyncEngine*] helpers.
type EngineRef struct{ World *ecs.World }

// MeshSet is stashed as a resource so the reset system can spawn a
// fresh wall of bricks with the right per-row colors.
type MeshSet struct {
	Meshes brickMeshes
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

// buildWorlds creates engine + game worlds, layers breakout-specific
// camera and game state on top of the engine defaults, builds two
// schedules, configures the render graph, spawns the scene, and
// compiles. Returns [app.Worlds] plus the breakout [app.App].
func buildWorlds(renderer *render.Renderer) (app.Worlds, *app.App) {
	engine := app.NewEngineWorld(renderer)
	ecs.SetResource(engine, breakoutCamera())

	game := ecs.New()
	ecs.Register[Paddle](game)
	ecs.Register[Ball](game)
	ecs.Register[Brick](game)
	ecs.Register[app.EngineEntity](game)
	ecs.SetResource(game, window.Window{})
	ecs.SetResource(game, EngineRef{World: engine})
	ecs.SetResource(game, GameState{Lives: startingLives})

	engineSchedule := ecs.NewSchedule()
	engineSchedule.Push("graphics_toggles", render.UpdateGraphicsToggles)
	engineSchedule.Push("transform_propagation", transform.UpdateGlobalTransforms)

	gameSchedule := ecs.NewSchedule()
	gameSchedule.Push("input", breakoutInputSystem)
	gameSchedule.Push("ball", breakoutBallSystem)
	gameSchedule.Push("win", breakoutWinSystem)
	gameSchedule.Push("reset", breakoutResetSystem)
	gameSchedule.Push("sync", breakoutSyncSystem)

	worlds := app.Worlds{
		Engine:         engine,
		Game:           game,
		EngineSchedule: engineSchedule,
		GameSchedule:   gameSchedule,
	}

	demo := breakoutApp()
	if demo.ConfigureRenderGraph != nil {
		demo.ConfigureRenderGraph(engine, renderer)
	}

	meshes, err := registerBreakoutMeshes(renderer)
	if err != nil {
		log.Fatal(err)
	}
	ecs.SetResource(game, MeshSet{Meshes: meshes})
	spawnBreakoutScene(worlds, meshes)
	spawnBreakoutSun(engine)

	if err := renderer.Graph.Compile(renderer.Device); err != nil {
		log.Fatal(err)
	}
	return worlds, demo
}

// breakoutCamera returns a near-top-down camera tilted forward a bit
// so block tops + front faces are both visible, giving the
// directional light something to shade.
func breakoutCamera() render.Camera {
	camera := render.DefaultCamera()
	camera.Eye = transform.Vec3{0, 11, 4}
	camera.Target = transform.Vec3{0, 0, 0}
	camera.Up = transform.Vec3{0, 1, 0}
	camera.FovYRadians = 0.9
	return camera
}

// spawnBreakoutSun puts a directional light on the engine world
// pointing diagonally down (mostly -Y with a -Z lean) so the brick
// tops and the brick front faces both pick up Lambert shading.
func spawnBreakoutSun(engine *ecs.World) {
	mask := ecs.MaskOf[transform.LocalTransform](engine) |
		ecs.MaskOf[transform.GlobalTransform](engine) |
		ecs.MaskOf[transform.LocalTransformDirty](engine) |
		ecs.MaskOf[render.Light](engine)
	sun := engine.Spawn(mask)
	local := transform.IdentityLocalTransform()
	local.Rotation = transform.QuatFromAxisAngle(-1.1, transform.Vec3{1, 0, 0})
	ecs.Set(engine, sun, local)
	ecs.Set(engine, sun, transform.IdentityGlobalTransform())
	ecs.Set(engine, sun, render.Light{
		Type:      render.LightTypeDirectional,
		Color:     transform.Vec3{1.0, 0.98, 0.9},
		Intensity: 1.6,
	})
}

// breakoutApp wires the render pipeline: mesh -> fxaa -> present.
// No sky or grid; mesh is the first writer of scene_color and its
// clear-on-load supplies the dark background.
func breakoutApp() *app.App {
	return &app.App{
		ConfigureRenderGraph: func(world *ecs.World, renderer *render.Renderer) {
			if _, err := render.AddMeshPass(renderer); err != nil {
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

// breakoutResetSystem rebuilds the brick wall and resets the ball
// when the input system sets [GameState.RequestReset].
func breakoutResetSystem(game *ecs.World) {
	state := ecs.Resource[GameState](game)
	if !state.RequestReset {
		return
	}
	state.RequestReset = false
	engine := ecs.Resource[EngineRef](game).World

	brickMask := ecs.MaskOf[Brick](game) | ecs.MaskOf[app.EngineEntity](game)
	var dead []ecs.Entity
	for entity := range game.Query(brickMask, 0) {
		dead = append(dead, entity)
	}
	for _, entity := range dead {
		app.DespawnLinked(engine, game, entity)
	}

	ballMask := ecs.MaskOf[Ball](game) | ecs.MaskOf[app.EngineEntity](game)
	game.ForEach(ballMask, 0, func(_ ecs.Entity, table *ecs.Archetype, index int) {
		balls, _ := ecs.Column[Ball](game, table)
		b := &balls[index]
		b.Launched = false
		b.Velocity = transform.Vec3{0, 0, 0}
	})

	state.Score = 0
	state.Lives = startingLives
	state.Won = false
	state.Lost = false
	state.Started = false

	meshes := ecs.Resource[MeshSet](game).Meshes
	respawnBricks(app.Worlds{Engine: engine, Game: game}, meshes)
}

func respawnBricks(worlds app.Worlds, meshes brickMeshes) {
	engineMask := ecs.MaskOf[transform.LocalTransform](worlds.Engine) |
		ecs.MaskOf[transform.GlobalTransform](worlds.Engine) |
		ecs.MaskOf[transform.LocalTransformDirty](worlds.Engine) |
		ecs.MaskOf[render.RenderMesh](worlds.Engine)
	brickMask := ecs.MaskOf[Brick](worlds.Game) | ecs.MaskOf[app.EngineEntity](worlds.Game)

	brickWidth := brickHalfWidth * 2
	totalWidth := brickWidth * brickCols
	startX := -totalWidth*0.5 + brickHalfWidth
	for row := 0; row < brickRows; row++ {
		rowMesh := meshes.Rows[row%len(meshes.Rows)]
		score := (brickRows - row) * 10
		z := float32(brickStartZ) + float32(row)*brickRowGap
		for col := 0; col < brickCols; col++ {
			x := startX + float32(col)*brickWidth
			pos := transform.Vec3{x, 0, z}

			engineEntity := worlds.Engine.Spawn(engineMask)
			local := transform.FromTranslation(pos)
			local.Scale = transform.Vec3{brickHalfWidth * 2, brickHalfY * 2, brickHalfDepth * 2}
			ecs.Set(worlds.Engine, engineEntity, local)
			ecs.Set(worlds.Engine, engineEntity, transform.IdentityGlobalTransform())
			ecs.Set(worlds.Engine, engineEntity, render.RenderMesh{Mesh: rowMesh})

			gameEntity := worlds.Game.Spawn(brickMask)
			ecs.Set(worlds.Game, gameEntity, Brick{
				Position: pos,
				HalfSize: transform.Vec3{brickHalfWidth, brickHalfY, brickHalfDepth},
				Score:    score,
			})
			ecs.Set(worlds.Game, gameEntity, app.EngineEntity{Entity: engineEntity})
		}
	}
}
