package main

import (
	"log"
	"os"

	"github.com/cogentcore/webgpu/wgpu"

	"github.com/matthewjberger/indigo/app"
	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/internal/render/pass"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/render/asset"
	"github.com/matthewjberger/indigo/transform"
)

type PaletteResource struct {
	Palette brickPalette
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
	worlds, err := app.NewWorlds(renderer)
	if err != nil {
		log.Fatal(err)
	}
	engine := worlds.Engine

	ecs.SetResource(engine, breakoutCamera())

	ecs.Register[Paddle](worlds.Game)
	ecs.Register[Ball](worlds.Game)
	ecs.Register[Brick](worlds.Game)
	ecs.SetResource(worlds.Game, GameState{Lives: startingLives})

	worlds.EngineSchedule.Push("graphics_toggles", render.UpdateGraphicsToggles)
	worlds.EngineSchedule.Push("animations", asset.UpdateAnimationPlayers)
	worlds.EngineSchedule.Push("transform_propagation", transform.UpdateGlobalTransforms)

	worlds.GameSchedule.Push("input", breakoutInputSystem)
	worlds.GameSchedule.Push("ball", breakoutBallSystem)
	worlds.GameSchedule.Push("win", breakoutWinSystem)
	worlds.GameSchedule.Push("reset", breakoutResetSystem)
	worlds.GameSchedule.Push("sync", breakoutSyncSystem)

	ecs.SetResource(engine, buildBreakoutHud(worlds.UI))

	demo := breakoutApp()
	if demo.ConfigureRenderGraph != nil {
		demo.ConfigureRenderGraph(engine, renderer)
	}

	assets := ecs.MustResource[asset.MeshAssetsResource](engine).Assets
	palette, err := registerBreakoutPalette(renderer.Device, assets)
	if err != nil {
		log.Fatal(err)
	}
	ecs.SetResource(worlds.Game, PaletteResource{Palette: palette})
	spawnBreakoutScene(worlds, palette)
	spawnBreakoutSun(engine)

	if err := renderer.Graph.Compile(renderer.Device); err != nil {
		log.Fatal(err)
	}
	return worlds, demo
}

func breakoutCamera() render.Camera {
	camera := render.DefaultCamera()
	camera.Eye = transform.Vec3{0, 11, 4}
	camera.Target = transform.Vec3{0, 0, 0}
	camera.Up = transform.Vec3{0, 1, 0}
	camera.FovYRadians = 0.9
	return camera
}

func spawnBreakoutSun(engine *ecs.World) {
	mask := ecs.MustMaskOf[transform.LocalTransform](engine) |
		ecs.MustMaskOf[transform.GlobalTransform](engine) |
		ecs.MustMaskOf[transform.LocalTransformDirty](engine) |
		ecs.MustMaskOf[render.Light](engine)
	sun := engine.Spawn(mask)
	local := transform.IdentityLocalTransform()
	local.Rotation = transform.QuatFromAxisAngle(-1.1, transform.Vec3{1, 0, 0})
	ecs.Set(engine, sun, local)
	ecs.Set(engine, sun, transform.IdentityGlobalTransform())
	ecs.Set(engine, sun, render.Light{
		Type:      render.LightTypeDirectional,
		Color:     transform.Vec3{1.0, 0.98, 0.9},
		Intensity: 6.0,
	})
}

func breakoutApp() *app.App {
	return &app.App{
		ConfigureRenderGraph: func(world *ecs.World, renderer *render.Renderer) {
			arrays := ecs.MustResource[asset.MaterialTextureArraysResource](world).Arrays
			registry := ecs.MustResource[asset.MaterialRegistryResource](world).Registry
			ibl := ecs.MustResource[pass.IBLResource](world).IBL
			shadow := ecs.MustResource[pass.ShadowResource](world).Shadow
			spotShadow := ecs.MustResource[pass.SpotShadowResource](world).Shadow
			pointShadow := ecs.MustResource[pass.PointShadowResource](world).Shadow
			if _, err := pass.AddShadowDepthPass(renderer, shadow); err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddSpotShadowPass(renderer, spotShadow); err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddPointShadowPass(renderer, pointShadow); err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddMeshPass(renderer, arrays, registry, ibl, shadow, spotShadow, pointShadow); err != nil {
				log.Fatal(err)
			}
			if _, _, err := pass.AddSsaoPass(renderer, renderer.AspectRatio); err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddAutoExposurePass(renderer); err != nil {
				log.Fatal(err)
			}
			_, bloomMipView, err := pass.AddBloomPass(renderer)
			if err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddPostProcessPass(renderer, bloomMipView); err != nil {
				log.Fatal(err)
			}
			_, fxaaOutputID, err := pass.AddFxaaPass(renderer)
			if err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddUiQuadPass(renderer, fxaaOutputID); err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddUiTextPass(renderer, fxaaOutputID); err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddPresentPass(renderer, fxaaOutputID); err != nil {
				log.Fatal(err)
			}
		},
	}
}

func breakoutResetSystem(game *ecs.World) {
	state := ecs.MustResource[GameState](game)
	if !state.RequestReset {
		return
	}
	state.RequestReset = false
	engine := ecs.MustResource[app.EngineRef](game).World

	brickMask := ecs.MustMaskOf[Brick](game) | ecs.MustMaskOf[app.EngineEntity](game)
	var dead []ecs.Entity
	for entity := range game.Query(brickMask, 0) {
		dead = append(dead, entity)
	}
	for _, entity := range dead {
		app.DespawnLinked(engine, game, entity)
	}

	ballMask := ecs.MustMaskOf[Ball](game) | ecs.MustMaskOf[app.EngineEntity](game)
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

	palette := ecs.MustResource[PaletteResource](game).Palette
	respawnBricks(app.Worlds{Engine: engine, Game: game}, palette)
}

func respawnBricks(worlds app.Worlds, palette brickPalette) {
	engineMask := ecs.MustMaskOf[transform.LocalTransform](worlds.Engine) |
		ecs.MustMaskOf[transform.GlobalTransform](worlds.Engine) |
		ecs.MustMaskOf[transform.LocalTransformDirty](worlds.Engine) |
		ecs.MustMaskOf[asset.RenderMesh](worlds.Engine) |
		ecs.MustMaskOf[asset.Material](worlds.Engine)
	brickMask := ecs.MustMaskOf[Brick](worlds.Game) | ecs.MustMaskOf[app.EngineEntity](worlds.Game)

	brickWidth := brickHalfWidth * 2
	totalWidth := brickWidth * brickCols
	startX := -totalWidth*0.5 + brickHalfWidth
	for row := 0; row < brickRows; row++ {
		rowColor := palette.Rows[row%len(palette.Rows)]
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
			ecs.Set(worlds.Engine, engineEntity, asset.RenderMesh{Mesh: palette.Cube})
			ecs.Set(worlds.Engine, engineEntity, asset.AlbedoMaterial(rowColor))

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
