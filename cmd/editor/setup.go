// Shared bootstrap used by both the desktop (!js) and wasm (js)
// entry points. Builds an engine world via [app.NewEngineWorld]
// plus a game world that holds Spinner state, wires schedules +
// the render graph, then spawns the demo scene.
//
// Sibling files in cmd/editor/:
//
//   - hud_layout.go   — HudHandles + buildHud + menu/builder helpers
//   - hud.go          — per-frame HudContext + refresh + input handlers
//   - world.go        — initializeWorldEntities + light orbs + spinner system
//   - loader.go       — defaultGltf embed + loadGltf* helpers
//   - picking.go      — handlePickResult + applyEntitySelection
//   - main.go         — native entry (GLFW)
//   - main_js.go      — wasm entry (canvas)
package main

import (
	"log"
	"os"

	"github.com/cogentcore/webgpu/wgpu"

	"github.com/matthewjberger/indigo/app"
	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/render/asset"
	"github.com/matthewjberger/indigo/render/pass"
	"github.com/matthewjberger/indigo/transform"
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

func buildWorlds(renderer *render.Renderer) (app.Worlds, *app.App) {
	worlds, err := app.NewWorlds(renderer)
	if err != nil {
		log.Fatal(err)
	}
	engine := worlds.Engine

	ecs.SetResource(engine, render.DefaultCamera())
	ecs.SetResource(engine, render.DefaultPanOrbitController())
	ecs.SetResource(engine, pass.NewPicking())
	installPickCycleState(engine)
	ecs.SetResource(engine, render.NewGizmos())

	ecs.Register[Spinner](worlds.Game)
	ecs.SetResource(engine, buildHud(worlds.UI))

	worlds.EngineSchedule.Push("graphics_toggles", render.UpdateGraphicsToggles)
	worlds.EngineSchedule.Push("gizmos", pass.UpdateGizmos)
	worlds.EngineSchedule.Push("pan_orbit_camera", render.UpdatePanOrbitCamera)
	worlds.EngineSchedule.Push("animations", asset.UpdateAnimationPlayers)
	worlds.EngineSchedule.Push("transform_propagation", transform.UpdateGlobalTransforms)
	worlds.EngineSchedule.Push("bounding_volume_lines", pass.UpdateBoundingVolumeLines)
	worlds.EngineSchedule.Push("normal_lines", pass.UpdateNormalLines)
	worlds.EngineSchedule.Push("skeleton_lines", pass.UpdateSkeletonLines)
	_ = advanceSpinners

	demo := editorApp()
	if demo.Initialize != nil {
		demo.Initialize(engine)
	}
	if demo.ConfigureRenderGraph != nil {
		demo.ConfigureRenderGraph(engine, renderer)
	}
	primitives := ecs.MustResource[asset.Primitives](engine)
	initializeWorldEntities(worlds,
		[]asset.MeshHandle{primitives.UnitTriangle, primitives.UnitQuad, primitives.UnitCube},
		[]string{"Triangle", "Quad", "Cube"},
		primitives.UnitSphere,
	)
	loadDefaultGltf(engine, renderer)

	if err := renderer.Graph.Compile(renderer.Device); err != nil {
		log.Fatal(err)
	}
	return worlds, demo
}

// editorApp wires the editor's render pipeline: sky -> mesh -> grid
// -> fxaa -> present. Each [pass.Add*Pass] helper builds the
// underlying [render.Pass] and registers it with the graph.
func editorApp() *app.App {
	return &app.App{
		ConfigureRenderGraph: func(world *ecs.World, renderer *render.Renderer) {
			skinning, err := pass.NewSkinningCompute(renderer.Device)
			if err != nil {
				log.Fatal(err)
			}
			ecs.SetResource(world, pass.SkinningComputeResource{Compute: skinning})
			if _, err := pass.AddSkinningComputePass(renderer, skinning); err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddSkyPass(renderer); err != nil {
				log.Fatal(err)
			}
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
			if _, err := pass.AddSkinnedMeshPass(renderer); err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddInstancedMeshPass(renderer); err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddMeshPass(renderer, arrays, registry, ibl, shadow, spotShadow, pointShadow); err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddPickProxyPass(renderer); err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddPickingPass(renderer); err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddSelectionMaskPass(renderer); err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddGridPass(renderer); err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddLinesPass(renderer); err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddOutlinePass(renderer); err != nil {
				log.Fatal(err)
			}
			if _, _, err := pass.AddSsaoPass(renderer, renderer.AspectRatio); err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddOitMeshPass(renderer); err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddSkinnedMeshOitPass(renderer); err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddOitCompositePass(renderer); err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddAutoExposurePass(renderer); err != nil {
				log.Fatal(err)
			}
			bloomPass, err := pass.AddBloomPass(renderer)
			if err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddPostProcessPass(renderer, bloomPass); err != nil {
				log.Fatal(err)
			}
			_, fxaaOutputID, err := pass.AddFxaaPass(renderer)
			if err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddGizmoPass(renderer, fxaaOutputID); err != nil {
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
		PreRender: func(engine *ecs.World) {
			drawLightGizmos(engine)
			drawCameraGizmos(engine)
		},
	}
}
