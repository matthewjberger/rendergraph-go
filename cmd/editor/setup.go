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
	ecs.SetResource(engine, NewKhronosBrowser())
	ecs.SetResource(engine, NewPolyhavenBrowser())
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

func editorApp() *app.App {
	return &app.App{
		ConfigureRenderGraph: func(world *ecs.World, renderer *render.Renderer) {
			skinning, err := pass.NewSkinningCompute(renderer.Device)
			if err != nil {
				log.Fatal(err)
			}
			ecs.SetResource(world, pass.SkinningComputeResource{Compute: skinning})

			arrays := ecs.MustResource[asset.MaterialTextureArraysResource](world).Arrays
			registry := ecs.MustResource[asset.MaterialRegistryResource](world).Registry
			ibl := ecs.MustResource[pass.IBLResource](world).IBL
			shadow := ecs.MustResource[pass.ShadowResource](world).Shadow
			spotShadow := ecs.MustResource[pass.SpotShadowResource](world).Shadow
			pointShadow := ecs.MustResource[pass.PointShadowResource](world).Shadow

			var bloomMipView func() *wgpu.TextureView
			var fxaaOutputID render.ResourceID

			steps := []struct {
				name string
				add  func() error
			}{
				{"skinning_compute", func() error { _, err := pass.AddSkinningComputePass(renderer, skinning); return err }},
				{"sky", func() error { _, err := pass.AddSkyPass(renderer); return err }},
				{"shadow_depth", func() error { _, err := pass.AddShadowDepthPass(renderer, shadow); return err }},
				{"spot_shadow", func() error { _, err := pass.AddSpotShadowPass(renderer, spotShadow); return err }},
				{"point_shadow", func() error { _, err := pass.AddPointShadowPass(renderer, pointShadow); return err }},
				{"mesh", func() error {
					_, err := pass.AddMeshPass(renderer, arrays, registry, ibl, shadow, spotShadow, pointShadow)
					return err
				}},
				{"skinned_mesh", func() error { _, err := pass.AddSkinnedMeshPass(renderer); return err }},
				{"pick_proxy", func() error { _, err := pass.AddPickProxyPass(renderer); return err }},
				{"picking", func() error { _, err := pass.AddPickingPass(renderer); return err }},
				{"selection_mask", func() error { _, err := pass.AddSelectionMaskPass(renderer); return err }},
				{"grid", func() error { _, err := pass.AddGridPass(renderer); return err }},
				{"lines", func() error { _, err := pass.AddLinesPass(renderer); return err }},
				{"outline", func() error { _, err := pass.AddOutlinePass(renderer); return err }},
				{"ssao", func() error { _, _, err := pass.AddSsaoPass(renderer, renderer.AspectRatio); return err }},
				{"oit_mesh", func() error { _, err := pass.AddOitMeshPass(renderer); return err }},
				{"skinned_mesh_oit", func() error { _, err := pass.AddSkinnedMeshOitPass(renderer); return err }},
				{"oit_composite", func() error { _, err := pass.AddOitCompositePass(renderer); return err }},
				{"auto_exposure", func() error { _, err := pass.AddAutoExposurePass(renderer); return err }},
				{"bloom", func() error {
					var err error
					_, bloomMipView, err = pass.AddBloomPass(renderer)
					return err
				}},
				{"postprocess", func() error { _, err := pass.AddPostProcessPass(renderer, bloomMipView); return err }},
				{"fxaa", func() error {
					var err error
					_, fxaaOutputID, err = pass.AddFxaaPass(renderer)
					return err
				}},
				{"gizmo", func() error { _, err := pass.AddGizmoPass(renderer, fxaaOutputID); return err }},
				{"ui_quad", func() error { _, err := pass.AddUiQuadPass(renderer, fxaaOutputID); return err }},
				{"ui_text", func() error { _, err := pass.AddUiTextPass(renderer, fxaaOutputID); return err }},
				{"present", func() error { _, err := pass.AddPresentPass(renderer, fxaaOutputID); return err }},
			}
			for _, step := range steps {
				if err := step.add(); err != nil {
					log.Fatalf("render graph: %s: %v", step.name, err)
				}
			}
		},
		PreRender: func(engine *ecs.World) {
			drawLightGizmos(engine)
			drawCameraGizmos(engine)
		},
	}
}
