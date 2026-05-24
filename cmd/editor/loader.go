package main

import (
	"bytes"
	_ "embed"
	"fmt"
	"log"
	"path/filepath"
	"strings"
	"testing/fstest"

	"github.com/matthewjberger/indigo/app"
	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/render/asset"
	"github.com/matthewjberger/indigo/transform"
	"github.com/matthewjberger/indigo/ui"
)

//go:embed assets/DamagedHelmet.glb
var defaultGltf []byte

func loadDefaultGltf(engine *ecs.World, renderer *render.Renderer) {
	if _, err := loadGltfBytes(engine, renderer, "DamagedHelmet.glb", defaultGltf); err != nil {
		log.Printf("gltf load failed: %v", err)
	}
}

func loadGltfBytes(engine *ecs.World, renderer *render.Renderer, label string, data []byte) ([]ecs.Entity, error) {
	assets := ecs.MustResource[asset.MeshAssetsResource](engine).Assets
	skinnedAssets := ecs.MustResource[asset.SkinnedMeshAssetsResource](engine).Assets
	arrays := ecs.MustResource[asset.MaterialTextureArraysResource](engine).Arrays
	scene, err := asset.LoadGltfReader(renderer.Device, renderer.Queue, assets, skinnedAssets, arrays, label, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	return spawnLoadedSceneNamed(engine, renderer, scene, label)
}

func drainKhronosPending(worlds app.Worlds, renderer *render.Renderer) {
	browser := *ecs.MustResource[*KhronosBrowser](worlds.Engine)
	pending := browser.TakePending()
	if pending == nil {
		return
	}
	if _, err := loadKhronosPending(worlds.Engine, renderer, pending); err != nil {
		log.Printf("khronos load failed: %v", err)
		return
	}
	if worlds.UI != nil {
		ui.MarkLayoutDirty(worlds.UI)
	}
}

func loadKhronosPending(engine *ecs.World, renderer *render.Renderer, pending *PendingKhronos) ([]ecs.Entity, error) {
	assets := ecs.MustResource[asset.MeshAssetsResource](engine).Assets
	skinnedAssets := ecs.MustResource[asset.SkinnedMeshAssetsResource](engine).Assets
	arrays := ecs.MustResource[asset.MaterialTextureArraysResource](engine).Arrays
	opts := asset.LoadGltfOptions{Label: pending.DisplayName}
	if len(pending.Resources) > 0 {
		fsys := make(fstest.MapFS, len(pending.Resources))
		for name, data := range pending.Resources {
			fsys[name] = &fstest.MapFile{Data: data}
		}
		opts.FS = fsys
	}
	scene, err := asset.LoadGltfReaderOpts(renderer.Device, renderer.Queue, assets, skinnedAssets, arrays, bytes.NewReader(pending.GltfBytes), opts)
	if err != nil {
		return nil, err
	}
	return spawnLoadedSceneNamed(engine, renderer, scene, pending.DisplayName)
}

func loadGltfInto(engine *ecs.World, renderer *render.Renderer, path string) ([]ecs.Entity, error) {
	assets := ecs.MustResource[asset.MeshAssetsResource](engine).Assets
	skinnedAssets := ecs.MustResource[asset.SkinnedMeshAssetsResource](engine).Assets
	arrays := ecs.MustResource[asset.MaterialTextureArraysResource](engine).Arrays
	scene, err := asset.LoadGltfFile(renderer.Device, renderer.Queue, assets, skinnedAssets, arrays, path)
	if err != nil {
		return nil, err
	}
	return spawnLoadedSceneNamed(engine, renderer, scene, filepath.Base(path))
}

func spawnLoadedSceneNamed(engine *ecs.World, renderer *render.Renderer, scene *asset.LoadedScene, label string) ([]ecs.Entity, error) {
	viewerMode := ecs.MustResource[render.Graphics](engine).ViewerMode
	if viewerMode {
		clearLoadedScenes(engine)
	}
	entities := asset.SpawnLoadedScene(engine, scene, renderer.Device)
	baseName := strings.TrimSuffix(label, filepath.Ext(label))
	nameMask := ecs.MustMaskOf[app.Name](engine)
	for i, e := range entities {
		name := scene.Nodes[i].Name
		if name == "" {
			name = fmt.Sprintf("%s/node_%d", baseName, i)
		}
		engine.AddComponents(e, nameMask)
		ecs.Set(engine, e, app.Name{Value: name})
	}
	attachSceneLights(engine, scene, entities)
	attachSceneCameras(engine, scene, entities)
	log.Printf("gltf loaded: %s (%d nodes, %d meshes, %d materials, %d animations, %d lights, %d cameras)",
		label, len(scene.Nodes), len(scene.Meshes), len(scene.Materials), len(scene.Animations), len(scene.Lights), len(scene.Cameras))
	if len(scene.Roots) > 0 {
		applyEntitySelection(engine, transform.FindGroupRoot(engine, entities[scene.Roots[0]]))
	}
	if viewerMode {
		hud := ecs.MustResource[HudHandles](engine)
		hud.ModelRoots = loadedSceneRoots(engine)
		hud.FramePending = true
	}
	return entities, nil
}

func attachSceneLights(engine *ecs.World, scene *asset.LoadedScene, entities []ecs.Entity) {
	if len(scene.Lights) == 0 {
		return
	}
	lightMask := ecs.MustMaskOf[render.Light](engine)
	for i, node := range scene.Nodes {
		if node.LightIndex < 0 || node.LightIndex >= len(scene.Lights) {
			continue
		}
		src := scene.Lights[node.LightIndex]
		var ty render.LightType
		switch src.Type {
		case asset.LoadedLightPoint:
			ty = render.LightTypePoint
		case asset.LoadedLightSpot:
			ty = render.LightTypeSpot
		default:
			ty = render.LightTypeDirectional
		}
		light := render.Light{
			Type:           ty,
			Color:          transform.Vec3{src.Color[0], src.Color[1], src.Color[2]},
			Intensity:      src.Intensity,
			Range:          src.Range,
			InnerConeAngle: src.InnerConeAngle,
			OuterConeAngle: src.OuterConeAngle,
		}
		engine.AddComponents(entities[i], lightMask)
		ecs.Set(engine, entities[i], light)
	}
}

func attachSceneCameras(engine *ecs.World, scene *asset.LoadedScene, entities []ecs.Entity) {
	if len(scene.Cameras) == 0 {
		return
	}
	camMask := ecs.MustMaskOf[render.CameraMarker](engine)
	for i, node := range scene.Nodes {
		if node.CameraIndex < 0 || node.CameraIndex >= len(scene.Cameras) {
			continue
		}
		engine.AddComponents(entities[i], camMask)
		ecs.Set(engine, entities[i], render.CameraMarker{})
		spawnCameraPickProxy(engine, entities[i])
	}
}
