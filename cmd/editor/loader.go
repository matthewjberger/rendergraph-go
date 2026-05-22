// glTF load + scene-spawn helpers for the editor. The embedded
// DamagedHelmet.glb is the auto-loaded default at startup; the
// loadGltf* helpers also serve as the entry points for the
// wasm canvas drop callback and the native GLFW drop callback.
package main

import (
	"bytes"
	_ "embed"
	"fmt"
	"log"
	"path/filepath"
	"strings"

	"indigo/app"
	"indigo/ecs"
	"indigo/render"
	"indigo/render/asset"
)

// defaultGltf is the .glb the editor auto-loads at startup so the
// view isn't just primitives. Embedded so both the native build
// and the wasm build see it without depending on a filesystem
// path.
//
//go:embed assets/DamagedHelmet.glb
var defaultGltf []byte

// loadDefaultGltf spawns the embedded DamagedHelmet scene at
// startup. Logs and continues on error rather than aborting boot
// so a broken asset doesn't prevent the editor from launching.
func loadDefaultGltf(engine *ecs.World, renderer *render.Renderer) {
	if _, err := loadGltfBytes(engine, renderer, "DamagedHelmet.glb", defaultGltf); err != nil {
		log.Printf("gltf load failed: %v", err)
	}
}

// loadGltfBytes parses a glTF / glb buffer and spawns its scene
// through the shared [spawnLoadedSceneNamed] helper. Used by both
// the editor's startup auto-load and the wasm drop callback.
func loadGltfBytes(engine *ecs.World, renderer *render.Renderer, label string, data []byte) ([]ecs.Entity, error) {
	assets := ecs.MustResource[asset.MeshAssetsResource](engine).Assets
	arrays := ecs.MustResource[asset.MaterialTextureArraysResource](engine).Arrays
	scene, err := asset.LoadGltfReader(renderer.Device, renderer.Queue, assets, arrays, label, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	return spawnLoadedSceneNamed(engine, scene, label)
}

// loadGltfInto loads path via [asset.LoadGltfFile] and forwards to
// the bytes-based spawn helper. Native-only — wasm calls
// [loadGltfBytes] directly with bytes fetched from a drop event.
func loadGltfInto(engine *ecs.World, renderer *render.Renderer, path string) ([]ecs.Entity, error) {
	assets := ecs.MustResource[asset.MeshAssetsResource](engine).Assets
	arrays := ecs.MustResource[asset.MaterialTextureArraysResource](engine).Arrays
	scene, err := asset.LoadGltfFile(renderer.Device, renderer.Queue, assets, arrays, path)
	if err != nil {
		return nil, err
	}
	return spawnLoadedSceneNamed(engine, scene, filepath.Base(path))
}

func spawnLoadedSceneNamed(engine *ecs.World, scene *asset.LoadedScene, label string) ([]ecs.Entity, error) {
	entities := asset.SpawnLoadedScene(engine, scene)
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
	log.Printf("gltf loaded: %s (%d nodes, %d meshes, %d materials, %d animations)",
		label, len(scene.Nodes), len(scene.Meshes), len(scene.Materials), len(scene.Animations))
	if len(scene.Roots) > 0 {
		applyEntitySelection(engine, entities[scene.Roots[0]])
	}
	return entities, nil
}
