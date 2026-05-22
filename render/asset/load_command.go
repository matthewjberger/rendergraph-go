package asset

import (
	"bytes"
	"log"

	"indigo/ecs"
	"indigo/render"
)

// LoadGltfBytes is a render-queue command that decodes a glTF /
// glb byte buffer into the engine's asset registries at frame
// setup. The decode + GPU uploads happen inside the renderer's
// drain step, before any pass writes the swapchain, so UI
// callbacks (drop handlers, file-open menus, network responses)
// can fire-and-forget instead of stalling on synchronous GPU work.
//
// OnLoaded receives the resulting scene on the same drain
// callback. Callers that want to spawn the scene into entities
// supply a closure; callers that only care about side effects
// (caching, atlas warm-up) can pass nil.
//
// Strategic shape: this is the path that future async net /
// disk-based glTF loads will use. The decode currently happens
// inline because the engine has no worker threads yet; once a
// worker pool lands, decode + staging move there and the queued
// command only sees a finished CPU buffer ready for GPU upload.
type LoadGltfBytes struct {
	Label    string
	Bytes    []byte
	OnLoaded func(world *ecs.World, scene *LoadedScene)
}

// Apply decodes the bytes through [LoadGltfReader] and invokes
// OnLoaded. Errors log and skip rather than panicking so a bad
// drop doesn't take down the editor.
func (c LoadGltfBytes) Apply(world *ecs.World, renderer *render.Renderer) {
	assets := ecs.MustResource[MeshAssetsResource](world).Assets
	arrays := ecs.MustResource[MaterialTextureArraysResource](world).Arrays
	scene, err := LoadGltfReader(renderer.Device, renderer.Queue, assets, arrays, c.Label, bytes.NewReader(c.Bytes))
	if err != nil {
		log.Printf("load_gltf_bytes: %s: %v", c.Label, err)
		return
	}
	if c.OnLoaded != nil {
		c.OnLoaded(world, scene)
	}
}
