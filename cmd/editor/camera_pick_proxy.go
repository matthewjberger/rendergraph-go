package main

import (
	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/render/asset"
	"github.com/matthewjberger/indigo/transform"
)

// spawnCameraPickProxy attaches an invisible child cube to the
// camera entity so the GPU pick can resolve clicks on the camera
// gizmo. The cube uses AlphaModePickProxy: mesh / prepass discard,
// OIT stamps entity_id only with zero color and reveal contribution.
// The proxy carries a render.PickProxy redirect so applySelection
// hands selection back to the owning camera entity.
func spawnCameraPickProxy(engine *ecs.World, camera ecs.Entity) {
	primitives, ok := ecs.Resource[asset.Primitives](engine)
	if !ok || primitives == nil {
		return
	}
	mask := ecs.MustMaskOf[transform.LocalTransform](engine) |
		ecs.MustMaskOf[transform.GlobalTransform](engine) |
		ecs.MustMaskOf[transform.LocalTransformDirty](engine) |
		ecs.MustMaskOf[transform.Parent](engine) |
		ecs.MustMaskOf[asset.RenderMesh](engine) |
		ecs.MustMaskOf[asset.Material](engine) |
		ecs.MustMaskOf[render.PickProxy](engine)
	proxy := engine.Spawn(mask)
	local := transform.IdentityLocalTransform()
	local.Scale = cameraGizmoPickExtents()
	ecs.Set(engine, proxy, local)
	ecs.Set(engine, proxy, transform.IdentityGlobalTransform())
	ecs.Set(engine, proxy, transform.Parent{Entity: camera})
	ecs.Set(engine, proxy, asset.RenderMesh{Mesh: primitives.UnitCube})
	ecs.Set(engine, proxy, asset.Material{
		BaseColor:              [4]float32{0, 0, 0, 0},
		BaseColorLayer:         asset.NoTextureLayer,
		NormalLayer:            asset.NoTextureLayer,
		MetallicRoughnessLayer: asset.NoTextureLayer,
		OcclusionLayer:         asset.NoTextureLayer,
		EmissiveLayer:          asset.NoTextureLayer,
		AlphaMode:              asset.AlphaModePickProxy,
		AlphaCutoff:            0.5,
		IOR:                    1.5,
		Unlit:                  true,
	})
	ecs.Set(engine, proxy, render.PickProxy{Target: camera})
}
