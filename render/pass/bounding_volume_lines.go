package pass

import (
	"indigo/ecs"
	"indigo/render"
	"indigo/render/asset"
	"indigo/transform"
)

// UpdateBoundingVolumeLines emits one wireframe AABB per entity that
// has a RenderMesh + GlobalTransform when [render.GraphicsSettings.ShowBounds]
// is true. Reads the mesh's local-space AABB from MeshAssets and
// pushes 12 world-space line segments into the [Lines] resource for
// the next line-pass frame.
func UpdateBoundingVolumeLines(world *ecs.World) {
	settings, ok := ecs.Resource[render.GraphicsSettings](world)
	if !ok || !settings.ShowBounds {
		return
	}
	linesRes, ok := ecs.Resource[LinesResource](world)
	if !ok {
		return
	}
	assetsRes, ok := ecs.Resource[asset.MeshAssetsResource](world)
	if !ok {
		return
	}
	assets := assetsRes.Assets
	lines := linesRes.Lines

	color := [4]float32{0.4, 0.95, 0.55, 0.9}
	meshMask := ecs.MustMaskOf[asset.RenderMesh](world)
	world.ForEach(meshMask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		mesh, ok := ecs.Get[asset.RenderMesh](world, entity)
		if !ok {
			return
		}
		global, ok := ecs.Get[transform.GlobalTransform](world, entity)
		if !ok {
			return
		}
		bounds := assets.Bounds(mesh.Mesh)
		if bounds == (asset.BoundingVolume{}) {
			return
		}
		lines.AddBox(bounds, &global.Matrix, color)
	})
}
