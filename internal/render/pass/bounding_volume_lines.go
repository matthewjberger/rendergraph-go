package pass

import (
	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/render/asset"
	"github.com/matthewjberger/indigo/transform"
)

func UpdateBoundingVolumeLines(world *ecs.World) {
	settings, ok := ecs.Resource[render.Graphics](world)
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
	mask := ecs.MustMaskOf[asset.RenderMesh](world) | ecs.MustMaskOf[transform.GlobalTransform](world)
	world.ForEach(mask, 0, func(_ ecs.Entity, table *ecs.Archetype, index int) {
		meshes, _ := ecs.Column[asset.RenderMesh](world, table)
		globals, _ := ecs.Column[transform.GlobalTransform](world, table)
		if meshes == nil || globals == nil {
			return
		}
		bounds := assets.Bounds(meshes[index].Mesh)
		if bounds == (asset.BoundingVolume{}) {
			return
		}
		lines.AddBox(bounds, &globals[index].Matrix, color)
	})
}
