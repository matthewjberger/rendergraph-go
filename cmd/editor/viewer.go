package main

import (
	"math"

	"github.com/go-gl/mathgl/mgl32"

	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/render/asset"
	"github.com/matthewjberger/indigo/transform"
)

func loadedSceneRoots(engine *ecs.World) []ecs.Entity {
	rootMask := ecs.MustMaskOf[transform.GroupRoot](engine)
	var roots []ecs.Entity
	engine.ForEach(rootMask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		roots = append(roots, entity)
	})
	return roots
}

func collectSubtree(engine *ecs.World, entity ecs.Entity, out []ecs.Entity) []ecs.Entity {
	out = append(out, entity)
	for _, child := range transform.QueryChildren(engine, entity) {
		out = collectSubtree(engine, child, out)
	}
	return out
}

func clearLoadedScenes(engine *ecs.World) {
	transformMask := ecs.MustMaskOf[transform.GlobalTransform](engine)
	keepMask := ecs.MustMaskOf[viewerKeyLight](engine)
	var doomed []ecs.Entity
	engine.ForEach(transformMask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		if engine.HasComponents(entity, keepMask) {
			return
		}
		doomed = append(doomed, entity)
	})
	if len(doomed) == 0 {
		return
	}
	for _, entity := range doomed {
		engine.Despawn(entity)
	}
	transform.InvalidateChildrenCache(engine)
	applySelection(engine, 0)
}

func frameCameraOnRoots(engine *ecs.World, roots []ecs.Entity) {
	if len(roots) == 0 {
		return
	}

	meshAssets := ecs.MustResource[asset.MeshAssetsResource](engine).Assets
	skinnedAssets := ecs.MustResource[asset.SkinnedMeshAssetsResource](engine).Assets

	minPoint := mgl32.Vec3{math.MaxFloat32, math.MaxFloat32, math.MaxFloat32}
	maxPoint := mgl32.Vec3{-math.MaxFloat32, -math.MaxFloat32, -math.MaxFloat32}
	found := false

	accumulate := func(entity ecs.Entity, bounds asset.BoundingVolume) {
		global, ok := ecs.Get[transform.GlobalTransform](engine, entity)
		if !ok {
			return
		}
		for corner := 0; corner < 8; corner++ {
			local := mgl32.Vec3{
				pickCorner(corner, 0, bounds.Min[0], bounds.Max[0]),
				pickCorner(corner, 1, bounds.Min[1], bounds.Max[1]),
				pickCorner(corner, 2, bounds.Min[2], bounds.Max[2]),
			}
			world := global.Matrix.Mul4x1(local.Vec4(1)).Vec3()
			minPoint = mgl32.Vec3{minf(minPoint[0], world[0]), minf(minPoint[1], world[1]), minf(minPoint[2], world[2])}
			maxPoint = mgl32.Vec3{maxf(maxPoint[0], world[0]), maxf(maxPoint[1], world[1]), maxf(maxPoint[2], world[2])}
			found = true
		}
	}

	var entities []ecs.Entity
	for _, root := range roots {
		entities = collectSubtree(engine, root, entities)
	}
	for _, entity := range entities {
		if mesh, ok := ecs.Get[asset.RenderMesh](engine, entity); ok {
			accumulate(entity, meshAssets.Bounds(mesh.Mesh))
		}
		if mesh, ok := ecs.Get[asset.InstancedMesh](engine, entity); ok {
			accumulate(entity, meshAssets.Bounds(mesh.Mesh))
		}
		if mesh, ok := ecs.Get[asset.SkinnedMesh](engine, entity); ok {
			if entry, ok := skinnedAssets.Lookup(mesh.Mesh); ok {
				accumulate(entity, entry.Bounds)
			}
		}
	}

	if !found {
		return
	}

	center := minPoint.Add(maxPoint).Mul(0.5)
	halfDiagonal := maxPoint.Sub(minPoint).Mul(0.5).Len()
	if halfDiagonal < 1e-3 {
		halfDiagonal = 1e-3
	}

	camera := ecs.MustResource[render.Camera](engine)
	halfFov := camera.FovYRadians * 0.5
	radius := halfDiagonal
	if sin := float32(math.Sin(float64(halfFov))); sin > 1e-4 {
		radius = halfDiagonal / sin
	}
	radius = radius * 1.2

	controller := ecs.MustResource[render.PanOrbitController](engine)
	controller.TargetFocus = center
	controller.TargetRadius = radius
	controller.TargetYaw = 0
	controller.TargetPitch = 0.3
	controller.ZoomLower = halfDiagonal * 0.001
	if controller.ZoomUpper < radius*1.5 {
		controller.ZoomUpper = radius * 1.5
	}
}

func pickCorner(corner, axis int, lo, hi float32) float32 {
	if corner&(1<<axis) != 0 {
		return hi
	}
	return lo
}

func minf(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func maxf(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
