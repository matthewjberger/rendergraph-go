package asset

import (
	"github.com/go-gl/mathgl/mgl32"

	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/transform"
)

// LodLevel is one entry in a LodChain: the mesh to draw while the object's
// projected screen diameter is at least MinScreenPixels. Levels are ordered
// highest detail first; the last level is used below all thresholds.
type LodLevel struct {
	Mesh            MeshHandle
	MinScreenPixels float32
}

// LodChain selects a RenderMesh per frame by projected screen size, matching
// nightshade's MeshLodChain / mesh_culling.wgsl LOD selection.
type LodChain struct {
	Levels []LodLevel
}

// UpdateLodSelection picks each LodChain entity's active mesh from its screen
// diameter and writes it into the entity's RenderMesh.
func UpdateLodSelection(world *ecs.World) {
	camera, ok := ecs.Resource[render.Camera](world)
	if !ok || camera == nil {
		return
	}
	rendererResource, ok := ecs.Resource[render.RendererResource](world)
	if !ok || rendererResource.Renderer == nil {
		return
	}
	renderer := rendererResource.Renderer
	aspect := renderer.AspectRatio()
	screenHeight := float32(renderer.Config.Height)
	projection := render.CameraProjection(camera, aspect)
	viewProjection := projection.Mul4(render.CameraView(camera))
	projectionScaleY := projection[5]
	meshAssets := ecs.MustResource[MeshAssetsResource](world).Assets

	mask := ecs.MustMaskOf[LodChain](world)
	world.ForEach(mask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		chain, ok := ecs.Get[LodChain](world, entity)
		if !ok || len(chain.Levels) == 0 {
			return
		}
		mesh, ok := ecs.GetMut[RenderMesh](world, entity)
		if !ok {
			return
		}
		global, ok := ecs.Get[transform.GlobalTransform](world, entity)
		if !ok {
			return
		}
		position := transform.GlobalTransformTranslation(global)
		radius := meshAssets.Bounds(chain.Levels[0].Mesh).Radius() * maxAxisScale(global.Matrix)

		clip := viewProjection.Mul4x1(mgl32.Vec4{position.X(), position.Y(), position.Z(), 1})
		screenDiameter := float32(0)
		if clip.W() > 0 {
			screenDiameter = radius * projectionScaleY * screenHeight / clip.W()
		}

		level := len(chain.Levels) - 1
		for index := 0; index < len(chain.Levels)-1; index++ {
			if screenDiameter >= chain.Levels[index].MinScreenPixels {
				level = index
				break
			}
		}
		mesh.Mesh = chain.Levels[level].Mesh
	})
}

func maxAxisScale(matrix mgl32.Mat4) float32 {
	x := mgl32.Vec3{matrix[0], matrix[1], matrix[2]}.Len()
	y := mgl32.Vec3{matrix[4], matrix[5], matrix[6]}.Len()
	z := mgl32.Vec3{matrix[8], matrix[9], matrix[10]}.Len()
	max := x
	if y > max {
		max = y
	}
	if z > max {
		max = z
	}
	return max
}
