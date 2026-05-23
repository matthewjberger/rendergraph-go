package pass

import (
	"github.com/go-gl/mathgl/mgl32"

	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/render/asset"
	"github.com/matthewjberger/indigo/transform"
)

// UpdateNormalLines emits one world-space line per vertex of every
// drawn mesh when [render.Graphics.ShowNormals] is true.
// Each line starts at the vertex's world position and points along
// the vertex's world-space normal for
// [render.Graphics.NormalLineLength] units.
//
// The world normal uses the upper-3x3 of the model matrix as the
// normal matrix. That's accurate for uniform scale; non-uniform
// scale would need a true transpose-inverse normal matrix, but
// transform.Mat4 only carries TRS so the approximation is fine
// for the editor scene.
//
// Push runs every frame so the lines pass picks the data up next
// draw. Frame-by-frame regeneration is fine for an interactive
// debug overlay; cost scales with total visible vertex count.
func UpdateNormalLines(world *ecs.World) {
	settings, ok := ecs.Resource[render.Graphics](world)
	if !ok || !settings.ShowNormals {
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
	length := settings.Lines.NormalLength
	if length <= 0 {
		length = 0.05
	}
	color := settings.Lines.NormalColor

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
		vertices := assets.CpuVertices(mesh.Mesh)
		if len(vertices) == 0 {
			return
		}
		m := global.Matrix
		for i := range vertices {
			v := &vertices[i]
			worldPos := m.Mul4x1(mgl32.Vec4{v.Position[0], v.Position[1], v.Position[2], 1})
			worldNormal := mgl32.Vec3{
				m[0]*v.Normal[0] + m[4]*v.Normal[1] + m[8]*v.Normal[2],
				m[1]*v.Normal[0] + m[5]*v.Normal[1] + m[9]*v.Normal[2],
				m[2]*v.Normal[0] + m[6]*v.Normal[1] + m[10]*v.Normal[2],
			}
			worldNormal = worldNormal.Normalize()
			start := [3]float32{worldPos.X(), worldPos.Y(), worldPos.Z()}
			end := [3]float32{
				start[0] + worldNormal.X()*length,
				start[1] + worldNormal.Y()*length,
				start[2] + worldNormal.Z()*length,
			}
			lines.AddSegment(start, end, color)
		}
	})

	skinnedAssetsRes, hasSkinnedAssets := ecs.Resource[asset.SkinnedMeshAssetsResource](world)
	if !hasSkinnedAssets || skinnedAssetsRes == nil || skinnedAssetsRes.Assets == nil {
		return
	}
	skinnedAssets := skinnedAssetsRes.Assets
	skinnedMask := ecs.MustMaskOf[asset.SkinnedMesh](world)
	world.ForEach(skinnedMask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		skinned, _ := ecs.Get[asset.SkinnedMesh](world, entity)
		if skinned == nil || skinned.Skin == nil || len(skinned.Skin.Joints) == 0 {
			return
		}
		entry, ok := skinnedAssets.Lookup(skinned.Mesh)
		if !ok || entry == nil || len(entry.CpuVertices) == 0 {
			return
		}
		skin := skinned.Skin
		joints := make([]mgl32.Mat4, len(skin.Joints))
		for jointIndex, jointEntity := range skin.Joints {
			if global, ok := ecs.Get[transform.GlobalTransform](world, jointEntity); ok {
				joints[jointIndex] = global.Matrix.Mul4(skin.InverseBindMatrices[jointIndex])
			} else {
				joints[jointIndex] = skin.InverseBindMatrices[jointIndex]
			}
		}
		for i := range entry.CpuVertices {
			v := &entry.CpuVertices[i]
			var pos mgl32.Vec3
			var normal mgl32.Vec3
			for k := 0; k < 4; k++ {
				w := v.JointWeights[k]
				if w <= 0 {
					continue
				}
				idx := int(v.JointIndices[k])
				if idx < 0 || idx >= len(joints) {
					continue
				}
				m := joints[idx]
				p := m.Mul4x1(mgl32.Vec4{v.Position[0], v.Position[1], v.Position[2], 1})
				pos = pos.Add(mgl32.Vec3{p.X() * w, p.Y() * w, p.Z() * w})
				n := mgl32.Vec3{
					m[0]*v.Normal[0] + m[4]*v.Normal[1] + m[8]*v.Normal[2],
					m[1]*v.Normal[0] + m[5]*v.Normal[1] + m[9]*v.Normal[2],
					m[2]*v.Normal[0] + m[6]*v.Normal[1] + m[10]*v.Normal[2],
				}
				normal = normal.Add(n.Mul(w))
			}
			if normal.Len() < 1e-6 {
				continue
			}
			normal = normal.Normalize()
			start := [3]float32{pos.X(), pos.Y(), pos.Z()}
			end := [3]float32{
				start[0] + normal.X()*length,
				start[1] + normal.Y()*length,
				start[2] + normal.Z()*length,
			}
			lines.AddSegment(start, end, color)
		}
	})
}
