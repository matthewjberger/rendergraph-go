package asset

import "github.com/go-gl/mathgl/mgl32"

// InstancedMesh renders many copies of one mesh under a single
// entity. Each instance carries a local transform relative to the
// entity's GlobalTransform (the parent); a compute pass multiplies
// parent * local to produce the per-instance world matrix the draw
// reads. Distinct from spawning one entity per copy: an
// InstancedMesh is a single ECS entity whose sub-instances move
// with the parent.
type InstancedMesh struct {
	Mesh      MeshHandle
	Instances []mgl32.Mat4
}
