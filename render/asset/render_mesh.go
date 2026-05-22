package asset

// RenderMesh is an ECS component pointing at a mesh in the engine's
// [MeshAssets] registry. Entities with [transform.GlobalTransform] +
// RenderMesh are drawn by the mesh pass; the handle picks which mesh.
//
// Data-oriented mirror of nightshade's RENDER_MESH: a thin handle, no
// per-entity material/mesh data here, no inheritance.
type RenderMesh struct {
	Mesh MeshHandle
}
