package asset

import (
	"fmt"
	"math"

	"github.com/cogentcore/webgpu/wgpu"
)

// MeshHandle is an opaque index into a [MeshAssets] registry. Zero is
// not a special value; the registry hands the first registered mesh
// out as handle 0.
type MeshHandle uint32

// Primitives bundles the built-in unit-mesh handles produced by
// [RegisterPrimitives]. Stored as a typed resource on the engine
// world so apps that want a quick cube or quad can pull the handle
// off the world without keeping a reference to the renderer.
type Primitives struct {
	UnitTriangle MeshHandle
	UnitQuad     MeshHandle
	UnitCube     MeshHandle
	UnitPlane    MeshHandle
	UnitSphere   MeshHandle
}

// RegisterPrimitives uploads the unit triangle, quad, and cube into
// assets and returns their handles. Apps usually call this through
// [indigo/app.NewEngineWorld] which installs the result as a
// [Primitives] resource.
func RegisterPrimitives(device *wgpu.Device, assets *MeshAssets) (Primitives, error) {
	tri, err := assets.Register(device, "unit_triangle", UnitTriangleVertices)
	if err != nil {
		return Primitives{}, err
	}
	quad, err := assets.Register(device, "unit_quad", UnitQuadVertices)
	if err != nil {
		return Primitives{}, err
	}
	cube, err := assets.Register(device, "unit_cube", UnitCubeVertices)
	if err != nil {
		return Primitives{}, err
	}
	plane, err := assets.Register(device, "unit_plane", UnitPlaneVertices)
	if err != nil {
		return Primitives{}, err
	}
	sphere, err := assets.Register(device, "unit_sphere", UnitSphereVertices())
	if err != nil {
		return Primitives{}, err
	}
	return Primitives{UnitTriangle: tri, UnitQuad: quad, UnitCube: cube, UnitPlane: plane, UnitSphere: sphere}, nil
}

// MeshVertex is the input layout the engine's stock mesh shader
// expects: position, normal, tangent, uv, color, each padded to
// vec4 stride for std430 alignment. UV holds two coordinate sets:
// xy is TEXCOORD_0, zw is TEXCOORD_1. Tangent's w stores handedness
// for normal mapping. Custom passes that bypass MeshAssets can use
// their own layout.
type MeshVertex struct {
	Position [4]float32
	Normal   [4]float32
	Tangent  [4]float32
	UV       [4]float32
	Color    [4]float32
}

// meshEntry is the renderer's per-mesh record: GPU vertex buffer,
// vertex count, and an optional base-color texture handle. Meshes
// loaded from glTF carry their texture here so the mesh pass binds
// it automatically when drawing instances of that handle.
//
// CpuVertices keeps a copy of the source vertex slice so debug
// passes (normal visualization) can transform per-vertex normals
// into world space without round-tripping through a compute
// shader. Allocates roughly 40 bytes per vertex per mesh.
type meshEntry struct {
	Name        string
	Vertices    *wgpu.Buffer
	VertexCount uint32
	Texture     TextureID
	Bounds      BoundingVolume
	CpuVertices []MeshVertex
}

// MeshAssets is the engine's per-renderer mesh registry. Stored on
// the engine ECS world via [MeshAssetsResource] so passes that
// want to draw a mesh can look up its GPU buffer by handle. The
// registry is "list of vertex buffers indexed by handle."
type MeshAssets struct {
	entries []meshEntry
}

// MeshAssetsResource wraps a *MeshAssets so it can be installed on an
// ECS world via Go-type-keyed resources. Mutations through the wrapped
// pointer persist; freecs-go keeps a stable pointer to the wrapper.
type MeshAssetsResource struct {
	Assets *MeshAssets
}

// NewMeshAssets returns an empty registry. Use [MeshAssets.Register]
// to add meshes.
func NewMeshAssets() *MeshAssets { return &MeshAssets{} }

// Register uploads vertices to a new GPU buffer and returns the handle
// callers should attach to entities via [RenderMesh].
func (assets *MeshAssets) Register(device *wgpu.Device, name string, vertices []MeshVertex) (MeshHandle, error) {
	buffer, err := device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    name + " vertex buffer",
		Contents: wgpu.ToBytes(vertices),
		Usage:    wgpu.BufferUsageVertex,
	})
	if err != nil {
		return 0, fmt.Errorf("mesh assets: %s vertex buffer: %w", name, err)
	}
	cpu := make([]MeshVertex, len(vertices))
	copy(cpu, vertices)
	handle := MeshHandle(len(assets.entries))
	assets.entries = append(assets.entries, meshEntry{
		Name:        name,
		Vertices:    buffer,
		VertexCount: uint32(len(vertices)),
		Bounds:      ComputeBounds(vertices),
		CpuVertices: cpu,
	})
	return handle, nil
}

// CpuVertices returns the CPU-side vertex slice for a registered
// mesh, or nil for an unknown handle. Used by debug overlays
// (normal visualization) that need to iterate vertices without
// reading them back from the GPU.
func (assets *MeshAssets) CpuVertices(handle MeshHandle) []MeshVertex {
	if int(handle) >= len(assets.entries) {
		return nil
	}
	return assets.entries[handle].CpuVertices
}

// Lookup returns the per-mesh entry for handle. Callers that pass an
// out-of-range handle get a false result and should skip the entity.
func (assets *MeshAssets) Lookup(handle MeshHandle) (*meshEntry, bool) {
	if int(handle) >= len(assets.entries) {
		return nil, false
	}
	return &assets.entries[handle], true
}

// AttachTexture sets the base-color texture handle for a mesh.
// Callers register the texture via [TextureCache.Register] first and
// then bind its handle here.
func (assets *MeshAssets) AttachTexture(handle MeshHandle, texture TextureID) {
	if int(handle) >= len(assets.entries) {
		return
	}
	assets.entries[handle].Texture = texture
}

// Bounds returns the AABB of the registered mesh in its local
// coordinate space. Zero-extent for an unknown handle.
func (assets *MeshAssets) Bounds(handle MeshHandle) BoundingVolume {
	if int(handle) >= len(assets.entries) {
		return BoundingVolume{}
	}
	return assets.entries[handle].Bounds
}

// Count returns the number of registered meshes.
func (assets *MeshAssets) Count() int { return len(assets.entries) }

// Release frees every GPU buffer owned by the registry.
func (assets *MeshAssets) Release() {
	for index := range assets.entries {
		if assets.entries[index].Vertices != nil {
			assets.entries[index].Vertices.Release()
			assets.entries[index].Vertices = nil
		}
	}
	assets.entries = nil
}

// UnitTriangleVertices is a 1-unit XY triangle facing +Z, with red /
// green / blue corners.
var UnitTriangleVertices = []MeshVertex{
	{Position: [4]float32{0.5, -0.5, 0.0, 1.0}, Normal: defaultNormalZ, Tangent: defaultTangent, UV: [4]float32{1, 1, 0, 0}, Color: [4]float32{1.0, 0.0, 0.0, 1.0}},
	{Position: [4]float32{-0.5, -0.5, 0.0, 1.0}, Normal: defaultNormalZ, Tangent: defaultTangent, UV: [4]float32{0, 1, 0, 0}, Color: [4]float32{0.0, 1.0, 0.0, 1.0}},
	{Position: [4]float32{0.0, 0.5, 0.0, 1.0}, Normal: defaultNormalZ, Tangent: defaultTangent, UV: [4]float32{0.5, 0, 0, 0}, Color: [4]float32{0.0, 0.0, 1.0, 1.0}},
}

var (
	defaultNormalZ = [4]float32{0, 0, 1, 0}
	defaultTangent = [4]float32{1, 0, 0, 1}
)

// UnitQuadVertices is a 1-unit XY quad facing +Z, two triangles wound
// CCW, with red / green / blue / yellow corners.
var UnitQuadVertices = []MeshVertex{
	{Position: [4]float32{-0.5, -0.5, 0.0, 1.0}, Normal: defaultNormalZ, Tangent: defaultTangent, UV: [4]float32{0, 1, 0, 0}, Color: [4]float32{1.0, 0.0, 0.0, 1.0}},
	{Position: [4]float32{0.5, -0.5, 0.0, 1.0}, Normal: defaultNormalZ, Tangent: defaultTangent, UV: [4]float32{1, 1, 0, 0}, Color: [4]float32{0.0, 1.0, 0.0, 1.0}},
	{Position: [4]float32{0.5, 0.5, 0.0, 1.0}, Normal: defaultNormalZ, Tangent: defaultTangent, UV: [4]float32{1, 0, 0, 0}, Color: [4]float32{0.0, 0.0, 1.0, 1.0}},
	{Position: [4]float32{-0.5, -0.5, 0.0, 1.0}, Normal: defaultNormalZ, Tangent: defaultTangent, UV: [4]float32{0, 1, 0, 0}, Color: [4]float32{1.0, 0.0, 0.0, 1.0}},
	{Position: [4]float32{0.5, 0.5, 0.0, 1.0}, Normal: defaultNormalZ, Tangent: defaultTangent, UV: [4]float32{1, 0, 0, 0}, Color: [4]float32{0.0, 0.0, 1.0, 1.0}},
	{Position: [4]float32{-0.5, 0.5, 0.0, 1.0}, Normal: defaultNormalZ, Tangent: defaultTangent, UV: [4]float32{0, 0, 0, 0}, Color: [4]float32{1.0, 1.0, 0.0, 1.0}},
}

// UnitPlaneVertices is a 1x1 horizontal plane on the XZ axes,
// normal pointing +Y, white vertex color so the material's
// BaseColor reads cleanly. Use this for ground planes / receivers
// where the rainbow UnitQuad's per-corner color tint would
// pollute the lit output.
var UnitPlaneVertices = func() []MeshVertex {
	const half = float32(0.5)
	const up = float32(1.0)
	normal := [4]float32{0, 1, 0, 0}
	tangent := [4]float32{1, 0, 0, 1}
	white := [4]float32{1, 1, 1, 1}
	corners := [4][2]float32{
		{-half, -half},
		{half, -half},
		{half, half},
		{-half, half},
	}
	uvs := [4][4]float32{
		{0, 1, 0, 0},
		{1, 1, 0, 0},
		{1, 0, 0, 0},
		{0, 0, 0, 0},
	}
	verts := make([]MeshVertex, 0, 6)
	add := func(i int) {
		verts = append(verts, MeshVertex{
			Position: [4]float32{corners[i][0], 0, corners[i][1], up},
			Normal:   normal,
			Tangent:  tangent,
			UV:       uvs[i],
			Color:    white,
		})
	}
	add(0)
	add(2)
	add(1)
	add(0)
	add(3)
	add(2)
	return verts
}()

// UnitCubeVertices is a 1-unit cube centered at origin, 36 vertices
// (6 faces x 2 triangles x 3 vertices), all wound CCW when viewed from
// outside. Each face gets its own distinct color so spinning cubes
// read as 3D rather than as flat colored disks.
var UnitCubeVertices = func() []MeshVertex {
	const s = 0.5
	red := [4]float32{0.9, 0.2, 0.2, 1.0}
	green := [4]float32{0.2, 0.85, 0.3, 1.0}
	blue := [4]float32{0.3, 0.45, 0.95, 1.0}
	yellow := [4]float32{0.95, 0.85, 0.2, 1.0}
	cyan := [4]float32{0.2, 0.85, 0.9, 1.0}
	magenta := [4]float32{0.85, 0.3, 0.85, 1.0}

	face := func(a, b, c, d [3]float32, normal [4]float32, color [4]float32) []MeshVertex {
		v := func(p [3]float32, uv [2]float32) MeshVertex {
			return MeshVertex{
				Position: [4]float32{p[0], p[1], p[2], 1.0},
				Normal:   normal,
				Tangent:  defaultTangent,
				UV:       [4]float32{uv[0], uv[1], 0, 0},
				Color:    color,
			}
		}
		return []MeshVertex{
			v(a, [2]float32{0, 1}), v(b, [2]float32{1, 1}), v(c, [2]float32{1, 0}),
			v(a, [2]float32{0, 1}), v(c, [2]float32{1, 0}), v(d, [2]float32{0, 0}),
		}
	}

	plusZ := face(
		[3]float32{-s, -s, s}, [3]float32{s, -s, s},
		[3]float32{s, s, s}, [3]float32{-s, s, s},
		[4]float32{0, 0, 1, 0}, blue,
	)
	minusZ := face(
		[3]float32{s, -s, -s}, [3]float32{-s, -s, -s},
		[3]float32{-s, s, -s}, [3]float32{s, s, -s},
		[4]float32{0, 0, -1, 0}, yellow,
	)
	plusX := face(
		[3]float32{s, -s, s}, [3]float32{s, -s, -s},
		[3]float32{s, s, -s}, [3]float32{s, s, s},
		[4]float32{1, 0, 0, 0}, red,
	)
	minusX := face(
		[3]float32{-s, -s, -s}, [3]float32{-s, -s, s},
		[3]float32{-s, s, s}, [3]float32{-s, s, -s},
		[4]float32{-1, 0, 0, 0}, cyan,
	)
	plusY := face(
		[3]float32{-s, s, s}, [3]float32{s, s, s},
		[3]float32{s, s, -s}, [3]float32{-s, s, -s},
		[4]float32{0, 1, 0, 0}, green,
	)
	minusY := face(
		[3]float32{-s, -s, -s}, [3]float32{s, -s, -s},
		[3]float32{s, -s, s}, [3]float32{-s, -s, s},
		[4]float32{0, -1, 0, 0}, magenta,
	)

	out := make([]MeshVertex, 0, 36)
	out = append(out, plusZ...)
	out = append(out, minusZ...)
	out = append(out, plusX...)
	out = append(out, minusX...)
	out = append(out, plusY...)
	out = append(out, minusY...)
	return out
}()

// UnitSphereVertices returns a 24-segment UV sphere of radius 0.5
// expanded into non-indexed triangles.
func UnitSphereVertices() []MeshVertex {
	const segments = 24
	const radius = 0.5
	grid := make([]MeshVertex, 0, (segments+1)*(segments+1))
	for i := 0; i <= segments; i++ {
		lat := math.Pi * float64(i) / float64(segments)
		v := float32(float64(i) / float64(segments))
		sinLat := float32(math.Sin(lat))
		cosLat := float32(math.Cos(lat))
		for j := 0; j <= segments; j++ {
			lon := 2 * math.Pi * float64(j) / float64(segments)
			u := float32(float64(j) / float64(segments))
			sinLon := float32(math.Sin(lon))
			cosLon := float32(math.Cos(lon))
			x := radius * sinLat * cosLon
			y := radius * cosLat
			z := radius * sinLat * sinLon
			nx, ny, nz := sinLat*cosLon, cosLat, sinLat*sinLon
			grid = append(grid, MeshVertex{
				Position: [4]float32{x, y, z, 1},
				Normal:   [4]float32{nx, ny, nz, 0},
				Tangent:  [4]float32{1, 0, 0, 1},
				UV:       [4]float32{u, v, 0, 0},
				Color:    [4]float32{1, 1, 1, 1},
			})
		}
	}
	out := make([]MeshVertex, 0, segments*segments*6)
	for i := 0; i < segments; i++ {
		rowA := i * (segments + 1)
		rowB := (i + 1) * (segments + 1)
		for j := 0; j < segments; j++ {
			a := grid[rowA+j]
			b := grid[rowA+j+1]
			c := grid[rowB+j]
			d := grid[rowB+j+1]
			out = append(out, a, b, c, c, b, d)
		}
	}
	return out
}
