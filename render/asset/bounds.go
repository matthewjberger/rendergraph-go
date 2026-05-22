package asset

import "math"

// BoundingVolume is an axis-aligned bounding box in mesh-local
// coordinates. Stored alongside each registered mesh so render passes
// and editor systems can ask "what's the local-space extent of this
// mesh" without re-walking the vertex data every frame.
type BoundingVolume struct {
	Min [3]float32
	Max [3]float32
}

// ComputeBounds returns the AABB enclosing every vertex position in
// vertices. An empty slice produces a zero-extent box at the origin.
func ComputeBounds(vertices []MeshVertex) BoundingVolume {
	if len(vertices) == 0 {
		return BoundingVolume{}
	}
	min := [3]float32{
		float32(math.Inf(1)), float32(math.Inf(1)), float32(math.Inf(1)),
	}
	max := [3]float32{
		float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)),
	}
	for i := range vertices {
		p := vertices[i].Position
		for axis := 0; axis < 3; axis++ {
			if p[axis] < min[axis] {
				min[axis] = p[axis]
			}
			if p[axis] > max[axis] {
				max[axis] = p[axis]
			}
		}
	}
	return BoundingVolume{Min: min, Max: max}
}

// Corners returns the eight corner points of the AABB.
func (b BoundingVolume) Corners() [8][3]float32 {
	return [8][3]float32{
		{b.Min[0], b.Min[1], b.Min[2]},
		{b.Max[0], b.Min[1], b.Min[2]},
		{b.Max[0], b.Max[1], b.Min[2]},
		{b.Min[0], b.Max[1], b.Min[2]},
		{b.Min[0], b.Min[1], b.Max[2]},
		{b.Max[0], b.Min[1], b.Max[2]},
		{b.Max[0], b.Max[1], b.Max[2]},
		{b.Min[0], b.Max[1], b.Max[2]},
	}
}

// EdgeIndices returns the 24 corner-index pairs (12 edges) that make
// up the box's wireframe. Caller pairs these with [Corners].
var BoundingBoxEdges = [24]uint8{
	0, 1, 1, 2, 2, 3, 3, 0,
	4, 5, 5, 6, 6, 7, 7, 4,
	0, 4, 1, 5, 2, 6, 3, 7,
}
