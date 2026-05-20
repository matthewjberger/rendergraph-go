package render

import "rendergraph-go/transform"

// LightType picks one of the three light shapes from glTF's
// KHR_lights_punctual, matching nightshade's enum.
type LightType uint32

const (
	LightTypeDirectional LightType = 0
	LightTypePoint       LightType = 1
	LightTypeSpot        LightType = 2
)

// Light is the ECS component attached to entities that emit light.
// The entity's [transform.GlobalTransform] supplies position (for
// point/spot) and direction (for directional/spot via the -Z column
// of the transform's rotation), matching the glTF convention
// nightshade uses.
//
// Color is linear RGB; intensity is a scalar multiplier on top of
// the color. Range gates point/spot attenuation; cone angles control
// spot falloff (unused for directional lights). cast_shadows et al.
// from nightshade are dropped for the slice; our mesh pass doesn't
// have shadows yet.
type Light struct {
	Type           LightType
	Color          transform.Vec3
	Intensity      float32
	Range          float32
	InnerConeAngle float32
	OuterConeAngle float32
}

// MaxLights is the fixed-size cap on lights uploaded to the mesh
// shader each frame. Mirrors the typical uniform-array approach for
// forward shading; can be promoted to a storage buffer if we ever
// need more than this.
const MaxLights = 8
