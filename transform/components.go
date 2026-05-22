// Package transform owns the engine's transform hierarchy: the
// LocalTransform / GlobalTransform / Parent components plus the
// propagation system that turns local positions in a parent chain
// into world-space matrices for the renderer.
//
// The transform package is independent of the renderer; render passes
// read GlobalTransform out of the ECS world the same way any other
// system does.
package transform

import "indigo/ecs"

// LocalTransform is position, rotation, and scale relative to the
// parent entity (or world origin if the entity has no Parent).
// AsMatrix builds the resulting matrix in T*R*S order.
type LocalTransform struct {
	Translation Vec3
	Rotation    Quat
	Scale       Vec3
}

// IdentityLocalTransform returns the identity transform: zero
// translation, identity rotation, unit scale.
func IdentityLocalTransform() LocalTransform {
	return LocalTransform{
		Translation: Vec3{0, 0, 0},
		Rotation:    QuatIdentity(),
		Scale:       Vec3{1, 1, 1},
	}
}

// FromTranslation returns a transform at the given position with
// identity rotation and unit scale.
func FromTranslation(translation Vec3) LocalTransform {
	return LocalTransform{
		Translation: translation,
		Rotation:    QuatIdentity(),
		Scale:       Vec3{1, 1, 1},
	}
}

// AsMatrix builds the 4x4 matrix for this transform in T*R*S order.
func AsMatrix(t *LocalTransform) Mat4 {
	matrix := QuatToMat4(QuatNormalize(t.Rotation))

	matrix[0] *= t.Scale[0]
	matrix[1] *= t.Scale[0]
	matrix[2] *= t.Scale[0]

	matrix[4] *= t.Scale[1]
	matrix[5] *= t.Scale[1]
	matrix[6] *= t.Scale[1]

	matrix[8] *= t.Scale[2]
	matrix[9] *= t.Scale[2]
	matrix[10] *= t.Scale[2]

	matrix[12] = t.Translation[0]
	matrix[13] = t.Translation[1]
	matrix[14] = t.Translation[2]

	return matrix
}

// GlobalTransform is the world-space matrix the renderer reads.
// Maintained by [UpdateGlobalTransforms]; never set directly outside
// the propagation system.
type GlobalTransform struct {
	Matrix Mat4
}

// GlobalTransformTranslation extracts the translation column from a
// world-space transform's matrix.
func GlobalTransformTranslation(g *GlobalTransform) Vec3 {
	return Vec3{g.Matrix[12], g.Matrix[13], g.Matrix[14]}
}

// IdentityGlobalTransform returns an identity world-space transform.
func IdentityGlobalTransform() GlobalTransform {
	return GlobalTransform{Matrix: Mat4Identity()}
}

// Parent links a child entity to its parent. An entity with
// IsRoot=true (or no Parent component at all) is a root. Set Parent
// through [UpdateParent] rather than writing it directly so the
// children cache invalidates correctly.
type Parent struct {
	Entity ecs.Entity
	IsRoot bool
}

// LocalTransformDirty is the marker component the propagation system
// uses to find entities whose [GlobalTransform] needs recomputing.
// [MarkDirty] sets it and cascades it through descendants via the
// children cache.
type LocalTransformDirty struct{}

// IgnoreParentScale on a child strips the parent's scale out of the
// world matrix composition. Use it for UI overlays, sprites, or
// world-aligned markers that need to keep their own scale regardless
// of what the parent does.
type IgnoreParentScale struct{}
