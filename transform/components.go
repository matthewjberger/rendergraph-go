// Package transform is the data-oriented port of nightshade's
// transform subsystem: components for positioning entities, a parent
// relationship that links into the ECS, and a propagation system that
// produces world-space matrices for the renderer.
//
// The transform package is independent of the renderer; the mesh pass
// reads [GlobalTransform] values out of the ECS world the same way any
// other system does.
package transform

import "rendergraph-go/ecs"

// LocalTransform is position, rotation, and scale relative to the
// parent entity (or world origin if the entity has no [Parent]).
//
// The matrix derived from this is TRS order (Translation × Rotation ×
// Scale). Mirrors nightshade's LocalTransform component, identical
// semantics.
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

// AsMatrix builds the 4x4 matrix for this transform in TRS order.
// Free function (not a method on a value type) keeps the API
// data-oriented: state in, matrix out.
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
// Maintained by [UpdateGlobalTransforms]; never set directly.
type GlobalTransform struct {
	Matrix Mat4
}

// Translation extracts the translation column from a global transform.
func (g *GlobalTransform) Translation() Vec3 {
	return Vec3{g.Matrix[12], g.Matrix[13], g.Matrix[14]}
}

// IdentityGlobalTransform returns an identity world-space transform.
func IdentityGlobalTransform() GlobalTransform {
	return GlobalTransform{Matrix: Mat4Identity()}
}

// Parent links a child entity to its parent. nightshade uses
// `Parent(Option<Entity>)`; Go has no Option, so we keep a separate
// IsRoot flag. An entity with IsRoot=true (or no Parent component at
// all) is a root.
type Parent struct {
	Entity ecs.Entity
	IsRoot bool
}

// LocalTransformDirty is the marker component the transform system
// uses to find entities whose [GlobalTransform] needs recomputing.
// Mirrors nightshade's LOCAL_TRANSFORM_DIRTY flag.
type LocalTransformDirty struct{}

// MarkDirty adds [LocalTransformDirty] to an entity if it is not
// already marked. Use this from any system that mutates a
// [LocalTransform] or changes a [Parent] link.
//
// The Has-then-Add check is intentional: adding a component triggers
// an archetype migration, so we want to skip it when the entity is
// already in the dirty archetype.
func MarkDirty(world *ecs.World, entity ecs.Entity) {
	if !ecs.Has[LocalTransformDirty](world, entity) {
		ecs.Add[LocalTransformDirty](world, entity)
	}
}
