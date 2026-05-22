package transform

import (
	"log"
	"sort"

	"indigo/ecs"
)

// MaxHierarchyDepth is the maximum parent-chain depth the propagation
// system will walk. Entities deeper than this are treated as
// disconnected from their broken / too-deep chain so a runaway parent
// link cannot accumulate unbounded translation.
const MaxHierarchyDepth = 256

// TransformState is the world-scoped resource that caches the
// parent → children index used by [MarkDirty]'s descendant cascade
// and by the [QueryChildren] family. The cache lives across frames
// and is invalidated whenever [UpdateParent] or [RemoveParent]
// changes a parent link; the first lookup after an invalidation
// rebuilds it via one ForEach over Parent components.
//
// The scratch slices ([dirty]/[depths]/[order]) are reused across
// frames so [UpdateGlobalTransforms] doesn't allocate per call.
type TransformState struct {
	ChildrenCache      map[ecs.Entity][]ecs.Entity
	ChildrenCacheValid bool

	dirty            []ecs.Entity
	depths           []int
	order            []int
	depthCapReported bool
}

// NewTransformState returns an empty transform state with the
// children cache pre-allocated.
func NewTransformState() TransformState {
	return TransformState{
		ChildrenCache:      make(map[ecs.Entity][]ecs.Entity, 16),
		ChildrenCacheValid: false,
	}
}

// MarkDirty adds [LocalTransformDirty] to entity and to every
// descendant in the children cache. Call this from any system that
// mutates a LocalTransform; the propagation system on the next frame
// will rebuild every dirty entity's GlobalTransform.
//
// [AssignLocalTransform] is the convenience that writes a value and
// calls MarkDirty in one shot.
func MarkDirty(world *ecs.World, entity ecs.Entity) {
	validateChildrenCache(world)
	state := ecs.MustResource[TransformState](world)
	markEntity(world, entity)
	stack := []ecs.Entity{entity}
	for len(stack) > 0 {
		current := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		for _, child := range state.ChildrenCache[current] {
			if ecs.Has[LocalTransformDirty](world, child) {
				continue
			}
			markEntity(world, child)
			stack = append(stack, child)
		}
	}
}

func markEntity(world *ecs.World, entity ecs.Entity) {
	if !ecs.Has[LocalTransformDirty](world, entity) {
		ecs.Add[LocalTransformDirty](world, entity)
	}
}

// AssignLocalTransform writes value to entity's LocalTransform and
// marks the entity (and its descendants) dirty so the propagation
// system rebuilds their GlobalTransforms next frame. Prefer this to
// writing LocalTransform directly: it closes the foot-gun of forgetting
// the MarkDirty call.
func AssignLocalTransform(world *ecs.World, entity ecs.Entity, value LocalTransform) {
	if existing, ok := ecs.GetMut[LocalTransform](world, entity); ok {
		*existing = value
	} else {
		ecs.Set(world, entity, value)
	}
	MarkDirty(world, entity)
}

// UpdateParent assigns parent as the child's new parent, invalidates
// the children cache so the next dirty cascade sees the new layout,
// and marks the child dirty so the propagation system recomputes its
// world matrix against the new parent chain. To detach a child from
// any parent (make it a root), call [RemoveParent] instead — entity
// IDs in this engine start at zero so there's no in-band "no parent"
// sentinel to overload UpdateParent with.
func UpdateParent(world *ecs.World, child ecs.Entity, parent ecs.Entity) {
	state := ecs.MustResource[TransformState](world)
	state.ChildrenCacheValid = false
	ecs.Set(world, child, Parent{Entity: parent})
	MarkDirty(world, child)
}

// RemoveParent detaches child from its current parent, making it a
// root. Invalidates the children cache and marks the child dirty.
func RemoveParent(world *ecs.World, child ecs.Entity) {
	state := ecs.MustResource[TransformState](world)
	state.ChildrenCacheValid = false
	ecs.Set(world, child, Parent{IsRoot: true})
	MarkDirty(world, child)
}

// QueryChildren returns the direct children of entity (one level
// deep). The returned slice is owned by the cache; callers must not
// mutate it. An entity with no Parent links pointing at it returns
// an empty slice.
func QueryChildren(world *ecs.World, entity ecs.Entity) []ecs.Entity {
	validateChildrenCache(world)
	state := ecs.MustResource[TransformState](world)
	return state.ChildrenCache[entity]
}

// QueryDescendants returns every entity in the subtree rooted at
// entity, in DFS pre-order (children of a node visited before the
// node's siblings). entity itself is NOT included.
func QueryDescendants(world *ecs.World, entity ecs.Entity) []ecs.Entity {
	validateChildrenCache(world)
	state := ecs.MustResource[TransformState](world)
	var out []ecs.Entity
	stack := append([]ecs.Entity(nil), state.ChildrenCache[entity]...)
	for len(stack) > 0 {
		current := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		out = append(out, current)
		stack = append(stack, state.ChildrenCache[current]...)
	}
	return out
}

// QueryAncestors returns every parent in entity's chain, root-first.
// Empty when entity is itself a root.
func QueryAncestors(world *ecs.World, entity ecs.Entity) []ecs.Entity {
	var out []ecs.Entity
	current := entity
	for depth := 0; depth <= MaxHierarchyDepth; depth++ {
		parent, ok := ecs.Get[Parent](world, current)
		if !ok || parent.IsRoot {
			break
		}
		out = append(out, parent.Entity)
		current = parent.Entity
	}
	// reverse to root-first
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out
}

// FindGroupRoot walks entity's parent chain until it hits a root and
// returns that root. If entity is itself a root (or has no Parent),
// FindGroupRoot returns entity. Useful for selection systems that
// want to operate on whole prefabs from any clicked child.
func FindGroupRoot(world *ecs.World, entity ecs.Entity) ecs.Entity {
	current := entity
	for depth := 0; depth <= MaxHierarchyDepth; depth++ {
		parent, ok := ecs.Get[Parent](world, current)
		if !ok || parent.IsRoot {
			return current
		}
		current = parent.Entity
	}
	return current
}

// validateChildrenCache rebuilds the parent → children index from
// one ForEach over Parent components if the cache has been marked
// invalid. Idempotent + cheap when the cache is already valid (just
// a flag check).
func validateChildrenCache(world *ecs.World) {
	state := ecs.MustResource[TransformState](world)
	if state.ChildrenCacheValid {
		return
	}
	for k := range state.ChildrenCache {
		delete(state.ChildrenCache, k)
	}
	parentMask := ecs.MustMaskOf[Parent](world)
	world.ForEach(parentMask, 0, func(entity ecs.Entity, table *ecs.Archetype, index int) {
		parents, _ := ecs.Column[Parent](world, table)
		p := &parents[index]
		if p.IsRoot {
			return
		}
		state.ChildrenCache[p.Entity] = append(state.ChildrenCache[p.Entity], entity)
	})
	state.ChildrenCacheValid = true
}

// InvalidateChildrenCache is the escape hatch for code that bypasses
// [UpdateParent] / [RemoveParent] and mutates Parent directly. Call
// it after any such mutation so the next [MarkDirty] / query sees
// the new tree. Routine code should use the helpers and never need
// this.
func InvalidateChildrenCache(world *ecs.World) {
	state := ecs.MustResource[TransformState](world)
	state.ChildrenCacheValid = false
}

// UpdateGlobalTransforms is the per-frame transform propagation
// system. Collects every entity marked LocalTransformDirty, orders
// them shallow-first by parent-chain depth, and recomputes each
// one's GlobalTransform as parent.global * local (with the parent's
// scale optionally stripped when the child carries
// [IgnoreParentScale]).
//
// Cascade is handled at [MarkDirty] time through the children cache,
// so this system doesn't need to expand the dirty set. It just sorts
// and recomputes.
func UpdateGlobalTransforms(world *ecs.World) {
	state := ecs.MustResource[TransformState](world)
	state.dirty = state.dirty[:0]
	state.depths = state.depths[:0]
	state.order = state.order[:0]

	localMask := ecs.MustMaskOf[LocalTransform](world)
	globalMask := ecs.MustMaskOf[GlobalTransform](world)
	dirtyMask := ecs.MustMaskOf[LocalTransformDirty](world)

	world.ForEach(localMask|globalMask|dirtyMask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		state.dirty = append(state.dirty, entity)
	})

	if len(state.dirty) == 0 {
		return
	}

	for _, entity := range state.dirty {
		depth := depthOf(world, entity)
		state.depths = append(state.depths, depth)
		if depth > MaxHierarchyDepth && !state.depthCapReported {
			log.Printf("transform: hierarchy depth exceeded MaxHierarchyDepth=%d for entity %v; treating as disconnected", MaxHierarchyDepth, entity)
			state.depthCapReported = true
		}
	}

	for index := range state.dirty {
		state.order = append(state.order, index)
	}
	sort.SliceStable(state.order, func(i, j int) bool {
		return state.depths[state.order[i]] < state.depths[state.order[j]]
	})

	for _, index := range state.order {
		entity := state.dirty[index]
		depth := state.depths[index]
		matrix := computeMatrix(world, entity, depth)
		if global, ok := ecs.GetMut[GlobalTransform](world, entity); ok {
			global.Matrix = matrix
		}
		ecs.Remove[LocalTransformDirty](world, entity)
	}
}

// depthOf returns the entity's parent-chain depth: 0 for roots, one
// plus the parent's depth otherwise. Returns MaxHierarchyDepth+1
// when the chain exceeds the limit so the propagation step treats
// those entities as disconnected.
func depthOf(world *ecs.World, entity ecs.Entity) int {
	current := entity
	for depth := 0; depth <= MaxHierarchyDepth; depth++ {
		parent, hasParent := ecs.Get[Parent](world, current)
		if !hasParent || parent.IsRoot {
			return depth
		}
		current = parent.Entity
	}
	return MaxHierarchyDepth + 1
}

// computeMatrix produces the entity's world-space matrix as
// parent.global * local. When the child has [IgnoreParentScale] the
// parent's basis is renormalized before the multiply so the child
// keeps its own scale regardless of the parent's. Roots and
// too-deep entities fall back to local.
func computeMatrix(world *ecs.World, entity ecs.Entity, depth int) Mat4 {
	local, ok := ecs.Get[LocalTransform](world, entity)
	if !ok {
		return Mat4Identity()
	}
	localMatrix := AsMatrix(local)

	if depth > MaxHierarchyDepth {
		return localMatrix
	}

	parent, hasParent := ecs.Get[Parent](world, entity)
	if !hasParent || parent.IsRoot {
		return localMatrix
	}
	parentGlobal, ok := ecs.Get[GlobalTransform](world, parent.Entity)
	if !ok {
		return localMatrix
	}
	parentMatrix := parentGlobal.Matrix
	if ecs.Has[IgnoreParentScale](world, entity) {
		parentMatrix = stripScale(parentMatrix)
	}
	return parentMatrix.Mul4(localMatrix)
}

// stripScale rebuilds m with each of the rotation columns
// renormalized to unit length, leaving translation alone. Used to
// implement IgnoreParentScale: a child that ignores its parent's
// scale composes against a parent matrix that has been collapsed
// back to rotation + translation only.
func stripScale(m Mat4) Mat4 {
	out := m
	col0 := Vec3{out[0], out[1], out[2]}
	col1 := Vec3{out[4], out[5], out[6]}
	col2 := Vec3{out[8], out[9], out[10]}
	if length := col0.Len(); length > 0 {
		inv := 1.0 / length
		out[0] *= inv
		out[1] *= inv
		out[2] *= inv
	}
	if length := col1.Len(); length > 0 {
		inv := 1.0 / length
		out[4] *= inv
		out[5] *= inv
		out[6] *= inv
	}
	if length := col2.Len(); length > 0 {
		inv := 1.0 / length
		out[8] *= inv
		out[9] *= inv
		out[10] *= inv
	}
	return out
}
