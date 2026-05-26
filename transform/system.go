package transform

import (
	"log"
	"sort"

	"github.com/matthewjberger/indigo/ecs"
)

const MaxHierarchyDepth = 256

type State struct {
	ChildrenCache      map[ecs.Entity][]ecs.Entity
	ChildrenCacheValid bool

	dirty            []ecs.Entity
	depths           []int
	order            []int
	depthCapReported bool
}

func NewState() State {
	return State{
		ChildrenCache:      make(map[ecs.Entity][]ecs.Entity, 16),
		ChildrenCacheValid: false,
	}
}

func MarkDirty(world *ecs.World, entity ecs.Entity) {
	validateChildrenCache(world)
	state := ecs.MustResource[State](world)
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

func AssignLocalTransform(world *ecs.World, entity ecs.Entity, value LocalTransform) {
	if existing, ok := ecs.GetMut[LocalTransform](world, entity); ok {
		*existing = value
	} else {
		ecs.Set(world, entity, value)
	}
	MarkDirty(world, entity)
}

func UpdateParent(world *ecs.World, child ecs.Entity, parent ecs.Entity) {
	state := ecs.MustResource[State](world)
	state.ChildrenCacheValid = false
	ecs.Set(world, child, Parent{Entity: parent})
	MarkDirty(world, child)
}

func RemoveParent(world *ecs.World, child ecs.Entity) {
	state := ecs.MustResource[State](world)
	state.ChildrenCacheValid = false
	ecs.Set(world, child, Parent{IsRoot: true})
	MarkDirty(world, child)
}

func QueryChildren(world *ecs.World, entity ecs.Entity) []ecs.Entity {
	validateChildrenCache(world)
	state := ecs.MustResource[State](world)
	return state.ChildrenCache[entity]
}

func QueryDescendants(world *ecs.World, entity ecs.Entity) []ecs.Entity {
	validateChildrenCache(world)
	state := ecs.MustResource[State](world)
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

	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out
}

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

func validateChildrenCache(world *ecs.World) {
	state := ecs.MustResource[State](world)
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

func InvalidateChildrenCache(world *ecs.World) {
	state := ecs.MustResource[State](world)
	state.ChildrenCacheValid = false
}

func UpdateGlobalTransforms(world *ecs.World) {
	state := ecs.MustResource[State](world)
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
		parentMatrix = StripScale(parentMatrix)
	}
	return parentMatrix.Mul4(localMatrix)
}

func StripScale(m Mat4) Mat4 {
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
