package transform_test

import (
	"testing"

	"indigo/ecs"
	"indigo/transform"
)

func newWorld(t *testing.T) *ecs.World {
	t.Helper()
	world := ecs.New()
	ecs.Register[transform.LocalTransform](world)
	ecs.Register[transform.GlobalTransform](world)
	ecs.Register[transform.Parent](world)
	ecs.Register[transform.LocalTransformDirty](world)
	ecs.Register[transform.IgnoreParentScale](world)
	ecs.SetResource(world, transform.NewTransformState())
	return world
}

func spawnAt(t *testing.T, world *ecs.World, position transform.Vec3) ecs.Entity {
	t.Helper()
	entity := world.Spawn(0)
	ecs.Set(world, entity, transform.FromTranslation(position))
	ecs.Set(world, entity, transform.IdentityGlobalTransform())
	transform.MarkDirty(world, entity)
	return entity
}

func TestUpdateGlobalTransformsRoot(t *testing.T) {
	world := newWorld(t)
	entity := spawnAt(t, world, transform.Vec3{1, 2, 3})

	transform.UpdateGlobalTransforms(world)

	global, ok := ecs.Get[transform.GlobalTransform](world, entity)
	if !ok {
		t.Fatal("GlobalTransform missing after update")
	}
	got := transform.GlobalTransformTranslation(global)
	want := transform.Vec3{1, 2, 3}
	if got != want {
		t.Fatalf("translation = %v, want %v", got, want)
	}
}

func TestUpdateGlobalTransformsClearsDirty(t *testing.T) {
	world := newWorld(t)
	entity := spawnAt(t, world, transform.Vec3{0, 0, 0})

	if !ecs.Has[transform.LocalTransformDirty](world, entity) {
		t.Fatal("entity should be dirty before update")
	}
	transform.UpdateGlobalTransforms(world)
	if ecs.Has[transform.LocalTransformDirty](world, entity) {
		t.Fatal("dirty flag should be cleared after update")
	}
}

func TestUpdateGlobalTransformsParentChain(t *testing.T) {
	world := newWorld(t)
	parent := spawnAt(t, world, transform.Vec3{10, 0, 0})
	child := spawnAt(t, world, transform.Vec3{0, 5, 0})
	ecs.Set(world, child, transform.Parent{Entity: parent})

	transform.UpdateGlobalTransforms(world)

	global, ok := ecs.Get[transform.GlobalTransform](world, child)
	if !ok {
		t.Fatal("child GlobalTransform missing")
	}
	got := transform.GlobalTransformTranslation(global)
	want := transform.Vec3{10, 5, 0}
	if got != want {
		t.Fatalf("child world translation = %v, want %v", got, want)
	}
}

func TestUpdateGlobalTransformsRootFlag(t *testing.T) {
	world := newWorld(t)
	a := spawnAt(t, world, transform.Vec3{10, 0, 0})
	b := spawnAt(t, world, transform.Vec3{0, 5, 0})
	ecs.Set(world, b, transform.Parent{Entity: a, IsRoot: true})

	transform.UpdateGlobalTransforms(world)

	global, _ := ecs.Get[transform.GlobalTransform](world, b)
	got := transform.GlobalTransformTranslation(global)
	want := transform.Vec3{0, 5, 0}
	if got != want {
		t.Fatalf("IsRoot=true should ignore parent, got %v want %v", got, want)
	}
}

func TestUpdateGlobalTransformsCycleTerminates(t *testing.T) {
	world := newWorld(t)
	a := spawnAt(t, world, transform.Vec3{7, 7, 7})
	b := spawnAt(t, world, transform.Vec3{0, 0, 0})
	ecs.Set(world, a, transform.Parent{Entity: b})
	ecs.Set(world, b, transform.Parent{Entity: a})

	transform.UpdateGlobalTransforms(world)

	global, _ := ecs.Get[transform.GlobalTransform](world, a)
	got := transform.GlobalTransformTranslation(global)
	want := transform.Vec3{7, 7, 7}
	if got != want {
		t.Fatalf("cycle truncation should leave a's chain at a.local, got %v want %v", got, want)
	}
}

func TestUpdateGlobalTransformsDepthLimitIsBounded(t *testing.T) {
	world := newWorld(t)
	previous := spawnAt(t, world, transform.Vec3{1, 0, 0})
	var tail ecs.Entity
	for index := 0; index < transform.MaxHierarchyDepth+2; index++ {
		entity := spawnAt(t, world, transform.Vec3{1, 0, 0})
		ecs.Set(world, entity, transform.Parent{Entity: previous})
		previous = entity
		tail = entity
	}

	transform.UpdateGlobalTransforms(world)

	global, _ := ecs.Get[transform.GlobalTransform](world, tail)
	got := transform.GlobalTransformTranslation(global)
	if got[0] > float32(transform.MaxHierarchyDepth) {
		t.Fatalf("walker accumulated more than MaxHierarchyDepth locals: %v", got)
	}
	if got[0] == 0 {
		t.Fatal("walker truncated everything; expected partial accumulation")
	}
}

func TestMarkDirtyIsIdempotent(t *testing.T) {
	world := newWorld(t)
	entity := world.Spawn(0)
	ecs.Set(world, entity, transform.IdentityLocalTransform())

	transform.MarkDirty(world, entity)
	transform.MarkDirty(world, entity)
	if !ecs.Has[transform.LocalTransformDirty](world, entity) {
		t.Fatal("expected dirty flag present")
	}
}

func TestAsMatrixTranslationOnly(t *testing.T) {
	local := transform.FromTranslation(transform.Vec3{4, 5, 6})
	matrix := transform.AsMatrix(&local)
	if matrix[12] != 4 || matrix[13] != 5 || matrix[14] != 6 {
		t.Fatalf("translation row = (%v,%v,%v); want (4,5,6)", matrix[12], matrix[13], matrix[14])
	}
	if matrix[0] != 1 || matrix[5] != 1 || matrix[10] != 1 {
		t.Fatalf("identity diagonal corrupted: %v %v %v", matrix[0], matrix[5], matrix[10])
	}
}
