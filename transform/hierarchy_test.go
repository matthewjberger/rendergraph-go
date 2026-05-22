package transform_test

import (
	"testing"

	"indigo/ecs"
	"indigo/transform"
)

func TestAssignLocalTransformMarksDirty(t *testing.T) {
	world := newWorld(t)
	entity := spawnAt(t, world, transform.Vec3{1, 2, 3})
	transform.UpdateGlobalTransforms(world)
	if ecs.Has[transform.LocalTransformDirty](world, entity) {
		t.Fatal("dirty flag should be cleared after first propagation")
	}
	transform.AssignLocalTransform(world, entity, transform.FromTranslation(transform.Vec3{4, 5, 6}))
	if !ecs.Has[transform.LocalTransformDirty](world, entity) {
		t.Fatal("AssignLocalTransform must mark the entity dirty")
	}
	got, _ := ecs.Get[transform.LocalTransform](world, entity)
	if got.Translation != (transform.Vec3{4, 5, 6}) {
		t.Errorf("translation = %v, want {4 5 6}", got.Translation)
	}
}

func TestMarkDirtyCascadesToDescendants(t *testing.T) {
	world := newWorld(t)
	root := spawnAt(t, world, transform.Vec3{0, 0, 0})
	child := spawnAt(t, world, transform.Vec3{1, 0, 0})
	grandchild := spawnAt(t, world, transform.Vec3{1, 0, 0})
	transform.UpdateParent(world, child, root)
	transform.UpdateParent(world, grandchild, child)
	transform.UpdateGlobalTransforms(world)

	transform.MarkDirty(world, root)
	if !ecs.Has[transform.LocalTransformDirty](world, child) {
		t.Error("child should be marked dirty when root is marked")
	}
	if !ecs.Has[transform.LocalTransformDirty](world, grandchild) {
		t.Error("grandchild should be marked dirty when root is marked")
	}
}

func TestUpdateParentInvalidatesChildrenCache(t *testing.T) {
	world := newWorld(t)
	root := spawnAt(t, world, transform.Vec3{0, 0, 0})
	child := spawnAt(t, world, transform.Vec3{1, 0, 0})
	transform.UpdateParent(world, child, root)
	transform.UpdateGlobalTransforms(world)

	if got := transform.QueryChildren(world, root); len(got) != 1 || got[0] != child {
		t.Errorf("children of root = %v, want [%v]", got, child)
	}

	newRoot := spawnAt(t, world, transform.Vec3{10, 0, 0})
	transform.UpdateParent(world, child, newRoot)

	if got := transform.QueryChildren(world, root); len(got) != 0 {
		t.Errorf("old root children = %v, want empty", got)
	}
	if got := transform.QueryChildren(world, newRoot); len(got) != 1 || got[0] != child {
		t.Errorf("new root children = %v, want [%v]", got, child)
	}
}

func TestRemoveParentDetaches(t *testing.T) {
	world := newWorld(t)
	root := spawnAt(t, world, transform.Vec3{0, 0, 0})
	child := spawnAt(t, world, transform.Vec3{1, 0, 0})
	transform.UpdateParent(world, child, root)
	transform.UpdateGlobalTransforms(world)

	transform.RemoveParent(world, child)
	if got := transform.QueryChildren(world, root); len(got) != 0 {
		t.Errorf("after detach, root children = %v, want empty", got)
	}
	parent, _ := ecs.Get[transform.Parent](world, child)
	if !parent.IsRoot {
		t.Error("after RemoveParent, child should be marked IsRoot")
	}
}

func TestQueryDescendantsAndAncestors(t *testing.T) {
	world := newWorld(t)
	root := spawnAt(t, world, transform.Vec3{0, 0, 0})
	a := spawnAt(t, world, transform.Vec3{1, 0, 0})
	b := spawnAt(t, world, transform.Vec3{0, 1, 0})
	ab := spawnAt(t, world, transform.Vec3{0, 0, 1})
	transform.UpdateParent(world, a, root)
	transform.UpdateParent(world, b, root)
	transform.UpdateParent(world, ab, a)

	desc := transform.QueryDescendants(world, root)
	if len(desc) != 3 {
		t.Errorf("descendants of root = %v, want 3 entries", desc)
	}
	seen := map[ecs.Entity]bool{}
	for _, e := range desc {
		seen[e] = true
	}
	if !seen[a] || !seen[b] || !seen[ab] {
		t.Errorf("descendants missing entries: a=%v b=%v ab=%v in %v", seen[a], seen[b], seen[ab], desc)
	}

	anc := transform.QueryAncestors(world, ab)
	if len(anc) != 2 || anc[0] != root || anc[1] != a {
		t.Errorf("ancestors of ab = %v, want [root=%v a=%v]", anc, root, a)
	}
}

func TestFindGroupRoot(t *testing.T) {
	world := newWorld(t)
	root := spawnAt(t, world, transform.Vec3{0, 0, 0})
	mid := spawnAt(t, world, transform.Vec3{1, 0, 0})
	leaf := spawnAt(t, world, transform.Vec3{2, 0, 0})
	transform.UpdateParent(world, mid, root)
	transform.UpdateParent(world, leaf, mid)

	if got := transform.FindGroupRoot(world, leaf); got != root {
		t.Errorf("root of leaf = %v, want %v", got, root)
	}
	if got := transform.FindGroupRoot(world, root); got != root {
		t.Errorf("root of root = %v, want %v (self)", got, root)
	}
}

func TestIgnoreParentScaleStripsScale(t *testing.T) {
	world := newWorld(t)
	root := spawnAt(t, world, transform.Vec3{0, 0, 0})
	// Set root scale to 10 uniformly
	transform.AssignLocalTransform(world, root, transform.LocalTransform{
		Translation: transform.Vec3{0, 0, 0},
		Rotation:    transform.QuatIdentity(),
		Scale:       transform.Vec3{10, 10, 10},
	})
	child := spawnAt(t, world, transform.Vec3{1, 0, 0})
	transform.UpdateParent(world, child, root)
	ecs.Add[transform.IgnoreParentScale](world, child)
	transform.MarkDirty(world, child)
	transform.UpdateGlobalTransforms(world)

	global, _ := ecs.Get[transform.GlobalTransform](world, child)
	got := transform.GlobalTransformTranslation(global)
	// With IgnoreParentScale, child's local (1, 0, 0) should land at
	// world (1, 0, 0) — not (10, 0, 0) which is what unstripped scale
	// would produce.
	if got != (transform.Vec3{1, 0, 0}) {
		t.Errorf("IgnoreParentScale child world position = %v, want {1 0 0}", got)
	}

	// Sanity check: without the marker the child WOULD scale to 10.
	other := spawnAt(t, world, transform.Vec3{1, 0, 0})
	transform.UpdateParent(world, other, root)
	transform.UpdateGlobalTransforms(world)
	otherGlobal, _ := ecs.Get[transform.GlobalTransform](world, other)
	otherGot := transform.GlobalTransformTranslation(otherGlobal)
	if otherGot != (transform.Vec3{10, 0, 0}) {
		t.Errorf("non-IgnoreParentScale child world position = %v, want {10 0 0}", otherGot)
	}
}
