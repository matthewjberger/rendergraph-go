package app_test

import (
	"testing"

	"indigo/app"
	"indigo/ecs"
	"indigo/transform"
)

func newEngineGame(t *testing.T) (*ecs.World, *ecs.World) {
	t.Helper()
	engine := ecs.New()
	ecs.Register[transform.LocalTransform](engine)
	ecs.Register[transform.GlobalTransform](engine)
	ecs.Register[transform.Parent](engine)
	ecs.Register[transform.LocalTransformDirty](engine)
	ecs.Register[transform.IgnoreParentScale](engine)
	ecs.SetResource(engine, transform.NewTransformState())

	game := ecs.New()
	ecs.Register[app.EngineEntity](game)

	return engine, game
}

func newLinkedPair(t *testing.T, engine, game *ecs.World) (engineEntity ecs.Entity, gameEntity ecs.Entity, link app.EngineEntity) {
	t.Helper()
	engineEntity = engine.Spawn(0)
	ecs.Set(engine, engineEntity, transform.IdentityLocalTransform())

	gameEntity = game.Spawn(0)
	link = app.EngineEntity{Entity: engineEntity}
	ecs.Set(game, gameEntity, link)

	return
}

func TestSyncEngineTransformWritesAndMarksDirty(t *testing.T) {
	engine, game := newEngineGame(t)
	engineEntity, _, link := newLinkedPair(t, engine, game)

	target := transform.FromTranslation(transform.Vec3{1, 2, 3})
	app.SyncEngineTransform(engine, link, target)

	got, ok := ecs.Get[transform.LocalTransform](engine, engineEntity)
	if !ok {
		t.Fatal("engine entity missing LocalTransform after sync")
	}
	if got.Translation != target.Translation {
		t.Fatalf("translation = %v, want %v", got.Translation, target.Translation)
	}
	if !ecs.Has[transform.LocalTransformDirty](engine, engineEntity) {
		t.Fatal("sync should mark engine entity dirty")
	}
}

func TestSyncEngineTransformStaleLinkIsNoOp(t *testing.T) {
	engine, game := newEngineGame(t)
	engineEntity, _, link := newLinkedPair(t, engine, game)
	engine.Despawn(engineEntity)

	defer func() {
		if recover() != nil {
			t.Fatal("stale link should not panic")
		}
	}()
	app.SyncEngineTransform(engine, link, transform.FromTranslation(transform.Vec3{9, 9, 9}))
}

func TestSyncEngineTranslationLeavesRotationIntact(t *testing.T) {
	engine, game := newEngineGame(t)
	engineEntity, _, link := newLinkedPair(t, engine, game)
	rotation := transform.QuatFromAxisAngle(1.5, transform.Vec3{0, 1, 0})
	ecs.Set(engine, engineEntity, transform.LocalTransform{
		Translation: transform.Vec3{0, 0, 0},
		Rotation:    rotation,
		Scale:       transform.Vec3{1, 1, 1},
	})

	app.SyncEngineTranslation(engine, link, transform.Vec3{4, 5, 6})

	got, _ := ecs.Get[transform.LocalTransform](engine, engineEntity)
	if got.Translation != (transform.Vec3{4, 5, 6}) {
		t.Fatalf("translation = %v, want (4 5 6)", got.Translation)
	}
	if got.Rotation != rotation {
		t.Fatalf("rotation changed: %v want %v", got.Rotation, rotation)
	}
}

func TestSyncEngineRotationLeavesTranslationIntact(t *testing.T) {
	engine, game := newEngineGame(t)
	engineEntity, _, link := newLinkedPair(t, engine, game)
	ecs.Set(engine, engineEntity, transform.FromTranslation(transform.Vec3{4, 5, 6}))

	rotation := transform.QuatFromAxisAngle(0.7, transform.Vec3{1, 0, 0})
	app.SyncEngineRotation(engine, link, rotation)

	got, _ := ecs.Get[transform.LocalTransform](engine, engineEntity)
	if got.Translation != (transform.Vec3{4, 5, 6}) {
		t.Fatalf("translation changed: %v", got.Translation)
	}
	if got.Rotation != rotation {
		t.Fatalf("rotation = %v, want %v", got.Rotation, rotation)
	}
}

func TestDespawnLinkedRemovesBoth(t *testing.T) {
	engine, game := newEngineGame(t)
	engineEntity, gameEntity, _ := newLinkedPair(t, engine, game)

	app.DespawnLinked(engine, game, gameEntity)

	if _, ok := ecs.Get[app.EngineEntity](game, gameEntity); ok {
		t.Fatal("game entity should be despawned")
	}
	if _, ok := ecs.Get[transform.LocalTransform](engine, engineEntity); ok {
		t.Fatal("engine entity should be despawned")
	}
}

func TestDespawnLinkedWithoutLinkOnlyTouchesGame(t *testing.T) {
	engine, game := newEngineGame(t)
	gameEntity := game.Spawn(0)
	engineEntity := engine.Spawn(0)
	ecs.Set(engine, engineEntity, transform.IdentityLocalTransform())

	app.DespawnLinked(engine, game, gameEntity)

	if _, ok := ecs.Get[transform.LocalTransform](engine, engineEntity); !ok {
		t.Fatal("unlinked engine entity should not be despawned")
	}
}
