// Scene-population helpers: spawns the demo grid + sun + light
// orbs at editor startup, plus the per-frame game-side spinner
// system that feeds rotation back into the engine world.
package main

import (
	"math"
	"strconv"

	"indigo/app"
	"indigo/ecs"
	"indigo/render"
	"indigo/render/asset"
	"indigo/transform"
	"indigo/window"
)

// initializeWorldEntities populates the engine and game worlds for
// the editor's demo scene. Engine entities get visual + transform
// components; each engine entity has a paired game entity carrying
// its Spinner state and an EngineEntity bridge. Light entities are
// added so the clustered lighting and shadow paths have local
// lights to cull and shadow casters.
//
// orbMesh is the mesh handle used for the visible body of each
// light orb; the orb's material is marked unlit so the mesh reads
// at its light's color regardless of which other lights illuminate
// it.
func initializeWorldEntities(worlds app.Worlds, meshes []asset.MeshHandle, meshNames []string, orbMesh asset.MeshHandle) {
	const (
		gridExtent  = 3
		gridSpacing = 1.5
	)

	lightMask := ecs.MustMaskOf[transform.LocalTransform](worlds.Engine) |
		ecs.MustMaskOf[transform.GlobalTransform](worlds.Engine) |
		ecs.MustMaskOf[transform.LocalTransformDirty](worlds.Engine) |
		ecs.MustMaskOf[render.Light](worlds.Engine) |
		ecs.MustMaskOf[app.Name](worlds.Engine)

	// Default sun: positioned up and back, aimed so the light's
	// -Z forward direction lands on the origin. Gives the scene a
	// classic upper-right key-light cast.
	sun := worlds.Engine.Spawn(lightMask)
	sunYaw := transform.QuatFromAxisAngle(1.821, transform.Vec3{0, 1, 0})
	sunPitch := transform.QuatFromAxisAngle(-0.969, transform.Vec3{1, 0, 0})
	sunTransform := transform.LocalTransform{
		Translation: transform.Vec3{4, 6, -1},
		Rotation:    sunYaw.Mul(sunPitch),
		Scale:       transform.Vec3{1, 1, 1},
	}
	ecs.Set(worlds.Engine, sun, sunTransform)
	ecs.Set(worlds.Engine, sun, transform.IdentityGlobalTransform())
	ecs.Set(worlds.Engine, sun, render.Light{
		Type:           render.LightTypeDirectional,
		Color:          transform.Vec3{1.0, 0.95, 0.8},
		Intensity:      3.5,
		Range:          100.0,
		InnerConeAngle: float32(math.Pi / 6),
		OuterConeAngle: float32(math.Pi / 4),
	})
	ecs.Set(worlds.Engine, sun, app.Name{Value: "Sun"})

	spawnLightOrbs(worlds, orbMesh)
	spawnGroundPlane(worlds)
	spawnDemoCamera(worlds)

	engineMask := ecs.MustMaskOf[transform.LocalTransform](worlds.Engine) |
		ecs.MustMaskOf[transform.GlobalTransform](worlds.Engine) |
		ecs.MustMaskOf[transform.LocalTransformDirty](worlds.Engine) |
		ecs.MustMaskOf[asset.RenderMesh](worlds.Engine) |
		ecs.MustMaskOf[app.Name](worlds.Engine)

	gameMask := ecs.MustMaskOf[Spinner](worlds.Game) |
		ecs.MustMaskOf[app.EngineEntity](worlds.Game)

	index := 0
	for x := -gridExtent; x <= gridExtent; x++ {
		for z := -gridExtent; z <= gridExtent; z++ {
			if x >= -1 && x <= 1 && z >= -1 && z <= 1 {
				continue
			}
			engineEntity := worlds.Engine.Spawn(engineMask)
			local := transform.FromTranslation(transform.Vec3{
				float32(x) * gridSpacing,
				0,
				float32(z) * gridSpacing,
			})
			local.Rotation = transform.QuatFromAxisAngle(
				float32(index)*0.4,
				transform.Vec3{0, 1, 0},
			)
			ecs.Set(worlds.Engine, engineEntity, local)
			ecs.Set(worlds.Engine, engineEntity, transform.IdentityGlobalTransform())
			meshIndex := index % len(meshes)
			ecs.Set(worlds.Engine, engineEntity, asset.RenderMesh{Mesh: meshes[meshIndex]})
			ecs.Set(worlds.Engine, engineEntity, app.Name{Value: meshNames[meshIndex] + " " + strconv.Itoa(index+1)})

			gameEntity := worlds.Game.Spawn(gameMask)
			ecs.Set(worlds.Game, gameEntity, Spinner{
				Axis:     transform.Vec3{0, 1, 0},
				Speed:    float32(math.Pi / 2),
				Rotation: local.Rotation,
			})
			ecs.Set(worlds.Game, gameEntity, app.EngineEntity{Entity: engineEntity})

			index++
		}
	}
}

// advanceSpinners is the game-side system. Walks the (Spinner |
// EngineEntity) archetype with [ecs.World.ForEach] for table-level
// column access, accumulates per-entity rotation, and writes the
// result to the engine world through [app.SyncEngineRotation]. Game
// state is the source of truth; reaching across into the engine world
// goes through the named sync API.
func advanceSpinners(game *ecs.World) {
	delta := ecs.MustResource[window.Window](game).Timing.DeltaSeconds
	engineRef := ecs.MustResource[app.EngineRef](game)

	spinnerMask := ecs.MustMaskOf[Spinner](game) | ecs.MustMaskOf[app.EngineEntity](game)

	game.ForEach(spinnerMask, 0, func(_ ecs.Entity, table *ecs.Archetype, index int) {
		spinners, _ := ecs.Column[Spinner](game, table)
		links, _ := ecs.Column[app.EngineEntity](game, table)

		s := &spinners[index]
		step := transform.QuatFromAxisAngle(s.Speed*delta, s.Axis)
		s.Rotation = step.Mul(s.Rotation).Normalize()

		app.SyncEngineRotation(engineRef.World, links[index], s.Rotation)
	})
}

// spawnGroundPlane spawns a flat 40x40 quad lying on the XZ plane
// at y=-0.5 so cast shadows have a surface to land on. Matte
// light-gray material; no special components beyond the standard
// renderable bundle.
func spawnGroundPlane(worlds app.Worlds) {
	engine := worlds.Engine
	primitives := ecs.MustResource[asset.Primitives](engine)
	groundMask := ecs.MustMaskOf[transform.LocalTransform](engine) |
		ecs.MustMaskOf[transform.GlobalTransform](engine) |
		ecs.MustMaskOf[transform.LocalTransformDirty](engine) |
		ecs.MustMaskOf[asset.RenderMesh](engine) |
		ecs.MustMaskOf[asset.Material](engine) |
		ecs.MustMaskOf[app.Name](engine)
	ground := engine.Spawn(groundMask)
	local := transform.IdentityLocalTransform()
	local.Translation = transform.Vec3{0, -3.0, 0}
	local.Scale = transform.Vec3{60, 1, 60}
	ecs.Set(engine, ground, local)
	ecs.Set(engine, ground, transform.IdentityGlobalTransform())
	ecs.Set(engine, ground, asset.RenderMesh{Mesh: primitives.UnitPlane})
	ecs.Set(engine, ground, asset.AlbedoMaterial([4]float32{0.5, 0.5, 0.5, 1.0}))
	ecs.Set(engine, ground, app.Name{Value: "Ground"})
}

// spawnLightOrbs scatters a handful of colored point lights and a
// downward spot light around the origin so the clustered light
// pipeline has something to cull. Each light entity also carries a
// small unlit cube the same color as its light, making the orb's
// position visible even when its light doesn't reach a given
// fragment.
func spawnLightOrbs(worlds app.Worlds, orbMesh asset.MeshHandle) {
	const orbScale float32 = 0.12
	scale := transform.Vec3{orbScale, orbScale, orbScale}

	type point struct {
		Name      string
		Position  transform.Vec3
		Color     transform.Vec3
		Intensity float32
		Range     float32
	}
	points := []point{
		{Name: "Red Orb", Position: transform.Vec3{2.0, 0.4, 0}, Color: transform.Vec3{1.0, 0.1, 0.1}, Intensity: 60, Range: 8.0},
		{Name: "Green Orb", Position: transform.Vec3{-2.0, 0.4, 0}, Color: transform.Vec3{0.1, 1.0, 0.2}, Intensity: 60, Range: 8.0},
		{Name: "Blue Orb", Position: transform.Vec3{0, 0.4, 2.0}, Color: transform.Vec3{0.1, 0.3, 1.0}, Intensity: 60, Range: 8.0},
		{Name: "Magenta Orb", Position: transform.Vec3{0, 0.4, -2.0}, Color: transform.Vec3{1.0, 0.2, 1.0}, Intensity: 60, Range: 8.0},
	}
	for _, p := range points {
		spawnPointOrb(worlds, orbMesh, p.Name, p.Position, scale, p.Color, p.Intensity, p.Range)
	}

	spawnSpotOrb(worlds, orbMesh,
		"Yellow Spot",
		transform.Vec3{0, 2.5, 0},
		scale,
		transform.Vec3{1.0, 0.95, 0.3},
		120.0,
		20.0,
		float32(math.Pi/8),
		float32(math.Pi/5),
	)
}

func spawnPointOrb(worlds app.Worlds, orbMesh asset.MeshHandle, name string, position, scale, color transform.Vec3, intensity, range_ float32) {
	mask := ecs.MustMaskOf[transform.LocalTransform](worlds.Engine) |
		ecs.MustMaskOf[transform.GlobalTransform](worlds.Engine) |
		ecs.MustMaskOf[transform.LocalTransformDirty](worlds.Engine) |
		ecs.MustMaskOf[asset.RenderMesh](worlds.Engine) |
		ecs.MustMaskOf[asset.Material](worlds.Engine) |
		ecs.MustMaskOf[render.Light](worlds.Engine) |
		ecs.MustMaskOf[app.Name](worlds.Engine)

	entity := worlds.Engine.Spawn(mask)
	ecs.Set(worlds.Engine, entity, transform.LocalTransform{
		Translation: position,
		Rotation:    transform.IdentityLocalTransform().Rotation,
		Scale:       scale,
	})
	ecs.Set(worlds.Engine, entity, transform.IdentityGlobalTransform())
	ecs.Set(worlds.Engine, entity, asset.RenderMesh{Mesh: orbMesh})

	orbMaterial := asset.EmissiveMaterial([3]float32{color[0], color[1], color[2]}, 8.0)
	orbMaterial.Unlit = true
	ecs.Set(worlds.Engine, entity, orbMaterial)

	ecs.Set(worlds.Engine, entity, render.Light{
		Type:        render.LightTypePoint,
		Color:       color,
		Intensity:   intensity,
		Range:       range_,
		CastShadows: true,
		ShadowBias:  0.005,
	})
	ecs.Set(worlds.Engine, entity, app.Name{Value: name})
}

// spawnSpotOrb spawns a spot-light orb whose cone aims straight
// down (-Y). The light entity's rotation rotates +Z to -Y so the
// transform's -Z column (which the renderer reads as the light's
// world-space direction) lands on (0, -1, 0).
func spawnSpotOrb(worlds app.Worlds, orbMesh asset.MeshHandle, name string, position, scale, color transform.Vec3, intensity, range_, innerCone, outerCone float32) {
	mask := ecs.MustMaskOf[transform.LocalTransform](worlds.Engine) |
		ecs.MustMaskOf[transform.GlobalTransform](worlds.Engine) |
		ecs.MustMaskOf[transform.LocalTransformDirty](worlds.Engine) |
		ecs.MustMaskOf[asset.RenderMesh](worlds.Engine) |
		ecs.MustMaskOf[asset.Material](worlds.Engine) |
		ecs.MustMaskOf[render.Light](worlds.Engine) |
		ecs.MustMaskOf[app.Name](worlds.Engine)

	entity := worlds.Engine.Spawn(mask)
	// Rotate +Z to -Y: rotation around +X by -PI/2.
	rotation := transform.QuatFromAxisAngle(-float32(math.Pi/2), transform.Vec3{1, 0, 0})
	ecs.Set(worlds.Engine, entity, transform.LocalTransform{
		Translation: position,
		Rotation:    rotation,
		Scale:       scale,
	})
	ecs.Set(worlds.Engine, entity, transform.IdentityGlobalTransform())
	ecs.Set(worlds.Engine, entity, asset.RenderMesh{Mesh: orbMesh})

	orbMaterial := asset.EmissiveMaterial([3]float32{color[0], color[1], color[2]}, 8.0)
	orbMaterial.Unlit = true
	ecs.Set(worlds.Engine, entity, orbMaterial)

	ecs.Set(worlds.Engine, entity, render.Light{
		Type:           render.LightTypeSpot,
		Color:          color,
		Intensity:      intensity,
		Range:          range_,
		InnerConeAngle: innerCone,
		OuterConeAngle: outerCone,
		CastShadows:    true,
		ShadowBias:     0.005,
	})
	ecs.Set(worlds.Engine, entity, app.Name{Value: name})
}

// spawnDemoCamera spawns a tagged scene-camera entity so the
// camera gizmo system has something to draw. The entity has no
// render mesh; the gizmo is the only visualization.
func spawnDemoCamera(worlds app.Worlds) {
	mask := ecs.MustMaskOf[transform.LocalTransform](worlds.Engine) |
		ecs.MustMaskOf[transform.GlobalTransform](worlds.Engine) |
		ecs.MustMaskOf[transform.LocalTransformDirty](worlds.Engine) |
		ecs.MustMaskOf[render.CameraMarker](worlds.Engine) |
		ecs.MustMaskOf[app.Name](worlds.Engine)
	entity := worlds.Engine.Spawn(mask)
	yaw := transform.QuatFromAxisAngle(float32(math.Pi/4), transform.Vec3{0, 1, 0})
	pitch := transform.QuatFromAxisAngle(-float32(math.Pi/12), transform.Vec3{1, 0, 0})
	ecs.Set(worlds.Engine, entity, transform.LocalTransform{
		Translation: transform.Vec3{-3, 1.5, 4},
		Rotation:    yaw.Mul(pitch),
		Scale:       transform.Vec3{1, 1, 1},
	})
	ecs.Set(worlds.Engine, entity, transform.IdentityGlobalTransform())
	ecs.Set(worlds.Engine, entity, render.CameraMarker{})
	ecs.Set(worlds.Engine, entity, app.Name{Value: "Camera"})
}
