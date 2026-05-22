package main

import (
	"fmt"

	"indigo/app"
	"indigo/ecs"
	"indigo/render"
	"indigo/render/asset"
	"indigo/transform"
	"indigo/window"
)

// Field geometry. The play area is a rectangle in the world's XZ
// plane (Y is up). The camera looks straight down at the field.
const (
	fieldMinX   float32 = -5.0
	fieldMaxX   float32 = 5.0
	fieldMinZ   float32 = -6.0
	fieldMaxZ   float32 = 6.0
	fieldBottom float32 = fieldMaxZ - 1.5

	paddleHalfWidth float32 = 0.9
	paddleHalfDepth float32 = 0.25
	paddleHalfY     float32 = 0.25
	paddleSpeed     float32 = 7.0

	ballRadius float32 = 0.22
	ballSpeed  float32 = 6.0

	brickHalfWidth float32 = 0.45
	brickHalfDepth float32 = 0.22
	brickHalfY     float32 = 0.22
	brickRows              = 4
	brickCols              = 9
	brickStartZ    float32 = -4.5
	brickRowGap    float32 = 0.65

	startingLives = 3
)

// spawnBreakoutScene populates both worlds with the paddle, ball, and
// brick wall. Engine entities carry transforms + RenderMesh +
// Material; game entities carry the gameplay components and an
// EngineEntity link to the engine-side render twin. Every entity
// uses the same white-cube mesh; the tint comes from the per-entity
// Material component.
func spawnBreakoutScene(worlds app.Worlds, palette brickPalette) {
	engineMask := ecs.MustMaskOf[transform.LocalTransform](worlds.Engine) |
		ecs.MustMaskOf[transform.GlobalTransform](worlds.Engine) |
		ecs.MustMaskOf[transform.LocalTransformDirty](worlds.Engine) |
		ecs.MustMaskOf[asset.RenderMesh](worlds.Engine) |
		ecs.MustMaskOf[asset.Material](worlds.Engine)

	paddleMask := ecs.MustMaskOf[Paddle](worlds.Game) | ecs.MustMaskOf[app.EngineEntity](worlds.Game)
	ballMask := ecs.MustMaskOf[Ball](worlds.Game) | ecs.MustMaskOf[app.EngineEntity](worlds.Game)
	brickMask := ecs.MustMaskOf[Brick](worlds.Game) | ecs.MustMaskOf[app.EngineEntity](worlds.Game)

	paddlePos := transform.Vec3{0, 0, fieldBottom}
	paddleEngine := worlds.Engine.Spawn(engineMask)
	paddleLocal := transform.FromTranslation(paddlePos)
	paddleLocal.Scale = transform.Vec3{paddleHalfWidth * 2, paddleHalfY * 2, paddleHalfDepth * 2}
	ecs.Set(worlds.Engine, paddleEngine, paddleLocal)
	ecs.Set(worlds.Engine, paddleEngine, transform.IdentityGlobalTransform())
	ecs.Set(worlds.Engine, paddleEngine, asset.RenderMesh{Mesh: palette.Cube})
	ecs.Set(worlds.Engine, paddleEngine, asset.Material{BaseColor: palette.Paddle})

	paddleGame := worlds.Game.Spawn(paddleMask)
	ecs.Set(worlds.Game, paddleGame, Paddle{
		Position: paddlePos,
		HalfSize: transform.Vec3{paddleHalfWidth, paddleHalfY, paddleHalfDepth},
		Speed:    paddleSpeed,
	})
	ecs.Set(worlds.Game, paddleGame, app.EngineEntity{Entity: paddleEngine})

	ballPos := transform.Vec3{0, 0, fieldBottom - paddleHalfDepth - ballRadius}
	ballEngine := worlds.Engine.Spawn(engineMask)
	ballLocal := transform.FromTranslation(ballPos)
	ballLocal.Scale = transform.Vec3{ballRadius * 2, ballRadius * 2, ballRadius * 2}
	ecs.Set(worlds.Engine, ballEngine, ballLocal)
	ecs.Set(worlds.Engine, ballEngine, transform.IdentityGlobalTransform())
	ecs.Set(worlds.Engine, ballEngine, asset.RenderMesh{Mesh: palette.Cube})
	ecs.Set(worlds.Engine, ballEngine, asset.Material{BaseColor: palette.Ball})

	ballGame := worlds.Game.Spawn(ballMask)
	ecs.Set(worlds.Game, ballGame, Ball{
		Position: ballPos,
		Velocity: transform.Vec3{0, 0, 0},
		Radius:   ballRadius,
		Launched: false,
	})
	ecs.Set(worlds.Game, ballGame, app.EngineEntity{Entity: ballEngine})

	brickWidth := brickHalfWidth * 2
	totalWidth := brickWidth * brickCols
	startX := -totalWidth*0.5 + brickHalfWidth
	for row := 0; row < brickRows; row++ {
		rowColor := palette.Rows[row%len(palette.Rows)]
		score := (brickRows - row) * 10
		z := float32(brickStartZ) + float32(row)*brickRowGap
		for col := 0; col < brickCols; col++ {
			x := startX + float32(col)*brickWidth
			pos := transform.Vec3{x, 0, z}

			engineEntity := worlds.Engine.Spawn(engineMask)
			local := transform.FromTranslation(pos)
			local.Scale = transform.Vec3{brickHalfWidth * 2, brickHalfY * 2, brickHalfDepth * 2}
			ecs.Set(worlds.Engine, engineEntity, local)
			ecs.Set(worlds.Engine, engineEntity, transform.IdentityGlobalTransform())
			ecs.Set(worlds.Engine, engineEntity, asset.RenderMesh{Mesh: palette.Cube})
			ecs.Set(worlds.Engine, engineEntity, asset.Material{BaseColor: rowColor})

			gameEntity := worlds.Game.Spawn(brickMask)
			ecs.Set(worlds.Game, gameEntity, Brick{
				Position: pos,
				HalfSize: transform.Vec3{brickHalfWidth, brickHalfY, brickHalfDepth},
				Score:    score,
			})
			ecs.Set(worlds.Game, gameEntity, app.EngineEntity{Entity: engineEntity})
		}
	}
}

// breakoutInputSystem reads the engine world's [render.Input] (filled
// by the platform callbacks each frame) and writes paddle motion plus
// gameplay edge events (launch, reset) into the game world. A and D
// are held checks via [render.InputIsKeyDown]; space and R are
// edge-triggered via KeysJustDown.
func breakoutInputSystem(game *ecs.World) {
	engineRef := ecs.MustResource[app.EngineRef](game)
	engine := engineRef.World
	input := ecs.MustResource[render.Input](engine)
	state := ecs.MustResource[GameState](game)
	delta := ecs.MustResource[window.Window](game).Timing.DeltaSeconds

	leftHeld := render.InputIsKeyDown(input, 'A')
	rightHeld := render.InputIsKeyDown(input, 'D')
	launchPressed := false
	for _, key := range input.KeysJustDown {
		switch key {
		case ' ':
			launchPressed = true
		case 'R':
			if state.Won || state.Lost {
				state.RequestReset = true
			}
		}
	}

	paddleMask := ecs.MustMaskOf[Paddle](game) | ecs.MustMaskOf[app.EngineEntity](game)
	game.ForEach(paddleMask, 0, func(_ ecs.Entity, table *ecs.Archetype, index int) {
		paddles, _ := ecs.Column[Paddle](game, table)
		p := &paddles[index]
		move := float32(0)
		if leftHeld {
			move -= 1
		}
		if rightHeld {
			move += 1
		}
		p.Position[0] += move * p.Speed * delta
		minX := fieldMinX + p.HalfSize[0]
		maxX := fieldMaxX - p.HalfSize[0]
		if p.Position[0] < minX {
			p.Position[0] = minX
		}
		if p.Position[0] > maxX {
			p.Position[0] = maxX
		}
	})

	if !launchPressed || state.Lost || state.Won {
		return
	}

	state.Started = true
	ballMask := ecs.MustMaskOf[Ball](game) | ecs.MustMaskOf[app.EngineEntity](game)
	game.ForEach(ballMask, 0, func(_ ecs.Entity, table *ecs.Archetype, index int) {
		balls, _ := ecs.Column[Ball](game, table)
		b := &balls[index]
		if b.Launched {
			return
		}
		b.Launched = true
		b.Velocity = transform.Vec3{0.5, 0, -1}.Normalize().Mul(ballSpeed)
	})
}

// breakoutBallSystem advances the ball position, walls/ceiling
// bouncing, then bounces off the paddle and any bricks it overlaps.
// Brick hits despawn the linked engine entity through the named sync
// API. The ball stays glued to the paddle while unlaunched.
func breakoutBallSystem(game *ecs.World) {
	engineRef := ecs.MustResource[app.EngineRef](game)
	engine := engineRef.World
	state := ecs.MustResource[GameState](game)
	delta := ecs.MustResource[window.Window](game).Timing.DeltaSeconds

	var paddlePos transform.Vec3
	var paddleHalf transform.Vec3
	paddleMask := ecs.MustMaskOf[Paddle](game) | ecs.MustMaskOf[app.EngineEntity](game)
	game.ForEach(paddleMask, 0, func(_ ecs.Entity, table *ecs.Archetype, index int) {
		paddles, _ := ecs.Column[Paddle](game, table)
		paddlePos = paddles[index].Position
		paddleHalf = paddles[index].HalfSize
	})

	type brickHit struct {
		gameEntity ecs.Entity
		score      int
	}
	var hits []brickHit

	ballMask := ecs.MustMaskOf[Ball](game) | ecs.MustMaskOf[app.EngineEntity](game)
	game.ForEach(ballMask, 0, func(_ ecs.Entity, table *ecs.Archetype, index int) {
		balls, _ := ecs.Column[Ball](game, table)
		b := &balls[index]

		if !b.Launched {
			b.Position[0] = paddlePos[0]
			b.Position[2] = paddlePos[2] - paddleHalfDepth - b.Radius
			return
		}

		b.Position[0] += b.Velocity[0] * delta
		b.Position[2] += b.Velocity[2] * delta

		if b.Position[0]-b.Radius < fieldMinX {
			b.Position[0] = fieldMinX + b.Radius
			b.Velocity[0] = -b.Velocity[0]
		}
		if b.Position[0]+b.Radius > fieldMaxX {
			b.Position[0] = fieldMaxX - b.Radius
			b.Velocity[0] = -b.Velocity[0]
		}
		if b.Position[2]-b.Radius < fieldMinZ {
			b.Position[2] = fieldMinZ + b.Radius
			b.Velocity[2] = -b.Velocity[2]
		}

		if collideCircleBox(b.Position, b.Radius, paddlePos, paddleHalf) {
			b.Velocity[2] = -absFloat(b.Velocity[2])
			offset := (b.Position[0] - paddlePos[0]) / paddleHalf[0]
			b.Velocity[0] = clampFloat(offset, -1, 1) * ballSpeed * 0.7
			b.Velocity = b.Velocity.Normalize().Mul(ballSpeed)
			b.Position[2] = paddlePos[2] - paddleHalf[2] - b.Radius
		}

		brickMask := ecs.MustMaskOf[Brick](game) | ecs.MustMaskOf[app.EngineEntity](game)
		game.ForEach(brickMask, 0, func(brickEntity ecs.Entity, brickTable *ecs.Archetype, brickIndex int) {
			bricks, _ := ecs.Column[Brick](game, brickTable)
			brick := &bricks[brickIndex]
			if !collideCircleBox(b.Position, b.Radius, brick.Position, brick.HalfSize) {
				return
			}
			overlapX := (brick.HalfSize[0] + b.Radius) - absFloat(b.Position[0]-brick.Position[0])
			overlapZ := (brick.HalfSize[2] + b.Radius) - absFloat(b.Position[2]-brick.Position[2])
			if overlapX < overlapZ {
				b.Velocity[0] = -b.Velocity[0]
				if b.Position[0] < brick.Position[0] {
					b.Position[0] -= overlapX
				} else {
					b.Position[0] += overlapX
				}
			} else {
				b.Velocity[2] = -b.Velocity[2]
				if b.Position[2] < brick.Position[2] {
					b.Position[2] -= overlapZ
				} else {
					b.Position[2] += overlapZ
				}
			}
			hits = append(hits, brickHit{gameEntity: brickEntity, score: brick.Score})
		})

		if b.Position[2]-b.Radius > fieldMaxZ {
			b.Launched = false
			b.Velocity = transform.Vec3{0, 0, 0}
			state.Lives--
			if state.Lives <= 0 {
				state.Lost = true
			}
		}
	})

	for _, hit := range hits {
		state.Score += hit.score
		app.DespawnLinked(engine, game, hit.gameEntity)
	}
}

// breakoutWinSystem flags the GameState as won when the brick wall
// is empty.
func breakoutWinSystem(game *ecs.World) {
	state := ecs.MustResource[GameState](game)
	if state.Won || state.Lost {
		return
	}
	if game.CountQuery(ecs.MustMaskOf[Brick](game), 0) == 0 {
		state.Won = true
	}
}

// breakoutSyncSystem writes the game-side positions of the paddle,
// ball, and any surviving bricks into their engine counterparts'
// LocalTransforms via the [app] sync API. The bricks don't move so
// this is technically redundant for them, but it keeps the pattern
// uniform with the spinner demo.
func breakoutSyncSystem(game *ecs.World) {
	engineRef := ecs.MustResource[app.EngineRef](game)
	engine := engineRef.World

	paddleMask := ecs.MustMaskOf[Paddle](game) | ecs.MustMaskOf[app.EngineEntity](game)
	game.ForEach(paddleMask, 0, func(_ ecs.Entity, table *ecs.Archetype, index int) {
		paddles, _ := ecs.Column[Paddle](game, table)
		links, _ := ecs.Column[app.EngineEntity](game, table)
		app.SyncEngineTranslation(engine, links[index], paddles[index].Position)
	})

	ballMask := ecs.MustMaskOf[Ball](game) | ecs.MustMaskOf[app.EngineEntity](game)
	game.ForEach(ballMask, 0, func(_ ecs.Entity, table *ecs.Archetype, index int) {
		balls, _ := ecs.Column[Ball](game, table)
		links, _ := ecs.Column[app.EngineEntity](game, table)
		app.SyncEngineTranslation(engine, links[index], balls[index].Position)
	})
}

// titleForState renders the current score/lives/status into the
// window title text. Stand-in for proper text rendering.
func titleForState(state *GameState) string {
	suffix := ""
	switch {
	case state.Won:
		suffix = "  (you win, press R)"
	case state.Lost:
		suffix = "  (game over, press R)"
	case !state.Started:
		suffix = "  (A/D to move, space to launch)"
	}
	return fmt.Sprintf("breakout -- score %d   lives %d%s", state.Score, state.Lives, suffix)
}

func collideCircleBox(circle transform.Vec3, radius float32, box transform.Vec3, half transform.Vec3) bool {
	dx := absFloat(circle[0] - box[0])
	dz := absFloat(circle[2] - box[2])
	if dx > half[0]+radius || dz > half[2]+radius {
		return false
	}
	if dx <= half[0] || dz <= half[2] {
		return true
	}
	cornerDX := dx - half[0]
	cornerDZ := dz - half[2]
	return cornerDX*cornerDX+cornerDZ*cornerDZ <= radius*radius
}

func absFloat(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func clampFloat(x, lo, hi float32) float32 {
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}
