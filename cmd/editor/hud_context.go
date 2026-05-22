// HudContext is the per-frame pointer-bundle every HUD system
// reads — gathered once at frame start so refresh / input
// functions don't repeat `ecs.MustResource[...]` lookups. This
// file also holds the tiny formatting helpers that produce the
// inspector's POS/ROT/SCL/MAT label strings.
package main

import (
	"fmt"

	"indigo/app"
	"indigo/ecs"
	"indigo/render"
	"indigo/transform"
	"indigo/ui"
)

// HudContext is the per-frame snapshot of pointers HUD systems
// share. Computed once at the top of the frame and passed by
// pointer to every refresh / handler function so they stop
// repeating ecs.MustResource[...] lookups.
type HudContext struct {
	Worlds  app.Worlds
	Engine  *ecs.World
	UI      *ecs.World
	Input   *render.Input
	Hud     *HudHandles
	Pointer *ui.PointerState
	Gizmo   *render.Gizmos
}

func newHudContext(worlds app.Worlds) *HudContext {
	if worlds.UI == nil {
		return nil
	}
	return &HudContext{
		Worlds:  worlds,
		Engine:  worlds.Engine,
		UI:      worlds.UI,
		Input:   ecs.MustResource[render.Input](worlds.Engine),
		Hud:     ecs.MustResource[HudHandles](worlds.Engine),
		Pointer: ecs.MustResource[ui.PointerState](worlds.UI),
		Gizmo:   *ecs.MustResource[*render.Gizmos](worlds.Engine),
	}
}

func (c *HudContext) setText(entity ecs.Entity, content string) {
	if t, ok := ecs.GetMut[ui.Text](c.UI, entity); ok {
		t.Content = content
	}
}

func (c *HudContext) setColor(entity ecs.Entity, rgba [4]float32) {
	if col, ok := ecs.GetMut[ui.Color](c.UI, entity); ok {
		col.RGBA = rgba
	}
}

// caretBlinkPeriodSeconds is the half-period of the text-input
// caret blink: the caret is visible for one period and hidden for
// the next, so the full cycle is 2 * caretBlinkPeriodSeconds.
const caretBlinkPeriodSeconds = 0.5

// caretVisible reports whether the blinking caret should be drawn
// this frame given the engine uptime in seconds.
func caretVisible(uptime float32) bool {
	return int(uptime/caretBlinkPeriodSeconds)%2 == 0
}

// namedEntity pairs an engine entity with its display name for
// the hierarchy tree's sorted row list. Lives in this file
// because it's a HUD-side projection of [app.Name].
type namedEntity struct {
	Entity ecs.Entity
	Name   string
}

func formatVec3(v transform.Vec3) string {
	return fmt.Sprintf("%.2f %.2f %.2f", v[0], v[1], v[2])
}

func formatQuat(q transform.Quat) string {
	return fmt.Sprintf("%.2f %.2f %.2f %.2f", q.V.X(), q.V.Y(), q.V.Z(), q.W)
}

func formatRGBA(c [4]float32) string {
	return fmt.Sprintf("%.2f %.2f %.2f", c[0], c[1], c[2])
}
