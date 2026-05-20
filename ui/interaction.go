package ui

import (
	"indigo/ecs"
)

// PointerState mirrors the engine-side mouse for the UI world's
// consumption: position in pixel coords (viewport-relative) plus
// down/up edges for the primary button. The platform main loop
// copies this from [render.Input] into the UI world once per frame.
//
// OverUI is updated by [InteractionSystem] each frame: true while
// the pointer is over any Interactive UI node. Apps consult this
// (typically from a pick-result handler) to suppress 3D selection
// when the user clicks the HUD.
type PointerState struct {
	X, Y         float32
	LeftDown     bool
	LeftJustDown bool
	LeftJustUp   bool
	OverUI       bool

	pressedEntity ecs.Entity
	hasPressed    bool
}

// InteractionSystem walks every UI entity that has both Node and
// Interactive components, updates Hovered and Pressed based on the
// current PointerState, and emits [EntityClicked] events on a
// press-release sequence that started and ended on the same entity.
//
// Hit testing is the topmost-hit-wins variant: entities later in
// iteration order shadow earlier ones. For the current scene
// complexity (a few buttons) the ordering is stable enough; layered
// menus would want explicit z-order, deferred.
func InteractionSystem(world *ecs.World) {
	pointer := ecs.Resource[PointerState](world)

	mask := ecs.MaskOf[Node](world) | ecs.MaskOf[Interactive](world)

	var hit ecs.Entity
	hasHit := false
	world.ForEach(mask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		node, _ := ecs.Get[Node](world, entity)
		if node.Resolved.Contains(pointer.X, pointer.Y) {
			hit = entity
			hasHit = true
		}
	})
	pointer.OverUI = hasHit

	var clearList, hoverList, pressList []ecs.Entity
	world.ForEach(mask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		if hasHit && entity == hit {
			hoverList = append(hoverList, entity)
			if pointer.LeftDown {
				pressList = append(pressList, entity)
			}
		} else {
			clearList = append(clearList, entity)
		}
	})

	for _, e := range clearList {
		if i, ok := ecs.GetMut[Interactive](world, e); ok {
			i.Hovered = false
			i.Pressed = false
		}
	}
	for _, e := range hoverList {
		if i, ok := ecs.GetMut[Interactive](world, e); ok {
			i.Hovered = true
			i.Pressed = pointer.LeftDown
		}
	}
	_ = pressList

	if pointer.LeftJustDown && hasHit {
		pointer.pressedEntity = hit
		pointer.hasPressed = true
	}
	if pointer.LeftJustUp {
		if pointer.hasPressed && hasHit && pointer.pressedEntity == hit {
			ecs.Send(world, EntityClicked{Entity: hit})
		}
		pointer.hasPressed = false
	}
}
