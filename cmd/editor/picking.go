package main

import (
	"indigo/app"
	"indigo/ecs"
	"indigo/render"
	"indigo/ui"
)

// handlePickResult routes the pick result, except when a gizmo
// drag is active, the pointer is over UI, or the pointer is over
// a gizmo axis - in all three cases the pick lands on whatever
// scene mesh was behind the overlay and changing selection would
// be wrong.
func handlePickResult(worlds app.Worlds, pickedID uint32) {
	gizmo := *ecs.Resource[*render.Gizmos](worlds.Engine)
	if gizmo != nil && gizmo.Dragging {
		return
	}
	if worlds.UI != nil {
		pointer := ecs.Resource[ui.PointerState](worlds.UI)
		if pointer.OverUI {
			return
		}
	}
	if gizmo != nil && gizmo.HoverAxis >= 0 {
		return
	}
	applySelection(worlds.Engine, pickedID)
	refreshHudLabel(worlds, pickedID)
}

// applySelection clears every existing [render.Selected] tag, then
// tags the entity whose ID matches pickedID. ID 0 means "picked
// nothing" and just clears.
func applySelection(engine *ecs.World, pickedID uint32) {
	selectedMask := ecs.MaskOf[render.Selected](engine)
	var toDeselect []ecs.Entity
	engine.ForEach(selectedMask, 0, func(e ecs.Entity, _ *ecs.Archetype, _ int) {
		toDeselect = append(toDeselect, e)
	})
	for _, e := range toDeselect {
		ecs.Remove[render.Selected](engine, e)
	}

	if pickedID == 0 {
		return
	}
	renderMask := ecs.MaskOf[render.RenderMesh](engine)
	var picked ecs.Entity
	found := false
	engine.ForEach(renderMask, 0, func(e ecs.Entity, _ *ecs.Archetype, _ int) {
		if !found && e.ID == pickedID {
			picked = e
			found = true
		}
	})
	if found {
		ecs.Add[render.Selected](engine, picked)
	}
}
