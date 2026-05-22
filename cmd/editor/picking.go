package main

import (
	"indigo/app"
	"indigo/ecs"
	"indigo/render"
	"indigo/render/asset"
	"indigo/ui"
)

// handlePickResult routes the pick result, except when a gizmo
// drag is active, the pointer is over UI, or the pointer is over
// a gizmo axis - in all three cases the pick lands on whatever
// scene mesh was behind the overlay and changing selection would
// be wrong.
func handlePickResult(worlds app.Worlds, pickedID uint32) {
	gizmo := *ecs.MustResource[*render.Gizmos](worlds.Engine)
	if gizmo != nil && gizmo.Dragging {
		return
	}
	if worlds.UI != nil {
		pointer := ecs.MustResource[ui.PointerState](worlds.UI)
		if pointer.OverUI {
			return
		}
	}
	if gizmo != nil && gizmo.HoverAxis >= 0 {
		return
	}
	applySelection(worlds.Engine, pickedID)
}

// applySelection clears every existing [render.Selected] tag and
// tags the entity whose ID matches pickedID. ID 0 means "picked
// nothing" and just clears.
func applySelection(engine *ecs.World, pickedID uint32) {
	clearSelection(engine)
	if pickedID == 0 {
		return
	}
	renderMask := ecs.MustMaskOf[asset.RenderMesh](engine)
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

// applyEntitySelection clears every existing Selected tag and tags
// the supplied entity directly. Used by the entity tree to select
// any named entity (including the Sun light, which has no
// RenderMesh and would be missed by ID-only [applySelection]).
func applyEntitySelection(engine *ecs.World, entity ecs.Entity) {
	clearSelection(engine)
	if entity.ID == 0 {
		return
	}
	ecs.Add[render.Selected](engine, entity)
}

func clearSelection(engine *ecs.World) {
	selectedMask := ecs.MustMaskOf[render.Selected](engine)
	var toDeselect []ecs.Entity
	engine.ForEach(selectedMask, 0, func(e ecs.Entity, _ *ecs.Archetype, _ int) {
		toDeselect = append(toDeselect, e)
	})
	for _, e := range toDeselect {
		ecs.Remove[render.Selected](engine, e)
	}
}
