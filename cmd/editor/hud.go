package main

import (
	"strconv"

	"indigo/app"
	"indigo/ecs"
	"indigo/render"
	"indigo/ui"
)

func syncUiPointer(worlds app.Worlds) {
	if worlds.UI == nil {
		return
	}
	input := ecs.Resource[render.Input](worlds.Engine)
	pointer := ecs.Resource[ui.PointerState](worlds.UI)
	prevDown := pointer.LeftDown
	pointer.X = input.MousePosition[0]
	pointer.Y = input.MousePosition[1]
	pointer.LeftDown = input.LeftDown
	pointer.LeftJustDown = input.LeftDown && !prevDown
	pointer.LeftJustUp = !input.LeftDown && prevDown
}

func handleUiClicks(worlds app.Worlds) {
	if worlds.UI == nil {
		return
	}
	hud := ecs.Resource[HudHandles](worlds.Engine)
	gizmo := *ecs.Resource[*render.Gizmos](worlds.Engine)
	for _, evt := range ecs.DrainEvents[ui.EntityClicked](worlds.UI) {
		switch evt.Entity {
		case hud.ClearButton:
			applySelection(worlds.Engine, 0)
			refreshHudLabel(worlds, 0)
		case hud.TranslateButton:
			gizmo.Mode = render.GizmoTranslate
		case hud.RotateButton:
			gizmo.Mode = render.GizmoRotate
		case hud.ScaleButton:
			gizmo.Mode = render.GizmoScale
		}
	}
}

func refreshHudLabel(worlds app.Worlds, entityID uint32) {
	if worlds.UI == nil {
		return
	}
	hud := ecs.Resource[HudHandles](worlds.Engine)
	label, ok := ecs.GetMut[ui.Text](worlds.UI, hud.StatusLabel)
	if !ok {
		return
	}
	if entityID == 0 {
		label.Content = "Pick a cube"
	} else {
		label.Content = "Selected " + strconv.FormatUint(uint64(entityID), 10)
	}
}

func refreshModeButtons(worlds app.Worlds) {
	if worlds.UI == nil {
		return
	}
	hud := ecs.Resource[HudHandles](worlds.Engine)
	gizmo := *ecs.Resource[*render.Gizmos](worlds.Engine)
	setModeColor(worlds.UI, hud.TranslateButton, gizmo.Mode == render.GizmoTranslate)
	setModeColor(worlds.UI, hud.RotateButton, gizmo.Mode == render.GizmoRotate)
	setModeColor(worlds.UI, hud.ScaleButton, gizmo.Mode == render.GizmoScale)
}

func setModeColor(world *ecs.World, entity ecs.Entity, active bool) {
	color, ok := ecs.GetMut[ui.Color](world, entity)
	if !ok {
		return
	}
	if active {
		color.RGBA = [4]float32{0.22, 0.5, 0.86, 1}
	} else {
		color.RGBA = [4]float32{0.18, 0.21, 0.28, 1}
	}
}
