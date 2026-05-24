package main

import (
	"github.com/matthewjberger/indigo/app"
	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/internal/render/pass"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/ui"
)

func syncUiPointer(worlds app.Worlds) {
	if worlds.UI == nil {
		return
	}
	input := ecs.MustResource[render.Input](worlds.Engine)
	pointer := ecs.MustResource[ui.PointerState](worlds.UI)
	prevLeft := pointer.LeftDown
	prevRight := pointer.RightDown
	pointer.X = input.MousePosition[0]
	pointer.Y = input.MousePosition[1]
	pointer.LeftDown = input.LeftDown
	pointer.LeftJustDown = input.LeftDown && !prevLeft
	pointer.LeftJustUp = !input.LeftDown && prevLeft
	pointer.RightDown = input.RightDown
	pointer.RightJustDown = input.RightDown && !prevRight
	pointer.RightJustUp = !input.RightDown && prevRight
}

func driveTextInputs(worlds app.Worlds) {
	if worlds.UI == nil {
		return
	}
	input := ecs.MustResource[render.Input](worlds.Engine)
	hud := ecs.MustResource[HudHandles](worlds.Engine)
	pointer := ecs.MustResource[ui.PointerState](worlds.UI)

	focusedNow := pointer.FocusedEntity == hud.InspectorName

	if focusedNow && pointer.LeftJustDown {
		placeCaretAtPointer(worlds.UI, hud.InspectorName, pointer.X)
	}

	backspace := false
	enter := false
	left := false
	right := false
	for _, k := range input.KeysJustDown {
		switch k {
		case '\b':
			backspace = true
		case '\n', '\r':
			enter = true
		case '\x01':
			left = true
		case '\x02':
			right = true
		}
	}
	ui.AdvanceTextInputs(worlds.UI, input.Chars, backspace, enter)

	if focusedNow && (left || right) {
		if ti, ok := ecs.GetMut[ui.TextInput](worlds.UI, hud.InspectorName); ok {
			runes := []rune(ti.Buffer)
			if left && ti.Caret > 0 {
				ti.Caret--
			}
			if right && ti.Caret < len(runes) {
				ti.Caret++
			}
		}
	}

	for _, evt := range ecs.DrainEvents[ui.TextCommitted](worlds.UI) {
		if evt.Entity != hud.InspectorName {
			continue
		}
		commitInspectorName(worlds, evt.Value)
	}

	if hud.NameFocusedPrev && !focusedNow {
		if ti, ok := ecs.Get[ui.TextInput](worlds.UI, hud.InspectorName); ok {
			commitInspectorName(worlds, ti.Buffer)
		}
	}
	hud.NameFocusedPrev = focusedNow
}

func commitInspectorName(worlds app.Worlds, value string) {
	target, ok := pass.SelectedTarget(worlds.Engine)
	if !ok {
		return
	}
	if name, ok := ecs.GetMut[app.Name](worlds.Engine, target); ok {
		name.Value = value
	}
}

func placeCaretAtPointer(world *ecs.World, field ecs.Entity, pointerX float32) {
	node, ok := ecs.Get[ui.Node](world, field)
	if !ok {
		return
	}
	ti, ok := ecs.GetMut[ui.TextInput](world, field)
	if !ok {
		return
	}
	label, ok := ecs.Get[ui.Text](world, field)
	if !ok {
		return
	}
	scale := label.Scale
	if scale <= 0 {
		scale = 1
	}
	advance := float32(ui.FontGlyphWidth+1) * scale
	runes := []rune(ti.Buffer)
	labelWidth := float32(len(runes)) * advance
	originX := node.Resolved.X + (node.Resolved.Width-labelWidth)*0.5
	rel := pointerX - originX
	idx := int((rel + advance*0.5) / advance)
	if idx < 0 {
		idx = 0
	}
	if idx > len(runes) {
		idx = len(runes)
	}
	ti.Caret = idx
}

func handleRightClick(worlds app.Worlds) {
	if worlds.UI == nil {
		return
	}
	pointer := ecs.MustResource[ui.PointerState](worlds.UI)
	if !pointer.RightJustDown {
		return
	}
	hud := ecs.MustResource[HudHandles](worlds.Engine)
	for i, row := range hud.TreeRows {
		if i >= hud.TreeRowCount {
			break
		}
		nodeRef, ok := ecs.Get[ui.Node](worlds.UI, row)
		if !ok {
			continue
		}
		if !nodeRef.Resolved.Contains(pointer.X, pointer.Y) {
			continue
		}
		target := hud.TreeRowToEngine[i]
		hud.ContextTarget = target
		hud.OpenMenu = menuContextOpen
		moveContextMenu(worlds.UI, hud.ContextMenu, pointer.X, pointer.Y)
		return
	}
}

func moveContextMenu(world *ecs.World, menu menuPopup, x, y float32) {
	if node, ok := ecs.GetMut[ui.Node](world, menu.Panel); ok {
		node.X = x
		node.Y = y
	}
	ui.MarkLayoutDirty(world)
}

func handleUiClicks(worlds app.Worlds) {
	if worlds.UI == nil {
		return
	}
	hud := ecs.MustResource[HudHandles](worlds.Engine)
	gizmo := *ecs.MustResource[*render.Gizmos](worlds.Engine)

	clickHandled := false
	for _, evt := range ecs.DrainEvents[ui.EntityClicked](worlds.UI) {
		clickHandled = true
		switch evt.Entity {
		case hud.TranslateButton:
			gizmo.Mode = render.GizmoTranslate
			hud.OpenMenu = menuClosed
		case hud.RotateButton:
			gizmo.Mode = render.GizmoRotate
			hud.OpenMenu = menuClosed
		case hud.ScaleButton:
			gizmo.Mode = render.GizmoScale
			hud.OpenMenu = menuClosed
		case hud.FileButton:
			hud.OpenMenu = toggleMenu(hud.OpenMenu, menuFileOpen)
		case hud.EditButton:
			hud.OpenMenu = toggleMenu(hud.OpenMenu, menuEditOpen)
		case hud.ViewButton:
			hud.OpenMenu = toggleMenu(hud.OpenMenu, menuViewOpen)
		case hud.AssetsButton:
			hud.OpenMenu = toggleMenu(hud.OpenMenu, menuAssetsOpen)
		case hud.ViewerButton:
			settings := ecs.MustResource[render.Graphics](worlds.Engine)
			settings.ViewerMode = !settings.ViewerMode
			hud.OpenMenu = menuClosed
			if settings.ViewerMode {
				hud.ModelRoots = loadedSceneRoots(worlds.Engine)
				hud.FramePending = true
			}
		default:
			if handleMenuItem(worlds, hud, evt.Entity) {
				hud.OpenMenu = menuClosed
				break
			}
			if handleKhronosRow(worlds, hud, evt.Entity) {
				break
			}
			for i, row := range hud.TreeRows {
				if evt.Entity == row {
					target := hud.TreeRowToEngine[i]
					applyEntitySelection(worlds.Engine, target)
					hud.OpenMenu = menuClosed
					break
				}
			}
		}
	}

	if hud.OpenMenu == menuClosed {
		return
	}

	pointer := ecs.MustResource[ui.PointerState](worlds.UI)
	if !pointer.LeftJustDown || clickHandled {
		return
	}
	if pointerOverMenuGeometry(worlds.UI, hud) {
		return
	}
	hud.OpenMenu = menuClosed
}

func toggleMenu(current, target int) int {
	if current == target {
		return menuClosed
	}
	return target
}

func handleMenuItem(worlds app.Worlds, hud *HudHandles, entity ecs.Entity) bool {
	if idx := matchItem(hud.FileMenu, entity); idx >= 0 {
		if idx == 3 {
			hud.RequestExit = true
		}
		return true
	}
	if idx := matchItem(hud.EditMenu, entity); idx >= 0 {
		if idx == 2 {
			applySelection(worlds.Engine, 0)
		}
		return true
	}
	if idx := matchItem(hud.AssetsMenu, entity); idx >= 0 {
		if idx == 0 {
			hud.KhronosOpen = !hud.KhronosOpen
			if hud.KhronosOpen {
				(*ecs.MustResource[*KhronosBrowser](worlds.Engine)).EnsureLoaded()
			}
		}
		return true
	}
	if idx := matchItem(hud.ViewMenu, entity); idx >= 0 {
		switch idx {
		case 0:
			controller := ecs.MustResource[render.PanOrbitController](worlds.Engine)
			defaults := render.DefaultPanOrbitController()
			controller.TargetYaw = defaults.TargetYaw
			controller.TargetPitch = defaults.TargetPitch
			controller.TargetRadius = defaults.TargetRadius
			controller.TargetFocus = defaults.TargetFocus
		case 1:
			settings := ecs.MustResource[render.Graphics](worlds.Engine)
			settings.ShowGrid = !settings.ShowGrid
		case 2:
			settings := ecs.MustResource[render.Graphics](worlds.Engine)
			settings.ShowSky = !settings.ShowSky
		case 3:
			settings := ecs.MustResource[render.Graphics](worlds.Engine)
			settings.ShowBounds = !settings.ShowBounds
		case 4:
			settings := ecs.MustResource[render.Graphics](worlds.Engine)
			settings.ShowNormals = !settings.ShowNormals
		case 5:
			settings := ecs.MustResource[render.Graphics](worlds.Engine)
			settings.ShowSkeletons = !settings.ShowSkeletons
		}
		return true
	}
	if idx := matchItem(hud.ContextMenu, entity); idx >= 0 {
		switch idx {
		case 0:
			worlds.Engine.Despawn(hud.ContextTarget)
			hud.ContextTarget = ecs.Entity{}
		}
		return true
	}
	return false
}

func handleKhronosRow(worlds app.Worlds, hud *HudHandles, entity ecs.Entity) bool {
	for i, row := range hud.Khronos.Rows {
		if row != entity {
			continue
		}
		entryIndex := hud.Khronos.RowToEntry[i]
		if entryIndex < 0 {
			return true
		}
		browser := *ecs.MustResource[*KhronosBrowser](worlds.Engine)
		entries := browser.Entries()
		if entryIndex < len(entries) {
			browser.FetchEntry(entries[entryIndex])
		}
		return true
	}
	return false
}

func matchItem(menu menuPopup, entity ecs.Entity) int {
	for i := 0; i < menu.Count; i++ {
		if menu.Items[i] == entity {
			return i
		}
	}
	return -1
}

func pointerOverMenuGeometry(uiWorld *ecs.World, hud *HudHandles) bool {
	pointer := ecs.MustResource[ui.PointerState](uiWorld)
	var menu menuPopup
	var button ecs.Entity
	switch hud.OpenMenu {
	case menuFileOpen:
		menu, button = hud.FileMenu, hud.FileButton
	case menuEditOpen:
		menu, button = hud.EditMenu, hud.EditButton
	case menuViewOpen:
		menu, button = hud.ViewMenu, hud.ViewButton
	case menuAssetsOpen:
		menu, button = hud.AssetsMenu, hud.AssetsButton
	case menuContextOpen:
		menu = hud.ContextMenu
	default:
		return false
	}
	if button.ID != 0 && nodeContains(uiWorld, button, pointer.X, pointer.Y) {
		return true
	}
	if nodeContains(uiWorld, menu.Panel, pointer.X, pointer.Y) {
		return true
	}
	for i := 0; i < menu.Count; i++ {
		if nodeContains(uiWorld, menu.Items[i], pointer.X, pointer.Y) {
			return true
		}
	}
	return false
}

func nodeContains(world *ecs.World, entity ecs.Entity, x, y float32) bool {
	node, ok := ecs.Get[ui.Node](world, entity)
	if !ok {
		return false
	}
	return node.Resolved.Contains(x, y)
}
