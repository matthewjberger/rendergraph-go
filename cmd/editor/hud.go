package main

import (
	"fmt"
	"sort"

	"indigo/app"
	"indigo/ecs"
	"indigo/render"
	"indigo/render/asset"
	"indigo/render/pass"
	"indigo/transform"
	"indigo/ui"
	"indigo/window"
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
		if row.ID == 0 {
			continue
		}
		nodeRef, ok := ecs.Get[ui.Node](worlds.UI, row)
		if !ok {
			continue
		}
		if !nodeRef.Resolved.Contains(pointer.X, pointer.Y) {
			continue
		}
		target := hud.TreeRowToEngine[i]
		if target.ID == 0 {
			return
		}
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
		default:
			if handleMenuItem(worlds, hud, evt.Entity) {
				hud.OpenMenu = menuClosed
				break
			}
			for i, row := range hud.TreeRows {
				if evt.Entity == row {
					target := hud.TreeRowToEngine[i]
					if target.ID != 0 {
						applyEntitySelection(worlds.Engine, target)
					}
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
			settings := ecs.MustResource[render.GraphicsSettings](worlds.Engine)
			settings.ShowGrid = !settings.ShowGrid
		case 2:
			settings := ecs.MustResource[render.GraphicsSettings](worlds.Engine)
			settings.ShowSky = !settings.ShowSky
		case 3:
			settings := ecs.MustResource[render.GraphicsSettings](worlds.Engine)
			settings.ShowBounds = !settings.ShowBounds
		}
		return true
	}
	if idx := matchItem(hud.ContextMenu, entity); idx >= 0 {
		switch idx {
		case 0:
			target := hud.ContextTarget
			if target.ID != 0 {
				worlds.Engine.Despawn(target)
			}
			hud.ContextTarget = ecs.Entity{}
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

func (c *HudContext) refreshMenuPopups() {
	setMenuVisible(c.UI, c.Hud.FileMenu, c.Hud.OpenMenu == menuFileOpen)
	setMenuVisible(c.UI, c.Hud.EditMenu, c.Hud.OpenMenu == menuEditOpen)
	setMenuVisible(c.UI, c.Hud.ViewMenu, c.Hud.OpenMenu == menuViewOpen)
	setMenuVisible(c.UI, c.Hud.ContextMenu, c.Hud.OpenMenu == menuContextOpen)
}

func (c *HudContext) refreshInteractiveHovers() {
	idle := [4]float32{0.14, 0.16, 0.20, 1}
	hover := [4]float32{0.32, 0.62, 0.98, 1}
	for _, m := range []menuPopup{c.Hud.FileMenu, c.Hud.EditMenu, c.Hud.ViewMenu, c.Hud.ContextMenu} {
		for i := 0; i < m.Count; i++ {
			paintHover(c.UI, m.Items[i], idle, hover)
		}
	}
	for _, b := range []ecs.Entity{c.Hud.FileButton, c.Hud.EditButton, c.Hud.ViewButton} {
		paintHover(c.UI, b, idle, hover)
	}

	selected, hasSelected := pass.SelectedTarget(c.Engine)
	for i, row := range c.Hud.TreeRows {
		if row.ID == 0 {
			continue
		}
		target := c.Hud.TreeRowToEngine[i]
		if hasSelected && target == selected {
			continue
		}
		interactive, ok := ecs.Get[ui.Interactive](c.UI, row)
		if !ok {
			continue
		}
		color, ok := ecs.GetMut[ui.Color](c.UI, row)
		if !ok {
			continue
		}
		if interactive.Hovered && color.RGBA[3] > 0.001 {
			color.RGBA = [4]float32{0.22, 0.32, 0.55, 1}
		}
	}
}

func paintHover(world *ecs.World, entity ecs.Entity, idle, hover [4]float32) {
	color, ok := ecs.GetMut[ui.Color](world, entity)
	if !ok || color.RGBA[3] <= 0.001 {
		return
	}
	interactive, ok := ecs.Get[ui.Interactive](world, entity)
	if !ok {
		return
	}
	if interactive.Hovered {
		color.RGBA = hover
	} else {
		color.RGBA = idle
	}
}

func setMenuVisible(world *ecs.World, menu menuPopup, visible bool) {
	var panelAlpha float32
	var textAlpha float32
	if visible {
		panelAlpha = 1
		textAlpha = 1
	}
	if color, ok := ecs.GetMut[ui.Color](world, menu.Panel); ok {
		color.RGBA = [4]float32{0.10, 0.11, 0.14, panelAlpha}
	}
	for i := 0; i < menu.Count; i++ {
		if color, ok := ecs.GetMut[ui.Color](world, menu.Items[i]); ok {
			if visible {
				color.RGBA = [4]float32{0.14, 0.16, 0.20, 1}
			} else {
				color.RGBA = [4]float32{0, 0, 0, 0}
			}
		}
		if text, ok := ecs.GetMut[ui.Text](world, menu.Items[i]); ok {
			text.Color[3] = textAlpha
		}
	}
}

func (c *HudContext) refreshHudLayout() {
	w := ecs.MustResource[window.Window](c.UI).Viewport
	width := float32(w.Width)
	height := float32(w.Height)
	changed := false
	if node, ok := ecs.GetMut[ui.Node](c.UI, c.Hud.TopBar); ok && node.Width != width {
		node.Width = width
		changed = true
	}
	panelHeight := height - hudTopBarHeight
	if node, ok := ecs.GetMut[ui.Node](c.UI, c.Hud.LeftPanel); ok && node.Height != panelHeight {
		node.Height = panelHeight
		changed = true
	}
	if node, ok := ecs.GetMut[ui.Node](c.UI, c.Hud.RightPanel); ok && node.Height != panelHeight {
		node.Height = panelHeight
		changed = true
	}
	if changed {
		ui.MarkLayoutDirty(c.UI)
	}
}

func (c *HudContext) refreshModeButtons() {
	setModeColor(c.UI, c.Hud.TranslateButton, c.Gizmo.Mode == render.GizmoTranslate)
	setModeColor(c.UI, c.Hud.RotateButton, c.Gizmo.Mode == render.GizmoRotate)
	setModeColor(c.UI, c.Hud.ScaleButton, c.Gizmo.Mode == render.GizmoScale)
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

type namedEntity struct {
	Entity ecs.Entity
	Name   string
}

func (c *HudContext) refreshEntityTree() {
	selected, hasSelected := pass.SelectedTarget(c.Engine)

	nameMask := ecs.MustMaskOf[app.Name](c.Engine)
	var entries []namedEntity
	c.Engine.ForEach(nameMask, 0, func(entity ecs.Entity, table *ecs.Archetype, columnIndex int) {
		names, _ := ecs.Column[app.Name](c.Engine, table)
		entries = append(entries, namedEntity{Entity: entity, Name: names[columnIndex].Value})
	})
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].Entity.ID < entries[j].Entity.ID
	})

	const rowStride float32 = hudTreeRowHeight + 4
	maxScroll := len(entries) - hudTreeRowCount
	if maxScroll < 0 {
		maxScroll = 0
	}
	maxScrollPixels := float32(maxScroll) * rowStride
	if c.Hud.TreeScrollPixels > maxScrollPixels {
		c.Hud.TreeScrollPixels = maxScrollPixels
	}
	if c.Hud.TreeScrollPixels < 0 {
		c.Hud.TreeScrollPixels = 0
	}
	c.Hud.TreeScrollIndex = int(c.Hud.TreeScrollPixels / rowStride)
	if c.Hud.TreeScrollIndex > maxScroll {
		c.Hud.TreeScrollIndex = maxScroll
	}
	offset := c.Hud.TreeScrollIndex

	for i := 0; i < hudTreeRowCount; i++ {
		row := c.Hud.TreeRows[i]
		idx := i + offset
		if idx < len(entries) {
			c.Hud.TreeRowToEngine[i] = entries[idx].Entity
			c.setText(row, entries[idx].Name)
			if hasSelected && entries[idx].Entity == selected {
				c.setColor(row, [4]float32{0.22, 0.5, 0.86, 1})
			} else {
				c.setColor(row, [4]float32{0.10, 0.11, 0.14, 1})
			}
		} else {
			c.Hud.TreeRowToEngine[i] = ecs.Entity{}
			c.setText(row, "")
			c.setColor(row, [4]float32{0, 0, 0, 0})
		}
	}
}

func (c *HudContext) updateTreeScroll() {
	if c.Input.Wheel != 0 {
		leftNode, ok := ecs.Get[ui.Node](c.UI, c.Hud.LeftPanel)
		if ok && leftNode.Resolved.Contains(c.Input.MousePosition[0], c.Input.MousePosition[1]) {
			c.Hud.TreeScrollPixels -= c.Input.Wheel * 40
			c.Input.Wheel = 0
		}
	}
	const rowStride float32 = hudTreeRowHeight + 4
	if c.Hud.TreeScrollPixels < 0 {
		c.Hud.TreeScrollPixels = 0
	}
	c.Hud.TreeScrollIndex = int(c.Hud.TreeScrollPixels / rowStride)
}

func (c *HudContext) refreshInspector() {
	target, hasTarget := pass.SelectedTarget(c.Engine)
	editing := c.Pointer.FocusedEntity == c.Hud.InspectorName

	if !hasTarget {
		c.syncInspectorName("No entity selected", false)
		c.setText(c.Hud.InspectorID, "")
		c.setText(c.Hud.TranslationLabel, "")
		c.setText(c.Hud.RotationLabel, "")
		c.setText(c.Hud.ScaleLabel, "")
		c.setText(c.Hud.MaterialLabel, "")
		return
	}

	name := "(unnamed)"
	if n, ok := ecs.Get[app.Name](c.Engine, target); ok {
		name = n.Value
	}
	c.syncInspectorName(name, editing)
	c.setText(c.Hud.InspectorID, fmt.Sprintf("ID   %d", target.ID))

	if local, ok := ecs.Get[transform.LocalTransform](c.Engine, target); ok {
		c.setText(c.Hud.TranslationLabel, fmt.Sprintf("POS  %s", formatVec3(local.Translation)))
		c.setText(c.Hud.RotationLabel, fmt.Sprintf("ROT  %s", formatQuat(local.Rotation)))
		c.setText(c.Hud.ScaleLabel, fmt.Sprintf("SCL  %s", formatVec3(local.Scale)))
	} else {
		c.setText(c.Hud.TranslationLabel, "")
		c.setText(c.Hud.RotationLabel, "")
		c.setText(c.Hud.ScaleLabel, "")
	}

	if mat, ok := ecs.Get[asset.Material](c.Engine, target); ok {
		c.setText(c.Hud.MaterialLabel, fmt.Sprintf("MAT  %s", formatRGBA(mat.BaseColor)))
	} else {
		c.setText(c.Hud.MaterialLabel, "")
	}
}

func (c *HudContext) syncInspectorName(name string, editing bool) {
	entity := c.Hud.InspectorName
	if !editing {
		if ti, ok := ecs.GetMut[ui.TextInput](c.UI, entity); ok {
			ti.Buffer = name
			if ti.Caret > len(name) {
				ti.Caret = len(name)
			}
		}
		c.setText(entity, name)
	}
	if editing {
		c.setColor(entity, [4]float32{0.18, 0.30, 0.55, 1})
	} else {
		c.setColor(entity, [4]float32{0.12, 0.13, 0.16, 1})
	}
}

func (c *HudContext) updateInspectorCaret() {
	editing := c.Pointer.FocusedEntity == c.Hud.InspectorName
	caretColor, ok := ecs.GetMut[ui.Color](c.UI, c.Hud.InspectorCaret)
	if !ok {
		return
	}
	if !editing {
		caretColor.RGBA[3] = 0
		return
	}
	field, ok := ecs.Get[ui.Node](c.UI, c.Hud.InspectorName)
	if !ok {
		return
	}
	ti, ok := ecs.Get[ui.TextInput](c.UI, c.Hud.InspectorName)
	if !ok {
		return
	}
	label, ok := ecs.Get[ui.Text](c.UI, c.Hud.InspectorName)
	if !ok {
		return
	}
	scale := label.Scale
	if scale <= 0 {
		scale = 1
	}
	advance := float32(ui.FontGlyphWidth+1) * scale
	glyphH := float32(ui.FontGlyphHeight) * scale
	bufferRunes := []rune(ti.Buffer)
	caret := ti.Caret
	if caret > len(bufferRunes) {
		caret = len(bufferRunes)
	}
	labelWidth := float32(len(bufferRunes)) * advance
	originX := field.Resolved.X + (field.Resolved.Width-labelWidth)*0.5
	caretX := originX + float32(caret)*advance
	caretY := field.Resolved.Y + (field.Resolved.Height-glyphH)*0.5

	caretNode, ok := ecs.GetMut[ui.Node](c.UI, c.Hud.InspectorCaret)
	if !ok {
		return
	}
	caretNode.X = caretX
	caretNode.Y = caretY
	caretNode.Height = glyphH
	caretNode.Resolved = ui.Rect{X: caretX, Y: caretY, Width: caretNode.Width, Height: glyphH}
	uptime := ecs.MustResource[window.Window](c.Engine).Timing.UptimeSeconds
	alpha := float32(0)
	if caretVisible(uptime) {
		alpha = 1
	}
	caretColor.RGBA = [4]float32{0.95, 0.96, 0.98, alpha}
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
