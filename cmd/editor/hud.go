package main

import (
	"fmt"
	"sort"

	"github.com/matthewjberger/indigo/app"
	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/internal/render/pass"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/render/asset"
	"github.com/matthewjberger/indigo/transform"
	"github.com/matthewjberger/indigo/ui"
	"github.com/matthewjberger/indigo/window"
)

func (c *HudContext) refreshMenuPopups() {
	setMenuVisible(c.UI, c.Hud.FileMenu, c.Hud.OpenMenu == menuFileOpen)
	setMenuVisible(c.UI, c.Hud.EditMenu, c.Hud.OpenMenu == menuEditOpen)
	setMenuVisible(c.UI, c.Hud.ViewMenu, c.Hud.OpenMenu == menuViewOpen)
	setMenuVisible(c.UI, c.Hud.AssetsMenu, c.Hud.OpenMenu == menuAssetsOpen)
	setMenuVisible(c.UI, c.Hud.ContextMenu, c.Hud.OpenMenu == menuContextOpen)
}

func (c *HudContext) refreshInteractiveHovers() {
	idle := [4]float32{0.14, 0.16, 0.20, 1}
	hover := [4]float32{0.32, 0.62, 0.98, 1}
	for _, m := range []menuPopup{c.Hud.FileMenu, c.Hud.EditMenu, c.Hud.ViewMenu, c.Hud.AssetsMenu, c.Hud.ContextMenu} {
		for i := 0; i < m.Count; i++ {
			paintHover(c.UI, m.Items[i], idle, hover)
		}
	}
	for _, b := range []ecs.Entity{c.Hud.FileButton, c.Hud.EditButton, c.Hud.ViewButton, c.Hud.AssetsButton} {
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
	ui.SetVisible(world, menu.Panel, visible)
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

func (c *HudContext) refreshFps() {
	fps := ecs.MustResource[window.Window](c.Engine).Timing.FramesPerSec
	c.setText(c.Hud.FpsLabel, fmt.Sprintf("%.0f FPS", fps))
}

func (c *HudContext) refreshLoadingProgress() {
	queue := ecs.MustResource[asset.LoadingQueueResource](c.Engine).Queue
	active := queue.Active()
	ui.SetVisible(c.UI, c.Hud.LoadingBar, active)
	if !active {
		return
	}
	completed, total := queue.Progress()
	width := hudLoadingBarWidth * queue.Fraction()
	if node, ok := ecs.GetMut[ui.Node](c.UI, c.Hud.LoadingFill); ok && node.Width != width {
		node.Width = width
		ui.MarkLayoutDirty(c.UI)
	}
	label := queue.Label()
	if label == "" {
		label = "Loading textures"
	}
	c.setText(c.Hud.LoadingLabel, fmt.Sprintf("%s  %d / %d", label, completed, total))
}

func (c *HudContext) refreshModeButtons() {
	setModeColor(c.UI, c.Hud.TranslateButton, c.Gizmo.Mode == render.GizmoTranslate)
	setModeColor(c.UI, c.Hud.RotateButton, c.Gizmo.Mode == render.GizmoRotate)
	setModeColor(c.UI, c.Hud.ScaleButton, c.Gizmo.Mode == render.GizmoScale)
}

func (c *HudContext) refreshViewerCheckbox() {
	on := ecs.MustResource[render.Graphics](c.Engine).ViewerMode
	ui.SetVisible(c.UI, c.Hud.ViewerCheck, on)
	if interactive, ok := ecs.Get[ui.Interactive](c.UI, c.Hud.ViewerButton); ok && interactive.Hovered {
		c.setColor(c.Hud.ViewerButton, [4]float32{0.20, 0.22, 0.28, 1})
	} else {
		c.setColor(c.Hud.ViewerButton, [4]float32{0.12, 0.13, 0.16, 1})
	}
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

	c.Hud.TreeRowCount = 0
	for i := 0; i < hudTreeRowCount; i++ {
		row := c.Hud.TreeRows[i]
		idx := i + offset
		if idx < len(entries) {
			c.Hud.TreeRowToEngine[i] = entries[idx].Entity
			c.Hud.TreeRowCount++
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
	c.updateCaret(c.Hud.InspectorName, c.Hud.InspectorCaret)
}

func (c *HudContext) updateCaret(fieldEntity, caretEntity ecs.Entity) {
	caretColor, ok := ecs.GetMut[ui.Color](c.UI, caretEntity)
	if !ok {
		return
	}
	if c.Pointer.FocusedEntity != fieldEntity {
		caretColor.RGBA[3] = 0
		return
	}
	field, ok := ecs.Get[ui.Node](c.UI, fieldEntity)
	if !ok {
		return
	}
	ti, ok := ecs.Get[ui.TextInput](c.UI, fieldEntity)
	if !ok {
		return
	}
	label, ok := ecs.Get[ui.Text](c.UI, fieldEntity)
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

	caretNode, ok := ecs.GetMut[ui.Node](c.UI, caretEntity)
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
