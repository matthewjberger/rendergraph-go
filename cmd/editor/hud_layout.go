// HUD layout types, constants, and builder helpers. Everything in
// this file describes the *shape* of the editor's retained-mode UI
// (sizes, panel layout, menu structure). Per-frame refresh +
// input handling live in sibling files.
package main

import (
	"indigo/ecs"
	"indigo/ui"
)

const (
	hudTopBarHeight    float32 = 36
	hudLeftPanelWidth  float32 = 240
	hudRightPanelWidth float32 = 320
	hudTreeRowCount    int     = 28
	hudTreeRowHeight   float32 = 22
	hudMenuPopupWidth  float32 = 150
	hudMenuItemHeight  float32 = 22

	menuClosed      = 0
	menuFileOpen    = 1
	menuEditOpen    = 2
	menuViewOpen    = 3
	menuContextOpen = 4
)

// menuPopupZ is the render z-index for popup panels and items.
// Anything below this stays underneath; tree rows, inspector text,
// and other HUD content use ZIndex 0. Items render on top of the
// panel via the +1 offset.
const menuPopupZ int32 = 100

// menuPopup is the visible content of one drop-down menu: a panel
// plus its item rows. Per-menu helpers populate the items with
// per-frame content; the platform layer toggles visibility by
// writing Color.RGBA[3] across the panel and items based on which
// menu is open.
type menuPopup struct {
	Panel ecs.Entity
	Items [4]ecs.Entity
	Count int
}

type HudHandles struct {
	TopBar          ecs.Entity
	FileButton      ecs.Entity
	EditButton      ecs.Entity
	ViewButton      ecs.Entity
	TranslateButton ecs.Entity
	RotateButton    ecs.Entity
	ScaleButton     ecs.Entity

	FileMenu menuPopup
	EditMenu menuPopup
	ViewMenu menuPopup

	// ContextMenu is the right-click popup for tree rows. The
	// platform layer repositions its panel to the mouse on
	// right-press and stores the target entity in ContextTarget
	// so the item action knows which entity to operate on.
	ContextMenu   menuPopup
	ContextTarget ecs.Entity

	// Which top-bar menu (if any) is currently open. Only one
	// popup is shown at a time; flipping a new one closes the
	// previous.
	OpenMenu int

	LeftPanel ecs.Entity
	TreeTitle ecs.Entity
	TreeRows  [hudTreeRowCount]ecs.Entity

	RightPanel       ecs.Entity
	InspectorTitle   ecs.Entity
	InspectorName    ecs.Entity
	InspectorCaret   ecs.Entity
	InspectorID      ecs.Entity
	TranslationLabel ecs.Entity
	RotationLabel    ecs.Entity
	ScaleLabel       ecs.Entity
	MaterialLabel    ecs.Entity

	TreeRowToEngine [hudTreeRowCount]ecs.Entity

	RequestExit     bool
	NameFocusedPrev bool

	TreeScrollPixels float32
	TreeScrollIndex  int
}

func buildHud(world *ecs.World) HudHandles {
	b := ui.NewBuilder(world)
	var h HudHandles

	h.TopBar = b.Node(ui.Node{
		X: 0, Y: 0,
		Width: 1280, Height: hudTopBarHeight,
		Anchor: ui.AnchorTopLeft,
	}).Color(ui.Color{RGBA: [4]float32{0.08, 0.09, 0.12, 1}}).Entity()

	menuBar := b.Node(ui.Node{
		X: 0, Y: 0,
		Width: 280, Height: hudTopBarHeight,
		Anchor:  ui.AnchorTopLeft,
		Layout:  ui.LayoutRow,
		Padding: 6, Spacing: 4,
	}).Color(ui.Color{RGBA: [4]float32{0, 0, 0, 0}}).Entity()
	b.Push(menuBar)
	h.FileButton = buildMenuButton(b, "FILE")
	h.EditButton = buildMenuButton(b, "EDIT")
	h.ViewButton = buildMenuButton(b, "VIEW")
	b.Pop()

	modeBar := b.Node(ui.Node{
		X: 0, Y: 0,
		Width: 360, Height: hudTopBarHeight,
		Anchor:  ui.AnchorTopRight,
		Layout:  ui.LayoutRow,
		Padding: 6, Spacing: 6,
	}).Color(ui.Color{RGBA: [4]float32{0, 0, 0, 0}}).Entity()
	b.Push(modeBar)
	h.TranslateButton = buildModeButton(b, "TRANSLATE")
	h.RotateButton = buildModeButton(b, "ROTATE")
	h.ScaleButton = buildModeButton(b, "SCALE")
	b.Pop()

	h.LeftPanel = b.Node(ui.Node{
		X: 0, Y: hudTopBarHeight,
		Width: hudLeftPanelWidth, Height: 684,
		Anchor:  ui.AnchorTopLeft,
		Layout:  ui.LayoutColumn,
		Padding: 10, Spacing: 4,
	}).Color(ui.Color{RGBA: [4]float32{0.06, 0.07, 0.09, 1}}).Entity()
	b.Push(h.LeftPanel)
	h.TreeTitle = b.Node(ui.Node{Width: hudLeftPanelWidth - 20, Height: 22}).Text(ui.Text{
		Content: "HIERARCHY",
		Color:   [4]float32{0.72, 0.78, 0.88, 1},
		Scale:   1.6,
	}).Entity()
	for i := 0; i < hudTreeRowCount; i++ {
		row := b.Node(ui.Node{
			Width: hudLeftPanelWidth - 20, Height: hudTreeRowHeight,
		}).
			Color(ui.Color{RGBA: [4]float32{0, 0, 0, 0}}).
			Interactive().
			Text(ui.Text{
				Content: "",
				Color:   [4]float32{0.9, 0.92, 0.96, 1},
				Scale:   1.4,
			}).Entity()
		h.TreeRows[i] = row
	}
	b.Pop()

	h.RightPanel = b.Node(ui.Node{
		X: 0, Y: hudTopBarHeight,
		Width: hudRightPanelWidth, Height: 684,
		Anchor:  ui.AnchorTopRight,
		Layout:  ui.LayoutColumn,
		Padding: 12, Spacing: 6,
	}).Color(ui.Color{RGBA: [4]float32{0.06, 0.07, 0.09, 1}}).Entity()
	b.Push(h.RightPanel)
	h.InspectorTitle = b.Node(ui.Node{Width: hudRightPanelWidth - 24, Height: 22}).Text(ui.Text{
		Content: "INSPECTOR",
		Color:   [4]float32{0.72, 0.78, 0.88, 1},
		Scale:   1.6,
	}).Entity()
	h.InspectorName = b.Node(ui.Node{
		Width: hudRightPanelWidth - 24, Height: 22,
	}).
		Color(ui.Color{RGBA: [4]float32{0.12, 0.13, 0.16, 1}}).
		Interactive().
		Text(ui.Text{
			Content: "",
			Color:   [4]float32{0.9, 0.92, 0.96, 1},
			Scale:   1.4,
		}).Entity()
	// TextInput component lives on the same entity so editing the
	// label rewrites Text.Content and a TextCommitted event flushes
	// the buffer back to the selected entity's [app.Name].
	ecs.Set(world, h.InspectorName, ui.TextInput{})

	// Caret: a thin colored rectangle the platform layer
	// repositions every frame to the current insertion point in
	// the inspector name buffer. Hidden (Color alpha 0) when the
	// name field isn't focused.
	h.InspectorCaret = b.Node(ui.Node{
		X: 0, Y: 0, Width: 2, Height: 14,
		Anchor: ui.AnchorTopLeft,
		ZIndex: 10,
	}).Color(ui.Color{RGBA: [4]float32{0, 0, 0, 0}}).Entity()
	h.InspectorID = inspectorLabel(b, "")
	h.TranslationLabel = inspectorLabel(b, "")
	h.RotationLabel = inspectorLabel(b, "")
	h.ScaleLabel = inspectorLabel(b, "")
	h.MaterialLabel = inspectorLabel(b, "")
	b.Pop()

	// Menu popups built LAST so their entities sit at the end of
	// their archetypes' per-frame iteration order. The UI quad +
	// text passes have no explicit z-ordering, so a later instance
	// blends on top of earlier ones; without this, the tree rows and
	// inspector labels (created earlier in their archetypes) would
	// paint over the open popup's items.
	const buttonOffset float32 = 6
	const buttonStride float32 = 64 + 4
	h.FileMenu = buildMenuPopup(b, buttonOffset+0*buttonStride, hudTopBarHeight,
		[]string{"NEW SCENE", "OPEN", "SAVE", "QUIT"})
	h.EditMenu = buildMenuPopup(b, buttonOffset+1*buttonStride, hudTopBarHeight,
		[]string{"UNDO", "REDO", "DESELECT"})
	h.ViewMenu = buildMenuPopup(b, buttonOffset+2*buttonStride, hudTopBarHeight,
		[]string{"RESET CAMERA", "TOGGLE GRID", "TOGGLE SKY", "TOGGLE BOUNDS", "TOGGLE NORMALS"})
	h.ContextMenu = buildMenuPopup(b, 0, 0,
		[]string{"DELETE"})

	return h
}

func buildMenuPopup(b *ui.Builder, anchorX, anchorY float32, items []string) menuPopup {
	height := float32(8) + float32(len(items))*hudMenuItemHeight + float32(len(items)-1)*2
	// Interactive on the panel absorbs clicks on the panel
	// background so they don't fall through to the close-on-outside-
	// click logic. ZIndex makes both the panel quad and its item
	// text render strictly on top of the HUD beneath them, replacing
	// the prior archetype-creation-order hack.
	panel := b.Node(ui.Node{
		X: anchorX, Y: anchorY,
		Width: hudMenuPopupWidth, Height: height,
		Anchor:  ui.AnchorTopLeft,
		Layout:  ui.LayoutColumn,
		Padding: 4, Spacing: 2,
		ZIndex: menuPopupZ,
	}).
		Color(ui.Color{RGBA: [4]float32{0, 0, 0, 0}}).
		Interactive().
		Entity()

	var popup menuPopup
	popup.Panel = panel
	popup.Count = len(items)
	b.Push(panel)
	for i, label := range items {
		popup.Items[i] = b.Node(ui.Node{
			Width: hudMenuPopupWidth - 8, Height: hudMenuItemHeight,
			ZIndex: menuPopupZ + 1,
		}).
			Color(ui.Color{RGBA: [4]float32{0, 0, 0, 0}}).
			Interactive().
			Text(ui.Text{
				Content: label,
				Color:   [4]float32{0.92, 0.94, 0.98, 0},
				Scale:   1.4,
			}).Entity()
	}
	b.Pop()
	return popup
}

func buildMenuButton(b *ui.Builder, text string) ecs.Entity {
	return b.Node(ui.Node{Width: 64, Height: 24}).
		Color(ui.Color{RGBA: [4]float32{0.14, 0.16, 0.20, 1}}).
		Interactive().
		Text(ui.Text{
			Content: text,
			Color:   [4]float32{0.85, 0.88, 0.94, 1},
			Scale:   1.4,
		}).Entity()
}

func buildModeButton(b *ui.Builder, text string) ecs.Entity {
	return b.Node(ui.Node{Width: 104, Height: 24}).
		Color(ui.Color{RGBA: [4]float32{0.18, 0.21, 0.28, 1}}).
		Interactive().
		Text(ui.Text{
			Content: text,
			Color:   [4]float32{0.95, 0.96, 0.98, 1},
			Scale:   1.4,
		}).Entity()
}

func inspectorLabel(b *ui.Builder, initial string) ecs.Entity {
	return b.Node(ui.Node{Width: hudRightPanelWidth - 24, Height: 20}).Text(ui.Text{
		Content: initial,
		Color:   [4]float32{0.9, 0.92, 0.96, 1},
		Scale:   1.4,
	}).Entity()
}
