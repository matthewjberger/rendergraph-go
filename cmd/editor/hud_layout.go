package main

import (
	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/ui"
)

const (
	hudTopBarHeight    float32 = 36
	hudLeftPanelWidth  float32 = 240
	hudRightPanelWidth float32 = 320
	hudTreeRowCount    int     = 28
	hudTreeRowHeight   float32 = 22
	hudMenuPopupWidth  float32 = 150
	hudMenuItemHeight  float32 = 22

	hudLoadingBarWidth  float32 = 320
	hudLoadingBarHeight float32 = 24

	menuClosed      = 0
	menuFileOpen    = 1
	menuEditOpen    = 2
	menuViewOpen    = 3
	menuContextOpen = 4
	menuAssetsOpen  = 5
)

const menuPopupZ int32 = 100

type menuPopup struct {
	Panel ecs.Entity
	Items [8]ecs.Entity
	Count int
}

type HudHandles struct {
	TopBar          ecs.Entity
	FpsLabel        ecs.Entity
	FileButton      ecs.Entity
	EditButton      ecs.Entity
	ViewButton      ecs.Entity
	AssetsButton    ecs.Entity
	ViewerButton    ecs.Entity
	ViewerCheck     ecs.Entity
	ViewerLabel     ecs.Entity
	TranslateButton ecs.Entity
	RotateButton    ecs.Entity
	ScaleButton     ecs.Entity

	FileMenu   menuPopup
	EditMenu   menuPopup
	ViewMenu   menuPopup
	AssetsMenu menuPopup

	Khronos     khronosHandles
	KhronosOpen bool

	ContextMenu   menuPopup
	ContextTarget ecs.Entity

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
	TreeRowCount    int

	RequestExit     bool
	NameFocusedPrev bool

	FramePending bool
	ModelRoots   []ecs.Entity

	LoadingBar   ecs.Entity
	LoadingFill  ecs.Entity
	LoadingLabel ecs.Entity

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

	h.FpsLabel = b.Node(ui.Node{
		X: 0, Y: 0,
		Width: 120, Height: hudTopBarHeight,
		Anchor: ui.AnchorTopCenter,
		ZIndex: 1,
	}).Text(ui.Text{
		Content: "",
		Color:   [4]float32{0.6, 0.66, 0.78, 1},
		Scale:   1.4,
	}).Entity()

	menuBar := b.Node(ui.Node{
		X: 0, Y: 0,
		Width: 520, Height: hudTopBarHeight,
		Anchor:  ui.AnchorTopLeft,
		Layout:  ui.LayoutRow,
		Padding: 6, Spacing: 4,
	}).Color(ui.Color{RGBA: [4]float32{0, 0, 0, 0}}).Entity()
	b.Push(menuBar)
	h.FileButton = buildMenuButton(b, "FILE")
	h.EditButton = buildMenuButton(b, "EDIT")
	h.ViewButton = buildMenuButton(b, "VIEW")
	h.AssetsButton = buildMenuButton(b, "ASSETS")
	h.ViewerButton, h.ViewerCheck = buildCheckbox(b)
	h.ViewerLabel = b.Node(ui.Node{Width: 70, Height: 24}).
		Color(ui.Color{RGBA: [4]float32{0, 0, 0, 0}}).
		Text(ui.Text{
			Content: "VIEWER",
			Color:   [4]float32{0.85, 0.88, 0.94, 1},
			Scale:   1.4,
		}).Entity()
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

	ecs.Set(world, h.InspectorName, ui.TextInput{})

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

	const buttonOffset float32 = 6
	const buttonStride float32 = 64 + 4
	h.FileMenu = buildMenuPopup(b, buttonOffset+0*buttonStride, hudTopBarHeight,
		[]string{"NEW SCENE", "OPEN", "SAVE", "QUIT"})
	h.EditMenu = buildMenuPopup(b, buttonOffset+1*buttonStride, hudTopBarHeight,
		[]string{"UNDO", "REDO", "DESELECT"})
	h.ViewMenu = buildMenuPopup(b, buttonOffset+2*buttonStride, hudTopBarHeight,
		[]string{"RESET CAMERA", "TOGGLE GRID", "TOGGLE SKY", "TOGGLE BOUNDS", "TOGGLE NORMALS", "TOGGLE SKELETONS"})
	h.AssetsMenu = buildMenuPopup(b, buttonOffset+3*buttonStride, hudTopBarHeight,
		[]string{"KHRONOS SAMPLES"})
	h.ContextMenu = buildMenuPopup(b, 0, 0,
		[]string{"DELETE"})

	h.Khronos = buildKhronosPanel(b)

	h.LoadingBar = b.Node(ui.Node{
		X: 0, Y: 40,
		Width: hudLoadingBarWidth, Height: hudLoadingBarHeight,
		Anchor: ui.AnchorBottomCenter,
		ZIndex: 200,
		Hidden: true,
	}).Color(ui.Color{RGBA: [4]float32{0.08, 0.09, 0.12, 0.92}}).Entity()
	b.Push(h.LoadingBar)
	h.LoadingFill = b.Node(ui.Node{
		X: 0, Y: 0,
		Width: hudLoadingBarWidth, Height: hudLoadingBarHeight,
		ZIndex: 201,
	}).Color(ui.Color{RGBA: [4]float32{0.32, 0.62, 0.98, 1}}).Entity()
	h.LoadingLabel = b.Node(ui.Node{
		X: 8, Y: 5,
		Width: hudLoadingBarWidth - 16, Height: hudLoadingBarHeight,
		ZIndex: 202,
	}).Color(ui.Color{RGBA: [4]float32{0, 0, 0, 0}}).Text(ui.Text{
		Content: "",
		Color:   [4]float32{0.95, 0.97, 1, 1},
		Scale:   1.4,
	}).Entity()
	b.Pop()

	return h
}

func buildMenuPopup(b *ui.Builder, anchorX, anchorY float32, items []string) menuPopup {
	height := float32(8) + float32(len(items))*hudMenuItemHeight + float32(len(items)-1)*2

	panel := b.Node(ui.Node{
		X: anchorX, Y: anchorY,
		Width: hudMenuPopupWidth, Height: height,
		Anchor:  ui.AnchorTopLeft,
		Layout:  ui.LayoutColumn,
		Padding: 4, Spacing: 2,
		ZIndex: menuPopupZ,
		Hidden: true,
	}).
		Color(ui.Color{RGBA: [4]float32{0.10, 0.11, 0.14, 1}}).
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
			Color(ui.Color{RGBA: [4]float32{0.14, 0.16, 0.20, 1}}).
			Interactive().
			Text(ui.Text{
				Content: label,
				Color:   [4]float32{0.92, 0.94, 0.98, 1},
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

func buildCheckbox(b *ui.Builder) (ecs.Entity, ecs.Entity) {
	box := b.Node(ui.Node{Width: 20, Height: 20}).
		Color(ui.Color{RGBA: [4]float32{0.12, 0.13, 0.16, 1}}).
		Interactive().
		Entity()
	b.Push(box)
	inner := b.Node(ui.Node{
		X: 4, Y: 4,
		Width: 12, Height: 12,
		ZIndex: 1,
	}).Color(ui.Color{RGBA: [4]float32{0.32, 0.62, 0.98, 1}}).Entity()
	b.Pop()
	return box, inner
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
