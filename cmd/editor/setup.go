// Shared bootstrap used by both the desktop (!js) and wasm (js)
// entry points. Builds an engine world via [app.NewEngineWorld] plus
// a game world that holds Spinner state, wires schedules + the
// render graph, then spawns the demo scene.
package main

import (
	"bytes"
	_ "embed"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/cogentcore/webgpu/wgpu"

	"indigo/app"
	"indigo/ecs"
	"indigo/render"
	"indigo/render/asset"
	"indigo/render/pass"
	"indigo/transform"
	"indigo/ui"
	"indigo/window"
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

	// Which top-bar menu (if any) is currently open. Matches
	// nightshade's active_context_menu resource: only one popup at a
	// time, flipping a new one closes the previous.
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

func setupLogging() {
	switch os.Getenv("WGPU_LOG_LEVEL") {
	case "OFF":
		wgpu.SetLogLevel(wgpu.LogLevelOff)
	case "ERROR":
		wgpu.SetLogLevel(wgpu.LogLevelError)
	case "WARN":
		wgpu.SetLogLevel(wgpu.LogLevelWarn)
	case "INFO":
		wgpu.SetLogLevel(wgpu.LogLevelInfo)
	case "DEBUG":
		wgpu.SetLogLevel(wgpu.LogLevelDebug)
	case "TRACE":
		wgpu.SetLogLevel(wgpu.LogLevelTrace)
	}
}

func buildWorlds(renderer *render.Renderer) (app.Worlds, *app.App) {
	worlds, err := app.NewWorlds(renderer)
	if err != nil {
		log.Fatal(err)
	}
	engine := worlds.Engine

	ecs.SetResource(engine, render.DefaultCamera())
	ecs.SetResource(engine, render.DefaultPanOrbitController())
	ecs.SetResource(engine, pass.NewPicking())
	ecs.SetResource(engine, render.NewGizmos())

	ecs.Register[Spinner](worlds.Game)
	ecs.SetResource(engine, buildHud(worlds.UI))

	worlds.EngineSchedule.Push("graphics_toggles", render.UpdateGraphicsToggles)
	worlds.EngineSchedule.Push("gizmos", pass.UpdateGizmos)
	worlds.EngineSchedule.Push("pan_orbit_camera", render.UpdatePanOrbitCamera)
	worlds.EngineSchedule.Push("animations", asset.UpdateAnimationPlayers)
	worlds.EngineSchedule.Push("transform_propagation", transform.UpdateGlobalTransforms)
	worlds.EngineSchedule.Push("bounding_volume_lines", pass.UpdateBoundingVolumeLines)
	_ = advanceSpinners

	demo := editorApp()
	if demo.Initialize != nil {
		demo.Initialize(engine)
	}
	if demo.ConfigureRenderGraph != nil {
		demo.ConfigureRenderGraph(engine, renderer)
	}
	primitives := ecs.MustResource[asset.Primitives](engine)
	initializeWorldEntities(worlds,
		[]asset.MeshHandle{primitives.UnitTriangle, primitives.UnitQuad, primitives.UnitCube},
		[]string{"Triangle", "Quad", "Cube"},
	)
	loadDefaultGltf(engine, renderer)

	if err := renderer.Graph.Compile(renderer.Device); err != nil {
		log.Fatal(err)
	}
	return worlds, demo
}

// defaultGltf is the .glb the editor auto-loads at startup so the
// view isn't just primitives. Embedded so both the native build
// and the wasm build see it without depending on a filesystem
// path.
//
//go:embed assets/DamagedHelmet.glb
var defaultGltf []byte

// loadDefaultGltf spawns the embedded DamagedHelmet scene at
// startup. Logs and continues on error rather than aborting boot
// so a broken asset doesn't prevent the editor from launching.
func loadDefaultGltf(engine *ecs.World, renderer *render.Renderer) {
	if _, err := loadGltfBytes(engine, renderer, "DamagedHelmet.glb", defaultGltf); err != nil {
		log.Printf("gltf load failed: %v", err)
	}
}

// loadGltfBytes parses a glTF / glb buffer and spawns its scene
// through the shared [spawnLoadedSceneNamed] helper. Used by both
// the editor's startup auto-load and the wasm drop callback.
func loadGltfBytes(engine *ecs.World, renderer *render.Renderer, label string, data []byte) ([]ecs.Entity, error) {
	assets := ecs.MustResource[asset.MeshAssetsResource](engine).Assets
	arrays := ecs.MustResource[asset.MaterialTextureArraysResource](engine).Arrays
	scene, err := asset.LoadGltfReader(renderer.Device, renderer.Queue, assets, arrays, label, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	return spawnLoadedSceneNamed(engine, scene, label)
}

// loadGltfInto loads path via [asset.LoadGltfFile] and forwards to
// the bytes-based spawn helper. Native-only — wasm calls
// [loadGltfBytes] directly with bytes fetched from a drop event.
func loadGltfInto(engine *ecs.World, renderer *render.Renderer, path string) ([]ecs.Entity, error) {
	assets := ecs.MustResource[asset.MeshAssetsResource](engine).Assets
	arrays := ecs.MustResource[asset.MaterialTextureArraysResource](engine).Arrays
	scene, err := asset.LoadGltfFile(renderer.Device, renderer.Queue, assets, arrays, path)
	if err != nil {
		return nil, err
	}
	return spawnLoadedSceneNamed(engine, scene, filepath.Base(path))
}

func spawnLoadedSceneNamed(engine *ecs.World, scene *asset.LoadedScene, label string) ([]ecs.Entity, error) {
	entities := asset.SpawnLoadedScene(engine, scene)
	baseName := strings.TrimSuffix(label, filepath.Ext(label))
	nameMask := ecs.MustMaskOf[app.Name](engine)
	for i, e := range entities {
		name := scene.Nodes[i].Name
		if name == "" {
			name = fmt.Sprintf("%s/node_%d", baseName, i)
		}
		engine.AddComponents(e, nameMask)
		ecs.Set(engine, e, app.Name{Value: name})
	}
	log.Printf("gltf loaded: %s (%d nodes, %d meshes, %d materials, %d animations)",
		label, len(scene.Nodes), len(scene.Meshes), len(scene.Materials), len(scene.Animations))
	if len(scene.Roots) > 0 {
		applyEntitySelection(engine, entities[scene.Roots[0]])
	}
	return entities, nil
}

// editorApp wires the editor's render pipeline: sky -> mesh -> grid
// -> fxaa -> present. Each [render.Add*Pass] helper builds the
// underlying [render.Pass] and registers it with the graph.
func editorApp() *app.App {
	return &app.App{
		ConfigureRenderGraph: func(world *ecs.World, renderer *render.Renderer) {
			if _, err := pass.AddSkyPass(renderer); err != nil {
				log.Fatal(err)
			}
			arrays := ecs.MustResource[asset.MaterialTextureArraysResource](world).Arrays
			if _, err := pass.AddMeshPass(renderer, arrays); err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddPickingPass(renderer); err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddSelectionMaskPass(renderer); err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddGridPass(renderer); err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddLinesPass(renderer); err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddOutlinePass(renderer); err != nil {
				log.Fatal(err)
			}
			_, fxaaOutputID, err := pass.AddFxaaPass(renderer)
			if err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddGizmoPass(renderer, fxaaOutputID); err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddUiQuadPass(renderer, fxaaOutputID); err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddUiTextPass(renderer, fxaaOutputID); err != nil {
				log.Fatal(err)
			}
			if _, err := pass.AddPresentPass(renderer, fxaaOutputID); err != nil {
				log.Fatal(err)
			}
		},
	}
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
		[]string{"RESET CAMERA", "TOGGLE GRID", "TOGGLE SKY", "TOGGLE BOUNDS"})
	h.ContextMenu = buildMenuPopup(b, 0, 0,
		[]string{"DELETE"})

	return h
}

// menuPopupZ is the render z-index for popup panels and items.
// Anything below this stays underneath; tree rows, inspector text,
// and other HUD content use ZIndex 0. Items render on top of the
// panel via the +1 offset.
const menuPopupZ int32 = 100

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

// initializeWorldEntities spawns a 5x5 grid of engine entities
// cycling through the supplied mesh handles for visual variety, plus
// a matching game entity per engine entity carrying its Spinner state
// and EngineEntity bridge. Also spawns a sun-like directional light
// pointing straight down.
func initializeWorldEntities(worlds app.Worlds, meshes []asset.MeshHandle, meshNames []string) {
	const (
		gridExtent  = 3
		gridSpacing = 1.5
	)

	lightMask := ecs.MustMaskOf[transform.LocalTransform](worlds.Engine) |
		ecs.MustMaskOf[transform.GlobalTransform](worlds.Engine) |
		ecs.MustMaskOf[transform.LocalTransformDirty](worlds.Engine) |
		ecs.MustMaskOf[render.Light](worlds.Engine) |
		ecs.MustMaskOf[app.Name](worlds.Engine)

	sun := worlds.Engine.Spawn(lightMask)
	sunTransform := transform.IdentityLocalTransform()
	sunTransform.Rotation = transform.QuatFromAxisAngle(-float32(math.Pi/2), transform.Vec3{1, 0, 0})
	ecs.Set(worlds.Engine, sun, sunTransform)
	ecs.Set(worlds.Engine, sun, transform.IdentityGlobalTransform())
	ecs.Set(worlds.Engine, sun, render.Light{
		Type:      render.LightTypeDirectional,
		Color:     transform.Vec3{1.0, 0.95, 0.85},
		Intensity: 1.4,
	})
	ecs.Set(worlds.Engine, sun, app.Name{Value: "Sun"})

	engineMask := ecs.MustMaskOf[transform.LocalTransform](worlds.Engine) |
		ecs.MustMaskOf[transform.GlobalTransform](worlds.Engine) |
		ecs.MustMaskOf[transform.LocalTransformDirty](worlds.Engine) |
		ecs.MustMaskOf[asset.RenderMesh](worlds.Engine) |
		ecs.MustMaskOf[app.Name](worlds.Engine)

	gameMask := ecs.MustMaskOf[Spinner](worlds.Game) |
		ecs.MustMaskOf[app.EngineEntity](worlds.Game)

	index := 0
	for x := -gridExtent; x <= gridExtent; x++ {
		for z := -gridExtent; z <= gridExtent; z++ {
			if x >= -1 && x <= 1 && z >= -1 && z <= 1 {
				continue
			}
			engineEntity := worlds.Engine.Spawn(engineMask)
			local := transform.FromTranslation(transform.Vec3{
				float32(x) * gridSpacing,
				0,
				float32(z) * gridSpacing,
			})
			local.Rotation = transform.QuatFromAxisAngle(
				float32(index)*0.4,
				transform.Vec3{0, 1, 0},
			)
			ecs.Set(worlds.Engine, engineEntity, local)
			ecs.Set(worlds.Engine, engineEntity, transform.IdentityGlobalTransform())
			meshIndex := index % len(meshes)
			ecs.Set(worlds.Engine, engineEntity, asset.RenderMesh{Mesh: meshes[meshIndex]})
			ecs.Set(worlds.Engine, engineEntity, app.Name{Value: meshNames[meshIndex] + " " + strconv.Itoa(index+1)})

			gameEntity := worlds.Game.Spawn(gameMask)
			ecs.Set(worlds.Game, gameEntity, Spinner{
				Axis:     transform.Vec3{0, 1, 0},
				Speed:    float32(math.Pi / 2),
				Rotation: local.Rotation,
			})
			ecs.Set(worlds.Game, gameEntity, app.EngineEntity{Entity: engineEntity})

			index++
		}
	}
}

// advanceSpinners is the game-side system. Walks the (Spinner |
// EngineEntity) archetype with [ecs.World.ForEach] for table-level
// column access, accumulates per-entity rotation, and writes the
// result to the engine world through [app.SyncEngineRotation]. Game
// state is the source of truth; reaching across into the engine world
// goes through the named sync API.
func advanceSpinners(game *ecs.World) {
	delta := ecs.MustResource[window.Window](game).Timing.DeltaSeconds
	engineRef := ecs.MustResource[app.EngineRef](game)

	spinnerMask := ecs.MustMaskOf[Spinner](game) | ecs.MustMaskOf[app.EngineEntity](game)

	game.ForEach(spinnerMask, 0, func(_ ecs.Entity, table *ecs.Archetype, index int) {
		spinners, _ := ecs.Column[Spinner](game, table)
		links, _ := ecs.Column[app.EngineEntity](game, table)

		s := &spinners[index]
		step := transform.QuatFromAxisAngle(s.Speed*delta, s.Axis)
		s.Rotation = step.Mul(s.Rotation).Normalize()

		app.SyncEngineRotation(engineRef.World, links[index], s.Rotation)
	})
}
