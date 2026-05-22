//go:build !js

// Command indigo is the engine demo entry point: a grid of spinning
// triangles driven by a dual-world ECS through the engine's render
// graph, viewed through a pan-orbit camera.
package main

import (
	"errors"
	"log"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/cogentcore/webgpu/wgpuglfw"
	"github.com/go-gl/glfw/v3.3/glfw"

	"indigo/app"
	"indigo/ecs"
	"indigo/render"
	"indigo/render/pass"
	"indigo/transform"
	"indigo/ui"
	"indigo/window"
)

func init() {
	runtime.LockOSThread()
}

func main() {
	setupLogging()

	if err := glfw.Init(); err != nil {
		log.Fatal(err)
	}
	defer glfw.Terminate()

	glfw.WindowHint(glfw.ClientAPI, glfw.NoAPI)
	glfwWindow, err := glfw.CreateWindow(1280, 720, "indigo", nil, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer glfwWindow.Destroy()

	instance := wgpu.CreateInstance(nil)
	defer instance.Release()

	surface := instance.CreateSurface(wgpuglfw.GetSurfaceDescriptor(glfwWindow))

	width, height := glfwWindow.GetSize()
	renderer, err := render.NewRenderer(instance, surface, uint32(width), uint32(height))
	if err != nil {
		log.Fatal(err)
	}
	defer renderer.Release()

	worlds, demo := buildWorlds(renderer)
	installInputCallbacks(glfwWindow, worlds.Engine)
	installDropCallback(glfwWindow, worlds, renderer)

	glfwWindow.SetSizeCallback(func(_ *glfw.Window, w, h int) {
		if w <= 0 || h <= 0 {
			return
		}
		if err := renderer.Resize(uint32(w), uint32(h)); err != nil {
			log.Printf("resize error: %v", err)
		}
		viewport := window.ViewportSize{Width: uint32(w), Height: uint32(h)}
		ecs.MustResource[window.Window](worlds.Engine).Viewport = viewport
		if worlds.UI != nil {
			ecs.MustResource[window.Window](worlds.UI).Viewport = viewport
		}
	})

	last := time.Now()
	for !glfwWindow.ShouldClose() {
		glfw.PollEvents()

		now := time.Now()
		delta := float32(now.Sub(last).Seconds())
		last = now

		syncUiPointer(worlds)
		ctx := newHudContext(worlds)
		ctx.refreshHudLayout()
		// Consume wheel before TickFrame so pan-orbit doesn't also
		// zoom on the same wheel event.
		ctx.updateTreeScroll()

		app.TickFrame(worlds, demo, delta)
		handleRightClick(worlds)
		driveTextInputs(worlds)
		handleUiClicks(worlds)
		ctx.refreshModeButtons()
		ctx.refreshMenuPopups()
		ctx.refreshInteractiveHovers()
		ctx.refreshEntityTree()
		ctx.refreshInspector()
		ctx.updateInspectorCaret()

		if ctx.Hud.RequestExit {
			glfwWindow.SetShouldClose(true)
		}

		switch err := render.RenderFrame(renderer, worlds.Engine); {
		case err == nil:
		case errors.Is(err, render.ErrSurfaceLost):
			renderer.Reconfigure()
		default:
			log.Fatal(err)
		}

		pass.ProcessPickingReadback(renderer, worlds.Engine)

		if picking := ecs.MustResource[*pass.Picking](worlds.Engine); (*picking).Result != nil {
			result := (*picking).Result
			(*picking).Result = nil
			handlePickResult(worlds, result.EntityID)
		}

		app.PostFrame(worlds)
	}
}

// installDropCallback wires GLFW's drag-drop handler to load each
// dropped .gltf or .glb file via [loadGltfInto] and refresh the UI
// layout so newly spawned entities show up in the hierarchy panel.
func installDropCallback(glfwWindow *glfw.Window, worlds app.Worlds, renderer *render.Renderer) {
	glfwWindow.SetDropCallback(func(_ *glfw.Window, paths []string) {
		for _, path := range paths {
			ext := strings.ToLower(filepath.Ext(path))
			if ext != ".gltf" && ext != ".glb" {
				log.Printf("skip drop: unsupported extension %q", path)
				continue
			}
			if _, err := loadGltfInto(worlds.Engine, renderer, path); err != nil {
				log.Printf("gltf load failed: %v", err)
				continue
			}
			if worlds.UI != nil {
				ui.MarkLayoutDirty(worlds.UI)
			}
		}
	})
}

// installInputCallbacks wires GLFW's cursor / mouse-button / scroll
// callbacks to accumulate frame deltas into the engine world's Input
// resource. tickFrame zeroes the deltas at the end of each frame.
func installInputCallbacks(glfwWindow *glfw.Window, engine *ecs.World) {
	var previousMouse transform.Vec2
	var haveMouse bool

	glfwWindow.SetCursorPosCallback(func(_ *glfw.Window, x, y float64) {
		input := ecs.MustResource[render.Input](engine)
		current := transform.Vec2{float32(x), float32(y)}
		if haveMouse {
			input.MouseDelta = input.MouseDelta.Add(current.Sub(previousMouse))
		}
		input.MousePosition = current
		previousMouse = current
		haveMouse = true
	})

	glfwWindow.SetMouseButtonCallback(func(_ *glfw.Window, button glfw.MouseButton, action glfw.Action, _ glfw.ModifierKey) {
		input := ecs.MustResource[render.Input](engine)
		pressed := action == glfw.Press
		switch button {
		case glfw.MouseButtonLeft:
			input.LeftDown = pressed
			if pressed {
				picking := *ecs.MustResource[*pass.Picking](engine)
				if picking != nil {
					altHeld := glfwWindow.GetKey(glfw.KeyLeftAlt) == glfw.Press || glfwWindow.GetKey(glfw.KeyRightAlt) == glfw.Press
					shiftHeld := glfwWindow.GetKey(glfw.KeyLeftShift) == glfw.Press || glfwWindow.GetKey(glfw.KeyRightShift) == glfw.Press
					recordPickModifiers(engine, altHeld, shiftHeld)
					pass.QueuePick(picking, uint32(input.MousePosition[0]), uint32(input.MousePosition[1]))
				}
			}
		case glfw.MouseButtonRight:
			input.RightDown = pressed
		case glfw.MouseButtonMiddle:
			input.MiddleDown = pressed
		}
	})

	glfwWindow.SetScrollCallback(func(_ *glfw.Window, _, yOffset float64) {
		input := ecs.MustResource[render.Input](engine)
		input.Wheel += float32(yOffset)
	})

	glfwWindow.SetKeyCallback(func(w *glfw.Window, key glfw.Key, _ int, action glfw.Action, _ glfw.ModifierKey) {
		if key == glfw.KeyEscape && action == glfw.Press {
			w.SetShouldClose(true)
			return
		}
		r, ok := glfwKeyRune(key)
		if !ok {
			return
		}
		input := ecs.MustResource[render.Input](engine)
		switch action {
		case glfw.Press:
			render.InputMarkKeyDown(input, r)
		case glfw.Release:
			render.InputMarkKeyUp(input, r)
		}
	})

	glfwWindow.SetCharCallback(func(_ *glfw.Window, char rune) {
		input := ecs.MustResource[render.Input](engine)
		input.Chars = append(input.Chars, char)
	})
}

// glfwKeyRune maps a subset of GLFW key codes to the uppercase ASCII
// runes the Input resource expects: A-Z, space, and digits.
func glfwKeyRune(key glfw.Key) (rune, bool) {
	switch {
	case key >= glfw.KeyA && key <= glfw.KeyZ:
		return rune(key), true
	case key == glfw.KeySpace:
		return ' ', true
	case key >= glfw.Key0 && key <= glfw.Key9:
		return rune(key), true
	case key == glfw.KeyBackspace:
		return '\b', true
	case key == glfw.KeyEnter:
		return '\n', true
	case key == glfw.KeyLeft:
		return '\x01', true
	case key == glfw.KeyRight:
		return '\x02', true
	}
	return 0, false
}
