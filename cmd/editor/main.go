//go:build !js

// Command indigo is the engine demo entry point: a grid of spinning
// triangles driven by a dual-world ECS through the engine's render
// graph, viewed through a pan-orbit camera.
package main

import (
	"errors"
	"log"
	"runtime"
	"time"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/cogentcore/webgpu/wgpuglfw"
	"github.com/go-gl/glfw/v3.3/glfw"

	"indigo/app"
	"indigo/ecs"
	"indigo/render"
	"indigo/transform"
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

	glfwWindow.SetSizeCallback(func(_ *glfw.Window, w, h int) {
		if w <= 0 || h <= 0 {
			return
		}
		if err := renderer.Resize(uint32(w), uint32(h)); err != nil {
			log.Printf("resize error: %v", err)
		}
		ecs.Resource[window.Window](worlds.Engine).Viewport = window.ViewportSize{
			Width:  uint32(w),
			Height: uint32(h),
		}
	})

	last := time.Now()
	for !glfwWindow.ShouldClose() {
		glfw.PollEvents()

		now := time.Now()
		delta := float32(now.Sub(last).Seconds())
		last = now

		app.TickFrame(worlds, demo, delta)

		switch err := render.RenderFrame(renderer, worlds.Engine); {
		case err == nil:
		case errors.Is(err, render.ErrSurfaceLost):
			renderer.Reconfigure()
		default:
			log.Fatal(err)
		}

		render.ProcessPickingReadback(renderer, worlds.Engine)

		if picking := ecs.Resource[*render.Picking](worlds.Engine); (*picking).Result != nil {
			result := (*picking).Result
			(*picking).Result = nil
			applySelection(worlds.Engine, result.EntityID)
			glfwWindow.SetTitle(pickTitle(result))
		}

		app.PostFrame(worlds)
	}
}

func pickTitle(result *render.PickResult) string {
	if result.EntityID == 0 {
		return "indigo editor — picked: (nothing)"
	}
	return "indigo editor — picked entity " + uintToStr(result.EntityID)
}

// applySelection deselects every currently-selected entity and adds
// the Selected tag to the entity matching pickedID (if any). The
// outline post-process picks this up next frame.
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

func uintToStr(v uint32) string {
	if v == 0 {
		return "0"
	}
	var buf [12]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	return string(buf[i:])
}

// installInputCallbacks wires GLFW's cursor / mouse-button / scroll
// callbacks to accumulate frame deltas into the engine world's Input
// resource. tickFrame zeroes the deltas at the end of each frame.
func installInputCallbacks(glfwWindow *glfw.Window, engine *ecs.World) {
	var previousMouse transform.Vec2
	var haveMouse bool

	glfwWindow.SetCursorPosCallback(func(_ *glfw.Window, x, y float64) {
		input := ecs.Resource[render.Input](engine)
		current := transform.Vec2{float32(x), float32(y)}
		if haveMouse {
			input.MouseDelta = input.MouseDelta.Add(current.Sub(previousMouse))
		}
		input.MousePosition = current
		previousMouse = current
		haveMouse = true
	})

	glfwWindow.SetMouseButtonCallback(func(_ *glfw.Window, button glfw.MouseButton, action glfw.Action, _ glfw.ModifierKey) {
		input := ecs.Resource[render.Input](engine)
		pressed := action == glfw.Press
		switch button {
		case glfw.MouseButtonLeft:
			input.LeftDown = pressed
			if pressed {
				picking := *ecs.Resource[*render.Picking](engine)
				if picking != nil {
					render.QueuePick(picking, uint32(input.MousePosition[0]), uint32(input.MousePosition[1]))
				}
			}
		case glfw.MouseButtonRight:
			input.RightDown = pressed
		case glfw.MouseButtonMiddle:
			input.MiddleDown = pressed
		}
	})

	glfwWindow.SetScrollCallback(func(_ *glfw.Window, _, yOffset float64) {
		input := ecs.Resource[render.Input](engine)
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
		input := ecs.Resource[render.Input](engine)
		switch action {
		case glfw.Press:
			render.InputMarkKeyDown(input, r)
		case glfw.Release:
			render.InputMarkKeyUp(input, r)
		}
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
	}
	return 0, false
}
