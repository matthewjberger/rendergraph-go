//go:build !js

// Command breakout is the engine's breakout-clone demo: a paddle, a
// ball, and a wall of cubes rendered as 3D primitives through the
// engine's render graph.
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
	glfwWindow, err := glfw.CreateWindow(1280, 720, "breakout", nil, nil)
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
		glfwWindow.SetTitle(titleForState(ecs.Resource[GameState](worlds.Game)))

		switch err := render.RenderFrame(renderer, worlds.Engine); {
		case err == nil:
		case errors.Is(err, render.ErrSurfaceLost):
			renderer.Reconfigure()
		default:
			log.Fatal(err)
		}

		app.PostFrame(worlds)
	}
}

// installInputCallbacks wires the GLFW cursor / mouse-button / scroll
// callbacks onto the engine world's Input resource, plus keyboard
// down/up for held A/D paddle motion, edge-triggered space (launch),
// and R (reset).
func installInputCallbacks(glfwWindow *glfw.Window, engine *ecs.World) {
	glfwWindow.SetCursorPosCallback(func(_ *glfw.Window, x, y float64) {
		input := ecs.Resource[render.Input](engine)
		input.MousePosition[0] = float32(x)
		input.MousePosition[1] = float32(y)
	})

	glfwWindow.SetMouseButtonCallback(func(_ *glfw.Window, button glfw.MouseButton, action glfw.Action, _ glfw.ModifierKey) {
		input := ecs.Resource[render.Input](engine)
		pressed := action == glfw.Press
		switch button {
		case glfw.MouseButtonLeft:
			input.LeftDown = pressed
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

// glfwKeyRune maps the breakout-relevant subset of GLFW key codes to
// the uppercase ASCII runes the Input resource expects.
func glfwKeyRune(key glfw.Key) (rune, bool) {
	switch {
	case key >= glfw.KeyA && key <= glfw.KeyZ:
		return rune(key), true
	case key == glfw.KeySpace:
		return ' ', true
	}
	return 0, false
}
