//go:build !js

// Command rendergraph-go is the engine demo entry point: a grid of spinning
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

	"rendergraph-go/ecs"
	"rendergraph-go/render"
	"rendergraph-go/transform"
	"rendergraph-go/window"
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
	glfwWindow, err := glfw.CreateWindow(1280, 720, "rendergraph-go", nil, nil)
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

		tickFrame(worlds, demo, delta)

		switch err := renderer.RenderFrame(worlds.Engine); {
		case err == nil:
		case errors.Is(err, render.ErrSurfaceLost):
			renderer.Reconfigure()
		default:
			log.Fatal(err)
		}
	}
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
		if action != glfw.Press {
			return
		}
		if key == glfw.KeyEscape {
			w.SetShouldClose(true)
			return
		}
		if key >= glfw.KeyA && key <= glfw.KeyZ {
			input := ecs.Resource[render.Input](engine)
			input.KeysJustDown = append(input.KeysJustDown, rune(key))
		}
	})
}
