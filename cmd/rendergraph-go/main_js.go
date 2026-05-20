//go:build js

package main

import (
	"syscall/js"
	"time"

	"github.com/cogentcore/webgpu/wgpu"

	"rendergraph-go/ecs"
	"rendergraph-go/render"
	"rendergraph-go/transform"
	"rendergraph-go/window"
)

func main() {
	setupLogging()

	doc := js.Global().Get("document")
	canvas := doc.Call("getElementById", "canvas")
	if !canvas.Truthy() {
		js.Global().Get("console").Call("error", "no <canvas id=\"canvas\"> found")
		return
	}

	width := uint32(canvas.Get("width").Int())
	height := uint32(canvas.Get("height").Int())

	instance := wgpu.CreateInstance(nil)
	if instance == nil {
		js.Global().Get("console").Call("error", "WebGPU not supported in this browser")
		return
	}

	surface := instance.CreateSurface(&wgpu.SurfaceDescriptor{Canvas: canvas})

	renderer, err := render.NewRenderer(instance, surface, width, height)
	if err != nil {
		js.Global().Get("console").Call("error", "renderer init failed: "+err.Error())
		return
	}

	worlds, demo := buildWorlds(renderer)
	installCanvasInputListeners(canvas, worlds.Engine)

	resizeFunc := js.FuncOf(func(_ js.Value, args []js.Value) any {
		entries := args[0]
		if entries.Length() == 0 {
			return nil
		}
		entry := entries.Index(0)
		contentBoxSize := entry.Get("contentBoxSize")
		if !contentBoxSize.Truthy() || contentBoxSize.Length() == 0 {
			return nil
		}
		box := contentBoxSize.Index(0)
		w := uint32(box.Get("inlineSize").Int())
		h := uint32(box.Get("blockSize").Int())
		if w == 0 || h == 0 {
			return nil
		}
		canvas.Set("width", w)
		canvas.Set("height", h)
		if err := renderer.Resize(w, h); err != nil {
			js.Global().Get("console").Call("error", "resize failed: "+err.Error())
		}
		ecs.Resource[window.Window](worlds.Engine).Viewport = window.ViewportSize{
			Width:  w,
			Height: h,
		}
		return nil
	})
	resizeObserver := js.Global().Get("ResizeObserver").New(resizeFunc)
	resizeObserver.Call("observe", canvas)

	last := time.Now()
	var frame js.Func
	frame = js.FuncOf(func(_ js.Value, _ []js.Value) any {
		now := time.Now()
		delta := float32(now.Sub(last).Seconds())
		last = now

		tickFrame(worlds, demo, delta)

		if err := renderer.RenderFrame(worlds.Engine); err != nil {
			js.Global().Get("console").Call("error", "render error: "+err.Error())
		}

		js.Global().Call("requestAnimationFrame", frame)
		return nil
	})
	js.Global().Call("requestAnimationFrame", frame)

	select {}
}

// installCanvasInputListeners attaches DOM pointer / wheel handlers to
// the canvas. Each handler accumulates into the engine world's Input
// resource; tickFrame consumes the snapshot and clears per-frame
// deltas. The js.Funcs are intentionally never released because main
// never returns.
func installCanvasInputListeners(canvas js.Value, engine *ecs.World) {
	canvas.Call("addEventListener", "mousemove", js.FuncOf(func(_ js.Value, args []js.Value) any {
		if len(args) == 0 {
			return nil
		}
		event := args[0]
		input := ecs.Resource[render.Input](engine)
		x := float32(event.Get("offsetX").Float())
		y := float32(event.Get("offsetY").Float())
		dx := float32(event.Get("movementX").Float())
		dy := float32(event.Get("movementY").Float())
		input.MousePosition = transform.Vec2{x, y}
		input.MouseDelta = input.MouseDelta.Add(transform.Vec2{dx, dy})
		return nil
	}))

	canvas.Call("addEventListener", "mousedown", js.FuncOf(func(_ js.Value, args []js.Value) any {
		if len(args) == 0 {
			return nil
		}
		event := args[0]
		setMouseButton(engine, event.Get("button").Int(), true)
		return nil
	}))

	canvas.Call("addEventListener", "mouseup", js.FuncOf(func(_ js.Value, args []js.Value) any {
		if len(args) == 0 {
			return nil
		}
		event := args[0]
		setMouseButton(engine, event.Get("button").Int(), false)
		return nil
	}))

	canvas.Call("addEventListener", "wheel", js.FuncOf(func(_ js.Value, args []js.Value) any {
		if len(args) == 0 {
			return nil
		}
		event := args[0]
		event.Call("preventDefault")
		input := ecs.Resource[render.Input](engine)
		deltaY := float32(event.Get("deltaY").Float())
		input.Wheel -= deltaY * 0.01
		return nil
	}), map[string]any{"passive": false})

	canvas.Call("addEventListener", "contextmenu", js.FuncOf(func(_ js.Value, args []js.Value) any {
		if len(args) > 0 {
			args[0].Call("preventDefault")
		}
		return nil
	}))

	js.Global().Get("window").Call("addEventListener", "keydown", js.FuncOf(func(_ js.Value, args []js.Value) any {
		if len(args) == 0 {
			return nil
		}
		event := args[0]
		key := event.Get("key").String()
		if len(key) != 1 {
			return nil
		}
		r := []rune(key)[0]
		if r >= 'a' && r <= 'z' {
			r -= 'a' - 'A'
		}
		if r < 'A' || r > 'Z' {
			return nil
		}
		input := ecs.Resource[render.Input](engine)
		input.KeysJustDown = append(input.KeysJustDown, r)
		return nil
	}))
}

func setMouseButton(engine *ecs.World, button int, pressed bool) {
	input := ecs.Resource[render.Input](engine)
	switch button {
	case 0:
		input.LeftDown = pressed
	case 1:
		input.MiddleDown = pressed
	case 2:
		input.RightDown = pressed
	}
}
