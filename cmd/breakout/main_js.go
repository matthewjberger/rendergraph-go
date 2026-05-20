//go:build js

package main

import (
	"syscall/js"
	"time"

	"github.com/cogentcore/webgpu/wgpu"

	"indigo/app"
	"indigo/ecs"
	"indigo/render"
	"indigo/window"
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

	titleEl := doc.Call("getElementById", "title")

	last := time.Now()
	var frame js.Func
	frame = js.FuncOf(func(_ js.Value, _ []js.Value) any {
		now := time.Now()
		delta := float32(now.Sub(last).Seconds())
		last = now

		app.TickFrame(worlds, demo, delta)
		title := titleForState(ecs.Resource[GameState](worlds.Game))
		if titleEl.Truthy() {
			titleEl.Set("textContent", title)
		}
		doc.Set("title", title)

		if err := render.RenderFrame(renderer, worlds.Engine); err != nil {
			js.Global().Get("console").Call("error", "render error: "+err.Error())
		}

		app.PostFrame(worlds)

		js.Global().Call("requestAnimationFrame", frame)
		return nil
	})
	js.Global().Call("requestAnimationFrame", frame)

	select {}
}

func installCanvasInputListeners(canvas js.Value, engine *ecs.World) {
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
		r, ok := domKeyRune(args[0])
		if !ok {
			return nil
		}
		input := ecs.Resource[render.Input](engine)
		render.InputMarkKeyDown(input, r)
		return nil
	}))

	js.Global().Get("window").Call("addEventListener", "keyup", js.FuncOf(func(_ js.Value, args []js.Value) any {
		if len(args) == 0 {
			return nil
		}
		r, ok := domKeyRune(args[0])
		if !ok {
			return nil
		}
		input := ecs.Resource[render.Input](engine)
		render.InputMarkKeyUp(input, r)
		return nil
	}))
}

func domKeyRune(event js.Value) (rune, bool) {
	key := event.Get("key").String()
	if key == " " {
		return ' ', true
	}
	if len(key) != 1 {
		return 0, false
	}
	r := []rune(key)[0]
	if r >= 'a' && r <= 'z' {
		r -= 'a' - 'A'
	}
	if r >= 'A' && r <= 'Z' {
		return r, true
	}
	return 0, false
}
