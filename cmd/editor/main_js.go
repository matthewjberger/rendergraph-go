//go:build js

package main

import (
	"log"
	"path/filepath"
	"strings"
	"sync"
	"syscall/js"
	"time"

	"github.com/cogentcore/webgpu/wgpu"

	"github.com/matthewjberger/indigo/app"
	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/internal/render/pass"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/transform"
	"github.com/matthewjberger/indigo/ui"
	"github.com/matthewjberger/indigo/window"
)

var pendingDrops struct {
	sync.Mutex
	files []pendingDrop
}

type pendingDrop struct {
	name string
	data []byte
}

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
	installCanvasDropListener(canvas)

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
		viewport := window.ViewportSize{Width: w, Height: h}
		ecs.MustResource[window.Window](worlds.Engine).Viewport = viewport
		if worlds.UI != nil {
			ecs.MustResource[window.Window](worlds.UI).Viewport = viewport
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

		drainPendingDrops(worlds, renderer)
		syncUiPointer(worlds)
		ctx := newHudContext(worlds)
		ctx.refreshHudLayout()
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

		if err := render.RenderFrame(renderer, worlds.Engine); err != nil {
			js.Global().Get("console").Call("error", "render error: "+err.Error())
		}

		pass.ProcessPickingReadback(renderer, worlds.Engine)

		if picking := ecs.MustResource[*pass.Picking](worlds.Engine); (*picking).Result != nil {
			result := (*picking).Result
			(*picking).Result = nil
			handlePickResult(worlds, result.EntityID)
		}

		app.PostFrame(worlds)

		js.Global().Call("requestAnimationFrame", frame)
		return nil
	})
	js.Global().Call("requestAnimationFrame", frame)

	select {}
}

func installCanvasInputListeners(canvas js.Value, engine *ecs.World) {
	canvas.Call("addEventListener", "mousemove", js.FuncOf(func(_ js.Value, args []js.Value) any {
		if len(args) == 0 {
			return nil
		}
		event := args[0]
		input := ecs.MustResource[render.Input](engine)
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
		button := event.Get("button").Int()
		setMouseButton(engine, button, true)
		if button == 0 {
			if uiRef, ok := ecs.Resource[ui.WorldRef](engine); ok && uiRef != nil && uiRef.World != nil {
				if uiPointer, ok := ecs.Resource[ui.PointerState](uiRef.World); ok && uiPointer != nil && uiPointer.OverUI {
					return nil
				}
			}
			picking := *ecs.MustResource[*pass.Picking](engine)
			if picking != nil {
				input := ecs.MustResource[render.Input](engine)
				altHeld := event.Get("altKey").Bool()
				shiftHeld := event.Get("shiftKey").Bool()
				recordPickModifiers(engine, altHeld, shiftHeld)
				pass.QueuePick(picking, uint32(input.MousePosition[0]), uint32(input.MousePosition[1]))
			}
		}
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
		input := ecs.MustResource[render.Input](engine)
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
		r, ok := domKeyRune(args[0])
		if !ok {
			return nil
		}
		input := ecs.MustResource[render.Input](engine)
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
		input := ecs.MustResource[render.Input](engine)
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
	switch {
	case r >= 'A' && r <= 'Z':
		return r, true
	case r >= '0' && r <= '9':
		return r, true
	}
	return 0, false
}

func setMouseButton(engine *ecs.World, button int, pressed bool) {
	input := ecs.MustResource[render.Input](engine)
	switch button {
	case 0:
		input.LeftDown = pressed
	case 1:
		input.MiddleDown = pressed
	case 2:
		input.RightDown = pressed
	}
}

func installCanvasDropListener(canvas js.Value) {
	canvas.Call("addEventListener", "dragover", js.FuncOf(func(_ js.Value, args []js.Value) any {
		if len(args) > 0 {
			args[0].Call("preventDefault")
		}
		return nil
	}), map[string]any{"passive": false})

	canvas.Call("addEventListener", "drop", js.FuncOf(func(_ js.Value, args []js.Value) any {
		if len(args) == 0 {
			return nil
		}
		event := args[0]
		event.Call("preventDefault")
		event.Call("stopPropagation")
		dt := event.Get("dataTransfer")
		if !dt.Truthy() {
			return nil
		}
		files := dt.Get("files")
		if !files.Truthy() {
			return nil
		}
		count := files.Length()
		for index := 0; index < count; index++ {
			readDroppedFile(files.Index(index))
		}
		return nil
	}), map[string]any{"passive": false})
}

func readDroppedFile(file js.Value) {
	name := file.Get("name").String()
	promise := file.Call("arrayBuffer")
	var thenFn js.Func
	thenFn = js.FuncOf(func(_ js.Value, thenArgs []js.Value) any {
		defer thenFn.Release()
		if len(thenArgs) == 0 {
			return nil
		}
		buffer := thenArgs[0]
		uint8Array := js.Global().Get("Uint8Array").New(buffer)
		data := make([]byte, uint8Array.Length())
		js.CopyBytesToGo(data, uint8Array)
		pendingDrops.Lock()
		pendingDrops.files = append(pendingDrops.files, pendingDrop{name: name, data: data})
		pendingDrops.Unlock()
		return nil
	})
	promise.Call("then", thenFn)
}

func drainPendingDrops(worlds app.Worlds, renderer *render.Renderer) {
	pendingDrops.Lock()
	drops := pendingDrops.files
	pendingDrops.files = nil
	pendingDrops.Unlock()
	for _, drop := range drops {
		ext := strings.ToLower(filepath.Ext(drop.name))
		if ext != ".gltf" && ext != ".glb" {
			log.Printf("skip drop: unsupported extension %q", drop.name)
			continue
		}
		if _, err := loadGltfBytes(worlds.Engine, renderer, drop.name, drop.data); err != nil {
			log.Printf("gltf load failed: %v", err)
			continue
		}
		if worlds.UI != nil {
			ui.MarkLayoutDirty(worlds.UI)
		}
	}
}
