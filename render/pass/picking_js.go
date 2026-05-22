//go:build js

package pass

import (
	"syscall/js"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"

	"indigo/ecs"
	"indigo/render"
)

// pickingJSState holds per-Picking JavaScript callbacks the readback
// path needs to keep alive across rAF invocations. We can't store
// js.Func on the platform-neutral [Picking] struct (it doesn't exist
// on native builds), so a package-level map[*Picking]*pickingJSState
// keeps the binding without polluting shared types.
type pickingJSState struct {
	onResolve js.Func
	onReject  js.Func
}

var pickingJSStates = map[*Picking]*pickingJSState{}

// ProcessPickingReadback advances any in-flight pick request by one
// step. Build-tag-split because wasm cannot use cogentcore/webgpu's
// blocking [wgpu.Buffer.MapAsync] from inside a requestAnimationFrame
// callback: that path goes through jsx.Await, which parks the calling
// goroutine on <-done, and the Promise can only resolve once Go
// yields back to JS - deadlock.
//
// This implementation drops under the cogentcore wrapper and calls
// GPUBuffer.mapAsync directly via syscall/js, attaching a .then()
// callback (js.FuncOf) that flips picking.mapped from a fresh
// goroutine spawned by the JS event loop. The rAF callback issues
// mapAsync, returns immediately, and the next rAF observes the flag
// and reads the mapped bytes.
//
// Buffer's underlying js.Value is read via unsafe.Pointer because
// the field is unexported in cogentcore/webgpu. The buffer struct
// holds exactly one field on JS (jsValue js.Value), so a pointer
// cast is well-defined within the pinned cogentcore version.
func ProcessPickingReadback(_ *render.Renderer, world *ecs.World) {
	if !ecs.HasResource[*Picking](world) {
		return
	}
	picking := *ecs.MustResource[*Picking](world)
	if picking == nil || !picking.hasInFlight || picking.staging == nil {
		return
	}

	if !picking.mapping {
		issueMapAsyncJS(picking)
		return
	}

	picking.mu.Lock()
	mapped := picking.mapped
	mapErr := picking.mapErr
	picking.mu.Unlock()

	if !mapped {
		return
	}
	if mapErr != nil {
		clearPickingJS(picking)
		resetPicking(picking)
		return
	}

	bytes := picking.staging.GetMappedRange(0, uint(stagingRowStride))
	var entityID uint32
	if len(bytes) >= 4 {
		entityID = *(*uint32)(unsafe.Pointer(&bytes[0]))
	}
	picking.staging.Unmap()

	picking.Result = &PickResult{EntityID: entityID, Request: picking.requested}
	clearPickingJS(picking)
	resetPicking(picking)
}

func issueMapAsyncJS(picking *Picking) {
	jsBuf := bufferJSValue(picking.staging)
	if !jsBuf.Truthy() {
		resetPicking(picking)
		return
	}

	state, ok := pickingJSStates[picking]
	if !ok {
		state = &pickingJSState{}
		state.onResolve = js.FuncOf(func(_ js.Value, _ []js.Value) any {
			picking.mu.Lock()
			picking.mapped = true
			picking.mu.Unlock()
			return nil
		})
		state.onReject = js.FuncOf(func(_ js.Value, args []js.Value) any {
			picking.mu.Lock()
			picking.mapped = true
			if len(args) > 0 {
				picking.mapErr = jsRejectError(args[0])
			} else {
				picking.mapErr = errMapRejected
			}
			picking.mu.Unlock()
			return nil
		})
		pickingJSStates[picking] = state
	}

	promise := jsBuf.Call("mapAsync", uint32(wgpu.MapModeRead), 0, stagingRowStride)
	promise.Call("then", state.onResolve, state.onReject)
	picking.mapping = true
}

func clearPickingJS(picking *Picking) {
	state, ok := pickingJSStates[picking]
	if !ok {
		return
	}
	state.onResolve.Release()
	state.onReject.Release()
	delete(pickingJSStates, picking)
}

// bufferJSValue extracts cogentcore/webgpu's unexported jsValue
// field. The wrapper is `type Buffer struct { jsValue js.Value }`
// and there's no public accessor, so this is the only path. The
// size assertion below fails to compile if a future dependency
// bump adds fields, surfacing the breakage loudly instead of as a
// silent memory-read at runtime.
var _ [unsafe.Sizeof(wgpu.Buffer{}) - unsafe.Sizeof(js.Value{})]struct{} = [0]struct{}{}

func bufferJSValue(buffer *wgpu.Buffer) js.Value {
	if buffer == nil {
		return js.Null()
	}
	return *(*js.Value)(unsafe.Pointer(buffer))
}

type mapAsyncError struct{ message string }

func (e *mapAsyncError) Error() string { return e.message }

var errMapRejected = &mapAsyncError{message: "picking readback: mapAsync rejected"}

func jsRejectError(reason js.Value) error {
	if reason.Type() == js.TypeString {
		return &mapAsyncError{message: "picking readback: " + reason.String()}
	}
	if message := reason.Get("message"); message.Truthy() {
		return &mapAsyncError{message: "picking readback: " + message.String()}
	}
	return errMapRejected
}
