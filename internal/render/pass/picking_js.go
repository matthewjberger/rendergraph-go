//go:build js

package pass

import (
	"syscall/js"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"

	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
)

type pickingJSState struct {
	onResolve js.Func
	onReject  js.Func
}

var pickingJSStates = map[*Picking]*pickingJSState{}

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
