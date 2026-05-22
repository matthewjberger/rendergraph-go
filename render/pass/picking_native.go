//go:build !js

package pass

import (
	"fmt"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"

	"indigo/ecs"
	"indigo/render"
)

// ProcessPickingReadback advances any in-flight pick request by one
// step. Call after [RenderFrame] each frame from the main loop.
// Non-blocking: issues MapAsync once and on subsequent frames just
// checks whether the GPU has flushed the result; [Device.Poll] with
// wait=false drives the binding's internal queue.
//
// Native build. The wasm build has a no-op stub in picking_js.go
// because cogentcore's MapAsync blocks on a JS promise via
// jsx.Await, and blocking a goroutine inside a requestAnimationFrame
// callback while waiting for that promise produces a deadlock (the
// promise can only resolve once Go returns control to JS).
func ProcessPickingReadback(renderer *render.Renderer, world *ecs.World) {
	if !ecs.HasResource[*Picking](world) {
		return
	}
	picking := *ecs.MustResource[*Picking](world)
	if picking == nil || !picking.hasInFlight || picking.staging == nil {
		return
	}

	if !picking.mapping {
		err := picking.staging.MapAsync(wgpu.MapModeRead, 0, stagingRowStride, func(status wgpu.BufferMapAsyncStatus) {
			picking.mu.Lock()
			defer picking.mu.Unlock()
			picking.mapped = true
			if status != wgpu.BufferMapAsyncStatusSuccess {
				picking.mapErr = fmt.Errorf("picking readback: map status %d", status)
			}
		})
		if err != nil {
			resetPicking(picking)
			return
		}
		picking.mapping = true
	}

	renderer.Device.Poll(false, nil)

	picking.mu.Lock()
	mapped := picking.mapped
	mapErr := picking.mapErr
	picking.mu.Unlock()

	if !mapped {
		return
	}
	if mapErr != nil {
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
	resetPicking(picking)
}
