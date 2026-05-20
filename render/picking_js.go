//go:build js

package render

import "indigo/ecs"

// ProcessPickingReadback on wasm is a no-op that clears the
// in-flight bookkeeping without ever issuing MapAsync. The picking
// pass still records the GPU copy, but no result is published; the
// title/selection callback that reads Picking.Result simply never
// sees a value.
//
// Why: cogentcore/webgpu's Buffer.MapAsync goes through jsx.Await,
// which blocks the calling goroutine on `<-done` until a JS Promise
// resolves. Inside a requestAnimationFrame callback that's a deadlock:
// the Promise can only resolve once Go yields back to JS, and Go
// can't yield while the calling goroutine is blocked. Real wasm
// picking needs the readback handled outside the frame callback (a
// long-lived listener goroutine, or going under cogentcore's wrapper
// to attach a .then() that toggles a Go flag without awaiting). Left
// as follow-up.
func ProcessPickingReadback(_ *Renderer, world *ecs.World) {
	if !ecs.HasResource[*Picking](world) {
		return
	}
	picking := *ecs.Resource[*Picking](world)
	if picking == nil || !picking.hasInFlight {
		return
	}
	resetPicking(picking)
}
