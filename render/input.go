package render

import "rendergraph-go/transform"

// Input is the engine's per-frame input snapshot. Stored as a typed
// resource on the engine world; platform glue (GLFW callbacks on
// desktop, DOM listeners on wasm) writes into it once per frame.
//
// The fields are the minimum a camera controller plus a few keyboard
// toggles need. Keep this flat and value-typed so the platform layer
// never has to think about allocation.
//
// KeysJustDown is the set of keys that transitioned from up to down
// during the current frame, encoded as uppercase ASCII runes. Cleared
// in BeginFrame so consumers see edge-triggered presses, not held.
type Input struct {
	MousePosition transform.Vec2
	MouseDelta    transform.Vec2
	Wheel         float32

	LeftDown   bool
	RightDown  bool
	MiddleDown bool

	KeysJustDown []rune
}

// BeginFrame zeroes the per-frame deltas and key edges (mouse
// movement, wheel scroll, just-pressed keys) so the input state at
// the start of each frame is "no input this frame yet." Held buttons
// stay held.
func (i *Input) BeginFrame() {
	i.MouseDelta = transform.Vec2{0, 0}
	i.Wheel = 0
	i.KeysJustDown = i.KeysJustDown[:0]
}
