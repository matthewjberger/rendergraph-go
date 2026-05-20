package render

import "indigo/transform"

// Input is the engine's per-frame input snapshot. Stored as a typed
// resource on the engine world; platform glue (GLFW callbacks on
// desktop, DOM listeners on wasm) writes into it once per frame.
//
// KeysDown is the set of keys currently held, encoded as uppercase
// ASCII runes (plus space for ' '). Survives across frames until the
// platform reports a key-up.
//
// KeysJustDown is the set of keys that transitioned from up to down
// during the current frame. Cleared by [InputBeginFrame] so consumers
// see edge-triggered presses, not held.
type Input struct {
	MousePosition transform.Vec2
	MouseDelta    transform.Vec2
	Wheel         float32

	LeftDown   bool
	RightDown  bool
	MiddleDown bool

	KeysDown     map[rune]struct{}
	KeysJustDown []rune
}

// NewInput returns an Input with its key maps allocated.
func NewInput() Input {
	return Input{KeysDown: make(map[rune]struct{}, 16)}
}

// InputMarkKeyDown records both the held state and the edge-
// triggered press for a key. Auto-allocates KeysDown on first use so
// zero-valued Input is usable.
func InputMarkKeyDown(i *Input, key rune) {
	if i.KeysDown == nil {
		i.KeysDown = make(map[rune]struct{}, 16)
	}
	i.KeysDown[key] = struct{}{}
	i.KeysJustDown = append(i.KeysJustDown, key)
}

// InputMarkKeyUp clears the held state for a key.
func InputMarkKeyUp(i *Input, key rune) {
	delete(i.KeysDown, key)
}

// InputIsKeyDown reports whether key is currently held.
func InputIsKeyDown(i *Input, key rune) bool {
	_, ok := i.KeysDown[key]
	return ok
}

// InputBeginFrame zeroes the per-frame deltas and key edges (mouse
// movement, wheel scroll, just-pressed keys) so the input state at
// the start of each frame is "no input this frame yet." Held buttons
// and held keys stay held.
func InputBeginFrame(i *Input) {
	i.MouseDelta = transform.Vec2{0, 0}
	i.Wheel = 0
	i.KeysJustDown = i.KeysJustDown[:0]
}
