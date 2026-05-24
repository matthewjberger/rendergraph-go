package window

type ViewportSize struct {
	Width  uint32
	Height uint32
}

type Timing struct {
	DeltaSeconds  float32
	UptimeSeconds float32
	FramesPerSec  float32
	FrameCounter  uint64

	fpsElapsedSeconds float32
}

type Window struct {
	Viewport ViewportSize
	Timing   Timing
}

// Advance updates per-frame timing. FramesPerSec is sampled over a one-second
// window — frames are counted until at least a second has elapsed, then the
// count becomes the reported rate and the counter resets — so the value is
// stable rather than the jittery instantaneous 1/delta.
func Advance(timing *Timing, delta float32) {
	timing.DeltaSeconds = delta
	timing.UptimeSeconds += delta
	timing.FrameCounter++
	timing.fpsElapsedSeconds += delta
	if timing.fpsElapsedSeconds >= 1.0 {
		timing.FramesPerSec = float32(timing.FrameCounter) / timing.fpsElapsedSeconds
		timing.FrameCounter = 0
		timing.fpsElapsedSeconds = 0
	}
}
