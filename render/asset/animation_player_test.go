package asset

import (
	"math"
	"testing"

	"indigo/transform"
)

func TestFindKeyframeBeforeFirst(t *testing.T) {
	times := []float32{0.5, 1.0, 1.5}
	a, b, f := findKeyframe(times, 0)
	if a != 0 || b != 0 || f != 0 {
		t.Errorf("before first = (%d,%d,%f), want (0,0,0)", a, b, f)
	}
}

func TestFindKeyframeAfterLast(t *testing.T) {
	times := []float32{0.5, 1.0, 1.5}
	a, b, f := findKeyframe(times, 2.0)
	if a != 2 || b != 2 || f != 0 {
		t.Errorf("after last = (%d,%d,%f), want (2,2,0)", a, b, f)
	}
}

func TestFindKeyframeMidSpan(t *testing.T) {
	times := []float32{0.0, 1.0, 2.0}
	a, b, f := findKeyframe(times, 0.5)
	if a != 0 || b != 1 {
		t.Errorf("mid span keys = (%d,%d), want (0,1)", a, b)
	}
	if math.Abs(float64(f-0.5)) > 1e-5 {
		t.Errorf("mid span factor = %f, want 0.5", f)
	}
}

func TestFindKeyframeExactBoundary(t *testing.T) {
	times := []float32{0.0, 1.0, 2.0}
	a, b, _ := findKeyframe(times, 1.0)
	// Exact match on times[1]: returns the span starting at index 1
	// with factor 0 (or the previous span with factor 1; both produce
	// the same sampled value).
	if a < 0 || b > 2 {
		t.Errorf("exact-boundary keys out of range: (%d,%d)", a, b)
	}
}

func TestSampleVec3Linear(t *testing.T) {
	outputs := [][3]float32{{0, 0, 0}, {10, 20, 30}}
	got := sampleVec3(outputs, 0, 1, 0.5, InterpolationLinear)
	want := transform.Vec3{5, 10, 15}
	if got != want {
		t.Errorf("lerp half = %v, want %v", got, want)
	}
}

func TestSampleVec3Step(t *testing.T) {
	outputs := [][3]float32{{0, 0, 0}, {10, 20, 30}}
	got := sampleVec3(outputs, 0, 1, 0.9, InterpolationStep)
	want := transform.Vec3{0, 0, 0}
	if got != want {
		t.Errorf("step at factor 0.9 = %v, want first keyframe %v", got, want)
	}
}

func TestSampleQuatLinearSlerp(t *testing.T) {
	// Identity to 90 deg yaw around +Y. Halfway through should be a
	// 45 deg yaw quaternion: (0, sin(22.5), 0, cos(22.5)) ... wait
	// 90 deg yaw quaternion is (0, sin(45), 0, cos(45)) = (0, .7071, 0, .7071).
	// Slerp halfway should give us 45 deg yaw: (0, sin(22.5), 0, cos(22.5)).
	outputs := [][4]float32{
		{0, 0, 0, 1},
		{0, float32(math.Sin(math.Pi / 4)), 0, float32(math.Cos(math.Pi / 4))},
	}
	got := sampleQuat(outputs, 0, 1, 0.5, InterpolationLinear)
	expectedY := float32(math.Sin(math.Pi / 8))
	expectedW := float32(math.Cos(math.Pi / 8))
	if math.Abs(float64(got.V[1]-expectedY)) > 1e-4 || math.Abs(float64(got.W-expectedW)) > 1e-4 {
		t.Errorf("slerp halfway = (V=%v, W=%f), want V.y=%f W=%f", got.V, got.W, expectedY, expectedW)
	}
}

func TestAdvancePlayerLoops(t *testing.T) {
	p := &AnimationPlayer{
		Clips:       []AnimationClip{{Duration: 1.0}},
		CurrentClip: 0,
		Time:        0.8,
		Speed:       1,
		Looping:     true,
		Playing:     true,
	}
	advancePlayer(p, 0.5)
	if math.Abs(float64(p.Time-0.3)) > 1e-5 {
		t.Errorf("looped time = %f, want 0.3", p.Time)
	}
	if !p.Playing {
		t.Error("looped player should still be playing")
	}
}

func TestAdvancePlayerClampsAndStops(t *testing.T) {
	p := &AnimationPlayer{
		Clips:       []AnimationClip{{Duration: 1.0}},
		CurrentClip: 0,
		Time:        0.8,
		Speed:       1,
		Looping:     false,
		Playing:     true,
	}
	advancePlayer(p, 0.5)
	if p.Time != 1.0 {
		t.Errorf("clamped time = %f, want 1.0", p.Time)
	}
	if p.Playing {
		t.Error("non-looping player should stop at clip end")
	}
}
