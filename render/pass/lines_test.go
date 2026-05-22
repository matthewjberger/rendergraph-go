package pass

import (
	"testing"

	"github.com/go-gl/mathgl/mgl32"

	"indigo/render/asset"
	"indigo/transform"
)

func TestLinesAddSegment(t *testing.T) {
	l := &Lines{}
	l.AddSegment([3]float32{1, 2, 3}, [3]float32{4, 5, 6}, [4]float32{1, 0, 0, 1})
	if len(l.Segments) != 1 {
		t.Fatalf("len = %d, want 1", len(l.Segments))
	}
	s := l.Segments[0]
	if s.Start != [4]float32{1, 2, 3, 1} {
		t.Errorf("start = %v, want [1 2 3 1]", s.Start)
	}
	if s.End != [4]float32{4, 5, 6, 1} {
		t.Errorf("end = %v, want [4 5 6 1]", s.End)
	}
	if s.Color != [4]float32{1, 0, 0, 1} {
		t.Errorf("color = %v, want [1 0 0 1]", s.Color)
	}
}

func TestLinesAddBoxEmitsTwelveEdges(t *testing.T) {
	l := &Lines{}
	bounds := asset.BoundingVolume{Min: [3]float32{-1, -1, -1}, Max: [3]float32{1, 1, 1}}
	identity := transform.Mat4(mgl32.Ident4())
	l.AddBox(bounds, &identity, [4]float32{0, 1, 0, 1})
	if len(l.Segments) != 12 {
		t.Fatalf("len = %d, want 12 (one per cube edge)", len(l.Segments))
	}
	for i, seg := range l.Segments {
		if seg.Color != [4]float32{0, 1, 0, 1} {
			t.Errorf("segment %d color = %v, want [0 1 0 1]", i, seg.Color)
		}
		for axis := 0; axis < 3; axis++ {
			for _, v := range []float32{seg.Start[axis], seg.End[axis]} {
				if v != -1 && v != 1 {
					t.Errorf("segment %d axis %d value %f: every cube-edge endpoint must sit on a unit corner", i, axis, v)
				}
			}
		}
	}
}

func TestLinesAddBoxRespectsTransform(t *testing.T) {
	l := &Lines{}
	bounds := asset.BoundingVolume{Min: [3]float32{0, 0, 0}, Max: [3]float32{1, 1, 1}}
	translated := transform.Mat4(mgl32.Translate3D(10, 20, 30))
	l.AddBox(bounds, &translated, [4]float32{1, 1, 1, 1})
	if len(l.Segments) != 12 {
		t.Fatalf("len = %d, want 12", len(l.Segments))
	}
	// Every endpoint must land in the translated cube's corner set
	// {(10..11), (20..21), (30..31)} since the source bounds are a
	// unit cube at origin.
	for i, seg := range l.Segments {
		for axis, base := range []float32{10, 20, 30} {
			for _, v := range []float32{seg.Start[axis], seg.End[axis]} {
				if v != base && v != base+1 {
					t.Errorf("segment %d axis %d value %f not at translated corner", i, axis, v)
				}
			}
		}
	}
}
