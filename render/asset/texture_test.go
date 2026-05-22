package asset

import "testing"

func TestMipLevelCount(t *testing.T) {
	cases := []struct {
		w, h, want uint32
	}{
		{1, 1, 1},
		{2, 1, 2},
		{2, 2, 2},
		{4, 4, 3},
		{8, 1, 4},
		{16, 16, 5},
		{1024, 1024, 11},
		{1024, 512, 11},
		{3, 5, 3},
	}
	for _, tc := range cases {
		got := mipLevelCount(tc.w, tc.h)
		if got != tc.want {
			t.Errorf("mipLevelCount(%d,%d) = %d, want %d", tc.w, tc.h, got, tc.want)
		}
	}
}

func TestDownsampleRGBAEven(t *testing.T) {
	src := []byte{
		10, 20, 30, 40, 50, 60, 70, 80,
		90, 100, 110, 120, 130, 140, 150, 160,
	}
	dst, w, h := downsampleRGBA(src, 2, 2)
	if w != 1 || h != 1 {
		t.Fatalf("dims = %dx%d, want 1x1", w, h)
	}
	if len(dst) != 4 {
		t.Fatalf("len = %d, want 4", len(dst))
	}
	wantR := byte((10 + 50 + 90 + 130 + 2) >> 2)
	wantG := byte((20 + 60 + 100 + 140 + 2) >> 2)
	wantB := byte((30 + 70 + 110 + 150 + 2) >> 2)
	wantA := byte((40 + 80 + 120 + 160 + 2) >> 2)
	if dst[0] != wantR || dst[1] != wantG || dst[2] != wantB || dst[3] != wantA {
		t.Errorf("got %v, want [%d %d %d %d]", dst, wantR, wantG, wantB, wantA)
	}
}

func TestDownsampleRGBAOdd(t *testing.T) {
	pixels := make([]byte, 3*3*4)
	for i := range pixels {
		pixels[i] = byte(i + 1)
	}
	dst, w, h := downsampleRGBA(pixels, 3, 3)
	if w != 1 || h != 1 {
		t.Fatalf("dims = %dx%d, want 1x1", w, h)
	}
	if len(dst) != 4 {
		t.Fatalf("len = %d, want 4", len(dst))
	}
}

func TestDownsampleRGBAClampsToOne(t *testing.T) {
	src := []byte{255, 255, 255, 255}
	dst, w, h := downsampleRGBA(src, 1, 1)
	if w != 1 || h != 1 {
		t.Fatalf("1x1 source: dims = %dx%d, want 1x1", w, h)
	}
	if dst[0] != 255 {
		t.Errorf("dst = %v, want [255 255 255 255]", dst)
	}
}
