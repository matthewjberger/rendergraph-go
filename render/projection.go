package render

import (
	"math"

	"github.com/go-gl/mathgl/mgl32"
)

// PerspectiveZO builds a reverse-Z perspective matrix for a 0..1 clip-space
func PerspectiveZO(fovY, aspect, near, far float32) mgl32.Mat4 {
	f := float32(1.0 / math.Tan(float64(fovY)/2.0))
	depth := far - near
	return mgl32.Mat4{
		f / aspect, 0, 0, 0,
		0, f, 0, 0,
		0, 0, near / depth, -1,
		0, 0, near * far / depth, 0,
	}
}
