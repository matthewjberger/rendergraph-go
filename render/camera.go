package render

import "github.com/go-gl/mathgl/mgl32"

// Camera is the active view + projection settings. Stored as a typed
// resource on the ECS world; the mesh pass looks it up in Prepare and
// composes view × projection × global-transform per instance.
//
// Kept deliberately simple for the slice: a look-at view + a
// perspective projection driven by aspect/fov/near/far. nightshade has
// a richer ecs/camera module (multiple cameras, viewports, controllers);
// we'll grow into that shape as needed.
type Camera struct {
	Eye    mgl32.Vec3
	Target mgl32.Vec3
	Up     mgl32.Vec3

	FovYRadians float32
	Near        float32
	Far         float32
}

// DefaultCamera returns a camera positioned to look down -Z at the
// origin, with a 60° vertical FOV. The mesh pass reads aspect from the
// renderer each frame, so aspect isn't stored on the camera.
func DefaultCamera() Camera {
	return Camera{
		Eye:         mgl32.Vec3{0, 2, 6},
		Target:      mgl32.Vec3{0, 0, 0},
		Up:          mgl32.Vec3{0, 1, 0},
		FovYRadians: mgl32.DegToRad(60),
		Near:        0.1,
		Far:         1000.0,
	}
}

// CameraView returns the camera's view matrix.
func CameraView(c *Camera) mgl32.Mat4 {
	return mgl32.LookAtV(c.Eye, c.Target, c.Up)
}

// CameraProjection returns the camera's perspective projection
// composed with the OpenGL → wgpu depth-range remap.
func CameraProjection(c *Camera, aspect float32) mgl32.Mat4 {
	return PerspectiveZO(c.FovYRadians, aspect, c.Near, c.Far)
}

// CameraViewProjection returns CameraProjection × CameraView.
func CameraViewProjection(c *Camera, aspect float32) mgl32.Mat4 {
	return CameraProjection(c, aspect).Mul4(CameraView(c))
}
