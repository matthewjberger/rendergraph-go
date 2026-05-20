package render

import (
	"math"

	"github.com/go-gl/mathgl/mgl32"

	"indigo/ecs"
	"indigo/transform"
	"indigo/ui"
	"indigo/window"
)

// PanOrbitController is the data half of an arc-ball / pan-orbit
// camera controller. Held as a typed resource on the engine world.
// [UpdatePanOrbitCamera] reads it (together with [Input] and the
// surface size) and writes the result to the [Camera] resource.
//
// Two sets of fields: the current pose (Focus / Radius / Yaw /
// Pitch) the camera renders from this frame, and the target pose
// the controller is interpolating toward. The targets are what
// input mutates; smoothing pulls the current toward them.
//
// Mirrors nightshade's PanOrbitCamera component, with the gamepad /
// touch / modifier-key plumbing dropped: orbit on left drag, pan on
// right drag, zoom on scroll.
type PanOrbitController struct {
	Focus  transform.Vec3
	Radius float32
	Yaw    float32
	Pitch  float32

	TargetFocus  transform.Vec3
	TargetRadius float32
	TargetYaw    float32
	TargetPitch  float32

	OrbitSensitivity float32
	PanSensitivity   float32
	ZoomSensitivity  float32

	OrbitSmoothness float32
	PanSmoothness   float32
	ZoomSmoothness  float32

	PitchLower float32
	PitchUpper float32
	ZoomLower  float32
	ZoomUpper  float32
}

// DefaultPanOrbitController returns a controller orbiting the world
// origin at radius 6, with limits and sensitivities tuned for the
// engine demo grid.
func DefaultPanOrbitController() PanOrbitController {
	pitch := float32(math.Pi / 8)
	return PanOrbitController{
		Focus:  transform.Vec3{0, 0, 0},
		Radius: 8,
		Yaw:    0,
		Pitch:  pitch,

		TargetFocus:  transform.Vec3{0, 0, 0},
		TargetRadius: 8,
		TargetYaw:    0,
		TargetPitch:  pitch,

		OrbitSensitivity: 1.0,
		PanSensitivity:   1.0,
		ZoomSensitivity:  1.0,

		OrbitSmoothness: 0.1,
		PanSmoothness:   0.02,
		ZoomSmoothness:  0.1,

		PitchLower: -float32(math.Pi/2) + 0.01,
		PitchUpper: float32(math.Pi/2) - 0.01,
		ZoomLower:  0.5,
		ZoomUpper:  100,
	}
}

// UpdatePanOrbitCamera consumes a frame of [Input] from the engine
// world, advances the [PanOrbitController]'s target pose (orbit, pan,
// zoom), smooths the current pose toward the target, and writes the
// derived eye + target back to the [Camera] resource. Reads delta and
// viewport from the [window.Window] resource so the system signature
// is (*World), the shape an [ecs.Schedule] expects.
func UpdatePanOrbitCamera(world *ecs.World) {
	w := ecs.Resource[window.Window](world)
	controller := ecs.Resource[PanOrbitController](world)
	camera := ecs.Resource[Camera](world)
	input := ecs.Resource[Input](world)

	width := float32(w.Viewport.Width)
	height := float32(w.Viewport.Height)
	if width <= 0 || height <= 0 {
		return
	}
	viewport := transform.Vec2{width, height}
	yFov := camera.FovYRadians
	delta := w.Timing.DeltaSeconds

	if !mouseCapturedByOverlay(world) {
		applyPanOrbitInput(controller, input, viewport, yFov)
	}
	applyPanOrbitSmoothing(controller, delta)
	writePanOrbitCamera(controller, camera)
}

// mouseCapturedByOverlay reports whether a gizmo handle or HUD
// widget is receiving this frame's pointer. Pan-orbit skips input
// in that case so left-button drag of a handle doesn't also orbit.
func mouseCapturedByOverlay(world *ecs.World) bool {
	if ecs.HasResource[*Gizmos](world) {
		g := *ecs.Resource[*Gizmos](world)
		if g != nil && (g.Dragging || g.HoverAxis >= 0) {
			return true
		}
	}
	if ecs.HasResource[ui.WorldRef](world) {
		uiWorld := ecs.Resource[ui.WorldRef](world).World
		if uiWorld != nil && ecs.HasResource[ui.PointerState](uiWorld) {
			pointer := ecs.Resource[ui.PointerState](uiWorld)
			if pointer.OverUI {
				return true
			}
		}
	}
	return false
}

func applyPanOrbitInput(controller *PanOrbitController, input *Input, viewport transform.Vec2, yFov float32) {
	wantsOrbit := input.LeftDown && !input.RightDown
	wantsPan := input.RightDown && !input.LeftDown

	if wantsOrbit {
		twoPi := float32(math.Pi * 2)
		deltaX := (input.MouseDelta[0] / viewport[0]) * twoPi
		deltaY := (input.MouseDelta[1] / viewport[1]) * float32(math.Pi)
		controller.TargetYaw -= deltaX * controller.OrbitSensitivity
		controller.TargetPitch += deltaY * controller.OrbitSensitivity
		if controller.TargetPitch < controller.PitchLower {
			controller.TargetPitch = controller.PitchLower
		}
		if controller.TargetPitch > controller.PitchUpper {
			controller.TargetPitch = controller.PitchUpper
		}
	}

	if wantsPan {
		_, rotation := computePanOrbitPose(
			controller.TargetFocus,
			controller.TargetYaw,
			controller.TargetPitch,
			controller.TargetRadius,
		)
		right := rotation.Rotate(mgl32.Vec3{1, 0, 0})
		up := rotation.Rotate(mgl32.Vec3{0, 1, 0})

		worldPerPixel := (controller.TargetRadius * 2 * float32(math.Tan(float64(yFov*0.5)))) / viewport[1]
		scale := worldPerPixel * controller.PanSensitivity
		panRight := right.Mul(-input.MouseDelta[0] * scale)
		panUp := up.Mul(input.MouseDelta[1] * scale)
		controller.TargetFocus = controller.TargetFocus.Add(panRight).Add(panUp)
	}

	if input.Wheel != 0 {
		zoomDelta := -input.Wheel * controller.TargetRadius * 0.2 * controller.ZoomSensitivity
		controller.TargetRadius += zoomDelta
		if controller.TargetRadius < controller.ZoomLower {
			controller.TargetRadius = controller.ZoomLower
		}
		if controller.TargetRadius > controller.ZoomUpper {
			controller.TargetRadius = controller.ZoomUpper
		}
	}
}

func applyPanOrbitSmoothing(controller *PanOrbitController, delta float32) {
	controller.Yaw = lerpAndSnap(controller.Yaw, controller.TargetYaw, controller.OrbitSmoothness, delta)
	controller.Pitch = lerpAndSnap(controller.Pitch, controller.TargetPitch, controller.OrbitSmoothness, delta)
	controller.Radius = lerpAndSnap(controller.Radius, controller.TargetRadius, controller.ZoomSmoothness, delta)
	controller.Focus = lerpAndSnapVec3(controller.Focus, controller.TargetFocus, controller.PanSmoothness, delta)
}

func writePanOrbitCamera(controller *PanOrbitController, camera *Camera) {
	position, _ := computePanOrbitPose(controller.Focus, controller.Yaw, controller.Pitch, controller.Radius)
	camera.Eye = position
	camera.Target = controller.Focus
	camera.Up = mgl32.Vec3{0, 1, 0}
}

// computePanOrbitPose returns the camera position and orientation
// quaternion for given orbit parameters. Mirrors nightshade's
// compute_pan_orbit_transform.
func computePanOrbitPose(focus transform.Vec3, yaw, pitch, radius float32) (transform.Vec3, transform.Quat) {
	yawQuat := mgl32.QuatRotate(yaw, mgl32.Vec3{0, 1, 0})
	pitchQuat := mgl32.QuatRotate(-pitch, mgl32.Vec3{1, 0, 0})
	rotation := yawQuat.Mul(pitchQuat)
	offset := rotation.Rotate(mgl32.Vec3{0, 0, radius})
	position := focus.Add(offset)
	return position, rotation
}

const panOrbitSnapEpsilon = 0.001

func lerpAndSnap(from, to, smoothness, delta float32) float32 {
	t := float32(math.Pow(float64(smoothness), 7))
	result := from + (to-from)*(1-float32(math.Pow(float64(t), float64(delta))))
	if smoothness < 1 && abs(result-to) < panOrbitSnapEpsilon {
		return to
	}
	return result
}

func lerpAndSnapVec3(from, to transform.Vec3, smoothness, delta float32) transform.Vec3 {
	t := float32(math.Pow(float64(smoothness), 7))
	factor := 1 - float32(math.Pow(float64(t), float64(delta)))
	result := from.Add(to.Sub(from).Mul(factor))
	if smoothness < 1 && result.Sub(to).Len() < panOrbitSnapEpsilon {
		return to
	}
	return result
}

func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
