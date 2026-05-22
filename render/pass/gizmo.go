package pass

import (
	"math"

	"github.com/go-gl/mathgl/mgl32"

	"indigo/ecs"
	"indigo/render"
)

const GizmoAxisCount = 3

// gizmoAxisScreenLength is constant in screen space, not world
// space, so the gizmo keeps the same pixel size regardless of
// camera distance.
const gizmoAxisScreenLength float32 = 110

const (
	gizmoAxisThicknessPx      float32 = 3.5
	gizmoAxisHoverThicknessPx float32 = 6
	gizmoAxisHitThresholdPx   float32 = 9
	gizmoOriginRadiusPx       float32 = 4
)

// AxisColor returns red for X, green for Y, blue for Z.
func AxisColor(axis uint8) [4]float32 {
	switch axis {
	case 0:
		return [4]float32{0.95, 0.30, 0.30, 1}
	case 1:
		return [4]float32{0.40, 0.95, 0.30, 1}
	default:
		return [4]float32{0.30, 0.55, 0.95, 1}
	}
}

var AxisHoverColor = [4]float32{1.0, 0.85, 0.20, 1}

func AxisDirection(axis uint8) mgl32.Vec3 {
	switch axis {
	case 0:
		return mgl32.Vec3{1, 0, 0}
	case 1:
		return mgl32.Vec3{0, 1, 0}
	default:
		return mgl32.Vec3{0, 0, 1}
	}
}

// SelectedTarget returns the first entity carrying a [render.Selected]
// tag, or the zero entity if nothing is selected.
func SelectedTarget(world *ecs.World) (ecs.Entity, bool) {
	mask := ecs.MustMaskOf[render.Selected](world)
	var found ecs.Entity
	hasFound := false
	world.ForEach(mask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		if !hasFound {
			found = entity
			hasFound = true
		}
	})
	return found, hasFound
}

// GizmoScreenPositions projects the entity origin and the three
// axis endpoints into screen space. The bool per axis is false if
// the endpoint landed behind the camera.
func GizmoScreenPositions(origin mgl32.Vec3, viewProj mgl32.Mat4, viewport [2]float32, worldLength float32) (mgl32.Vec2, [3]mgl32.Vec2, [3]bool) {
	originScreen, originValid := projectWorld(origin, viewProj, viewport)
	var ends [3]mgl32.Vec2
	var valid [3]bool
	if !originValid {
		return originScreen, ends, valid
	}
	for axis := 0; axis < 3; axis++ {
		dir := AxisDirection(uint8(axis))
		end := mgl32.Vec3{
			origin.X() + dir.X()*worldLength,
			origin.Y() + dir.Y()*worldLength,
			origin.Z() + dir.Z()*worldLength,
		}
		ends[axis], valid[axis] = projectWorld(end, viewProj, viewport)
	}
	return originScreen, ends, valid
}

// AxisWorldLengthForScreen returns the world-space distance that
// projects to screenLength pixels at the depth of origin under a
// perspective projection with half-vertical-FOV tanHalfFovY.
// Mirrors nightshade's world_length_for_screen_pixels: a 1-unit
// world segment at view distance d covers viewport_height /
// (2*d*tanHalfFovY) pixels, so the inverse scales screenLength by
// 2*d*tanHalfFovY/viewport_height.
func AxisWorldLengthForScreen(origin mgl32.Vec3, viewProj mgl32.Mat4, viewport [2]float32, tanHalfFovY, screenLength float32) float32 {
	clip := viewProj.Mul4x1(mgl32.Vec4{origin.X(), origin.Y(), origin.Z(), 1})
	w := clip.W()
	if w <= 0 {
		return 0
	}
	return screenLength * 2 * w * tanHalfFovY / viewport[1]
}

// ProjectWorld is the public form of projectWorld for callers
// outside the package who need the same perspective-divide +
// pixel-mapping routine the gizmo overlay uses.
func ProjectWorld(p mgl32.Vec3, viewProj mgl32.Mat4, viewport [2]float32) (mgl32.Vec2, bool) {
	return projectWorld(p, viewProj, viewport)
}

func projectWorld(p mgl32.Vec3, viewProj mgl32.Mat4, viewport [2]float32) (mgl32.Vec2, bool) {
	clip := viewProj.Mul4x1(mgl32.Vec4{p.X(), p.Y(), p.Z(), 1})
	if clip.W() <= 0 {
		return mgl32.Vec2{}, false
	}
	ndcX := clip.X() / clip.W()
	ndcY := clip.Y() / clip.W()
	screenX := (ndcX*0.5 + 0.5) * viewport[0]
	screenY := (1 - (ndcY*0.5 + 0.5)) * viewport[1]
	return mgl32.Vec2{screenX, screenY}, true
}

// ParametricTOnSegment returns the signed parametric position of p
// projected perpendicularly onto the infinite line through a..b
// (NOT clamped to [0,1]). a corresponds to t=0, b to t=1. Mirrors
// nightshade's parametric_t_on_segment.
func ParametricTOnSegment(p, a, b mgl32.Vec2) float32 {
	ab := b.Sub(a)
	lengthSq := ab.Dot(ab)
	if lengthSq < 1e-6 {
		return 0
	}
	return p.Sub(a).Dot(ab) / lengthSq
}

// DistanceToSegment returns the perpendicular distance from point
// p to the line segment a..b, in the same coordinate space as the
// inputs. Used by the gizmo CPU hit test to decide whether the
// mouse is over an axis line.
func DistanceToSegment(p, a, b mgl32.Vec2) float32 {
	ab := b.Sub(a)
	ap := p.Sub(a)
	lengthSq := ab.Dot(ab)
	if lengthSq < 1e-6 {
		return ap.Len()
	}
	t := ap.Dot(ab) / lengthSq
	if t < 0 {
		t = 0
	} else if t > 1 {
		t = 1
	}
	proj := a.Add(ab.Mul(t))
	return p.Sub(proj).Len()
}

// RingSegmentCount is the number of straight-line chords the rotate
// gizmo decomposes each ring into. 64 is enough to look smooth at
// typical sizes without flooding the instance buffer.
const RingSegmentCount = 64

// RingBasis picks two world-space basis vectors that span the plane
// perpendicular to axis. The first is "right-ish," the second is
// "up-ish"; together with axis they form a right-handed frame.
// Adapted directly from nightshade's ring_basis: when axis is near
// world up, use world X as the seed instead.
func RingBasis(axis mgl32.Vec3) (mgl32.Vec3, mgl32.Vec3) {
	worldUp := mgl32.Vec3{0, 1, 0}
	if math.Abs(float64(axis.Y())) > 0.9 {
		worldUp = mgl32.Vec3{1, 0, 0}
	}
	u := worldUp.Cross(axis).Normalize()
	v := axis.Cross(u).Normalize()
	return u, v
}

// MouseRay casts a ray from the camera through the given pixel and
// returns its world-space origin (near plane) and direction. Uses
// the inverse of viewProj to unproject the NDC point.
func MouseRay(mouseX, mouseY float32, viewProj mgl32.Mat4, viewport [2]float32) (mgl32.Vec3, mgl32.Vec3) {
	invVP := viewProj.Inv()
	ndcX := mouseX/viewport[0]*2 - 1
	ndcY := 1 - mouseY/viewport[1]*2
	nearClip := invVP.Mul4x1(mgl32.Vec4{ndcX, ndcY, -1, 1})
	farClip := invVP.Mul4x1(mgl32.Vec4{ndcX, ndcY, 1, 1})
	if nearClip.W() == 0 || farClip.W() == 0 {
		return mgl32.Vec3{}, mgl32.Vec3{0, 0, -1}
	}
	near := mgl32.Vec3{nearClip.X() / nearClip.W(), nearClip.Y() / nearClip.W(), nearClip.Z() / nearClip.W()}
	far := mgl32.Vec3{farClip.X() / farClip.W(), farClip.Y() / farClip.W(), farClip.Z() / farClip.W()}
	return near, far.Sub(near).Normalize()
}

// RayPlaneIntersect returns the world-space point where the ray
// (origin, direction) crosses the plane defined by planePoint and
// planeNormal. The bool is false when the ray is parallel to the
// plane or the intersection is behind the origin.
func RayPlaneIntersect(origin, direction, planePoint, planeNormal mgl32.Vec3) (mgl32.Vec3, bool) {
	denom := direction.Dot(planeNormal)
	if math.Abs(float64(denom)) < 1e-6 {
		return mgl32.Vec3{}, false
	}
	t := planePoint.Sub(origin).Dot(planeNormal) / denom
	if t < 0 {
		return mgl32.Vec3{}, false
	}
	return origin.Add(direction.Mul(t)), true
}

// ProjectOntoPlane removes the component of v parallel to
// planeNormal so the result lies in the plane.
func ProjectOntoPlane(v, planeNormal mgl32.Vec3) mgl32.Vec3 {
	return v.Sub(planeNormal.Mul(v.Dot(planeNormal)))
}

// RingScreenPoints samples a world-space circle in the plane
// perpendicular to axis, of the given radius and centered at
// origin, and projects each sample into screen space. Returns the
// per-sample screen position plus a flag for whether the projection
// landed in front of the camera. RingSegmentCount + 1 entries are
// produced so callers can iterate consecutive pairs to form
// segments (the last point repeats the first).
func RingScreenPoints(origin mgl32.Vec3, axis mgl32.Vec3, radius float32, viewProj mgl32.Mat4, viewport [2]float32) ([RingSegmentCount + 1]mgl32.Vec2, [RingSegmentCount + 1]bool) {
	u, v := RingBasis(axis)
	var points [RingSegmentCount + 1]mgl32.Vec2
	var valid [RingSegmentCount + 1]bool
	for sample := 0; sample <= RingSegmentCount; sample++ {
		theta := float32(sample) / float32(RingSegmentCount) * 2 * float32(math.Pi)
		cos := float32(math.Cos(float64(theta)))
		sin := float32(math.Sin(float64(theta)))
		point := mgl32.Vec3{
			origin.X() + (u.X()*cos+v.X()*sin)*radius,
			origin.Y() + (u.Y()*cos+v.Y()*sin)*radius,
			origin.Z() + (u.Z()*cos+v.Z()*sin)*radius,
		}
		points[sample], valid[sample] = projectWorld(point, viewProj, viewport)
	}
	return points, valid
}
