package pass

import (
	"math"

	"github.com/go-gl/mathgl/mgl32"

	"indigo/ecs"
	"indigo/render"
	"indigo/transform"
)

// UpdateGizmos runs hit-testing + drag for the gizmo overlay. Must
// run BEFORE [render.UpdatePanOrbitCamera] in the engine schedule so the
// camera controller can read [render.Gizmos.Dragging] / [render.Gizmos.HoverAxis]
// this frame and skip its own input.
func UpdateGizmos(world *ecs.World) {
	if !ecs.HasResource[*render.Gizmos](world) {
		return
	}
	gizmoPtr := ecs.MustResource[*render.Gizmos](world)
	gizmo := *gizmoPtr
	if gizmo == nil {
		return
	}

	input := ecs.MustResource[render.Input](world)
	leftDown := input.LeftDown
	leftJustDown := leftDown && !gizmo.PrevLeftDown
	leftJustUp := !leftDown && gizmo.PrevLeftDown
	gizmo.PrevLeftDown = leftDown

	if gizmo.Mode == render.GizmoNone {
		gizmo.HoverAxis = -1
		gizmo.Dragging = false
		return
	}

	target, ok := SelectedTarget(world)
	if !ok {
		gizmo.HoverAxis = -1
		gizmo.Dragging = false
		return
	}

	global, ok := ecs.Get[transform.GlobalTransform](world, target)
	if !ok {
		gizmo.HoverAxis = -1
		return
	}
	origin := mgl32.Vec3{
		global.Matrix[12], global.Matrix[13], global.Matrix[14],
	}

	camera := ecs.MustResource[render.Camera](world)
	renderer := ecs.MustResource[render.RendererResource](world).Renderer
	aspect := renderer.AspectRatio()
	viewport := [2]float32{
		float32(renderer.Config.Width),
		float32(renderer.Config.Height),
	}
	viewProj := render.CameraViewProjection(camera, aspect)
	tanHalfFov := float32(math.Tan(float64(camera.FovYRadians) * 0.5))

	worldLength := AxisWorldLengthForScreen(origin, viewProj, viewport, tanHalfFov, gizmoAxisScreenLength)
	if worldLength <= 0 {
		gizmo.HoverAxis = -1
		return
	}
	originScreen, ends, valid := GizmoScreenPositions(origin, viewProj, viewport, worldLength)

	mousePos := mgl32.Vec2{input.MousePosition[0], input.MousePosition[1]}
	hoverAxis := -1
	bestDist := gizmoAxisHitThresholdPx
	if gizmo.Mode == render.GizmoRotate {
		for i := 0; i < 3; i++ {
			points, validPts := RingScreenPoints(origin, AxisDirection(uint8(i)), worldLength, viewProj, viewport)
			for sample := 0; sample < RingSegmentCount; sample++ {
				if !validPts[sample] || !validPts[sample+1] {
					continue
				}
				dist := DistanceToSegment(mousePos, points[sample], points[sample+1])
				if dist < bestDist {
					bestDist = dist
					hoverAxis = i
				}
			}
		}
	} else {
		for i := 0; i < 3; i++ {
			if !valid[i] {
				continue
			}
			dist := DistanceToSegment(mousePos, originScreen, ends[i])
			if dist < bestDist {
				bestDist = dist
				hoverAxis = i
			}
		}
	}
	gizmo.HoverAxis = hoverAxis

	if leftJustDown && hoverAxis >= 0 && !gizmo.Dragging {
		local, ok := ecs.Get[transform.LocalTransform](world, target)
		if ok && valid[hoverAxis] {
			axisDir := AxisDirection(uint8(hoverAxis))
			started := false
			initialT := float32(0)
			startVec := mgl32.Vec3{}

			if gizmo.Mode == render.GizmoRotate {
				rayOrigin, rayDir := MouseRay(input.MousePosition[0], input.MousePosition[1], viewProj, viewport)
				if hit, ok := RayPlaneIntersect(rayOrigin, rayDir, origin, axisDir); ok {
					sv := ProjectOntoPlane(hit.Sub(origin), axisDir)
					if sv.Len() > 1e-3 {
						startVec = sv.Normalize()
						started = true
					}
				}
			} else {
				initialT = ParametricTOnSegment(mousePos, originScreen, ends[hoverAxis])
				started = true
			}

			if started {
				gizmo.Dragging = true
				gizmo.DragAxis = uint8(hoverAxis)
				gizmo.DragMode = gizmo.Mode
				gizmo.StartLocal = *local
				gizmo.StartGlobalTrans = [3]float32{origin.X(), origin.Y(), origin.Z()}
				gizmo.AxisWorldDirection = [3]float32{axisDir.X(), axisDir.Y(), axisDir.Z()}
				gizmo.AxisWorldLengthDrag = worldLength
				gizmo.InitialT = initialT
				gizmo.StartWorldVector = [3]float32{startVec.X(), startVec.Y(), startVec.Z()}
			}
		}
	}

	if leftJustUp {
		gizmo.Dragging = false
	}

	if !gizmo.Dragging {
		return
	}

	startWorld := mgl32.Vec3{gizmo.StartGlobalTrans[0], gizmo.StartGlobalTrans[1], gizmo.StartGlobalTrans[2]}
	axisWorld := mgl32.Vec3{gizmo.AxisWorldDirection[0], gizmo.AxisWorldDirection[1], gizmo.AxisWorldDirection[2]}
	endWorld := startWorld.Add(axisWorld.Mul(gizmo.AxisWorldLengthDrag))

	dragOriginScreen, originOK := projectWorld(startWorld, viewProj, viewport)
	dragEndScreen, endOK := projectWorld(endWorld, viewProj, viewport)
	if !originOK || !endOK {
		return
	}
	currentT := ParametricTOnSegment(mousePos, dragOriginScreen, dragEndScreen)
	tDelta := currentT - gizmo.InitialT

	switch gizmo.DragMode {
	case render.GizmoTranslate:
		delta := tDelta * gizmo.AxisWorldLengthDrag
		applyLocalGizmo(world, target, func(t *transform.LocalTransform) {
			*t = gizmo.StartLocal
			t.Translation[0] = gizmo.StartLocal.Translation[0] + axisWorld.X()*delta
			t.Translation[1] = gizmo.StartLocal.Translation[1] + axisWorld.Y()*delta
			t.Translation[2] = gizmo.StartLocal.Translation[2] + axisWorld.Z()*delta
		})
	case render.GizmoScale:
		factor := 1 + tDelta
		if factor < 0.05 {
			factor = 0.05
		}
		applyLocalGizmo(world, target, func(t *transform.LocalTransform) {
			*t = gizmo.StartLocal
			t.Scale[gizmo.DragAxis] = gizmo.StartLocal.Scale[gizmo.DragAxis] * factor
		})
	case render.GizmoRotate:
		rayOrigin, rayDir := MouseRay(input.MousePosition[0], input.MousePosition[1], viewProj, viewport)
		hit, ok := RayPlaneIntersect(rayOrigin, rayDir, startWorld, axisWorld)
		if !ok {
			return
		}
		current := ProjectOntoPlane(hit.Sub(startWorld), axisWorld)
		if current.Len() < 1e-3 {
			return
		}
		currentUnit := current.Normalize()
		startUnit := mgl32.Vec3{gizmo.StartWorldVector[0], gizmo.StartWorldVector[1], gizmo.StartWorldVector[2]}
		cross := startUnit.Cross(currentUnit)
		dot := startUnit.Dot(currentUnit)
		signedAngle := float32(math.Atan2(float64(cross.Dot(axisWorld)), float64(dot)))
		rotation := transform.QuatFromAxisAngle(signedAngle, transform.Vec3{axisWorld.X(), axisWorld.Y(), axisWorld.Z()})
		applyLocalGizmo(world, target, func(t *transform.LocalTransform) {
			*t = gizmo.StartLocal
			t.Rotation = rotation.Mul(gizmo.StartLocal.Rotation).Normalize()
		})
	}
}

func applyLocalGizmo(world *ecs.World, entity ecs.Entity, mutate func(*transform.LocalTransform)) {
	local, ok := ecs.GetMut[transform.LocalTransform](world, entity)
	if !ok {
		return
	}
	mutate(local)
	transform.MarkDirty(world, entity)
}

// AddGizmoPass binds [NewGizmoPass] into the render graph writing
// into colorID (typically fxaa_output so the overlay lands on top
// of the antialiased scene before present).
func AddGizmoPass(renderer *render.Renderer, colorID render.ResourceID) (*render.Pass, error) {
	pass, err := NewGizmoPass(renderer.Device, renderer.SurfaceFormat, renderer.AspectRatio)
	if err != nil {
		return nil, err
	}
	if err := renderer.Graph.AddPass(pass, []render.SlotBinding{
		{Slot: "color", ResourceID: colorID},
	}); err != nil {
		return nil, err
	}
	return pass, nil
}
