package main

import (
	"math"

	"github.com/go-gl/mathgl/mgl32"

	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/internal/render/pass"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/transform"
)

const (
	lightGizmoSphereSegments = 32
	lightGizmoConeSegments   = 24
	lightGizmoArrowLength    = 1.5
	lightGizmoArrowHeadSize  = 0.25
)

func drawLightGizmos(engine *ecs.World) {
	if !ecs.HasResource[pass.LinesResource](engine) {
		return
	}
	lines := ecs.MustResource[pass.LinesResource](engine).Lines
	selectedMask := ecs.MustMaskOf[render.Selected](engine)
	engine.ForEach(selectedMask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		light, ok := ecs.Get[render.Light](engine, entity)
		if !ok {
			return
		}
		global, ok := ecs.Get[transform.GlobalTransform](engine, entity)
		if !ok {
			return
		}
		position := mgl32.Vec3{global.Matrix[12], global.Matrix[13], global.Matrix[14]}
		right := mgl32.Vec3{global.Matrix[0], global.Matrix[1], global.Matrix[2]}.Normalize()
		up := mgl32.Vec3{global.Matrix[4], global.Matrix[5], global.Matrix[6]}.Normalize()
		forward := mgl32.Vec3{-global.Matrix[8], -global.Matrix[9], -global.Matrix[10]}.Normalize()
		color := [4]float32{light.Color[0], light.Color[1], light.Color[2], 1}
		switch light.Type {
		case render.LightTypeDirectional:
			directionalLightLines(lines, position, forward, right, up, color)
		case render.LightTypePoint:
			pointLightLines(lines, position, max32(light.Range, 0.05), color)
		case render.LightTypeSpot:
			spotLightLines(lines, position, forward, right, up,
				max32(light.OuterConeAngle, 0.01), max32(light.Range, 0.05), color)
		}
	})
}

func directionalLightLines(lines *pass.Lines, position, forward, right, up mgl32.Vec3, color [4]float32) {
	direction := forward
	arrowEnd := position.Add(direction.Mul(lightGizmoArrowLength))
	lines.AddSegment(vecToArr(position), vecToArr(arrowEnd), color)
	headBack := arrowEnd.Sub(direction.Mul(lightGizmoArrowHeadSize))
	rightHalf := right.Mul(lightGizmoArrowHeadSize * 0.5)
	upHalf := up.Mul(lightGizmoArrowHeadSize * 0.5)
	lines.AddSegment(vecToArr(arrowEnd), vecToArr(headBack.Add(rightHalf)), color)
	lines.AddSegment(vecToArr(arrowEnd), vecToArr(headBack.Sub(rightHalf)), color)
	lines.AddSegment(vecToArr(arrowEnd), vecToArr(headBack.Add(upHalf)), color)
	lines.AddSegment(vecToArr(arrowEnd), vecToArr(headBack.Sub(upHalf)), color)
	const parallelOffset = float32(0.6)
	offsets := [4][2]float32{{parallelOffset, 0}, {-parallelOffset, 0}, {0, parallelOffset}, {0, -parallelOffset}}
	for _, off := range offsets {
		offset := right.Mul(off[0]).Add(up.Mul(off[1]))
		start := position.Add(offset)
		end := start.Add(direction.Mul(lightGizmoArrowLength))
		lines.AddSegment(vecToArr(start), vecToArr(end), color)
	}
}

func pointLightLines(lines *pass.Lines, center mgl32.Vec3, radius float32, color [4]float32) {
	axisX := mgl32.Vec3{1, 0, 0}
	axisY := mgl32.Vec3{0, 1, 0}
	axisZ := mgl32.Vec3{0, 0, 1}
	pushCircle(lines, center, axisX, axisY, radius, color)
	pushCircle(lines, center, axisY, axisZ, radius, color)
	pushCircle(lines, center, axisX, axisZ, radius, color)
	lines.AddSegment(vecToArr(center.Sub(axisX.Mul(radius))), vecToArr(center.Add(axisX.Mul(radius))), color)
	lines.AddSegment(vecToArr(center.Sub(axisY.Mul(radius))), vecToArr(center.Add(axisY.Mul(radius))), color)
	lines.AddSegment(vecToArr(center.Sub(axisZ.Mul(radius))), vecToArr(center.Add(axisZ.Mul(radius))), color)
}

func spotLightLines(lines *pass.Lines, apex, forward, right, up mgl32.Vec3, outerAngle, rangeUnits float32, color [4]float32) {
	direction := forward
	coneHeight := rangeUnits
	coneRadius := coneHeight * float32(math.Tan(float64(outerAngle)))
	coneCenter := apex.Add(direction.Mul(coneHeight))
	prev := coneRimPoint(coneCenter, right, up, coneRadius, 0)
	for index := 1; index <= lightGizmoConeSegments; index++ {
		theta := float32(index) / float32(lightGizmoConeSegments) * 2 * float32(math.Pi)
		next := coneRimPoint(coneCenter, right, up, coneRadius, theta)
		lines.AddSegment(vecToArr(prev), vecToArr(next), color)
		prev = next
	}
	const spokes = 4
	for index := 0; index < spokes; index++ {
		theta := float32(index) / float32(spokes) * 2 * float32(math.Pi)
		rim := coneRimPoint(coneCenter, right, up, coneRadius, theta)
		lines.AddSegment(vecToArr(apex), vecToArr(rim), color)
	}
	lines.AddSegment(vecToArr(apex), vecToArr(coneCenter), color)
}

func pushCircle(lines *pass.Lines, center, axisA, axisB mgl32.Vec3, radius float32, color [4]float32) {
	prev := center.Add(axisA.Mul(radius))
	for index := 1; index <= lightGizmoSphereSegments; index++ {
		theta := float32(index) / float32(lightGizmoSphereSegments) * 2 * float32(math.Pi)
		next := center.Add(axisA.Mul(float32(math.Cos(float64(theta))) * radius)).Add(axisB.Mul(float32(math.Sin(float64(theta))) * radius))
		lines.AddSegment(vecToArr(prev), vecToArr(next), color)
		prev = next
	}
}

func coneRimPoint(center, right, up mgl32.Vec3, radius, theta float32) mgl32.Vec3 {
	return center.Add(right.Mul(float32(math.Cos(float64(theta))) * radius)).Add(up.Mul(float32(math.Sin(float64(theta))) * radius))
}

func vecToArr(v mgl32.Vec3) [3]float32 { return [3]float32{v.X(), v.Y(), v.Z()} }

func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
