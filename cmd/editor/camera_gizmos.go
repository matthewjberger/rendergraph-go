package main

import (
	"github.com/go-gl/mathgl/mgl32"

	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/internal/render/pass"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/transform"
)

const (
	cameraBodyDepth      = 0.25
	cameraBodyHalfWidth  = 0.18
	cameraBodyHalfHeight = 0.12
	cameraLensLength     = 0.45
	cameraLensRadius     = 0.32
	cameraUpTriangleSize = 0.18
)

func cameraGizmoPickExtents() transform.Vec3 {
	width := float32(cameraBodyHalfWidth * 2)
	if w := float32(cameraLensRadius * 2); w > width {
		width = w
	}
	height := float32(cameraBodyHalfHeight * 2)
	if h := float32(cameraLensRadius * 2); h > height {
		height = h
	}
	depth := float32(cameraBodyDepth + cameraLensLength)
	return transform.Vec3{width, height, depth}
}

func drawCameraGizmos(engine *ecs.World) {
	if !ecs.HasResource[pass.LinesResource](engine) {
		return
	}
	lines := ecs.MustResource[pass.LinesResource](engine).Lines
	mask := ecs.MustMaskOf[render.CameraMarker](engine) | ecs.MustMaskOf[transform.GlobalTransform](engine)
	engine.ForEach(mask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		global, ok := ecs.Get[transform.GlobalTransform](engine, entity)
		if !ok {
			return
		}
		matrix := global.Matrix
		origin := mgl32.Vec3{matrix[12], matrix[13], matrix[14]}
		right := mgl32.Vec3{matrix[0], matrix[1], matrix[2]}.Normalize()
		up := mgl32.Vec3{matrix[4], matrix[5], matrix[6]}.Normalize()
		forward := mgl32.Vec3{-matrix[8], -matrix[9], -matrix[10]}.Normalize()
		color := [4]float32{0.9, 0.9, 0.95, 1}
		drawCameraGizmoFor(lines, origin, right, up, forward, color)
	})
}

func drawCameraGizmoFor(lines *pass.Lines, origin, right, up, forward mgl32.Vec3, color [4]float32) {
	rightHalf := right.Mul(cameraBodyHalfWidth)
	upHalf := up.Mul(cameraBodyHalfHeight)
	back := origin.Sub(forward.Mul(cameraBodyDepth))

	frontTL := origin.Sub(rightHalf).Add(upHalf)
	frontTR := origin.Add(rightHalf).Add(upHalf)
	frontBL := origin.Sub(rightHalf).Sub(upHalf)
	frontBR := origin.Add(rightHalf).Sub(upHalf)
	backTL := back.Sub(rightHalf).Add(upHalf)
	backTR := back.Add(rightHalf).Add(upHalf)
	backBL := back.Sub(rightHalf).Sub(upHalf)
	backBR := back.Add(rightHalf).Sub(upHalf)

	body := [][2]mgl32.Vec3{
		{frontTL, frontTR}, {frontTR, frontBR}, {frontBR, frontBL}, {frontBL, frontTL},
		{backTL, backTR}, {backTR, backBR}, {backBR, backBL}, {backBL, backTL},
		{frontTL, backTL}, {frontTR, backTR}, {frontBL, backBL}, {frontBR, backBR},
	}
	for _, seg := range body {
		lines.AddSegment(vecToArr(seg[0]), vecToArr(seg[1]), color)
	}

	lensRimCenter := origin.Add(forward.Mul(cameraLensLength))
	lensRightHalf := right.Mul(cameraLensRadius)
	lensUpHalf := up.Mul(cameraLensRadius)
	rimTL := lensRimCenter.Sub(lensRightHalf).Add(lensUpHalf)
	rimTR := lensRimCenter.Add(lensRightHalf).Add(lensUpHalf)
	rimBL := lensRimCenter.Sub(lensRightHalf).Sub(lensUpHalf)
	rimBR := lensRimCenter.Add(lensRightHalf).Sub(lensUpHalf)

	cone := [][2]mgl32.Vec3{
		{origin, rimTL}, {origin, rimTR}, {origin, rimBL}, {origin, rimBR},
		{rimTL, rimTR}, {rimTR, rimBR}, {rimBR, rimBL}, {rimBL, rimTL},
	}
	for _, seg := range cone {
		lines.AddSegment(vecToArr(seg[0]), vecToArr(seg[1]), color)
	}

	triBaseLeft := frontTL.Add(up.Mul(0.01))
	triBaseRight := frontTR.Add(up.Mul(0.01))
	triApex := origin.Add(up.Mul(cameraBodyHalfHeight + cameraUpTriangleSize)).Sub(forward.Mul(cameraBodyDepth * 0.5))
	lines.AddSegment(vecToArr(triBaseLeft), vecToArr(triBaseRight), color)
	lines.AddSegment(vecToArr(triBaseLeft), vecToArr(triApex), color)
	lines.AddSegment(vecToArr(triBaseRight), vecToArr(triApex), color)
}
