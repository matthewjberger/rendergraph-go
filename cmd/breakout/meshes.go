package main

import (
	"github.com/cogentcore/webgpu/wgpu"

	"indigo/render"
)

// coloredCubeVertices returns a 36-vertex unit cube with every face
// painted in the given color. Wound CCW from outside, same six-face
// layout the engine's UnitCubeVertices uses.
func coloredCubeVertices(color [4]float32) []render.MeshVertex {
	const s = 0.5

	face := func(a, b, c, d [3]float32) []render.MeshVertex {
		v := func(p [3]float32) render.MeshVertex {
			return render.MeshVertex{
				Position: [4]float32{p[0], p[1], p[2], 1.0},
				Color:    color,
			}
		}
		return []render.MeshVertex{v(a), v(b), v(c), v(a), v(c), v(d)}
	}

	plusZ := face(
		[3]float32{-s, -s, s}, [3]float32{s, -s, s},
		[3]float32{s, s, s}, [3]float32{-s, s, s},
	)
	minusZ := face(
		[3]float32{s, -s, -s}, [3]float32{-s, -s, -s},
		[3]float32{-s, s, -s}, [3]float32{s, s, -s},
	)
	plusX := face(
		[3]float32{s, -s, s}, [3]float32{s, -s, -s},
		[3]float32{s, s, -s}, [3]float32{s, s, s},
	)
	minusX := face(
		[3]float32{-s, -s, -s}, [3]float32{-s, -s, s},
		[3]float32{-s, s, s}, [3]float32{-s, s, -s},
	)
	plusY := face(
		[3]float32{-s, s, s}, [3]float32{s, s, s},
		[3]float32{s, s, -s}, [3]float32{-s, s, -s},
	)
	minusY := face(
		[3]float32{-s, -s, -s}, [3]float32{s, -s, -s},
		[3]float32{s, -s, s}, [3]float32{-s, -s, s},
	)

	out := make([]render.MeshVertex, 0, 36)
	out = append(out, plusZ...)
	out = append(out, minusZ...)
	out = append(out, plusX...)
	out = append(out, minusX...)
	out = append(out, plusY...)
	out = append(out, minusY...)
	return out
}

// brickMeshes holds one mesh handle per brick row color, plus the
// paddle and ball handles. The breakout app registers these on top
// of the engine's built-in primitives.
type brickMeshes struct {
	Rows   []render.MeshHandle
	Paddle render.MeshHandle
	Ball   render.MeshHandle
}

// registerBreakoutMeshes uploads the colored cubes for the brick
// rows, paddle, and ball into the given mesh registry.
func registerBreakoutMeshes(device *wgpu.Device, assets *render.MeshAssets) (brickMeshes, error) {
	rowColors := [][4]float32{
		{0.92, 0.28, 0.28, 1.0},
		{0.95, 0.62, 0.25, 1.0},
		{0.95, 0.85, 0.3, 1.0},
		{0.4, 0.85, 0.45, 1.0},
	}

	var out brickMeshes
	for index, color := range rowColors {
		handle, err := assets.Register(
			device,
			brickRowMeshName(index),
			coloredCubeVertices(color),
		)
		if err != nil {
			return out, err
		}
		out.Rows = append(out.Rows, handle)
	}

	paddle, err := assets.Register(
		device,
		"paddle",
		coloredCubeVertices([4]float32{0.88, 0.88, 0.92, 1.0}),
	)
	if err != nil {
		return out, err
	}
	out.Paddle = paddle

	ball, err := assets.Register(
		device,
		"ball",
		coloredCubeVertices([4]float32{0.35, 0.9, 0.95, 1.0}),
	)
	if err != nil {
		return out, err
	}
	out.Ball = ball
	return out, nil
}

func brickRowMeshName(index int) string {
	switch index {
	case 0:
		return "brick_red"
	case 1:
		return "brick_orange"
	case 2:
		return "brick_yellow"
	case 3:
		return "brick_green"
	}
	return "brick"
}
