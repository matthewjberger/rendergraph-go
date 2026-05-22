package main

import (
	"github.com/cogentcore/webgpu/wgpu"

	"indigo/render/asset"
)

// whiteCubeVertices returns a 36-vertex unit cube with every vertex
// colored white. Entities tint it through their [asset.Material]
// component; the vertex color stays as a neutral multiplicand so the
// material reads out exactly.
func whiteCubeVertices() []asset.MeshVertex {
	const s = 0.5
	white := [4]float32{1, 1, 1, 1}

	face := func(a, b, c, d [3]float32) []asset.MeshVertex {
		v := func(p [3]float32) asset.MeshVertex {
			return asset.MeshVertex{
				Position: [4]float32{p[0], p[1], p[2], 1.0},
				Color:    white,
			}
		}
		return []asset.MeshVertex{v(a), v(b), v(c), v(a), v(c), v(d)}
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

	out := make([]asset.MeshVertex, 0, 36)
	out = append(out, plusZ...)
	out = append(out, minusZ...)
	out = append(out, plusX...)
	out = append(out, minusX...)
	out = append(out, plusY...)
	out = append(out, minusY...)
	return out
}

// brickPalette holds the per-row material colors plus the paddle and
// ball colors. One white cube mesh + per-entity Material gives every
// piece its own tint without registering N mesh variants.
type brickPalette struct {
	Cube   asset.MeshHandle
	Rows   [][4]float32
	Paddle [4]float32
	Ball   [4]float32
}

// registerBreakoutPalette uploads the single white cube mesh used by
// the paddle, ball, and bricks and returns it alongside the per-row /
// per-element material colors.
func registerBreakoutPalette(device *wgpu.Device, assets *asset.MeshAssets) (brickPalette, error) {
	cube, err := assets.Register(device, "white_cube", whiteCubeVertices())
	if err != nil {
		return brickPalette{}, err
	}
	return brickPalette{
		Cube: cube,
		Rows: [][4]float32{
			{0.92, 0.28, 0.28, 1.0},
			{0.95, 0.62, 0.25, 1.0},
			{0.95, 0.85, 0.3, 1.0},
			{0.4, 0.85, 0.45, 1.0},
		},
		Paddle: [4]float32{0.88, 0.88, 0.92, 1.0},
		Ball:   [4]float32{0.35, 0.9, 0.95, 1.0},
	}, nil
}
