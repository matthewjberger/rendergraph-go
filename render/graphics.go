package render

import (
	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/ui"
)

type Graphics struct {
	ShowSky       bool
	ShowGrid      bool
	ShowBounds    bool
	ShowNormals   bool
	ShowSkeletons bool

	ViewerMode bool

	Exposure    float32
	FxaaEnabled bool
	Bloom       Bloom
	Ssao        Ssao

	Cull Cull

	Lines DebugLines
}

type Bloom struct {
	Enabled   bool
	Intensity float32
}

type Ssao struct {
	Enabled     bool
	Radius      float32
	Bias        float32
	Intensity   float32
	SampleCount int
}

type Cull struct {
	Enabled            bool
	MinScreenPixelSize float32
}

type DebugLines struct {
	NormalLength       float32
	NormalColor        [4]float32
	SkeletonJointSize  float32
	SkeletonJointColor [4]float32
	SkeletonBoneColor  [4]float32
}

func DefaultGraphics() Graphics {
	return Graphics{
		ShowSky:       true,
		ShowGrid:      true,
		ShowBounds:    false,
		ShowNormals:   false,
		ShowSkeletons: false,
		ViewerMode:    true,
		Exposure:      1.0,
		FxaaEnabled:   true,
		Bloom: Bloom{
			Enabled:   true,
			Intensity: 0.04,
		},
		Ssao: Ssao{
			Enabled:     true,
			Radius:      0.5,
			Bias:        0.025,
			Intensity:   1.5,
			SampleCount: 32,
		},
		Cull: Cull{
			Enabled:            true,
			MinScreenPixelSize: 1.0,
		},
		Lines: DebugLines{
			NormalLength:       0.08,
			NormalColor:        [4]float32{1.0, 0.92, 0.2, 0.95},
			SkeletonJointSize:  0.04,
			SkeletonJointColor: [4]float32{0.4, 1.0, 0.4, 1.0},
			SkeletonBoneColor:  [4]float32{1.0, 0.85, 0.2, 1.0},
		},
	}
}

func UpdateGraphicsToggles(world *ecs.World) {
	if ecs.HasResource[ui.WorldRef](world) {
		if ui.AnyTextInputFocused(ecs.MustResource[ui.WorldRef](world).World) {
			return
		}
	}
	input := ecs.MustResource[Input](world)
	settings := ecs.MustResource[Graphics](world)
	for _, key := range input.KeysJustDown {
		switch key {
		case 'G':
			settings.ShowGrid = !settings.ShowGrid
		case 'S':
			settings.ShowSky = !settings.ShowSky
		case 'F':
			settings.FxaaEnabled = !settings.FxaaEnabled
		case 'B':
			settings.ShowBounds = !settings.ShowBounds
		case 'N':
			settings.ShowNormals = !settings.ShowNormals
		case 'K':
			settings.ShowSkeletons = !settings.ShowSkeletons
		case 'V':
			settings.ViewerMode = !settings.ViewerMode
		}
	}
}
