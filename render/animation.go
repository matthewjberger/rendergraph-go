package render

// AnimationProperty names the per-channel animation target.
type AnimationProperty uint8

const (
	AnimationTranslation AnimationProperty = iota
	AnimationRotation
	AnimationScale
	AnimationMorphWeights
)

// AnimationInterpolation selects how sampler outputs are blended
// between keyframes.
type AnimationInterpolation uint8

const (
	InterpolationLinear AnimationInterpolation = iota
	InterpolationStep
	InterpolationCubicSpline
)

// AnimationSampler is the data behind one animation channel: a list
// of keyframe times paired with per-keyframe output values. The
// loader stores rotation outputs as flat [4]float32 quaternions,
// translation / scale as [3]float32, morph weights as variable-size
// []float32 per keyframe.
type AnimationSampler struct {
	Interpolation AnimationInterpolation
	Inputs        []float32
	Vec3Outputs   [][3]float32
	Vec4Outputs   [][4]float32
	ScalarOutputs [][]float32
}

// AnimationChannel binds a sampler to a target node + property.
// TargetNode indexes into [LoadedScene.Nodes].
type AnimationChannel struct {
	TargetNode int
	Property   AnimationProperty
	Sampler    AnimationSampler
}

// AnimationClip is one named clip from a glTF animation entry.
// Duration is the max input time across every sampler.
type AnimationClip struct {
	Name     string
	Duration float32
	Channels []AnimationChannel
}
