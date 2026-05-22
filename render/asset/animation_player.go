package asset

import (
	"math"

	"github.com/go-gl/mathgl/mgl32"

	"indigo/ecs"
	"indigo/transform"
	"indigo/window"
)

// AnimationPlayer is the per-entity component that drives glTF
// animation playback. One player owns a slice of [AnimationClip]
// and a map from glTF node index to the spawned ECS entity for
// that node (built by [SpawnLoadedScene] at load time). Each frame
// [UpdateAnimationPlayers] advances Time by delta*Speed, samples
// the current clip's channels at that time, and writes the
// resulting translation / rotation / scale into the targeted
// node's LocalTransform.
//
// Scope mirrors nightshade's AnimationPlayer minus skinning and
// cross-fade: a single clip plays at a time, looping is per-player,
// stopping at the end is the alternative.
type AnimationPlayer struct {
	Clips             []AnimationClip
	CurrentClip       int
	Time              float32
	Speed             float32
	Looping           bool
	Playing           bool
	NodeIndexToEntity map[int]ecs.Entity
}

// NewAnimationPlayer returns a player that auto-selects clip 0 and
// starts playing in a loop. Apps that want different startup state
// can mutate the returned struct before attaching it.
func NewAnimationPlayer(clips []AnimationClip, nodeIndexToEntity map[int]ecs.Entity) AnimationPlayer {
	player := AnimationPlayer{
		Clips:             clips,
		CurrentClip:       -1,
		Speed:             1,
		Looping:           true,
		NodeIndexToEntity: nodeIndexToEntity,
	}
	if len(clips) > 0 {
		player.CurrentClip = 0
		player.Playing = true
	}
	return player
}

// UpdateAnimationPlayers is the per-frame system: advances every
// AnimationPlayer's clock, samples the current clip's channels at
// the new time, and writes the resulting TRS into each targeted
// node's LocalTransform via [transform.AssignLocalTransform] (which
// marks the node dirty so the propagation system rebuilds its
// GlobalTransform next).
//
// Reads delta from the engine world's window timing resource. Put
// this in the engine schedule BEFORE transform_propagation so the
// world matrices that flow into the renderer reflect this frame's
// sampled poses.
func UpdateAnimationPlayers(world *ecs.World) {
	w, ok := ecs.Resource[window.Window](world)
	if !ok {
		return
	}
	delta := w.Timing.DeltaSeconds

	mask := ecs.MustMaskOf[AnimationPlayer](world)
	world.ForEach(mask, 0, func(entity ecs.Entity, table *ecs.Archetype, index int) {
		players, _ := ecs.Column[AnimationPlayer](world, table)
		p := &players[index]
		advancePlayer(p, delta)
		applyPlayer(world, p)
	})
}

func advancePlayer(p *AnimationPlayer, delta float32) {
	if !p.Playing || p.CurrentClip < 0 || p.CurrentClip >= len(p.Clips) {
		return
	}
	clip := &p.Clips[p.CurrentClip]
	p.Time += delta * p.Speed
	if clip.Duration <= 0 {
		return
	}
	if p.Looping {
		p.Time = float32(math.Mod(float64(p.Time), float64(clip.Duration)))
		if p.Time < 0 {
			p.Time += clip.Duration
		}
	} else if p.Time > clip.Duration {
		p.Time = clip.Duration
		p.Playing = false
	}
}

func applyPlayer(world *ecs.World, p *AnimationPlayer) {
	if p.CurrentClip < 0 || p.CurrentClip >= len(p.Clips) {
		return
	}
	clip := &p.Clips[p.CurrentClip]
	for i := range clip.Channels {
		channel := &clip.Channels[i]
		target, ok := p.NodeIndexToEntity[channel.TargetNode]
		if !ok {
			continue
		}
		sampleChannelInto(world, target, channel, p.Time)
	}
}

func sampleChannelInto(world *ecs.World, target ecs.Entity, channel *AnimationChannel, t float32) {
	keyA, keyB, factor := findKeyframe(channel.Sampler.Inputs, t)
	local, ok := ecs.GetMut[transform.LocalTransform](world, target)
	if !ok {
		return
	}
	changed := false
	switch channel.Property {
	case AnimationTranslation:
		if len(channel.Sampler.Vec3Outputs) > 0 {
			local.Translation = sampleVec3(channel.Sampler.Vec3Outputs, keyA, keyB, factor, channel.Sampler.Interpolation)
			changed = true
		}
	case AnimationScale:
		if len(channel.Sampler.Vec3Outputs) > 0 {
			local.Scale = sampleVec3(channel.Sampler.Vec3Outputs, keyA, keyB, factor, channel.Sampler.Interpolation)
			changed = true
		}
	case AnimationRotation:
		if len(channel.Sampler.Vec4Outputs) > 0 {
			local.Rotation = sampleQuat(channel.Sampler.Vec4Outputs, keyA, keyB, factor, channel.Sampler.Interpolation)
			changed = true
		}
	}
	if changed {
		transform.MarkDirty(world, target)
	}
}

// findKeyframe returns the indices of the keyframes bracketing t
// plus the interpolation factor in [0, 1]. Times must be sorted
// ascending (glTF guarantees this for AnimationSampler input
// accessors). Linear scan matches nightshade — clips have tens of
// keyframes typically, a binary search isn't worth the complexity.
func findKeyframe(times []float32, t float32) (int, int, float32) {
	if len(times) == 0 {
		return 0, 0, 0
	}
	if t <= times[0] || len(times) == 1 {
		return 0, 0, 0
	}
	last := len(times) - 1
	if t >= times[last] {
		return last, last, 0
	}
	for i := 0; i < last; i++ {
		if times[i+1] > t {
			span := times[i+1] - times[i]
			if span <= 0 {
				return i, i, 0
			}
			return i, i + 1, (t - times[i]) / span
		}
	}
	return last, last, 0
}

func sampleVec3(outputs [][3]float32, a, b int, factor float32, mode AnimationInterpolation) transform.Vec3 {
	va := outputs[a]
	vb := outputs[b]
	if mode == InterpolationStep || a == b {
		return transform.Vec3{va[0], va[1], va[2]}
	}
	return transform.Vec3{
		va[0] + (vb[0]-va[0])*factor,
		va[1] + (vb[1]-va[1])*factor,
		va[2] + (vb[2]-va[2])*factor,
	}
}

func sampleQuat(outputs [][4]float32, a, b int, factor float32, mode AnimationInterpolation) transform.Quat {
	va := outputs[a]
	vb := outputs[b]
	if mode == InterpolationStep || a == b {
		return transform.Quat{W: va[3], V: mgl32.Vec3{va[0], va[1], va[2]}}
	}
	qa := mgl32.Quat{W: va[3], V: mgl32.Vec3{va[0], va[1], va[2]}}
	qb := mgl32.Quat{W: vb[3], V: mgl32.Vec3{vb[0], vb[1], vb[2]}}
	return mgl32.QuatSlerp(qa, qb, factor)
}
