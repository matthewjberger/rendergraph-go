package pass

import (
	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/render/asset"
	"github.com/matthewjberger/indigo/transform"
)

func UpdateSkeletonLines(world *ecs.World) {
	settings, ok := ecs.Resource[render.Graphics](world)
	if !ok || settings == nil || !settings.ShowSkeletons {
		return
	}
	linesRes, ok := ecs.Resource[LinesResource](world)
	if !ok || linesRes == nil || linesRes.Lines == nil {
		return
	}
	lines := linesRes.Lines

	jointSize := settings.Lines.SkeletonJointSize
	if jointSize <= 0 {
		jointSize = 0.04
	}
	jointColor := settings.Lines.SkeletonJointColor
	if jointColor[3] == 0 {
		jointColor = [4]float32{0.4, 1.0, 0.4, 1.0}
	}
	boneColor := settings.Lines.SkeletonBoneColor
	if boneColor[3] == 0 {
		boneColor = [4]float32{1.0, 0.85, 0.2, 1.0}
	}

	mask := ecs.MustMaskOf[asset.SkinnedMesh](world)
	world.ForEach(mask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		skinned, _ := ecs.Get[asset.SkinnedMesh](world, entity)
		if skinned == nil || skinned.Skin == nil || len(skinned.Skin.Joints) == 0 {
			return
		}
		jointSet := make(map[ecs.Entity]struct{}, len(skinned.Skin.Joints))
		for _, jointEntity := range skinned.Skin.Joints {
			jointSet[jointEntity] = struct{}{}
		}
		for _, jointEntity := range skinned.Skin.Joints {
			global, ok := ecs.Get[transform.GlobalTransform](world, jointEntity)
			if !ok {
				continue
			}
			pos := [3]float32{global.Matrix[12], global.Matrix[13], global.Matrix[14]}
			pushSkeletonJointCross(lines, pos, jointSize, jointColor)

			parent, ok := ecs.Get[transform.Parent](world, jointEntity)
			if !ok || parent == nil {
				continue
			}
			if _, isJoint := jointSet[parent.Entity]; !isJoint {
				continue
			}
			parentGlobal, ok := ecs.Get[transform.GlobalTransform](world, parent.Entity)
			if !ok {
				continue
			}
			parentPos := [3]float32{parentGlobal.Matrix[12], parentGlobal.Matrix[13], parentGlobal.Matrix[14]}
			lines.AddOverlaySegment(parentPos, pos, boneColor)
		}
	})
}

func pushSkeletonJointCross(lines *Lines, position [3]float32, size float32, color [4]float32) {
	half := size * 0.5
	lines.AddOverlaySegment(
		[3]float32{position[0] - half, position[1], position[2]},
		[3]float32{position[0] + half, position[1], position[2]},
		color,
	)
	lines.AddOverlaySegment(
		[3]float32{position[0], position[1] - half, position[2]},
		[3]float32{position[0], position[1] + half, position[2]},
		color,
	)
	lines.AddOverlaySegment(
		[3]float32{position[0], position[1], position[2] - half},
		[3]float32{position[0], position[1], position[2] + half},
		color,
	)
}
