package pass

import (
	_ "embed"
	"fmt"
	"hash/fnv"
	"sort"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/go-gl/mathgl/mgl32"

	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render/asset"
	"github.com/matthewjberger/indigo/transform"
)

//go:embed animation_compute.wgsl
var animationComputeShader string

const (
	animPropertyTranslation uint32 = 0
	animPropertyRotation    uint32 = 1
	animPropertyScale       uint32 = 2
)

type gpuAnimBone struct {
	OutputIndex       uint32
	ParentOutputIndex int32
	CurChannelStart   uint32
	CurChannelCount   uint32
	BlendChannelStart uint32
	BlendChannelCount uint32
	Pad0              uint32
	Pad1              uint32
	RestTranslation   [4]float32
	RestRotation      [4]float32
	RestScale         [4]float32
}

type gpuAnimChannel struct {
	Property      uint32
	Interpolation uint32
	InputOffset   uint32
	KeyCount      uint32
	OutputOffset  uint32
	OutputStride  uint32
	Pad0          uint32
	Pad1          uint32
}

type gpuAnimSkeleton struct {
	BoneStart     uint32
	BoneCount     uint32
	Time          float32
	BlendFromTime float32
	BlendFactor   float32
	Pad0          uint32
	Pad1          uint32
	Pad2          uint32
	ArmatureRoot  mgl32.Mat4
}

type animSkeletonLayout struct {
	skinEntity   ecs.Entity
	playerEntity ecs.Entity
	boneStart    uint32
	boneCount    uint32
}

type ownedChannel struct {
	property      uint32
	interpolation uint32
	input         []float32
	values        [][4]float32
	stride        uint32
}

type flatChannels struct {
	channels []gpuAnimChannel
	times    []float32
	values   [][4]float32
}

func (f *flatChannels) push(channel *ownedChannel) {
	inputOffset := uint32(len(f.times))
	f.times = append(f.times, channel.input...)
	outputOffset := uint32(len(f.values))
	f.values = append(f.values, channel.values...)
	f.channels = append(f.channels, gpuAnimChannel{
		Property:      channel.property,
		Interpolation: channel.interpolation,
		InputOffset:   inputOffset,
		KeyCount:      uint32(len(channel.input)),
		OutputOffset:  outputOffset,
		OutputStride:  channel.stride,
	})
}

type AnimationCompute struct {
	pipeline  *wgpu.ComputePipeline
	layout    *wgpu.BindGroupLayout
	bindGroup *wgpu.BindGroup

	skeletonsBuffer *wgpu.Buffer
	bonesBuffer     *wgpu.Buffer
	channelsBuffer  *wgpu.Buffer
	timesBuffer     *wgpu.Buffer
	valuesBuffer    *wgpu.Buffer

	layouts          []animSkeletonLayout
	cachedSignature  uint64
	boneGen          uint64
	skeletonsScratch []gpuAnimSkeleton
}

func newAnimationCompute(device *wgpu.Device) (*AnimationCompute, error) {
	module, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "animation_compute shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: animationComputeShader},
	})
	if err != nil {
		return nil, fmt.Errorf("animation_compute: shader: %w", err)
	}
	defer module.Release()

	entries := make([]wgpu.BindGroupLayoutEntry, 6)
	for binding := range entries {
		bufferType := wgpu.BufferBindingTypeReadOnlyStorage
		if binding == 5 {
			bufferType = wgpu.BufferBindingTypeStorage
		}
		entries[binding] = wgpu.BindGroupLayoutEntry{
			Binding:    uint32(binding),
			Visibility: wgpu.ShaderStageCompute,
			Buffer:     wgpu.BufferBindingLayout{Type: bufferType},
		}
	}
	layout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label:   "animation_compute layout",
		Entries: entries,
	})
	if err != nil {
		return nil, fmt.Errorf("animation_compute: layout: %w", err)
	}

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "animation_compute pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{layout},
	})
	if err != nil {
		layout.Release()
		return nil, fmt.Errorf("animation_compute: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	pipeline, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Layout: pipelineLayout,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     module,
			EntryPoint: "main",
		},
	})
	if err != nil {
		layout.Release()
		return nil, fmt.Errorf("animation_compute: pipeline: %w", err)
	}

	return &AnimationCompute{
		pipeline:        pipeline,
		layout:          layout,
		skeletonsBuffer: emptyStorage(device, "anim skeletons"),
		bonesBuffer:     emptyStorage(device, "anim bones"),
		channelsBuffer:  emptyStorage(device, "anim channels"),
		timesBuffer:     emptyStorage(device, "anim times"),
		valuesBuffer:    emptyStorage(device, "anim values"),
		cachedSignature: 0,
		boneGen:         ^uint64(0),
	}, nil
}

func (ac *AnimationCompute) release() {
	for _, buf := range []*wgpu.Buffer{ac.skeletonsBuffer, ac.bonesBuffer, ac.channelsBuffer, ac.timesBuffer, ac.valuesBuffer} {
		if buf != nil {
			buf.Release()
		}
	}
	ac.skeletonsBuffer, ac.bonesBuffer, ac.channelsBuffer, ac.timesBuffer, ac.valuesBuffer = nil, nil, nil, nil, nil
	if ac.bindGroup != nil {
		ac.bindGroup.Release()
		ac.bindGroup = nil
	}
	if ac.pipeline != nil {
		ac.pipeline.Release()
		ac.pipeline = nil
	}
	if ac.layout != nil {
		ac.layout.Release()
		ac.layout = nil
	}
}

func (ac *AnimationCompute) update(world *ecs.World, device *wgpu.Device, queue *wgpu.Queue, encoder *wgpu.CommandEncoder, sc *SkinningCompute) bool {
	jointToPlayer := make(map[ecs.Entity]ecs.Entity)
	playerTime := make(map[ecs.Entity]float32)
	playerClip := make(map[ecs.Entity]int)

	playerMask := ecs.MustMaskOf[asset.AnimationPlayer](world)
	world.ForEach(playerMask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		player, ok := ecs.Get[asset.AnimationPlayer](world, entity)
		if !ok || player == nil || !player.Playing {
			return
		}
		if player.CurrentClip < 0 || player.CurrentClip >= len(player.Clips) {
			return
		}
		playerTime[entity] = player.Time
		playerClip[entity] = player.CurrentClip
		clip := &player.Clips[player.CurrentClip]
		for channelIndex := range clip.Channels {
			channel := &clip.Channels[channelIndex]
			if _, ok := gpuAnimProperty(channel.Property); !ok {
				continue
			}
			target, ok := player.NodeIndexToEntity[channel.TargetNode]
			if !ok {
				continue
			}
			if _, exists := jointToPlayer[target]; !exists {
				jointToPlayer[target] = entity
			}
		}
	})

	signature := animStaticSignature(world, sc, jointToPlayer, playerClip)
	if signature != ac.cachedSignature || ac.boneGen != sc.BoneGeneration() {
		ac.rebuildStatic(device, queue, encoder, world, sc, jointToPlayer)
		ac.cachedSignature = signature
		ac.boneGen = sc.BoneGeneration()
		ac.bindGroup = ac.createBindGroup(device, sc.BoneBuffer())
	}

	ac.uploadSkeletons(world, device, queue, encoder, playerTime)
	return len(ac.layouts) > 0
}

func (ac *AnimationCompute) rebuildStatic(device *wgpu.Device, queue *wgpu.Queue, encoder *wgpu.CommandEncoder, world *ecs.World, sc *SkinningCompute, jointToPlayer map[ecs.Entity]ecs.Entity) {
	var bones []gpuAnimBone
	var flat flatChannels
	ac.layouts = ac.layouts[:0]

	skinMask := ecs.MustMaskOf[asset.SkinnedMesh](world)
	var skinEntities []ecs.Entity
	world.ForEach(skinMask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		skinned, _ := ecs.Get[asset.SkinnedMesh](world, entity)
		if skinned == nil || skinned.Skin == nil {
			return
		}
		skinEntities = append(skinEntities, entity)
	})
	sort.Slice(skinEntities, func(i, j int) bool {
		if skinEntities[i].ID != skinEntities[j].ID {
			return skinEntities[i].ID < skinEntities[j].ID
		}
		return skinEntities[i].Generation < skinEntities[j].Generation
	})

	for _, skinEntity := range skinEntities {
		skinned, _ := ecs.Get[asset.SkinnedMesh](world, skinEntity)
		if skinned == nil || skinned.Skin == nil {
			continue
		}
		skin := skinned.Skin

		playerEntity, found := ecs.Entity{}, false
		for _, joint := range skin.Joints {
			if player, ok := jointToPlayer[joint]; ok {
				playerEntity, found = player, true
				break
			}
		}
		if !found {
			continue
		}
		baseBone, ok := sc.BaseBoneIndex(skinEntity)
		if !ok {
			continue
		}
		player, ok := ecs.Get[asset.AnimationPlayer](world, playerEntity)
		if !ok || player == nil {
			continue
		}

		localIndexOf := make(map[ecs.Entity]int, len(skin.Joints))
		for local, joint := range skin.Joints {
			localIndexOf[joint] = local
		}
		order := make([]int, len(skin.Joints))
		for i := range order {
			order[i] = i
		}
		sort.SliceStable(order, func(i, j int) bool {
			return jointDepth(world, skin.Joints, localIndexOf, order[i]) < jointDepth(world, skin.Joints, localIndexOf, order[j])
		})

		boneStart := uint32(len(bones))
		for _, local := range order {
			joint := skin.Joints[local]
			rest := transform.IdentityLocalTransform()
			if local, ok := ecs.Get[transform.LocalTransform](world, joint); ok && local != nil {
				rest = *local
			}
			parentOutput := int32(-1)
			if parent, ok := ecs.Get[transform.Parent](world, joint); ok && parent != nil && !parent.IsRoot {
				if parentLocal, inSkin := localIndexOf[parent.Entity]; inSkin {
					parentOutput = int32(baseBone + uint32(parentLocal))
				}
			}
			curStart, curCount := appendJointChannels(&flat, world, player, joint)
			bones = append(bones, gpuAnimBone{
				OutputIndex:       baseBone + uint32(local),
				ParentOutputIndex: parentOutput,
				CurChannelStart:   curStart,
				CurChannelCount:   curCount,
				RestTranslation:   [4]float32{rest.Translation[0], rest.Translation[1], rest.Translation[2], 0},
				RestRotation:      [4]float32{rest.Rotation.V[0], rest.Rotation.V[1], rest.Rotation.V[2], rest.Rotation.W},
				RestScale:         [4]float32{rest.Scale[0], rest.Scale[1], rest.Scale[2], 0},
			})
		}
		ac.layouts = append(ac.layouts, animSkeletonLayout{
			skinEntity:   skinEntity,
			playerEntity: playerEntity,
			boneStart:    boneStart,
			boneCount:    uint32(len(skin.Joints)),
		})
	}

	ac.bonesBuffer = ac.uploadStatic(device, queue, encoder, ac.bonesBuffer, "anim bones", animBytes(bones))
	ac.channelsBuffer = ac.uploadStatic(device, queue, encoder, ac.channelsBuffer, "anim channels", animBytes(flat.channels))
	ac.timesBuffer = ac.uploadStatic(device, queue, encoder, ac.timesBuffer, "anim times", animBytes(flat.times))
	ac.valuesBuffer = ac.uploadStatic(device, queue, encoder, ac.valuesBuffer, "anim values", animBytes(flat.values))

	skeletonCount := len(ac.layouts)
	if skeletonCount < 1 {
		skeletonCount = 1
	}
	if ac.skeletonsBuffer != nil {
		ac.skeletonsBuffer.Release()
	}
	buf, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "anim skeletons",
		Size:  uint64(skeletonCount) * uint64(unsafe.Sizeof(gpuAnimSkeleton{})),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err == nil {
		ac.skeletonsBuffer = buf
	}
}

func (ac *AnimationCompute) uploadStatic(device *wgpu.Device, queue *wgpu.Queue, encoder *wgpu.CommandEncoder, old *wgpu.Buffer, label string, data []byte) *wgpu.Buffer {
	if old != nil {
		old.Release()
	}
	size := uint64(len(data))
	if size < 16 {
		size = 16
	}
	buf, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: label,
		Size:  size,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil
	}
	if len(data) > 0 {
		writeBuffer(device, queue, encoder, buf, 0, data)
	}
	return buf
}

func (ac *AnimationCompute) uploadSkeletons(world *ecs.World, device *wgpu.Device, queue *wgpu.Queue, encoder *wgpu.CommandEncoder, playerTime map[ecs.Entity]float32) {
	if len(ac.layouts) == 0 {
		return
	}
	ac.skeletonsScratch = ac.skeletonsScratch[:0]
	for _, layout := range ac.layouts {
		time, ok := playerTime[layout.playerEntity]
		boneCount := layout.boneCount
		if !ok {
			boneCount = 0
		}
		ac.skeletonsScratch = append(ac.skeletonsScratch, gpuAnimSkeleton{
			BoneStart:    layout.boneStart,
			BoneCount:    boneCount,
			Time:         time,
			BlendFactor:  1,
			ArmatureRoot: skinArmatureRoot(world, layout.skinEntity),
		})
	}
	writeBuffer(device, queue, encoder, ac.skeletonsBuffer, 0, animBytes(ac.skeletonsScratch))
}

func (ac *AnimationCompute) createBindGroup(device *wgpu.Device, boneTransforms *wgpu.Buffer) *wgpu.BindGroup {
	if ac.bindGroup != nil {
		ac.bindGroup.Release()
		ac.bindGroup = nil
	}
	if boneTransforms == nil {
		return nil
	}
	bg, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "animation_compute bg",
		Layout: ac.layout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: ac.skeletonsBuffer, Offset: 0, Size: wgpu.WholeSize},
			{Binding: 1, Buffer: ac.bonesBuffer, Offset: 0, Size: wgpu.WholeSize},
			{Binding: 2, Buffer: ac.channelsBuffer, Offset: 0, Size: wgpu.WholeSize},
			{Binding: 3, Buffer: ac.timesBuffer, Offset: 0, Size: wgpu.WholeSize},
			{Binding: 4, Buffer: ac.valuesBuffer, Offset: 0, Size: wgpu.WholeSize},
			{Binding: 5, Buffer: boneTransforms, Offset: 0, Size: wgpu.WholeSize},
		},
	})
	if err != nil {
		return nil
	}
	return bg
}

func (ac *AnimationCompute) dispatch(encoder *wgpu.CommandEncoder) {
	if len(ac.layouts) == 0 || ac.bindGroup == nil {
		return
	}
	pass := encoder.BeginComputePass(&wgpu.ComputePassDescriptor{})
	pass.SetPipeline(ac.pipeline)
	pass.SetBindGroup(0, ac.bindGroup, nil)
	pass.DispatchWorkgroups((uint32(len(ac.layouts))+63)/64, 1, 1)
	pass.End()
	pass.Release()
}

func appendJointChannels(flat *flatChannels, world *ecs.World, player *asset.AnimationPlayer, joint ecs.Entity) (uint32, uint32) {
	start := uint32(len(flat.channels))
	count := uint32(0)
	if player.CurrentClip < 0 || player.CurrentClip >= len(player.Clips) {
		return start, count
	}
	clip := &player.Clips[player.CurrentClip]
	for channelIndex := range clip.Channels {
		channel := &clip.Channels[channelIndex]
		target, ok := player.NodeIndexToEntity[channel.TargetNode]
		if !ok || target != joint {
			continue
		}
		owned, ok := buildOwnedChannel(channel)
		if !ok {
			continue
		}
		flat.push(&owned)
		count++
	}
	return start, count
}

func buildOwnedChannel(channel *asset.AnimationChannel) (ownedChannel, bool) {
	property, ok := gpuAnimProperty(channel.Property)
	if !ok {
		return ownedChannel{}, false
	}
	sampler := &channel.Sampler
	interpolation := uint32(0)
	stride := uint32(1)
	switch sampler.Interpolation {
	case asset.InterpolationStep:
		interpolation = 1
	case asset.InterpolationCubicSpline:
		interpolation = 2
		stride = 3
	}
	var values [][4]float32
	if channel.Property == asset.AnimationRotation {
		if len(sampler.Vec4Outputs) == 0 {
			return ownedChannel{}, false
		}
		values = make([][4]float32, len(sampler.Vec4Outputs))
		copy(values, sampler.Vec4Outputs)
	} else {
		if len(sampler.Vec3Outputs) == 0 {
			return ownedChannel{}, false
		}
		values = make([][4]float32, len(sampler.Vec3Outputs))
		for index, value := range sampler.Vec3Outputs {
			values[index] = [4]float32{value[0], value[1], value[2], 0}
		}
	}
	return ownedChannel{
		property:      property,
		interpolation: interpolation,
		input:         sampler.Inputs,
		values:        values,
		stride:        stride,
	}, true
}

func gpuAnimProperty(property asset.AnimationProperty) (uint32, bool) {
	switch property {
	case asset.AnimationTranslation:
		return animPropertyTranslation, true
	case asset.AnimationRotation:
		return animPropertyRotation, true
	case asset.AnimationScale:
		return animPropertyScale, true
	default:
		return 0, false
	}
}

func jointDepth(world *ecs.World, joints []ecs.Entity, localIndexOf map[ecs.Entity]int, local int) uint32 {
	depth := uint32(0)
	current := joints[local]
	for {
		parent, ok := ecs.Get[transform.Parent](world, current)
		if !ok || parent == nil || parent.IsRoot {
			break
		}
		if _, inSkin := localIndexOf[parent.Entity]; !inSkin {
			break
		}
		depth++
		current = parent.Entity
		if depth > uint32(len(joints)) {
			break
		}
	}
	return depth
}

func skinArmatureRoot(world *ecs.World, skinEntity ecs.Entity) mgl32.Mat4 {
	skinned, ok := ecs.Get[asset.SkinnedMesh](world, skinEntity)
	if !ok || skinned == nil || skinned.Skin == nil {
		return mgl32.Ident4()
	}
	skin := skinned.Skin
	localIndexOf := make(map[ecs.Entity]int, len(skin.Joints))
	for local, joint := range skin.Joints {
		localIndexOf[joint] = local
	}
	for _, joint := range skin.Joints {
		parent, ok := ecs.Get[transform.Parent](world, joint)
		if !ok || parent == nil || parent.IsRoot {
			continue
		}
		if _, inSkin := localIndexOf[parent.Entity]; inSkin {
			continue
		}
		if global, ok := ecs.Get[transform.GlobalTransform](world, parent.Entity); ok && global != nil {
			return global.Matrix
		}
	}
	return mgl32.Ident4()
}

func animStaticSignature(world *ecs.World, sc *SkinningCompute, jointToPlayer map[ecs.Entity]ecs.Entity, playerClip map[ecs.Entity]int) uint64 {
	hasher := fnv.New64a()
	var scratch [8]byte
	skinMask := ecs.MustMaskOf[asset.SkinnedMesh](world)
	var skinEntities []ecs.Entity
	world.ForEach(skinMask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		skinned, _ := ecs.Get[asset.SkinnedMesh](world, entity)
		if skinned == nil || skinned.Skin == nil {
			return
		}
		skinEntities = append(skinEntities, entity)
	})
	sort.Slice(skinEntities, func(i, j int) bool {
		if skinEntities[i].ID != skinEntities[j].ID {
			return skinEntities[i].ID < skinEntities[j].ID
		}
		return skinEntities[i].Generation < skinEntities[j].Generation
	})
	for _, skinEntity := range skinEntities {
		skinned, _ := ecs.Get[asset.SkinnedMesh](world, skinEntity)
		if skinned == nil || skinned.Skin == nil {
			continue
		}
		skin := skinned.Skin
		playerEntity, found := ecs.Entity{}, false
		for _, joint := range skin.Joints {
			if player, ok := jointToPlayer[joint]; ok {
				playerEntity, found = player, true
				break
			}
		}
		if !found {
			continue
		}
		putUint64(&scratch, uint64(skinEntity.ID))
		hasher.Write(scratch[:])
		for _, joint := range skin.Joints {
			putUint64(&scratch, uint64(joint.ID))
			hasher.Write(scratch[:])
		}
		putUint64(&scratch, uint64(int64(playerClip[playerEntity])))
		hasher.Write(scratch[:])
	}
	return hasher.Sum64()
}

func emptyStorage(device *wgpu.Device, label string) *wgpu.Buffer {
	buf, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: label,
		Size:  16,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil
	}
	return buf
}

func animBytes[T any](items []T) []byte {
	if len(items) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&items[0])), len(items)*int(unsafe.Sizeof(items[0])))
}
