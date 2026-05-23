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
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/render/asset"
	"github.com/matthewjberger/indigo/transform"
)

//go:embed skinning_compute.wgsl
var skinningComputeShader string

type GpuSkinData struct {
	JointCount      uint32
	BaseBoneIndex   uint32
	BaseIbmIndex    uint32
	BaseOutputIndex uint32
}

type SkinnedInstance struct {
	BaseColor   [4]float32
	EntityID    uint32
	JointOffset uint32
	BaseLayer   uint32
	AlphaMode   uint32
}

type SkinnedDrawGroup struct {
	Mesh          asset.SkinnedMeshHandle
	FirstInstance uint32
	Count         uint32
}

type SkinningCompute struct {
	pipeline  *wgpu.ComputePipeline
	layout    *wgpu.BindGroupLayout
	bindGroup *wgpu.BindGroup

	boneBuffer     *wgpu.Buffer
	boneCap        int
	ibmBuffer      *wgpu.Buffer
	ibmCap         int
	skinDataBuffer *wgpu.Buffer
	skinDataCap    int
	jointBuffer    *wgpu.Buffer
	jointCap       int

	inverseBindMatrices []mgl32.Mat4
	skinData            []GpuSkinData
	entitySkinIndices   map[ecs.Entity]uint32
	jointEntities       []ecs.Entity
	totalJoints         uint32
	skinnedEntityIDs    []uint32
	staticUploaded      bool

	boneScratch []mgl32.Mat4

	lastBones []mgl32.Mat4

	generation uint64

	boneGeneration uint64

	animation *AnimationCompute

	instances       []SkinnedInstance
	instancesBuffer *wgpu.Buffer
	instancesCap    int
	instancesGen    uint64
	opaqueGroups    []SkinnedDrawGroup
	blendGroups     []SkinnedDrawGroup
}

func (sc *SkinningCompute) Generation() uint64 {
	return sc.generation
}

func (sc *SkinningCompute) InstancesBuffer() *wgpu.Buffer { return sc.instancesBuffer }

func (sc *SkinningCompute) InstancesGeneration() uint64 { return sc.instancesGen }

func (sc *SkinningCompute) OpaqueGroups() []SkinnedDrawGroup { return sc.opaqueGroups }

func (sc *SkinningCompute) BlendGroups() []SkinnedDrawGroup { return sc.blendGroups }

func (sc *SkinningCompute) BoneBuffer() *wgpu.Buffer { return sc.boneBuffer }

func (sc *SkinningCompute) BoneGeneration() uint64 { return sc.boneGeneration }

func (sc *SkinningCompute) BaseBoneIndex(entity ecs.Entity) (uint32, bool) {
	skinIndex, ok := sc.entitySkinIndices[entity]
	if !ok || int(skinIndex) >= len(sc.skinData) {
		return 0, false
	}
	return sc.skinData[skinIndex].BaseBoneIndex, true
}

type SkinningComputeResource struct {
	Compute *SkinningCompute
}

func AddSkinningComputePass(renderer *render.Renderer, sc *SkinningCompute) (*render.Pass, error) {
	pass := &render.Pass{
		Name:  "skinning_compute",
		State: sc,
		Execute: func(state any, context *render.PassContext) error {
			compute := state.(*SkinningCompute)
			if compute.Prepare(context.World, context.Device, context.Queue, context.Encoder) {
				compute.Dispatch(context.Encoder)
			}
			return nil
		},
		Release: func(state any) {
			state.(*SkinningCompute).Release()
		},
	}
	if err := renderer.Graph.AddPass(pass, nil); err != nil {
		return nil, err
	}
	return pass, nil
}

const matrixBytes = uint64(unsafe.Sizeof(mgl32.Mat4{}))

func NewSkinningCompute(device *wgpu.Device) (*SkinningCompute, error) {
	module, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "skinning_compute shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: skinningComputeShader},
	})
	if err != nil {
		return nil, fmt.Errorf("skinning_compute: shader: %w", err)
	}
	defer module.Release()

	layout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "skinning_compute layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("skinning_compute: layout: %w", err)
	}

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "skinning_compute pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{layout},
	})
	if err != nil {
		layout.Release()
		return nil, fmt.Errorf("skinning_compute: pipeline layout: %w", err)
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
		return nil, fmt.Errorf("skinning_compute: pipeline: %w", err)
	}

	animation, err := newAnimationCompute(device)
	if err != nil {
		pipeline.Release()
		layout.Release()
		return nil, err
	}

	return &SkinningCompute{
		pipeline:          pipeline,
		layout:            layout,
		entitySkinIndices: make(map[ecs.Entity]uint32),
		animation:         animation,
	}, nil
}

func (sc *SkinningCompute) Release() {
	for _, buf := range []*wgpu.Buffer{sc.boneBuffer, sc.ibmBuffer, sc.skinDataBuffer, sc.jointBuffer} {
		if buf != nil {
			buf.Release()
		}
	}
	sc.boneBuffer, sc.ibmBuffer, sc.skinDataBuffer, sc.jointBuffer = nil, nil, nil, nil
	if sc.instancesBuffer != nil {
		sc.instancesBuffer.Release()
		sc.instancesBuffer = nil
	}
	if sc.animation != nil {
		sc.animation.release()
		sc.animation = nil
	}
	if sc.bindGroup != nil {
		sc.bindGroup.Release()
		sc.bindGroup = nil
	}
	if sc.pipeline != nil {
		sc.pipeline.Release()
		sc.pipeline = nil
	}
	if sc.layout != nil {
		sc.layout.Release()
		sc.layout = nil
	}
}

func (sc *SkinningCompute) JointMatrixBuffer() *wgpu.Buffer {
	return sc.jointBuffer
}

func (sc *SkinningCompute) JointOffset(entity ecs.Entity) uint32 {
	skinIndex, ok := sc.entitySkinIndices[entity]
	if !ok || int(skinIndex) >= len(sc.skinData) {
		return 0
	}
	return sc.skinData[skinIndex].BaseOutputIndex
}

func (sc *SkinningCompute) needsRebuild(world *ecs.World) bool {
	if !sc.staticUploaded {
		return true
	}
	mask := ecs.MustMaskOf[asset.SkinnedMesh](world)
	mismatch := false
	count := 0
	world.ForEach(mask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		index := count
		count++
		if mismatch {
			return
		}
		if index >= len(sc.skinnedEntityIDs) || entity.ID != sc.skinnedEntityIDs[index] {
			mismatch = true
		}
	})
	if mismatch {
		return true
	}
	return count != len(sc.skinnedEntityIDs)
}

func (sc *SkinningCompute) rebuildStaticData(world *ecs.World) {
	sc.skinnedEntityIDs = sc.skinnedEntityIDs[:0]
	sc.inverseBindMatrices = sc.inverseBindMatrices[:0]
	sc.skinData = sc.skinData[:0]
	sc.jointEntities = sc.jointEntities[:0]
	for key := range sc.entitySkinIndices {
		delete(sc.entitySkinIndices, key)
	}
	sc.totalJoints = 0

	mask := ecs.MustMaskOf[asset.SkinnedMesh](world)
	boneEntityToIndex := make(map[ecs.Entity]uint32)

	world.ForEach(mask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		skinned, _ := ecs.Get[asset.SkinnedMesh](world, entity)
		if skinned == nil || skinned.Skin == nil {
			return
		}
		sc.skinnedEntityIDs = append(sc.skinnedEntityIDs, entity.ID)
		for _, jointEntity := range skinned.Skin.Joints {
			if _, seen := boneEntityToIndex[jointEntity]; !seen {
				boneEntityToIndex[jointEntity] = uint32(len(sc.jointEntities))
				sc.jointEntities = append(sc.jointEntities, jointEntity)
			}
		}
	})

	skinConfigs := make(map[uint64]uint32)
	var uniqueSkinIndex uint32
	var ibmOffset, outputOffset uint32

	world.ForEach(mask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		skinned, _ := ecs.Get[asset.SkinnedMesh](world, entity)
		if skinned == nil || skinned.Skin == nil {
			return
		}
		skin := skinned.Skin
		skinHash := hashSkin(skin)
		skinIndex, exists := skinConfigs[skinHash]
		if !exists {
			skinIndex = uniqueSkinIndex
			skinConfigs[skinHash] = skinIndex

			var baseBoneIndex uint32
			if len(skin.Joints) > 0 {
				baseBoneIndex = boneEntityToIndex[skin.Joints[0]]
			}
			jointCount := uint32(len(skin.Joints))
			if jointCount > uint32(asset.MaxJointsPerSkin) {
				jointCount = uint32(asset.MaxJointsPerSkin)
			}
			sc.skinData = append(sc.skinData, GpuSkinData{
				JointCount:      jointCount,
				BaseBoneIndex:   baseBoneIndex,
				BaseIbmIndex:    ibmOffset,
				BaseOutputIndex: outputOffset,
			})
			for jointIndex, ibm := range skin.InverseBindMatrices {
				if jointIndex >= asset.MaxJointsPerSkin {
					break
				}
				sc.inverseBindMatrices = append(sc.inverseBindMatrices, ibm)
			}
			ibmOffset += jointCount
			outputOffset += jointCount
			uniqueSkinIndex++
		}
		sc.entitySkinIndices[entity] = skinIndex
	})

	sc.totalJoints = outputOffset
	sc.staticUploaded = true
}

func hashSkin(skin *asset.Skin) uint64 {
	hasher := fnv.New64a()
	var scratch [8]byte
	for _, joint := range skin.Joints {
		putUint64(&scratch, uint64(joint.ID)|uint64(joint.Generation)<<32)
		hasher.Write(scratch[:])
	}
	for _, matrix := range skin.InverseBindMatrices {
		for component := 0; component < 16; component++ {
			putUint64(&scratch, uint64(mathFloat32Bits(matrix[component])))
			hasher.Write(scratch[:])
		}
	}
	return hasher.Sum64()
}

func (sc *SkinningCompute) buildInstances(world *ecs.World) {
	sc.instances = sc.instances[:0]
	sc.opaqueGroups = sc.opaqueGroups[:0]
	sc.blendGroups = sc.blendGroups[:0]

	type record struct {
		instance    SkinnedInstance
		mesh        asset.SkinnedMeshHandle
		transparent bool
	}
	var records []record
	mask := ecs.MustMaskOf[asset.SkinnedMesh](world) | ecs.MustMaskOf[transform.GlobalTransform](world)
	world.ForEach(mask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		sm, _ := ecs.Get[asset.SkinnedMesh](world, entity)
		if sm == nil || sm.Skin == nil {
			return
		}
		instance := SkinnedInstance{
			BaseColor:   [4]float32{1, 1, 1, 1},
			EntityID:    entity.ID,
			JointOffset: sc.JointOffset(entity),
			BaseLayer:   0xFFFFFFFF,
		}
		if material, ok := ecs.Get[asset.Material](world, entity); ok && material != nil {
			instance.BaseColor = material.BaseColor
			instance.BaseLayer = material.BaseColorLayer
			instance.AlphaMode = uint32(material.ToGPU().AlphaMode)
		}
		records = append(records, record{instance: instance, mesh: sm.Mesh, transparent: instance.AlphaMode == 2})
	})
	if len(records) == 0 {
		return
	}
	sort.SliceStable(records, func(i, j int) bool {
		if records[i].transparent != records[j].transparent {
			return !records[i].transparent
		}
		return records[i].mesh < records[j].mesh
	})
	for _, r := range records {
		sc.instances = append(sc.instances, r.instance)
	}
	index := 0
	for index < len(records) {
		transparent := records[index].transparent
		mesh := records[index].mesh
		first := uint32(index)
		for index < len(records) && records[index].transparent == transparent && records[index].mesh == mesh {
			index++
		}
		group := SkinnedDrawGroup{Mesh: mesh, FirstInstance: first, Count: uint32(index) - first}
		if transparent {
			sc.blendGroups = append(sc.blendGroups, group)
		} else {
			sc.opaqueGroups = append(sc.opaqueGroups, group)
		}
	}
}

func (sc *SkinningCompute) uploadInstances(device *wgpu.Device, queue *wgpu.Queue, encoder *wgpu.CommandEncoder) {
	count := len(sc.instances)
	if count == 0 {
		return
	}
	if sc.instancesCap < count {
		if sc.instancesBuffer != nil {
			sc.instancesBuffer.Release()
		}
		buf, err := device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: "skinned instances",
			Size:  uint64(count) * uint64(unsafe.Sizeof(SkinnedInstance{})),
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
		})
		if err != nil {
			return
		}
		sc.instancesBuffer = buf
		sc.instancesCap = count
		sc.instancesGen++
	}
	writeBuffer(device, queue, encoder, sc.instancesBuffer, 0, sliceBytes(sc.instances[:count]))
}

func (sc *SkinningCompute) collectBoneTransforms(world *ecs.World) {
	if cap(sc.boneScratch) < len(sc.jointEntities) {
		sc.boneScratch = make([]mgl32.Mat4, len(sc.jointEntities))
	}
	sc.boneScratch = sc.boneScratch[:len(sc.jointEntities)]
	for index, jointEntity := range sc.jointEntities {
		if global, ok := ecs.Get[transform.GlobalTransform](world, jointEntity); ok {
			sc.boneScratch[index] = global.Matrix
		} else {
			sc.boneScratch[index] = mgl32.Ident4()
		}
	}
}

func (sc *SkinningCompute) Prepare(world *ecs.World, device *wgpu.Device, queue *wgpu.Queue, encoder *wgpu.CommandEncoder) bool {
	rebuilt := false
	if sc.needsRebuild(world) {
		sc.rebuildStaticData(world)
		rebuilt = true
	}
	if sc.totalJoints == 0 || len(sc.skinData) == 0 {
		sc.instances = sc.instances[:0]
		sc.opaqueGroups = sc.opaqueGroups[:0]
		sc.blendGroups = sc.blendGroups[:0]
		return false
	}

	sc.buildInstances(world)
	sc.uploadInstances(device, queue, encoder)

	sc.collectBoneTransforms(world)

	grew := sc.ensureBuffers(device)
	if rebuilt || grew {
		if len(sc.skinData) > 0 {
			writeBuffer(device, queue, encoder, sc.skinDataBuffer, 0, skinDataBytes(sc.skinData))
		}
		if len(sc.inverseBindMatrices) > 0 {
			writeBuffer(device, queue, encoder, sc.ibmBuffer, 0, matrixSliceBytes(sc.inverseBindMatrices))
		}
	}
	if grew && sc.bindGroup != nil {
		sc.bindGroup.Release()
		sc.bindGroup = nil
	}
	if sc.bindGroup == nil {
		bg, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  "skinning_compute bg",
			Layout: sc.layout,
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: sc.boneBuffer, Offset: 0, Size: wgpu.WholeSize},
				{Binding: 1, Buffer: sc.ibmBuffer, Offset: 0, Size: wgpu.WholeSize},
				{Binding: 2, Buffer: sc.skinDataBuffer, Offset: 0, Size: wgpu.WholeSize},
				{Binding: 3, Buffer: sc.jointBuffer, Offset: 0, Size: wgpu.WholeSize},
			},
		})
		if err != nil {
			return false
		}
		sc.bindGroup = bg
	}

	bonesChanged := rebuilt || grew || !bonesEqual(sc.boneScratch, sc.lastBones)
	if bonesChanged && len(sc.boneScratch) > 0 {
		writeBuffer(device, queue, encoder, sc.boneBuffer, 0, matrixSliceBytes(sc.boneScratch))
		sc.lastBones = append(sc.lastBones[:0], sc.boneScratch...)
	}

	animActive := sc.animation.update(world, device, queue, encoder, sc)

	return bonesChanged || animActive
}

func bonesEqual(a, b []mgl32.Mat4) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func (sc *SkinningCompute) Dispatch(encoder *wgpu.CommandEncoder) {
	if sc.bindGroup == nil || sc.totalJoints == 0 {
		return
	}
	sc.animation.dispatch(encoder)
	pass := encoder.BeginComputePass(&wgpu.ComputePassDescriptor{})
	pass.SetPipeline(sc.pipeline)
	pass.SetBindGroup(0, sc.bindGroup, nil)
	pass.DispatchWorkgroups((sc.totalJoints+63)/64, 1, 1)
	pass.End()
	pass.Release()
}

func (sc *SkinningCompute) ensureBuffers(device *wgpu.Device) bool {
	grew := false
	joints := int(sc.totalJoints)
	if joints < 1 {
		joints = 1
	}
	if sc.boneCap < joints {
		sc.boneBuffer = recreateStorage(device, sc.boneBuffer, "skinning bones", uint64(joints)*matrixBytes)
		sc.boneCap = joints
		sc.boneGeneration++
		grew = true
	}
	if sc.ibmCap < joints {
		sc.ibmBuffer = recreateStorage(device, sc.ibmBuffer, "skinning inverse bind", uint64(joints)*matrixBytes)
		sc.ibmCap = joints
		grew = true
	}
	if sc.jointCap < joints {
		sc.jointBuffer = recreateStorage(device, sc.jointBuffer, "skinning joint matrices", uint64(joints)*matrixBytes)
		sc.jointCap = joints
		sc.generation++
		grew = true
	}
	skins := len(sc.skinData)
	if skins < 1 {
		skins = 1
	}
	if sc.skinDataCap < skins {
		sc.skinDataBuffer = recreateStorage(device, sc.skinDataBuffer, "skinning skin data", uint64(skins)*uint64(unsafe.Sizeof(GpuSkinData{})))
		sc.skinDataCap = skins
		grew = true
	}
	return grew
}

func recreateStorage(device *wgpu.Device, old *wgpu.Buffer, label string, size uint64) *wgpu.Buffer {
	if old != nil {
		old.Release()
	}
	buf, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: label,
		Size:  size,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil
	}
	return buf
}

func matrixSliceBytes(matrices []mgl32.Mat4) []byte {
	if len(matrices) == 0 {
		return nil
	}
	return sliceBytes(matrices)
}

func skinDataBytes(data []GpuSkinData) []byte {
	if len(data) == 0 {
		return nil
	}
	return sliceBytes(data)
}

func putUint64(scratch *[8]byte, value uint64) {
	for i := 0; i < 8; i++ {
		scratch[i] = byte(value >> (8 * i))
	}
}

func mathFloat32Bits(f float32) uint32 {
	return *(*uint32)(unsafe.Pointer(&f))
}
