package pass

import (
	_ "embed"
	"fmt"
	"hash/fnv"
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

// GpuSkinData is the per-unique-skin descriptor the compute reads
// to locate a joint's inputs / output in the shared buffers.
type GpuSkinData struct {
	JointCount      uint32
	BaseBoneIndex   uint32
	BaseIbmIndex    uint32
	BaseOutputIndex uint32
}

// SkinningCompute gathers every skinned entity's joints into shared
// engine-wide buffers and runs one compute dispatch over all joints
// to produce joint_matrices = bone_transform * inverse_bind. The
// skinned mesh + shadow passes index their slice of the output via
// a per-entity joint offset.
//
// Static data (skin_data, inverse_bind_matrices, the deduped joint
// entity list) is rebuilt only when the set of skinned entities
// changes. Bone transforms are re-collected every frame.
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

	// generation bumps whenever jointBuffer is reallocated, so the
	// render passes that cache a bind group referencing it can
	// detect staleness and rebuild.
	generation uint64
}

// Generation returns a counter that changes whenever the shared
// joint-matrix buffer is reallocated. Render passes compare it to
// drop bind groups that reference a freed buffer.
func (sc *SkinningCompute) Generation() uint64 {
	return sc.generation
}

// SkinningComputeResource exposes the compute on the engine world
// so the skinned mesh + shadow passes can read its output buffer
// and per-entity joint offsets.
type SkinningComputeResource struct {
	Compute *SkinningCompute
}

// AddSkinningComputePass registers the skinning dispatch as a
// render-graph pass and publishes the SkinningCompute as a world
// resource. Register before the shadow + skinned mesh passes so
// the joint-matrix buffer is populated before they sample it
// (passes with no slot deps run in registration order).
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

// NewSkinningCompute builds the compute pipeline + bind-group
// layout. Shared buffers grow lazily on the first frame that has
// skinned entities.
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

	return &SkinningCompute{
		pipeline:          pipeline,
		layout:            layout,
		entitySkinIndices: make(map[ecs.Entity]uint32),
	}, nil
}

// Release frees the pipeline, layout, and shared buffers.
func (sc *SkinningCompute) Release() {
	for _, buf := range []*wgpu.Buffer{sc.boneBuffer, sc.ibmBuffer, sc.skinDataBuffer, sc.jointBuffer} {
		if buf != nil {
			buf.Release()
		}
	}
	sc.boneBuffer, sc.ibmBuffer, sc.skinDataBuffer, sc.jointBuffer = nil, nil, nil, nil
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

// JointMatrixBuffer is the shared output buffer the render passes
// bind. Nil until the first frame with skinned entities.
func (sc *SkinningCompute) JointMatrixBuffer() *wgpu.Buffer {
	return sc.jointBuffer
}

// JointOffset returns the base index into the joint-matrix buffer
// for the entity's skin (its base_output_index). The render shader
// reads joint_matrices[JointOffset + joint_index].
func (sc *SkinningCompute) JointOffset(entity ecs.Entity) uint32 {
	skinIndex, ok := sc.entitySkinIndices[entity]
	if !ok || int(skinIndex) >= len(sc.skinData) {
		return 0
	}
	return sc.skinData[skinIndex].BaseOutputIndex
}

// needsRebuild reports whether the set of skinned entities changed
// since the last static rebuild.
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

// rebuildStaticData walks the skinned entities, dedupes joint
// entities into a single bone array, dedupes identical rig configs
// into shared skin_data + inverse-bind entries, and records each
// entity's skin index. Mirrors nightshade's SkinningCache.
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

// hashSkin produces a config hash from the joint entity ids + the
// inverse bind matrices, so two entities sharing a rig collapse to
// one skin_data entry.
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

// collectBoneTransforms reads each deduped joint entity's current
// world transform into the scratch slice (identity when missing).
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

// Prepare runs every frame from the skinned mesh pass: rebuilds
// static data when the entity set changed, (re)allocates shared
// buffers, uploads static + per-frame data. Returns false when
// there are no skinned entities to dispatch.
func (sc *SkinningCompute) Prepare(world *ecs.World, device *wgpu.Device, queue *wgpu.Queue, encoder *wgpu.CommandEncoder) bool {
	rebuilt := false
	if sc.needsRebuild(world) {
		sc.rebuildStaticData(world)
		rebuilt = true
	}
	if sc.totalJoints == 0 || len(sc.skinData) == 0 {
		return false
	}
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
	if len(sc.boneScratch) > 0 {
		writeBuffer(device, queue, encoder, sc.boneBuffer, 0, matrixSliceBytes(sc.boneScratch))
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
	return true
}

// Dispatch encodes the single skinning dispatch. Prepare must have
// returned true this frame.
func (sc *SkinningCompute) Dispatch(encoder *wgpu.CommandEncoder) {
	if sc.bindGroup == nil || sc.totalJoints == 0 {
		return
	}
	pass := encoder.BeginComputePass(&wgpu.ComputePassDescriptor{})
	pass.SetPipeline(sc.pipeline)
	pass.SetBindGroup(0, sc.bindGroup, nil)
	pass.DispatchWorkgroups((sc.totalJoints+63)/64, 1, 1)
	pass.End()
	pass.Release()
}

// ensureBuffers grows the four shared buffers to fit the current
// joint / skin counts. Returns true when any buffer was
// reallocated (so callers re-upload static data + rebuild the
// bind group).
func (sc *SkinningCompute) ensureBuffers(device *wgpu.Device) bool {
	grew := false
	joints := int(sc.totalJoints)
	if joints < 1 {
		joints = 1
	}
	if sc.boneCap < joints {
		sc.boneBuffer = recreateStorage(device, sc.boneBuffer, "skinning bones", uint64(joints)*matrixBytes)
		sc.boneCap = joints
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
	return unsafe.Slice((*byte)(unsafe.Pointer(&matrices[0])), len(matrices)*int(matrixBytes))
}

func skinDataBytes(data []GpuSkinData) []byte {
	if len(data) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data)*int(unsafe.Sizeof(GpuSkinData{})))
}

func putUint64(scratch *[8]byte, value uint64) {
	for i := 0; i < 8; i++ {
		scratch[i] = byte(value >> (8 * i))
	}
}

func mathFloat32Bits(f float32) uint32 {
	return *(*uint32)(unsafe.Pointer(&f))
}
