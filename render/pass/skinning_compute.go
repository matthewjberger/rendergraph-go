package pass

import (
	_ "embed"
	"fmt"
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

// SkinningCompute owns the compute pipeline + bind-group layout
// that multiplies per-skin bone transforms by inverse-bind matrices
// each frame to produce the joint-matrix buffer the skinned mesh
// vertex shader reads. One dispatch per Skin, sized to the skin's
// joint count.
type SkinningCompute struct {
	pipeline       *wgpu.ComputePipeline
	layout         *wgpu.BindGroupLayout
	bindGroupCache map[*wgpu.Buffer]*wgpu.BindGroup
}

// SkinningComputeResource exposes the compute on the world.
type SkinningComputeResource struct {
	Compute *SkinningCompute
}

// AddSkinningComputePass registers the compute as a render-graph
// pass so the graph schedules it on the frame encoder. No slot
// reads / writes — the joint matrix buffers live on Skin
// components, not in the resource graph. Register before shadow
// and mesh passes so its writes complete before they sample.
func AddSkinningComputePass(renderer *render.Renderer, sc *SkinningCompute) (*render.Pass, error) {
	pass := &render.Pass{
		Name:  "skinning_compute",
		State: sc,
		Execute: func(state any, context *render.PassContext) error {
			compute := state.(*SkinningCompute)
			DispatchSkinningCompute(compute, context.World, context.Device, context.Encoder)
			return nil
		},
		Release: func(state any) {
			compute := state.(*SkinningCompute)
			compute.Release()
		},
	}
	if err := renderer.Graph.AddPass(pass, nil); err != nil {
		return nil, err
	}
	return pass, nil
}

// NewSkinningCompute builds the shared compute pipeline + layout.
// Per-skin bind groups are built lazily on the first dispatch
// for that skin and cached against the skin's joint buffer.
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
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},
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
		pipeline:       pipeline,
		layout:         layout,
		bindGroupCache: make(map[*wgpu.Buffer]*wgpu.BindGroup),
	}, nil
}

// Release frees the pipeline + layout + cached bind groups.
func (sc *SkinningCompute) Release() {
	for _, bg := range sc.bindGroupCache {
		bg.Release()
	}
	sc.bindGroupCache = nil
	if sc.pipeline != nil {
		sc.pipeline.Release()
		sc.pipeline = nil
	}
	if sc.layout != nil {
		sc.layout.Release()
		sc.layout = nil
	}
}

// UploadSkinBones writes each Skin's per-frame bone transforms
// (joint entity GlobalTransform matrices) into its bone-transforms
// buffer. Also uploads InverseBindMatrices the first time a skin
// is seen. Runs in the engine schedule before the renderer
// dispatches the compute.
func UploadSkinBones(world *ecs.World) {
	if !ecs.HasResource[render.RendererResource](world) {
		return
	}
	queue := ecs.MustResource[render.RendererResource](world).Renderer.Queue
	mask := ecs.MustMaskOf[asset.SkinnedMesh](world)
	matSize := int(unsafe.Sizeof(mgl32.Mat4{}))
	world.ForEach(mask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		sm, _ := ecs.Get[asset.SkinnedMesh](world, entity)
		if sm == nil || sm.Skin == nil {
			return
		}
		skin := sm.Skin
		if skin.BoneTransformsBuffer == nil || skin.InverseBindBuffer == nil {
			return
		}
		if len(skin.Joints) != len(skin.InverseBindMatrices) || len(skin.Joints) != len(skin.BoneScratch) {
			return
		}
		if !skin.InverseBindUploaded && len(skin.InverseBindMatrices) > 0 {
			data := unsafe.Slice((*byte)(unsafe.Pointer(&skin.InverseBindMatrices[0])), len(skin.InverseBindMatrices)*matSize)
			queue.WriteBuffer(skin.InverseBindBuffer, 0, data)
			skin.InverseBindUploaded = true
		}
		for index, jointEntity := range skin.Joints {
			global, ok := ecs.Get[transform.GlobalTransform](world, jointEntity)
			if !ok {
				skin.BoneScratch[index] = mgl32.Ident4()
				continue
			}
			skin.BoneScratch[index] = global.Matrix
		}
		if len(skin.BoneScratch) == 0 {
			return
		}
		data := unsafe.Slice((*byte)(unsafe.Pointer(&skin.BoneScratch[0])), len(skin.BoneScratch)*matSize)
		queue.WriteBuffer(skin.BoneTransformsBuffer, 0, data)
	})
}

// DispatchSkinningCompute encodes one compute dispatch per Skin,
// computing joint_matrices = bone_transforms * inverse_bind on the
// GPU. Must run on the renderer encoder before any pass that reads
// JointMatrixBuffer (skinned_mesh, shadow_depth skinned path).
func DispatchSkinningCompute(sc *SkinningCompute, world *ecs.World, device *wgpu.Device, encoder *wgpu.CommandEncoder) {
	if sc == nil || sc.pipeline == nil {
		return
	}
	mask := ecs.MustMaskOf[asset.SkinnedMesh](world)
	var skins []*asset.Skin
	world.ForEach(mask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		sm, _ := ecs.Get[asset.SkinnedMesh](world, entity)
		if sm == nil || sm.Skin == nil || sm.Skin.JointMatrixBuffer == nil {
			return
		}
		skins = append(skins, sm.Skin)
	})
	if len(skins) == 0 {
		return
	}

	pass := encoder.BeginComputePass(&wgpu.ComputePassDescriptor{})
	pass.SetPipeline(sc.pipeline)
	for _, skin := range skins {
		count := uint32(len(skin.Joints))
		if count == 0 {
			continue
		}
		bg := sc.bindGroupCache[skin.JointMatrixBuffer]
		if bg == nil {
			created, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
				Label:  "skinning_compute bg",
				Layout: sc.layout,
				Entries: []wgpu.BindGroupEntry{
					{Binding: 0, Buffer: skin.BoneTransformsBuffer, Offset: 0, Size: wgpu.WholeSize},
					{Binding: 1, Buffer: skin.InverseBindBuffer, Offset: 0, Size: wgpu.WholeSize},
					{Binding: 2, Buffer: skin.JointMatrixBuffer, Offset: 0, Size: wgpu.WholeSize},
				},
			})
			if err != nil {
				continue
			}
			sc.bindGroupCache[skin.JointMatrixBuffer] = created
			bg = created
		}
		pass.SetBindGroup(0, bg, nil)
		groups := (count + 63) / 64
		pass.DispatchWorkgroups(groups, 1, 1)
	}
	pass.End()
	pass.Release()
}
