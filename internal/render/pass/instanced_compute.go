package pass

import (
	"fmt"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/go-gl/mathgl/mgl32"

	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/render/asset"
	"github.com/matthewjberger/indigo/transform"
)

type instancedComputeEntity struct {
	capacity      int
	uploadedCount int
	localBuffer   *wgpu.Buffer
	worldBuffer   *wgpu.Buffer
	uniformBuffer *wgpu.Buffer
	bindGroup     *wgpu.BindGroup
	instanceCount uint32
	generation    uint64
	seen          bool
}

type InstancedCompute struct {
	pipeline *wgpu.ComputePipeline
	layout   *wgpu.BindGroupLayout
	entities map[ecs.Entity]*instancedComputeEntity
}

type InstancedComputeResource struct {
	Compute *InstancedCompute
}

func NewInstancedCompute(device *wgpu.Device) (*InstancedCompute, error) {
	module, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "instanced_transform shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: instancedTransformShader},
	})
	if err != nil {
		return nil, fmt.Errorf("instanced_compute: shader: %w", err)
	}
	defer module.Release()

	layout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "instanced_transform layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("instanced_compute: layout: %w", err)
	}
	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "instanced_transform pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{layout},
	})
	if err != nil {
		layout.Release()
		return nil, fmt.Errorf("instanced_compute: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()
	pipeline, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Layout:  pipelineLayout,
		Compute: wgpu.ProgrammableStageDescriptor{Module: module, EntryPoint: "main"},
	})
	if err != nil {
		layout.Release()
		return nil, fmt.Errorf("instanced_compute: pipeline: %w", err)
	}
	return &InstancedCompute{
		pipeline: pipeline,
		layout:   layout,
		entities: make(map[ecs.Entity]*instancedComputeEntity),
	}, nil
}

func AddInstancedComputePass(renderer *render.Renderer, ic *InstancedCompute) (*render.Pass, error) {
	pass := &render.Pass{
		Name:  "instanced_compute",
		State: ic,
		Execute: func(state any, context *render.PassContext) error {
			compute := state.(*InstancedCompute)
			compute.prepare(context.World, context.Device, context.Queue, context.Encoder)
			compute.dispatch(context.Encoder)
			return nil
		},
		Release: func(state any) { state.(*InstancedCompute).Release() },
	}
	if err := renderer.Graph.AddPass(pass, nil); err != nil {
		return nil, err
	}
	return pass, nil
}

func (ic *InstancedCompute) WorldBuffer(entity ecs.Entity) *wgpu.Buffer {
	if es := ic.entities[entity]; es != nil {
		return es.worldBuffer
	}
	return nil
}

func (ic *InstancedCompute) InstanceCount(entity ecs.Entity) uint32 {
	if es := ic.entities[entity]; es != nil {
		return es.instanceCount
	}
	return 0
}

func (ic *InstancedCompute) Generation(entity ecs.Entity) uint64 {
	if es := ic.entities[entity]; es != nil {
		return es.generation
	}
	return 0
}

func (ic *InstancedCompute) prepare(world *ecs.World, device *wgpu.Device, queue *wgpu.Queue, encoder *wgpu.CommandEncoder) {
	for _, es := range ic.entities {
		es.seen = false
	}
	matBytes := int(unsafe.Sizeof(mgl32.Mat4{}))
	mask := ecs.MustMaskOf[asset.InstancedMesh](world) | ecs.MustMaskOf[transform.GlobalTransform](world)
	world.ForEach(mask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		inst, _ := ecs.Get[asset.InstancedMesh](world, entity)
		if inst == nil || len(inst.Instances) == 0 {
			return
		}
		global, ok := ecs.Get[transform.GlobalTransform](world, entity)
		if !ok {
			return
		}
		es := ic.entities[entity]
		if es == nil {
			es = &instancedComputeEntity{}
			ic.entities[entity] = es
		}
		es.seen = true
		count := len(inst.Instances)
		if es.capacity < count {
			if es.localBuffer != nil {
				es.localBuffer.Release()
			}
			if es.worldBuffer != nil {
				es.worldBuffer.Release()
			}
			localBuf, err := device.CreateBuffer(&wgpu.BufferDescriptor{
				Label: "instanced local matrices",
				Size:  uint64(count) * uint64(matBytes),
				Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
			})
			if err != nil {
				return
			}
			worldBuf, err := device.CreateBuffer(&wgpu.BufferDescriptor{
				Label: "instanced world matrices",
				Size:  uint64(count) * uint64(matBytes),
				Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
			})
			if err != nil {
				localBuf.Release()
				return
			}
			es.localBuffer = localBuf
			es.worldBuffer = worldBuf
			es.capacity = count
			es.uploadedCount = 0
			es.generation++
			if es.bindGroup != nil {
				es.bindGroup.Release()
				es.bindGroup = nil
			}
		}
		if es.uniformBuffer == nil {
			buf, err := device.CreateBuffer(&wgpu.BufferDescriptor{
				Label: "instanced compute uniform",
				Size:  uint64(unsafe.Sizeof(instancedComputeUniform{})),
				Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
			})
			if err != nil {
				return
			}
			es.uniformBuffer = buf
		}
		if es.uploadedCount != count {
			writeBuffer(device, queue, encoder, es.localBuffer, 0, unsafe.Slice((*byte)(unsafe.Pointer(&inst.Instances[0])), count*matBytes))
			es.uploadedCount = count
		}
		cu := instancedComputeUniform{ParentTransform: global.Matrix, InstanceCount: uint32(count)}
		writeBuffer(device, queue, encoder, es.uniformBuffer, 0, bytesOf(&cu))
		if es.bindGroup == nil {
			bg, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
				Label:  "instanced compute bg",
				Layout: ic.layout,
				Entries: []wgpu.BindGroupEntry{
					{Binding: 0, Buffer: es.localBuffer, Offset: 0, Size: wgpu.WholeSize},
					{Binding: 1, Buffer: es.worldBuffer, Offset: 0, Size: wgpu.WholeSize},
					{Binding: 2, Buffer: es.uniformBuffer, Offset: 0, Size: uint64(unsafe.Sizeof(instancedComputeUniform{}))},
				},
			})
			if err != nil {
				return
			}
			es.bindGroup = bg
		}
		es.instanceCount = uint32(count)
	})
	for entity, es := range ic.entities {
		if es.seen {
			continue
		}
		releaseInstancedComputeEntity(es)
		delete(ic.entities, entity)
	}
}

func (ic *InstancedCompute) dispatch(encoder *wgpu.CommandEncoder) {
	if len(ic.entities) == 0 {
		return
	}
	pass := encoder.BeginComputePass(&wgpu.ComputePassDescriptor{})
	pass.SetPipeline(ic.pipeline)
	for _, es := range ic.entities {
		if es.bindGroup == nil || es.instanceCount == 0 {
			continue
		}
		pass.SetBindGroup(0, es.bindGroup, nil)
		pass.DispatchWorkgroups((es.instanceCount+63)/64, 1, 1)
	}
	pass.End()
	pass.Release()
}

func releaseInstancedComputeEntity(es *instancedComputeEntity) {
	for _, b := range []*wgpu.Buffer{es.localBuffer, es.worldBuffer, es.uniformBuffer} {
		if b != nil {
			b.Release()
		}
	}
	if es.bindGroup != nil {
		es.bindGroup.Release()
	}
}

func (ic *InstancedCompute) Release() {
	for _, es := range ic.entities {
		releaseInstancedComputeEntity(es)
	}
	ic.entities = nil
	if ic.pipeline != nil {
		ic.pipeline.Release()
		ic.pipeline = nil
	}
	if ic.layout != nil {
		ic.layout.Release()
		ic.layout = nil
	}
}
