package pass

import (
	_ "embed"
	"fmt"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/go-gl/mathgl/mgl32"
)

//go:embed mesh_culling.wgsl
var meshCullingShader string

type CullingUniforms struct {
	FrustumPlanes      [6]mgl32.Vec4
	ViewProjection     mgl32.Mat4
	ScreenSize         mgl32.Vec2
	MinScreenPixelSize float32
	ProjectionScaleY   float32
	Enabled            uint32
	Pad0               uint32
	Pad1               uint32
	Pad2               uint32
}

type bucketCullParams struct {
	BoundsCenter mgl32.Vec3
	BoundsRadius float32
	ObjectCount  uint32
	Pad0         uint32
	Pad1         uint32
	Pad2         uint32
}

type meshCullingPipeline struct {
	cullPipeline   *wgpu.ComputePipeline
	frameLayout    *wgpu.BindGroupLayout
	bucketLayout   *wgpu.BindGroupLayout
	frameBuffer    *wgpu.Buffer
	frameBindGroup *wgpu.BindGroup
}

func newMeshCullingPipeline(device *wgpu.Device) (*meshCullingPipeline, error) {
	frameLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "mesh_culling frame layout",
		Entries: []wgpu.BindGroupLayoutEntry{{
			Binding:    0,
			Visibility: wgpu.ShaderStageCompute,
			Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform},
		}},
	})
	if err != nil {
		return nil, fmt.Errorf("mesh_culling: frame layout: %w", err)
	}

	bucketLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "mesh_culling bucket layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform}},
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},
		},
	})
	if err != nil {
		frameLayout.Release()
		return nil, fmt.Errorf("mesh_culling: bucket layout: %w", err)
	}

	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "mesh_culling shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: meshCullingShader},
	})
	if err != nil {
		frameLayout.Release()
		bucketLayout.Release()
		return nil, fmt.Errorf("mesh_culling: shader: %w", err)
	}
	defer shader.Release()

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "mesh_culling pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{frameLayout, bucketLayout},
	})
	if err != nil {
		frameLayout.Release()
		bucketLayout.Release()
		return nil, fmt.Errorf("mesh_culling: pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	cullPipeline, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Layout: pipelineLayout,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     shader,
			EntryPoint: "cull",
		},
	})
	if err != nil {
		frameLayout.Release()
		bucketLayout.Release()
		return nil, fmt.Errorf("mesh_culling: cull pipeline: %w", err)
	}

	frameBuffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "mesh_culling frame uniform",
		Size:  uint64(unsafe.Sizeof(CullingUniforms{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		cullPipeline.Release()
		frameLayout.Release()
		bucketLayout.Release()
		return nil, fmt.Errorf("mesh_culling: frame buffer: %w", err)
	}

	frameBindGroup, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:   "mesh_culling frame bg",
		Layout:  frameLayout,
		Entries: []wgpu.BindGroupEntry{{Binding: 0, Buffer: frameBuffer, Offset: 0, Size: uint64(unsafe.Sizeof(CullingUniforms{}))}},
	})
	if err != nil {
		frameBuffer.Release()
		cullPipeline.Release()
		frameLayout.Release()
		bucketLayout.Release()
		return nil, fmt.Errorf("mesh_culling: frame bg: %w", err)
	}

	return &meshCullingPipeline{
		cullPipeline:   cullPipeline,
		frameLayout:    frameLayout,
		bucketLayout:   bucketLayout,
		frameBuffer:    frameBuffer,
		frameBindGroup: frameBindGroup,
	}, nil
}

func (p *meshCullingPipeline) release() {
	if p.frameBindGroup != nil {
		p.frameBindGroup.Release()
	}
	if p.frameBuffer != nil {
		p.frameBuffer.Release()
	}
	if p.cullPipeline != nil {
		p.cullPipeline.Release()
	}
	if p.frameLayout != nil {
		p.frameLayout.Release()
	}
	if p.bucketLayout != nil {
		p.bucketLayout.Release()
	}
}

func ExtractFrustumPlanes(viewProj mgl32.Mat4) [6]mgl32.Vec4 {
	row := func(r int) mgl32.Vec4 {
		return mgl32.Vec4{viewProj[r], viewProj[4+r], viewProj[8+r], viewProj[12+r]}
	}
	r0 := row(0)
	r1 := row(1)
	r2 := row(2)
	r3 := row(3)
	planes := [6]mgl32.Vec4{
		r3.Add(r0),
		r3.Sub(r0),
		r3.Add(r1),
		r3.Sub(r1),
		r2,
		r3.Sub(r2),
	}
	for i := range planes {
		n := mgl32.Vec3{planes[i][0], planes[i][1], planes[i][2]}.Len()
		if n > 1e-6 {
			inv := 1.0 / n
			planes[i] = mgl32.Vec4{planes[i][0] * inv, planes[i][1] * inv, planes[i][2] * inv, planes[i][3] * inv}
		}
	}
	return planes
}
