package pass

import (
	_ "embed"
	"fmt"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"

	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/render/asset"
)

//go:embed mesh_depth_prepass.wgsl
var meshDepthPrepassShader string

type depthPrepass struct {
	pipeline   *wgpu.RenderPipeline
	matsLayout *wgpu.BindGroupLayout
	matsBg     *wgpu.BindGroup
}

func newDepthPrepassPipeline(device *wgpu.Device, viewProjLayout, handleLayout *wgpu.BindGroupLayout, materialRegistry *asset.MaterialRegistry) (*depthPrepass, error) {
	module, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "mesh depth prepass shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: meshDepthPrepassShader},
	})
	if err != nil {
		return nil, fmt.Errorf("mesh depth prepass: shader: %w", err)
	}
	defer module.Release()

	matsLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "mesh depth prepass materials layout",
		Entries: []wgpu.BindGroupLayoutEntry{{
			Binding:    0,
			Visibility: wgpu.ShaderStageVertex,
			Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage},
		}},
	})
	if err != nil {
		return nil, fmt.Errorf("mesh depth prepass: materials layout: %w", err)
	}

	matsBg, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "mesh depth prepass materials bg",
		Layout: matsLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: materialRegistry.Buffer(), Offset: 0, Size: wgpu.WholeSize},
		},
	})
	if err != nil {
		matsLayout.Release()
		return nil, fmt.Errorf("mesh depth prepass: materials bg: %w", err)
	}

	layout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "mesh depth prepass pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{viewProjLayout, handleLayout, matsLayout},
	})
	if err != nil {
		matsBg.Release()
		matsLayout.Release()
		return nil, fmt.Errorf("mesh depth prepass: pipeline layout: %w", err)
	}
	defer layout.Release()

	pipeline, err := device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "mesh depth prepass pipeline",
		Layout: layout,
		Vertex: wgpu.VertexState{
			Module:     module,
			EntryPoint: "vertex_main",
			Buffers: []wgpu.VertexBufferLayout{{
				ArrayStride: uint64(unsafe.Sizeof(asset.MeshVertex{})),
				StepMode:    wgpu.VertexStepModeVertex,
				Attributes: []wgpu.VertexAttribute{
					{Format: wgpu.VertexFormatFloat32x4, Offset: 0, ShaderLocation: 0},
					{Format: wgpu.VertexFormatFloat32x4, Offset: 16, ShaderLocation: 1},
					{Format: wgpu.VertexFormatFloat32x4, Offset: 32, ShaderLocation: 2},
					{Format: wgpu.VertexFormatFloat32x4, Offset: 48, ShaderLocation: 3},
					{Format: wgpu.VertexFormatFloat32x4, Offset: 64, ShaderLocation: 4},
				},
			}},
		},
		Primitive: wgpu.PrimitiveState{
			Topology:  wgpu.PrimitiveTopologyTriangleList,
			FrontFace: wgpu.FrontFaceCCW,
			CullMode:  wgpu.CullModeBack,
		},
		DepthStencil: &wgpu.DepthStencilState{
			Format:            render.DepthFormat,
			DepthWriteEnabled: true,
			DepthCompare:      wgpu.CompareFunctionLess,
			StencilFront:      wgpu.StencilFaceState{Compare: wgpu.CompareFunctionAlways},
			StencilBack:       wgpu.StencilFaceState{Compare: wgpu.CompareFunctionAlways},
		},
		Multisample: wgpu.MultisampleState{Count: 1, Mask: 0xFFFFFFFF},
		Fragment: &wgpu.FragmentState{
			Module:     module,
			EntryPoint: "fragment_main",
			Targets:    []wgpu.ColorTargetState{},
		},
	})
	if err != nil {
		matsBg.Release()
		matsLayout.Release()
		return nil, fmt.Errorf("mesh depth prepass: pipeline: %w", err)
	}
	return &depthPrepass{
		pipeline:   pipeline,
		matsLayout: matsLayout,
		matsBg:     matsBg,
	}, nil
}

func (d *depthPrepass) release() {
	if d == nil {
		return
	}
	if d.matsBg != nil {
		d.matsBg.Release()
	}
	if d.matsLayout != nil {
		d.matsLayout.Release()
	}
	if d.pipeline != nil {
		d.pipeline.Release()
	}
}
