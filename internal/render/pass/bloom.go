package pass

import (
	_ "embed"
	"fmt"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"

	"github.com/matthewjberger/indigo/render"
)

//go:embed bloom_downsample.wgsl
var bloomDownsampleShader string

//go:embed bloom_upsample.wgsl
var bloomUpsampleShader string

const BloomMipCount = 5

type bloomDownsampleParams struct {
	MipLevel  uint32
	Threshold float32
	Knee      float32
	Pad       uint32
}

type bloomUpsampleParams struct {
	FilterRadius float32
	Pad0         float32
	Pad1         float32
	Pad2         float32
}

type bloomPassState struct {
	device *wgpu.Device

	downsamplePipeline *wgpu.RenderPipeline
	upsamplePipeline   *wgpu.RenderPipeline
	downsampleLayout   *wgpu.BindGroupLayout
	upsampleLayout     *wgpu.BindGroupLayout
	sampler            *wgpu.Sampler

	downsampleParams []*wgpu.Buffer
	upsampleParams   *wgpu.Buffer

	mipTextures []*wgpu.Texture
	mipViews    []*wgpu.TextureView
	mipWidth    []uint32
	mipHeight   []uint32
	current     [2]uint32

	downsampleBg []*wgpu.BindGroup
	upsampleBg   []*wgpu.BindGroup

	threshold    float32
	knee         float32
	filterRadius float32
}

func NewBloomPass(device *wgpu.Device) (*render.Pass, error) {
	state := &bloomPassState{
		device:       device,
		threshold:    1.0,
		knee:         0.5,
		filterRadius: 0.005,
	}

	downsampleLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "bloom downsample bg layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: wgpu.ShaderStageFragment,
				Texture: wgpu.TextureBindingLayout{
					SampleType:    wgpu.TextureSampleTypeFloat,
					ViewDimension: wgpu.TextureViewDimension2D,
				},
			},
			{
				Binding:    1,
				Visibility: wgpu.ShaderStageFragment,
				Sampler:    wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeFiltering},
			},
			{
				Binding:    2,
				Visibility: wgpu.ShaderStageFragment,
				Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform},
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("bloom: downsample layout: %w", err)
	}
	state.downsampleLayout = downsampleLayout

	upsampleLayout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "bloom upsample bg layout",
		Entries: []wgpu.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: wgpu.ShaderStageFragment,
				Texture: wgpu.TextureBindingLayout{
					SampleType:    wgpu.TextureSampleTypeFloat,
					ViewDimension: wgpu.TextureViewDimension2D,
				},
			},
			{
				Binding:    1,
				Visibility: wgpu.ShaderStageFragment,
				Sampler:    wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeFiltering},
			},
			{
				Binding:    2,
				Visibility: wgpu.ShaderStageFragment,
				Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform},
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("bloom: upsample layout: %w", err)
	}
	state.upsampleLayout = upsampleLayout

	sampler, err := device.CreateSampler(&wgpu.SamplerDescriptor{
		Label:         "bloom sampler",
		AddressModeU:  wgpu.AddressModeClampToEdge,
		AddressModeV:  wgpu.AddressModeClampToEdge,
		AddressModeW:  wgpu.AddressModeClampToEdge,
		MagFilter:     wgpu.FilterModeLinear,
		MinFilter:     wgpu.FilterModeLinear,
		MipmapFilter:  wgpu.MipmapFilterModeNearest,
		MaxAnisotropy: 1,
	})
	if err != nil {
		return nil, fmt.Errorf("bloom: sampler: %w", err)
	}
	state.sampler = sampler

	for index := 0; index < BloomMipCount; index++ {
		buffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: "bloom downsample params",
			Size:  uint64(unsafe.Sizeof(bloomDownsampleParams{})),
			Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
		})
		if err != nil {
			return nil, fmt.Errorf("bloom: downsample params %d: %w", index, err)
		}
		state.downsampleParams = append(state.downsampleParams, buffer)
	}

	upsampleParams, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bloom upsample params",
		Size:  uint64(unsafe.Sizeof(bloomUpsampleParams{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("bloom: upsample params: %w", err)
	}
	state.upsampleParams = upsampleParams

	downsampleShader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "bloom downsample shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: bloomDownsampleShader},
	})
	if err != nil {
		return nil, fmt.Errorf("bloom: downsample shader: %w", err)
	}
	defer downsampleShader.Release()
	upsampleShader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "bloom upsample shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: bloomUpsampleShader},
	})
	if err != nil {
		return nil, fmt.Errorf("bloom: upsample shader: %w", err)
	}
	defer upsampleShader.Release()

	downsamplePipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "bloom downsample pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{downsampleLayout},
	})
	if err != nil {
		return nil, fmt.Errorf("bloom: downsample pipeline layout: %w", err)
	}
	defer downsamplePipelineLayout.Release()

	upsamplePipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "bloom upsample pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{upsampleLayout},
	})
	if err != nil {
		return nil, fmt.Errorf("bloom: upsample pipeline layout: %w", err)
	}
	defer upsamplePipelineLayout.Release()

	downsamplePipeline, err := device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "bloom downsample pipeline",
		Layout: downsamplePipelineLayout,
		Vertex: wgpu.VertexState{Module: downsampleShader, EntryPoint: "vertex_main"},
		Primitive: wgpu.PrimitiveState{
			Topology: wgpu.PrimitiveTopologyTriangleList,
			CullMode: wgpu.CullModeNone,
		},
		Multisample: wgpu.MultisampleState{Count: 1, Mask: 0xFFFFFFFF},
		Fragment: &wgpu.FragmentState{
			Module:     downsampleShader,
			EntryPoint: "fragment_main",
			Targets: []wgpu.ColorTargetState{{
				Format:    render.HdrFormat,
				WriteMask: wgpu.ColorWriteMaskAll,
			}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("bloom: downsample pipeline: %w", err)
	}
	state.downsamplePipeline = downsamplePipeline

	upsamplePipeline, err := device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "bloom upsample pipeline",
		Layout: upsamplePipelineLayout,
		Vertex: wgpu.VertexState{Module: upsampleShader, EntryPoint: "vertex_main"},
		Primitive: wgpu.PrimitiveState{
			Topology: wgpu.PrimitiveTopologyTriangleList,
			CullMode: wgpu.CullModeNone,
		},
		Multisample: wgpu.MultisampleState{Count: 1, Mask: 0xFFFFFFFF},
		Fragment: &wgpu.FragmentState{
			Module:     upsampleShader,
			EntryPoint: "fragment_main",
			Targets: []wgpu.ColorTargetState{{
				Format:    render.HdrFormat,
				WriteMask: wgpu.ColorWriteMaskAll,
				Blend: &wgpu.BlendState{
					Color: wgpu.BlendComponent{
						SrcFactor: wgpu.BlendFactorOne,
						DstFactor: wgpu.BlendFactorOne,
						Operation: wgpu.BlendOperationAdd,
					},
					Alpha: wgpu.BlendComponent{
						SrcFactor: wgpu.BlendFactorOne,
						DstFactor: wgpu.BlendFactorOne,
						Operation: wgpu.BlendOperationAdd,
					},
				},
			}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("bloom: upsample pipeline: %w", err)
	}
	state.upsamplePipeline = upsamplePipeline

	return &render.Pass{
		Name:                 "bloom",
		Reads:                []string{"input"},
		Writes:               []string{"output"},
		State:                state,
		Prepare:              bloomPrepare,
		Execute:              bloomExecute,
		InvalidateBindGroups: bloomInvalidate,
		Release:              bloomRelease,
	}, nil
}

func bloomPrepare(s any, context *render.PassContext) error {
	state := s.(*bloomPassState)

	inputView, err := context.TextureView("input")
	if err != nil {
		return err
	}
	_ = inputView

	width, height, err := bloomInputSize(context)
	if err != nil {
		return err
	}
	if width == 0 || height == 0 {
		return nil
	}
	if state.current[0] != width || state.current[1] != height {
		if err := bloomBuildMipChain(state, width, height); err != nil {
			return err
		}
		state.current = [2]uint32{width, height}
		state.downsampleBg = nil
		state.upsampleBg = nil
	}

	for index := 0; index < BloomMipCount; index++ {
		params := bloomDownsampleParams{
			MipLevel:  uint32(index),
			Threshold: state.threshold,
			Knee:      state.knee,
		}
		writeBuffer(context.Device, context.Queue, context.Encoder, state.downsampleParams[index], 0, bytesOf(&params))
	}
	upsampleParams := bloomUpsampleParams{FilterRadius: state.filterRadius}
	writeBuffer(context.Device, context.Queue, context.Encoder, state.upsampleParams, 0, bytesOf(&upsampleParams))

	if len(state.downsampleBg) == 0 {
		state.downsampleBg = make([]*wgpu.BindGroup, BloomMipCount)
		state.upsampleBg = make([]*wgpu.BindGroup, BloomMipCount-1)
		for index := 0; index < BloomMipCount; index++ {
			source := inputView
			if index > 0 {
				source = state.mipViews[index-1]
			}
			bg, err := context.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
				Label:  "bloom downsample bg",
				Layout: state.downsampleLayout,
				Entries: []wgpu.BindGroupEntry{
					{Binding: 0, TextureView: source},
					{Binding: 1, Sampler: state.sampler},
					{Binding: 2, Buffer: state.downsampleParams[index], Offset: 0, Size: uint64(unsafe.Sizeof(bloomDownsampleParams{}))},
				},
			})
			if err != nil {
				return fmt.Errorf("bloom: downsample bg %d: %w", index, err)
			}
			state.downsampleBg[index] = bg
		}
		for index := 0; index < BloomMipCount-1; index++ {
			source := state.mipViews[index+1]
			bg, err := context.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
				Label:  "bloom upsample bg",
				Layout: state.upsampleLayout,
				Entries: []wgpu.BindGroupEntry{
					{Binding: 0, TextureView: source},
					{Binding: 1, Sampler: state.sampler},
					{Binding: 2, Buffer: state.upsampleParams, Offset: 0, Size: uint64(unsafe.Sizeof(bloomUpsampleParams{}))},
				},
			})
			if err != nil {
				return fmt.Errorf("bloom: upsample bg %d: %w", index, err)
			}
			state.upsampleBg[index] = bg
		}
	}
	return nil
}

func bloomExecute(s any, context *render.PassContext) error {
	state := s.(*bloomPassState)
	if state.current[0] == 0 || len(state.mipViews) == 0 {
		return nil
	}

	for index := 0; index < BloomMipCount; index++ {
		view := state.mipViews[index]
		passEnc := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
			Label: "bloom downsample",
			ColorAttachments: []wgpu.RenderPassColorAttachment{{
				View:       view,
				LoadOp:     wgpu.LoadOpClear,
				StoreOp:    wgpu.StoreOpStore,
				ClearValue: wgpu.Color{},
			}},
		})
		passEnc.SetPipeline(state.downsamplePipeline)
		passEnc.SetBindGroup(0, state.downsampleBg[index], nil)
		passEnc.Draw(3, 1, 0, 0)
		passEnc.End()
		passEnc.Release()
	}

	for index := BloomMipCount - 2; index >= 0; index-- {
		view := state.mipViews[index]
		passEnc := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
			Label: "bloom upsample",
			ColorAttachments: []wgpu.RenderPassColorAttachment{{
				View:    view,
				LoadOp:  wgpu.LoadOpLoad,
				StoreOp: wgpu.StoreOpStore,
			}},
		})
		passEnc.SetPipeline(state.upsamplePipeline)
		passEnc.SetBindGroup(0, state.upsampleBg[index], nil)
		passEnc.Draw(3, 1, 0, 0)
		passEnc.End()
		passEnc.Release()
	}

	outputAttachment, err := context.ColorAttachment("output")
	if err != nil {
		return err
	}
	_ = outputAttachment
	return nil
}

func bloomInvalidate(s any) {
	state := s.(*bloomPassState)
	for index := range state.downsampleBg {
		if state.downsampleBg[index] != nil {
			state.downsampleBg[index].Release()
		}
	}
	state.downsampleBg = nil
	for index := range state.upsampleBg {
		if state.upsampleBg[index] != nil {
			state.upsampleBg[index].Release()
		}
	}
	state.upsampleBg = nil
}

func bloomRelease(s any) {
	state := s.(*bloomPassState)
	bloomInvalidate(state)
	for index := range state.mipViews {
		if state.mipViews[index] != nil {
			state.mipViews[index].Release()
		}
		if state.mipTextures[index] != nil {
			state.mipTextures[index].Release()
		}
	}
	state.mipViews = nil
	state.mipTextures = nil
	if state.upsampleParams != nil {
		state.upsampleParams.Release()
	}
	for index := range state.downsampleParams {
		state.downsampleParams[index].Release()
	}
	if state.sampler != nil {
		state.sampler.Release()
	}
	if state.downsamplePipeline != nil {
		state.downsamplePipeline.Release()
	}
	if state.upsamplePipeline != nil {
		state.upsamplePipeline.Release()
	}
	if state.downsampleLayout != nil {
		state.downsampleLayout.Release()
	}
	if state.upsampleLayout != nil {
		state.upsampleLayout.Release()
	}
}

func bloomInputSize(context *render.PassContext) (uint32, uint32, error) {
	id, ok := context.Slots["input"]
	if !ok {
		return 0, 0, fmt.Errorf("bloom: input slot not bound")
	}
	handle := context.Resources.Handles[id]
	return handle.Width, handle.Height, nil
}

func bloomBuildMipChain(state *bloomPassState, width, height uint32) error {
	for index := range state.mipViews {
		if state.mipViews[index] != nil {
			state.mipViews[index].Release()
		}
		if state.mipTextures[index] != nil {
			state.mipTextures[index].Release()
		}
	}
	state.mipTextures = make([]*wgpu.Texture, 0, BloomMipCount)
	state.mipViews = make([]*wgpu.TextureView, 0, BloomMipCount)
	state.mipWidth = make([]uint32, 0, BloomMipCount)
	state.mipHeight = make([]uint32, 0, BloomMipCount)

	currentWidth := width
	currentHeight := height
	for index := 0; index < BloomMipCount; index++ {
		currentWidth = currentWidth / 2
		currentHeight = currentHeight / 2
		if currentWidth < 1 {
			currentWidth = 1
		}
		if currentHeight < 1 {
			currentHeight = 1
		}
		tex, err := state.device.CreateTexture(&wgpu.TextureDescriptor{
			Label: "bloom mip",
			Size: wgpu.Extent3D{
				Width:              currentWidth,
				Height:             currentHeight,
				DepthOrArrayLayers: 1,
			},
			MipLevelCount: 1,
			SampleCount:   1,
			Dimension:     wgpu.TextureDimension2D,
			Format:        render.HdrFormat,
			Usage:         wgpu.TextureUsageRenderAttachment | wgpu.TextureUsageTextureBinding,
		})
		if err != nil {
			return fmt.Errorf("bloom: mip %d texture: %w", index, err)
		}
		view, err := tex.CreateView(nil)
		if err != nil {
			tex.Release()
			return fmt.Errorf("bloom: mip %d view: %w", index, err)
		}
		state.mipTextures = append(state.mipTextures, tex)
		state.mipViews = append(state.mipViews, view)
		state.mipWidth = append(state.mipWidth, currentWidth)
		state.mipHeight = append(state.mipHeight, currentHeight)
	}
	return nil
}

func BloomMipView(p *render.Pass) *wgpu.TextureView {
	state, ok := p.State.(*bloomPassState)
	if !ok || len(state.mipViews) == 0 {
		return nil
	}
	return state.mipViews[0]
}

func AddBloomPass(renderer *render.Renderer) (*render.Pass, error) {
	pass, err := NewBloomPass(renderer.Device)
	if err != nil {
		return nil, err
	}
	if err := renderer.Graph.AddPass(pass, []render.SlotBinding{
		{Slot: "input", ResourceID: renderer.SceneColorID},
		{Slot: "output", ResourceID: renderer.SceneColorID},
	}); err != nil {
		return nil, err
	}
	return pass, nil
}
