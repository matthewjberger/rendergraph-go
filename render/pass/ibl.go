package pass

import (
	_ "embed"
	"fmt"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"
)

//go:embed brdf_lut.wgsl
var brdfLutShader string

//go:embed procedural_to_cubemap.wgsl
var proceduralCubemapShader string

//go:embed cubemap_mipgen.wgsl
var cubemapMipgenShader string

//go:embed filter_envmap.wgsl
var filterEnvmapShader string

// IBL grid sizes.
const (
	BrdfLutSize           uint32 = 256
	IrradianceSize        uint32 = 64
	PrefilteredSize       uint32 = 512
	PrefilteredMipLevels  uint32 = 5
	IrradianceSamples     uint32 = 1024
	PrefilteredSamples    uint32 = 512
	ProceduralCubemapSize uint32 = 1024
)

// IBL owns every GPU resource the image-based-lighting pipeline
// needs: the pre-integrated BRDF LUT, the captured environment
// cubemap (with full mip chain), the Lambertian-filtered
// irradiance cubemap, and the GGX-prefiltered specular cubemap.
// One-time generation at startup; the mesh fragment shader
// samples these every frame to evaluate ambient diffuse + ambient
// specular.
type IBL struct {
	BrdfLut         *wgpu.Texture
	BrdfLutView     *wgpu.TextureView
	Cubemap         *wgpu.Texture
	CubemapView     *wgpu.TextureView
	Irradiance      *wgpu.Texture
	IrradianceView  *wgpu.TextureView
	Prefiltered     *wgpu.Texture
	PrefilteredView *wgpu.TextureView
	Sampler         *wgpu.Sampler
}

// IBLResource is the typed wrapper apps put on the engine world
// so passes can look the bundle up via [ecs.MustResource].
type IBLResource struct {
	IBL *IBL
}

// NewIBL allocates every IBL texture and runs the four compute
// dispatches that fill them: BRDF LUT integration, procedural sky
// -> cubemap capture, cubemap mip generation, and Lambertian +
// GGX environment filtering. Designed to be called once at
// startup; results are reused every frame.
func NewIBL(device *wgpu.Device, queue *wgpu.Queue) (*IBL, error) {
	ibl := &IBL{}

	sampler, err := device.CreateSampler(&wgpu.SamplerDescriptor{
		Label:         "ibl sampler",
		AddressModeU:  wgpu.AddressModeClampToEdge,
		AddressModeV:  wgpu.AddressModeClampToEdge,
		AddressModeW:  wgpu.AddressModeClampToEdge,
		MagFilter:     wgpu.FilterModeLinear,
		MinFilter:     wgpu.FilterModeLinear,
		MipmapFilter:  wgpu.MipmapFilterModeLinear,
		LodMinClamp:   0,
		LodMaxClamp:   float32(PrefilteredMipLevels),
		MaxAnisotropy: 1,
	})
	if err != nil {
		return nil, fmt.Errorf("ibl: sampler: %w", err)
	}
	ibl.Sampler = sampler

	if err := generateBrdfLut(ibl, device, queue); err != nil {
		ibl.Release()
		return nil, err
	}
	if err := captureProceduralCubemap(ibl, device, queue); err != nil {
		ibl.Release()
		return nil, err
	}
	if err := generateCubemapMipmaps(ibl.Cubemap, device, queue); err != nil {
		ibl.Release()
		return nil, err
	}
	if err := filterEnvironmentMap(ibl, device, queue); err != nil {
		ibl.Release()
		return nil, err
	}

	return ibl, nil
}

// Rebake re-runs the procedural sky capture + mip generation +
// environment filter. The cubemap textures are reused (same size,
// same format); the irradiance + prefiltered textures are
// reallocated because [filterEnvironmentMap] currently creates
// fresh outputs. Callers that cached IBL views must rebuild bind
// groups after Rebake — meshPass invalidates via [InvalidateIBL].
//
// The BRDF LUT isn't recomputed — it's a static integral that
// doesn't depend on the environment.
//
// Intended call site is a render command drained at frame setup
// so the GPU work happens before any pass writes the swapchain.
func (ibl *IBL) Rebake(device *wgpu.Device, queue *wgpu.Queue) error {
	if ibl.PrefilteredView != nil {
		ibl.PrefilteredView.Release()
		ibl.PrefilteredView = nil
	}
	if ibl.Prefiltered != nil {
		ibl.Prefiltered.Release()
		ibl.Prefiltered = nil
	}
	if ibl.IrradianceView != nil {
		ibl.IrradianceView.Release()
		ibl.IrradianceView = nil
	}
	if ibl.Irradiance != nil {
		ibl.Irradiance.Release()
		ibl.Irradiance = nil
	}

	if err := captureProceduralCubemap(ibl, device, queue); err != nil {
		return err
	}
	if err := generateCubemapMipmaps(ibl.Cubemap, device, queue); err != nil {
		return err
	}
	if err := filterEnvironmentMap(ibl, device, queue); err != nil {
		return err
	}
	return nil
}

// Release frees every GPU resource the IBL bundle owns.
func (ibl *IBL) Release() {
	if ibl.Sampler != nil {
		ibl.Sampler.Release()
	}
	if ibl.PrefilteredView != nil {
		ibl.PrefilteredView.Release()
	}
	if ibl.Prefiltered != nil {
		ibl.Prefiltered.Release()
	}
	if ibl.IrradianceView != nil {
		ibl.IrradianceView.Release()
	}
	if ibl.Irradiance != nil {
		ibl.Irradiance.Release()
	}
	if ibl.CubemapView != nil {
		ibl.CubemapView.Release()
	}
	if ibl.Cubemap != nil {
		ibl.Cubemap.Release()
	}
	if ibl.BrdfLutView != nil {
		ibl.BrdfLutView.Release()
	}
	if ibl.BrdfLut != nil {
		ibl.BrdfLut.Release()
	}
}

// generateBrdfLut runs the BRDF LUT compute shader once into a
// 256x256 rgba16float texture. RG channels hold the GGX
// split-sum integration the IBL specular term reads; B holds the
// Charlie sheen integration (unused by indigo today but cheap to
// pre-compute alongside).
func generateBrdfLut(ibl *IBL, device *wgpu.Device, queue *wgpu.Queue) error {
	tex, err := device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "brdf lut",
		Size: wgpu.Extent3D{
			Width:              BrdfLutSize,
			Height:             BrdfLutSize,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     wgpu.TextureDimension2D,
		Format:        wgpu.TextureFormatRGBA16Float,
		Usage:         wgpu.TextureUsageTextureBinding | wgpu.TextureUsageStorageBinding,
	})
	if err != nil {
		return fmt.Errorf("ibl: brdf lut texture: %w", err)
	}
	ibl.BrdfLut = tex

	view, err := tex.CreateView(nil)
	if err != nil {
		return fmt.Errorf("ibl: brdf lut view: %w", err)
	}
	ibl.BrdfLutView = view

	layout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "brdf lut layout",
		Entries: []wgpu.BindGroupLayoutEntry{{
			Binding:    0,
			Visibility: wgpu.ShaderStageCompute,
			StorageTexture: wgpu.StorageTextureBindingLayout{
				Access:        wgpu.StorageTextureAccessWriteOnly,
				Format:        wgpu.TextureFormatRGBA16Float,
				ViewDimension: wgpu.TextureViewDimension2D,
			},
		}},
	})
	if err != nil {
		return fmt.Errorf("ibl: brdf lut bgl: %w", err)
	}
	defer layout.Release()

	bindGroup, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "brdf lut bind group",
		Layout: layout,
		Entries: []wgpu.BindGroupEntry{{
			Binding:     0,
			TextureView: view,
		}},
	})
	if err != nil {
		return fmt.Errorf("ibl: brdf lut bind group: %w", err)
	}
	defer bindGroup.Release()

	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "brdf lut shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: brdfLutShader},
	})
	if err != nil {
		return fmt.Errorf("ibl: brdf lut shader: %w", err)
	}
	defer shader.Release()

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "brdf lut pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{layout},
	})
	if err != nil {
		return fmt.Errorf("ibl: brdf lut pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	pipeline, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Layout: pipelineLayout,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     shader,
			EntryPoint: "main",
		},
	})
	if err != nil {
		return fmt.Errorf("ibl: brdf lut pipeline: %w", err)
	}
	defer pipeline.Release()

	encoder, err := device.CreateCommandEncoder(&wgpu.CommandEncoderDescriptor{
		Label: "brdf lut encoder",
	})
	if err != nil {
		return fmt.Errorf("ibl: brdf lut encoder: %w", err)
	}
	pass := encoder.BeginComputePass(&wgpu.ComputePassDescriptor{})
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups(BrdfLutSize/16, BrdfLutSize/16, 1)
	pass.End()
	pass.Release()

	cmd, err := encoder.Finish(nil)
	if err != nil {
		encoder.Release()
		return fmt.Errorf("ibl: brdf lut encoder finish: %w", err)
	}
	encoder.Release()
	defer cmd.Release()
	queue.Submit(cmd)

	return nil
}

// proceduralUniform mirrors the WGSL ProceduralUniform struct.
type proceduralUniform struct {
	OutputSize uint32
	Time       float32
	Pad0       uint32
	Pad1       uint32
}

// captureProceduralCubemap allocates the 1024x1024x6 cubemap with
// a full mip chain and runs the procedural sky shader to fill mip
// 0 of all six faces. The mip chain itself is populated by
// generateCubemapMipmaps in a follow-up pass.
func captureProceduralCubemap(ibl *IBL, device *wgpu.Device, queue *wgpu.Queue) error {
	mips := mipLevelCount(ProceduralCubemapSize)

	tex, err := device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "procedural cubemap",
		Size: wgpu.Extent3D{
			Width:              ProceduralCubemapSize,
			Height:             ProceduralCubemapSize,
			DepthOrArrayLayers: 6,
		},
		MipLevelCount: mips,
		SampleCount:   1,
		Dimension:     wgpu.TextureDimension2D,
		Format:        wgpu.TextureFormatRGBA16Float,
		Usage:         wgpu.TextureUsageTextureBinding | wgpu.TextureUsageStorageBinding,
	})
	if err != nil {
		return fmt.Errorf("ibl: cubemap texture: %w", err)
	}
	ibl.Cubemap = tex

	cubeView, err := tex.CreateView(&wgpu.TextureViewDescriptor{
		Label:           "procedural cubemap cube view",
		Dimension:       wgpu.TextureViewDimensionCube,
		BaseMipLevel:    0,
		MipLevelCount:   mips,
		BaseArrayLayer:  0,
		ArrayLayerCount: 6,
		Aspect:          wgpu.TextureAspectAll,
	})
	if err != nil {
		return fmt.Errorf("ibl: cubemap cube view: %w", err)
	}
	ibl.CubemapView = cubeView

	storageView, err := tex.CreateView(&wgpu.TextureViewDescriptor{
		Label:           "procedural cubemap storage view (mip 0)",
		Dimension:       wgpu.TextureViewDimension2DArray,
		BaseMipLevel:    0,
		MipLevelCount:   1,
		BaseArrayLayer:  0,
		ArrayLayerCount: 6,
		Aspect:          wgpu.TextureAspectAll,
	})
	if err != nil {
		return fmt.Errorf("ibl: cubemap storage view: %w", err)
	}
	defer storageView.Release()

	uniformBuf, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "procedural uniform",
		Size:  uint64(unsafe.Sizeof(proceduralUniform{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return fmt.Errorf("ibl: procedural uniform buffer: %w", err)
	}
	defer uniformBuf.Release()

	uniform := proceduralUniform{OutputSize: ProceduralCubemapSize}
	queue.WriteBuffer(uniformBuf, 0, unsafe.Slice((*byte)(unsafe.Pointer(&uniform)), unsafe.Sizeof(uniform)))

	layout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "procedural cubemap bgl",
		Entries: []wgpu.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: wgpu.ShaderStageCompute,
				Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform},
			},
			{
				Binding:    1,
				Visibility: wgpu.ShaderStageCompute,
				StorageTexture: wgpu.StorageTextureBindingLayout{
					Access:        wgpu.StorageTextureAccessWriteOnly,
					Format:        wgpu.TextureFormatRGBA16Float,
					ViewDimension: wgpu.TextureViewDimension2DArray,
				},
			},
		},
	})
	if err != nil {
		return fmt.Errorf("ibl: procedural bgl: %w", err)
	}
	defer layout.Release()

	bindGroup, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "procedural cubemap bind group",
		Layout: layout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: uniformBuf, Offset: 0, Size: uint64(unsafe.Sizeof(uniform))},
			{Binding: 1, TextureView: storageView},
		},
	})
	if err != nil {
		return fmt.Errorf("ibl: procedural bind group: %w", err)
	}
	defer bindGroup.Release()

	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "procedural cubemap shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: proceduralCubemapShader},
	})
	if err != nil {
		return fmt.Errorf("ibl: procedural shader: %w", err)
	}
	defer shader.Release()

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "procedural cubemap pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{layout},
	})
	if err != nil {
		return fmt.Errorf("ibl: procedural pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	pipeline, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Layout: pipelineLayout,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     shader,
			EntryPoint: "main",
		},
	})
	if err != nil {
		return fmt.Errorf("ibl: procedural pipeline: %w", err)
	}
	defer pipeline.Release()

	encoder, err := device.CreateCommandEncoder(&wgpu.CommandEncoderDescriptor{
		Label: "procedural cubemap encoder",
	})
	if err != nil {
		return fmt.Errorf("ibl: procedural encoder: %w", err)
	}
	pass := encoder.BeginComputePass(&wgpu.ComputePassDescriptor{})
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups((ProceduralCubemapSize+15)/16, (ProceduralCubemapSize+15)/16, 6)
	pass.End()
	pass.Release()

	cmd, err := encoder.Finish(nil)
	if err != nil {
		encoder.Release()
		return fmt.Errorf("ibl: procedural encoder finish: %w", err)
	}
	encoder.Release()
	defer cmd.Release()
	queue.Submit(cmd)

	return nil
}

type mipParams struct {
	DstSize uint32
	Pad0    uint32
	Pad1    uint32
	Pad2    uint32
}

// generateCubemapMipmaps fills mips 1..N of an existing cubemap
// by sampling the previous mip via the bilinear sampler. One
// compute dispatch per mip level.
func generateCubemapMipmaps(cubemap *wgpu.Texture, device *wgpu.Device, queue *wgpu.Queue) error {
	mipCount := mipLevelCount(ProceduralCubemapSize)
	if mipCount <= 1 {
		return nil
	}

	sampler, err := device.CreateSampler(&wgpu.SamplerDescriptor{
		Label:         "mipgen sampler",
		AddressModeU:  wgpu.AddressModeClampToEdge,
		AddressModeV:  wgpu.AddressModeClampToEdge,
		AddressModeW:  wgpu.AddressModeClampToEdge,
		MagFilter:     wgpu.FilterModeLinear,
		MinFilter:     wgpu.FilterModeLinear,
		MipmapFilter:  wgpu.MipmapFilterModeLinear,
		MaxAnisotropy: 1,
	})
	if err != nil {
		return fmt.Errorf("ibl: mipgen sampler: %w", err)
	}
	defer sampler.Release()

	layout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "cubemap mipgen bgl",
		Entries: []wgpu.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: wgpu.ShaderStageCompute,
				Texture: wgpu.TextureBindingLayout{
					SampleType:    wgpu.TextureSampleTypeFloat,
					ViewDimension: wgpu.TextureViewDimension2DArray,
				},
			},
			{
				Binding:    1,
				Visibility: wgpu.ShaderStageCompute,
				Sampler:    wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeFiltering},
			},
			{
				Binding:    2,
				Visibility: wgpu.ShaderStageCompute,
				StorageTexture: wgpu.StorageTextureBindingLayout{
					Access:        wgpu.StorageTextureAccessWriteOnly,
					Format:        wgpu.TextureFormatRGBA16Float,
					ViewDimension: wgpu.TextureViewDimension2DArray,
				},
			},
			{
				Binding:    3,
				Visibility: wgpu.ShaderStageCompute,
				Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform},
			},
		},
	})
	if err != nil {
		return fmt.Errorf("ibl: mipgen bgl: %w", err)
	}
	defer layout.Release()

	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "cubemap mipgen shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: cubemapMipgenShader},
	})
	if err != nil {
		return fmt.Errorf("ibl: mipgen shader: %w", err)
	}
	defer shader.Release()

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "cubemap mipgen pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{layout},
	})
	if err != nil {
		return fmt.Errorf("ibl: mipgen pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	pipeline, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Layout: pipelineLayout,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     shader,
			EntryPoint: "main",
		},
	})
	if err != nil {
		return fmt.Errorf("ibl: mipgen pipeline: %w", err)
	}
	defer pipeline.Release()

	uniformBuf, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "mipgen uniform",
		Size:  uint64(unsafe.Sizeof(mipParams{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return fmt.Errorf("ibl: mipgen uniform: %w", err)
	}
	defer uniformBuf.Release()

	for mip := uint32(1); mip < mipCount; mip++ {
		srcLevel := mip - 1
		dstSize := ProceduralCubemapSize >> mip

		srcView, err := cubemap.CreateView(&wgpu.TextureViewDescriptor{
			Label:           "mipgen src view",
			Dimension:       wgpu.TextureViewDimension2DArray,
			BaseMipLevel:    srcLevel,
			MipLevelCount:   1,
			BaseArrayLayer:  0,
			ArrayLayerCount: 6,
			Aspect:          wgpu.TextureAspectAll,
		})
		if err != nil {
			return fmt.Errorf("ibl: mipgen src view: %w", err)
		}

		dstView, err := cubemap.CreateView(&wgpu.TextureViewDescriptor{
			Label:           "mipgen dst view",
			Dimension:       wgpu.TextureViewDimension2DArray,
			BaseMipLevel:    mip,
			MipLevelCount:   1,
			BaseArrayLayer:  0,
			ArrayLayerCount: 6,
			Aspect:          wgpu.TextureAspectAll,
		})
		if err != nil {
			srcView.Release()
			return fmt.Errorf("ibl: mipgen dst view: %w", err)
		}

		params := mipParams{DstSize: dstSize}
		queue.WriteBuffer(uniformBuf, 0, unsafe.Slice((*byte)(unsafe.Pointer(&params)), unsafe.Sizeof(params)))

		bindGroup, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  "mipgen bind group",
			Layout: layout,
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, TextureView: srcView},
				{Binding: 1, Sampler: sampler},
				{Binding: 2, TextureView: dstView},
				{Binding: 3, Buffer: uniformBuf, Offset: 0, Size: uint64(unsafe.Sizeof(params))},
			},
		})
		if err != nil {
			srcView.Release()
			dstView.Release()
			return fmt.Errorf("ibl: mipgen bind group: %w", err)
		}

		encoder, err := device.CreateCommandEncoder(&wgpu.CommandEncoderDescriptor{
			Label: "mipgen encoder",
		})
		if err != nil {
			srcView.Release()
			dstView.Release()
			bindGroup.Release()
			return fmt.Errorf("ibl: mipgen encoder: %w", err)
		}
		pass := encoder.BeginComputePass(&wgpu.ComputePassDescriptor{})
		pass.SetPipeline(pipeline)
		pass.SetBindGroup(0, bindGroup, nil)
		groups := (dstSize + 15) / 16
		if groups == 0 {
			groups = 1
		}
		pass.DispatchWorkgroups(groups, groups, 6)
		pass.End()
		pass.Release()

		cmd, err := encoder.Finish(nil)
		if err != nil {
			encoder.Release()
			srcView.Release()
			dstView.Release()
			bindGroup.Release()
			return fmt.Errorf("ibl: mipgen encoder finish: %w", err)
		}
		encoder.Release()
		queue.Submit(cmd)
		cmd.Release()
		bindGroup.Release()
		dstView.Release()
		srcView.Release()
	}

	return nil
}

type filterParams struct {
	Face           uint32
	OutputSize     uint32
	Roughness      float32
	SampleCount    uint32
	SourceWidth    uint32
	SourceHeight   uint32
	Distribution   uint32
	SourceMipCount uint32
}

// filterEnvironmentMap allocates the irradiance + prefiltered
// cubemaps and runs the filter compute shader twice: once with
// the Lambertian distribution (writes irradiance mip 0) and once
// per mip level with the GGX distribution at the roughness
// matching that mip.
//
// Uses its own sampler — NOT the shared [IBL.Sampler] — because
// the GGX importance-sampling math inside filter_envmap.wgsl
// requests mip levels up to log2(SourceWidth) (typically 10 for
// a 1024 cubemap) when roughness approaches 1.0. The shared
// sampler clamps LOD at PrefilteredMipLevels (5) so the mesh
// pass can't accidentally read past the prefiltered chain — that
// clamp silently blocks the filter pass from sampling the high
// mips of the SOURCE cubemap, collapsing the per-mip filter
// quality and making rough surfaces look identical to shinier
// ones.
func filterEnvironmentMap(ibl *IBL, device *wgpu.Device, queue *wgpu.Queue) error {
	filterSampler, err := device.CreateSampler(&wgpu.SamplerDescriptor{
		Label:         "ibl filter pass sampler",
		AddressModeU:  wgpu.AddressModeClampToEdge,
		AddressModeV:  wgpu.AddressModeClampToEdge,
		AddressModeW:  wgpu.AddressModeClampToEdge,
		MagFilter:     wgpu.FilterModeLinear,
		MinFilter:     wgpu.FilterModeLinear,
		MipmapFilter:  wgpu.MipmapFilterModeLinear,
		LodMinClamp:   0,
		LodMaxClamp:   32,
		MaxAnisotropy: 1,
	})
	if err != nil {
		return fmt.Errorf("ibl: filter sampler: %w", err)
	}
	defer filterSampler.Release()
	irradiance, err := device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "ibl irradiance",
		Size: wgpu.Extent3D{
			Width:              IrradianceSize,
			Height:             IrradianceSize,
			DepthOrArrayLayers: 6,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     wgpu.TextureDimension2D,
		Format:        wgpu.TextureFormatRGBA16Float,
		Usage:         wgpu.TextureUsageTextureBinding | wgpu.TextureUsageStorageBinding,
	})
	if err != nil {
		return fmt.Errorf("ibl: irradiance texture: %w", err)
	}
	ibl.Irradiance = irradiance

	irradianceCube, err := irradiance.CreateView(&wgpu.TextureViewDescriptor{
		Label:           "ibl irradiance cube view",
		Dimension:       wgpu.TextureViewDimensionCube,
		BaseMipLevel:    0,
		MipLevelCount:   1,
		BaseArrayLayer:  0,
		ArrayLayerCount: 6,
		Aspect:          wgpu.TextureAspectAll,
	})
	if err != nil {
		return fmt.Errorf("ibl: irradiance cube view: %w", err)
	}
	ibl.IrradianceView = irradianceCube

	prefiltered, err := device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "ibl prefiltered",
		Size: wgpu.Extent3D{
			Width:              PrefilteredSize,
			Height:             PrefilteredSize,
			DepthOrArrayLayers: 6,
		},
		MipLevelCount: PrefilteredMipLevels,
		SampleCount:   1,
		Dimension:     wgpu.TextureDimension2D,
		Format:        wgpu.TextureFormatRGBA16Float,
		Usage:         wgpu.TextureUsageTextureBinding | wgpu.TextureUsageStorageBinding,
	})
	if err != nil {
		return fmt.Errorf("ibl: prefiltered texture: %w", err)
	}
	ibl.Prefiltered = prefiltered

	prefilteredCube, err := prefiltered.CreateView(&wgpu.TextureViewDescriptor{
		Label:           "ibl prefiltered cube view",
		Dimension:       wgpu.TextureViewDimensionCube,
		BaseMipLevel:    0,
		MipLevelCount:   PrefilteredMipLevels,
		BaseArrayLayer:  0,
		ArrayLayerCount: 6,
		Aspect:          wgpu.TextureAspectAll,
	})
	if err != nil {
		return fmt.Errorf("ibl: prefiltered cube view: %w", err)
	}
	ibl.PrefilteredView = prefilteredCube

	layout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "filter envmap bgl",
		Entries: []wgpu.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: wgpu.ShaderStageCompute,
				Texture: wgpu.TextureBindingLayout{
					SampleType:    wgpu.TextureSampleTypeFloat,
					ViewDimension: wgpu.TextureViewDimensionCube,
				},
			},
			{
				Binding:    1,
				Visibility: wgpu.ShaderStageCompute,
				Sampler:    wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeFiltering},
			},
			{
				Binding:    2,
				Visibility: wgpu.ShaderStageCompute,
				StorageTexture: wgpu.StorageTextureBindingLayout{
					Access:        wgpu.StorageTextureAccessWriteOnly,
					Format:        wgpu.TextureFormatRGBA16Float,
					ViewDimension: wgpu.TextureViewDimension2DArray,
				},
			},
			{
				Binding:    3,
				Visibility: wgpu.ShaderStageCompute,
				Buffer:     wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform},
			},
		},
	})
	if err != nil {
		return fmt.Errorf("ibl: filter bgl: %w", err)
	}
	defer layout.Release()

	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "filter envmap shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: filterEnvmapShader},
	})
	if err != nil {
		return fmt.Errorf("ibl: filter shader: %w", err)
	}
	defer shader.Release()

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "filter envmap pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{layout},
	})
	if err != nil {
		return fmt.Errorf("ibl: filter pipeline layout: %w", err)
	}
	defer pipelineLayout.Release()

	pipeline, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Layout: pipelineLayout,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     shader,
			EntryPoint: "main",
		},
	})
	if err != nil {
		return fmt.Errorf("ibl: filter pipeline: %w", err)
	}
	defer pipeline.Release()

	uniformBuf, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "filter uniform",
		Size:  uint64(unsafe.Sizeof(filterParams{})),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return fmt.Errorf("ibl: filter uniform: %w", err)
	}
	defer uniformBuf.Release()

	srcMipCount := mipLevelCount(ProceduralCubemapSize)

	dispatchFilter := func(label string, view *wgpu.TextureView, params filterParams, outputSize uint32) error {
		queue.WriteBuffer(uniformBuf, 0, unsafe.Slice((*byte)(unsafe.Pointer(&params)), unsafe.Sizeof(params)))

		bindGroup, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  label + " bind group",
			Layout: layout,
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, TextureView: ibl.CubemapView},
				{Binding: 1, Sampler: filterSampler},
				{Binding: 2, TextureView: view},
				{Binding: 3, Buffer: uniformBuf, Offset: 0, Size: uint64(unsafe.Sizeof(params))},
			},
		})
		if err != nil {
			return fmt.Errorf("ibl: %s bind group: %w", label, err)
		}
		defer bindGroup.Release()

		encoder, err := device.CreateCommandEncoder(&wgpu.CommandEncoderDescriptor{
			Label: label + " encoder",
		})
		if err != nil {
			return fmt.Errorf("ibl: %s encoder: %w", label, err)
		}
		_ = label
		pass := encoder.BeginComputePass(&wgpu.ComputePassDescriptor{})
		pass.SetPipeline(pipeline)
		pass.SetBindGroup(0, bindGroup, nil)
		groups := (outputSize + 15) / 16
		if groups == 0 {
			groups = 1
		}
		pass.DispatchWorkgroups(groups, groups, 6)
		pass.End()
		pass.Release()

		cmd, err := encoder.Finish(nil)
		if err != nil {
			encoder.Release()
			return fmt.Errorf("ibl: %s encoder finish: %w", label, err)
		}
		encoder.Release()
		queue.Submit(cmd)
		cmd.Release()
		return nil
	}

	irradianceStorageView, err := irradiance.CreateView(&wgpu.TextureViewDescriptor{
		Label:           "ibl irradiance storage view",
		Dimension:       wgpu.TextureViewDimension2DArray,
		BaseMipLevel:    0,
		MipLevelCount:   1,
		BaseArrayLayer:  0,
		ArrayLayerCount: 6,
		Aspect:          wgpu.TextureAspectAll,
	})
	if err != nil {
		return fmt.Errorf("ibl: irradiance storage view: %w", err)
	}
	defer irradianceStorageView.Release()

	if err := dispatchFilter("filter irradiance", irradianceStorageView, filterParams{
		Face:           0,
		OutputSize:     IrradianceSize,
		Roughness:      1.0,
		SampleCount:    IrradianceSamples,
		SourceWidth:    ProceduralCubemapSize,
		SourceHeight:   ProceduralCubemapSize,
		Distribution:   0,
		SourceMipCount: srcMipCount,
	}, IrradianceSize); err != nil {
		return err
	}

	for mip := uint32(0); mip < PrefilteredMipLevels; mip++ {
		mipSize := PrefilteredSize >> mip
		roughness := float32(mip) / float32(PrefilteredMipLevels-1)

		mipView, err := prefiltered.CreateView(&wgpu.TextureViewDescriptor{
			Label:           "ibl prefiltered storage view",
			Dimension:       wgpu.TextureViewDimension2DArray,
			BaseMipLevel:    mip,
			MipLevelCount:   1,
			BaseArrayLayer:  0,
			ArrayLayerCount: 6,
			Aspect:          wgpu.TextureAspectAll,
		})
		if err != nil {
			return fmt.Errorf("ibl: prefiltered mip view: %w", err)
		}

		err = dispatchFilter(fmt.Sprintf("filter prefiltered mip %d", mip), mipView, filterParams{
			Face:           0,
			OutputSize:     mipSize,
			Roughness:      roughness,
			SampleCount:    PrefilteredSamples,
			SourceWidth:    ProceduralCubemapSize,
			SourceHeight:   ProceduralCubemapSize,
			Distribution:   1,
			SourceMipCount: srcMipCount,
		}, mipSize)
		mipView.Release()
		if err != nil {
			return err
		}
	}

	return nil
}

// mipLevelCount returns the number of mip levels needed to walk
// `size` down to 1x1. Matches asset.mipLevelCount but lives in
// the pass package so the IBL setup doesn't pull the asset
// package in.
func mipLevelCount(size uint32) uint32 {
	if size <= 1 {
		return 1
	}
	count := uint32(0)
	current := size
	for current > 0 {
		count++
		current >>= 1
	}
	return count
}
