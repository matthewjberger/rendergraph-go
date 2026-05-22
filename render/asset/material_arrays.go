package asset

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"
)

// MaterialTextureArrays is the GPU storage for every PBR texture
// every loaded material references. Two 2D array textures (sRGB
// for color data — base color + emissive; linear for everything
// else: normal, metallic-roughness, occlusion) hold up to MaxLayers
// slices at LayerSize x LayerSize each, with a full mip chain per
// layer. A single anisotropic sampler is bound alongside them; the
// per-texture wrap modes are encoded in the packed layer integer
// the material references (bits 16-19) and applied in the shader
// at sample time.
//
// Mirrors nightshade's material_texture_arrays.rs (1024x1024x256,
// 16x anisotropic, sRGB/linear pair, name-keyed deduplication).
type MaterialTextureArrays struct {
	LayerSize uint32
	MaxLayers uint32
	MipLevels uint32

	SRGB       *wgpu.Texture
	Linear     *wgpu.Texture
	SRGBView   *wgpu.TextureView
	LinearView *wgpu.TextureView
	Sampler    *wgpu.Sampler

	srgbNext    uint32
	linearNext  uint32
	layerByName map[string]uint32
}

// MaterialTextureArraysResource wraps the cache as a typed engine
// resource so passes can look it up via [ecs.MustResource].
type MaterialTextureArraysResource struct {
	Arrays *MaterialTextureArrays
}

// NoTextureLayer is the sentinel packed layer value meaning "this
// material slot has no texture; the shader should fall back to the
// per-material factor only." Stored as 0xFFFFFFFF so a valid
// (layer 0, repeat, repeat) never collides with it.
const NoTextureLayer uint32 = 0xFFFFFFFF

// WrapMode codes packed into the upper bits of a layer value.
// Lower 16 bits hold the layer index; bits 16-17 hold the U wrap
// mode, bits 18-19 hold V. Encoding matches nightshade's
// SamplerWrap -> u32 mapping so a shader port doesn't have to
// remap.
type WrapMode uint8

const (
	WrapRepeat         WrapMode = 0
	WrapMirroredRepeat WrapMode = 1
	WrapClampToEdge    WrapMode = 2
)

// PackLayer returns the u32 value the GPU material buffer stores
// for a (layer, wrapU, wrapV) tuple. Lower 16 bits: layer; bits
// 16-17: wrapU; bits 18-19: wrapV. The shader extracts via
// `layer & 0xFFFF` and `(packed >> 16) & 0x3` / `(packed >> 18) & 0x3`.
func PackLayer(layer uint32, wrapU, wrapV WrapMode) uint32 {
	return (layer & 0xFFFF) | (uint32(wrapU) << 16) | (uint32(wrapV) << 18)
}

// NewMaterialTextureArrays allocates the two texture arrays with
// nightshade's default 1024x1024x256 layout. Apps that need
// different limits can construct via [NewMaterialTextureArraysWith].
func NewMaterialTextureArrays(device *wgpu.Device) (*MaterialTextureArrays, error) {
	return NewMaterialTextureArraysWith(device, 1024, 256)
}

// NewMaterialTextureArraysWith is the explicit-size constructor.
func NewMaterialTextureArraysWith(device *wgpu.Device, layerSize, maxLayers uint32) (*MaterialTextureArrays, error) {
	mips := mipLevelCount(layerSize, layerSize)
	arrays := &MaterialTextureArrays{
		LayerSize:   layerSize,
		MaxLayers:   maxLayers,
		MipLevels:   mips,
		layerByName: make(map[string]uint32, 64),
	}

	srgb, err := device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "material srgb texture array",
		Size: wgpu.Extent3D{
			Width:              layerSize,
			Height:             layerSize,
			DepthOrArrayLayers: maxLayers,
		},
		MipLevelCount: mips,
		SampleCount:   1,
		Dimension:     wgpu.TextureDimension2D,
		Format:        wgpu.TextureFormatRGBA8UnormSrgb,
		Usage:         wgpu.TextureUsageTextureBinding | wgpu.TextureUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("material arrays: srgb texture: %w", err)
	}
	arrays.SRGB = srgb

	linear, err := device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "material linear texture array",
		Size: wgpu.Extent3D{
			Width:              layerSize,
			Height:             layerSize,
			DepthOrArrayLayers: maxLayers,
		},
		MipLevelCount: mips,
		SampleCount:   1,
		Dimension:     wgpu.TextureDimension2D,
		Format:        wgpu.TextureFormatRGBA8Unorm,
		Usage:         wgpu.TextureUsageTextureBinding | wgpu.TextureUsageCopyDst,
	})
	if err != nil {
		srgb.Release()
		return nil, fmt.Errorf("material arrays: linear texture: %w", err)
	}
	arrays.Linear = linear

	arrayDim := wgpu.TextureViewDimension2DArray
	srgbView, err := srgb.CreateView(&wgpu.TextureViewDescriptor{
		Label:           "material srgb array view",
		Format:          wgpu.TextureFormatRGBA8UnormSrgb,
		Dimension:       arrayDim,
		BaseMipLevel:    0,
		MipLevelCount:   mips,
		BaseArrayLayer:  0,
		ArrayLayerCount: maxLayers,
		Aspect:          wgpu.TextureAspectAll,
	})
	if err != nil {
		srgb.Release()
		linear.Release()
		return nil, fmt.Errorf("material arrays: srgb view: %w", err)
	}
	arrays.SRGBView = srgbView

	linearView, err := linear.CreateView(&wgpu.TextureViewDescriptor{
		Label:           "material linear array view",
		Format:          wgpu.TextureFormatRGBA8Unorm,
		Dimension:       arrayDim,
		BaseMipLevel:    0,
		MipLevelCount:   mips,
		BaseArrayLayer:  0,
		ArrayLayerCount: maxLayers,
		Aspect:          wgpu.TextureAspectAll,
	})
	if err != nil {
		srgbView.Release()
		srgb.Release()
		linear.Release()
		return nil, fmt.Errorf("material arrays: linear view: %w", err)
	}
	arrays.LinearView = linearView

	sampler, err := device.CreateSampler(&wgpu.SamplerDescriptor{
		Label:         "material array sampler",
		AddressModeU:  wgpu.AddressModeClampToEdge,
		AddressModeV:  wgpu.AddressModeClampToEdge,
		AddressModeW:  wgpu.AddressModeClampToEdge,
		MagFilter:     wgpu.FilterModeLinear,
		MinFilter:     wgpu.FilterModeLinear,
		MipmapFilter:  wgpu.MipmapFilterModeLinear,
		LodMinClamp:   0,
		LodMaxClamp:   float32(mips),
		MaxAnisotropy: 16,
	})
	if err != nil {
		linearView.Release()
		srgbView.Release()
		linear.Release()
		srgb.Release()
		return nil, fmt.Errorf("material arrays: sampler: %w", err)
	}
	arrays.Sampler = sampler

	return arrays, nil
}

// Release frees every GPU resource owned by the arrays.
func (m *MaterialTextureArrays) Release() {
	if m.Sampler != nil {
		m.Sampler.Release()
		m.Sampler = nil
	}
	if m.SRGBView != nil {
		m.SRGBView.Release()
		m.SRGBView = nil
	}
	if m.LinearView != nil {
		m.LinearView.Release()
		m.LinearView = nil
	}
	if m.SRGB != nil {
		m.SRGB.Release()
		m.SRGB = nil
	}
	if m.Linear != nil {
		m.Linear.Release()
		m.Linear = nil
	}
}

// Upload writes pixels into the next free layer of the chosen
// array and returns that layer index. Pixels must be RGBA8 sized
// exactly LayerSize x LayerSize x 4 bytes. The loader's image
// decoder is responsible for resizing to the array's tile size.
//
// name acts as a deduplication key (combined with the color
// space): uploading the same (name, space) twice returns the
// existing layer without re-uploading.
func (m *MaterialTextureArrays) Upload(queue *wgpu.Queue, name string, space TextureColorSpace, pixels []byte) (uint32, error) {
	key := name + colorSpaceTag(space)
	if layer, ok := m.layerByName[key]; ok {
		return layer, nil
	}
	expected := int(m.LayerSize * m.LayerSize * 4)
	if len(pixels) != expected {
		return 0, fmt.Errorf("material arrays: %q: expected %d bytes (%dx%d RGBA8), got %d", name, expected, m.LayerSize, m.LayerSize, len(pixels))
	}
	var (
		target *wgpu.Texture
		layer  uint32
	)
	if space == TextureSRGB {
		if m.srgbNext >= m.MaxLayers {
			return 0, fmt.Errorf("material arrays: sRGB array full (%d layers)", m.MaxLayers)
		}
		target = m.SRGB
		layer = m.srgbNext
		m.srgbNext++
	} else {
		if m.linearNext >= m.MaxLayers {
			return 0, fmt.Errorf("material arrays: linear array full (%d layers)", m.MaxLayers)
		}
		target = m.Linear
		layer = m.linearNext
		m.linearNext++
	}

	level := pixels
	w := m.LayerSize
	h := m.LayerSize
	for mip := uint32(0); mip < m.MipLevels; mip++ {
		queue.WriteTexture(
			&wgpu.ImageCopyTexture{
				Texture:  target,
				MipLevel: mip,
				Origin: wgpu.Origin3D{
					Z: layer,
				},
				Aspect: wgpu.TextureAspectAll,
			},
			level,
			&wgpu.TextureDataLayout{
				Offset:       0,
				BytesPerRow:  w * 4,
				RowsPerImage: h,
			},
			&wgpu.Extent3D{
				Width:              w,
				Height:             h,
				DepthOrArrayLayers: 1,
			},
		)
		if mip+1 >= m.MipLevels {
			break
		}
		level, w, h = downsampleRGBA(level, w, h)
	}

	m.layerByName[key] = layer
	return layer, nil
}

func colorSpaceTag(space TextureColorSpace) string {
	if space == TextureSRGB {
		return "|srgb"
	}
	return "|linear"
}
