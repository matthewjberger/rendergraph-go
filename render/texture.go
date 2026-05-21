package render

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"
)

type TextureID uint32

// Texture is a GPU-uploaded 2D image plus the view + sampler the mesh
// pass binds. Stored in a [TextureCache] and referenced from
// [MeshAssets] entries by handle.
type Texture struct {
	Texture *wgpu.Texture
	View    *wgpu.TextureView
	Sampler *wgpu.Sampler
	Width   uint32
	Height  uint32
	Mips    uint32
	Name    string
}

func (t *Texture) Release() {
	if t.View != nil {
		t.View.Release()
		t.View = nil
	}
	if t.Sampler != nil {
		t.Sampler.Release()
		t.Sampler = nil
	}
	if t.Texture != nil {
		t.Texture.Release()
		t.Texture = nil
	}
}

type TextureCache struct {
	entries []*Texture
	white   TextureID
}

type TextureCacheResource struct{ Cache *TextureCache }

func NewTextureCache() *TextureCache { return &TextureCache{} }

func (c *TextureCache) Lookup(handle TextureID) (*Texture, bool) {
	if handle == 0 || int(handle) > len(c.entries) {
		return nil, false
	}
	return c.entries[handle-1], true
}

func (c *TextureCache) Count() int { return len(c.entries) }

func (c *TextureCache) White() TextureID { return c.white }

func (c *TextureCache) Release() {
	for _, t := range c.entries {
		if t != nil {
			t.Release()
		}
	}
	c.entries = nil
	c.white = 0
}

// TextureColorSpace picks sRGB vs linear encoding for a texture
// upload. Base color and emissive maps go in as sRGB; normal,
// metallic-roughness, and occlusion maps stay linear so their data
// isn't gamma-curved before the shader reads it.
type TextureColorSpace uint8

const (
	TextureSRGB TextureColorSpace = iota
	TextureLinear
)

// SamplerSettings carries the wrap modes + min/mag filters extracted
// from a glTF texture sampler. Zero value is "repeat both axes,
// linear filtering with linear mip blending" which matches the glTF
// default sampler behavior.
type SamplerSettings struct {
	WrapU        wgpu.AddressMode
	WrapV        wgpu.AddressMode
	MagFilter    wgpu.FilterMode
	MinFilter    wgpu.FilterMode
	MipmapFilter wgpu.MipmapFilterMode
}

// DefaultSamplerSettings returns the glTF-default sampler config.
func DefaultSamplerSettings() SamplerSettings {
	return SamplerSettings{
		WrapU:        wgpu.AddressModeRepeat,
		WrapV:        wgpu.AddressModeRepeat,
		MagFilter:    wgpu.FilterModeLinear,
		MinFilter:    wgpu.FilterModeLinear,
		MipmapFilter: wgpu.MipmapFilterModeLinear,
	}
}

// Register uploads pixels to a new 2D texture (with a full mip
// chain) and returns its handle. Pixels are tightly packed RGBA8 in
// row-major order, width*height*4 bytes long. The sampler is built
// from settings.
func (c *TextureCache) Register(device *wgpu.Device, queue *wgpu.Queue, name string, width, height uint32, pixels []byte, space TextureColorSpace, settings SamplerSettings) (TextureID, error) {
	if len(pixels) != int(width*height*4) {
		return 0, fmt.Errorf("texture %q: expected %d bytes, got %d", name, width*height*4, len(pixels))
	}
	format := wgpu.TextureFormatRGBA8UnormSrgb
	if space == TextureLinear {
		format = wgpu.TextureFormatRGBA8Unorm
	}
	mips := mipLevelCount(width, height)
	tex, err := device.CreateTexture(&wgpu.TextureDescriptor{
		Label: name,
		Size: wgpu.Extent3D{
			Width:              width,
			Height:             height,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: mips,
		SampleCount:   1,
		Dimension:     wgpu.TextureDimension2D,
		Format:        format,
		Usage:         wgpu.TextureUsageTextureBinding | wgpu.TextureUsageCopyDst,
	})
	if err != nil {
		return 0, fmt.Errorf("texture %q: create: %w", name, err)
	}
	level := pixels
	w := width
	h := height
	for mip := uint32(0); mip < mips; mip++ {
		queue.WriteTexture(
			&wgpu.ImageCopyTexture{
				Texture:  tex,
				MipLevel: mip,
				Aspect:   wgpu.TextureAspectAll,
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
		if mip+1 >= mips {
			break
		}
		level, w, h = downsampleRGBA(level, w, h)
	}
	view, err := tex.CreateView(nil)
	if err != nil {
		tex.Release()
		return 0, fmt.Errorf("texture %q: view: %w", name, err)
	}
	sampler, err := device.CreateSampler(&wgpu.SamplerDescriptor{
		Label:         name + " sampler",
		AddressModeU:  settings.WrapU,
		AddressModeV:  settings.WrapV,
		MagFilter:     settings.MagFilter,
		MinFilter:     settings.MinFilter,
		MipmapFilter:  settings.MipmapFilter,
		LodMinClamp:   0,
		LodMaxClamp:   float32(mips),
		MaxAnisotropy: 1,
	})
	if err != nil {
		view.Release()
		tex.Release()
		return 0, fmt.Errorf("texture %q: sampler: %w", name, err)
	}
	entry := &Texture{
		Texture: tex,
		View:    view,
		Sampler: sampler,
		Width:   width,
		Height:  height,
		Mips:    mips,
		Name:    name,
	}
	c.entries = append(c.entries, entry)
	return TextureID(len(c.entries)), nil
}

// EnsureWhite registers a 1x1 opaque-white texture and caches its
// handle. Meshes without a base-color texture are bound to this so
// the shader's `textureSample(...)` call still has something to
// sample (returning vec4(1.0)).
func (c *TextureCache) EnsureWhite(device *wgpu.Device, queue *wgpu.Queue) (TextureID, error) {
	if c.white != 0 {
		return c.white, nil
	}
	h, err := c.Register(device, queue, "white_1x1", 1, 1, []byte{255, 255, 255, 255}, TextureSRGB, DefaultSamplerSettings())
	if err != nil {
		return 0, err
	}
	c.white = h
	return h, nil
}

// mipLevelCount returns ⌊log2(max(w,h))⌋ + 1, the standard full mip
// chain count for a 2D texture.
func mipLevelCount(width, height uint32) uint32 {
	max := width
	if height > max {
		max = height
	}
	levels := uint32(1)
	for max > 1 {
		max >>= 1
		levels++
	}
	return levels
}

// downsampleRGBA halves width and height with a 2x2 box filter,
// clamping odd dimensions to 1.
func downsampleRGBA(src []byte, srcW, srcH uint32) ([]byte, uint32, uint32) {
	dstW := srcW / 2
	if dstW < 1 {
		dstW = 1
	}
	dstH := srcH / 2
	if dstH < 1 {
		dstH = 1
	}
	dst := make([]byte, dstW*dstH*4)
	for y := uint32(0); y < dstH; y++ {
		for x := uint32(0); x < dstW; x++ {
			sx := x * 2
			sy := y * 2
			sxNext := sx + 1
			if sxNext >= srcW {
				sxNext = srcW - 1
			}
			syNext := sy + 1
			if syNext >= srcH {
				syNext = srcH - 1
			}
			for c := uint32(0); c < 4; c++ {
				sum := uint32(src[(sy*srcW+sx)*4+c]) +
					uint32(src[(sy*srcW+sxNext)*4+c]) +
					uint32(src[(syNext*srcW+sx)*4+c]) +
					uint32(src[(syNext*srcW+sxNext)*4+c])
				dst[(y*dstW+x)*4+c] = byte((sum + 2) >> 2)
			}
		}
	}
	return dst, dstW, dstH
}
