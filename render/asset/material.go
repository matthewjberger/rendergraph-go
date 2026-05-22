package asset

// AlphaMode maps the glTF 2.0 material.alphaMode enum.
type AlphaMode uint8

const (
	AlphaModeOpaque AlphaMode = iota
	AlphaModeMask
	AlphaModeBlend
)

// Material is the per-entity surface-appearance component. Each
// texture slot is a packed layer value (lower 16 bits = layer
// index in [MaterialTextureArrays], upper bits = per-axis wrap
// mode codes). [NoTextureLayer] means "no texture in this slot —
// the shader should use the factor only."
//
// Mirrors the runtime shape nightshade's Material keeps: texture
// slots reference layers into shared sRGB / linear texture arrays,
// per-material factors / scales / alpha flags ride alongside.
type Material struct {
	BaseColor              [4]float32
	BaseColorLayer         uint32
	MetallicRoughnessLayer uint32
	NormalLayer            uint32
	OcclusionLayer         uint32
	EmissiveLayer          uint32

	MetallicFactor    float32
	RoughnessFactor   float32
	NormalScale       float32
	OcclusionStrength float32
	EmissiveFactor    [3]float32

	AlphaMode   AlphaMode
	AlphaCutoff float32
	DoubleSided bool
	Unlit       bool
}

// DefaultMaterial returns a fully-opaque white material with the
// glTF-spec defaults (metallic=1.0, roughness=1.0, alpha=opaque).
// All texture slots default to [NoTextureLayer] so the shader's
// sampling helpers fall back to the per-material factors.
func DefaultMaterial() Material {
	return Material{
		BaseColor:              [4]float32{1, 1, 1, 1},
		BaseColorLayer:         NoTextureLayer,
		MetallicRoughnessLayer: NoTextureLayer,
		NormalLayer:            NoTextureLayer,
		OcclusionLayer:         NoTextureLayer,
		EmissiveLayer:          NoTextureLayer,
		MetallicFactor:         1.0,
		RoughnessFactor:        1.0,
		NormalScale:            1.0,
		OcclusionStrength:      1.0,
		AlphaCutoff:            0.5,
	}
}

// MaterialGPU is the std430-aligned per-material struct the mesh
// pass uploads to its global storage buffer. Six 16-byte rows;
// the WGSL Material struct mirrors this layout exactly so the
// driver can read it without padding surprises.
type MaterialGPU struct {
	BaseColor      [4]float32 // row 0
	EmissiveFactor [3]float32 // row 1
	AlphaMode      uint32

	BaseLayer              uint32 // row 2
	EmissiveLayer          uint32
	NormalLayer            uint32
	MetallicRoughnessLayer uint32

	OcclusionLayer    uint32 // row 3
	NormalScale       float32
	OcclusionStrength float32
	MetallicFactor    float32

	RoughnessFactor float32 // row 4
	AlphaCutoff     float32
	Unlit           uint32
	Pad0            uint32

	Pad1 [4]float32 // row 5 (reserved for future PBR extensions)
}

// MaterialGPUSize is the byte stride of a single [MaterialGPU]
// entry in the global storage buffer.
const MaterialGPUSize = uint64(96)

// ToGPU converts a CPU-side [Material] into the GPU layout the
// shader expects. NoTextureLayer-marked slots stay as the sentinel
// so the WGSL helpers can branch on it.
func (m Material) ToGPU() MaterialGPU {
	var alpha uint32
	switch m.AlphaMode {
	case AlphaModeMask:
		alpha = 1
	case AlphaModeBlend:
		alpha = 2
	}
	var unlit uint32
	if m.Unlit {
		unlit = 1
	}
	return MaterialGPU{
		BaseColor:              m.BaseColor,
		EmissiveFactor:         m.EmissiveFactor,
		AlphaMode:              alpha,
		BaseLayer:              m.BaseColorLayer,
		EmissiveLayer:          m.EmissiveLayer,
		NormalLayer:            m.NormalLayer,
		MetallicRoughnessLayer: m.MetallicRoughnessLayer,
		OcclusionLayer:         m.OcclusionLayer,
		NormalScale:            m.NormalScale,
		OcclusionStrength:      m.OcclusionStrength,
		MetallicFactor:         m.MetallicFactor,
		RoughnessFactor:        m.RoughnessFactor,
		AlphaCutoff:            m.AlphaCutoff,
		Unlit:                  unlit,
	}
}
