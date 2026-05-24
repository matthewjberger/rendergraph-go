package asset

import (
	"math"
	"unsafe"
)

type AlphaMode uint8

const (
	AlphaModeOpaque AlphaMode = iota
	AlphaModeMask
	AlphaModeBlend
)

const (
	NormalMapFlipY        uint32 = 1
	NormalMapTwoComponent uint32 = 2
)

// TextureTransform is the KHR_texture_transform UV transform for one texture
type TextureTransform struct {
	Offset   [2]float32
	Rotation float32
	Scale    [2]float32
	UVSet    uint32
}

func IdentityTextureTransform() TextureTransform {
	return TextureTransform{Scale: [2]float32{1, 1}}
}

func (t TextureTransform) packed() (row0 [4]float32, row1 [4]float32) {
	cosR := float32(math.Cos(float64(t.Rotation)))
	sinR := float32(math.Sin(float64(t.Rotation)))
	row0 = [4]float32{cosR * t.Scale[0], sinR * t.Scale[1], t.Offset[0], float32(t.UVSet)}
	row1 = [4]float32{-sinR * t.Scale[0], cosR * t.Scale[1], t.Offset[1], 0}
	return
}

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
	EmissiveStrength  float32

	AlphaMode   AlphaMode
	AlphaCutoff float32
	DoubleSided bool
	Unlit       bool

	IOR float32

	NormalMapFlags uint32

	BaseTransform              TextureTransform
	NormalTransform            TextureTransform
	MetallicRoughnessTransform TextureTransform
	OcclusionTransform         TextureTransform
	EmissiveTransform          TextureTransform

	SpecularFactor      float32
	SpecularColorFactor [3]float32
	SpecularLayer       uint32
	SpecularColorLayer  uint32

	TransmissionFactor  float32
	TransmissionLayer   uint32
	Thickness           float32
	ThicknessLayer      uint32
	AttenuationColor    [3]float32
	AttenuationDistance float32
	Dispersion          float32

	AnisotropyStrength float32
	AnisotropyRotation float32
	AnisotropyLayer    uint32

	ClearcoatFactor          float32
	ClearcoatRoughnessFactor float32
	ClearcoatNormalScale     float32
	ClearcoatLayer           uint32
	ClearcoatRoughnessLayer  uint32
	ClearcoatNormalLayer     uint32

	SheenColorFactor     [3]float32
	SheenRoughnessFactor float32
	SheenColorLayer      uint32
	SheenRoughnessLayer  uint32

	IridescenceFactor         float32
	IridescenceIor            float32
	IridescenceThicknessMin   float32
	IridescenceThicknessMax   float32
	IridescenceLayer          uint32
	IridescenceThicknessLayer uint32

	DiffuseTransmissionFactor      float32
	DiffuseTransmissionColorFactor [3]float32
	DiffuseTransmissionColorLayer  uint32

	BlendOpaqueAlphaThreshold float32
}

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
		IOR:                    1.5,
		EmissiveStrength:       1.0,

		BaseTransform:              IdentityTextureTransform(),
		NormalTransform:            IdentityTextureTransform(),
		MetallicRoughnessTransform: IdentityTextureTransform(),
		OcclusionTransform:         IdentityTextureTransform(),
		EmissiveTransform:          IdentityTextureTransform(),

		SpecularFactor:      1.0,
		SpecularColorFactor: [3]float32{1, 1, 1},
		SpecularLayer:       NoTextureLayer,
		SpecularColorLayer:  NoTextureLayer,

		TransmissionLayer: NoTextureLayer,
		ThicknessLayer:    NoTextureLayer,
		AttenuationColor:  [3]float32{1, 1, 1},

		AnisotropyLayer: NoTextureLayer,

		ClearcoatLayer:          NoTextureLayer,
		ClearcoatRoughnessLayer: NoTextureLayer,
		ClearcoatNormalLayer:    NoTextureLayer,
		ClearcoatNormalScale:    1.0,

		SheenColorFactor:    [3]float32{0, 0, 0},
		SheenColorLayer:     NoTextureLayer,
		SheenRoughnessLayer: NoTextureLayer,

		IridescenceIor:            1.3,
		IridescenceThicknessMin:   100.0,
		IridescenceThicknessMax:   400.0,
		IridescenceLayer:          NoTextureLayer,
		IridescenceThicknessLayer: NoTextureLayer,

		DiffuseTransmissionColorFactor: [3]float32{1, 1, 1},
		DiffuseTransmissionColorLayer:  NoTextureLayer,

		BlendOpaqueAlphaThreshold: 0.99,
	}
}

func EmissiveMaterial(color [3]float32, strength float32) Material {
	m := DefaultMaterial()
	m.BaseColor = [4]float32{color[0], color[1], color[2], 1.0}
	m.EmissiveFactor = color
	m.EmissiveStrength = strength
	return m
}

func AlbedoMaterial(color [4]float32) Material {
	m := DefaultMaterial()
	m.BaseColor = color
	return m
}

type texTransformGPU struct {
	Row0 [4]float32
	Row1 [4]float32
}

// MaterialGPU mirrors the WGSL Material struct (std430).
type MaterialGPU struct {
	BaseColor      [4]float32
	EmissiveFactor [3]float32
	AlphaMode      uint32

	BaseLayer              uint32
	EmissiveLayer          uint32
	NormalLayer            uint32
	MetallicRoughnessLayer uint32

	OcclusionLayer    uint32
	NormalScale       float32
	OcclusionStrength float32
	MetallicFactor    float32

	RoughnessFactor float32
	AlphaCutoff     float32
	Unlit           uint32
	IOR             float32

	EmissiveStrength float32
	Pad1a            float32
	Pad1b            float32
	Pad1c            float32

	NormalMapFlags     uint32
	SpecularFactor     float32
	SpecularLayer      uint32
	SpecularColorLayer uint32

	SpecularColorFactor [3]float32
	TransmissionFactor  float32

	TransmissionLayer   uint32
	Thickness           float32
	ThicknessLayer      uint32
	AttenuationDistance float32

	AttenuationColor [3]float32
	Dispersion       float32

	AnisotropyStrength    float32
	AnisotropyRotationCos float32
	AnisotropyRotationSin float32
	AnisotropyLayer       uint32

	ClearcoatFactor          float32
	ClearcoatRoughnessFactor float32
	ClearcoatNormalScale     float32
	ClearcoatLayer           uint32

	ClearcoatRoughnessLayer uint32
	ClearcoatNormalLayer    uint32
	SheenColorLayer         uint32
	SheenRoughnessLayer     uint32

	SheenColorFactor     [3]float32
	SheenRoughnessFactor float32

	IridescenceFactor       float32
	IridescenceIor          float32
	IridescenceThicknessMin float32
	IridescenceThicknessMax float32

	IridescenceLayer              uint32
	IridescenceThicknessLayer     uint32
	DiffuseTransmissionFactor     float32
	DiffuseTransmissionColorLayer uint32

	DiffuseTransmissionColorFactor [3]float32
	BlendOpaqueAlphaThreshold      float32

	BaseTransform              texTransformGPU
	NormalTransform            texTransformGPU
	MetallicRoughnessTransform texTransformGPU
	OcclusionTransform         texTransformGPU
	EmissiveTransform          texTransformGPU
}

const MaterialGPUSize = uint64(432)

type _ [uintptr(MaterialGPUSize) - unsafe.Sizeof(MaterialGPU{})]byte
type _ [unsafe.Sizeof(MaterialGPU{}) - uintptr(MaterialGPUSize)]byte

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
	ior := m.IOR
	if ior <= 0 {
		ior = 1.5
	}
	emissiveStrength := m.EmissiveStrength
	if emissiveStrength <= 0 {
		emissiveStrength = 1.0
	}
	cosR := float32(math.Cos(float64(m.AnisotropyRotation)))
	sinR := float32(math.Sin(float64(m.AnisotropyRotation)))

	blendOpaqueAlphaThreshold := m.BlendOpaqueAlphaThreshold
	if blendOpaqueAlphaThreshold <= 0 {
		blendOpaqueAlphaThreshold = 0.99
	}

	baseRow0, baseRow1 := m.BaseTransform.packed()
	normalRow0, normalRow1 := m.NormalTransform.packed()
	mrRow0, mrRow1 := m.MetallicRoughnessTransform.packed()
	occRow0, occRow1 := m.OcclusionTransform.packed()
	emiRow0, emiRow1 := m.EmissiveTransform.packed()

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
		IOR:                    ior,
		EmissiveStrength:       emissiveStrength,

		NormalMapFlags:      m.NormalMapFlags,
		SpecularFactor:      m.SpecularFactor,
		SpecularLayer:       m.SpecularLayer,
		SpecularColorLayer:  m.SpecularColorLayer,
		SpecularColorFactor: m.SpecularColorFactor,

		TransmissionFactor:  m.TransmissionFactor,
		TransmissionLayer:   m.TransmissionLayer,
		Thickness:           m.Thickness,
		ThicknessLayer:      m.ThicknessLayer,
		AttenuationDistance: m.AttenuationDistance,
		AttenuationColor:    m.AttenuationColor,
		Dispersion:          m.Dispersion,

		AnisotropyStrength:    m.AnisotropyStrength,
		AnisotropyRotationCos: cosR,
		AnisotropyRotationSin: sinR,
		AnisotropyLayer:       m.AnisotropyLayer,

		ClearcoatFactor:          m.ClearcoatFactor,
		ClearcoatRoughnessFactor: m.ClearcoatRoughnessFactor,
		ClearcoatNormalScale:     m.ClearcoatNormalScale,
		ClearcoatLayer:           m.ClearcoatLayer,
		ClearcoatRoughnessLayer:  m.ClearcoatRoughnessLayer,
		ClearcoatNormalLayer:     m.ClearcoatNormalLayer,

		SheenColorFactor:     m.SheenColorFactor,
		SheenRoughnessFactor: m.SheenRoughnessFactor,
		SheenColorLayer:      m.SheenColorLayer,
		SheenRoughnessLayer:  m.SheenRoughnessLayer,

		IridescenceFactor:         m.IridescenceFactor,
		IridescenceIor:            m.IridescenceIor,
		IridescenceThicknessMin:   m.IridescenceThicknessMin,
		IridescenceThicknessMax:   m.IridescenceThicknessMax,
		IridescenceLayer:          m.IridescenceLayer,
		IridescenceThicknessLayer: m.IridescenceThicknessLayer,

		DiffuseTransmissionFactor:      m.DiffuseTransmissionFactor,
		DiffuseTransmissionColorFactor: m.DiffuseTransmissionColorFactor,
		DiffuseTransmissionColorLayer:  m.DiffuseTransmissionColorLayer,

		BlendOpaqueAlphaThreshold: blendOpaqueAlphaThreshold,

		BaseTransform:              texTransformGPU{Row0: baseRow0, Row1: baseRow1},
		NormalTransform:            texTransformGPU{Row0: normalRow0, Row1: normalRow1},
		MetallicRoughnessTransform: texTransformGPU{Row0: mrRow0, Row1: mrRow1},
		OcclusionTransform:         texTransformGPU{Row0: occRow0, Row1: occRow1},
		EmissiveTransform:          texTransformGPU{Row0: emiRow0, Row1: emiRow1},
	}
}
