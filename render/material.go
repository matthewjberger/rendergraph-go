package render

// AlphaMode maps the glTF 2.0 material.alphaMode enum.
type AlphaMode uint8

const (
	AlphaModeOpaque AlphaMode = iota
	AlphaModeMask
	AlphaModeBlend
)

// Material is the per-entity surface-appearance component. The mesh
// shader currently samples BaseColorTexture and multiplies the result
// by vertex color and BaseColor. The remaining PBR fields are
// captured by the glTF loader and stored on the entity so future
// passes (proper PBR shading, alpha-masked draws, emissive bloom)
// can consume them without revisiting the importer.
type Material struct {
	BaseColor                [4]float32
	BaseColorTexture         TextureID
	MetallicFactor           float32
	RoughnessFactor          float32
	MetallicRoughnessTexture TextureID
	NormalTexture            TextureID
	NormalScale              float32
	OcclusionTexture         TextureID
	OcclusionStrength        float32
	EmissiveFactor           [3]float32
	EmissiveTexture          TextureID
	AlphaMode                AlphaMode
	AlphaCutoff              float32
	DoubleSided              bool
}

// DefaultMaterial returns a fully-opaque white material with the
// glTF-spec defaults (metallic=1.0, roughness=1.0, alpha=opaque).
func DefaultMaterial() Material {
	return Material{
		BaseColor:         [4]float32{1, 1, 1, 1},
		MetallicFactor:    1.0,
		RoughnessFactor:   1.0,
		NormalScale:       1.0,
		OcclusionStrength: 1.0,
		AlphaCutoff:       0.5,
	}
}

// materialDataUniform is the WGSL-aligned material entry layout the
// mesh pass packs into per-handle storage buffers. Kept narrow until
// the mesh shader actually consumes the extra PBR fields.
type materialDataUniform struct {
	BaseColor [4]float32
}

const materialDataSize = uint64(16)
