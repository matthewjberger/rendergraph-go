package asset

import (
	"encoding/json"

	"github.com/qmuntal/gltf"

	"github.com/matthewjberger/indigo/ecs"
)

// MaterialVariantMapping ties one or more KHR_materials_variants variant names
// to the material a primitive uses when that variant is active.
type MaterialVariantMapping struct {
	VariantNames []string
	Material     Material
}

// MaterialVariants is attached to a spawned primitive that declares
// KHR_materials_variants mappings. Default is the primitive's base material.
type MaterialVariants struct {
	Default  Material
	Mappings []MaterialVariantMapping
}

func (v *MaterialVariants) MaterialFor(variant string) Material {
	for index := range v.Mappings {
		for _, name := range v.Mappings[index].VariantNames {
			if name == variant {
				return v.Mappings[index].Material
			}
		}
	}
	return v.Default
}

func readVariantNames(doc *gltf.Document) []string {
	if doc == nil || doc.Extensions == nil {
		return nil
	}
	raw, ok := doc.Extensions["KHR_materials_variants"]
	if !ok {
		return nil
	}
	body, err := jsonRawFromExt(raw)
	if err != nil {
		return nil
	}
	var payload struct {
		Variants []struct {
			Name string `json:"name"`
		} `json:"variants"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil
	}
	names := make([]string, len(payload.Variants))
	for index, variant := range payload.Variants {
		names[index] = variant.Name
	}
	return names
}

func readPrimitiveVariants(prim *gltf.Primitive, materials []Material, variantNames []string, fallback Material) *MaterialVariants {
	if prim == nil || prim.Extensions == nil {
		return nil
	}
	raw, ok := prim.Extensions["KHR_materials_variants"]
	if !ok {
		return nil
	}
	body, err := jsonRawFromExt(raw)
	if err != nil {
		return nil
	}
	var payload struct {
		Mappings []struct {
			Material int   `json:"material"`
			Variants []int `json:"variants"`
		} `json:"mappings"`
	}
	if err := json.Unmarshal(body, &payload); err != nil || len(payload.Mappings) == 0 {
		return nil
	}
	result := &MaterialVariants{Default: fallback}
	for _, mapping := range payload.Mappings {
		if mapping.Material < 0 || mapping.Material >= len(materials) {
			continue
		}
		names := make([]string, 0, len(mapping.Variants))
		for _, index := range mapping.Variants {
			if index >= 0 && index < len(variantNames) {
				names = append(names, variantNames[index])
			}
		}
		if len(names) == 0 {
			continue
		}
		result.Mappings = append(result.Mappings, MaterialVariantMapping{VariantNames: names, Material: materials[mapping.Material]})
	}
	if len(result.Mappings) == 0 {
		return nil
	}
	return result
}

// ApplyMaterialVariant switches every variant-tagged entity to the material
// mapped to variantName, falling back to each primitive's default material
// when variantName is empty or unmapped.
func ApplyMaterialVariant(world *ecs.World, variantName string) {
	mask := ecs.MustMaskOf[MaterialVariants](world)
	world.ForEach(mask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		variants, ok := ecs.Get[MaterialVariants](world, entity)
		if !ok {
			return
		}
		material := variants.Default
		if variantName != "" {
			material = variants.MaterialFor(variantName)
		}
		if existing, ok := ecs.GetMut[Material](world, entity); ok {
			*existing = material
		}
	})
}
