package pass

import (
	"reflect"
	"regexp"
	"strings"
	"testing"

	"github.com/matthewjberger/indigo/render/asset"
)

// TestMaterialWGSLLayoutMatchesGPU computes the std430 byte offset of every
// field in the WGSL Material struct (the single source in material_struct.wgsl)
// and asserts the field count, per-field offsets, and total size all match the
// Go MaterialGPU the registry is written from. This catches drift the eye
// misses — added/removed fields, a reorder at the same total size, or a vec3
// not padded to a 16-byte boundary — all of which would read garbage on the
// GPU (the failure mode behind the black-transmission bug).
func TestMaterialWGSLLayoutMatchesGPU(t *testing.T) {
	type sizeAlign struct{ size, align int }
	types := map[string]sizeAlign{
		"f32":              {4, 4},
		"u32":              {4, 4},
		"i32":              {4, 4},
		"vec2<f32>":        {8, 8},
		"vec3<f32>":        {12, 16},
		"vec4<f32>":        {16, 16},
		"TextureTransform": {32, 16},
	}

	start := strings.Index(shaderMaterialStruct, "struct Material {")
	if start < 0 {
		t.Fatal("Material struct not found in material_struct.wgsl")
	}
	body := shaderMaterialStruct[start:]
	body = body[strings.Index(body, "{")+1 : strings.Index(body, "};")]

	fieldRe := regexp.MustCompile(`(?m)^\s*\w+:\s*([\w<>]+)\s*,`)
	matches := fieldRe.FindAllStringSubmatch(body, -1)
	if len(matches) == 0 {
		t.Fatal("no Material fields parsed")
	}

	alignUp := func(x, a int) int { return (x + a - 1) / a * a }
	wgslOffsets := make([]int, len(matches))
	offset, maxAlign := 0, 1
	for i, m := range matches {
		info, ok := types[m[1]]
		if !ok {
			t.Fatalf("unhandled WGSL field type %q", m[1])
		}
		offset = alignUp(offset, info.align)
		wgslOffsets[i] = offset
		offset += info.size
		if info.align > maxAlign {
			maxAlign = info.align
		}
	}
	total := alignUp(offset, maxAlign)

	if uint64(total) != asset.MaterialGPUSize {
		t.Fatalf("WGSL Material std430 size = %d, want MaterialGPUSize = %d", total, asset.MaterialGPUSize)
	}

	gpu := reflect.TypeOf(asset.MaterialGPU{})
	if gpu.NumField() != len(wgslOffsets) {
		t.Fatalf("field count: Go MaterialGPU has %d, WGSL Material has %d", gpu.NumField(), len(wgslOffsets))
	}
	for i := 0; i < gpu.NumField(); i++ {
		if got := int(gpu.Field(i).Offset); got != wgslOffsets[i] {
			t.Fatalf("field %d (%s): Go offset %d != WGSL std430 offset %d (layout drifted)",
				i, gpu.Field(i).Name, got, wgslOffsets[i])
		}
	}
}
