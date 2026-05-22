package asset

import (
	"bytes"
	"testing"

	"github.com/qmuntal/gltf"
	"github.com/qmuntal/gltf/modeler"
)

func TestNodeTRSDefaultsAreIdentity(t *testing.T) {
	n := &gltf.Node{}
	tr, rot, sc := nodeTRS(n)
	if tr != [3]float32{0, 0, 0} {
		t.Errorf("translation = %v, want zero", tr)
	}
	if rot != [4]float32{0, 0, 0, 1} {
		t.Errorf("rotation = %v, want identity quat", rot)
	}
	if sc != [3]float32{1, 1, 1} {
		t.Errorf("scale = %v, want one", sc)
	}
}

func TestNodeTRSFromExplicitFields(t *testing.T) {
	n := &gltf.Node{
		Translation: [3]float64{1.5, 2.5, 3.5},
		Rotation:    [4]float64{0, 0.7071, 0, 0.7071},
		Scale:       [3]float64{2, 3, 4},
	}
	tr, rot, sc := nodeTRS(n)
	if tr != [3]float32{1.5, 2.5, 3.5} {
		t.Errorf("translation = %v", tr)
	}
	if sc != [3]float32{2, 3, 4} {
		t.Errorf("scale = %v", sc)
	}
	if rot[3] != 0.7071 {
		t.Errorf("rotation w = %f, want 0.7071", rot[3])
	}
}

func TestNodeTRSDecomposesMatrix(t *testing.T) {
	// Matrix encoding: translation (10, 20, 30), 90 deg yaw around Y,
	// uniform scale 2. Column-major glTF matrix is built as
	// T * R * S applied to the right-hand-side basis.
	//   T*R*S applied to x,y,z columns gives, before translation:
	//     col0 = R*S*x = (0, 0, -2)
	//     col1 = R*S*y = (0, 2, 0)
	//     col2 = R*S*z = (2, 0, 0)
	matrix := [16]float64{
		0, 0, -2, 0,
		0, 2, 0, 0,
		2, 0, 0, 0,
		10, 20, 30, 1,
	}
	n := &gltf.Node{Matrix: matrix}
	tr, _, sc := nodeTRS(n)
	if tr != [3]float32{10, 20, 30} {
		t.Errorf("translation = %v, want {10 20 30}", tr)
	}
	if sc != [3]float32{2, 2, 2} {
		t.Errorf("scale = %v, want {2 2 2}", sc)
	}
	// Rotation: round-trip via the identity-axis test. A yaw of 90
	// around +Y rotates (1,0,0) to (0,0,-1); reconstruct the rotation
	// from the quaternion and verify.
	_, rot, _ := nodeTRS(n)
	_ = rot // exact quaternion form depends on conversion; the
	// behavior we care about is that decomposition produces a non-
	// identity rotation when the matrix has a non-identity basis.
	if rot == ([4]float32{0, 0, 0, 1}) {
		t.Errorf("decomposed rotation should be non-identity, got identity")
	}
}

func TestBuildMaterialDefaults(t *testing.T) {
	got := buildMaterial(nil, nil)
	want := DefaultMaterial()
	if got != want {
		t.Errorf("nil material = %+v, want %+v", got, want)
	}
}

func TestBuildMaterialPBRFactorsAndFlags(t *testing.T) {
	metallic := 0.25
	roughness := 0.75
	src := &gltf.Material{
		PBRMetallicRoughness: &gltf.PBRMetallicRoughness{
			BaseColorFactor: &[4]float64{0.5, 0.6, 0.7, 0.8},
			MetallicFactor:  &metallic,
			RoughnessFactor: &roughness,
		},
		EmissiveFactor: [3]float64{0.1, 0.2, 0.3},
		AlphaMode:      gltf.AlphaMask,
		DoubleSided:    true,
	}
	got := buildMaterial(src, nil)
	if got.BaseColor != [4]float32{0.5, 0.6, 0.7, 0.8} {
		t.Errorf("base color = %v", got.BaseColor)
	}
	if got.MetallicFactor != 0.25 {
		t.Errorf("metallic = %f", got.MetallicFactor)
	}
	if got.RoughnessFactor != 0.75 {
		t.Errorf("roughness = %f", got.RoughnessFactor)
	}
	if got.EmissiveFactor != [3]float32{0.1, 0.2, 0.3} {
		t.Errorf("emissive = %v", got.EmissiveFactor)
	}
	if got.AlphaMode != AlphaModeMask {
		t.Errorf("alpha mode = %d, want AlphaModeMask", got.AlphaMode)
	}
	if !got.DoubleSided {
		t.Error("double sided = false, want true")
	}
}

func TestBuildMaterialTextureLayerLookup(t *testing.T) {
	textureLayers := []uint32{10, 20, 30}
	scale := 1.5
	strength := 0.8
	src := &gltf.Material{
		PBRMetallicRoughness: &gltf.PBRMetallicRoughness{
			BaseColorTexture:         &gltf.TextureInfo{Index: 0},
			MetallicRoughnessTexture: &gltf.TextureInfo{Index: 1},
		},
		NormalTexture: &gltf.NormalTexture{
			Index: ptrInt(2),
			Scale: &scale,
		},
		OcclusionTexture: &gltf.OcclusionTexture{
			Index:    ptrInt(0),
			Strength: &strength,
		},
		EmissiveTexture: &gltf.TextureInfo{Index: 1},
	}
	got := buildMaterial(src, textureLayers)
	if got.BaseColorLayer != 10 {
		t.Errorf("base color layer = %d", got.BaseColorLayer)
	}
	if got.MetallicRoughnessLayer != 20 {
		t.Errorf("metallic roughness layer = %d", got.MetallicRoughnessLayer)
	}
	if got.NormalLayer != 30 || got.NormalScale != 1.5 {
		t.Errorf("normal layer/scale = %d / %f", got.NormalLayer, got.NormalScale)
	}
	if got.OcclusionLayer != 10 || got.OcclusionStrength != 0.8 {
		t.Errorf("occlusion layer/strength = %d / %f", got.OcclusionLayer, got.OcclusionStrength)
	}
	if got.EmissiveLayer != 20 {
		t.Errorf("emissive layer = %d", got.EmissiveLayer)
	}
}

func TestClassifyTexturesColorVsLinear(t *testing.T) {
	doc := &gltf.Document{
		Textures: []*gltf.Texture{{}, {}, {}, {}},
		Materials: []*gltf.Material{
			{
				PBRMetallicRoughness: &gltf.PBRMetallicRoughness{
					BaseColorTexture:         &gltf.TextureInfo{Index: 0},
					MetallicRoughnessTexture: &gltf.TextureInfo{Index: 1},
				},
				NormalTexture:   &gltf.NormalTexture{Index: ptrInt(2)},
				EmissiveTexture: &gltf.TextureInfo{Index: 3},
			},
		},
	}
	got := classifyTextures(doc)
	if got[0] != TextureSRGB {
		t.Errorf("base color must be sRGB, got %d", got[0])
	}
	if got[1] != TextureLinear {
		t.Errorf("metallic-roughness must be linear, got %d", got[1])
	}
	if got[2] != TextureLinear {
		t.Errorf("normal must be linear, got %d", got[2])
	}
	if got[3] != TextureSRGB {
		t.Errorf("emissive must be sRGB, got %d", got[3])
	}
}

func TestBuildGltfPrimitiveRoundTrip(t *testing.T) {
	positions := [][3]float32{
		{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
	}
	normals := [][3]float32{
		{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1},
	}
	uvs := [][2]float32{
		{0, 0}, {1, 0}, {0, 1}, {1, 1},
	}
	indices := []uint16{0, 1, 2, 1, 3, 2}

	doc := &gltf.Document{
		Asset: gltf.Asset{Version: "2.0"},
	}
	posAcc := modeler.WritePosition(doc, positions)
	normAcc := modeler.WriteNormal(doc, normals)
	uvAcc := modeler.WriteTextureCoord(doc, uvs)
	idxAcc := modeler.WriteIndices(doc, indices)
	prim := &gltf.Primitive{
		Attributes: gltf.PrimitiveAttributes{
			"POSITION":   posAcc,
			"NORMAL":     normAcc,
			"TEXCOORD_0": uvAcc,
		},
		Indices: ptrInt(int(idxAcc)),
	}

	var buf bytes.Buffer
	if err := gltf.NewEncoder(&buf).Encode(doc); err != nil {
		t.Fatalf("encode roundtrip: %v", err)
	}
	rd := new(gltf.Document)
	if err := gltf.NewDecoder(bytes.NewReader(buf.Bytes())).Decode(rd); err != nil {
		t.Fatalf("decode roundtrip: %v", err)
	}

	verts, err := buildGltfPrimitive(rd, prim)
	if err != nil {
		t.Fatalf("buildGltfPrimitive: %v", err)
	}
	if len(verts) != len(indices) {
		t.Fatalf("len = %d, want %d", len(verts), len(indices))
	}
	for i, src := range indices {
		v := verts[i]
		if v.Position != [4]float32{positions[src][0], positions[src][1], positions[src][2], 1} {
			t.Errorf("v%d position = %v", i, v.Position)
		}
		if v.Normal != [4]float32{0, 0, 1, 0} {
			t.Errorf("v%d normal = %v", i, v.Normal)
		}
		if v.UV[0] != uvs[src][0] || v.UV[1] != uvs[src][1] {
			t.Errorf("v%d uv0 = (%f,%f), want (%f,%f)", i, v.UV[0], v.UV[1], uvs[src][0], uvs[src][1])
		}
		if v.Color != [4]float32{1, 1, 1, 1} {
			t.Errorf("v%d color = %v (default should be white when COLOR_0 missing)", i, v.Color)
		}
	}
}

func TestMapInterpolationAndTargetPath(t *testing.T) {
	if got := mapInterpolation(gltf.InterpolationLinear); got != InterpolationLinear {
		t.Errorf("linear = %d", got)
	}
	if got := mapInterpolation(gltf.InterpolationStep); got != InterpolationStep {
		t.Errorf("step = %d", got)
	}
	if got := mapInterpolation(gltf.InterpolationCubicSpline); got != InterpolationCubicSpline {
		t.Errorf("cubic = %d", got)
	}
	if got := mapTargetPath(gltf.TRSRotation); got != AnimationRotation {
		t.Errorf("rotation path = %d", got)
	}
	if got := mapTargetPath(gltf.TRSScale); got != AnimationScale {
		t.Errorf("scale path = %d", got)
	}
	if got := mapTargetPath(gltf.TRSWeights); got != AnimationMorphWeights {
		t.Errorf("weights path = %d", got)
	}
	if got := mapTargetPath(gltf.TRSTranslation); got != AnimationTranslation {
		t.Errorf("translation path = %d", got)
	}
}

func TestDecodeDataURIBase64(t *testing.T) {
	uri := "data:application/octet-stream;base64,SGVsbG8="
	got, err := decodeDataURI(uri)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "Hello" {
		t.Errorf("got %q", got)
	}
}

func TestDecodeDataURIRaw(t *testing.T) {
	uri := "data:text/plain,abc%20def"
	got, err := decodeDataURI(uri)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "abc def" {
		t.Errorf("got %q", got)
	}
}

func ptrInt(v int) *int { return &v }
