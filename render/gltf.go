package render

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/go-gl/mathgl/mgl32"
	"github.com/qmuntal/gltf"
	"github.com/qmuntal/gltf/modeler"

	"indigo/ecs"
	"indigo/transform"
)

// LoadedScene is the result of parsing a glTF file: a flat list of
// nodes that together describe a scene tree, plus the registered
// mesh + material handles every primitive references. Use
// [SpawnLoadedScene] to materialize the tree into ECS entities, or
// walk Nodes yourself for custom spawn behavior.
type LoadedScene struct {
	Label      string
	Nodes      []SceneNode
	Roots      []int
	Meshes     []LoadedMesh
	Materials  []Material
	Animations []AnimationClip
}

// SceneNode is one glTF node: a TRS transform, optional mesh +
// material assignment, and child node indices. Multi-primitive
// meshes expand into ChildPrimitives so each primitive lands on its
// own entity (matching the glTF "one mesh, many primitives" model).
type SceneNode struct {
	Name            string
	Translation     [3]float32
	Rotation        [4]float32
	Scale           [3]float32
	Children        []int
	Mesh            MeshHandle
	HasMesh         bool
	Material        Material
	ChildPrimitives []SceneNodePrimitive
}

// SceneNodePrimitive is one mesh primitive in a multi-primitive glTF
// mesh node. The spawner gives each primitive its own child entity
// so they can carry distinct materials and per-primitive bounds.
type SceneNodePrimitive struct {
	Mesh     MeshHandle
	Material Material
}

// LoadedMesh names a registered mesh from this load.
type LoadedMesh struct {
	Handle MeshHandle
	Name   string
}

// LoadGltfFile reads a glTF or glb file from disk and forwards to
// [LoadGltfReader]. External-file image URIs are resolved relative
// to the parent directory of path.
func LoadGltfFile(device *wgpu.Device, queue *wgpu.Queue, assets *MeshAssets, cache *TextureCache, path string) (*LoadedScene, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("gltf %q: open: %w", path, err)
	}
	defer f.Close()
	opts := LoadGltfOptions{
		Label:   filepath.Base(path),
		BaseDir: filepath.Dir(path),
	}
	return LoadGltfReaderOpts(device, queue, assets, cache, f, opts)
}

// LoadGltfOptions tweaks reader-side behavior. Label is the name
// prefix used for registered meshes and textures; BaseDir is the
// directory used to resolve external image URIs (empty disables
// external resolution).
type LoadGltfOptions struct {
	Label   string
	BaseDir string
}

// LoadGltfReader is the LoadGltfReaderOpts shortcut with default
// options. Kept for callers that don't need a base dir.
func LoadGltfReader(device *wgpu.Device, queue *wgpu.Queue, assets *MeshAssets, cache *TextureCache, label string, r io.Reader) (*LoadedScene, error) {
	return LoadGltfReaderOpts(device, queue, assets, cache, r, LoadGltfOptions{Label: label})
}

// LoadGltfReaderOpts parses a glTF / glb document from r, uploads
// every primitive to assets, decodes every texture into cache,
// captures animation data, and returns a LoadedScene that mirrors
// the glTF node hierarchy. Caller can then [SpawnLoadedScene] to
// materialize entities.
func LoadGltfReaderOpts(device *wgpu.Device, queue *wgpu.Queue, assets *MeshAssets, cache *TextureCache, r io.Reader, opts LoadGltfOptions) (*LoadedScene, error) {
	label := opts.Label
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("gltf %q: read: %w", label, err)
	}
	dec := gltf.NewDecoder(bytes.NewReader(data))
	doc := new(gltf.Document)
	if err := dec.Decode(doc); err != nil {
		return nil, fmt.Errorf("gltf %q: decode: %w", label, err)
	}

	scene := &LoadedScene{Label: label}

	textureColorSpace := classifyTextures(doc)
	textureIDs, err := uploadTextures(device, queue, cache, label, doc, textureColorSpace, opts.BaseDir)
	if err != nil {
		return nil, err
	}

	materials := make([]Material, len(doc.Materials))
	for i := range doc.Materials {
		materials[i] = buildMaterial(doc.Materials[i], textureIDs)
	}
	scene.Materials = materials

	type primitiveResult struct {
		Handle   MeshHandle
		Material Material
	}
	meshPrimitives := make([][]primitiveResult, len(doc.Meshes))
	for meshIdx, mesh := range doc.Meshes {
		results := make([]primitiveResult, 0, len(mesh.Primitives))
		for primIdx, prim := range mesh.Primitives {
			vertices, err := buildGltfPrimitive(doc, prim)
			if err != nil {
				return nil, fmt.Errorf("gltf %q: mesh %d primitive %d: %w", label, meshIdx, primIdx, err)
			}
			name := mesh.Name
			if name == "" {
				name = fmt.Sprintf("%s/mesh_%d", label, meshIdx)
			}
			if len(mesh.Primitives) > 1 {
				name = fmt.Sprintf("%s.%d", name, primIdx)
			}
			handle, err := assets.Register(device, name, vertices)
			if err != nil {
				return nil, err
			}
			material := DefaultMaterial()
			if prim.Material != nil {
				material = materials[*prim.Material]
			}
			if material.BaseColorTexture != 0 {
				assets.AttachTexture(handle, material.BaseColorTexture)
			}
			results = append(results, primitiveResult{Handle: handle, Material: material})
			scene.Meshes = append(scene.Meshes, LoadedMesh{Handle: handle, Name: name})
		}
		meshPrimitives[meshIdx] = results
	}

	scene.Animations, err = readAnimations(doc)
	if err != nil {
		return nil, fmt.Errorf("gltf %q: %w", label, err)
	}

	scene.Nodes = make([]SceneNode, len(doc.Nodes))
	for i, node := range doc.Nodes {
		out := SceneNode{
			Name:        node.Name,
			Translation: nodeTranslation(node),
			Rotation:    nodeRotation(node),
			Scale:       nodeScale(node),
			Children:    childIndicesOf(node),
		}
		if node.Mesh != nil {
			prims := meshPrimitives[*node.Mesh]
			switch len(prims) {
			case 0:
			case 1:
				out.HasMesh = true
				out.Mesh = prims[0].Handle
				out.Material = prims[0].Material
			default:
				out.ChildPrimitives = make([]SceneNodePrimitive, len(prims))
				for j, p := range prims {
					out.ChildPrimitives[j] = SceneNodePrimitive{Mesh: p.Handle, Material: p.Material}
				}
			}
		}
		scene.Nodes[i] = out
	}

	if len(doc.Scenes) > 0 {
		sceneIdx := 0
		if doc.Scene != nil {
			sceneIdx = *doc.Scene
		}
		for _, idx := range doc.Scenes[sceneIdx].Nodes {
			scene.Roots = append(scene.Roots, int(idx))
		}
	} else {
		assigned := make(map[int]bool, len(scene.Nodes))
		for _, n := range scene.Nodes {
			for _, child := range n.Children {
				assigned[child] = true
			}
		}
		for i := range scene.Nodes {
			if !assigned[i] {
				scene.Roots = append(scene.Roots, i)
			}
		}
	}
	return scene, nil
}

// SpawnLoadedScene materializes every node in scene as an entity on
// world, preserving the glTF hierarchy via transform.Parent. Returns
// every spawned entity in node-index order (entities[i] is the
// entity for scene.Nodes[i]); roots are at entities[scene.Roots[k]].
//
// world must have transform.LocalTransform, transform.Parent,
// RenderMesh, and Material registered. The standard
// [indigo/app.NewEngineWorld] does this.
func SpawnLoadedScene(world *ecs.World, scene *LoadedScene) []ecs.Entity {
	entities := make([]ecs.Entity, len(scene.Nodes))
	localMask := ecs.MustMaskOf[transform.LocalTransform](world)
	globalMask := ecs.MustMaskOf[transform.GlobalTransform](world)
	dirtyMask := ecs.MustMaskOf[transform.LocalTransformDirty](world)
	parentMask := ecs.MustMaskOf[transform.Parent](world)
	meshMask := ecs.MustMaskOf[RenderMesh](world)
	materialMask := ecs.MustMaskOf[Material](world)

	transformMask := localMask | globalMask | dirtyMask
	for i := range scene.Nodes {
		entities[i] = world.Spawn(transformMask)
		ecs.Set(world, entities[i], transform.IdentityGlobalTransform())
	}
	for i, node := range scene.Nodes {
		entity := entities[i]
		local := transform.LocalTransform{
			Translation: transform.Vec3{node.Translation[0], node.Translation[1], node.Translation[2]},
			Rotation: transform.Quat{
				V: mgl32.Vec3{node.Rotation[0], node.Rotation[1], node.Rotation[2]},
				W: node.Rotation[3],
			},
			Scale: transform.Vec3{node.Scale[0], node.Scale[1], node.Scale[2]},
		}
		ecs.Set(world, entity, local)
		if node.HasMesh {
			world.AddComponents(entity, meshMask|materialMask)
			ecs.Set(world, entity, RenderMesh{Mesh: node.Mesh})
			ecs.Set(world, entity, node.Material)
		}
		for _, child := range node.Children {
			world.AddComponents(entities[child], parentMask)
			ecs.Set(world, entities[child], transform.Parent{Entity: entity})
		}
		for _, prim := range node.ChildPrimitives {
			child := world.Spawn(transformMask | parentMask | meshMask | materialMask)
			ecs.Set(world, child, transform.IdentityLocalTransform())
			ecs.Set(world, child, transform.IdentityGlobalTransform())
			ecs.Set(world, child, transform.Parent{Entity: entity})
			ecs.Set(world, child, RenderMesh{Mesh: prim.Mesh})
			ecs.Set(world, child, prim.Material)
		}
	}
	return entities
}

func nodeTranslation(node *gltf.Node) [3]float32 {
	if node.Translation != [3]float64{} {
		return [3]float32{float32(node.Translation[0]), float32(node.Translation[1]), float32(node.Translation[2])}
	}
	if node.Matrix != [16]float64{} {
		return [3]float32{float32(node.Matrix[12]), float32(node.Matrix[13]), float32(node.Matrix[14])}
	}
	return [3]float32{0, 0, 0}
}

func nodeRotation(node *gltf.Node) [4]float32 {
	if node.Rotation != [4]float64{} {
		return [4]float32{float32(node.Rotation[0]), float32(node.Rotation[1]), float32(node.Rotation[2]), float32(node.Rotation[3])}
	}
	return [4]float32{0, 0, 0, 1}
}

func nodeScale(node *gltf.Node) [3]float32 {
	if node.Scale != [3]float64{} {
		return [3]float32{float32(node.Scale[0]), float32(node.Scale[1]), float32(node.Scale[2])}
	}
	return [3]float32{1, 1, 1}
}

func childIndicesOf(node *gltf.Node) []int {
	if len(node.Children) == 0 {
		return nil
	}
	out := make([]int, len(node.Children))
	for i, c := range node.Children {
		out[i] = int(c)
	}
	return out
}

func classifyTextures(doc *gltf.Document) []TextureColorSpace {
	spaces := make([]TextureColorSpace, len(doc.Textures))
	for i := range spaces {
		spaces[i] = TextureLinear
	}
	mark := func(info *gltf.TextureInfo, space TextureColorSpace) {
		if info == nil {
			return
		}
		if int(info.Index) < len(spaces) {
			spaces[info.Index] = space
		}
	}
	for _, mat := range doc.Materials {
		if mat.PBRMetallicRoughness != nil {
			mark(mat.PBRMetallicRoughness.BaseColorTexture, TextureSRGB)
		}
		if mat.EmissiveTexture != nil {
			mark(mat.EmissiveTexture, TextureSRGB)
		}
	}
	return spaces
}

func uploadTextures(device *wgpu.Device, queue *wgpu.Queue, cache *TextureCache, label string, doc *gltf.Document, spaces []TextureColorSpace, baseDir string) ([]TextureID, error) {
	out := make([]TextureID, len(doc.Textures))
	for i, tex := range doc.Textures {
		if tex.Source == nil {
			continue
		}
		img := doc.Images[*tex.Source]
		pixels, w, h, err := decodeGltfImage(doc, img, baseDir)
		if err != nil {
			return nil, fmt.Errorf("gltf %q: image %d: %w", label, i, err)
		}
		name := fmt.Sprintf("%s/tex_%d", label, i)
		if img.Name != "" {
			name = label + "/" + img.Name
		}
		settings := samplerSettingsFor(doc, tex)
		id, err := cache.Register(device, queue, name, w, h, pixels, spaces[i], settings)
		if err != nil {
			return nil, err
		}
		out[i] = id
	}
	return out, nil
}

func samplerSettingsFor(doc *gltf.Document, tex *gltf.Texture) SamplerSettings {
	s := DefaultSamplerSettings()
	if tex.Sampler == nil {
		return s
	}
	sampler := doc.Samplers[*tex.Sampler]
	s.WrapU = mapWrap(sampler.WrapS)
	s.WrapV = mapWrap(sampler.WrapT)
	s.MagFilter = mapMagFilter(sampler.MagFilter)
	s.MinFilter, s.MipmapFilter = mapMinFilter(sampler.MinFilter)
	return s
}

func mapWrap(mode gltf.WrappingMode) wgpu.AddressMode {
	switch mode {
	case gltf.WrapClampToEdge:
		return wgpu.AddressModeClampToEdge
	case gltf.WrapMirroredRepeat:
		return wgpu.AddressModeMirrorRepeat
	default:
		return wgpu.AddressModeRepeat
	}
}

func mapMagFilter(filter gltf.MagFilter) wgpu.FilterMode {
	if filter == gltf.MagNearest {
		return wgpu.FilterModeNearest
	}
	return wgpu.FilterModeLinear
}

func mapMinFilter(filter gltf.MinFilter) (wgpu.FilterMode, wgpu.MipmapFilterMode) {
	switch filter {
	case gltf.MinNearest, gltf.MinNearestMipMapNearest:
		return wgpu.FilterModeNearest, wgpu.MipmapFilterModeNearest
	case gltf.MinLinearMipMapNearest:
		return wgpu.FilterModeLinear, wgpu.MipmapFilterModeNearest
	case gltf.MinNearestMipMapLinear:
		return wgpu.FilterModeNearest, wgpu.MipmapFilterModeLinear
	default:
		return wgpu.FilterModeLinear, wgpu.MipmapFilterModeLinear
	}
}

func buildMaterial(src *gltf.Material, textureIDs []TextureID) Material {
	out := DefaultMaterial()
	if src == nil {
		return out
	}
	if src.PBRMetallicRoughness != nil {
		pbr := src.PBRMetallicRoughness
		bc := pbr.BaseColorFactorOrDefault()
		out.BaseColor = [4]float32{float32(bc[0]), float32(bc[1]), float32(bc[2]), float32(bc[3])}
		out.MetallicFactor = float32(pbr.MetallicFactorOrDefault())
		out.RoughnessFactor = float32(pbr.RoughnessFactorOrDefault())
		if pbr.BaseColorTexture != nil {
			out.BaseColorTexture = textureIDs[pbr.BaseColorTexture.Index]
		}
		if pbr.MetallicRoughnessTexture != nil {
			out.MetallicRoughnessTexture = textureIDs[pbr.MetallicRoughnessTexture.Index]
		}
	}
	if src.NormalTexture != nil {
		if src.NormalTexture.Index != nil {
			out.NormalTexture = textureIDs[*src.NormalTexture.Index]
		}
		out.NormalScale = float32(src.NormalTexture.ScaleOrDefault())
	}
	if src.OcclusionTexture != nil {
		if src.OcclusionTexture.Index != nil {
			out.OcclusionTexture = textureIDs[*src.OcclusionTexture.Index]
		}
		out.OcclusionStrength = float32(src.OcclusionTexture.StrengthOrDefault())
	}
	if src.EmissiveTexture != nil {
		out.EmissiveTexture = textureIDs[src.EmissiveTexture.Index]
	}
	if src.EmissiveFactor != [3]float64{} {
		out.EmissiveFactor = [3]float32{float32(src.EmissiveFactor[0]), float32(src.EmissiveFactor[1]), float32(src.EmissiveFactor[2])}
	}
	switch src.AlphaMode {
	case gltf.AlphaMask:
		out.AlphaMode = AlphaModeMask
	case gltf.AlphaBlend:
		out.AlphaMode = AlphaModeBlend
	default:
		out.AlphaMode = AlphaModeOpaque
	}
	out.AlphaCutoff = float32(src.AlphaCutoffOrDefault())
	out.DoubleSided = src.DoubleSided
	return out
}

func buildGltfPrimitive(doc *gltf.Document, prim *gltf.Primitive) ([]MeshVertex, error) {
	posAttr, ok := prim.Attributes["POSITION"]
	if !ok {
		return nil, fmt.Errorf("primitive missing POSITION attribute")
	}
	positions, err := modeler.ReadPosition(doc, doc.Accessors[posAttr], nil)
	if err != nil {
		return nil, fmt.Errorf("read position: %w", err)
	}

	var normals [][3]float32
	if idx, ok := prim.Attributes["NORMAL"]; ok {
		normals, err = modeler.ReadNormal(doc, doc.Accessors[idx], nil)
		if err != nil {
			return nil, fmt.Errorf("read normal: %w", err)
		}
	}

	var tangents [][4]float32
	if idx, ok := prim.Attributes["TANGENT"]; ok {
		tangents, err = modeler.ReadTangent(doc, doc.Accessors[idx], nil)
		if err != nil {
			return nil, fmt.Errorf("read tangent: %w", err)
		}
	}

	var uvs [][2]float32
	if idx, ok := prim.Attributes["TEXCOORD_0"]; ok {
		uvs, err = modeler.ReadTextureCoord(doc, doc.Accessors[idx], nil)
		if err != nil {
			return nil, fmt.Errorf("read texcoord_0: %w", err)
		}
	}

	var uvs1 [][2]float32
	if idx, ok := prim.Attributes["TEXCOORD_1"]; ok {
		uvs1, err = modeler.ReadTextureCoord(doc, doc.Accessors[idx], nil)
		if err != nil {
			return nil, fmt.Errorf("read texcoord_1: %w", err)
		}
	}

	var colors [][4]uint8
	if idx, ok := prim.Attributes["COLOR_0"]; ok {
		colors, err = modeler.ReadColor(doc, doc.Accessors[idx], nil)
		if err != nil {
			return nil, fmt.Errorf("read color: %w", err)
		}
	}

	indices, err := readIndices(doc, prim)
	if err != nil {
		return nil, err
	}

	expanded := make([]MeshVertex, 0, len(indices))
	for _, srcIdx := range indices {
		v := MeshVertex{
			Position: [4]float32{positions[srcIdx][0], positions[srcIdx][1], positions[srcIdx][2], 1},
			Normal:   defaultNormalZ,
			Tangent:  defaultTangent,
			Color:    [4]float32{1, 1, 1, 1},
		}
		if int(srcIdx) < len(normals) {
			v.Normal = [4]float32{normals[srcIdx][0], normals[srcIdx][1], normals[srcIdx][2], 0}
		}
		if int(srcIdx) < len(tangents) {
			v.Tangent = tangents[srcIdx]
		}
		var u0x, u0y, u1x, u1y float32
		if int(srcIdx) < len(uvs) {
			u0x = uvs[srcIdx][0]
			u0y = uvs[srcIdx][1]
		}
		if int(srcIdx) < len(uvs1) {
			u1x = uvs1[srcIdx][0]
			u1y = uvs1[srcIdx][1]
		}
		v.UV = [4]float32{u0x, u0y, u1x, u1y}
		if int(srcIdx) < len(colors) {
			c := colors[srcIdx]
			v.Color = [4]float32{float32(c[0]) / 255, float32(c[1]) / 255, float32(c[2]) / 255, float32(c[3]) / 255}
		}
		expanded = append(expanded, v)
	}
	return expanded, nil
}

func readIndices(doc *gltf.Document, prim *gltf.Primitive) ([]uint32, error) {
	if prim.Indices == nil {
		posAttr := prim.Attributes["POSITION"]
		count := doc.Accessors[posAttr].Count
		out := make([]uint32, count)
		for i := range out {
			out[i] = uint32(i)
		}
		return out, nil
	}
	idx, err := modeler.ReadIndices(doc, doc.Accessors[*prim.Indices], nil)
	if err != nil {
		return nil, fmt.Errorf("read indices: %w", err)
	}
	return idx, nil
}

func readAnimations(doc *gltf.Document) ([]AnimationClip, error) {
	if len(doc.Animations) == 0 {
		return nil, nil
	}
	out := make([]AnimationClip, len(doc.Animations))
	for i, anim := range doc.Animations {
		clip := AnimationClip{Name: anim.Name}
		samplers := make([]AnimationSampler, len(anim.Samplers))
		for sIdx, s := range anim.Samplers {
			sampler := AnimationSampler{Interpolation: mapInterpolation(s.Interpolation)}
			inputAcc := doc.Accessors[s.Input]
			inputs := make([]float32, inputAcc.Count)
			if err := accessorScalarFloat(doc, inputAcc, inputs); err != nil {
				return nil, fmt.Errorf("animation %d sampler %d input: %w", i, sIdx, err)
			}
			sampler.Inputs = inputs
			outputAcc := doc.Accessors[s.Output]
			switch outputAcc.Type {
			case gltf.AccessorVec3:
				vals := make([][3]float32, outputAcc.Count)
				if err := accessorVec3(doc, outputAcc, vals); err != nil {
					return nil, fmt.Errorf("animation %d sampler %d output: %w", i, sIdx, err)
				}
				sampler.Vec3Outputs = vals
			case gltf.AccessorVec4:
				vals := make([][4]float32, outputAcc.Count)
				if err := accessorVec4(doc, outputAcc, vals); err != nil {
					return nil, fmt.Errorf("animation %d sampler %d output: %w", i, sIdx, err)
				}
				sampler.Vec4Outputs = vals
			case gltf.AccessorScalar:
				vals := make([]float32, outputAcc.Count)
				if err := accessorScalarFloat(doc, outputAcc, vals); err != nil {
					return nil, fmt.Errorf("animation %d sampler %d output: %w", i, sIdx, err)
				}
				sampler.ScalarOutputs = [][]float32{vals}
			}
			samplers[sIdx] = sampler
			for _, t := range inputs {
				if t > clip.Duration {
					clip.Duration = t
				}
			}
		}
		clip.Channels = make([]AnimationChannel, 0, len(anim.Channels))
		for _, c := range anim.Channels {
			if c.Sampler < 0 || c.Sampler >= len(samplers) {
				continue
			}
			node := -1
			if c.Target.Node != nil {
				node = int(*c.Target.Node)
			}
			ch := AnimationChannel{
				TargetNode: node,
				Property:   mapTargetPath(c.Target.Path),
				Sampler:    samplers[c.Sampler],
			}
			clip.Channels = append(clip.Channels, ch)
		}
		out[i] = clip
	}
	return out, nil
}

func mapInterpolation(mode gltf.Interpolation) AnimationInterpolation {
	switch mode {
	case gltf.InterpolationStep:
		return InterpolationStep
	case gltf.InterpolationCubicSpline:
		return InterpolationCubicSpline
	default:
		return InterpolationLinear
	}
}

func mapTargetPath(path gltf.TRSProperty) AnimationProperty {
	switch path {
	case gltf.TRSRotation:
		return AnimationRotation
	case gltf.TRSScale:
		return AnimationScale
	case gltf.TRSWeights:
		return AnimationMorphWeights
	default:
		return AnimationTranslation
	}
}

func accessorBytes(doc *gltf.Document, acc *gltf.Accessor) ([]byte, error) {
	if acc.BufferView == nil {
		return nil, fmt.Errorf("accessor without bufferView not supported")
	}
	view := doc.BufferViews[*acc.BufferView]
	buf := doc.Buffers[view.Buffer]
	start := int(view.ByteOffset) + int(acc.ByteOffset)
	componentSize := acc.ComponentType.ByteSize() * acc.Type.Components()
	stride := int(view.ByteStride)
	if stride == 0 {
		stride = componentSize
	}
	end := start + (int(acc.Count)-1)*stride + componentSize
	if end > len(buf.Data) {
		return nil, fmt.Errorf("accessor out of buffer range")
	}
	if stride == componentSize {
		return buf.Data[start:end], nil
	}
	out := make([]byte, int(acc.Count)*componentSize)
	for i := 0; i < acc.Count; i++ {
		copy(out[int(i)*componentSize:], buf.Data[start+int(i)*stride:start+int(i)*stride+componentSize])
	}
	return out, nil
}

func accessorScalarFloat(doc *gltf.Document, acc *gltf.Accessor, out []float32) error {
	raw, err := accessorBytes(doc, acc)
	if err != nil {
		return err
	}
	if len(raw) < int(acc.Count)*4 {
		return fmt.Errorf("scalar accessor expects float32")
	}
	for i := 0; i < acc.Count; i++ {
		bits := binary.LittleEndian.Uint32(raw[i*4 : i*4+4])
		out[i] = math.Float32frombits(bits)
	}
	return nil
}

func accessorVec3(doc *gltf.Document, acc *gltf.Accessor, out [][3]float32) error {
	raw, err := accessorBytes(doc, acc)
	if err != nil {
		return err
	}
	if len(raw) < int(acc.Count)*12 {
		return fmt.Errorf("vec3 accessor expects 12 bytes per element")
	}
	for i := 0; i < acc.Count; i++ {
		base := i * 12
		out[i][0] = math.Float32frombits(binary.LittleEndian.Uint32(raw[base : base+4]))
		out[i][1] = math.Float32frombits(binary.LittleEndian.Uint32(raw[base+4 : base+8]))
		out[i][2] = math.Float32frombits(binary.LittleEndian.Uint32(raw[base+8 : base+12]))
	}
	return nil
}

func accessorVec4(doc *gltf.Document, acc *gltf.Accessor, out [][4]float32) error {
	raw, err := accessorBytes(doc, acc)
	if err != nil {
		return err
	}
	if len(raw) < int(acc.Count)*16 {
		return fmt.Errorf("vec4 accessor expects 16 bytes per element")
	}
	for i := 0; i < acc.Count; i++ {
		base := i * 16
		out[i][0] = math.Float32frombits(binary.LittleEndian.Uint32(raw[base : base+4]))
		out[i][1] = math.Float32frombits(binary.LittleEndian.Uint32(raw[base+4 : base+8]))
		out[i][2] = math.Float32frombits(binary.LittleEndian.Uint32(raw[base+8 : base+12]))
		out[i][3] = math.Float32frombits(binary.LittleEndian.Uint32(raw[base+12 : base+16]))
	}
	return nil
}

func accessorMat4(doc *gltf.Document, acc *gltf.Accessor, out [][4][4]float32) error {
	raw, err := accessorBytes(doc, acc)
	if err != nil {
		return err
	}
	const matBytes = 64
	if len(raw) < int(acc.Count)*matBytes {
		return fmt.Errorf("mat4 accessor expects 64 bytes per element")
	}
	for i := 0; i < acc.Count; i++ {
		base := i * matBytes
		for col := 0; col < 4; col++ {
			for row := 0; row < 4; row++ {
				offset := base + col*16 + row*4
				out[i][col][row] = math.Float32frombits(binary.LittleEndian.Uint32(raw[offset : offset+4]))
			}
		}
	}
	return nil
}

func decodeGltfImage(doc *gltf.Document, img *gltf.Image, baseDir string) ([]byte, uint32, uint32, error) {
	raw, err := imageBytes(doc, img, baseDir)
	if err != nil {
		return nil, 0, 0, err
	}
	decoded, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		return nil, 0, 0, fmt.Errorf("decode: %w", err)
	}
	bounds := decoded.Bounds()
	w := uint32(bounds.Dx())
	h := uint32(bounds.Dy())
	pixels := make([]byte, w*h*4)
	for y := 0; y < int(h); y++ {
		for x := 0; x < int(w); x++ {
			r, g, b, a := decoded.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			i := (y*int(w) + x) * 4
			pixels[i+0] = byte(r >> 8)
			pixels[i+1] = byte(g >> 8)
			pixels[i+2] = byte(b >> 8)
			pixels[i+3] = byte(a >> 8)
		}
	}
	return pixels, w, h, nil
}

func imageBytes(doc *gltf.Document, img *gltf.Image, baseDir string) ([]byte, error) {
	if img.BufferView != nil {
		view := doc.BufferViews[*img.BufferView]
		buf := doc.Buffers[view.Buffer]
		start := int(view.ByteOffset)
		end := start + int(view.ByteLength)
		if end > len(buf.Data) {
			return nil, fmt.Errorf("bufferView %d out of range", *img.BufferView)
		}
		return buf.Data[start:end], nil
	}
	if img.URI != "" {
		if strings.HasPrefix(img.URI, "data:") {
			return decodeDataURI(img.URI)
		}
		if baseDir == "" {
			return nil, fmt.Errorf("external image %q has no base dir to resolve against", img.URI)
		}
		unescaped, err := url.QueryUnescape(img.URI)
		if err != nil {
			return nil, fmt.Errorf("unescape image URI %q: %w", img.URI, err)
		}
		path := filepath.Join(baseDir, unescaped)
		raw, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("read image %q: %w", path, err)
		}
		return raw, nil
	}
	return nil, fmt.Errorf("image has no source data")
}

func decodeDataURI(uri string) ([]byte, error) {
	comma := strings.IndexByte(uri, ',')
	if comma < 0 {
		return nil, fmt.Errorf("malformed data URI")
	}
	header := uri[5:comma]
	payload := uri[comma+1:]
	if strings.HasSuffix(header, ";base64") {
		return base64.StdEncoding.DecodeString(payload)
	}
	unescaped, err := url.QueryUnescape(payload)
	if err != nil {
		return nil, err
	}
	return []byte(unescaped), nil
}
