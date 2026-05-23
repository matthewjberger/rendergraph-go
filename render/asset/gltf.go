package asset

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
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
	xdraw "golang.org/x/image/draw"

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
	Lights     []LoadedLight
	Cameras    []LoadedCamera
	// SkinSpecs[i] holds the joint node indices + inverse-bind
	// matrices for glTF skin i. SpawnLoadedScene materializes each
	// SkinSpec into an asset.Skin once joint entities exist.
	SkinSpecs []SkinSpec
	// skinnedPrimSpawned records the child entities
	// SpawnLoadedScene creates for multi-primitive skinned mesh
	// nodes so the per-skin resolution pass can attach SkinnedMesh
	// components after the joint entities exist.
	skinnedPrimSpawned []skinnedSpawnRecord
}

// SkinSpec describes one glTF skin without GPU resources: the
// joint nodes' indices into LoadedScene.Nodes and the
// inverse-bind matrices in joint order. Resolved into an
// asset.Skin at spawn time.
type SkinSpec struct {
	JointNodeIndices    []int
	InverseBindMatrices []mgl32.Mat4
}

// skinnedSpawnRecord tracks each multi-primitive skinned child
// entity SpawnLoadedScene creates so the post-spawn skin-
// resolution loop can attach SkinnedMesh components after the
// asset.Skin instances exist.
type skinnedSpawnRecord struct {
	NodeIdx     int
	SkinnedMesh SkinnedMeshHandle
	Entity      ecs.Entity
}

// LoadedLightType mirrors render.LightType without depending on
// the render package (which already imports this package).
type LoadedLightType uint8

const (
	LoadedLightDirectional LoadedLightType = iota
	LoadedLightPoint
	LoadedLightSpot
)

// LoadedLight is one entry from the document's
// KHR_lights_punctual.lights array. Nodes that reference it via
// their own KHR_lights_punctual extension store the index in
// SceneNode.LightIndex.
type LoadedLight struct {
	Name           string
	Type           LoadedLightType
	Color          [3]float32
	Intensity      float32
	Range          float32
	InnerConeAngle float32
	OuterConeAngle float32
}

// LoadedCamera carries the perspective parameters of a glTF
// camera. Nodes referencing it store the index in
// SceneNode.CameraIndex.
type LoadedCamera struct {
	Name        string
	FovYRadians float32
	Near        float32
	Far         float32
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
	// SkinIndex points into LoadedScene.SkinSpecs when this node
	// has a glTF skin assigned; -1 otherwise. When >= 0, Mesh /
	// ChildPrimitives are SkinnedMeshHandles rather than
	// MeshHandles.
	SkinIndex int
	// LightIndex points into LoadedScene.Lights when this node
	// carries a KHR_lights_punctual light reference; -1 otherwise.
	LightIndex int
	// CameraIndex points into LoadedScene.Cameras when this node
	// has a camera; -1 otherwise.
	CameraIndex int
	// SkinnedMesh holds the skinned counterpart of Mesh when
	// SkinIndex >= 0 and the primitive carried JOINTS_0 attrs.
	SkinnedMesh       SkinnedMeshHandle
	HasSkinnedMesh    bool
	ChildSkinnedPrims []SceneNodeSkinnedPrimitive
}

// SceneNodeSkinnedPrimitive is one rigged primitive of a
// multi-primitive glTF mesh node.
type SceneNodeSkinnedPrimitive struct {
	Mesh     SkinnedMeshHandle
	Material Material
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
func LoadGltfFile(device *wgpu.Device, queue *wgpu.Queue, assets *MeshAssets, skinnedAssets *SkinnedMeshAssets, arrays *MaterialTextureArrays, path string) (*LoadedScene, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("gltf %q: open: %w", path, err)
	}
	defer f.Close()
	opts := LoadGltfOptions{
		Label:   filepath.Base(path),
		BaseDir: filepath.Dir(path),
	}
	return LoadGltfReaderOpts(device, queue, assets, skinnedAssets, arrays, f, opts)
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
func LoadGltfReader(device *wgpu.Device, queue *wgpu.Queue, assets *MeshAssets, skinnedAssets *SkinnedMeshAssets, arrays *MaterialTextureArrays, label string, r io.Reader) (*LoadedScene, error) {
	return LoadGltfReaderOpts(device, queue, assets, skinnedAssets, arrays, r, LoadGltfOptions{Label: label})
}

// LoadGltfReaderOpts parses a glTF / glb document from r, uploads
// every primitive to assets, decodes every texture into cache,
// captures animation data, and returns a LoadedScene that mirrors
// the glTF node hierarchy. Caller can then [SpawnLoadedScene] to
// materialize entities.
func LoadGltfReaderOpts(device *wgpu.Device, queue *wgpu.Queue, assets *MeshAssets, skinnedAssets *SkinnedMeshAssets, arrays *MaterialTextureArrays, r io.Reader, opts LoadGltfOptions) (*LoadedScene, error) {
	label := opts.Label
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("gltf %q: read: %w", label, err)
	}
	var dec *gltf.Decoder
	if opts.BaseDir != "" {
		dec = gltf.NewDecoderFS(bytes.NewReader(data), os.DirFS(opts.BaseDir))
	} else {
		dec = gltf.NewDecoder(bytes.NewReader(data))
	}
	doc := new(gltf.Document)
	if err := dec.Decode(doc); err != nil {
		return nil, fmt.Errorf("gltf %q: decode: %w", label, err)
	}

	scene := &LoadedScene{Label: label}

	textureColorSpace := classifyTextures(doc)
	textureLayers, err := uploadTextures(queue, arrays, label, doc, textureColorSpace, opts.BaseDir)
	if err != nil {
		return nil, err
	}

	materials := make([]Material, len(doc.Materials))
	for i := range doc.Materials {
		materials[i] = buildMaterial(doc.Materials[i], textureLayers)
	}
	scene.Materials = materials

	type primitiveResult struct {
		Handle        MeshHandle
		SkinnedHandle SkinnedMeshHandle
		Skinned       bool
		Material      Material
	}
	meshPrimitives := make([][]primitiveResult, len(doc.Meshes))
	for meshIdx, mesh := range doc.Meshes {
		results := make([]primitiveResult, 0, len(mesh.Primitives))
		for primIdx, prim := range mesh.Primitives {
			name := mesh.Name
			if name == "" {
				name = fmt.Sprintf("%s/mesh_%d", label, meshIdx)
			}
			if len(mesh.Primitives) > 1 {
				name = fmt.Sprintf("%s.%d", name, primIdx)
			}
			material := DefaultMaterial()
			if prim.Material != nil {
				material = materials[*prim.Material]
			}
			if _, hasJoints := prim.Attributes["JOINTS_0"]; hasJoints && skinnedAssets != nil {
				vertices, err := buildSkinnedGltfPrimitive(doc, prim)
				if err != nil {
					return nil, fmt.Errorf("gltf %q: mesh %d primitive %d (skinned): %w", label, meshIdx, primIdx, err)
				}
				handle, err := skinnedAssets.Register(device, name, vertices)
				if err != nil {
					return nil, err
				}
				results = append(results, primitiveResult{SkinnedHandle: handle, Skinned: true, Material: material})
				continue
			}
			vertices, err := buildGltfPrimitive(doc, prim)
			if err != nil {
				return nil, fmt.Errorf("gltf %q: mesh %d primitive %d: %w", label, meshIdx, primIdx, err)
			}
			handle, err := assets.Register(device, name, vertices)
			if err != nil {
				return nil, err
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

	scene.SkinSpecs, err = readSkins(doc)
	if err != nil {
		return nil, fmt.Errorf("gltf %q: %w", label, err)
	}

	scene.Lights = readPunctualLights(doc)
	scene.Cameras = readCameras(doc)

	scene.Nodes = make([]SceneNode, len(doc.Nodes))
	for i, node := range doc.Nodes {
		translation, rotation, scale := nodeTRS(node)
		out := SceneNode{
			Name:        node.Name,
			Translation: translation,
			Rotation:    rotation,
			Scale:       scale,
			Children:    childIndicesOf(node),
			SkinIndex:   -1,
			LightIndex:  -1,
			CameraIndex: -1,
		}
		if node.Skin != nil {
			out.SkinIndex = int(*node.Skin)
		}
		if node.Camera != nil {
			out.CameraIndex = int(*node.Camera)
		}
		out.LightIndex = readNodeLightIndex(node.Extensions)
		if node.Mesh != nil {
			prims := meshPrimitives[*node.Mesh]
			// Pre-classify primitives so multi-primitive nodes
			// route static and skinned variants down their
			// respective slots.
			staticPrims := make([]SceneNodePrimitive, 0, len(prims))
			skinnedPrims := make([]SceneNodeSkinnedPrimitive, 0, len(prims))
			for _, p := range prims {
				if p.Skinned {
					skinnedPrims = append(skinnedPrims, SceneNodeSkinnedPrimitive{Mesh: p.SkinnedHandle, Material: p.Material})
				} else {
					staticPrims = append(staticPrims, SceneNodePrimitive{Mesh: p.Handle, Material: p.Material})
				}
			}
			switch {
			case len(staticPrims) == 1 && len(skinnedPrims) == 0:
				out.HasMesh = true
				out.Mesh = staticPrims[0].Mesh
				out.Material = staticPrims[0].Material
			case len(skinnedPrims) == 1 && len(staticPrims) == 0:
				out.HasSkinnedMesh = true
				out.SkinnedMesh = skinnedPrims[0].Mesh
				out.Material = skinnedPrims[0].Material
			default:
				if len(staticPrims) > 0 {
					out.ChildPrimitives = staticPrims
				}
				if len(skinnedPrims) > 0 {
					out.ChildSkinnedPrims = skinnedPrims
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
// SpawnLoadedScene needs a *wgpu.Device to allocate per-skin
// joint matrix storage buffers. Static-only scenes still work
// when device is nil (the function skips skin materialization).
func SpawnLoadedScene(world *ecs.World, scene *LoadedScene, device *wgpu.Device) []ecs.Entity {
	entities := make([]ecs.Entity, len(scene.Nodes))
	localMask := ecs.MustMaskOf[transform.LocalTransform](world)
	globalMask := ecs.MustMaskOf[transform.GlobalTransform](world)
	dirtyMask := ecs.MustMaskOf[transform.LocalTransformDirty](world)
	parentMask := ecs.MustMaskOf[transform.Parent](world)
	groupRootMask := ecs.MustMaskOf[transform.GroupRoot](world)
	meshMask := ecs.MustMaskOf[RenderMesh](world)
	skinnedMeshMask := ecs.MustMaskOf[SkinnedMesh](world)
	materialMask := ecs.MustMaskOf[Material](world)

	transformMask := localMask | globalMask | dirtyMask
	for i := range scene.Nodes {
		entities[i] = world.Spawn(transformMask)
		ecs.Set(world, entities[i], transform.IdentityGlobalTransform())
	}
	// Synthesize an asset-wrapper entity above every glTF top-level
	// node so the whole loaded asset selects as a single cluster
	// even when the glTF has multiple sibling root nodes (Flight
	// Helmet, BoomBox, ABeautifulGame all do). The wrapper carries
	// GroupRoot + an identity transform; the scene's original roots
	// reparent to it.
	assetRoot := world.Spawn(transformMask | parentMask | groupRootMask)
	ecs.Set(world, assetRoot, transform.IdentityLocalTransform())
	ecs.Set(world, assetRoot, transform.IdentityGlobalTransform())
	ecs.Set(world, assetRoot, transform.Parent{IsRoot: true})
	ecs.Set(world, assetRoot, transform.GroupRoot{})
	for _, rootIdx := range scene.Roots {
		if rootIdx >= 0 && rootIdx < len(entities) {
			world.AddComponents(entities[rootIdx], parentMask)
			ecs.Set(world, entities[rootIdx], transform.Parent{Entity: assetRoot})
		}
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
		if node.HasSkinnedMesh {
			world.AddComponents(entity, skinnedMeshMask|materialMask)
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
		for _, prim := range node.ChildSkinnedPrims {
			child := world.Spawn(transformMask | parentMask | skinnedMeshMask | materialMask)
			ecs.Set(world, child, transform.IdentityLocalTransform())
			ecs.Set(world, child, transform.IdentityGlobalTransform())
			ecs.Set(world, child, transform.Parent{Entity: entity})
			ecs.Set(world, child, prim.Material)
			scene.skinnedPrimSpawned = append(scene.skinnedPrimSpawned, skinnedSpawnRecord{
				NodeIdx:     i,
				SkinnedMesh: prim.Mesh,
				Entity:      child,
			})
		}
	}

	// Materialize each glTF skin into an asset.Skin, then write the
	// SkinnedMesh component on every entity that referenced that
	// skin (both single-primitive nodes and multi-primitive
	// children). joint matrix uploads happen each frame via
	// PrepareSkinMatrices.
	skinResources := make([]*Skin, len(scene.SkinSpecs))
	if device != nil {
		for skinIdx, spec := range scene.SkinSpecs {
			if len(spec.JointNodeIndices) == 0 {
				continue
			}
			skin, err := NewSkin(device, len(spec.JointNodeIndices))
			if err != nil {
				continue
			}
			for jointIdx, nodeIdx := range spec.JointNodeIndices {
				if nodeIdx >= 0 && nodeIdx < len(entities) {
					skin.Joints[jointIdx] = entities[nodeIdx]
				}
			}
			if len(spec.InverseBindMatrices) == len(skin.InverseBindMatrices) {
				copy(skin.InverseBindMatrices, spec.InverseBindMatrices)
			}
			skinResources[skinIdx] = skin
		}
	}
	for i, node := range scene.Nodes {
		if node.SkinIndex < 0 || node.SkinIndex >= len(skinResources) {
			continue
		}
		skin := skinResources[node.SkinIndex]
		if skin == nil {
			continue
		}
		if node.HasSkinnedMesh {
			ecs.Set(world, entities[i], SkinnedMesh{Mesh: node.SkinnedMesh, Skin: skin})
		}
	}
	for _, record := range scene.skinnedPrimSpawned {
		node := scene.Nodes[record.NodeIdx]
		if node.SkinIndex < 0 || node.SkinIndex >= len(skinResources) {
			continue
		}
		skin := skinResources[node.SkinIndex]
		if skin == nil {
			continue
		}
		ecs.Set(world, record.Entity, SkinnedMesh{Mesh: record.SkinnedMesh, Skin: skin})
	}

	// The loader sets transform.Parent directly via ecs.Set above,
	// which bypasses the UpdateParent helper. The transform
	// children cache is keyed on Parent values and would stay
	// stale (no children mapped for any of the new entities)
	// without an explicit invalidate.
	transform.InvalidateChildrenCache(world)

	// Auto-attach an AnimationPlayer to the first scene root when
	// the source glTF carries clips. The player owns a copy of the
	// clips plus the per-load node-index-to-entity map so per-frame
	// channel sampling can resolve target nodes in O(1).
	if len(scene.Animations) > 0 && len(scene.Roots) > 0 {
		nodeIndexToEntity := make(map[int]ecs.Entity, len(entities))
		for i, e := range entities {
			nodeIndexToEntity[i] = e
		}
		root := entities[scene.Roots[0]]
		playerMask := ecs.MustMaskOf[AnimationPlayer](world)
		world.AddComponents(root, playerMask)
		ecs.Set(world, root, NewAnimationPlayer(scene.Animations, nodeIndexToEntity))
	}
	return entities
}

// identityNodeMatrix is the column-major identity matrix qmuntal/gltf
// fills in for unspecified node.Matrix fields. Used to detect when a
// node's transform actually lives in the Matrix slot vs. when it
// lives in the TRS slots and Matrix is just default-populated.
var identityNodeMatrix = [16]float64{
	1, 0, 0, 0,
	0, 1, 0, 0,
	0, 0, 1, 0,
	0, 0, 0, 1,
}

// nodeTRS returns the node's TRS as three components. glTF lets a
// node specify its transform two ways: a 4x4 Matrix field, or the
// separate Translation / Rotation / Scale fields. The qmuntal/gltf
// decoder always populates both — Matrix is set to identity when
// the JSON only specifies TRS, and TRS is set to defaults when the
// JSON only specifies Matrix. We check for a non-identity Matrix
// first; if it's identity, the real transform lives in the TRS
// slots.
func nodeTRS(node *gltf.Node) (translation [3]float32, rotation [4]float32, scale [3]float32) {
	if node.Matrix != identityNodeMatrix && node.Matrix != [16]float64{} {
		return decomposeMatrix(node.Matrix)
	}
	translation = [3]float32{0, 0, 0}
	if node.Translation != [3]float64{} {
		translation = [3]float32{float32(node.Translation[0]), float32(node.Translation[1]), float32(node.Translation[2])}
	}
	rotation = [4]float32{0, 0, 0, 1}
	if node.Rotation != [4]float64{} {
		rotation = [4]float32{float32(node.Rotation[0]), float32(node.Rotation[1]), float32(node.Rotation[2]), float32(node.Rotation[3])}
	}
	scale = [3]float32{1, 1, 1}
	if node.Scale != [3]float64{} {
		scale = [3]float32{float32(node.Scale[0]), float32(node.Scale[1]), float32(node.Scale[2])}
	}
	return
}

// decomposeMatrix splits a column-major 4x4 affine TRS matrix into
// its translation, rotation, and scale components. Translation
// lives in the fourth column. Scale magnitudes are the lengths of
// the first three columns; the columns are then renormalized into
// a pure rotation matrix which is converted to a quaternion.
// Assumes no shear (glTF matrices are required to be decomposable).
func decomposeMatrix(matrix [16]float64) (translation [3]float32, rotation [4]float32, scale [3]float32) {
	translation = [3]float32{float32(matrix[12]), float32(matrix[13]), float32(matrix[14])}

	col0 := mgl32.Vec3{float32(matrix[0]), float32(matrix[1]), float32(matrix[2])}
	col1 := mgl32.Vec3{float32(matrix[4]), float32(matrix[5]), float32(matrix[6])}
	col2 := mgl32.Vec3{float32(matrix[8]), float32(matrix[9]), float32(matrix[10])}
	scale = [3]float32{col0.Len(), col1.Len(), col2.Len()}
	if scale[0] == 0 {
		scale[0] = 1
	}
	if scale[1] == 0 {
		scale[1] = 1
	}
	if scale[2] == 0 {
		scale[2] = 1
	}

	// Detect mirror (negative determinant) and flip one axis so
	// the residual rotation has a positive determinant. Without
	// this fix Mat4ToQuat on a left-handed basis returns garbage.
	det := col0.Cross(col1).Dot(col2)
	if det < 0 {
		col0 = col0.Mul(-1)
		scale[0] = -scale[0]
	}

	rot := mgl32.Mat4{
		col0[0] / mgl32.Abs(scale[0]), col0[1] / mgl32.Abs(scale[0]), col0[2] / mgl32.Abs(scale[0]), 0,
		col1[0] / scale[1], col1[1] / scale[1], col1[2] / scale[1], 0,
		col2[0] / scale[2], col2[1] / scale[2], col2[2] / scale[2], 0,
		0, 0, 0, 1,
	}
	quat := mgl32.Mat4ToQuat(rot)
	rotation = [4]float32{quat.V[0], quat.V[1], quat.V[2], quat.W}
	return
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

// readEmissiveStrengthExt parses the KHR_materials_emissive_strength
// payload off a glTF material's extensions map. Returns 1.0 when
// the extension is absent or malformed.
func readEmissiveStrengthExt(ext gltf.Extensions) float32 {
	if ext == nil {
		return 1.0
	}
	raw, ok := ext["KHR_materials_emissive_strength"]
	if !ok {
		return 1.0
	}
	type payload struct {
		EmissiveStrength float64 `json:"emissiveStrength"`
	}
	body, err := jsonRawFromExt(raw)
	if err != nil {
		return 1.0
	}
	var decoded payload
	if err := json.Unmarshal(body, &decoded); err != nil {
		return 1.0
	}
	if decoded.EmissiveStrength <= 0 {
		return 1.0
	}
	return float32(decoded.EmissiveStrength)
}

// readPunctualLights extracts the document-level
// KHR_lights_punctual.lights array. Returns nil when the
// extension is absent.
func readPunctualLights(doc *gltf.Document) []LoadedLight {
	if doc == nil || doc.Extensions == nil {
		return nil
	}
	raw, ok := doc.Extensions["KHR_lights_punctual"]
	if !ok {
		return nil
	}
	type spotPayload struct {
		InnerConeAngle float64 `json:"innerConeAngle"`
		OuterConeAngle float64 `json:"outerConeAngle"`
	}
	type lightPayload struct {
		Name      string       `json:"name"`
		Type      string       `json:"type"`
		Color     [3]float64   `json:"color"`
		Intensity *float64     `json:"intensity"`
		Range     *float64     `json:"range"`
		Spot      *spotPayload `json:"spot"`
	}
	type payload struct {
		Lights []lightPayload `json:"lights"`
	}
	body, err := jsonRawFromExt(raw)
	if err != nil {
		return nil
	}
	var decoded payload
	if err := json.Unmarshal(body, &decoded); err != nil {
		return nil
	}
	out := make([]LoadedLight, 0, len(decoded.Lights))
	for _, l := range decoded.Lights {
		color := [3]float32{1, 1, 1}
		if l.Color != [3]float64{0, 0, 0} {
			color = [3]float32{float32(l.Color[0]), float32(l.Color[1]), float32(l.Color[2])}
		}
		intensity := float32(1)
		if l.Intensity != nil {
			intensity = float32(*l.Intensity)
		}
		var rng float32
		if l.Range != nil {
			rng = float32(*l.Range)
		}
		var inner, outer float32
		if l.Spot != nil {
			inner = float32(l.Spot.InnerConeAngle)
			outer = float32(l.Spot.OuterConeAngle)
		} else {
			outer = float32(math.Pi / 4)
		}
		var ty LoadedLightType
		switch l.Type {
		case "point":
			ty = LoadedLightPoint
		case "spot":
			ty = LoadedLightSpot
		default:
			ty = LoadedLightDirectional
		}
		out = append(out, LoadedLight{
			Name:           l.Name,
			Type:           ty,
			Color:          color,
			Intensity:      intensity,
			Range:          rng,
			InnerConeAngle: inner,
			OuterConeAngle: outer,
		})
	}
	return out
}

// readCameras extracts the document's perspective camera array.
// Orthographic cameras are skipped (the engine's view camera is
// perspective-only today).
func readCameras(doc *gltf.Document) []LoadedCamera {
	if doc == nil || len(doc.Cameras) == 0 {
		return nil
	}
	out := make([]LoadedCamera, 0, len(doc.Cameras))
	for _, cam := range doc.Cameras {
		if cam.Perspective == nil {
			out = append(out, LoadedCamera{Name: cam.Name, FovYRadians: float32(math.Pi / 3), Near: 0.1, Far: 1000})
			continue
		}
		far := float32(1000)
		if cam.Perspective.Zfar != nil {
			far = float32(*cam.Perspective.Zfar)
		}
		out = append(out, LoadedCamera{
			Name:        cam.Name,
			FovYRadians: float32(cam.Perspective.Yfov),
			Near:        float32(cam.Perspective.Znear),
			Far:         far,
		})
	}
	return out
}

// readNodeLightIndex peels the node-level KHR_lights_punctual
// payload (which carries a single "light" index into the
// document's lights array). Returns -1 when absent.
func readNodeLightIndex(ext gltf.Extensions) int {
	if ext == nil {
		return -1
	}
	raw, ok := ext["KHR_lights_punctual"]
	if !ok {
		return -1
	}
	type payload struct {
		Light int `json:"light"`
	}
	body, err := jsonRawFromExt(raw)
	if err != nil {
		return -1
	}
	var decoded payload
	if err := json.Unmarshal(body, &decoded); err != nil {
		return -1
	}
	return decoded.Light
}

// jsonRawFromExt normalizes the three shapes an extension value
// can take with qmuntal/gltf (raw bytes, json.RawMessage, or a
// pre-decoded map) into a JSON byte slice the caller can
// unmarshal into a typed payload.
func jsonRawFromExt(raw any) ([]byte, error) {
	switch v := raw.(type) {
	case json.RawMessage:
		return v, nil
	case []byte:
		return v, nil
	case string:
		return []byte(v), nil
	default:
		return json.Marshal(v)
	}
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

// uploadTextures decodes every glTF texture, resizes to the array
// tile size, uploads it into the appropriate (sRGB or linear)
// material texture array, and returns the packed layer + wrap-mode
// value the mesh shader consumes. Index in the result matches the
// glTF texture index.
func uploadTextures(queue *wgpu.Queue, arrays *MaterialTextureArrays, label string, doc *gltf.Document, spaces []TextureColorSpace, baseDir string) ([]uint32, error) {
	out := make([]uint32, len(doc.Textures))
	for i := range out {
		out[i] = NoTextureLayer
	}
	for i, tex := range doc.Textures {
		if tex.Source == nil {
			continue
		}
		img := doc.Images[*tex.Source]
		pixels, w, h, err := decodeGltfImage(doc, img, baseDir)
		if err != nil {
			return nil, fmt.Errorf("gltf %q: image %d: %w", label, i, err)
		}
		resized := resizeRGBA(pixels, w, h, arrays.LayerSize, arrays.LayerSize)
		name := fmt.Sprintf("%s/tex_%d", label, i)
		if img.Name != "" {
			name = label + "/" + img.Name
		}
		layer, err := arrays.Upload(queue, name, spaces[i], resized)
		if err != nil {
			return nil, err
		}
		wrapU, wrapV := samplerWrapModes(doc, tex)
		out[i] = PackLayer(layer, wrapU, wrapV)
	}
	return out, nil
}

// samplerWrapModes returns the per-axis wrap codes for a glTF
// texture's sampler. Defaults to Repeat / Repeat when the texture
// doesn't reference a sampler.
func samplerWrapModes(doc *gltf.Document, tex *gltf.Texture) (WrapMode, WrapMode) {
	if tex.Sampler == nil {
		return WrapRepeat, WrapRepeat
	}
	s := doc.Samplers[*tex.Sampler]
	return mapWrap(s.WrapS), mapWrap(s.WrapT)
}

func mapWrap(mode gltf.WrappingMode) WrapMode {
	switch mode {
	case gltf.WrapClampToEdge:
		return WrapClampToEdge
	case gltf.WrapMirroredRepeat:
		return WrapMirroredRepeat
	default:
		return WrapRepeat
	}
}

// resizeRGBA bilinearly rescales tightly-packed RGBA8 pixels to
// the target size. When the source already matches it returns the
// input slice unchanged. The mesh texture arrays require all
// layers to be the same size, so any non-matching glTF image gets
// rescaled at load time.
func resizeRGBA(src []byte, srcW, srcH, dstW, dstH uint32) []byte {
	if srcW == dstW && srcH == dstH {
		return src
	}
	source := &image.RGBA{
		Pix:    src,
		Stride: int(srcW) * 4,
		Rect:   image.Rect(0, 0, int(srcW), int(srcH)),
	}
	dst := image.NewRGBA(image.Rect(0, 0, int(dstW), int(dstH)))
	xdraw.BiLinear.Scale(dst, dst.Rect, source, source.Rect, xdraw.Src, nil)
	return dst.Pix
}

func buildMaterial(src *gltf.Material, textureLayers []uint32) Material {
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
			out.BaseColorLayer = textureLayers[pbr.BaseColorTexture.Index]
		}
		if pbr.MetallicRoughnessTexture != nil {
			out.MetallicRoughnessLayer = textureLayers[pbr.MetallicRoughnessTexture.Index]
		}
	}
	if src.NormalTexture != nil {
		if src.NormalTexture.Index != nil {
			out.NormalLayer = textureLayers[*src.NormalTexture.Index]
		}
		out.NormalScale = float32(src.NormalTexture.ScaleOrDefault())
	}
	if src.OcclusionTexture != nil {
		if src.OcclusionTexture.Index != nil {
			out.OcclusionLayer = textureLayers[*src.OcclusionTexture.Index]
		}
		out.OcclusionStrength = float32(src.OcclusionTexture.StrengthOrDefault())
	}
	if src.EmissiveTexture != nil {
		out.EmissiveLayer = textureLayers[src.EmissiveTexture.Index]
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
	out.EmissiveStrength = readEmissiveStrengthExt(src.Extensions)
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

// buildSkinnedGltfPrimitive mirrors buildGltfPrimitive but also
// reads JOINTS_0 + WEIGHTS_0 and packs them into the per-vertex
// SkinnedMeshVertex layout the skinned mesh pass expects.
func buildSkinnedGltfPrimitive(doc *gltf.Document, prim *gltf.Primitive) ([]SkinnedMeshVertex, error) {
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

	jointAccessor, hasJoints := prim.Attributes["JOINTS_0"]
	if !hasJoints {
		return nil, fmt.Errorf("primitive missing JOINTS_0 attribute")
	}
	jointsRaw, err := modeler.ReadJoints(doc, doc.Accessors[jointAccessor], nil)
	if err != nil {
		return nil, fmt.Errorf("read joints_0: %w", err)
	}

	weightAccessor, hasWeights := prim.Attributes["WEIGHTS_0"]
	if !hasWeights {
		return nil, fmt.Errorf("primitive missing WEIGHTS_0 attribute")
	}
	weightsRaw, err := modeler.ReadWeights(doc, doc.Accessors[weightAccessor], nil)
	if err != nil {
		return nil, fmt.Errorf("read weights_0: %w", err)
	}

	indices, err := readIndices(doc, prim)
	if err != nil {
		return nil, err
	}

	expanded := make([]SkinnedMeshVertex, 0, len(indices))
	for _, srcIdx := range indices {
		v := SkinnedMeshVertex{
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
		if int(srcIdx) < len(jointsRaw) {
			j := jointsRaw[srcIdx]
			v.JointIndices = [4]uint32{uint32(j[0]), uint32(j[1]), uint32(j[2]), uint32(j[3])}
		}
		if int(srcIdx) < len(weightsRaw) {
			w := weightsRaw[srcIdx]
			sum := w[0] + w[1] + w[2] + w[3]
			if sum > 0 {
				inv := 1.0 / sum
				v.JointWeights = [4]float32{w[0] * inv, w[1] * inv, w[2] * inv, w[3] * inv}
			} else {
				v.JointWeights = [4]float32{1, 0, 0, 0}
			}
		} else {
			v.JointWeights = [4]float32{1, 0, 0, 0}
		}
		expanded = append(expanded, v)
	}
	return expanded, nil
}

// readSkins extracts every glTF skin's joint node indices and
// inverse-bind matrices. Uses the dedicated
// modeler.ReadInverseBindMatrices accessor so column-major
// glTF storage decodes into [col][row]float32 as expected, then
// reshuffles into the column-major flat layout mgl32.Mat4 uses.
//
// Pads the IBM slice up to len(joints) with identity matrices to
// match the reference engine's behavior when a skin's IBM
// accessor is short (or absent).
func readSkins(doc *gltf.Document) ([]SkinSpec, error) {
	if len(doc.Skins) == 0 {
		return nil, nil
	}
	specs := make([]SkinSpec, len(doc.Skins))
	for skinIdx, skin := range doc.Skins {
		spec := SkinSpec{
			JointNodeIndices: make([]int, len(skin.Joints)),
		}
		for j, jointNodeIdx := range skin.Joints {
			spec.JointNodeIndices[j] = int(jointNodeIdx)
		}
		spec.InverseBindMatrices = make([]mgl32.Mat4, len(spec.JointNodeIndices))
		for k := range spec.InverseBindMatrices {
			spec.InverseBindMatrices[k] = mgl32.Ident4()
		}
		if skin.InverseBindMatrices != nil {
			accessor := doc.Accessors[*skin.InverseBindMatrices]
			mats, err := modeler.ReadInverseBindMatrices(doc, accessor, nil)
			if err != nil {
				return nil, fmt.Errorf("skin %d: read inverse bind matrices: %w", skinIdx, err)
			}
			count := len(mats)
			if count > len(spec.InverseBindMatrices) {
				count = len(spec.InverseBindMatrices)
			}
			for k := 0; k < count; k++ {
				m := mats[k]
				// modeler returns matrices as [row][col]float32:
				// m[r][c] is the value at row r, column c. The
				// modeler's own TestReadInverseBindMatrices verifies
				// this -- a buffer storing column-major (col0, col1,
				// col2, col3) of (1,0,0,0)(2,0,0,0)(3,0,0,0)(4,0,0,0)
				// decodes to m[0]={1,2,3,4}, which is row 0.
				//
				// mgl32.Mat4 stores column-major flat (col 0 rows
				// 0..3 at indices 0..3, col 1 at 4..7, etc.), so
				// rearrange m[row][col] -> column-major flat.
				spec.InverseBindMatrices[k] = mgl32.Mat4{
					m[0][0], m[1][0], m[2][0], m[3][0],
					m[0][1], m[1][1], m[2][1], m[3][1],
					m[0][2], m[1][2], m[2][2], m[3][2],
					m[0][3], m[1][3], m[2][3], m[3][3],
				}
			}
		}
		specs[skinIdx] = spec
	}
	return specs, nil
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
	// .RGBA() returns premultiplied values, which zero out RGB
	// on transparent pixels. OPAQUE-mode materials with alpha-
	// textured base color need the unmodified RGB so glTF's
	// "alpha is ignored" semantics show the texture's color.
	for y := 0; y < int(h); y++ {
		for x := 0; x < int(w); x++ {
			c := color.NRGBAModel.Convert(decoded.At(bounds.Min.X+x, bounds.Min.Y+y)).(color.NRGBA)
			i := (y*int(w) + x) * 4
			pixels[i+0] = c.R
			pixels[i+1] = c.G
			pixels[i+2] = c.B
			pixels[i+3] = c.A
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
