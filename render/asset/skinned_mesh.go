package asset

import (
	"fmt"
	"unsafe"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/go-gl/mathgl/mgl32"

	"github.com/matthewjberger/indigo/ecs"
)

// MaxJointsPerSkin caps the joint count any one skinned mesh can
// carry. 128 fits typical character rigs without inflating the
// per-frame upload cost; rigs that exceed it need to be split.
const MaxJointsPerSkin = 128

// SkinnedMeshVertex extends MeshVertex with the two glTF skinning
// attributes (JOINTS_0 + WEIGHTS_0). Stored as a parallel layout
// rather than a wider MeshVertex so the static mesh path keeps its
// smaller stride for non-skinned geometry.
type SkinnedMeshVertex struct {
	Position     [4]float32
	Normal       [4]float32
	Tangent      [4]float32
	UV           [4]float32
	Color        [4]float32
	JointIndices [4]uint32
	JointWeights [4]float32
}

// SkinnedMeshHandle is an opaque index into the skinned mesh
// registry. Distinct from [MeshHandle] so the static mesh path
// can't accidentally pull a skinned buffer.
type SkinnedMeshHandle uint32

// Skin owns the per-frame joint binding state for one skinned
// mesh instance. Joints are engine entities whose GlobalTransform
// matrices feed the bone-transforms input each frame; a compute
// pass multiplies them by InverseBindMatrices to produce the
// JointMatrixBuffer the vertex shader reads:
//
//	joint_matrix[i] = global(joints[i]) * inverse_bind[i]
//
// The CPU pushes the per-frame bone transforms; the GPU does the
// mat4 multiply.
type Skin struct {
	Joints               []ecs.Entity
	InverseBindMatrices  []mgl32.Mat4
	BoneScratch          []mgl32.Mat4
	BoneTransformsBuffer *wgpu.Buffer
	InverseBindBuffer    *wgpu.Buffer
	JointMatrixBuffer    *wgpu.Buffer
	InverseBindUploaded  bool
}

// NewSkin allocates the per-skin storage buffers sized for the
// given joint count. Caller fills Joints + InverseBindMatrices,
// then the first frame's compute dispatch uploads + computes
// joint matrices.
func NewSkin(device *wgpu.Device, jointCount int) (*Skin, error) {
	if jointCount <= 0 || jointCount > MaxJointsPerSkin {
		return nil, fmt.Errorf("skin: joint count %d outside [1, %d]", jointCount, MaxJointsPerSkin)
	}
	size := uint64(jointCount) * uint64(unsafe.Sizeof(mgl32.Mat4{}))
	bones, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "skin bone transforms",
		Size:  size,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("skin: bones buffer: %w", err)
	}
	ibms, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "skin inverse bind matrices",
		Size:  size,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		bones.Release()
		return nil, fmt.Errorf("skin: ibm buffer: %w", err)
	}
	out, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "skin joint matrices",
		Size:  size,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		bones.Release()
		ibms.Release()
		return nil, fmt.Errorf("skin: joint buffer: %w", err)
	}
	return &Skin{
		Joints:               make([]ecs.Entity, jointCount),
		InverseBindMatrices:  make([]mgl32.Mat4, jointCount),
		BoneScratch:          make([]mgl32.Mat4, jointCount),
		BoneTransformsBuffer: bones,
		InverseBindBuffer:    ibms,
		JointMatrixBuffer:    out,
	}, nil
}

// Release frees every per-skin storage buffer.
func (s *Skin) Release() {
	if s.JointMatrixBuffer != nil {
		s.JointMatrixBuffer.Release()
		s.JointMatrixBuffer = nil
	}
	if s.BoneTransformsBuffer != nil {
		s.BoneTransformsBuffer.Release()
		s.BoneTransformsBuffer = nil
	}
	if s.InverseBindBuffer != nil {
		s.InverseBindBuffer.Release()
		s.InverseBindBuffer = nil
	}
}

type skinnedMeshEntry struct {
	Name        string
	Vertices    *wgpu.Buffer
	VertexCount uint32
	Bounds      BoundingVolume
	CpuVertices []SkinnedMeshVertex
}

// SkinnedMeshAssets is the per-renderer registry of skinned mesh
// GPU buffers. Parallel to [MeshAssets].
type SkinnedMeshAssets struct {
	entries []skinnedMeshEntry
}

// NewSkinnedMeshAssets returns an empty registry.
func NewSkinnedMeshAssets() *SkinnedMeshAssets {
	return &SkinnedMeshAssets{}
}

// Register uploads the vertex slice to a GPU buffer and returns
// the handle assigned to it.
func (assets *SkinnedMeshAssets) Register(device *wgpu.Device, name string, vertices []SkinnedMeshVertex) (SkinnedMeshHandle, error) {
	if len(vertices) == 0 {
		return 0, fmt.Errorf("skinned mesh %q: empty vertex slice", name)
	}
	buffer, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "skinned mesh vertex buffer: " + name,
		Size:  uint64(len(vertices)) * uint64(unsafe.Sizeof(SkinnedMeshVertex{})),
		Usage: wgpu.BufferUsageVertex | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return 0, fmt.Errorf("skinned mesh %q: buffer: %w", name, err)
	}
	device.GetQueue().WriteBuffer(buffer, 0, unsafe.Slice((*byte)(unsafe.Pointer(&vertices[0])), len(vertices)*int(unsafe.Sizeof(SkinnedMeshVertex{}))))
	cpu := make([]SkinnedMeshVertex, len(vertices))
	copy(cpu, vertices)
	bounds := computeSkinnedBounds(cpu)
	handle := SkinnedMeshHandle(len(assets.entries))
	assets.entries = append(assets.entries, skinnedMeshEntry{
		Name:        name,
		Vertices:    buffer,
		VertexCount: uint32(len(vertices)),
		Bounds:      bounds,
		CpuVertices: cpu,
	})
	return handle, nil
}

// Lookup returns the entry for handle. Returns false when the
// handle is out of range.
func (assets *SkinnedMeshAssets) Lookup(handle SkinnedMeshHandle) (*skinnedMeshEntry, bool) {
	if int(handle) >= len(assets.entries) {
		return nil, false
	}
	return &assets.entries[handle], true
}

// Release frees every GPU buffer the registry owns.
func (assets *SkinnedMeshAssets) Release() {
	for index := range assets.entries {
		if assets.entries[index].Vertices != nil {
			assets.entries[index].Vertices.Release()
			assets.entries[index].Vertices = nil
		}
	}
	assets.entries = nil
}

// SkinnedMeshAssetsResource exposes the registry as an ECS
// resource so passes can look up vertex buffers by handle.
type SkinnedMeshAssetsResource struct {
	Assets *SkinnedMeshAssets
}

// SkinnedMesh tags an entity as a rigged mesh. Mesh is the handle
// into [SkinnedMeshAssets]; Skin owns the per-frame joint matrix
// array the vertex shader samples. Distinct from [RenderMesh] so
// the static and skinned passes iterate disjoint sets.
type SkinnedMesh struct {
	Mesh SkinnedMeshHandle
	Skin *Skin
}

func computeSkinnedBounds(vertices []SkinnedMeshVertex) BoundingVolume {
	if len(vertices) == 0 {
		return BoundingVolume{}
	}
	minPt := [3]float32{vertices[0].Position[0], vertices[0].Position[1], vertices[0].Position[2]}
	maxPt := minPt
	for index := 1; index < len(vertices); index++ {
		for component := 0; component < 3; component++ {
			value := vertices[index].Position[component]
			if value < minPt[component] {
				minPt[component] = value
			}
			if value > maxPt[component] {
				maxPt[component] = value
			}
		}
	}
	return BoundingVolume{Min: minPt, Max: maxPt}
}
