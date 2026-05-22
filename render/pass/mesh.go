package pass

import (
	_ "embed"
	"sync/atomic"

	"github.com/cogentcore/webgpu/wgpu"

	"indigo/ecs"
	"indigo/render"
	"indigo/render/asset"
)

//go:embed mesh.wgsl
var meshShader string

// sharedMeshPassState exposes the mesh pass's per-handle bookkeeping
// to sibling passes (shadow_depth today) that draw the same
// geometry without duplicating the per-entity slot tracking.
// Lifetime: stays valid from NewMeshPass through meshRelease.
var sharedMeshPassState atomic.Value

// meshPassState is the long-lived state the mesh pass keeps.
//
//   - viewProjBindGroup (group 0): view * projection uniform.
//   - globalBindGroup   (group 1): clustered-lighting bindings
//     (lights / light_grid / light_indices / cluster_uniforms /
//     view_matrix) + sRGB / linear texture array views + the
//     shared sampler. The cluster compute passes write into the
//     same lights / light_grid / light_indices buffers from
//     [clusterResources], so the read-only views the mesh shader
//     binds here are consistent the moment the compute passes
//     finish.
//   - per-handle bind group (group 2): models / materials /
//     entity_ids storage buffers for that handle's instances.
//   - iblBindGroup (group 3): irradiance cube + prefiltered cube +
//     BRDF LUT + ibl sampler. Built once at pass setup from the
//     [IBL] bundle the engine world owns.
type meshPassState struct {
	pipeline       *wgpu.RenderPipeline
	viewProjLayout *wgpu.BindGroupLayout
	globalBgLayout *wgpu.BindGroupLayout
	handleBgLayout *wgpu.BindGroupLayout
	iblBgLayout    *wgpu.BindGroupLayout

	viewProjBuffer    *wgpu.Buffer
	viewProjBindGroup *wgpu.BindGroup

	globalBindGroup *wgpu.BindGroup
	iblBindGroup    *wgpu.BindGroup

	clusters      *clusterResources
	buildIndirect *buildIndirectPipeline

	perHandle     map[asset.MeshHandle]*handleInstances
	entityHandle  map[ecs.Entity]asset.MeshHandle
	sortedHandles []asset.MeshHandle

	aspectFn func() float32

	// clusterUniformsScratch is reused across frames to avoid
	// reallocating the per-frame ClusterUniforms upload value.
	clusterUniformsScratch ClusterUniforms
	lightScratch           []LightGPU
}

// NewMeshPass builds the engine's instanced PBR mesh pass.
//
// View * projection lives in a small uniform updated every frame;
// per-entity model matrices, materials, and entity IDs live in
// per-handle storage buffers, sparse-updated via [ecs.IterChanged]
// on the respective components. Each entity gets a stable slot in
// its handle's buffers so the GPU side only writes the entries
// that changed this frame. Materials are sampled from the global
// [asset.MaterialTextureArrays] resource, bound once at pass setup.
//
// The constructor orchestrates four sibling files' worth of setup
// code: bind-group layouts + buffers + pipeline live in
// [mesh_layouts.go], per-handle bookkeeping in [mesh_handle.go],
// per-frame work in [mesh_frame.go].
func NewMeshPass(device *wgpu.Device, surfaceFormat wgpu.TextureFormat, aspect func() float32, arrays *asset.MaterialTextureArrays, registry *asset.MaterialRegistry, ibl *IBL, shadow *Shadow, spotShadow *SpotShadow, pointShadow *PointShadow) (*render.Pass, error) {
	state := &meshPassState{
		perHandle:    make(map[asset.MeshHandle]*handleInstances, 4),
		entityHandle: make(map[ecs.Entity]asset.MeshHandle, 64),
		aspectFn:     aspect,
	}

	viewProjLayout, err := createViewProjLayout(device)
	if err != nil {
		return nil, err
	}
	state.viewProjLayout = viewProjLayout

	globalBgLayout, err := createGlobalBgLayout(device)
	if err != nil {
		return nil, err
	}
	state.globalBgLayout = globalBgLayout

	handleBgLayout, err := createHandleBgLayout(device)
	if err != nil {
		return nil, err
	}
	state.handleBgLayout = handleBgLayout

	iblBgLayout, err := createIblBgLayout(device)
	if err != nil {
		return nil, err
	}
	state.iblBgLayout = iblBgLayout

	iblBindGroup, err := createIblBindGroup(device, iblBgLayout, ibl)
	if err != nil {
		return nil, err
	}
	state.iblBindGroup = iblBindGroup

	viewProjBuffer, err := createViewProjBuffer(device)
	if err != nil {
		return nil, err
	}
	state.viewProjBuffer = viewProjBuffer

	viewProjBindGroup, err := createViewProjBindGroup(device, viewProjLayout, viewProjBuffer)
	if err != nil {
		return nil, err
	}
	state.viewProjBindGroup = viewProjBindGroup

	clusters, err := newClusterResources(device)
	if err != nil {
		return nil, err
	}
	state.clusters = clusters

	globalBindGroup, err := createGlobalBindGroup(device, globalBgLayout, clusters, arrays, registry, shadow, shadow.UniformBuffer, spotShadow, pointShadow)
	if err != nil {
		return nil, err
	}
	state.globalBindGroup = globalBindGroup

	pipeline, err := createMeshPipeline(device, surfaceFormat, viewProjLayout, globalBgLayout, handleBgLayout, iblBgLayout)
	if err != nil {
		return nil, err
	}
	state.pipeline = pipeline

	buildIndirect, err := newBuildIndirectPipeline(device)
	if err != nil {
		return nil, err
	}
	state.buildIndirect = buildIndirect

	sharedMeshPassState.Store(state)

	return &render.Pass{
		Name:    "mesh",
		Writes:  []string{"color", "depth", "entity_id", "view_normals"},
		State:   state,
		Prepare: meshPrepare,
		Execute: meshExecute,
		Release: meshRelease,
	}, nil
}
