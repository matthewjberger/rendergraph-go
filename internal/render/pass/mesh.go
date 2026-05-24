package pass

import (
	_ "embed"
	"sync/atomic"

	"github.com/cogentcore/webgpu/wgpu"

	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/render/asset"
)

//go:embed mesh.wgsl
var meshShader string

var sharedMeshPassState atomic.Value

type meshPassState struct {
	pipeline       *wgpu.RenderPipeline
	depthPrepass   *depthPrepass
	viewProjLayout *wgpu.BindGroupLayout
	globalBgLayout *wgpu.BindGroupLayout
	handleBgLayout *wgpu.BindGroupLayout
	iblBgLayout    *wgpu.BindGroupLayout

	viewProjBuffer    *wgpu.Buffer
	viewProjBindGroup *wgpu.BindGroup

	globalBindGroup *wgpu.BindGroup
	iblBindGroup    *wgpu.BindGroup

	clusters    *clusterResources
	meshCulling *meshCullingPipeline

	perHandle     map[asset.MeshHandle]*handleInstances
	entityHandle  map[ecs.Entity]asset.MeshHandle
	sortedHandles []asset.MeshHandle

	aspectFn func() float32

	clusterUniformsScratch ClusterUniforms
	lightScratch           []LightGPU
	identityScratch        []uint32
}

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

	culling, err := newMeshCullingPipeline(device)
	if err != nil {
		return nil, err
	}
	state.meshCulling = culling

	prepass, err := newDepthPrepassPipeline(device, state.viewProjLayout, state.handleBgLayout, registry, arrays)
	if err != nil {
		return nil, err
	}
	state.depthPrepass = prepass

	sharedMeshPassState.Store(state)

	return &render.Pass{
		Name:    "mesh",
		Writes:  []string{"color", "depth", "entity_id", "view_normals"},
		Prepare: func(c *render.PassContext) error { return meshPrepare(state, c) },
		Execute: func(c *render.PassContext) error { return meshExecute(state, c) },
		Release: func() { meshRelease(state) },
	}, nil
}
