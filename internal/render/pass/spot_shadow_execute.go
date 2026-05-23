package pass

import (
	"github.com/cogentcore/webgpu/wgpu"

	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/render/asset"
)

func spotShadowExecute(s any, context *render.PassContext) error {
	state := s.(*spotShadowPassState)
	shadow := state.shadow
	if shadow.ActiveCount == 0 {
		return nil
	}

	meshState, ok := findMeshPassState(context.World)
	if !ok {
		return nil
	}
	assets := ecs.MustResource[asset.MeshAssetsResource](context.World).Assets
	lightMask := ecs.MustMaskOf[render.Light](context.World)

	passEnc := context.Encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		Label: "spot shadow atlas",
		DepthStencilAttachment: &wgpu.RenderPassDepthStencilAttachment{
			View:            shadow.AtlasView,
			DepthLoadOp:     wgpu.LoadOpClear,
			DepthStoreOp:    wgpu.StoreOpStore,
			DepthClearValue: 1.0,
		},
	})
	defer passEnc.Release()
	passEnc.SetPipeline(state.pipeline)

	for index := uint32(0); index < shadow.ActiveCount; index++ {
		passEnc.SetBindGroup(0, state.slotBgs[index], nil)
		for _, handle := range meshState.sortedHandles {
			bucket := meshState.perHandle[handle]
			entry, ok := assets.Lookup(handle)
			if !ok {
				continue
			}
			shadowBg, err := ensureShadowHandleBindGroup(bucket, context.Device, state.handleBgLayout)
			if err != nil {
				passEnc.End()
				return err
			}
			passEnc.SetBindGroup(1, shadowBg, nil)
			passEnc.SetVertexBuffer(0, entry.Vertices, 0, wgpu.WholeSize)
			drawNonLightInstances(passEnc, bucket, entry.VertexCount, lightMask, context.World)
		}
	}
	passEnc.End()
	return nil
}
