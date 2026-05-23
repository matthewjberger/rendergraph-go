//go:build !js

package pass

import (
	"github.com/cogentcore/webgpu/wgpu"

	"github.com/matthewjberger/indigo/render"
)

func dispatchCullPasses(state *meshPassState, context *render.PassContext) {
	if len(state.sortedHandles) == 0 {
		return
	}

	cullPass := context.Encoder.BeginComputePass(&wgpu.ComputePassDescriptor{})
	cullPass.SetBindGroup(0, state.meshCulling.frameBindGroup, nil)
	cullPass.SetPipeline(state.meshCulling.cullPipeline)
	for _, handle := range state.sortedHandles {
		bucket := state.perHandle[handle]
		if bucket.cullBindGroup == nil {
			continue
		}
		count := uint32(len(bucket.slotEntity))
		groups := (count + 63) / 64
		if groups == 0 {
			continue
		}
		cullPass.SetBindGroup(1, bucket.cullBindGroup, nil)
		cullPass.DispatchWorkgroups(groups, 1, 1)
	}
	cullPass.End()
	cullPass.Release()
}
