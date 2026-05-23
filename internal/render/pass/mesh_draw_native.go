//go:build !js

package pass

import "github.com/cogentcore/webgpu/wgpu"

func drawHandle(pass *wgpu.RenderPassEncoder, bucket *handleInstances, _, _ uint32) {
	pass.DrawIndirect(bucket.indirectBuffer, 0)
}
