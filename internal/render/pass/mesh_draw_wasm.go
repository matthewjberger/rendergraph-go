//go:build js

package pass

import "github.com/cogentcore/webgpu/wgpu"

func drawHandle(pass *wgpu.RenderPassEncoder, bucket *handleInstances, vertexCount, instanceCount uint32) {
	_ = bucket
	pass.Draw(vertexCount, instanceCount, 0, 0)
}
