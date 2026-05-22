//go:build !js

package pass

import "github.com/cogentcore/webgpu/wgpu"

// writeBuffer copies data into buffer at offset using the queue's
// fast path. On native wgpu this is a direct copy with no staging
// step.
func writeBuffer(_ *wgpu.Device, queue *wgpu.Queue, _ *wgpu.CommandEncoder, buffer *wgpu.Buffer, offset uint64, data []byte) {
	queue.WriteBuffer(buffer, offset, data)
}
