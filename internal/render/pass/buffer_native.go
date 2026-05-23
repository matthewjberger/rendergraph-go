//go:build !js

package pass

import "github.com/cogentcore/webgpu/wgpu"

func writeBuffer(_ *wgpu.Device, queue *wgpu.Queue, _ *wgpu.CommandEncoder, buffer *wgpu.Buffer, offset uint64, data []byte) {
	queue.WriteBuffer(buffer, offset, data)
}
