//go:build js

package pass

import "github.com/cogentcore/webgpu/wgpu"

func writeBuffer(device *wgpu.Device, _ *wgpu.Queue, encoder *wgpu.CommandEncoder, buffer *wgpu.Buffer, offset uint64, data []byte) {
	if len(data) == 0 {
		return
	}
	staging, err := device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    "write_buffer staging",
		Contents: data,
		Usage:    wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return
	}
	encoder.CopyBufferToBuffer(staging, 0, buffer, offset, uint64(len(data)))
	staging.Release()
}
