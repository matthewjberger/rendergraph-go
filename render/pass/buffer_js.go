//go:build js

package pass

import "github.com/cogentcore/webgpu/wgpu"

// writeBuffer copies data into buffer at offset by routing through a
// per-call CreateBufferInit staging buffer plus a CopyBufferToBuffer
// on the encoder.
//
// Why not queue.WriteBuffer directly? cogentcore/webgpu's WriteBuffer
// goes through jsx.BytesToJS, which constructs a typed-array VIEW
// over wasm linear memory. When the Go runtime grows wasm memory the
// underlying ArrayBuffer detaches and the next WriteBuffer call
// fails. CreateBufferInit uses js.CopyBytesToJS, which copies the
// bytes into a JS-owned ArrayBuffer so it's immune to subsequent
// wasm memory growth.
//
// The staging buffer is released immediately; wgpu keeps a reference
// alive via the recorded copy command until queue.Submit consumes it.
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
