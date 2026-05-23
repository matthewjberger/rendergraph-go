//go:build js

package pass

import "github.com/cogentcore/webgpu/wgpu"

// Wasm skips the depth prepass (it caused browser-only rasterizer
// artifacts) and draws the main mesh pass standalone: DepthWrite on
// with Less, so it owns the depth buffer.
const meshRunPrepass = false

const meshMainDepthWrite = true

var meshMainDepthCompare = wgpu.CompareFunctionLess
