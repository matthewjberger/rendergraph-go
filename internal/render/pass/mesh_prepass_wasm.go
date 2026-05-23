//go:build js

package pass

import "github.com/cogentcore/webgpu/wgpu"

const meshRunPrepass = false

const meshMainDepthWrite = true

var meshMainDepthCompare = wgpu.CompareFunctionLess
