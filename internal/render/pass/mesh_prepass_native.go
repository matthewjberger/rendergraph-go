//go:build !js

package pass

import "github.com/cogentcore/webgpu/wgpu"

const meshRunPrepass = true

const meshMainDepthWrite = false

var meshMainDepthCompare = wgpu.CompareFunctionLessEqual
