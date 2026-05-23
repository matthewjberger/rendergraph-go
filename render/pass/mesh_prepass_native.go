//go:build !js

package pass

import "github.com/cogentcore/webgpu/wgpu"

// Native runs a depth prepass: the main mesh pass then renders with
// DepthWrite off and LessEqual so co-planar fragments survive.
const meshRunPrepass = true

const meshMainDepthWrite = false

var meshMainDepthCompare = wgpu.CompareFunctionLessEqual
