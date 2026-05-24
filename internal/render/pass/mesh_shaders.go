package pass

import _ "embed"

// The four mesh shaders (static/skinned × forward/OIT) share an identical body
// of structs, bindings, and PBR/shadow/IBL/transmission helpers. Rather than
// duplicate ~1,100 lines four times, that body lives in common.wgsl and each
// shader is composed from it plus a vertex variant (static vs skinned) and a
// fragment variant (forward vs OIT). This mirrors nightshade's naga_oil module
// composition, adapted to Go's lack of a WGSL preprocessor.

//go:embed mesh_parts/common.wgsl
var shaderCommon string

//go:embed mesh_parts/vertex_static.wgsl
var shaderVertexStatic string

//go:embed mesh_parts/vertex_skinned.wgsl
var shaderVertexSkinned string

//go:embed mesh_parts/fragment_forward.wgsl
var shaderFragmentForward string

//go:embed mesh_parts/fragment_oit.wgsl
var shaderFragmentOit string

func composeShader(parts ...string) string {
	out := ""
	for _, part := range parts {
		out += part + "\n"
	}
	return out
}

var (
	meshShader           = composeShader(shaderCommon, shaderVertexStatic, shaderFragmentForward)
	meshOitShader        = composeShader(shaderCommon, shaderVertexStatic, shaderFragmentOit)
	skinnedMeshShader    = composeShader(shaderCommon, shaderVertexSkinned, shaderFragmentForward)
	skinnedMeshOitShader = composeShader(shaderCommon, shaderVertexSkinned, shaderFragmentOit)
)
