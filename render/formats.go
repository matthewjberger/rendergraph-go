package render

import "github.com/cogentcore/webgpu/wgpu"

// EntityIdFormat is the texture format used for the mesh pass's
// entity-id MRT output and the picking pass's readback source. The
// format is u32 per pixel; the mesh shader bitcasts its u32 entity
// id into the texture's raw bits and the picking shader bitcasts
// them back out.
const EntityIdFormat = wgpu.TextureFormatR32Uint

// SelectionMaskFormat is the texture format the selection mask is
// rendered into: a single-channel 8-bit Unorm target where 1.0
// marks pixels covered by selected entities.
const SelectionMaskFormat = wgpu.TextureFormatR8Unorm
