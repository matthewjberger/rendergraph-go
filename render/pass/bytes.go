package pass

import "unsafe"

// bytesOf returns the raw byte view of v's pointed-at memory without
// allocating. Used at GPU upload boundaries where the value's Go
// layout matches the WGSL buffer layout exactly.
func bytesOf[T any](v *T) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(v)), unsafe.Sizeof(*v))
}

// bytesOfN returns size bytes starting at v. Used when uploading a
// prefix of a Go slice's backing array without copying.
func bytesOfN[T any](v *T, size uint64) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(v)), size)
}
