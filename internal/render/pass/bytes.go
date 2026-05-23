package pass

import "unsafe"

func bytesOf[T any](v *T) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(v)), unsafe.Sizeof(*v))
}

func bytesOfN[T any](v *T, size uint64) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(v)), size)
}
