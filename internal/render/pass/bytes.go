package pass

import "unsafe"

func bytesOf[T any](v *T) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(v)), unsafe.Sizeof(*v))
}

func bytesOfN[T any](v *T, size uint64) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(v)), size)
}

func sliceBytes[T any](s []T) []byte {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*int(unsafe.Sizeof(s[0])))
}
