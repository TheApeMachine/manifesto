package tensor

import "unsafe"

/*
getBufferPointer returns the base address of a byte slice as a
uintptr. Used by the platform-specific madvise wrappers.

Marked nolint:unused on platforms that don't reference it directly;
the linker keeps the symbol because the mmap_linux / mmap_darwin
files call it via build tags.
*/
func getBufferPointer(buffer []byte) uintptr {
	if len(buffer) == 0 {
		return 0
	}

	return uintptr(unsafe.Pointer(&buffer[0]))
}
