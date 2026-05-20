//go:build linux

package tensor

import (
	"fmt"
	"syscall"
)

/*
Linux mmap primitives. Anonymous private mappings with read/write
protections. MADV_DONTNEED reclaims physical pages without unmapping;
MADV_HUGEPAGE asks the kernel to back the mapping with transparent
huge pages.

The allocator returns (nil, error) on failure instead of falling back
to make([]byte, n); mixing heap buffers with mmap-backed buffers
would cause Munmap to be called on heap memory and corrupt the
process. Callers must handle the error path.
*/

func mmapAlloc(bytesNeeded int) ([]byte, error) {
	buffer, err := syscall.Mmap(
		-1,
		0,
		bytesNeeded,
		syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_ANON|syscall.MAP_PRIVATE,
	)

	if err != nil {
		return nil, fmt.Errorf("tensor/mmap: anonymous Mmap(%d) failed: %w", bytesNeeded, err)
	}

	return buffer, nil
}

func mmapFree(buffer []byte) {
	if len(buffer) == 0 {
		return
	}

	_ = syscall.Munmap(buffer)
}

func mmapAdviseDontNeed(buffer []byte) {
	if len(buffer) == 0 {
		return
	}

	_ = madviseRaw(buffer, syscallMadvDontNeed)
}

func mmapAdviseHugePage(buffer []byte) {
	if len(buffer) == 0 {
		return
	}

	_ = madviseRaw(buffer, syscallMadvHugePage)
}

const (
	syscallMadvDontNeed = 4
	syscallMadvHugePage = 14
)

func madviseRaw(buffer []byte, advice int) error {
	_, _, errno := syscall.Syscall(
		syscall.SYS_MADVISE,
		uintptr(getBufferPointer(buffer)),
		uintptr(len(buffer)),
		uintptr(advice),
	)

	if errno != 0 {
		return errno
	}

	return nil
}
