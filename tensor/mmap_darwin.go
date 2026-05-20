//go:build darwin

package tensor

import (
	"fmt"
	"syscall"
)

/*
Darwin mmap primitives. Anonymous private mappings. The allocator
returns (nil, error) on failure; mixing heap buffers with mmap-backed
buffers would corrupt the process when Munmap runs on heap memory.
*/

const (
	madvFreeReusable = 7
)

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

	_ = madviseRaw(buffer, madvFreeReusable)
}

func mmapAdviseHugePage(buffer []byte) {
	// Darwin doesn't have a portable "huge page" hint that survives
	// outside the VM subsystem. Leave as no-op; the mmap allocation
	// already prefers superpage-friendly alignment.
}

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
