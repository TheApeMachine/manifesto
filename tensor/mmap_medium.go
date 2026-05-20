package tensor

import (
	"sync"
)

/*
Tier 2 allocator: 1 MiB – 1 GiB allocations via anonymous mmap with
per-size-class free lists. Freed buffers are MADV_DONTNEED'd on
release.
*/

const (
	mediumMinClass = 20 // 1 MiB
	mediumMaxClass = 30 // 1 GiB
)

type mediumBlock struct {
	next *mediumBlock
	data []byte
}

type mediumPool struct {
	mu      sync.Mutex
	classes [mediumMaxClass - mediumMinClass + 1]*mediumBlock
}

var defaultMedium = &mediumPool{}

/*
mmapMedium returns a 64-byte aligned byte slice of at least bytes
size, drawn from the medium-tier free list or freshly mmap'd if no
slot is free. Returns (nil, err) when mmap fails or the requested
size is outside the medium-tier range.
*/
func mmapMedium(bytesNeeded int) ([]byte, error) {
	class := mediumClass(bytesNeeded)

	if class < 0 {
		return nil, ErrAllocatorExhausted
	}

	classBytes := 1 << uint(mediumMinClass+class)

	defaultMedium.mu.Lock()
	head := defaultMedium.classes[class]

	if head != nil {
		defaultMedium.classes[class] = head.next
		defaultMedium.mu.Unlock()

		return head.data, nil
	}
	defaultMedium.mu.Unlock()

	return mmapAlloc(classBytes)
}

/*
mmapMediumRelease puts a buffer back on the appropriate free list
after MADV_DONTNEED-ing its pages. If the buffer's size class is out
of range, the buffer is unmapped instead.
*/
func mmapMediumRelease(buffer []byte) {
	class := mediumClass(cap(buffer))

	if class < 0 {
		mmapFree(buffer)
		return
	}

	mmapAdviseDontNeed(buffer)

	block := &mediumBlock{data: buffer}

	defaultMedium.mu.Lock()
	block.next = defaultMedium.classes[class]
	defaultMedium.classes[class] = block
	defaultMedium.mu.Unlock()
}

func mediumClass(bytesNeeded int) int {
	if bytesNeeded < (1 << uint(mediumMinClass)) {
		return -1
	}

	if bytesNeeded > (1 << uint(mediumMaxClass)) {
		return -1
	}

	class := 0
	target := 1 << uint(mediumMinClass)

	for target < bytesNeeded {
		target <<= 1
		class++
	}

	if class > mediumMaxClass-mediumMinClass {
		return -1
	}

	return class
}
