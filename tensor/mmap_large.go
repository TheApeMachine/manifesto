package tensor

import (
	"fmt"
	"math"
	"sync"
)

/*
Tier 3 allocator: ≥ 1 GiB allocations via anonymous mmap with
huge-page advice. Freed buffers go on an indefinite free list and are
never returned to the OS.
*/

type largeBlock struct {
	bytes int
	data  []byte
	next  *largeBlock
}

type largePool struct {
	mu   sync.Mutex
	head *largeBlock
}

var defaultLarge = &largePool{}

/*
mmapLarge returns an mmap'd buffer of at least bytesNeeded size,
drawn from the free list or freshly allocated. Returns (nil, err) on
mmap failure or overflow during huge-page rounding.
*/
func mmapLarge(bytesNeeded int) ([]byte, error) {
	rounded, err := roundToHugePage(bytesNeeded)

	if err != nil {
		return nil, err
	}

	defaultLarge.mu.Lock()
	previous := (*largeBlock)(nil)
	cursor := defaultLarge.head

	for cursor != nil {
		if cursor.bytes >= bytesNeeded {
			if previous == nil {
				defaultLarge.head = cursor.next
				defaultLarge.mu.Unlock()

				return cursor.data[:bytesNeeded:cursor.bytes], nil
			}

			previous.next = cursor.next
			defaultLarge.mu.Unlock()

			return cursor.data[:bytesNeeded:cursor.bytes], nil
		}

		previous = cursor
		cursor = cursor.next
	}
	defaultLarge.mu.Unlock()

	allocated, err := mmapAlloc(rounded)

	if err != nil {
		return nil, err
	}

	mmapAdviseHugePage(allocated)

	return allocated[:bytesNeeded:rounded], nil
}

/*
mmapLargeRelease parks a buffer on the indefinite free list.
*/
func mmapLargeRelease(buffer []byte) {
	mmapAdviseDontNeed(buffer)

	block := &largeBlock{bytes: cap(buffer), data: buffer}

	defaultLarge.mu.Lock()
	block.next = defaultLarge.head
	defaultLarge.head = block
	defaultLarge.mu.Unlock()
}

/*
roundToHugePage rounds a byte count up to the next 2 MiB boundary.
Returns an error if the rounded value overflows int.
*/
func roundToHugePage(bytesNeeded int) (int, error) {
	const hugePage = 2 * 1024 * 1024

	if bytesNeeded > math.MaxInt-(hugePage-1) {
		return 0, fmt.Errorf("tensor/mmap: byte count %d overflows on huge-page rounding", bytesNeeded)
	}

	return (bytesNeeded + hugePage - 1) &^ (hugePage - 1), nil
}
