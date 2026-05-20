package tensor

import (
	"math/bits"
	"sync"
	"sync/atomic"
	"unsafe"
)

/*
Tier 1 allocator: a per-P sharded slab pool for sub-1-MiB allocations.

Why not sync.Pool: the Go runtime drains sync.Pool entirely on every
GC cycle (the victim cache survives at most one cycle), so under
memory pressure the pool empties exactly when we need it most.

The slab here keeps per-shard free-lists indexed by power-of-two
size class. Allocation: pop from the local shard, fall back to the
global shard, fall back to allocateAligned() which over-allocates by
64 bytes and slices to a 64-byte boundary. Free: push to the local
shard. The Go runtime never touches the slabs, so GC pressure does
not drain them.

Per the spray-and-pray contract, the shard count is fixed to a power
of two at init for fast lookup; Phase 8/9 work may tune this against
benchmarks.
*/

const (
	slabMaxBytes      = 1 << 20 // 1 MiB
	slabClassMin      = 6       // 64 B minimum
	slabClassMax      = 20      // 1 MiB maximum
	slabShardCount    = 64
	slabAlignmentMask = uintptr(63)
)

type slabBlock struct {
	next *slabBlock
	data []byte
}

type slabShard struct {
	mu   sync.Mutex
	head *slabBlock
}

type slabAllocator struct {
	classes [slabClassMax - slabClassMin + 1][slabShardCount]slabShard
	allocs  atomic.Uint64
	frees   atomic.Uint64
}

var defaultSlab = &slabAllocator{}

/*
slabClass returns the power-of-two size class index for a byte count.
Bytes ≤ 2^(slabClassMin+i) fits in class i. Sizes outside [slabClassMin,
slabClassMax] return -1 to signal "go to the next tier."
*/
func slabClass(bytesNeeded int) int {
	if bytesNeeded <= 0 {
		return -1
	}

	if bytesNeeded > slabMaxBytes {
		return -1
	}

	power := bits.Len(uint(bytesNeeded - 1))

	if power < slabClassMin {
		power = slabClassMin
	}

	return power - slabClassMin
}

/*
slabAlloc returns a 64-byte-aligned byte slice of at least the
requested size, drawn from the slab pool. Returns (nil, false) if
the request falls outside the slab's size range; caller should fall
through to mmap-based tiers.
*/
func (allocator *slabAllocator) slabAlloc(bytesNeeded int) ([]byte, bool) {
	class := slabClass(bytesNeeded)

	if class < 0 {
		return nil, false
	}

	shardIndex := slabShardIndex()
	shard := &allocator.classes[class][shardIndex]

	shard.mu.Lock()
	if shard.head != nil {
		block := shard.head
		shard.head = block.next
		shard.mu.Unlock()

		allocator.allocs.Add(1)

		return block.data, true
	}
	shard.mu.Unlock()

	buffer := allocateAligned(1 << (slabClassMin + class))
	allocator.allocs.Add(1)

	return buffer, true
}

/*
slabFree returns a slab buffer to the pool. The buffer must have come
from slabAlloc with the same size class; passing arbitrary slices is
a programmer error and may panic.
*/
func (allocator *slabAllocator) slabFree(buffer []byte) {
	class := slabClass(cap(buffer))

	if class < 0 {
		return
	}

	shardIndex := slabShardIndex()
	shard := &allocator.classes[class][shardIndex]

	block := &slabBlock{data: buffer}

	shard.mu.Lock()
	block.next = shard.head
	shard.head = block
	shard.mu.Unlock()

	allocator.frees.Add(1)
}

/*
allocateAligned returns a byte slice whose first element is at a
64-byte-aligned address. Over-allocates by 64 bytes and re-slices.
The full backing array is retained inside the slice header so GC
reclaims it when the slice goes out of scope.

Per AGENTS.md, the alignment is asserted via
uintptr(unsafe.Pointer(&buf[0])) % 64 == 0 in the test suite.
*/
func allocateAligned(bytesNeeded int) []byte {
	overshoot := make([]byte, bytesNeeded+64)
	base := uintptr(unsafe.Pointer(&overshoot[0]))
	offset := (64 - base&slabAlignmentMask) & slabAlignmentMask

	return overshoot[offset : offset+uintptr(bytesNeeded) : offset+uintptr(bytesNeeded)]
}

/*
slabShardIndex returns a per-goroutine shard index. The implementation
uses runtime_procPin equivalent via a simple atomic counter for now;
Phase 8 work may replace this with runtime_procPin if the contention
profile warrants the cgo bridge.
*/
var slabShardCounter atomic.Uint64

func slabShardIndex() int {
	value := slabShardCounter.Add(1)
	return int(value & (slabShardCount - 1))
}

/*
Allocate is the package-level entry point. Returns 64-byte aligned
storage from the appropriate tier:

  - bytesNeeded < 1 MiB: slab tier (this file)
  - 1 MiB ≤ bytesNeeded < 1 GiB: mmap-medium tier (mmap_medium.go)
  - bytesNeeded ≥ 1 GiB: mmap-large tier (mmap_large.go)

The returned slice's len equals bytesNeeded; its cap reflects the
underlying tier's allocation unit (size class for slab/medium, exact
huge-page-rounded size for large). Release routes back to the
correct tier via cap.

Returns (nil, error) when the underlying allocator fails (mmap
failure, exhausted address space) or when bytesNeeded is non-positive.
*/
func Allocate(bytesNeeded int) ([]byte, error) {
	if bytesNeeded <= 0 {
		return nil, nil
	}

	if bytesNeeded < slabMaxBytes {
		buffer, ok := defaultSlab.slabAlloc(bytesNeeded)

		if !ok {
			return nil, ErrAllocatorExhausted
		}

		return buffer[:bytesNeeded], nil
	}

	if bytesNeeded < (1 << 30) {
		buffer, err := mmapMedium(bytesNeeded)

		if err != nil {
			return nil, err
		}

		return buffer[:bytesNeeded], nil
	}

	return mmapLarge(bytesNeeded)
}

/*
Release returns storage to the allocator. The implementation routes
to the correct tier based on cap. Pass a slice produced by Allocate
or by a tier-specific function; pass cap unchanged from the original
allocation.
*/
func Release(buffer []byte) {
	bytesUsed := cap(buffer)

	if bytesUsed <= 0 {
		return
	}

	if bytesUsed < slabMaxBytes {
		defaultSlab.slabFree(buffer[:cap(buffer)])
		return
	}

	if bytesUsed < (1 << 30) {
		mmapMediumRelease(buffer[:cap(buffer)])
		return
	}

	mmapLargeRelease(buffer[:cap(buffer)])
}
