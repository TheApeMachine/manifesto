package tensor

import (
	"sync"
	"sync/atomic"

	"github.com/theapemachine/manifesto/dtype"
)

/*
Arena is an opt-in bump allocator for forward-pass scratch tensors.
Backed by a single mmap region drawn from the tiered allocator,
Arena hands out tensors whose Close is a no-op; the entire arena
resets wholesale via Reset between forward passes.

Per §2.7 of TENSOR_BACKEND_REWRITE.md, arena tensors flag themselves
so that lifetime accounting elsewhere stays correct. The state
machine still applies — a post-Reset access on an outstanding arena
tensor handle panics via the reader counter.

This is a host-only construct. Backend implementations may use their
own arena strategies (Metal heap, CUDA stream-ordered allocator).
*/
type Arena struct {
	mu      sync.Mutex
	storage []byte
	offset  int
	epoch   atomic.Uint64
}

/*
NewArena allocates an Arena of the given total byte capacity. The
underlying buffer is drawn from the tiered allocator and is therefore
64-byte-aligned. Indeterminate contents on first acquisition; reset
recycles without zeroing. Returns nil + non-nil error on allocator
failure so callers can fail fast at construction.
*/
func NewArena(capacity int) (*Arena, error) {
	storage, err := Allocate(capacity)

	if err != nil {
		return nil, err
	}

	return &Arena{storage: storage}, nil
}

/*
Bytes returns the arena's total capacity in bytes.
*/
func (arena *Arena) Bytes() int {
	return len(arena.storage)
}

/*
Available returns the bytes remaining before the next allocation
fails. Best-effort; under concurrent access the result is only a
hint.
*/
func (arena *Arena) Available() int {
	arena.mu.Lock()
	defer arena.mu.Unlock()

	return len(arena.storage) - arena.offset
}

/*
allocateBytes hands out alignedBytes-aligned scratch space from the
bump arena. Returns nil if the arena is exhausted.
*/
func (arena *Arena) allocateBytes(bytesNeeded int) []byte {
	if bytesNeeded <= 0 {
		return nil
	}

	arena.mu.Lock()
	defer arena.mu.Unlock()

	aligned := (arena.offset + 63) &^ 63

	if aligned+bytesNeeded > len(arena.storage) {
		return nil
	}

	slice := arena.storage[aligned : aligned+bytesNeeded : aligned+bytesNeeded]
	arena.offset = aligned + bytesNeeded

	return slice
}

/*
Reset invalidates every outstanding arena tensor handle and rewinds
the bump pointer to zero. Subsequent native-view calls on tensors
handed out before Reset will trip the epoch check and return
ErrTensorClosed.
*/
func (arena *Arena) Reset() {
	arena.mu.Lock()
	arena.offset = 0
	arena.mu.Unlock()
	arena.epoch.Add(1)

	mmapAdviseDontNeed(arena.storage)
}

/*
Close releases the arena's backing storage. After Close, the arena
cannot be reused.
*/
func (arena *Arena) Close() {
	arena.mu.Lock()
	storage := arena.storage
	arena.storage = nil
	arena.offset = 0
	arena.mu.Unlock()

	if len(storage) > 0 {
		Release(storage)
	}
}

/*
Epoch returns the current arena epoch. Arena tensors capture this on
creation and compare on every access; Reset increments it.
*/
func (arena *Arena) Epoch() uint64 {
	return arena.epoch.Load()
}

/*
New returns a host tensor backed by arena scratch storage. The
returned tensor's Close is a no-op; lifetime is governed by the
arena's Reset. dtype must be unpacked (Int4/Bool are rejected to keep
the bump arithmetic simple — sparse and packed live in their own
phases).
*/
func (arena *Arena) New(shape Shape, asType dtype.DType) (Tensor, error) {
	if asType.IsPacked() {
		return nil, ErrDTypeUnsupported
	}

	bytesNeeded, err := shape.Bytes(asType)

	if err != nil {
		return nil, err
	}

	buffer := arena.allocateBytes(bytesNeeded)

	if buffer == nil {
		return nil, ErrAllocatorExhausted
	}

	return newArenaHostTensor(arena, shape, asType, buffer), nil
}
