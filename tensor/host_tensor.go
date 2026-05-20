package tensor

import (
	"context"
	"slices"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
HostTensor is the host-resident Tensor implementation. Storage is a
single 64-byte-aligned byte slice from the tiered allocator; native
typed views are produced via unsafe.Slice over that buffer.

The state machine in state.go is enforced through state (atomic) and
the reader counter readers (also atomic). Mutating native views
increment readers; transitions to StatePending or StateClosed wait
for readers to drain (the implementation polls on a sync.Cond — Go
1.24+'s runtime.AddCleanup would be the ideal hook but isn't required
for correctness).
*/
type HostTensor struct {
	backend  *HostBackend
	shape    Shape
	dtype    dtype.DType
	layout   Layout
	bytes    []byte
	state    atomic.Uint32
	readers  atomic.Int32
	cond     *sync.Cond
	mu       sync.Mutex
	closed   atomic.Bool
	arena    *Arena
	epoch    uint64
	parent   *HostTensor
	gradFlag atomic.Bool
	grad     atomic.Pointer[HostTensor]
}

/*
newHostTensor builds a HostTensor over the given byte buffer. The
backend is optional (nil for arena-backed tensors). The shape and
dtype determine the buffer layout but are not re-validated against
len(bytes); the caller must size correctly.
*/
func newHostTensor(
	backend *HostBackend,
	shape Shape,
	asType dtype.DType,
	buffer []byte,
) *HostTensor {
	host := &HostTensor{
		backend: backend,
		shape:   shape,
		dtype:   asType,
		layout:  LayoutDense,
		bytes:   buffer,
	}

	host.cond = sync.NewCond(&host.mu)
	host.state.Store(uint32(StateReady))

	return host
}

/*
newArenaHostTensor builds a HostTensor whose Close is a no-op and
which is invalidated by Arena.Reset.
*/
func newArenaHostTensor(
	arena *Arena,
	shape Shape,
	asType dtype.DType,
	buffer []byte,
) *HostTensor {
	host := newHostTensor(nil, shape, asType, buffer)
	host.arena = arena
	host.epoch = arena.Epoch()

	return host
}

/*
Shape returns the tensor's shape.
*/
func (host *HostTensor) Shape() Shape {
	return host.shape
}

/*
DType returns the tensor's storage dtype.
*/
func (host *HostTensor) DType() dtype.DType {
	return host.dtype
}

/*
Layout returns the tensor's storage layout.
*/
func (host *HostTensor) Layout() Layout {
	return host.layout
}

/*
Location reports Host.
*/
func (host *HostTensor) Location() Location {
	return Host
}

/*
Len returns the logical element count.
*/
func (host *HostTensor) Len() int {
	return host.shape.Len()
}

/*
Bytes returns the storage footprint in bytes.
*/
func (host *HostTensor) Bytes() int {
	return len(host.bytes)
}

/*
State returns the current lifecycle state.
*/
func (host *HostTensor) State() State {
	return State(host.state.Load())
}

/*
Sync blocks until the tensor is StateReady. The host backend is
synchronous so Sync only blocks if the tensor is in StateMutating;
StatePending never happens for host tensors but the state machine
treats it correctly.
*/
func (host *HostTensor) Sync(ctx context.Context) error {
	for {
		state := State(host.state.Load())

		if state == StateReady {
			return nil
		}

		if state == StateClosed {
			return ErrTensorClosed
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		host.mu.Lock()
		state = State(host.state.Load())
		if state == StateReady || state == StateClosed {
			host.mu.Unlock()
			continue
		}
		host.cond.Wait()
		host.mu.Unlock()
	}
}

/*
Ready returns a channel that closes when the tensor reaches
StateReady. Used by callers that prefer select over polling.
*/
func (host *HostTensor) Ready() <-chan struct{} {
	channel := make(chan struct{})

	go func() {
		_ = host.Sync(context.Background())
		close(channel)
	}()

	return channel
}

/*
Close releases the tensor's storage. Idempotent. Arena-backed
tensors do not release storage; Close is a no-op for those.

Host tensors do not have async backend operations, so Close does not
block waiting for outstanding native views. Concurrent Close +
Native() across goroutines is a documented programmer responsibility
(see §2.8 aliasing rules). The state machine still flips to
StateClosed so subsequent native-view calls return ErrTensorClosed,
but any slice already held into the buffer is a dangling reference
after Close completes.
*/
func (host *HostTensor) Close() error {
	if !host.closed.CompareAndSwap(false, true) {
		return nil
	}

	host.state.Store(uint32(StateClosed))
	host.cond.Broadcast()

	if host.arena != nil || host.parent != nil {
		host.bytes = nil
		return nil
	}

	buffer := host.bytes
	host.bytes = nil

	if len(buffer) > 0 {
		Release(buffer)
	}

	return nil
}

/*
acquireView verifies the tensor is in a state that permits view
acquisition. Returns ErrTensorClosed / ErrTensorInTransit as
appropriate. Host tensors do not have a reader-counter blocking
contract; the returned slice's lifetime is the caller's responsibility
(see §2.8 aliasing rules).
*/
func (host *HostTensor) acquireView() error {
	state := State(host.state.Load())

	if state == StateClosed {
		return ErrTensorClosed
	}

	if state == StatePending {
		return ErrTensorInTransit
	}

	if host.arena != nil && host.arena.Epoch() != host.epoch {
		return ErrTensorClosed
	}

	return nil
}

/*
releaseView is a no-op on the host backend; kept for symmetry with
device backends where a view may pin GPU memory until released.
*/
func (host *HostTensor) releaseView() {}

/*
RawBytes returns the storage bytes plus the storage dtype. The
returned slice is a fresh copy; the caller owns it.
*/
func (host *HostTensor) RawBytes() (dtype.DType, []byte, error) {
	if err := host.acquireView(); err != nil {
		return dtype.Invalid, nil, err
	}

	out := slices.Clone(host.bytes)
	host.releaseView()

	return host.dtype, out, nil
}

/*
RequiresGrad reports whether autograd is enabled.
*/
func (host *HostTensor) RequiresGrad() bool {
	return host.gradFlag.Load()
}

/*
SetRequiresGrad flips the autograd flag.
*/
func (host *HostTensor) SetRequiresGrad(yes bool) error {
	host.gradFlag.Store(yes)
	return nil
}

/*
Grad returns the accumulated gradient tensor. Phase 11 fills this in
with finite-difference parity tests. For now we return ErrNoAutograd
unless a gradient has been explicitly stored.
*/
func (host *HostTensor) Grad() (Tensor, error) {
	gradient := host.grad.Load()

	if gradient == nil {
		return nil, ErrNoAutograd
	}

	return gradient, nil
}

/*
GradFn returns the GradFn recorded by the forward kernel that
produced this tensor. nil if no autograd path is active.
*/
func (host *HostTensor) GradFn() GradFn {
	return nil
}

// nativeView returns a typed slice over the tensor's bytes when the
// dtype matches. Caller must call releaseView when done.
func nativeView[T any](host *HostTensor, want dtype.DType) ([]T, error) {
	if err := host.acquireView(); err != nil {
		return nil, err
	}

	if host.dtype != want {
		host.releaseView()
		return nil, ErrDTypeMismatch
	}

	if len(host.bytes) == 0 {
		host.releaseView()
		return nil, nil
	}

	var zero T
	elementSize := int(unsafe.Sizeof(zero))
	elements := len(host.bytes) / elementSize

	pointer := (*T)(unsafe.Pointer(&host.bytes[0]))

	return unsafe.Slice(pointer, elements), nil
}

func (host *HostTensor) Float64Native() ([]float64, error) {
	return nativeView[float64](host, dtype.Float64)
}

func (host *HostTensor) Float32Native() ([]float32, error) {
	return nativeView[float32](host, dtype.Float32)
}

func (host *HostTensor) Float16Native() ([]dtype.F16, error) {
	return nativeView[dtype.F16](host, dtype.Float16)
}

func (host *HostTensor) BFloat16Native() ([]dtype.BF16, error) {
	return nativeView[dtype.BF16](host, dtype.BFloat16)
}

func (host *HostTensor) Float8E4M3Native() ([]dtype.F8E4M3, error) {
	return nativeView[dtype.F8E4M3](host, dtype.Float8E4M3)
}

func (host *HostTensor) Float8E5M2Native() ([]dtype.F8E5M2, error) {
	return nativeView[dtype.F8E5M2](host, dtype.Float8E5M2)
}

func (host *HostTensor) Int64Native() ([]int64, error) {
	return nativeView[int64](host, dtype.Int64)
}

func (host *HostTensor) Int32Native() ([]int32, error) {
	return nativeView[int32](host, dtype.Int32)
}

func (host *HostTensor) Int16Native() ([]int16, error) {
	return nativeView[int16](host, dtype.Int16)
}

func (host *HostTensor) Int8Native() ([]int8, error) {
	return nativeView[int8](host, dtype.Int8)
}

func (host *HostTensor) Uint64Native() ([]uint64, error) {
	return nativeView[uint64](host, dtype.Uint64)
}

func (host *HostTensor) Uint32Native() ([]uint32, error) {
	return nativeView[uint32](host, dtype.Uint32)
}

func (host *HostTensor) Uint16Native() ([]uint16, error) {
	return nativeView[uint16](host, dtype.Uint16)
}

func (host *HostTensor) Uint8Native() ([]uint8, error) {
	return nativeView[uint8](host, dtype.Uint8)
}

/*
BoolNative wraps the storage bytes as a BitVector. Each logical
element is one bit, packed eight per byte little-endian.
*/
func (host *HostTensor) BoolNative() (BitVector, error) {
	if err := host.acquireView(); err != nil {
		return BitVector{}, err
	}

	if host.dtype != dtype.Bool {
		host.releaseView()
		return BitVector{}, ErrDTypeMismatch
	}

	view := NewBitVector(host.bytes, host.shape.Len())
	return view, nil
}

/*
Int4Native wraps the storage bytes as an Int4Vector.
*/
func (host *HostTensor) Int4Native() (Int4Vector, error) {
	if err := host.acquireView(); err != nil {
		return Int4Vector{}, err
	}

	if host.dtype != dtype.Int4 {
		host.releaseView()
		return Int4Vector{}, ErrDTypeMismatch
	}

	pairs := unsafe.Slice((*dtype.Int4Pair)(unsafe.Pointer(&host.bytes[0])), len(host.bytes))
	return NewInt4Vector(pairs, host.shape.Len()), nil
}

/*
Slice returns a zero-copy subview of the tensor's contiguous storage.
*/
func (host *HostTensor) Slice(start, length int) (Tensor, error) {
	if start < 0 || length < 0 || start+length > host.shape.Len() {
		return nil, ErrShapeInvalid
	}

	elementSize, err := host.dtype.Size()

	if err != nil {
		return nil, err
	}

	if elementSize == 0 {
		return nil, ErrLayoutUnsupported
	}

	byteStart := start * elementSize
	byteEnd := byteStart + length*elementSize

	if byteEnd > len(host.bytes) {
		return nil, ErrShapeInvalid
	}

	newShape, err := NewShape([]int{length})

	if err != nil {
		return nil, err
	}

	child := newHostTensor(host.backend, newShape, host.dtype, host.bytes[byteStart:byteEnd])
	child.parent = host

	return child, nil
}

/*
Reshape returns a metadata-only view with the given dimensions.
*/
func (host *HostTensor) Reshape(dims []int) (Tensor, error) {
	newShape, err := host.shape.ReshapeTo(dims)

	if err != nil {
		return nil, err
	}

	child := newHostTensor(host.backend, newShape, host.dtype, host.bytes)
	child.parent = host

	return child, nil
}

var _ Tensor = (*HostTensor)(nil)
