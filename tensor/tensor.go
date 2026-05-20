/*
Package tensor is the dtype-aware tensor abstraction for the platform.
Every backend (host, Metal, CUDA, XLA) implements the Backend interface
defined in backend.go and produces Tensor values that conform to the
interface defined here.

This file defines the Tensor interface itself plus its small set of
companion types (Location, the View error sentinels, and the public
constructor helpers that delegate to the new tiered allocator).

The contract is documented in TENSOR_BACKEND_REWRITE.md §2.3 and §2.8.
Read those before extending this file. Particularly: native accessors
return plain typed slices aliasing storage, not View[T] wrappers;
lifetime is enforced through the State machine defined in state.go,
not through reader counters.

Per the spray-and-pray contract (VERIFICATION_STATUS.md), this
package's HostBackend has correct scalar behaviour. The Metal, CUDA,
and XLA implementations exist as separate packages and may be in
"attempted" or "needs-platform-setup" status until their phases
complete.
*/
package tensor

import (
	"context"

	"github.com/theapemachine/manifesto/dtype"
)

/*
Location identifies where a tensor's storage physically lives. The
enum is a string so it serializes cleanly and so the zero value
("") is distinguishable from any real backend.
*/
type Location string

const (
	Host    Location = "host"
	Metal   Location = "metal"
	CUDA    Location = "cuda"
	XLA     Location = "xla"
	Network Location = "network"
)

/*
Tensor is the backend-neutral handle to a chunk of resident storage.
It carries shape, dtype, layout, location, and lifecycle state. Native
view methods return typed slices aliasing the underlying storage; see
state.go for the rules that protect against use-after-close and
in-flight-upload races.
*/
type Tensor interface {
	Shape() Shape
	DType() dtype.DType
	Layout() Layout
	Location() Location

	// Len returns the logical element count.
	Len() int

	// Bytes returns the storage footprint in bytes, computed from
	// Shape × DType. Packed dtypes (Int4, Bool) report the packed size.
	Bytes() int

	// Close releases the storage back to the backend's allocator.
	// Blocks until the tensor's state machine reaches StateReady
	// (drains in-flight async ops) or the context attached to any
	// pending operation expires. Idempotent.
	Close() error

	// Slice returns a zero-copy 1-D subview into the tensor's
	// contiguous storage. start is in logical elements; length is
	// in logical elements. The returned tensor shares storage with
	// the parent and increments the parent's view counter; closing
	// the parent invalidates the slice. Only legal on dense
	// (LayoutDense) contiguous tensors; sparse tensors must use
	// their layout-specific accessors.
	Slice(start, length int) (Tensor, error)

	// Reshape returns a metadata-only view with the given
	// dimensions. Element count must match the original. For
	// rearrangements that change element order, use
	// tensor.Contiguous(tensor.Permute(...)).
	Reshape(dims []int) (Tensor, error)

	// Native typed views. Each returns a slice aliasing storage iff
	// DType matches; otherwise ErrDTypeMismatch. The slice is valid
	// until Close, Sync, the next mutating call on this tensor, or
	// the start of any async backend op touching this tensor.
	// Mutation through the slice is permitted; concurrent host
	// mutation across goroutines is the caller's responsibility.
	Float64Native() ([]float64, error)
	Float32Native() ([]float32, error)
	Float16Native() ([]dtype.F16, error)
	BFloat16Native() ([]dtype.BF16, error)
	Float8E4M3Native() ([]dtype.F8E4M3, error)
	Float8E5M2Native() ([]dtype.F8E5M2, error)
	Int64Native() ([]int64, error)
	Int32Native() ([]int32, error)
	Int16Native() ([]int16, error)
	Int8Native() ([]int8, error)
	Uint64Native() ([]uint64, error)
	Uint32Native() ([]uint32, error)
	Uint16Native() ([]uint16, error)
	Uint8Native() ([]uint8, error)
	BoolNative() (BitVector, error)
	Int4Native() (Int4Vector, error)

	// RawBytes returns the storage as bytes plus its dtype. Always
	// succeeds for a host-resident tensor; device-resident tensors
	// materialize a host copy through the backend's Download path.
	// Caller owns the returned slice and may modify it freely; the
	// slice does not alias tensor storage.
	RawBytes() (dtype.DType, []byte, error)

	// Lifecycle state machine. See state.go and §2.8 of
	// TENSOR_BACKEND_REWRITE.md.
	State() State
	Sync(ctx context.Context) error
	Ready() <-chan struct{}

	// Autograd hooks. See autograd.go and §2.18. The default
	// implementation for non-grad tensors returns false / nil; only
	// tensors flagged with SetRequiresGrad(true) participate.
	RequiresGrad() bool
	SetRequiresGrad(yes bool) error
	Grad() (Tensor, error)
	GradFn() GradFn
}
