package tensor

import "errors"

/*
Sentinel errors. Callers compare with errors.Is. Use these for the
common failure modes so the test suite and downstream code can branch
on intent rather than parsing strings.
*/
var (
	// ErrDTypeMismatch is returned by a Native accessor when the
	// requested element type does not match the tensor's storage dtype.
	ErrDTypeMismatch = errors.New("tensor: dtype mismatch")

	// ErrDTypeUnsupported is returned by Backend.Upload when the
	// source dtype is not in the backend's SupportedDTypes list and
	// the caller has not converted to a supported dtype first.
	ErrDTypeUnsupported = errors.New("tensor: dtype not supported by backend")

	// ErrLayoutUnsupported is returned when an operation requires a
	// dense tensor but received a sparse tensor (or vice versa), or
	// when a sparse layout is not implemented by the backend.
	ErrLayoutUnsupported = errors.New("tensor: layout not supported")

	// ErrShapeMismatch is returned when an operation's input shapes
	// are incompatible (matmul inner-dim mismatch, broadcast not
	// permitted, etc.).
	ErrShapeMismatch = errors.New("tensor: shape mismatch")

	// ErrTensorInTransit is returned when a host-side view is
	// requested on a tensor currently in StatePending (e.g. async
	// upload in flight). Call Sync first, or check Ready.
	ErrTensorInTransit = errors.New("tensor: in transit; call Sync first")

	// ErrTensorMutating is returned when an upload or other
	// state-changing op is requested on a tensor that has an
	// outstanding native view holding it in StateMutating.
	ErrTensorMutating = errors.New("tensor: native view outstanding")

	// ErrTensorClosed is returned when any operation is attempted on
	// a tensor whose Close has already run.
	ErrTensorClosed = errors.New("tensor: closed")

	// ErrBackendClosed is returned when an operation is attempted on
	// a backend whose Close has already run.
	ErrBackendClosed = errors.New("tensor: backend closed")

	// ErrShapeInvalid is returned by NewShape when the dimensions
	// fail validation (negative, overflow, etc.).
	ErrShapeInvalid = errors.New("tensor: invalid shape")

	// ErrNoAutograd is returned by Tensor.Grad on tensors that have
	// not had SetRequiresGrad(true) called.
	ErrNoAutograd = errors.New("tensor: autograd not enabled on this tensor")

	// ErrBackwardNotImplemented is returned by a forward kernel when
	// recording autograd is requested but the corresponding backward
	// kernel has not been registered. Caught at record time, not at
	// backward time, so missing coverage is visible immediately.
	ErrBackwardNotImplemented = errors.New("tensor: backward not implemented for this op")

	// ErrNeedsPlatformSetup is returned by skeletal backend bodies
	// that compile but require platform toolchain (CUDA, libnuma,
	// XLA runtime) that cannot be assumed at build time. See
	// VERIFICATION_STATUS.md.
	ErrNeedsPlatformSetup = errors.New("tensor: platform toolchain not available")

	// ErrAllocatorExhausted is returned when an allocator tier
	// cannot satisfy a request and the next tier is also full or
	// disabled. Distinct from ErrBackendClosed so callers can retry
	// with a smaller allocation.
	ErrAllocatorExhausted = errors.New("tensor: allocator exhausted")
)
