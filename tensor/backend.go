package tensor

import "github.com/theapemachine/manifesto/dtype"

/*
Backend owns persistent tensor storage for one compute Location. Each
of the four backend packages (host, metal, cuda, xla) implements this
interface; the host implementation lives in this package in
host_backend.go.

Upload takes raw bytes plus a source dtype. The backend chooses its
on-device storage dtype, which may differ from the input dtype; the
returned tensor's DType reports the on-device dtype. SupportedDTypes
lists which input dtypes are accepted without conversion at the
boundary; for others the caller must call into
pkg/dtype/convert first.
*/
type Backend interface {
	Location() Location

	// SupportedDTypes returns the input dtypes the backend accepts
	// without conversion. The on-device storage dtype may still
	// differ (e.g. Metal stores Float16 inputs as Float16 natively;
	// the host backend stores anything as itself).
	SupportedDTypes() []dtype.DType

	// SupportedLayouts returns the storage layouts the backend
	// implements. Backends without sparse support return
	// {LayoutDense}.
	SupportedLayouts() []Layout

	// Capabilities returns the backend's capacity, alignment, and
	// feature flags.
	Capabilities() Capabilities

	// Upload moves bytes from host memory into the backend's
	// resident storage. Shape × source dtype must equal len(bytes)
	// for unpacked dtypes; packed dtypes follow the standard
	// (length + N - 1) / N byte count rule.
	Upload(shape Shape, sourceDType dtype.DType, bytes []byte) (Tensor, error)

	// UploadAsync is the non-blocking variant. The returned tensor
	// is in StatePending until the upload event fires. Backends
	// without async support implement this as a synchronous Upload
	// followed by an immediate StateReady; Capabilities.SupportsAsync
	// reports which path is taken.
	UploadAsync(shape Shape, sourceDType dtype.DType, bytes []byte) (Tensor, error)

	// UploadSparse takes a values payload and a layout-specific set
	// of index tensors. Backends without sparse support return
	// ErrLayoutUnsupported.
	UploadSparse(
		shape Shape,
		valueDType dtype.DType,
		layout Layout,
		values []byte,
		indices []SparseIndex,
	) (SparseTensor, error)

	// Download materializes a host copy of a tensor's storage. The
	// returned dtype matches the device-side dtype. Caller owns the
	// returned slice.
	Download(input Tensor) (dtype.DType, []byte, error)

	// Close releases the backend's underlying resources. Outstanding
	// tensors are invalidated.
	Close() error
}

/*
Capabilities describes a backend's runtime properties. Used by the
orchestrator's residency planner and by the kernel dispatch tables.
*/
/*
MaxBytesUnlimited signals that the backend has no enforced storage
budget; the available pool is determined at runtime by the host's
free RAM (host backend) or by the device's driver/runtime (device
backends that choose not to advertise a static cap).
*/
const MaxBytesUnlimited int64 = 0

type Capabilities struct {
	// MaxBytes is the total resident storage budget in bytes.
	// MaxBytesUnlimited (zero) means "no enforced limit" — runtime-
	// determined — and is typical for the host backend once the
	// tiered allocator is in place.
	MaxBytes int64

	// SupportsAsync reports whether UploadAsync is genuinely
	// non-blocking on this backend.
	SupportsAsync bool

	// SupportsSparse reports whether UploadSparse returns real
	// sparse tensors. Backends with this false return
	// ErrLayoutUnsupported.
	SupportsSparse bool

	// SupportsAutograd reports whether backward kernels are
	// available for this backend's forward kernels. Phase 11.
	SupportsAutograd bool

	// NativeAlignment is the byte alignment that native views
	// guarantee. AVX-512 prefers 64; CUDA prefers 128; XLA varies.
	NativeAlignment int

	// NUMANodes reports the host backend's NUMA topology depth.
	// 1 on single-node hosts and on all non-host backends; > 1 on
	// multi-socket Linux hosts where NUMA pinning is active.
	NUMANodes int
}
