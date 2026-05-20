package tensor

import (
	"sync/atomic"

	"github.com/theapemachine/manifesto/dtype"
)

/*
HostBackend is the Backend implementation for host-resident storage.
Allocations route through the tiered allocator (slab + mmap-medium +
mmap-large); there is no fixed-size arena. Concurrent uploads scale
with cores because the slab and mmap free lists are sharded.
*/
type HostBackend struct {
	closed atomic.Bool
}

/*
NewHostBackend returns a host backend. There is no per-backend state
beyond the closed flag; the allocator package-level functions handle
storage.
*/
func NewHostBackend() *HostBackend {
	return &HostBackend{}
}

/*
Location reports Host.
*/
func (backend *HostBackend) Location() Location {
	return Host
}

/*
SupportedDTypes lists the dtypes the host backend accepts directly.
The host stores anything as itself, so every dtype is supported.
*/
func (backend *HostBackend) SupportedDTypes() []dtype.DType {
	return []dtype.DType{
		dtype.Float64,
		dtype.Float32,
		dtype.Float16,
		dtype.BFloat16,
		dtype.Float8E4M3,
		dtype.Float8E5M2,
		dtype.Int64,
		dtype.Int32,
		dtype.Int16,
		dtype.Int8,
		dtype.Int4,
		dtype.Uint64,
		dtype.Uint32,
		dtype.Uint16,
		dtype.Uint8,
		dtype.Bool,
	}
}

/*
SupportedLayouts lists the storage layouts the host backend
implements. Dense is always supported; CSR is the first sparse layout
to land. CSC / COO / BSR follow as kernels arrive.
*/
func (backend *HostBackend) SupportedLayouts() []Layout {
	return []Layout{LayoutDense, LayoutSparseCSR}
}

/*
Capabilities returns the host backend's runtime properties. MaxBytes
of zero means "no enforced limit; allocation is bounded only by the
host's available RAM at runtime" — the tiered allocator does not
impose a fixed budget on the host backend.
*/
func (backend *HostBackend) Capabilities() Capabilities {
	return Capabilities{
		MaxBytes:         MaxBytesUnlimited,
		SupportsAsync:    false,
		SupportsSparse:   true,
		SupportsAutograd: false,
		NativeAlignment:  64,
		NUMANodes:        NUMAQuery(),
	}
}

/*
Upload copies the given byte buffer into a freshly allocated
host-aligned storage and returns a Tensor that aliases it. The
input byte count must match the shape × dtype.
*/
func (backend *HostBackend) Upload(
	shape Shape,
	sourceDType dtype.DType,
	bytesIn []byte,
) (Tensor, error) {
	if backend.closed.Load() {
		return nil, ErrBackendClosed
	}

	if !shape.Valid() {
		return nil, ErrShapeInvalid
	}

	expected, err := shape.Bytes(sourceDType)

	if err != nil {
		return nil, err
	}

	if expected != len(bytesIn) {
		return nil, ErrShapeMismatch
	}

	buffer, err := Allocate(expected)

	if err != nil {
		return nil, err
	}

	copy(buffer, bytesIn)

	return newHostTensor(backend, shape, sourceDType, buffer), nil
}

/*
UploadAsync on the host backend is identical to Upload (the host
allocator is synchronous and the bytes are immediately accessible).
SupportsAsync reports false so callers know not to expect overlap.
*/
func (backend *HostBackend) UploadAsync(
	shape Shape,
	sourceDType dtype.DType,
	bytesIn []byte,
) (Tensor, error) {
	return backend.Upload(shape, sourceDType, bytesIn)
}

/*
UploadSparse stores sparse tensor data in the indicated layout. CSR
is implemented in this phase; CSC / COO / BSR follow the same shape
and land as their kernels do.
*/
func (backend *HostBackend) UploadSparse(
	shape Shape,
	valueDType dtype.DType,
	layout Layout,
	values []byte,
	indices []SparseIndex,
) (SparseTensor, error) {
	if backend.closed.Load() {
		return nil, ErrBackendClosed
	}

	if layout != LayoutSparseCSR {
		return nil, ErrLayoutUnsupported
	}

	expectedBytes, err := valueDType.BytesFor(nnzFromCSR(indices))

	if err != nil {
		return nil, err
	}

	if expectedBytes != len(values) {
		return nil, ErrShapeMismatch
	}

	valueShape, err := NewShape([]int{nnzFromCSR(indices)})

	if err != nil {
		return nil, err
	}

	valueTensor, err := backend.Upload(valueShape, valueDType, values)

	if err != nil {
		return nil, err
	}

	rowPtr := lookupSparseIndex(indices, "row_ptr")
	colIdx := lookupSparseIndex(indices, "col_idx")

	if rowPtr == nil || colIdx == nil {
		_ = valueTensor.Close()
		return nil, ErrShapeMismatch
	}

	return newHostSparseCSR(
		shape,
		valueDType,
		valueTensor,
		rowPtr,
		colIdx,
		nnzFromCSR(indices),
	), nil
}

func nnzFromCSR(indices []SparseIndex) int {
	colIdx := lookupSparseIndex(indices, "col_idx")

	if colIdx == nil {
		return 0
	}

	return colIdx.Len()
}

func lookupSparseIndex(indices []SparseIndex, name string) Tensor {
	for _, candidate := range indices {
		if candidate.Name == name {
			return candidate.Data
		}
	}

	return nil
}

/*
Download returns a fresh host-side byte copy of the tensor's storage
plus its dtype.
*/
func (backend *HostBackend) Download(input Tensor) (dtype.DType, []byte, error) {
	if backend.closed.Load() {
		return dtype.Invalid, nil, ErrBackendClosed
	}

	return input.RawBytes()
}

/*
Close marks the backend closed. Outstanding tensors remain valid
until their own Close runs; the closed flag prevents new uploads.
*/
func (backend *HostBackend) Close() error {
	backend.closed.Store(true)
	return nil
}

var _ Backend = (*HostBackend)(nil)
