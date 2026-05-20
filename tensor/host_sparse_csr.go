package tensor

import (
	"context"
	"sync/atomic"

	"github.com/theapemachine/manifesto/dtype"
)

/*
HostSparseCSR is the host-resident CSR (Compressed Sparse Row) tensor
implementation. Storage is three buffers drawn from the tiered
allocator: the row-pointer array (int32, rows+1 entries), the
column-index array (int32, nnz entries), and the values array
(value-dtype, nnz entries).

Per the spray-and-pray contract, this file establishes the on-host
CSR shape with correct scalar bodies. Sparse kernels (cuSPARSE,
MPS-Graph sparse, host SIMD) land in Phase 9 work; this Tensor
implementation supports them by exposing the index tensors through
the Indices() accessor.
*/
type HostSparseCSR struct {
	shape    Shape
	dtype    dtype.DType
	values   Tensor
	rowPtr   Tensor
	colIdx   Tensor
	nnz      int
	state    atomic.Uint32
	closed   atomic.Bool
	gradFlag atomic.Bool
}

/*
newHostSparseCSR builds a CSR tensor over three pre-uploaded host
tensors. Callers should go through Backend.UploadSparse rather than
constructing this directly.
*/
func newHostSparseCSR(
	shape Shape,
	valueDType dtype.DType,
	values Tensor,
	rowPtr Tensor,
	colIdx Tensor,
	nnz int,
) *HostSparseCSR {
	sparse := &HostSparseCSR{
		shape:  shape,
		dtype:  valueDType,
		values: values,
		rowPtr: rowPtr,
		colIdx: colIdx,
		nnz:    nnz,
	}

	sparse.state.Store(uint32(StateReady))
	return sparse
}

/*
Shape returns the logical dense shape of the sparse tensor.
*/
func (sparse *HostSparseCSR) Shape() Shape { return sparse.shape }

/*
DType returns the value dtype.
*/
func (sparse *HostSparseCSR) DType() dtype.DType { return sparse.dtype }

/*
Layout reports CSR.
*/
func (sparse *HostSparseCSR) Layout() Layout { return LayoutSparseCSR }

/*
Location reports Host.
*/
func (sparse *HostSparseCSR) Location() Location { return Host }

/*
Len returns the logical element count (rows × cols), not nnz.
*/
func (sparse *HostSparseCSR) Len() int { return sparse.shape.Len() }

/*
Bytes returns the total storage footprint across values + row pointer
+ column index arrays.
*/
func (sparse *HostSparseCSR) Bytes() int {
	return sparse.values.Bytes() + sparse.rowPtr.Bytes() + sparse.colIdx.Bytes()
}

/*
Close releases all three component tensors.
*/
func (sparse *HostSparseCSR) Close() error {
	if !sparse.closed.CompareAndSwap(false, true) {
		return nil
	}

	sparse.state.Store(uint32(StateClosed))

	if err := sparse.values.Close(); err != nil {
		return err
	}

	if err := sparse.rowPtr.Close(); err != nil {
		return err
	}

	return sparse.colIdx.Close()
}

/*
State returns the lifecycle state.
*/
func (sparse *HostSparseCSR) State() State {
	return State(sparse.state.Load())
}

/*
Sync is a no-op on host sparse tensors.
*/
func (sparse *HostSparseCSR) Sync(ctx context.Context) error {
	if State(sparse.state.Load()) == StateClosed {
		return ErrTensorClosed
	}

	return nil
}

/*
Ready returns a closed channel; host sparse tensors are always Ready
unless Closed.
*/
func (sparse *HostSparseCSR) Ready() <-chan struct{} {
	channel := make(chan struct{})
	close(channel)
	return channel
}

/*
NNZ returns the stored non-zero count.
*/
func (sparse *HostSparseCSR) NNZ() int {
	return sparse.nnz
}

/*
Values returns the dense 1-D values tensor.
*/
func (sparse *HostSparseCSR) Values() (Tensor, error) {
	return sparse.values, nil
}

/*
Indices returns the CSR-layout index tensors.
*/
func (sparse *HostSparseCSR) Indices() ([]SparseIndex, error) {
	return []SparseIndex{
		{Name: "row_ptr", Data: sparse.rowPtr},
		{Name: "col_idx", Data: sparse.colIdx},
	}, nil
}

/*
BlockSize is not meaningful for CSR; returns (0, 0, false).
*/
func (sparse *HostSparseCSR) BlockSize() (rows, cols int, ok bool) {
	return 0, 0, false
}

/*
RawBytes is not supported on sparse tensors; the storage is split
across multiple buffers with different dtypes. Callers needing bytes
should iterate Indices() and Values() separately.
*/
func (sparse *HostSparseCSR) RawBytes() (dtype.DType, []byte, error) {
	return dtype.Invalid, nil, ErrLayoutUnsupported
}

/*
Slice is not supported on sparse tensors; the layout doesn't admit
contiguous subviews without materializing dense.
*/
func (sparse *HostSparseCSR) Slice(start, length int) (Tensor, error) {
	return nil, ErrLayoutUnsupported
}

/*
Reshape is not supported on sparse tensors.
*/
func (sparse *HostSparseCSR) Reshape(dims []int) (Tensor, error) {
	return nil, ErrLayoutUnsupported
}

// Native accessors are not meaningful on sparse storage; the caller
// must go through Values() to get the dense values tensor and use
// that tensor's native accessors.

func (sparse *HostSparseCSR) Float64Native() ([]float64, error) {
	return nil, ErrLayoutUnsupported
}

func (sparse *HostSparseCSR) Float32Native() ([]float32, error) {
	return nil, ErrLayoutUnsupported
}

func (sparse *HostSparseCSR) Float16Native() ([]dtype.F16, error) {
	return nil, ErrLayoutUnsupported
}

func (sparse *HostSparseCSR) BFloat16Native() ([]dtype.BF16, error) {
	return nil, ErrLayoutUnsupported
}

func (sparse *HostSparseCSR) Float8E4M3Native() ([]dtype.F8E4M3, error) {
	return nil, ErrLayoutUnsupported
}

func (sparse *HostSparseCSR) Float8E5M2Native() ([]dtype.F8E5M2, error) {
	return nil, ErrLayoutUnsupported
}

func (sparse *HostSparseCSR) Int64Native() ([]int64, error) {
	return nil, ErrLayoutUnsupported
}

func (sparse *HostSparseCSR) Int32Native() ([]int32, error) {
	return nil, ErrLayoutUnsupported
}

func (sparse *HostSparseCSR) Int16Native() ([]int16, error) {
	return nil, ErrLayoutUnsupported
}

func (sparse *HostSparseCSR) Int8Native() ([]int8, error) {
	return nil, ErrLayoutUnsupported
}

func (sparse *HostSparseCSR) Uint64Native() ([]uint64, error) {
	return nil, ErrLayoutUnsupported
}

func (sparse *HostSparseCSR) Uint32Native() ([]uint32, error) {
	return nil, ErrLayoutUnsupported
}

func (sparse *HostSparseCSR) Uint16Native() ([]uint16, error) {
	return nil, ErrLayoutUnsupported
}

func (sparse *HostSparseCSR) Uint8Native() ([]uint8, error) {
	return nil, ErrLayoutUnsupported
}

func (sparse *HostSparseCSR) BoolNative() (BitVector, error) {
	return BitVector{}, ErrLayoutUnsupported
}

func (sparse *HostSparseCSR) Int4Native() (Int4Vector, error) {
	return Int4Vector{}, ErrLayoutUnsupported
}

func (sparse *HostSparseCSR) RequiresGrad() bool {
	return sparse.gradFlag.Load()
}

func (sparse *HostSparseCSR) SetRequiresGrad(yes bool) error {
	sparse.gradFlag.Store(yes)
	return nil
}

func (sparse *HostSparseCSR) Grad() (Tensor, error) {
	return nil, ErrNoAutograd
}

func (sparse *HostSparseCSR) GradFn() GradFn {
	return nil
}

var _ Tensor = (*HostSparseCSR)(nil)
var _ SparseTensor = (*HostSparseCSR)(nil)
