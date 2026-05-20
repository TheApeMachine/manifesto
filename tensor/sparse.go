package tensor

/*
SparseTensor extends Tensor with layout-specific accessors. The full
implementation lands in Phase 9; this surface defines the contract so
backends can stub UploadSparse with ErrLayoutUnsupported and kernels
can declare their layout requirements ahead of the implementation.
*/
type SparseTensor interface {
	Tensor

	// NNZ returns the number of stored non-zero elements.
	NNZ() int

	// Values returns a dense 1-D tensor of the stored non-zero
	// values. Dtype matches the parent tensor's DType.
	Values() (Tensor, error)

	// Indices returns the layout-specific index sets. The exact
	// contents depend on the SparseTensor's Layout:
	//   - CSR: [row_ptr (int32, rows+1), col_idx (int32, nnz)]
	//   - CSC: [col_ptr (int32, cols+1), row_idx (int32, nnz)]
	//   - COO: [dim_0_idx, dim_1_idx, ... dim_R_idx] (int32 each, nnz)
	//   - BSR: [row_ptr (int32, blocks_per_row+1), col_idx (int32, nnz_blocks)]
	Indices() ([]SparseIndex, error)

	// BlockSize returns the block dimensions for BSR layout. ok is
	// false for other layouts.
	BlockSize() (rows, cols int, ok bool)
}

/*
SparseIndex names an index tensor inside a sparse representation. The
Name field is conventional (e.g. "row_ptr", "col_idx", "dim_0_idx")
and lets kernels look up the index they need without positional
fragility.
*/
type SparseIndex struct {
	Name string
	Data Tensor
}
