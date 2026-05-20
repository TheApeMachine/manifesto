package tensor

import "fmt"

/*
Layout identifies the storage layout family of a tensor. Dense tensors
use LayoutDense and store contiguous row-major values. Sparse tensors
use one of LayoutSparseCSR / LayoutSparseCSC / LayoutSparseCOO /
LayoutSparseBSR and expose layout-specific accessors via the
SparseTensor interface (sparse.go).
*/
type Layout uint8

const (
	LayoutDense Layout = iota
	LayoutSparseCSR
	LayoutSparseCSC
	LayoutSparseCOO
	LayoutSparseBSR
)

/*
String returns a human-readable name for the layout.
*/
func (layout Layout) String() string {
	switch layout {
	case LayoutDense:
		return "dense"
	case LayoutSparseCSR:
		return "csr"
	case LayoutSparseCSC:
		return "csc"
	case LayoutSparseCOO:
		return "coo"
	case LayoutSparseBSR:
		return "bsr"
	}

	return fmt.Sprintf("layout(%d)", layout)
}

/*
IsSparse reports whether the layout is any of the sparse variants.
*/
func (layout Layout) IsSparse() bool {
	return layout != LayoutDense
}
