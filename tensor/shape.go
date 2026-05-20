package tensor

import (
	"fmt"
	"slices"

	"github.com/theapemachine/manifesto/dtype"
)

const maxInt = int(^uint(0) >> 1)

/*
Shape is a tensor's logical dimensions. Storage is strictly contiguous
row-major; there is no general-stride support (§5.6 of
TENSOR_BACKEND_REWRITE.md). Reshape preserves element order; transpose
materializes through tensor.Contiguous.

The zero value of Shape is invalid; use NewShape.
*/
type Shape struct {
	dims     []int
	elements int
	valid    bool
}

/*
NewShape validates the dimensions and computes the element count.
Empty input represents a scalar tensor (Len() == 1). Zero dimensions
are permitted and represent an empty tensor (Len() == 0). Negative
dimensions are rejected.
*/
func NewShape(dims []int) (Shape, error) {
	shape := Shape{
		dims:     slices.Clone(dims),
		elements: 1,
		valid:    true,
	}

	for dimensionIndex, dimension := range shape.dims {
		if dimension < 0 {
			return Shape{}, fmt.Errorf(
				"%w: dimension %d is negative (%d)",
				ErrShapeInvalid, dimensionIndex, dimension,
			)
		}

		if dimension == 0 {
			shape.elements = 0
			continue
		}

		if shape.elements > maxInt/dimension {
			return Shape{}, fmt.Errorf("%w: element count overflows int", ErrShapeInvalid)
		}

		shape.elements *= dimension
	}

	return shape, nil
}

/*
Dims returns a defensive copy of the dimensions.
*/
func (shape Shape) Dims() []int {
	return slices.Clone(shape.dims)
}

/*
Rank returns the number of dimensions.
*/
func (shape Shape) Rank() int {
	return len(shape.dims)
}

/*
Len returns the number of logical elements addressed by the shape.
*/
func (shape Shape) Len() int {
	return shape.elements
}

/*
Valid reports whether the shape was produced by NewShape.
*/
func (shape Shape) Valid() bool {
	return shape.valid
}

/*
Bytes returns the storage footprint for this shape under the given
dtype. Handles packed dtypes correctly.
*/
func (shape Shape) Bytes(at dtype.DType) (int, error) {
	if !shape.valid {
		return 0, ErrShapeInvalid
	}

	return at.BytesFor(shape.elements)
}

/*
Equal reports whether two shapes have the same dimensions in the same
order. Validity is part of the comparison; an invalid Shape is not
equal to any other Shape including itself.
*/
func (shape Shape) Equal(other Shape) bool {
	if !shape.valid || !other.valid {
		return false
	}

	return slices.Equal(shape.dims, other.dims) && shape.elements == other.elements
}

/*
ReshapeTo produces a new Shape with the supplied dimensions. The
element count must match the original; otherwise ErrShapeMismatch.
This is the metadata-only Reshape primitive; it does not move data.
*/
func (shape Shape) ReshapeTo(dims []int) (Shape, error) {
	target, err := NewShape(dims)

	if err != nil {
		return Shape{}, err
	}

	if target.elements != shape.elements {
		return Shape{}, fmt.Errorf(
			"%w: reshape product %d != original %d",
			ErrShapeMismatch, target.elements, shape.elements,
		)
	}

	return target, nil
}

/*
SliceTo produces a Shape representing a 1-D subview of length
elements. Used by Tensor.Slice. The resulting shape has rank 1; the
caller is expected to Reshape afterward if multi-dimensional access
is needed.
*/
func (shape Shape) SliceTo(length int) (Shape, error) {
	if !shape.valid {
		return Shape{}, ErrShapeInvalid
	}

	if length < 0 || length > shape.elements {
		return Shape{}, fmt.Errorf(
			"%w: slice length %d out of range [0, %d]",
			ErrShapeInvalid, length, shape.elements,
		)
	}

	return NewShape([]int{length})
}

/*
String returns a human-readable representation: "[2 3 4]".
*/
func (shape Shape) String() string {
	if !shape.valid {
		return "<invalid>"
	}

	return fmt.Sprintf("%v", shape.dims)
}
