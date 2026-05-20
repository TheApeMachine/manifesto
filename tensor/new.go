package tensor

import "github.com/theapemachine/manifesto/dtype"

/*
New returns a host tensor with uninitialized storage of the given
shape and dtype. Storage is drawn from the tiered allocator and is
NOT zeroed; callers that need zeroed memory call NewZeroed instead.

This is the canonical entry point for creating host tensors outside
of a Backend.Upload call (e.g. for output buffers of kernels).
*/
func New(shape Shape, asType dtype.DType) (Tensor, error) {
	if !shape.Valid() {
		return nil, ErrShapeInvalid
	}

	bytesNeeded, err := shape.Bytes(asType)

	if err != nil {
		return nil, err
	}

	buffer, err := Allocate(bytesNeeded)

	if err != nil {
		return nil, err
	}

	return newHostTensor(nil, shape, asType, buffer), nil
}

/*
NewZeroed returns a host tensor with explicitly zeroed storage of
the given shape and dtype. Panics if the underlying allocator
returns a non-HostTensor — that would mean the New contract has
been silently broken and a caller would otherwise receive
unzeroed memory.
*/
func NewZeroed(shape Shape, asType dtype.DType) (Tensor, error) {
	value, err := New(shape, asType)

	if err != nil {
		return nil, err
	}

	host, ok := value.(*HostTensor)

	if !ok {
		_ = value.Close()
		panic("tensor.NewZeroed: New returned non-HostTensor; allocator contract broken")
	}

	clear(host.bytes)

	return host, nil
}
