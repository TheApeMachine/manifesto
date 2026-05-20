package tensor

import "github.com/theapemachine/manifesto/dtype"

/*
Contiguous returns a freshly allocated contiguous tensor with the
same data as input. For host tensors the result aliases through
RawBytes + Upload; for device tensors the implementation materializes
through Download + Upload to the same backend.

Per §5.6 of TENSOR_BACKEND_REWRITE.md, transpose / permute / broadcast
operations live above this primitive. They produce a logical
rearrangement (currently no general-stride storage; rearrangements
materialize unconditionally) and then call Contiguous to produce a
fresh contiguous tensor.

Phase 8 fills in real transpose/permute kernels; this function is
their target.
*/
func Contiguous(input Tensor) (Tensor, error) {
	storeDType, bytes, err := input.RawBytes()

	if err != nil {
		return nil, err
	}

	return NewFromBytes(input.Shape(), storeDType, bytes)
}

/*
NewFromBytes returns a host tensor with the given dtype and shape,
populated from the byte slice. Caller owns the byte slice; this
function copies.
*/
func NewFromBytes(shape Shape, asType dtype.DType, bytes []byte) (Tensor, error) {
	backend := NewHostBackend()

	return backend.Upload(shape, asType, bytes)
}
