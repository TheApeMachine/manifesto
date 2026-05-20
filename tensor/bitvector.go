package tensor

import "fmt"

/*
BitVector is the host-side view returned by Tensor.BoolNative for
Bool-dtype tensors. Storage is one bit per logical element, packed
eight per byte in little-endian bit order: element 0 is bit 0 of
byte 0, element 8 is bit 0 of byte 1.
*/
type BitVector struct {
	data   []byte
	length int
}

/*
NewBitVector wraps an existing byte slice as a BitVector. length is
the logical element count. The caller is responsible for sizing data
to at least (length + 7) / 8 bytes; bits beyond length are read as
the underlying bytes and may be garbage.
*/
func NewBitVector(data []byte, length int) BitVector {
	return BitVector{data: data, length: length}
}

/*
Len returns the logical element count.
*/
func (vector BitVector) Len() int {
	return vector.length
}

/*
Bytes returns the underlying byte storage. Modifying the slice
modifies the tensor in place.
*/
func (vector BitVector) Bytes() []byte {
	return vector.data
}

/*
Get returns the bit at the given logical index. Panics on out-of-
range access to match Go's native slice indexing behaviour and
surface programmer errors loudly rather than masking them with a
false return.
*/
func (vector BitVector) Get(index int) bool {
	if index < 0 || index >= vector.length {
		panic(fmt.Sprintf("BitVector.Get: index %d out of range [0, %d)", index, vector.length))
	}

	return vector.data[index/8]&(1<<(uint(index)%8)) != 0
}

/*
Set writes the bit at the given logical index. Panics on out-of-
range access to match Go's native slice indexing behaviour.
*/
func (vector BitVector) Set(index int, value bool) {
	if index < 0 || index >= vector.length {
		panic(fmt.Sprintf("BitVector.Set: index %d out of range [0, %d)", index, vector.length))
	}

	mask := byte(1) << (uint(index) % 8)

	if value {
		vector.data[index/8] |= mask
		return
	}

	vector.data[index/8] &^= mask
}
