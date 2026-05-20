package tensor

import "github.com/theapemachine/manifesto/dtype"

/*
Int4Vector is the host-side view returned by Tensor.Int4Native for
Int4-dtype tensors. Storage is two int4 nibbles per byte, packed
little-endian (element 0 is the low nibble of byte 0).
*/
type Int4Vector struct {
	data   []dtype.Int4Pair
	length int
}

/*
NewInt4Vector wraps an existing slice of Int4Pair as an Int4Vector.
length is the logical element count and may be odd; in that case the
high nibble of the last byte is unused but still readable.
*/
func NewInt4Vector(data []dtype.Int4Pair, length int) Int4Vector {
	return Int4Vector{data: data, length: length}
}

/*
Len returns the logical element count.
*/
func (vector Int4Vector) Len() int {
	return vector.length
}

/*
Pairs returns the underlying packed storage. Modifying the slice
modifies the tensor in place.
*/
func (vector Int4Vector) Pairs() []dtype.Int4Pair {
	return vector.data
}

/*
Bytes returns the underlying storage as a byte slice. The length is
(vector.length + 1) / 2.
*/
func (vector Int4Vector) Bytes() []byte {
	out := make([]byte, len(vector.data))

	for index, pair := range vector.data {
		out[index] = pair.Bits()
	}

	return out
}

/*
Get returns the sign-extended int8 value at the given logical index.
*/
func (vector Int4Vector) Get(index int) int8 {
	if index < 0 || index >= vector.length {
		return 0
	}

	pair := vector.data[index/2]

	if index%2 == 0 {
		return pair.Lo()
	}

	return pair.Hi()
}

/*
Set writes the int4 value at the given logical index. Inputs outside
the int4 range clamp to the boundary.
*/
func (vector Int4Vector) Set(index int, value int8) {
	if index < 0 || index >= vector.length {
		return
	}

	pair := vector.data[index/2]

	if index%2 == 0 {
		vector.data[index/2] = pair.WithLo(value)
		return
	}

	vector.data[index/2] = pair.WithHi(value)
}
