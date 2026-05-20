package dtype

/*
Int4Pair packs two signed 4-bit integers into a single byte. The
low nibble (bits 0..3) is element index 2*i in a packed buffer, and
the high nibble (bits 4..7) is element index 2*i + 1. Sign extension
happens on read.

The packing convention matches the GPTQ / AWQ on-disk layout used by
the open-weights quantized model ecosystem.
*/
type Int4Pair uint8

const int4Min = int8(-8)
const int4Max = int8(7)

/*
NewInt4Pair builds a pair from two signed 4-bit values. Values outside
the int4 range [-8, 7] are clamped at the boundary.
*/
func NewInt4Pair(lo, hi int8) Int4Pair {
	return Int4Pair(int4Encode(lo) | (int4Encode(hi) << 4))
}

/*
Lo returns the sign-extended low nibble.
*/
func (pair Int4Pair) Lo() int8 {
	return int4Decode(uint8(pair) & 0x0f)
}

/*
Hi returns the sign-extended high nibble.
*/
func (pair Int4Pair) Hi() int8 {
	return int4Decode((uint8(pair) >> 4) & 0x0f)
}

/*
WithLo returns a new pair with the low nibble replaced by value.
*/
func (pair Int4Pair) WithLo(value int8) Int4Pair {
	return Int4Pair((uint8(pair) & 0xf0) | int4Encode(value))
}

/*
WithHi returns a new pair with the high nibble replaced by value.
*/
func (pair Int4Pair) WithHi(value int8) Int4Pair {
	return Int4Pair((uint8(pair) & 0x0f) | (int4Encode(value) << 4))
}

/*
Bits returns the raw byte representation.
*/
func (pair Int4Pair) Bits() uint8 {
	return uint8(pair)
}

func int4Encode(value int8) uint8 {
	if value < int4Min {
		value = int4Min
	}

	if value > int4Max {
		value = int4Max
	}

	return uint8(value) & 0x0f
}

func int4Decode(nibble uint8) int8 {
	if nibble&0x08 != 0 {
		return int8(nibble) - 16
	}

	return int8(nibble)
}
