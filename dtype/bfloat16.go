package dtype

import (
	"encoding/binary"
	"math"
)

/*
BF16 is the Google Brain 16-bit floating-point format: 1 sign bit, 8
exponent bits matching float32, 7 mantissa bits. Conversion to and from
float32 is a simple high-half / sign-extend operation.

Wire byte order is little-endian everywhere, matching safetensors,
GGUF, and every hardware target. The original Ollama-derived
big-endian methods were corrected here; if you need an internal
Ollama-compat path with big-endian framing, add it as a distinct named
function so the default wire format stays correct.
*/
type BF16 uint16

/*
NewBfloat16FromFloat32 converts a float32 to BF16 by truncating the
low 16 bits of the IEEE 754 binary32 representation. The conversion
does not round; this matches the hardware behaviour on every target
that supports a BF16 cast intrinsic.
*/
func NewBfloat16FromFloat32(value float32) BF16 {
	return BF16(math.Float32bits(value) >> 16)
}

/*
NewBfloat16FromBytes reads a little-endian BF16 from the first two
bytes of buf. Caller must guarantee len(buf) >= 2.
*/
func NewBfloat16FromBytes(buf []byte) BF16 {
	return BF16(binary.LittleEndian.Uint16(buf[:2]))
}

/*
Bytes returns the little-endian two-byte encoding of the BF16 value.
*/
func (bf16 *BF16) Bytes() []byte {
	var buf [2]byte
	binary.LittleEndian.PutUint16(buf[:], uint16(*bf16))

	return buf[:]
}

/*
Decode decodes a tightly packed little-endian BF16 byte slice into a
slice of BF16 values. The output length is len(buf)/2; trailing bytes
that do not form a complete BF16 are dropped.
*/
func (bf16 *BF16) Decode(buf []byte) []BF16 {
	pairs := len(buf) / 2
	out := make([]BF16, pairs)

	for index := range out {
		out[index] = NewBfloat16FromBytes(buf[index*2:])
	}

	return out
}

/*
Encode encodes a slice of BF16 values into a tightly packed
little-endian byte slice of length 2*len(values).
*/
func (bf16 *BF16) Encode(values []BF16) []byte {
	out := make([]byte, len(values)*2)

	for index, value := range values {
		binary.LittleEndian.PutUint16(out[index*2:], uint16(value))
	}

	return out
}

/*
DecodeFloat32 decodes a little-endian BF16 byte slice into a slice of
float32 values.
*/
func (bf16 *BF16) DecodeFloat32(buf []byte) []float32 {
	pairs := len(buf) / 2
	out := make([]float32, pairs)

	for index := range out {
		value := NewBfloat16FromBytes(buf[index*2:])
		out[index] = (&value).Float32()
	}

	return out
}

/*
EncodeFloat32 encodes a slice of float32 values into a tightly packed
little-endian BF16 byte slice. Truncation rounding only; no
round-to-nearest-even.
*/
func (bf16 *BF16) EncodeFloat32(values []float32) []byte {
	out := make([]byte, len(values)*2)

	for index, value := range values {
		converted := NewBfloat16FromFloat32(value)
		binary.LittleEndian.PutUint16(out[index*2:], uint16(converted))
	}

	return out
}

/*
Float32 returns the float32 value represented by the BF16, by shifting
the BF16 bits into the high 16 bits of a float32 representation. The
conversion is lossless from BF16 to float32.
*/
func (bf16 *BF16) Float32() float32 {
	return math.Float32frombits(uint32(*bf16) << 16)
}

/*
Bits returns the raw 16-bit representation of the BF16 value. Pointer
receiver matches Float32 and the other query methods on BF16.
*/
func (bf16 *BF16) Bits() uint16 {
	return uint16(*bf16)
}
