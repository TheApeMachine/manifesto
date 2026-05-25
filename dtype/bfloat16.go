package dtype

import (
	"encoding/binary"
	"math"
)

/*
BF16 is the Google Brain 16-bit floating-point format: 1 sign bit, 8
exponent bits matching float32, 7 mantissa bits. Conversion to and from
float32 is a high-half shift; rounding from float32 follows
round-to-nearest-even per the AVX-512, ARM NEON, and Apple Silicon
hardware narrowing convention so the Go scalar path stays bit-for-bit
parity with on-device casts.

Wire byte order is little-endian everywhere, matching safetensors, GGUF,
and every hardware target.
*/
type BF16 uint16

/*
NewBfloat16FromFloat32 converts a float32 to BF16 using round-to-nearest
with ties to even (RNE). RNE is the narrowing rule mandated by IEEE 754
and implemented by every BF16 hardware cast we ship against (AVX-512 BF
extensions, NEON BFCVT, Apple AMX bfdot/bfmla). Pure truncation — the
prior implementation — introduces a systematic negative bias that
compounds across deep networks; RNE matches the hardware so the CPU and
the GPU produce identical bits for the same input.

Algorithm: add 0x7fff (a half-ULP at the truncation point), then add the
low bit of the post-shift mantissa (the "to-even" tiebreaker). Truncate
to the high 16 bits. NaN and Inf propagate naturally because their
mantissa bits are preserved through the round.

The float32 special-value bit pattern for NaN survives RNE intact (we
preserve the high bit of the mantissa, so quiet NaNs stay quiet); +Inf /
-Inf likewise. Subnormals in float32 flush to zero in BF16 because the
narrowed exponent has the same range as float32 — there is no subnormal
gap to handle.
*/
func NewBfloat16FromFloat32(value float32) BF16 {
	bits := math.Float32bits(value)

	// Preserve NaN: any float32 NaN has exponent == 0xff and a non-zero
	// mantissa. RNE rounding can flip a quiet NaN to a signalling NaN by
	// touching the high mantissa bit, so we short-circuit and keep the
	// canonical quiet NaN representation of the high 16 bits.
	if bits&0x7fffffff > 0x7f800000 {
		return BF16(bits>>16) | 0x0040
	}

	rounded := bits + 0x7fff + ((bits >> 16) & 1)

	return BF16(rounded >> 16)
}

/*
NewBfloat16FromBytes reads a little-endian BF16 from the first two bytes
of buf. Caller must guarantee len(buf) >= 2.
*/
func NewBfloat16FromBytes(buf []byte) BF16 {
	return BF16(binary.LittleEndian.Uint16(buf[:2]))
}

/*
Bytes returns the little-endian two-byte encoding of the BF16 value.
Value receiver — the method does not mutate, and a value receiver lets
callers use both `bf16.Bytes()` and `(&bf16).Bytes()` without surprise.
*/
func (bf16 BF16) Bytes() []byte {
	var buf [2]byte
	binary.LittleEndian.PutUint16(buf[:], uint16(bf16))

	return buf[:]
}

/*
Float32 returns the float32 value represented by the BF16 by zero-padding
the lower 16 mantissa bits. The conversion is lossless from BF16 to
float32 because BF16 is structurally the high half of a float32.
*/
func (bf16 BF16) Float32() float32 {
	return math.Float32frombits(uint32(bf16) << 16)
}

/*
Bits returns the raw 16-bit representation of the BF16 value.
*/
func (bf16 BF16) Bits() uint16 {
	return uint16(bf16)
}

/*
DecodeBF16 decodes a tightly packed little-endian BF16 byte slice into a
slice of BF16 values. The output length is len(buf)/2; trailing bytes
that do not form a complete BF16 are dropped.

Package-level function rather than a method because the operation is
slice-codec, not state-bound: requiring callers to instantiate or
nil-cast a BF16 just to invoke this was an API wart called out in the
dtype review.
*/
func DecodeBF16(buf []byte) []BF16 {
	pairs := len(buf) / 2
	if pairs == 0 {
		return nil
	}

	out := make([]BF16, pairs)

	// Slice the input down to exactly the bytes we consume so the
	// compiler can elide the per-iteration bounds check on buf.
	src := buf[:pairs*2]

	for index := range out {
		out[index] = BF16(binary.LittleEndian.Uint16(src[index*2 : index*2+2]))
	}

	return out
}

/*
EncodeBF16 encodes a slice of BF16 values into a tightly packed
little-endian byte slice of length 2*len(values).
*/
func EncodeBF16(values []BF16) []byte {
	if len(values) == 0 {
		return nil
	}

	out := make([]byte, len(values)*2)

	for index, value := range values {
		binary.LittleEndian.PutUint16(out[index*2:index*2+2], uint16(value))
	}

	return out
}

/*
DecodeBF16ToFloat32 decodes a little-endian BF16 byte slice into a slice
of float32 values. Pre-sizes the bounds check so the inner loop is
allocation- and branch-free per element.
*/
func DecodeBF16ToFloat32(buf []byte) []float32 {
	pairs := len(buf) / 2
	if pairs == 0 {
		return nil
	}

	out := make([]float32, pairs)

	src := buf[:pairs*2]

	for index := range out {
		bits := uint32(binary.LittleEndian.Uint16(src[index*2:index*2+2])) << 16
		out[index] = math.Float32frombits(bits)
	}

	return out
}

/*
EncodeFloat32ToBF16 encodes a slice of float32 values into a tightly
packed little-endian BF16 byte slice. Uses RNE via NewBfloat16FromFloat32
so the bytes round-trip identically with the hardware cast path.
*/
func EncodeFloat32ToBF16(values []float32) []byte {
	if len(values) == 0 {
		return nil
	}

	out := make([]byte, len(values)*2)

	for index, value := range values {
		converted := NewBfloat16FromFloat32(value)
		binary.LittleEndian.PutUint16(out[index*2:index*2+2], uint16(converted))
	}

	return out
}

// Deprecated method shims. They forward to the package-level functions so
// existing callers in puter/manifesto keep compiling while the migration
// rolls through the rest of the tree. New code should use the package
// functions directly — these will be removed once every caller in the
// dependency graph is updated.

/*
Decode is a deprecated wrapper around DecodeBF16. Prefer the package-level
function.
*/
func (bf16 *BF16) Decode(buf []byte) []BF16 {
	_ = bf16
	return DecodeBF16(buf)
}

/*
Encode is a deprecated wrapper around EncodeBF16. Prefer the package-level
function.
*/
func (bf16 *BF16) Encode(values []BF16) []byte {
	_ = bf16
	return EncodeBF16(values)
}

/*
DecodeFloat32 is a deprecated wrapper around DecodeBF16ToFloat32. Prefer
the package-level function.
*/
func (bf16 *BF16) DecodeFloat32(buf []byte) []float32 {
	_ = bf16
	return DecodeBF16ToFloat32(buf)
}

/*
EncodeFloat32 is a deprecated wrapper around EncodeFloat32ToBF16. Prefer
the package-level function.
*/
func (bf16 *BF16) EncodeFloat32(values []float32) []byte {
	_ = bf16
	return EncodeFloat32ToBF16(values)
}
