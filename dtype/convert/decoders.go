package convert

import (
	"encoding/binary"
	"math"

	"github.com/theapemachine/manifesto/dtype"
)

// Each decoder asserts source-buffer adequacy once before the loop and
// slices the input down to exactly the bytes the loop consumes. That
// gives the Go compiler a single pre-loop range it can prove every
// indexed access against, so the per-iteration bounds checks fall out.
//
// The transient-scalar pattern `value := dtype.Foo(...); (&value).Float32()`
// is also avoided: pointer-receiver methods on a stack scalar force the
// value to escape to the heap. Replaced with direct bit manipulation so
// the loop body stays allocation-free.

const (
	bytesPerUint16 = 2
	bytesPerUint32 = 4
	bytesPerUint64 = 8
)

func decodeFloat64(buf []byte, out []float64) []float64 {
	required := len(out) * bytesPerUint64

	if len(buf) < required {
		panic("convert: buffer too short for float64 decode")
	}

	src := buf[:required]

	for index := range out {
		out[index] = math.Float64frombits(
			binary.LittleEndian.Uint64(src[index*bytesPerUint64 : index*bytesPerUint64+bytesPerUint64]),
		)
	}

	return out
}

func decodeFloat32ToFloat64(buf []byte, out []float64) []float64 {
	required := len(out) * bytesPerUint32

	if len(buf) < required {
		panic("convert: buffer too short for float32 decode")
	}

	src := buf[:required]

	for index := range out {
		bits := binary.LittleEndian.Uint32(src[index*bytesPerUint32 : index*bytesPerUint32+bytesPerUint32])
		out[index] = float64(math.Float32frombits(bits))
	}

	return out
}

func decodeFloat16ToFloat64(buf []byte, out []float64) []float64 {
	required := len(out) * bytesPerUint16

	if len(buf) < required {
		panic("convert: buffer too short for float16 decode")
	}

	src := buf[:required]

	for index := range out {
		bits := binary.LittleEndian.Uint16(src[index*bytesPerUint16 : index*bytesPerUint16+bytesPerUint16])
		out[index] = float64(dtype.Frombits(bits).Float32())
	}

	return out
}

func decodeBFloat16ToFloat64(buf []byte, out []float64) []float64 {
	required := len(out) * bytesPerUint16

	if len(buf) < required {
		panic("convert: buffer too short for bfloat16 decode")
	}

	src := buf[:required]

	for index := range out {
		// BF16 lossless-widens to float32 by zero-padding the low 16 bits.
		// Skipping NewBfloat16FromBytes + (&value).Float32() — both stack
		// roundtrips — keeps the value in a register.
		bits := uint32(binary.LittleEndian.Uint16(src[index*bytesPerUint16:index*bytesPerUint16+bytesPerUint16])) << 16
		out[index] = float64(math.Float32frombits(bits))
	}

	return out
}

func decodeFloat8E4M3ToFloat64(buf []byte, out []float64) []float64 {
	if len(buf) < len(out) {
		panic("convert: buffer too short for fp8e4m3 decode")
	}

	src := buf[:len(out)]

	for index, raw := range src {
		out[index] = float64(dtype.F8E4M3(raw).Float32())
	}

	return out
}

func decodeFloat8E5M2ToFloat64(buf []byte, out []float64) []float64 {
	if len(buf) < len(out) {
		panic("convert: buffer too short for fp8e5m2 decode")
	}

	src := buf[:len(out)]

	for index, raw := range src {
		out[index] = float64(dtype.F8E5M2(raw).Float32())
	}

	return out
}

func decodeInt64ToFloat64(buf []byte, out []float64) []float64 {
	required := len(out) * bytesPerUint64

	if len(buf) < required {
		panic("convert: buffer too short for int64 decode")
	}

	src := buf[:required]

	for index := range out {
		raw := binary.LittleEndian.Uint64(src[index*bytesPerUint64 : index*bytesPerUint64+bytesPerUint64])
		out[index] = float64(int64(raw))
	}

	return out
}

func decodeInt32ToFloat64(buf []byte, out []float64) []float64 {
	required := len(out) * bytesPerUint32

	if len(buf) < required {
		panic("convert: buffer too short for int32 decode")
	}

	src := buf[:required]

	for index := range out {
		raw := binary.LittleEndian.Uint32(src[index*bytesPerUint32 : index*bytesPerUint32+bytesPerUint32])
		out[index] = float64(int32(raw))
	}

	return out
}

func decodeInt16ToFloat64(buf []byte, out []float64) []float64 {
	required := len(out) * bytesPerUint16

	if len(buf) < required {
		panic("convert: buffer too short for int16 decode")
	}

	src := buf[:required]

	for index := range out {
		raw := binary.LittleEndian.Uint16(src[index*bytesPerUint16 : index*bytesPerUint16+bytesPerUint16])
		out[index] = float64(int16(raw))
	}

	return out
}

func decodeInt8ToFloat64(buf []byte, out []float64) []float64 {
	if len(buf) < len(out) {
		panic("convert: buffer too short for int8 decode")
	}

	src := buf[:len(out)]

	for index, raw := range src {
		out[index] = float64(int8(raw))
	}

	return out
}

func decodeUint64ToFloat64(buf []byte, out []float64) []float64 {
	required := len(out) * bytesPerUint64

	if len(buf) < required {
		panic("convert: buffer too short for uint64 decode")
	}

	src := buf[:required]

	for index := range out {
		out[index] = float64(binary.LittleEndian.Uint64(src[index*bytesPerUint64 : index*bytesPerUint64+bytesPerUint64]))
	}

	return out
}

func decodeUint32ToFloat64(buf []byte, out []float64) []float64 {
	required := len(out) * bytesPerUint32

	if len(buf) < required {
		panic("convert: buffer too short for uint32 decode")
	}

	src := buf[:required]

	for index := range out {
		out[index] = float64(binary.LittleEndian.Uint32(src[index*bytesPerUint32 : index*bytesPerUint32+bytesPerUint32]))
	}

	return out
}

func decodeUint16ToFloat64(buf []byte, out []float64) []float64 {
	required := len(out) * bytesPerUint16

	if len(buf) < required {
		panic("convert: buffer too short for uint16 decode")
	}

	src := buf[:required]

	for index := range out {
		out[index] = float64(binary.LittleEndian.Uint16(src[index*bytesPerUint16 : index*bytesPerUint16+bytesPerUint16]))
	}

	return out
}

func decodeUint8ToFloat64(buf []byte, out []float64) []float64 {
	if len(buf) < len(out) {
		panic("convert: buffer too short for uint8 decode")
	}

	src := buf[:len(out)]

	for index, raw := range src {
		out[index] = float64(raw)
	}

	return out
}

// decodeInt4ToFloat64 iterates the source byte slice once and writes both
// nibbles per pass. The previous index/2, index%2 form did an integer
// division and a modulo on every iteration and re-read the same source
// byte for adjacent elements; this rewrite eliminates the division, the
// modulo, and the duplicate load.
//
// Low nibble = element at even index. High nibble = element at odd index.
// Matches dtype/int4.go::Int4Pair (Lo == bits 0-3, Hi == bits 4-7), which
// is also the convention the GPU `int4_dequant` kernel reads.
func decodeInt4ToFloat64(buf []byte, out []float64) []float64 {
	required := (len(out) + 1) / 2

	if len(buf) < required {
		panic("convert: buffer too short for int4 decode")
	}

	src := buf[:required]

	for byteIndex, packed := range src {
		outIndex := byteIndex * 2

		if outIndex >= len(out) {
			break
		}

		// Sign-extend each nibble by placing it in the high half of an
		// int8 and arithmetic-right-shifting. int8 >> is sign-preserving
		// in Go, so 0xf0 (-16) >> 4 = -1; 0x70 (+112) >> 4 = +7.
		out[outIndex] = float64(int8(packed<<4) >> 4)

		if outIndex+1 < len(out) {
			out[outIndex+1] = float64(int8(packed) >> 4)
		}
	}

	return out
}
