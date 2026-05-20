/*
Package convert provides correctness-only scalar Go conversions between
the dtypes defined in pkg/dtype. It is the bridge that lets the
legacy float64 kernel path keep compiling while the per-backend
dtype-native rewrite is in flight (Phases 3-6 of
TENSOR_BACKEND_REWRITE.md).

The functions here operate on raw byte slices plus a source DType so
they do not depend on the tensor package. Phase 2 will rewrite the
bodies of these functions to dispatch to the five-host-ISA SIMD
kernels in pkg/backend/compute/convert; the public surface in this
file does not change.

Performance is explicitly NOT a goal of this file. Correctness is.
Every weight load that goes through these functions during the
transitional phases pays a Go-scalar conversion cost. Phase 2 fixes
that without changing call sites.
*/
package convert

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/theapemachine/manifesto/dtype"
)

/*
BytesToFloat64 decodes a byte buffer of the given source dtype into a
slice of float64 values. Supported source dtypes cover every entry of
the dtype enum that has a real numeric interpretation. Bool and
Complex128 are rejected explicitly because their semantics in float64
are ambiguous; callers should request the type-specific helper.
*/
func BytesToFloat64(source dtype.DType, buf []byte) ([]float64, error) {
	elements, err := source.LogicalElements(len(buf))

	if err != nil {
		return nil, err
	}

	out := make([]float64, elements)

	switch source {
	case dtype.Float64:
		return decodeFloat64(buf, out), nil
	case dtype.Float32:
		return decodeFloat32ToFloat64(buf, out), nil
	case dtype.Float16:
		return decodeFloat16ToFloat64(buf, out), nil
	case dtype.BFloat16:
		return decodeBFloat16ToFloat64(buf, out), nil
	case dtype.Float8E4M3:
		return decodeFloat8E4M3ToFloat64(buf, out), nil
	case dtype.Float8E5M2:
		return decodeFloat8E5M2ToFloat64(buf, out), nil
	case dtype.Int64:
		return decodeInt64ToFloat64(buf, out), nil
	case dtype.Int32:
		return decodeInt32ToFloat64(buf, out), nil
	case dtype.Int16:
		return decodeInt16ToFloat64(buf, out), nil
	case dtype.Int8:
		return decodeInt8ToFloat64(buf, out), nil
	case dtype.Uint64:
		return decodeUint64ToFloat64(buf, out), nil
	case dtype.Uint32:
		return decodeUint32ToFloat64(buf, out), nil
	case dtype.Uint16:
		return decodeUint16ToFloat64(buf, out), nil
	case dtype.Uint8:
		return decodeUint8ToFloat64(buf, out), nil
	case dtype.Int4:
		return decodeInt4ToFloat64(buf, out), nil
	}

	return nil, fmt.Errorf("convert: unsupported source dtype %s for float64 target", source)
}

/*
BytesToFloat32 decodes a byte buffer of the given source dtype into a
slice of float32 values. Same coverage as BytesToFloat64.
*/
func BytesToFloat32(source dtype.DType, buf []byte) ([]float32, error) {
	elements, err := source.LogicalElements(len(buf))

	if err != nil {
		return nil, err
	}

	out := make([]float32, elements)

	switch source {
	case dtype.Float64:
		for index := range out {
			out[index] = float32(math.Float64frombits(binary.LittleEndian.Uint64(buf[index*8:])))
		}

		return out, nil
	case dtype.Float32:
		for index := range out {
			out[index] = math.Float32frombits(binary.LittleEndian.Uint32(buf[index*4:]))
		}

		return out, nil
	case dtype.Float16:
		for index := range out {
			value := dtype.Frombits(binary.LittleEndian.Uint16(buf[index*2:]))
			out[index] = value.Float32()
		}

		return out, nil
	case dtype.BFloat16:
		for index := range out {
			value := dtype.NewBfloat16FromBytes(buf[index*2:])
			out[index] = (&value).Float32()
		}

		return out, nil
	case dtype.Float8E4M3:
		for index := range out {
			out[index] = dtype.F8E4M3(buf[index]).Float32()
		}

		return out, nil
	case dtype.Float8E5M2:
		for index := range out {
			out[index] = dtype.F8E5M2(buf[index]).Float32()
		}

		return out, nil
	}

	float64s, err := BytesToFloat64(source, buf)

	if err != nil {
		return nil, err
	}

	for index, value := range float64s {
		out[index] = float32(value)
	}

	return out, nil
}

/*
BytesToBFloat16 converts a byte buffer of the given source dtype into
a slice of BF16 values. Conversion goes through float32 with
truncation rounding (the canonical BF16 hardware behaviour).
*/
func BytesToBFloat16(source dtype.DType, buf []byte) ([]dtype.BF16, error) {
	float32s, err := BytesToFloat32(source, buf)

	if err != nil {
		return nil, err
	}

	out := make([]dtype.BF16, len(float32s))

	for index, value := range float32s {
		out[index] = dtype.NewBfloat16FromFloat32(value)
	}

	return out, nil
}

/*
BytesToFloat16 converts a byte buffer to IEEE 754 binary16 values.
*/
func BytesToFloat16(source dtype.DType, buf []byte) ([]dtype.F16, error) {
	float32s, err := BytesToFloat32(source, buf)

	if err != nil {
		return nil, err
	}

	out := make([]dtype.F16, len(float32s))

	for index, value := range float32s {
		out[index] = dtype.Fromfloat32(value)
	}

	return out, nil
}

/*
BytesToFloat8E4M3 converts a byte buffer to FP8 E4M3 values using
saturating round-to-nearest-even.
*/
func BytesToFloat8E4M3(source dtype.DType, buf []byte) ([]dtype.F8E4M3, error) {
	float32s, err := BytesToFloat32(source, buf)

	if err != nil {
		return nil, err
	}

	out := make([]dtype.F8E4M3, len(float32s))

	for index, value := range float32s {
		out[index] = dtype.NewF8E4M3FromFloat32(value)
	}

	return out, nil
}

/*
BytesToFloat8E5M2 converts a byte buffer to FP8 E5M2 values using
saturating round-to-nearest-even.
*/
func BytesToFloat8E5M2(source dtype.DType, buf []byte) ([]dtype.F8E5M2, error) {
	float32s, err := BytesToFloat32(source, buf)

	if err != nil {
		return nil, err
	}

	out := make([]dtype.F8E5M2, len(float32s))

	for index, value := range float32s {
		out[index] = dtype.NewF8E5M2FromFloat32(value)
	}

	return out, nil
}

/*
BytesToInt8 truncates floating-point and wider-integer inputs to int8.
Truncation is the standard "saturating cast" behaviour: values outside
the int8 range clamp to math.MinInt8 / math.MaxInt8.
*/
func BytesToInt8(source dtype.DType, buf []byte) ([]int8, error) {
	float64s, err := BytesToFloat64(source, buf)

	if err != nil {
		return nil, err
	}

	out := make([]int8, len(float64s))

	for index, value := range float64s {
		if value < float64(math.MinInt8) {
			out[index] = math.MinInt8
			continue
		}

		if value > float64(math.MaxInt8) {
			out[index] = math.MaxInt8
			continue
		}

		out[index] = int8(value)
	}

	return out, nil
}

/*
Float64ToBytes encodes float64 values into a tightly packed
little-endian byte slice.
*/
func Float64ToBytes(values []float64) []byte {
	out := make([]byte, len(values)*8)

	for index, value := range values {
		binary.LittleEndian.PutUint64(out[index*8:], math.Float64bits(value))
	}

	return out
}

/*
Float32ToBytes encodes float32 values into a tightly packed
little-endian byte slice.
*/
func Float32ToBytes(values []float32) []byte {
	out := make([]byte, len(values)*4)

	for index, value := range values {
		binary.LittleEndian.PutUint32(out[index*4:], math.Float32bits(value))
	}

	return out
}

/*
BFloat16ToBytes encodes BF16 values into a tightly packed
little-endian byte slice.
*/
func BFloat16ToBytes(values []dtype.BF16) []byte {
	out := make([]byte, len(values)*2)

	for index, value := range values {
		binary.LittleEndian.PutUint16(out[index*2:], value.Bits())
	}

	return out
}

/*
Float16ToBytes encodes binary16 values into a tightly packed
little-endian byte slice.
*/
func Float16ToBytes(values []dtype.F16) []byte {
	out := make([]byte, len(values)*2)

	for index, value := range values {
		binary.LittleEndian.PutUint16(out[index*2:], value.Bits())
	}

	return out
}

/*
Float8E4M3ToBytes encodes E4M3 values into a tightly packed byte slice.
*/
func Float8E4M3ToBytes(values []dtype.F8E4M3) []byte {
	out := make([]byte, len(values))

	for index, value := range values {
		out[index] = value.Bits()
	}

	return out
}

/*
Float8E5M2ToBytes encodes E5M2 values into a tightly packed byte slice.
*/
func Float8E5M2ToBytes(values []dtype.F8E5M2) []byte {
	out := make([]byte, len(values))

	for index, value := range values {
		out[index] = value.Bits()
	}

	return out
}
