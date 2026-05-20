package dtype

import (
	"math"
	"strconv"
)

/*
F32 represents IEEE 754 single-precision floating-point numbers (binary32).
The dtype constant for the same format is dtype.Float32; the helper type
is named F32 to avoid the constant/type collision in Go's flat namespace.
*/
type F32 uint32

// SmallestNonzeroF32 is the smallest positive subnormal float32 (1.4012985e-45).
const SmallestNonzeroF32 = F32(0x00000001)

/*
PrecisionFromfloat64 reports whether converting f64 to float32 is exact,
inexact, underflow, or overflow. Infinity and NaN always report
PrecisionExact even if payload bits are narrowed.
*/
func PrecisionFromfloat64(value float64) Precision {
	bits := math.Float64bits(value)

	if bits == 0 || bits == 0x8000000000000000 {
		return PrecisionExact
	}

	const coefficientMask uint64 = 0x000fffffffffffff
	const exponentShift uint64 = 52
	const exponentBias int32 = 1023
	const exponentMask uint64 = 0x7ff << exponentShift
	const dropMask uint64 = coefficientMask >> 29

	exponent := int32((bits&exponentMask)>>exponentShift) - exponentBias
	coefficient := bits & coefficientMask

	if exponent == 1024 {
		return PrecisionExact
	}

	if exponent < -149 {
		return PrecisionUnderflow
	}

	if exponent > 127 {
		return PrecisionOverflow
	}

	if (coefficient & dropMask) != 0 {
		return PrecisionInexact
	}

	if exponent < -126 {
		return PrecisionUnknown
	}

	return PrecisionExact
}

// F32Frombits returns the float32 number for IEEE 754 binary32 bits u32.
// F32Frombits(x.Bits()) == x.
func F32Frombits(bits uint32) F32 {
	return F32(bits)
}

// F32Fromfloat64 returns F32 converted from value using IEEE round-to-nearest-even.
func F32Fromfloat64(value float64) F32 {
	return F32(math.Float32bits(float32(value)))
}

// ErrInvalidF32NaNValue indicates the input was not an IEEE 754 NaN.
const ErrInvalidF32NaNValue = float32Error("float32: invalid NaN value, expected IEEE 754 NaN")

type float32Error string

func (err float32Error) Error() string {
	return string(err)
}

/*
F32FromNaN64ps converts nan to IEEE binary32 NaN while preserving signaling
vs quiet and as much payload as fits in 23 significand bits.
*/
func F32FromNaN64ps(nan float64) (F32, error) {
	const signalingNaN = F32(0x7f800001)

	bits := math.Float64bits(nan)
	sign := bits & 0x8000000000000000
	exponent := bits & 0x7ff0000000000000
	coefficient := bits & 0x000fffffffffffff

	if exponent != 0x7ff0000000000000 || coefficient == 0 {
		return signalingNaN, ErrInvalidF32NaNValue
	}

	result := uint32((sign >> 32) | 0x7f800000 | (coefficient >> 29))

	if result&0x007fffff == 0 {
		result |= 0x00000001
	}

	return F32(result), nil
}

// F32NaN returns a canonical quiet NaN (0x7fc00001).
func F32NaN() F32 {
	return F32(0x7fc00001)
}

// F32Inf returns positive or negative infinity.
func F32Inf(sign int) F32 {
	if sign >= 0 {
		return F32(0x7f800000)
	}

	return F32(0xff800000)
}

// Float32 returns the float32 value for f. This is lossless.
func (value F32) Float32() float32 {
	return math.Float32frombits(uint32(value))
}

// Float64 returns value widened to float64. This is lossless.
func (value F32) Float64() float64 {
	return float64(value.Float32())
}

// Bits returns the IEEE 754 binary32 representation of f.
func (value F32) Bits() uint32 {
	return uint32(value)
}

// IsNaN reports whether f is a not-a-number value.
func (value F32) IsNaN() bool {
	bits := uint32(value)
	return (bits&0x7f800000 == 0x7f800000) && (bits&0x007fffff != 0)
}

// IsQuietNaN reports whether f is a quiet NaN.
func (value F32) IsQuietNaN() bool {
	bits := uint32(value)
	return value.IsNaN() && (bits&0x00400000 != 0)
}

// IsInf reports whether f is infinity; sign selects positive, negative, or either.
func (value F32) IsInf(sign int) bool {
	bits := uint32(value)
	return ((bits == 0x7f800000) && sign >= 0) ||
		((bits == 0xff800000) && sign <= 0)
}

// IsFinite reports whether f is neither infinite nor NaN.
func (value F32) IsFinite() bool {
	return (uint32(value) & 0x7f800000) != 0x7f800000
}

// IsNormal reports whether f is normal (not zero, subnormal, infinite, or NaN).
func (value F32) IsNormal() bool {
	bits := uint32(value) & 0x7f800000
	return bits != 0x7f800000 && bits != 0
}

// Signbit reports whether f is negative or negative zero.
func (value F32) Signbit() bool {
	return (uint32(value) & 0x80000000) != 0
}

// String satisfies fmt.Stringer.
func (value F32) String() string {
	return strconv.FormatFloat(float64(value.Float32()), 'f', -1, 32)
}
