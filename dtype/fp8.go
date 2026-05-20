package dtype

import "math"

/*
Float8E4M3 is the FP8 variant with 4 exponent bits and 3 mantissa bits.
The encoding follows the OCP FP8 specification (which Nvidia and AMD
both implement on H100/B200 and MI300): no infinity, S.1111.111 is
NaN, exponent bias 7, denormals supported, max finite ~448.

Wire byte order is irrelevant for a one-byte type; serialization is
just the raw byte.

Conversion to and from float32 uses saturating round-to-nearest-even.
Inputs larger than max-finite saturate to ±max-finite. NaN inputs
produce the canonical NaN encoding (0x7f for positive, 0xff for
negative). Subnormal inputs are flushed through the FP8 subnormal
range rather than to zero.
*/
type F8E4M3 uint8

/*
Float8E5M2 is the FP8 variant with 5 exponent bits and 2 mantissa bits.
The encoding follows IEEE 754 conventions (infinity at S.11111.00,
NaN at S.11111.{01,10,11}, exponent bias 15), supporting a wider
range than E4M3 at the cost of mantissa precision.
*/
type F8E5M2 uint8

const (
	fp8E4M3MaxFinite float32 = 448.0
	fp8E5M2MaxFinite float32 = 57344.0
)

/*
Float32 converts an E4M3 value to float32. NaN inputs produce a quiet
NaN; finite values are exact within the representable range.
*/
func (value F8E4M3) Float32() float32 {
	bits := uint8(value)
	sign := uint32(bits>>7) << 31

	if bits&0x7f == 0x7f {
		return math.Float32frombits(sign | 0x7fc00000)
	}

	if bits&0x7f == 0 {
		return math.Float32frombits(sign)
	}

	exponent := uint32((bits >> 3) & 0x0f)
	mantissa := uint32(bits & 0x07)

	if exponent == 0 {
		shift := uint32(0)

		for mantissa&0x08 == 0 {
			mantissa <<= 1
			shift++
		}

		mantissa &= 0x07
		exponent = 1 - shift
	}

	biased := int32(exponent) - 7 + 127
	float32Bits := sign | (uint32(biased) << 23) | (mantissa << 20)

	return math.Float32frombits(float32Bits)
}

/*
NewFloat8E4M3FromFloat32 converts a float32 to E4M3 using
saturating round-to-nearest-even. Overflow saturates to ±max-finite
(0x7e / 0xfe). NaN inputs become the canonical NaN encoding.
*/
func NewF8E4M3FromFloat32(value float32) F8E4M3 {
	if math.IsNaN(float64(value)) {
		return F8E4M3(0x7f)
	}

	sign := uint8(0)

	if math.Signbit(float64(value)) {
		sign = 0x80
		value = -value
	}

	if math.IsInf(float64(value), 1) || value > fp8E4M3MaxFinite {
		return F8E4M3(sign | 0x7e)
	}

	if value == 0 {
		return F8E4M3(sign)
	}

	bits := math.Float32bits(value)
	exponent := int32((bits>>23)&0xff) - 127 + 7
	mantissa := bits & 0x7fffff

	if exponent <= 0 {
		if exponent < -3 {
			return F8E4M3(sign)
		}

		shift := uint32(20 - exponent)
		full := (mantissa | 0x800000) >> shift
		rounded := roundToNearestEven(full, mantissa|0x800000, shift)

		if rounded >= 0x08 {
			return F8E4M3(sign | 0x08)
		}

		return F8E4M3(sign | uint8(rounded))
	}

	if exponent >= 15 {
		return F8E4M3(sign | 0x7e)
	}

	mantissa8 := mantissa >> 20
	rounded := roundToNearestEven(mantissa8, mantissa, 20)

	if rounded >= 0x08 {
		mantissa8 = 0
		exponent++

		if exponent >= 15 {
			return F8E4M3(sign | 0x7e)
		}
	} else {
		mantissa8 = rounded
	}

	return F8E4M3(sign | uint8(exponent)<<3 | uint8(mantissa8))
}

/*
Bits returns the raw 8-bit representation.
*/
func (value F8E4M3) Bits() uint8 {
	return uint8(value)
}

/*
Float32 converts an E5M2 value to float32. The encoding follows
IEEE 754 with bias 15.
*/
func (value F8E5M2) Float32() float32 {
	bits := uint8(value)
	sign := uint32(bits>>7) << 31
	exponent := uint32((bits >> 2) & 0x1f)
	mantissa := uint32(bits & 0x03)

	if exponent == 0x1f {
		if mantissa == 0 {
			return math.Float32frombits(sign | 0x7f800000)
		}

		return math.Float32frombits(sign | 0x7fc00000 | (mantissa << 21))
	}

	if exponent == 0 {
		if mantissa == 0 {
			return math.Float32frombits(sign)
		}

		shift := uint32(0)

		for mantissa&0x04 == 0 {
			mantissa <<= 1
			shift++
		}

		mantissa &= 0x03
		exponent = 1 - shift
	}

	biased := int32(exponent) - 15 + 127
	float32Bits := sign | (uint32(biased) << 23) | (mantissa << 21)

	return math.Float32frombits(float32Bits)
}

/*
NewFloat8E5M2FromFloat32 converts a float32 to E5M2 using
saturating round-to-nearest-even. Overflow saturates to ±infinity
(0x7c / 0xfc).
*/
func NewF8E5M2FromFloat32(value float32) F8E5M2 {
	if math.IsNaN(float64(value)) {
		return F8E5M2(0x7f)
	}

	sign := uint8(0)

	if math.Signbit(float64(value)) {
		sign = 0x80
		value = -value
	}

	if math.IsInf(float64(value), 1) || value > fp8E5M2MaxFinite {
		return F8E5M2(sign | 0x7c)
	}

	if value == 0 {
		return F8E5M2(sign)
	}

	bits := math.Float32bits(value)
	exponent := int32((bits>>23)&0xff) - 127 + 15
	mantissa := bits & 0x7fffff

	if exponent <= 0 {
		if exponent < -1 {
			return F8E5M2(sign)
		}

		shift := uint32(21 - exponent)
		full := (mantissa | 0x800000) >> shift
		rounded := roundToNearestEven(full, mantissa|0x800000, shift)

		if rounded >= 0x04 {
			return F8E5M2(sign | 0x04)
		}

		return F8E5M2(sign | uint8(rounded))
	}

	if exponent >= 0x1f {
		return F8E5M2(sign | 0x7c)
	}

	mantissa8 := mantissa >> 21
	rounded := roundToNearestEven(mantissa8, mantissa, 21)

	if rounded >= 0x04 {
		mantissa8 = 0
		exponent++

		if exponent >= 0x1f {
			return F8E5M2(sign | 0x7c)
		}
	} else {
		mantissa8 = rounded
	}

	return F8E5M2(sign | uint8(exponent)<<2 | uint8(mantissa8))
}

/*
Bits returns the raw 8-bit representation.
*/
func (value F8E5M2) Bits() uint8 {
	return uint8(value)
}

/*
roundToNearestEven implements the IEEE 754 round-to-nearest-even rule
on truncated bits. truncated is the post-shift mantissa, full is the
pre-shift mantissa, shift is the number of dropped bits.
*/
func roundToNearestEven(truncated, full uint32, shift uint32) uint32 {
	if shift == 0 {
		return truncated
	}

	dropped := full & ((uint32(1) << shift) - 1)
	half := uint32(1) << (shift - 1)

	if dropped < half {
		return truncated
	}

	if dropped > half {
		return truncated + 1
	}

	if truncated&1 == 1 {
		return truncated + 1
	}

	return truncated
}
