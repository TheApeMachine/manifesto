package dtype

import (
	gomath "math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestPrecisionFromfloat32(test *testing.T) {
	Convey("Given float32 values", test, func() {
		Convey("It should classify exact finite values", func() {
			So(PrecisionFromfloat32(1.0), ShouldEqual, PrecisionExact)
			So(PrecisionFromfloat32(0.0), ShouldEqual, PrecisionExact)
			So(PrecisionFromfloat32(float32(gomath.Inf(1))), ShouldEqual, PrecisionExact)
		})

		Convey("It should classify values outside binary16 range", func() {
			So(PrecisionFromfloat32(gomath.MaxFloat32), ShouldEqual, PrecisionOverflow)
			So(PrecisionFromfloat32(gomath.SmallestNonzeroFloat32), ShouldEqual, PrecisionUnderflow)
		})

		Convey("It should classify dropped significand bits", func() {
			So(PrecisionFromfloat32(1.0001), ShouldEqual, PrecisionInexact)
		})
	})
}

func TestFrombits(test *testing.T) {
	Convey("Given IEEE binary16 bits", test, func() {
		float16 := Frombits(0x3c00)

		Convey("It should decode the value", func() {
			So(float16.Bits(), ShouldEqual, uint16(0x3c00))
			So(float16.Float32(), ShouldEqual, float32(1.0))
		})
	})
}

func TestFromfloat32(test *testing.T) {
	Convey("Given a float32 value", test, func() {
		float16 := Fromfloat32(1.0)

		Convey("It should encode the IEEE binary16 value", func() {
			So(float16.Bits(), ShouldEqual, uint16(0x3c00))
			So(float16.Float32(), ShouldEqual, float32(1.0))
		})
	})
}

func TestFromNaN32ps(test *testing.T) {
	Convey("Given a valid IEEE float32 NaN", test, func() {
		float16, err := FromNaN32ps(gomath.Float32frombits(0x7fc02000))

		Convey("It should preserve a NaN payload", func() {
			So(err, ShouldBeNil)
			So(float16, ShouldEqual, NaN())
			So(float16.IsNaN(), ShouldBeTrue)
		})
	})

	Convey("Given a non-NaN float32 value", test, func() {
		float16, err := FromNaN32ps(1.0)

		Convey("It should reject the input", func() {
			So(err, ShouldEqual, ErrInvalidNaNValue)
			So(float16, ShouldEqual, F16(0x7c01))
		})
	})
}

func TestNaN(test *testing.T) {
	Convey("Given the NaN constructor", test, func() {
		float16 := NaN()

		Convey("It should return a quiet NaN", func() {
			So(float16.Bits(), ShouldEqual, uint16(0x7e01))
			So(float16.IsNaN(), ShouldBeTrue)
			So(float16.IsQuietNaN(), ShouldBeTrue)
		})
	})
}

func TestInf(test *testing.T) {
	Convey("Given the infinity constructor", test, func() {
		positive := Inf(1)
		negative := Inf(-1)

		Convey("It should encode positive and negative infinity", func() {
			So(positive.Bits(), ShouldEqual, uint16(0x7c00))
			So(negative.Bits(), ShouldEqual, uint16(0xfc00))
			So(positive.IsInf(1), ShouldBeTrue)
			So(negative.IsInf(-1), ShouldBeTrue)
		})
	})
}

func TestFloat16_Float32(test *testing.T) {
	Convey("Given a F16 value", test, func() {
		float16 := Frombits(0x4200)

		Convey("It should convert to float32", func() {
			So(float16.Float32(), ShouldEqual, float32(3.0))
		})
	})
}

func TestFloat16_Bits(test *testing.T) {
	Convey("Given a F16 value", test, func() {
		float16 := Fromfloat32(-2.0)

		Convey("It should expose the raw bits", func() {
			So(float16.Bits(), ShouldEqual, uint16(0xc000))
		})
	})
}

func TestFloat16_IsNaN(test *testing.T) {
	Convey("Given F16 values", test, func() {
		Convey("It should detect NaN values", func() {
			So(NaN().IsNaN(), ShouldBeTrue)
			So(Fromfloat32(1.0).IsNaN(), ShouldBeFalse)
		})
	})
}

func TestFloat16_IsQuietNaN(test *testing.T) {
	Convey("Given F16 NaN values", test, func() {
		quiet := F16(0x7e01)
		signaling := F16(0x7c01)

		Convey("It should distinguish quiet NaN values", func() {
			So(quiet.IsQuietNaN(), ShouldBeTrue)
			So(signaling.IsQuietNaN(), ShouldBeFalse)
		})
	})
}

func TestFloat16_IsInf(test *testing.T) {
	Convey("Given F16 values", test, func() {
		Convey("It should detect infinity by sign", func() {
			So(Inf(1).IsInf(0), ShouldBeTrue)
			So(Inf(-1).IsInf(0), ShouldBeTrue)
			So(Inf(1).IsInf(-1), ShouldBeFalse)
			So(Fromfloat32(1.0).IsInf(0), ShouldBeFalse)
		})
	})
}

func TestFloat16_IsFinite(test *testing.T) {
	Convey("Given F16 values", test, func() {
		Convey("It should reject infinite and NaN values", func() {
			So(Fromfloat32(1.0).IsFinite(), ShouldBeTrue)
			So(Inf(1).IsFinite(), ShouldBeFalse)
			So(NaN().IsFinite(), ShouldBeFalse)
		})
	})
}

func TestFloat16_IsNormal(test *testing.T) {
	Convey("Given F16 values", test, func() {
		Convey("It should detect normal finite values", func() {
			So(Fromfloat32(1.0).IsNormal(), ShouldBeTrue)
			So(Frombits(0).IsNormal(), ShouldBeFalse)
			So(SmallestNonzero.IsNormal(), ShouldBeFalse)
			So(Inf(1).IsNormal(), ShouldBeFalse)
		})
	})
}

func TestFloat16_Signbit(test *testing.T) {
	Convey("Given positive and negative F16 values", test, func() {
		Convey("It should report the sign bit", func() {
			So(Fromfloat32(-1.0).Signbit(), ShouldBeTrue)
			So(Fromfloat32(1.0).Signbit(), ShouldBeFalse)
			So(Frombits(0x8000).Signbit(), ShouldBeTrue)
		})
	})
}

func TestFloat16_String(test *testing.T) {
	Convey("Given a F16 value", test, func() {
		float16 := Fromfloat32(1.5)

		Convey("It should format the float32 value", func() {
			So(float16.String(), ShouldEqual, "1.5")
		})
	})
}

func BenchmarkPrecisionFromfloat32(benchmark *testing.B) {
	for benchmark.Loop() {
		_ = PrecisionFromfloat32(1.0)
	}
}

func BenchmarkFrombits(benchmark *testing.B) {
	for benchmark.Loop() {
		_ = Frombits(0x3c00)
	}
}

func BenchmarkFromfloat32(benchmark *testing.B) {
	for benchmark.Loop() {
		_ = Fromfloat32(1.0)
	}
}

func BenchmarkFromNaN32ps(benchmark *testing.B) {
	nan := gomath.Float32frombits(0x7fc02000)

	for benchmark.Loop() {
		_, _ = FromNaN32ps(nan)
	}
}

func BenchmarkNaN(benchmark *testing.B) {
	for benchmark.Loop() {
		_ = NaN()
	}
}

func BenchmarkInf(benchmark *testing.B) {
	for benchmark.Loop() {
		_ = Inf(1)
	}
}

func BenchmarkFloat16_Float32(benchmark *testing.B) {
	float16 := Fromfloat32(1.0)

	for benchmark.Loop() {
		_ = float16.Float32()
	}
}

func BenchmarkFloat16_Bits(benchmark *testing.B) {
	float16 := Fromfloat32(1.0)

	for benchmark.Loop() {
		_ = float16.Bits()
	}
}

func BenchmarkFloat16_IsNaN(benchmark *testing.B) {
	float16 := NaN()

	for benchmark.Loop() {
		_ = float16.IsNaN()
	}
}

func BenchmarkFloat16_IsQuietNaN(benchmark *testing.B) {
	float16 := NaN()

	for benchmark.Loop() {
		_ = float16.IsQuietNaN()
	}
}

func BenchmarkFloat16_IsInf(benchmark *testing.B) {
	float16 := Inf(1)

	for benchmark.Loop() {
		_ = float16.IsInf(1)
	}
}

func BenchmarkFloat16_IsFinite(benchmark *testing.B) {
	float16 := Fromfloat32(1.0)

	for benchmark.Loop() {
		_ = float16.IsFinite()
	}
}

func BenchmarkFloat16_IsNormal(benchmark *testing.B) {
	float16 := Fromfloat32(1.0)

	for benchmark.Loop() {
		_ = float16.IsNormal()
	}
}

func BenchmarkFloat16_Signbit(benchmark *testing.B) {
	float16 := Fromfloat32(-1.0)

	for benchmark.Loop() {
		_ = float16.Signbit()
	}
}

func BenchmarkFloat16_String(benchmark *testing.B) {
	float16 := Fromfloat32(1.5)

	for benchmark.Loop() {
		_ = float16.String()
	}
}
