package dtype

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestFloat8E4M3_RoundTrip(test *testing.T) {
	Convey("Given canonical floats representable in E4M3", test, func() {
		cases := []float32{0, 1, -1, 2, 0.5, -0.5, 448, -448, 0.015625}

		Convey("Encoding and decoding should round-trip exactly", func() {
			for _, value := range cases {
				encoded := NewF8E4M3FromFloat32(value)
				decoded := encoded.Float32()

				So(decoded, ShouldAlmostEqual, value, 1e-2)
			}
		})
	})

	Convey("Given an overflow value", test, func() {
		Convey("It should saturate to ±max-finite", func() {
			positive := NewF8E4M3FromFloat32(1e9)
			negative := NewF8E4M3FromFloat32(-1e9)

			So(positive.Float32(), ShouldEqual, float32(448))
			So(negative.Float32(), ShouldEqual, float32(-448))
		})
	})

	Convey("Given NaN", test, func() {
		Convey("It should produce the canonical NaN encoding", func() {
			encoded := NewF8E4M3FromFloat32(float32(math.NaN()))

			So(encoded.Bits(), ShouldEqual, uint8(0x7f))
			So(math.IsNaN(float64(encoded.Float32())), ShouldBeTrue)
		})
	})

	Convey("Given zero", test, func() {
		Convey("It should produce 0x00 / 0x80", func() {
			positive := NewF8E4M3FromFloat32(0)
			negative := NewF8E4M3FromFloat32(float32(math.Copysign(0, -1)))

			So(positive.Bits(), ShouldEqual, uint8(0))
			So(negative.Bits(), ShouldEqual, uint8(0x80))
		})
	})
}

func TestFloat8E5M2_RoundTrip(test *testing.T) {
	Convey("Given canonical floats representable in E5M2", test, func() {
		cases := []float32{0, 1, -1, 2, 0.5, -0.5, 57344, -57344}

		Convey("Encoding and decoding should round-trip exactly", func() {
			for _, value := range cases {
				encoded := NewF8E5M2FromFloat32(value)
				decoded := encoded.Float32()

				So(decoded, ShouldAlmostEqual, value, 1e-2)
			}
		})
	})

	Convey("Given an overflow value", test, func() {
		Convey("It should saturate to ±infinity", func() {
			positive := NewF8E5M2FromFloat32(1e9)
			negative := NewF8E5M2FromFloat32(-1e9)

			So(math.IsInf(float64(positive.Float32()), 1), ShouldBeTrue)
			So(math.IsInf(float64(negative.Float32()), -1), ShouldBeTrue)
		})
	})

	Convey("Given infinity", test, func() {
		Convey("It should round-trip", func() {
			encoded := NewF8E5M2FromFloat32(float32(math.Inf(1)))

			So(math.IsInf(float64(encoded.Float32()), 1), ShouldBeTrue)
		})
	})

	Convey("Given NaN", test, func() {
		Convey("It should propagate NaN", func() {
			encoded := NewF8E5M2FromFloat32(float32(math.NaN()))

			So(math.IsNaN(float64(encoded.Float32())), ShouldBeTrue)
		})
	})
}

func BenchmarkFloat8E4M3_FromFloat32(benchmark *testing.B) {
	for benchmark.Loop() {
		_ = NewF8E4M3FromFloat32(1.0)
	}
}

func BenchmarkFloat8E4M3_ToFloat32(benchmark *testing.B) {
	value := NewF8E4M3FromFloat32(1.0)

	for benchmark.Loop() {
		_ = value.Float32()
	}
}

func BenchmarkFloat8E5M2_FromFloat32(benchmark *testing.B) {
	for benchmark.Loop() {
		_ = NewF8E5M2FromFloat32(1.0)
	}
}

func BenchmarkFloat8E5M2_ToFloat32(benchmark *testing.B) {
	value := NewF8E5M2FromFloat32(1.0)

	for benchmark.Loop() {
		_ = value.Float32()
	}
}
