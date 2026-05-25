package dtype

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNewBfloat16FromFloat32(test *testing.T) {
	Convey("Given a float32 value", test, func() {
		bf16 := NewBfloat16FromFloat32(1.0)

		Convey("It should create a BF16 value", func() {
			So(bf16, ShouldEqual, BF16(0x3f80))
			So(bf16.Float32(), ShouldEqual, 1.0)
		})
	})
}

func TestNewBfloat16FromFloat32_RoundToNearestEven(test *testing.T) {
	Convey("Given values that round under RNE", test, func() {
		Convey("A value whose discarded mantissa is exactly half-ULP rounds to even", func() {
			// float32 mantissa: high seven bits become BF16's mantissa, the
			// remaining sixteen bits are the rounded fraction. A bit pattern
			// with mantissa low bits == 0x008000 sits exactly at the
			// half-way mark between two BF16 representable values; RNE
			// breaks the tie toward the even (LSB=0) neighbour.
			value := math.Float32frombits(0x3f800000 | 0x008000)
			rounded := NewBfloat16FromFloat32(value)
			So(rounded, ShouldEqual, BF16(0x3f80))
		})

		Convey("A value whose discarded bits exceed half-ULP rounds up", func() {
			value := math.Float32frombits(0x3f800000 | 0x008001)
			rounded := NewBfloat16FromFloat32(value)
			So(rounded, ShouldEqual, BF16(0x3f81))
		})

		Convey("Half-ULP on an odd-LSB neighbour rounds up to the even one", func() {
			// 0x3f81_8000 sits exactly halfway between BF16(0x3f81) (odd
			// LSB) and BF16(0x3f82) (even LSB). RNE picks the even one →
			// 0x3f82. Truncation — what the old implementation did — would
			// have stopped at 0x3f81 and locked in a systematic downward
			// bias; this test prevents a regression to that behaviour.
			value := math.Float32frombits(0x3f818000)
			rounded := NewBfloat16FromFloat32(value)
			So(rounded, ShouldEqual, BF16(0x3f82))
		})

		Convey("Infinities and NaNs pass through with their exponent intact", func() {
			posInf := NewBfloat16FromFloat32(float32(math.Inf(1)))
			So(posInf, ShouldEqual, BF16(0x7f80))

			negInf := NewBfloat16FromFloat32(float32(math.Inf(-1)))
			So(negInf, ShouldEqual, BF16(0xff80))

			nan := NewBfloat16FromFloat32(float32(math.NaN()))
			So(math.IsNaN(float64(nan.Float32())), ShouldBeTrue)
		})
	})
}

func TestNewBfloat16FromBytes(test *testing.T) {
	Convey("Given a little-endian byte slice", test, func() {
		bf16 := NewBfloat16FromBytes([]byte{0x80, 0x3f})

		Convey("It should create a BF16 value", func() {
			So(bf16, ShouldEqual, BF16(0x3f80))
			So(bf16.Float32(), ShouldEqual, 1.0)
		})
	})
}

func TestBF16_Bytes(test *testing.T) {
	Convey("Given a BF16 value", test, func() {
		bf16 := NewBfloat16FromFloat32(1.0)

		Convey("It should return the little-endian bytes", func() {
			So(bf16.Bytes(), ShouldResemble, []byte{0x80, 0x3f})
		})
	})
}

func TestDecodeBF16(test *testing.T) {
	Convey("Given a little-endian byte slice with two BF16 values", test, func() {
		one := NewBfloat16FromFloat32(1.0)
		two := NewBfloat16FromFloat32(2.0)
		buf := []byte{0x80, 0x3f, 0x00, 0x40}

		Convey("DecodeBF16 returns both", func() {
			decoded := DecodeBF16(buf)
			So(decoded, ShouldResemble, []BF16{one, two})
		})

		Convey("The deprecated method shim agrees with the package function", func() {
			var receiver BF16
			So((&receiver).Decode(buf), ShouldResemble, DecodeBF16(buf))
		})
	})
}

func TestEncodeBF16(test *testing.T) {
	Convey("Given two BF16 values", test, func() {
		one := NewBfloat16FromFloat32(1.0)
		two := NewBfloat16FromFloat32(2.0)

		Convey("EncodeBF16 packs them little-endian", func() {
			encoded := EncodeBF16([]BF16{one, two})
			So(encoded, ShouldResemble, []byte{0x80, 0x3f, 0x00, 0x40})
		})

		Convey("The deprecated method shim agrees with the package function", func() {
			var receiver BF16
			So((&receiver).Encode([]BF16{one, two}), ShouldResemble, EncodeBF16([]BF16{one, two}))
		})
	})
}

func TestDecodeBF16ToFloat32(test *testing.T) {
	Convey("Given a little-endian BF16 byte slice", test, func() {
		buf := []byte{0x80, 0x3f, 0x00, 0x40}

		Convey("DecodeBF16ToFloat32 widens to float32", func() {
			So(DecodeBF16ToFloat32(buf), ShouldResemble, []float32{1.0, 2.0})
		})
	})
}

func TestEncodeFloat32ToBF16(test *testing.T) {
	Convey("Given float32 values", test, func() {
		Convey("EncodeFloat32ToBF16 round-trips through RNE narrowing", func() {
			encoded := EncodeFloat32ToBF16([]float32{1.0, 2.0})
			So(encoded, ShouldResemble, []byte{0x80, 0x3f, 0x00, 0x40})
		})
	})
}

func TestBF16_RoundTrip(test *testing.T) {
	Convey("DecodeBF16(EncodeBF16(values)) recovers the originals", test, func() {
		values := []BF16{
			NewBfloat16FromFloat32(-2.5),
			NewBfloat16FromFloat32(0.0),
			NewBfloat16FromFloat32(1.0),
			NewBfloat16FromFloat32(float32(math.Pi)),
		}

		So(DecodeBF16(EncodeBF16(values)), ShouldResemble, values)
	})
}

func TestBF16_Float32(test *testing.T) {
	Convey("Given a BF16 value", test, func() {
		bf16 := NewBfloat16FromFloat32(1.0)

		Convey("It should return the float32 value", func() {
			So(bf16.Float32(), ShouldEqual, 1.0)
		})
	})
}

func TestBF16_Bits(test *testing.T) {
	Convey("Given a BF16 value", test, func() {
		bf16 := NewBfloat16FromFloat32(-2.0)

		Convey("It should expose the raw bits", func() {
			So(bf16.Bits(), ShouldEqual, uint16(0xc000))
		})
	})
}

func BenchmarkBF16_NewBfloat16FromFloat32(benchmark *testing.B) {
	for benchmark.Loop() {
		_ = NewBfloat16FromFloat32(1.0)
	}
}

func BenchmarkBF16_NewBfloat16FromBytes(benchmark *testing.B) {
	for benchmark.Loop() {
		_ = NewBfloat16FromBytes([]byte{0x80, 0x3f})
	}
}

func BenchmarkBF16_Bytes(benchmark *testing.B) {
	bf16 := NewBfloat16FromFloat32(1.0)

	for benchmark.Loop() {
		_ = bf16.Bytes()
	}
}

func BenchmarkBF16_Float32(benchmark *testing.B) {
	bf16 := NewBfloat16FromFloat32(1.0)

	for benchmark.Loop() {
		_ = bf16.Float32()
	}
}

func BenchmarkDecodeBF16(benchmark *testing.B) {
	buf := EncodeBF16([]BF16{
		NewBfloat16FromFloat32(1.0),
		NewBfloat16FromFloat32(2.0),
		NewBfloat16FromFloat32(3.0),
		NewBfloat16FromFloat32(4.0),
	})

	for benchmark.Loop() {
		_ = DecodeBF16(buf)
	}
}
