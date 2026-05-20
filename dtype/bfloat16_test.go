package dtype

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNewBfloat16FromFloat32(test *testing.T) {
	Convey("Given a float32 value", test, func() {
		bf16 := NewBfloat16FromFloat32(1.0)

		Convey("It should create a BF16 value", func() {
			So(bf16, ShouldEqual, BF16(0x3f80))
			So((&bf16).Float32(), ShouldEqual, 1.0)
		})
	})
}

func TestNewBfloat16FromBytes(test *testing.T) {
	Convey("Given a little-endian byte slice", test, func() {
		bf16 := NewBfloat16FromBytes([]byte{0x80, 0x3f})

		Convey("It should create a BF16 value", func() {
			So(bf16, ShouldEqual, BF16(0x3f80))
			So((&bf16).Float32(), ShouldEqual, 1.0)
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

func TestBF16_Decode(test *testing.T) {
	Convey("Given a little-endian byte slice with two BF16 values", test, func() {
		one := NewBfloat16FromFloat32(1.0)
		two := NewBfloat16FromFloat32(2.0)
		buf := []byte{0x80, 0x3f, 0x00, 0x40}

		Convey("It should decode both", func() {
			decoded := (&one).Decode(buf)
			So(decoded, ShouldResemble, []BF16{one, two})
		})
	})
}

func TestBF16_Encode(test *testing.T) {
	Convey("Given two BF16 values", test, func() {
		one := NewBfloat16FromFloat32(1.0)
		two := NewBfloat16FromFloat32(2.0)

		Convey("It should encode as tightly packed little-endian bytes", func() {
			encoded := (&one).Encode([]BF16{one, two})
			So(encoded, ShouldResemble, []byte{0x80, 0x3f, 0x00, 0x40})
		})
	})
}

func TestBF16_DecodeFloat32(test *testing.T) {
	Convey("Given a little-endian byte slice", test, func() {
		bf16 := NewBfloat16FromFloat32(1.0)
		buf := []byte{0x80, 0x3f, 0x00, 0x40}

		Convey("It should decode to float32 values", func() {
			So(bf16.DecodeFloat32(buf), ShouldResemble, []float32{1.0, 2.0})
		})
	})
}

func TestBF16_EncodeFloat32(test *testing.T) {
	Convey("Given float32 values", test, func() {
		bf16 := NewBfloat16FromFloat32(1.0)

		Convey("It should encode as little-endian BF16 bytes", func() {
			encoded := bf16.EncodeFloat32([]float32{1.0, 2.0})
			So(encoded, ShouldResemble, []byte{0x80, 0x3f, 0x00, 0x40})
		})
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
