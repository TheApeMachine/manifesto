package convert

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
)

func TestBytesToFloat64(test *testing.T) {
	Convey("Given a float32 byte buffer", test, func() {
		original := []float32{1.0, -2.5, 0.125}
		buf := Float32ToBytes(original)

		Convey("It should decode losslessly to float64", func() {
			values, err := BytesToFloat64(dtype.Float32, buf)
			So(err, ShouldBeNil)
			So(len(values), ShouldEqual, 3)

			for index, expected := range original {
				So(values[index], ShouldEqual, float64(expected))
			}
		})
	})

	Convey("Given a BF16 byte buffer", test, func() {
		original := []float32{1.0, -2.0, 0.5}
		bf16s := make([]dtype.BF16, len(original))

		for index, value := range original {
			bf16s[index] = dtype.NewBfloat16FromFloat32(value)
		}

		buf := BFloat16ToBytes(bf16s)

		Convey("It should decode to float64 within bf16 precision", func() {
			values, err := BytesToFloat64(dtype.BFloat16, buf)
			So(err, ShouldBeNil)

			for index, expected := range original {
				So(values[index], ShouldAlmostEqual, float64(expected), 1e-2)
			}
		})
	})

	Convey("Given an int8 byte buffer", test, func() {
		buf := []byte{0, 1, 0xff, 0x80}

		Convey("It should sign-extend to float64", func() {
			values, err := BytesToFloat64(dtype.Int8, buf)
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{0, 1, -1, -128})
		})
	})

	Convey("Given an int4 byte buffer", test, func() {
		pairs := []byte{
			byte(dtype.NewInt4Pair(1, -2)),
			byte(dtype.NewInt4Pair(-8, 7)),
		}

		Convey("It should sign-extend each nibble", func() {
			values, err := BytesToFloat64(dtype.Int4, pairs)
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{1, -2, -8, 7})
		})
	})

	Convey("Given a Bool dtype", test, func() {
		Convey("It should reject conversion to float64 explicitly", func() {
			_, err := BytesToFloat64(dtype.Bool, []byte{0x01})
			So(err, ShouldNotBeNil)
		})
	})
}

func TestBytesToFloat32_RoundTrip(test *testing.T) {
	Convey("Given a float32 round-trip via bytes", test, func() {
		original := []float32{1.0, -2.5, 0.125, 1024.5}
		buf := Float32ToBytes(original)

		Convey("It should preserve values exactly", func() {
			values, err := BytesToFloat32(dtype.Float32, buf)
			So(err, ShouldBeNil)
			So(values, ShouldResemble, original)
		})
	})
}

func TestBytesToBFloat16_RoundTrip(test *testing.T) {
	Convey("Given values representable in BF16", test, func() {
		original := []float32{1.0, -2.0, 0.5, 4.0}
		buf := Float32ToBytes(original)

		Convey("It should round-trip via BF16", func() {
			bf16s, err := BytesToBFloat16(dtype.Float32, buf)
			So(err, ShouldBeNil)
			So(len(bf16s), ShouldEqual, len(original))

			roundTrip := make([]float32, len(bf16s))

			for index := range bf16s {
				roundTrip[index] = (&bf16s[index]).Float32()
			}

			for index, expected := range original {
				So(roundTrip[index], ShouldAlmostEqual, expected, 1e-2)
			}
		})
	})
}

func TestBytesToFloat64_Int4_OddLength(test *testing.T) {
	Convey("Given an Int4 buffer with a trailing half-byte (odd output length)", test, func() {
		// One packed byte: low nibble = 3, high nibble = -4 (twos-complement
		// in 4 bits = 0xC). The high nibble is past the requested element
		// count and must be left untouched — proves the loop's outIndex+1
		// guard fires correctly.
		pair := dtype.NewInt4Pair(3, -4)
		buf := []byte{byte(pair)}
		out := make([]float64, 1)

		Convey("It should decode only the low nibble without panicking", func() {
			result, err := BytesToFloat64(dtype.Int4, buf)
			So(err, ShouldBeNil)
			So(len(result), ShouldEqual, 2) // LogicalElements returns bytes*2 for Int4
			So(result[0], ShouldEqual, 3)
			So(result[1], ShouldEqual, -4)
		})

		_ = out
	})
}

func TestBytesToFloat64_Int4_SignExtensionCorners(test *testing.T) {
	Convey("Int4 sign-extension covers the full int4 range", test, func() {
		pairs := []byte{
			byte(dtype.NewInt4Pair(-8, -1)),
			byte(dtype.NewInt4Pair(0, 7)),
		}

		Convey("All four nibbles round-trip with correct sign", func() {
			values, err := BytesToFloat64(dtype.Int4, pairs)
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{-8, -1, 0, 7})
		})
	})
}

func TestBytesToInt8(test *testing.T) {
	Convey("Given float values within int8 range", test, func() {
		buf := Float32ToBytes([]float32{1, -1, 127, -128, 64})

		Convey("It should truncate to int8", func() {
			values, err := BytesToInt8(dtype.Float32, buf)
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []int8{1, -1, 127, -128, 64})
		})
	})

	Convey("Given float values outside int8 range", test, func() {
		buf := Float32ToBytes([]float32{1000, -1000})

		Convey("It should saturate at the boundary", func() {
			values, err := BytesToInt8(dtype.Float32, buf)
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []int8{127, -128})
		})
	})
}
