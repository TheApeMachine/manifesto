package dtype

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestDType_Size(test *testing.T) {
	Convey("Given each well-known dtype", test, func() {
		cases := map[DType]int{
			Float64:    8,
			Float32:    4,
			Float16:    2,
			BFloat16:   2,
			Float8E4M3: 1,
			Float8E5M2: 1,
			Int64:      8,
			Int32:      4,
			Int16:      2,
			Int8:       1,
			Uint64:     8,
			Uint32:     4,
			Uint16:     2,
			Uint8:      1,
			Complex64:  8,
			Complex128: 16,
		}

		Convey("Size should match the element width", func() {
			for dtype, expected := range cases {
				size, err := dtype.Size()
				So(err, ShouldBeNil)
				So(size, ShouldEqual, expected)
			}
		})
	})

	Convey("Given packed dtypes", test, func() {
		Convey("Size should report zero (caller uses BytesFor)", func() {
			size, err := Int4.Size()
			So(err, ShouldBeNil)
			So(size, ShouldEqual, 0)

			size, err = Bool.Size()
			So(err, ShouldBeNil)
			So(size, ShouldEqual, 0)
		})
	})

	Convey("Given Invalid", test, func() {
		Convey("Size should fail", func() {
			_, err := Invalid.Size()
			So(err, ShouldNotBeNil)
		})
	})
}

func TestDType_BytesFor(test *testing.T) {
	Convey("Given an unpacked dtype", test, func() {
		Convey("It should multiply elements by element width", func() {
			bytes, err := Float32.BytesFor(10)
			So(err, ShouldBeNil)
			So(bytes, ShouldEqual, 40)
		})
	})

	Convey("Given Int4 (packed 2-per-byte)", test, func() {
		Convey("It should round up odd element counts", func() {
			bytes, err := Int4.BytesFor(10)
			So(err, ShouldBeNil)
			So(bytes, ShouldEqual, 5)

			bytes, err = Int4.BytesFor(11)
			So(err, ShouldBeNil)
			So(bytes, ShouldEqual, 6)
		})
	})

	Convey("Given Bool (packed 8-per-byte)", test, func() {
		Convey("It should round up by 8", func() {
			bytes, err := Bool.BytesFor(1)
			So(err, ShouldBeNil)
			So(bytes, ShouldEqual, 1)

			bytes, err = Bool.BytesFor(8)
			So(err, ShouldBeNil)
			So(bytes, ShouldEqual, 1)

			bytes, err = Bool.BytesFor(9)
			So(err, ShouldBeNil)
			So(bytes, ShouldEqual, 2)
		})
	})
}

func TestDType_LogicalElements(test *testing.T) {
	Convey("Given an unpacked dtype", test, func() {
		Convey("It should divide byte count by element width", func() {
			elements, err := Float32.LogicalElements(40)
			So(err, ShouldBeNil)
			So(elements, ShouldEqual, 10)
		})

		Convey("It should reject misaligned byte counts", func() {
			_, err := Float32.LogicalElements(41)
			So(err, ShouldNotBeNil)
		})
	})

	Convey("Given packed dtypes", test, func() {
		Convey("Int4 doubles the byte count", func() {
			elements, err := Int4.LogicalElements(5)
			So(err, ShouldBeNil)
			So(elements, ShouldEqual, 10)
		})

		Convey("Bool octuples the byte count", func() {
			elements, err := Bool.LogicalElements(2)
			So(err, ShouldBeNil)
			So(elements, ShouldEqual, 16)
		})
	})
}

func TestDType_Parse(test *testing.T) {
	Convey("Given safetensors-cased identifiers", test, func() {
		cases := map[string]DType{
			"F64":    Float64,
			"F32":    Float32,
			"F16":    Float16,
			"BF16":   BFloat16,
			"F8E4M3": Float8E4M3,
			"F8E5M2": Float8E5M2,
			"I64":    Int64,
			"I32":    Int32,
			"I16":    Int16,
			"I8":     Int8,
			"I4":     Int4,
			"U64":    Uint64,
			"U32":    Uint32,
			"U16":    Uint16,
			"U8":     Uint8,
			"BOOL":   Bool,
			"C64":    Complex64,
			"C128":   Complex128,
		}

		Convey("Parse should round-trip through String", func() {
			for input, expected := range cases {
				dtype, err := Parse(input)
				So(err, ShouldBeNil)
				So(dtype, ShouldEqual, expected)
				So(dtype.String(), ShouldEqual, input)
			}
		})
	})

	Convey("Given canonical lowercase names", test, func() {
		Convey("Parse should accept and round-trip via Name", func() {
			dtype, err := Parse("bf16")
			So(err, ShouldBeNil)
			So(dtype, ShouldEqual, BFloat16)
			So(dtype.Name(), ShouldEqual, "bf16")
		})
	})

	Convey("Given historical aliases", test, func() {
		Convey("FP8_E4M3 maps to Float8E4M3", func() {
			dtype, err := Parse("FP8_E4M3")
			So(err, ShouldBeNil)
			So(dtype, ShouldEqual, Float8E4M3)
		})

		Convey("FP8_E5M2 maps to Float8E5M2", func() {
			dtype, err := Parse("FP8_E5M2")
			So(err, ShouldBeNil)
			So(dtype, ShouldEqual, Float8E5M2)
		})
	})

	Convey("Given an unknown identifier", test, func() {
		Convey("It should reject", func() {
			_, err := Parse("not-a-dtype")
			So(err, ShouldNotBeNil)
		})
	})
}

func TestDType_Classifiers(test *testing.T) {
	Convey("Given a floating-point dtype", test, func() {
		So(Float32.IsFloat(), ShouldBeTrue)
		So(BFloat16.IsFloat(), ShouldBeTrue)
		So(Float8E4M3.IsFloat(), ShouldBeTrue)
		So(Int32.IsFloat(), ShouldBeFalse)
	})

	Convey("Given a signed integer dtype", test, func() {
		So(Int32.IsSignedInt(), ShouldBeTrue)
		So(Int4.IsSignedInt(), ShouldBeTrue)
		So(Uint32.IsSignedInt(), ShouldBeFalse)
	})

	Convey("Given an unsigned integer dtype", test, func() {
		So(Uint32.IsUnsignedInt(), ShouldBeTrue)
		So(Uint8.IsUnsignedInt(), ShouldBeTrue)
		So(Int32.IsUnsignedInt(), ShouldBeFalse)
	})

	Convey("Given a complex dtype", test, func() {
		So(Complex64.IsComplex(), ShouldBeTrue)
		So(Complex128.IsComplex(), ShouldBeTrue)
		So(Float32.IsComplex(), ShouldBeFalse)
	})

	Convey("Given a packed dtype", test, func() {
		So(Int4.IsPacked(), ShouldBeTrue)
		So(Bool.IsPacked(), ShouldBeTrue)
		So(Int8.IsPacked(), ShouldBeFalse)
	})
}
