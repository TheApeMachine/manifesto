package dtype

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestInt4Pair_LoHi(test *testing.T) {
	Convey("Given a pair built from signed 4-bit values", test, func() {
		pair := NewInt4Pair(-3, 5)

		Convey("Lo and Hi should sign-extend correctly", func() {
			So(pair.Lo(), ShouldEqual, int8(-3))
			So(pair.Hi(), ShouldEqual, int8(5))
		})
	})

	Convey("Given a pair at the int4 boundaries", test, func() {
		pair := NewInt4Pair(int4Min, int4Max)

		Convey("Lo and Hi should preserve the extremes", func() {
			So(pair.Lo(), ShouldEqual, int8(-8))
			So(pair.Hi(), ShouldEqual, int8(7))
		})
	})

	Convey("Given inputs outside the int4 range", test, func() {
		below := NewInt4Pair(-100, 0)
		above := NewInt4Pair(0, 100)

		Convey("They should clamp at the boundary", func() {
			So(below.Lo(), ShouldEqual, int8(-8))
			So(above.Hi(), ShouldEqual, int8(7))
		})
	})
}

func TestInt4Pair_WithLoWithHi(test *testing.T) {
	Convey("Given a pair", test, func() {
		pair := NewInt4Pair(1, 2)

		Convey("WithLo should replace only the low nibble", func() {
			updated := pair.WithLo(-4)
			So(updated.Lo(), ShouldEqual, int8(-4))
			So(updated.Hi(), ShouldEqual, int8(2))
		})

		Convey("WithHi should replace only the high nibble", func() {
			updated := pair.WithHi(-4)
			So(updated.Lo(), ShouldEqual, int8(1))
			So(updated.Hi(), ShouldEqual, int8(-4))
		})
	})
}

func TestInt4Pair_BitsRoundTrip(test *testing.T) {
	Convey("Given an Int4Pair", test, func() {
		pair := NewInt4Pair(-5, 6)

		Convey("Bits should round-trip via Int4Pair(byte)", func() {
			reconstructed := Int4Pair(pair.Bits())
			So(reconstructed.Lo(), ShouldEqual, pair.Lo())
			So(reconstructed.Hi(), ShouldEqual, pair.Hi())
		})
	})
}
