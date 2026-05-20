package dtype

import (
	"encoding/json"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestInt64Value(t *testing.T) {
	Convey("Given JSON-decoded configuration scalars", t, func() {
		Convey("It should accept integer widths without loss", func() {
			value, err := Int64Value(int64(3072))
			So(err, ShouldBeNil)
			So(value, ShouldEqual, 3072)
		})

		Convey("It should accept whole-number float64 from encoding/json", func() {
			value, err := Int64Value(float64(36))
			So(err, ShouldBeNil)
			So(value, ShouldEqual, 36)
		})

		Convey("It should reject fractional float64 values", func() {
			_, err := Int64Value(float64(1.5))
			So(err, ShouldNotBeNil)
		})

		Convey("It should parse json.Number values", func() {
			value, err := Int64Value(json.Number("128"))
			So(err, ShouldBeNil)
			So(value, ShouldEqual, 128)
		})
	})
}

func TestFloat64Value(t *testing.T) {
	Convey("Given JSON-decoded hyperparameter scalars", t, func() {
		Convey("It should preserve float values such as epsilon", func() {
			value, err := Float64Value(float64(1e-6))
			So(err, ShouldBeNil)
			So(value, ShouldEqual, 1e-6)
		})
	})
}
