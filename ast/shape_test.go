package ast

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"

)

func TestNewDynamicShape(t *testing.T) {
	convey.Convey("Given a rank", t, func() {
		convey.Convey("It should produce all-dynamic axes", func() {
			shape := NewDynamicShape(3)
			convey.So(shape, convey.ShouldResemble, []int64{DynamicDim, DynamicDim, DynamicDim})
		})
	})
}
