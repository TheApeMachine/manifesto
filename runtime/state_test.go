package runtime

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
)

func TestNewStateStoreScalar(testingObject *testing.T) {
	convey.Convey("Given scalar runtime state", testingObject, func() {
		state, err := NewStateStore([]ast.StateDeclaration{
			{
				Name: "step_index",
				Type: "scalar",
				Init: "zero",
			},
		})
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should initialize and increment like a counter", func() {
			value, ok := state.Get("step_index")
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(value, convey.ShouldEqual, int64(0))

			err := state.Update("increment", "step_index")
			convey.So(err, convey.ShouldBeNil)

			value, ok = state.Get("step_index")
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(value, convey.ShouldEqual, int64(1))
		})
	})
}
