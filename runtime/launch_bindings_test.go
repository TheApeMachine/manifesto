package runtime

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
)

func TestDeriveLaunchBindings(test *testing.T) {
	convey.Convey("Given a graph whose boundary input carries token IDs", test, func() {
		graph := &ast.Graph{
			Inputs: []string{"input_ids", "position_offset"},
		}

		convey.Convey("It should bind N and T to the live token count", func() {
			bindings := DeriveLaunchBindings(graph, map[string]any{
				"input_ids":       []int{1, 2, 3, 4},
				"position_offset": []float32{0},
			})

			convey.So(bindings["N"], convey.ShouldEqual, int64(4))
			convey.So(bindings["T"], convey.ShouldEqual, int64(4))
		})
	})

	convey.Convey("Given no token-bearing inputs", test, func() {
		convey.Convey("It should return nil bindings", func() {
			bindings := DeriveLaunchBindings(&ast.Graph{}, map[string]any{
				"sigma": []float32{1, 0.5},
			})

			convey.So(bindings, convey.ShouldBeNil)
		})
	})
}
