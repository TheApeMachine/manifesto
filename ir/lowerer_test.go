package ir

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
)

func TestLowerer_Graph(t *testing.T) {
	convey.Convey("Given a manifest graph with execution dtype", t, func() {
		manifestGraph := &ast.Graph{
			Nodes: []*ast.GraphNode{
				{
					ID: "embed",
					Op: "embedding.token",
					ValueType: ast.ValueType{
						Shape:     []int64{ast.DynamicDim, ast.DynamicDim, 2048},
						DType:     dtype.BFloat16,
						Precision: dtype.BFloat16,
						Layout:    ast.LayoutDense,
						Memory:    ast.MemoryDevice,
					},
				},
			},
		}
		lowerer := NewLowerer()

		convey.Convey("It should lower value types into compute IR", func() {
			computeGraph, err := lowerer.Graph(manifestGraph)
			convey.So(err, convey.ShouldBeNil)
			convey.So(len(computeGraph.Nodes()), convey.ShouldEqual, 1)
			convey.So(computeGraph.Nodes()[0].ValueType().DType, convey.ShouldEqual, dtype.BFloat16)
			convey.So(computeGraph.Nodes()[0].ValueType().Precision, convey.ShouldEqual, dtype.BFloat16)
			convey.So(computeGraph.Nodes()[0].OperationID(), convey.ShouldEqual, OpID("embedding.token"))
			convey.So(computeGraph.Nodes()[0].Shape().Dims(), convey.ShouldResemble, []int{1, 1, 2048})
		})
	})
}
