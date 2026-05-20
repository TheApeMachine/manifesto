package ast

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/manifesto/dtype"
)

func TestGraph_ApplyExecutionDType(t *testing.T) {
	convey.Convey("Given a graph with nodes", t, func() {
		graph := &Graph{
			Nodes: []*GraphNode{
				{ID: "embed", Op: "embedding.token"},
				{ID: "linear", Op: "projection.linear"},
			},
		}

		convey.Convey("It should stamp execution dtype on the graph and every node", func() {
			graph.ApplyExecutionDType(dtype.BFloat16)

			convey.So(graph.ExecutionDType, convey.ShouldEqual, dtype.BFloat16)
			convey.So(graph.Nodes[0].ValueType.DType, convey.ShouldEqual, dtype.BFloat16)
			convey.So(graph.Nodes[0].ValueType.Precision, convey.ShouldEqual, dtype.BFloat16)
			convey.So(graph.Nodes[1].ValueType.Memory, convey.ShouldEqual, MemoryDevice)
		})
	})
}
