package compiler

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
)

func TestApplyExecutionDTypeFromConfigOrWeights(t *testing.T) {
	convey.Convey("Given a graph with bf16 weights and no config dtype", t, func() {
		graph := &ast.Graph{
			Nodes: []*ast.GraphNode{
				{
					ID: "linear",
					Weights: &ast.BoundWeight{
						TensorName: "linear.weight",
						DType:      dtype.BFloat16,
					},
				},
			},
		}

		convey.Convey("It should stamp bf16 execution dtype from weights", func() {
			result := applyExecutionDTypeFromConfigOrWeights(nil, graph, dtype.Float32)

			convey.So(result, convey.ShouldEqual, dtype.BFloat16)
			convey.So(graph.ExecutionDType, convey.ShouldEqual, dtype.BFloat16)
			convey.So(graph.Nodes[0].ValueType.DType, convey.ShouldEqual, dtype.BFloat16)
		})
	})

	convey.Convey("Given an explicit config dtype", t, func() {
		graph := &ast.Graph{
			Nodes: []*ast.GraphNode{
				{
					ID: "linear",
					Weights: &ast.BoundWeight{
						DType: dtype.BFloat16,
					},
				},
			},
		}
		config := map[string]any{"dtype": "float32"}

		convey.Convey("It should keep the configured dtype", func() {
			result := applyExecutionDTypeFromConfigOrWeights(config, graph, dtype.Float32)

			convey.So(result, convey.ShouldEqual, dtype.Float32)
			convey.So(graph.ExecutionDType, convey.ShouldEqual, dtype.Float32)
		})
	})
}
