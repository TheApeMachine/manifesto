package lower

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
)

func TestShapeInferencer_Apply(t *testing.T) {
	convey.Convey("Given a transformer-style topology chain", t, func() {
		topology := &ast.Topology{
			Inputs: []string{"input_ids"},
			Nodes: []ast.Node{
				{
					ID:  "embed",
					Op:  "embedding.token",
					In:  []string{"input_ids"},
					Out: []string{"hidden"},
					Config: map[string]any{
						"vocab_size": int64(128256),
						"d_model":    int64(2048),
					},
				},
				{
					ID:  "q_proj",
					Op:  "projection.linear",
					In:  []string{"hidden"},
					Out: []string{"query"},
					Config: map[string]any{
						"in_features":  int64(2048),
						"out_features": int64(2048),
					},
				},
				{
					ID:  "q_heads",
					Op:  "shape.view_as_heads",
					In:  []string{"query"},
					Out: []string{"query_heads"},
					Config: map[string]any{
						"num_heads": int64(32),
					},
				},
				{
					ID:  "merge",
					Op:  "shape.merge_heads",
					In:  []string{"query_heads"},
					Out: []string{"merged"},
				},
			},
		}
		lowerer := NewLowerer()

		convey.Convey("It should infer activation shapes on every node", func() {
			graph, err := lowerer.Topology(topology, dtype.Float32)
			convey.So(err, convey.ShouldBeNil)
			convey.So(graph.Nodes[0].ValueType.Shape, convey.ShouldResemble, []int64{ast.DynamicDim, ast.DynamicDim, 2048})
			convey.So(graph.Nodes[1].ValueType.Shape, convey.ShouldResemble, []int64{ast.DynamicDim, ast.DynamicDim, 2048})
			convey.So(graph.Nodes[2].ValueType.Shape, convey.ShouldResemble, []int64{ast.DynamicDim, ast.DynamicDim, 32, 64})
			convey.So(graph.Nodes[3].ValueType.Shape, convey.ShouldResemble, []int64{ast.DynamicDim, ast.DynamicDim, 2048})
		})
	})
}

func TestLowerer_Topology(t *testing.T) {
	convey.Convey("Given an expanded topology", t, func() {
		topology := &ast.Topology{
			Inputs: []string{"input_ids"},
			Nodes: []ast.Node{
				{
					ID:  "embed",
					Op:  "embedding.token",
					In:  []string{"input_ids"},
					Out: []string{"hidden"},
					Config: map[string]any{
						"d_model": int64(768),
					},
				},
			},
		}
		lowerer := NewLowerer()

		convey.Convey("It should lower nodes into graph IR with execution dtype and shape", func() {
			graph, err := lowerer.Topology(topology, dtype.Float32)
			convey.So(err, convey.ShouldBeNil)
			convey.So(len(graph.Nodes), convey.ShouldEqual, 1)
			convey.So(graph.Nodes[0].Op, convey.ShouldEqual, "embedding.token")
			convey.So(graph.ExecutionDType, convey.ShouldEqual, dtype.Float32)
			convey.So(graph.Nodes[0].ValueType.DType, convey.ShouldEqual, dtype.Float32)
			convey.So(graph.Nodes[0].ValueType.Shape, convey.ShouldResemble, []int64{ast.DynamicDim, ast.DynamicDim, 768})
		})
	})
}
