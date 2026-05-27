package runtime

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
)

func TestResolveGraphInputForGraphBatchesTokenIDs(testingObject *testing.T) {
	convey.Convey("Given a graph boundary that expects batched token IDs", testingObject, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		graph := &ast.Graph{
			Inputs: []string{"input_ids"},
			Nodes: []*ast.GraphNode{
				{
					ID:     "embed",
					Op:     "embedding.token",
					Inputs: []string{"input_ids"},
					InputTypes: []ir.PortType{
						{
							DType: dtype.Int32,
							ShapeSchema: ir.ShapeSchema{
								Dimensions: []ir.Dimension{
									{Symbol: "B"},
									{Symbol: "T"},
								},
							},
							Kind: ir.SemanticTokenIndex,
						},
					},
				},
			},
		}

		resolved, err := ResolveGraphInputForGraph(graph, "input_ids", []int{11, 22, 33}, memory)

		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should upload a [1,T] int32 tensor", func() {
			tensorValue, ok := resolved.(tensor.Tensor)

			convey.So(ok, convey.ShouldBeTrue)
			convey.So(tensorValue.Shape().Dims(), convey.ShouldResemble, []int{1, 3})
			convey.So(tensorValue.DType(), convey.ShouldEqual, dtype.Int32)

			values, err := tensorValue.Int32Native()
			convey.So(err, convey.ShouldBeNil)
			convey.So(values, convey.ShouldResemble, []int32{11, 22, 33})
		})
	})
}

func TestResolveGraphInputForGraphLeavesUnbatchedTokens(testingObject *testing.T) {
	convey.Convey("Given a graph boundary without a batched token contract", testingObject, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		graph := &ast.Graph{
			Inputs: []string{"ids"},
			Nodes: []*ast.GraphNode{
				{
					ID:     "consumer",
					Op:     "test.consumer",
					Inputs: []string{"ids"},
					InputTypes: []ir.PortType{
						{
							DType: dtype.Int32,
							ShapeSchema: ir.ShapeSchema{
								Dimensions: []ir.Dimension{{Symbol: "N"}},
							},
							Kind: ir.SemanticTokenIndex,
						},
					},
				},
			},
		}

		resolved, err := ResolveGraphInputForGraph(graph, "ids", []int{1, 2}, memory)

		convey.So(err, convey.ShouldBeNil)
		convey.So(resolved, convey.ShouldResemble, []int{1, 2})
	})
}

func TestResolveGraphInputForGraphUploadsScalarTimestep(testingObject *testing.T) {
	convey.Convey("Given a graph boundary that expects a float timestep vector", testingObject, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		graph := &ast.Graph{
			Inputs: []string{"timestep"},
			Nodes: []*ast.GraphNode{
				{
					ID:     "time_guidance_embed.time_proj",
					Op:     "embedding.timestep",
					Inputs: []string{"timestep"},
					InputTypes: []ir.PortType{
						{
							DType: dtype.Float32,
							ShapeSchema: ir.ShapeSchema{
								Dimensions: []ir.Dimension{{Symbol: "B"}},
							},
						},
					},
				},
			},
		}

		resolved, err := ResolveGraphInputForGraph(graph, "timestep", float32(250), memory)

		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should upload a [1] float32 tensor", func() {
			tensorValue, ok := resolved.(tensor.Tensor)

			convey.So(ok, convey.ShouldBeTrue)
			convey.So(tensorValue.Shape().Dims(), convey.ShouldResemble, []int{1})
			convey.So(tensorValue.DType(), convey.ShouldEqual, dtype.Float32)

			values, err := tensorValue.Float32Native()
			convey.So(err, convey.ShouldBeNil)
			convey.So(values, convey.ShouldResemble, []float32{250})
		})
	})
}
