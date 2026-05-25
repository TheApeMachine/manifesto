package typer

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
)

func TestInferTypesElementwiseChain(t *testing.T) {
	convey.Convey("Given a flat chain x → add → relu", t, func() {
		graph := &ast.Graph{
			Inputs:  []string{"x", "y"},
			Outputs: map[string]string{"out": "relu_out"},
			Nodes: []*ast.GraphNode{
				{ID: "added", Op: "math.add", Inputs: []string{"x", "y"}},
				{ID: "relu_out", Op: "activation.relu", Inputs: []string{"added"}},
			},
		}

		stats, edgeErrors, err := Infer(graph)

		convey.So(err, convey.ShouldBeNil)
		convey.So(stats.NodesTyped, convey.ShouldEqual, 2)
		convey.So(edgeErrors, convey.ShouldBeEmpty)

		convey.Convey("Every node carries an InputTypes and OutputType slot", func() {
			convey.So(len(graph.Nodes[0].InputTypes), convey.ShouldEqual, 2)
			convey.So(graph.Nodes[0].OutputType.DType, convey.ShouldEqual, dtype.Float32)
			convey.So(len(graph.Nodes[1].InputTypes), convey.ShouldEqual, 1)
			convey.So(graph.Nodes[1].OutputType.DType, convey.ShouldEqual, dtype.Float32)
		})
	})
}

func TestInferDerivesMatmulOutputShape(t *testing.T) {
	convey.Convey("Given a matmul whose operands are produced by explicitly-shaped nodes", t, func() {
		RegisterSpec("test.emit_mk", OpSpec{
			Inputs: []ir.PortType{anyTensor()},
			OutputDeriver: func(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
				return ir.PortType{
					DType:       dtype.Float32,
					ShapeSchema: shapeSymbols("M", "K"),
					Layout:      ir.LayoutContiguous,
				}, nil
			},
		})

		RegisterSpec("test.emit_kn", OpSpec{
			Inputs: []ir.PortType{anyTensor()},
			OutputDeriver: func(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
				return ir.PortType{
					DType:       dtype.Float32,
					ShapeSchema: shapeSymbols("K", "N"),
					Layout:      ir.LayoutContiguous,
				}, nil
			},
		})

		graph := &ast.Graph{
			Inputs:  []string{"x"},
			Outputs: map[string]string{"out": "product"},
			Nodes: []*ast.GraphNode{
				{ID: "left", Op: "test.emit_mk", Inputs: []string{"x"}},
				{ID: "right", Op: "test.emit_kn", Inputs: []string{"x"}},
				{ID: "product", Op: "math.matmul", Inputs: []string{"left", "right"}},
			},
		}

		_, edgeErrors, err := Infer(graph)

		convey.So(err, convey.ShouldBeNil)
		convey.So(edgeErrors, convey.ShouldBeEmpty)

		convey.Convey("Output shape is rank-2 [M, N]", func() {
			productNode := graph.Nodes[2]
			outputDims := productNode.OutputType.ShapeSchema.Dimensions
			convey.So(len(outputDims), convey.ShouldEqual, 2)
			convey.So(outputDims[0].Symbol, convey.ShouldEqual, "M")
			convey.So(outputDims[1].Symbol, convey.ShouldEqual, "N")
		})
	})
}

func TestInferSurfacesCastHintForDTypeMismatch(t *testing.T) {
	convey.Convey("Given an embedding.token op fed by a default Float32 graph input", t, func() {
		// Graph inputs default to anyTensor() (Float32). embedding.token's
		// spec demands an Int32 TokenIndex input. The typer should surface
		// a cast hint on that edge so the synthesis pass can insert a
		// shape.cast.
		graph := &ast.Graph{
			Inputs: []string{"x"},
			Nodes: []*ast.GraphNode{
				{
					ID:      "embed",
					Op:      "embedding.token",
					Inputs:  []string{"x"},
					Weights: &ast.BoundWeight{TensorName: "table"},
				},
			},
		}

		_, edgeErrors, err := Infer(graph)

		convey.So(err, convey.ShouldBeNil)
		convey.So(len(edgeErrors), convey.ShouldEqual, 1)
		convey.So(edgeErrors[0].AdaptorHint, convey.ShouldEqual, "cast")
		convey.So(edgeErrors[0].Producer, convey.ShouldEqual, "x")
		convey.So(edgeErrors[0].Consumer, convey.ShouldEqual, "embed")
		convey.So(edgeErrors[0].ConsumerSlot, convey.ShouldEqual, 0)
	})
}

func TestRunInsertsCastAdaptor(t *testing.T) {
	convey.Convey("Given a graph where a producer dtype mismatches the consumer's expected dtype", t, func() {
		// Register a fake op that demands Int32 input.
		RegisterSpec("test.demand_int", OpSpec{
			Inputs: []ir.PortType{
				{DType: dtype.Int32, ShapeSchema: shapeSymbols("N"), Layout: ir.LayoutContiguous},
			},
			OutputDeriver: deriveSameAsFirstInput(ir.SemanticGeneric),
		})

		graph := &ast.Graph{
			Inputs: []string{"x", "y"},
			Outputs: map[string]string{
				"out": "consumer",
			},
			Nodes: []*ast.GraphNode{
				{ID: "added", Op: "math.add", Inputs: []string{"x", "y"}},
				{ID: "consumer", Op: "test.demand_int", Inputs: []string{"added"}},
			},
		}

		stats, err := Run(graph, Options{})

		convey.So(err, convey.ShouldBeNil)
		convey.So(stats.Synthesis.CastsInserted, convey.ShouldEqual, 1)

		convey.Convey("A shape.cast node was spliced in front of the consumer", func() {
			convey.So(len(graph.Nodes), convey.ShouldEqual, 3)

			adaptorIndex := -1
			consumerIndex := -1

			for index, node := range graph.Nodes {
				if node.Op == "shape.cast" {
					adaptorIndex = index
				}

				if node.ID == "consumer" {
					consumerIndex = index
				}
			}

			convey.So(adaptorIndex, convey.ShouldBeGreaterThanOrEqualTo, 0)
			convey.So(consumerIndex, convey.ShouldBeGreaterThan, adaptorIndex)

			consumer := graph.Nodes[consumerIndex]
			convey.So(consumer.Inputs[0], convey.ShouldEqual, graph.Nodes[adaptorIndex].ID)
		})
	})
}

func TestRunInsertsTransposeAdaptor(t *testing.T) {
	convey.Convey("Given a producer with one layout meeting a consumer that wants another", t, func() {
		RegisterSpec("test.want_channelfirst", OpSpec{
			Inputs: []ir.PortType{
				{DType: dtype.Float32, ShapeSchema: shapeSymbols("N", "C"), Layout: ir.LayoutChannelFirst},
			},
			OutputDeriver: deriveSameAsFirstInput(ir.SemanticGeneric),
		})

		RegisterSpec("test.emit_channellast", OpSpec{
			Inputs: []ir.PortType{anyTensor()},
			OutputDeriver: func(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
				return ir.PortType{
					DType:       dtype.Float32,
					ShapeSchema: shapeSymbols("N", "C"),
					Layout:      ir.LayoutChannelLast,
				}, nil
			},
		})

		graph := &ast.Graph{
			Inputs: []string{"x"},
			Outputs: map[string]string{
				"out": "consumer",
			},
			Nodes: []*ast.GraphNode{
				{ID: "producer", Op: "test.emit_channellast", Inputs: []string{"x"}},
				{ID: "consumer", Op: "test.want_channelfirst", Inputs: []string{"producer"}},
			},
		}

		stats, err := Run(graph, Options{})

		convey.So(err, convey.ShouldBeNil)
		convey.So(stats.Synthesis.TransposesInserted, convey.ShouldEqual, 1)
	})
}

func TestRunReturnsHardFailureOnRankMismatch(t *testing.T) {
	convey.Convey("Given a producer with rank 1 and a consumer that requires rank 2", t, func() {
		RegisterSpec("test.emit_rank1", OpSpec{
			Inputs: []ir.PortType{anyTensor()},
			OutputDeriver: func(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
				return ir.PortType{
					DType:       dtype.Float32,
					ShapeSchema: shapeSymbols("N"),
					Layout:      ir.LayoutContiguous,
				}, nil
			},
		})

		RegisterSpec("test.want_rank2", OpSpec{
			Inputs: []ir.PortType{
				{DType: dtype.Float32, ShapeSchema: shapeSymbols("N", "D"), Layout: ir.LayoutContiguous},
			},
			OutputDeriver: deriveSameAsFirstInput(ir.SemanticGeneric),
		})

		graph := &ast.Graph{
			Inputs: []string{"x"},
			Nodes: []*ast.GraphNode{
				{ID: "producer", Op: "test.emit_rank1", Inputs: []string{"x"}},
				{ID: "consumer", Op: "test.want_rank2", Inputs: []string{"producer"}},
			},
		}

		_, err := Run(graph, Options{})

		convey.So(err, convey.ShouldNotBeNil)

		hardFailure, ok := err.(*HardFailure)
		convey.So(ok, convey.ShouldBeTrue)
		convey.So(len(hardFailure.Errors), convey.ShouldEqual, 1)
		convey.So(hardFailure.Errors[0].AdaptorHint, convey.ShouldEqual, "")
	})
}

func TestRunBindsSymbolsAcrossGraph(t *testing.T) {
	convey.Convey("Given a static [4, 8] producer feeding a [N, D] consumer", t, func() {
		RegisterSpec("test.emit_static48", OpSpec{
			Inputs: []ir.PortType{anyTensor()},
			OutputDeriver: func(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
				return ir.PortType{
					DType: dtype.Float32,
					ShapeSchema: ir.ShapeSchema{
						Dimensions: []ir.Dimension{
							{Static: 4},
							{Static: 8},
						},
					},
					Layout: ir.LayoutContiguous,
				}, nil
			},
		})

		graph := &ast.Graph{
			Inputs: []string{"x"},
			Nodes: []*ast.GraphNode{
				{ID: "producer", Op: "test.emit_static48", Inputs: []string{"x"}},
				{ID: "norm", Op: "math.rmsnorm", Inputs: []string{"producer"},
					Weights: &ast.BoundWeight{TensorName: "scale"}},
			},
		}

		stats, err := Run(graph, Options{})

		convey.So(err, convey.ShouldBeNil)
		convey.So(graph.Bindings, convey.ShouldNotBeNil)
		convey.So(graph.Bindings["N"], convey.ShouldEqual, int64(4))
		convey.So(graph.Bindings["D"], convey.ShouldEqual, int64(8))
		convey.So(stats.Infer.BindingsResolved, convey.ShouldBeGreaterThan, 0)
	})
}
