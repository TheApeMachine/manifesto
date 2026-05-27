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

func TestInferDerivesConcatOutputShape(t *testing.T) {
	convey.Convey("Given two rank-3 tensors concatenated across sequence", t, func() {
		RegisterSpec("test.emit_context", OpSpec{
			Inputs: []ir.PortType{anyTensor()},
			OutputDeriver: func(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
				return ir.PortType{
					DType: dtype.Float32,
					ShapeSchema: ir.ShapeSchema{
						Dimensions: []ir.Dimension{
							{Static: 1},
							{Static: 4},
							{Static: 3},
						},
					},
					Layout: ir.LayoutContiguous,
				}, nil
			},
		})

		RegisterSpec("test.emit_latents", OpSpec{
			Inputs: []ir.PortType{anyTensor()},
			OutputDeriver: func(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
				return ir.PortType{
					DType: dtype.Float32,
					ShapeSchema: ir.ShapeSchema{
						Dimensions: []ir.Dimension{
							{Static: 1},
							{Static: 5},
							{Static: 3},
						},
					},
					Layout: ir.LayoutContiguous,
				}, nil
			},
		})

		graph := &ast.Graph{
			Inputs: []string{"x"},
			Nodes: []*ast.GraphNode{
				{ID: "context", Op: "test.emit_context", Inputs: []string{"x"}},
				{ID: "latents", Op: "test.emit_latents", Inputs: []string{"x"}},
				{
					ID:     "joint",
					Op:     "shape.concat",
					Inputs: []string{"context", "latents"},
					Attributes: map[string]any{
						"dim": 1,
					},
				},
			},
		}

		_, edgeErrors, err := Infer(graph)

		convey.So(err, convey.ShouldBeNil)
		convey.So(edgeErrors, convey.ShouldBeEmpty)

		convey.Convey("Then the concat axis is summed", func() {
			outputDimensions := graph.Nodes[2].OutputType.ShapeSchema.Dimensions

			convey.So(len(outputDimensions), convey.ShouldEqual, 3)
			convey.So(outputDimensions[0].Static, convey.ShouldEqual, 1)
			convey.So(outputDimensions[1].Static, convey.ShouldEqual, 9)
			convey.So(outputDimensions[2].Static, convey.ShouldEqual, 3)
		})
	})
}

func TestInferDerivesBatchedEmbeddingOutputShape(t *testing.T) {
	convey.Convey("Given input_ids at a graph boundary", t, func() {
		graph := &ast.Graph{
			Inputs: []string{"input_ids"},
			Nodes: []*ast.GraphNode{
				{
					ID:     "embed",
					Op:     "embedding.token",
					Inputs: []string{"input_ids"},
					Attributes: map[string]any{
						"d_model": 8,
					},
					Weights: &ast.BoundWeight{TensorName: "model.embed_tokens.weight"},
				},
			},
		}

		_, edgeErrors, err := Infer(graph)

		convey.So(err, convey.ShouldBeNil)
		convey.So(edgeErrors, convey.ShouldBeEmpty)

		convey.Convey("Then token embedding preserves batch and sequence axes", func() {
			outputDimensions := graph.Nodes[0].OutputType.ShapeSchema.Dimensions

			convey.So(len(outputDimensions), convey.ShouldEqual, 3)
			convey.So(outputDimensions[0].Symbol, convey.ShouldEqual, "B")
			convey.So(outputDimensions[1].Symbol, convey.ShouldEqual, "T")
			convey.So(outputDimensions[2].Static, convey.ShouldEqual, 8)
		})
	})
}

func TestInferDerivesTimestepEmbeddingOutputShape(t *testing.T) {
	convey.Convey("Given timestep at a graph boundary", t, func() {
		graph := &ast.Graph{
			Inputs: []string{"timestep"},
			Nodes: []*ast.GraphNode{
				{
					ID:     "time_guidance_embed.time_proj",
					Op:     "embedding.timestep",
					Inputs: []string{"timestep"},
					Attributes: map[string]any{
						"dim": 256,
					},
				},
			},
		}

		_, edgeErrors, err := Infer(graph)

		convey.So(err, convey.ShouldBeNil)
		convey.So(edgeErrors, convey.ShouldBeEmpty)

		convey.Convey("Then timestep embedding appends the configured width", func() {
			outputDimensions := graph.Nodes[0].OutputType.ShapeSchema.Dimensions

			convey.So(len(outputDimensions), convey.ShouldEqual, 2)
			convey.So(outputDimensions[0].Symbol, convey.ShouldEqual, "B")
			convey.So(outputDimensions[1].Static, convey.ShouldEqual, 256)
		})
	})
}

func TestInferDerivesModulatedLayerNormOutputShape(t *testing.T) {
	convey.Convey("Given hidden states and modulation feeding ModulatedLayerNorm", t, func() {
		RegisterSpec("test.emit_modulated_hidden", OpSpec{
			Inputs: []ir.PortType{anyTensor()},
			OutputDeriver: func(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
				return ir.PortType{
					DType: dtype.Float32,
					ShapeSchema: ir.ShapeSchema{
						Dimensions: []ir.Dimension{
							{Static: 2},
							{Static: 3},
							{Static: 4},
						},
					},
					Layout: ir.LayoutContiguous,
					Kind:   ir.SemanticHiddenState,
				}, nil
			},
		})
		RegisterSpec("test.emit_modulation", OpSpec{
			Inputs: []ir.PortType{anyTensor()},
			OutputDeriver: func(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
				return ir.PortType{
					DType: dtype.Float32,
					ShapeSchema: ir.ShapeSchema{
						Dimensions: []ir.Dimension{
							{Static: 2},
							{Static: 24},
						},
					},
					Layout: ir.LayoutContiguous,
				}, nil
			},
		})

		graph := &ast.Graph{
			Inputs: []string{"x"},
			Nodes: []*ast.GraphNode{
				{ID: "hidden", Op: "test.emit_modulated_hidden", Inputs: []string{"x"}},
				{ID: "modulation", Op: "test.emit_modulation", Inputs: []string{"x"}},
				{
					ID:     "norm",
					Op:     "math.modulated_layernorm",
					Inputs: []string{"hidden", "modulation"},
				},
			},
		}

		_, edgeErrors, err := Infer(graph)

		convey.So(err, convey.ShouldBeNil)
		convey.So(edgeErrors, convey.ShouldBeEmpty)

		convey.Convey("Then output preserves the hidden-state shape", func() {
			outputDimensions := graph.Nodes[2].OutputType.ShapeSchema.Dimensions

			convey.So(len(outputDimensions), convey.ShouldEqual, 3)
			convey.So(outputDimensions[0].Static, convey.ShouldEqual, 2)
			convey.So(outputDimensions[1].Static, convey.ShouldEqual, 3)
			convey.So(outputDimensions[2].Static, convey.ShouldEqual, 4)
		})
	})
}

func TestInferDerivesMultiAxisRoPEOutputShape(t *testing.T) {
	convey.Convey("Given headed hidden states feeding MultiAxisRoPE", t, func() {
		RegisterSpec("test.emit_headed_hidden", OpSpec{
			Inputs: []ir.PortType{anyTensor()},
			OutputDeriver: func(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
				return ir.PortType{
					DType: dtype.Float32,
					ShapeSchema: ir.ShapeSchema{
						Dimensions: []ir.Dimension{
							{Static: 1},
							{Static: 8},
							{Static: 2},
							{Static: 8},
						},
					},
					Layout: ir.LayoutContiguous,
					Kind:   ir.SemanticHiddenState,
				}, nil
			},
		})

		graph := &ast.Graph{
			Inputs: []string{"x"},
			Nodes: []*ast.GraphNode{
				{ID: "hidden", Op: "test.emit_headed_hidden", Inputs: []string{"x"}},
				{
					ID:     "rope",
					Op:     "positional.multi_axis_rope",
					Inputs: []string{"hidden"},
				},
			},
		}

		_, edgeErrors, err := Infer(graph)

		convey.So(err, convey.ShouldBeNil)
		convey.So(edgeErrors, convey.ShouldBeEmpty)

		convey.Convey("Then output preserves the headed hidden-state shape", func() {
			outputDimensions := graph.Nodes[1].OutputType.ShapeSchema.Dimensions

			convey.So(len(outputDimensions), convey.ShouldEqual, 4)
			convey.So(outputDimensions[0].Static, convey.ShouldEqual, 1)
			convey.So(outputDimensions[1].Static, convey.ShouldEqual, 8)
			convey.So(outputDimensions[2].Static, convey.ShouldEqual, 2)
			convey.So(outputDimensions[3].Static, convey.ShouldEqual, 8)
		})
	})
}

func TestInferDerivesBatchedLastTokenOutputShape(t *testing.T) {
	convey.Convey("Given batched sequence hidden states", t, func() {
		RegisterSpec("test.emit_batched_hidden", OpSpec{
			Inputs: []ir.PortType{anyTensor()},
			OutputDeriver: func(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
				return ir.PortType{
					DType: dtype.Float32,
					ShapeSchema: ir.ShapeSchema{
						Dimensions: []ir.Dimension{
							{Static: 2},
							{Static: 5},
							{Static: 7},
						},
					},
					Layout: ir.LayoutContiguous,
				}, nil
			},
		})

		graph := &ast.Graph{
			Inputs: []string{"x"},
			Nodes: []*ast.GraphNode{
				{ID: "hidden", Op: "test.emit_batched_hidden", Inputs: []string{"x"}},
				{ID: "last", Op: "shape.last_token", Inputs: []string{"hidden"}},
			},
		}

		_, edgeErrors, err := Infer(graph)

		convey.So(err, convey.ShouldBeNil)
		convey.So(edgeErrors, convey.ShouldBeEmpty)

		convey.Convey("Then the sequence axis is removed", func() {
			outputDimensions := graph.Nodes[1].OutputType.ShapeSchema.Dimensions

			convey.So(len(outputDimensions), convey.ShouldEqual, 2)
			convey.So(outputDimensions[0].Static, convey.ShouldEqual, 2)
			convey.So(outputDimensions[1].Static, convey.ShouldEqual, 7)
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

/*
TestInferAssignsOutputTypeForUnknownOp guards against the planner crash
"port id=N: dtype INVALID rejected scalar size: dtype: unsupported
dtype 0" that surfaces when an unknown op leaves node.OutputType at the
zero value. The typer's unknown-op branch updates its internal
producerTypes map but must also write node.InputTypes and node.OutputType
so downstream passes (compiler/plan.TopologyForPlanning → ir.PlanWorkspace)
see typed ports.
*/
func TestInferAssignsOutputTypeForUnknownOp(t *testing.T) {
	convey.Convey("Given a chain: typed producer → unknown op → typed consumer", t, func() {
		graph := &ast.Graph{
			Inputs: []string{"x"},
			Nodes: []*ast.GraphNode{
				{ID: "added", Op: "math.add", Inputs: []string{"x", "x"}},
				{ID: "reshaped", Op: "shape.brand_new_op_typer_does_not_know", Inputs: []string{"added"}},
				{ID: "doubled", Op: "math.add", Inputs: []string{"reshaped", "reshaped"}},
			},
		}

		_, _, err := Infer(graph)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("The unknown op's OutputType is set, not the zero PortType", func() {
			unknown := graph.Nodes[1]
			convey.So(len(unknown.InputTypes), convey.ShouldEqual, 1)
			convey.So(unknown.InputTypes[0].DType, convey.ShouldEqual, dtype.Float32)
			convey.So(unknown.OutputType.DType, convey.ShouldEqual, dtype.Float32)
		})

		convey.Convey("The downstream typed consumer can still type its inputs", func() {
			consumer := graph.Nodes[2]
			convey.So(len(consumer.InputTypes), convey.ShouldEqual, 2)
			convey.So(consumer.InputTypes[0].DType, convey.ShouldEqual, dtype.Float32)
			convey.So(consumer.OutputType.DType, convey.ShouldEqual, dtype.Float32)
		})
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

func TestRunDoesNotBindNForRank4Norm(t *testing.T) {
	convey.Convey("Given a static rank-4 producer feeding an RMSNorm consumer", t, func() {
		RegisterSpec("test.emit_static_heads", OpSpec{
			Inputs: []ir.PortType{anyTensor()},
			OutputDeriver: func(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
				return ir.PortType{
					DType: dtype.Float32,
					ShapeSchema: ir.ShapeSchema{
						Dimensions: []ir.Dimension{
							{Static: 1},
							{Static: 4096},
							{Static: 24},
							{Static: 128},
						},
					},
					Layout: ir.LayoutContiguous,
				}, nil
			},
		})

		graph := &ast.Graph{
			Inputs: []string{"x"},
			Nodes: []*ast.GraphNode{
				{ID: "producer", Op: "test.emit_static_heads", Inputs: []string{"x"}},
				{ID: "norm", Op: "math.rmsnorm", Inputs: []string{"producer"},
					Weights: &ast.BoundWeight{TensorName: "scale"}},
			},
		}

		_, err := Run(graph, Options{})

		convey.So(err, convey.ShouldBeNil)
		convey.So(graph.Bindings["D"], convey.ShouldEqual, int64(128))
		_, exists := graph.Bindings["N"]
		convey.So(exists, convey.ShouldBeFalse)
	})
}
