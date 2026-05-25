package optimizer

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
)

func TestFuseStraightChain(t *testing.T) {
	convey.Convey("Given a straight chain of three elementwise nodes", t, func() {
		graph := &ast.Graph{
			Inputs:  []string{"x", "y"},
			Outputs: map[string]string{"result": "relu_out"},
			Nodes: []*ast.GraphNode{
				{ID: "add_out", Op: "math.add", Inputs: []string{"x", "y"}},
				{ID: "sigmoid_out", Op: "activation.sigmoid", Inputs: []string{"add_out"}},
				{ID: "relu_out", Op: "activation.relu", Inputs: []string{"sigmoid_out"}},
			},
		}

		stats, err := Fuse(graph)

		convey.So(err, convey.ShouldBeNil)
		convey.So(stats.Clusters, convey.ShouldEqual, 1)
		convey.So(stats.NodesFused, convey.ShouldEqual, 3)
		convey.So(stats.NodesRemoved, convey.ShouldEqual, 2)

		convey.Convey("Then the chain collapses to one fused node", func() {
			convey.So(len(graph.Nodes), convey.ShouldEqual, 1)
			convey.So(graph.Nodes[0].Op, convey.ShouldEqual, FuseOp)
			convey.So(graph.Nodes[0].Inputs, convey.ShouldResemble, []string{"x", "y"})

			fusionAny, ok := graph.Nodes[0].Attributes[FuseAttributeAST]
			convey.So(ok, convey.ShouldBeTrue)

			fusion, ok := fusionAny.(*FusionAST)
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(fusion.Root.Type, convey.ShouldEqual, NodeReLU)
			convey.So(fusion.Root.Children[0].Type, convey.ShouldEqual, NodeSigmoid)
			convey.So(fusion.Root.Children[0].Children[0].Type, convey.ShouldEqual, NodeAdd)
			convey.So(len(fusion.ContainedNodeIDs), convey.ShouldEqual, 3)
			convey.So(fusion.ContainedNodeIDs, convey.ShouldContain, "add_out")
			convey.So(fusion.ContainedNodeIDs, convey.ShouldContain, "sigmoid_out")
			convey.So(fusion.ContainedNodeIDs, convey.ShouldContain, "relu_out")
		})
	})
}

func TestFuseDAGSwiGLUStylePattern(t *testing.T) {
	convey.Convey("Given a SwiGLU-style pattern Mul(Sigmoid(x), y)", t, func() {
		graph := &ast.Graph{
			Inputs:  []string{"gate", "up"},
			Outputs: map[string]string{"result": "swiglu"},
			Nodes: []*ast.GraphNode{
				{ID: "sig_gate", Op: "activation.sigmoid", Inputs: []string{"gate"}},
				{ID: "swiglu", Op: "math.mul", Inputs: []string{"sig_gate", "up"}},
			},
		}

		stats, err := Fuse(graph)

		convey.So(err, convey.ShouldBeNil)
		convey.So(stats.Clusters, convey.ShouldEqual, 1)
		convey.So(stats.NodesFused, convey.ShouldEqual, 2)

		convey.Convey("Then both nodes collapse into a single fused expression", func() {
			convey.So(len(graph.Nodes), convey.ShouldEqual, 1)

			fusion := graph.Nodes[0].Attributes[FuseAttributeAST].(*FusionAST)
			convey.So(fusion.Root.Type, convey.ShouldEqual, NodeMul)
			convey.So(fusion.Root.Children[0].Type, convey.ShouldEqual, NodeSigmoid)
			convey.So(fusion.Root.Children[1].Type, convey.ShouldEqual, NodeInput)
			convey.So(fusion.InputPorts, convey.ShouldContain, "gate")
			convey.So(fusion.InputPorts, convey.ShouldContain, "up")
		})
	})
}

func TestFuseDAGResidualChain(t *testing.T) {
	convey.Convey("Given a residual chain Add(Add(a, b), c) feeding a ReLU", t, func() {
		graph := &ast.Graph{
			Inputs:  []string{"a", "b", "c"},
			Outputs: map[string]string{"y": "relu_out"},
			Nodes: []*ast.GraphNode{
				{ID: "inner_sum", Op: "math.add", Inputs: []string{"a", "b"}},
				{ID: "outer_sum", Op: "math.add", Inputs: []string{"inner_sum", "c"}},
				{ID: "relu_out", Op: "activation.relu", Inputs: []string{"outer_sum"}},
			},
		}

		stats, err := Fuse(graph)

		convey.So(err, convey.ShouldBeNil)
		convey.So(stats.Clusters, convey.ShouldEqual, 1)
		convey.So(stats.NodesFused, convey.ShouldEqual, 3)

		convey.Convey("Then all three nodes collapse into one fusion", func() {
			convey.So(len(graph.Nodes), convey.ShouldEqual, 1)
			fusion := graph.Nodes[0].Attributes[FuseAttributeAST].(*FusionAST)
			convey.So(fusion.Root.Type, convey.ShouldEqual, NodeReLU)
			convey.So(fusion.Root.Children[0].Type, convey.ShouldEqual, NodeAdd)
			convey.So(fusion.Root.Children[0].Children[0].Type, convey.ShouldEqual, NodeAdd)
		})
	})
}

func TestFuseStopsAtNonElementwise(t *testing.T) {
	convey.Convey("Given an elementwise chain interrupted by a matmul", t, func() {
		graph := &ast.Graph{
			Inputs:  []string{"x", "weight"},
			Outputs: map[string]string{"result": "relu_out"},
			Nodes: []*ast.GraphNode{
				{ID: "scaled", Op: "math.mul", Inputs: []string{"x", "x"}},
				{ID: "projected", Op: "projection.linear", Inputs: []string{"scaled"}},
				{ID: "relu_out", Op: "activation.relu", Inputs: []string{"projected"}},
			},
		}

		stats, err := Fuse(graph)

		convey.So(err, convey.ShouldBeNil)
		convey.So(stats.Clusters, convey.ShouldEqual, 0)

		convey.Convey("Then no cluster forms because the chain has a non-elementwise op", func() {
			convey.So(len(graph.Nodes), convey.ShouldEqual, 3)
			convey.So(graph.Nodes[0].Op, convey.ShouldEqual, "math.mul")
			convey.So(graph.Nodes[1].Op, convey.ShouldEqual, "projection.linear")
			convey.So(graph.Nodes[2].Op, convey.ShouldEqual, "activation.relu")
		})
	})
}

func TestFuseSkipsSharedIntermediate(t *testing.T) {
	convey.Convey("Given an elementwise output consumed by two downstream nodes", t, func() {
		// add_out is consumed by both relu_out and projected — so the
		// fuser must keep it materialized rather than absorb it.
		graph := &ast.Graph{
			Inputs:  []string{"x", "y"},
			Outputs: map[string]string{"residual": "relu_out", "projected": "projected"},
			Nodes: []*ast.GraphNode{
				{ID: "add_out", Op: "math.add", Inputs: []string{"x", "y"}},
				{ID: "relu_out", Op: "activation.relu", Inputs: []string{"add_out"}},
				{ID: "projected", Op: "projection.linear", Inputs: []string{"add_out"}},
			},
		}

		stats, err := Fuse(graph)

		convey.So(err, convey.ShouldBeNil)
		convey.So(stats.Clusters, convey.ShouldEqual, 0)

		convey.Convey("Then the original three nodes survive unchanged", func() {
			convey.So(len(graph.Nodes), convey.ShouldEqual, 3)
		})
	})
}

func TestRewriteEliminatesIdentity(t *testing.T) {
	convey.Convey("Given a graph with a shape.identity node", t, func() {
		graph := &ast.Graph{
			Inputs:  []string{"x"},
			Outputs: map[string]string{"y": "downstream"},
			Nodes: []*ast.GraphNode{
				{ID: "id_out", Op: "shape.identity", Inputs: []string{"x"}},
				{ID: "downstream", Op: "math.add", Inputs: []string{"id_out", "id_out"}},
			},
		}

		stats, err := Rewrite(graph)

		convey.So(err, convey.ShouldBeNil)
		convey.So(stats.IdentitiesRemoved, convey.ShouldEqual, 1)

		convey.Convey("Then downstream consumers rewire to the identity's input", func() {
			convey.So(len(graph.Nodes), convey.ShouldEqual, 1)
			convey.So(graph.Nodes[0].ID, convey.ShouldEqual, "downstream")
			convey.So(graph.Nodes[0].Inputs, convey.ShouldResemble, []string{"x", "x"})
		})
	})
}

func TestRewriteFlagsScaleFold(t *testing.T) {
	convey.Convey("Given a math.mul by a scalar feeding a single projection.linear", t, func() {
		graph := &ast.Graph{
			Inputs:  []string{"x"},
			Outputs: map[string]string{"y": "projected"},
			Nodes: []*ast.GraphNode{
				{
					ID:     "scaled",
					Op:     "math.mul",
					Inputs: []string{"x"},
					Attributes: map[string]any{
						"scalar": 0.5,
					},
				},
				{
					ID:     "projected",
					Op:     "projection.linear",
					Inputs: []string{"scaled"},
				},
			},
		}

		stats, err := Rewrite(graph)

		convey.So(err, convey.ShouldBeNil)
		convey.So(stats.ScalesFolded, convey.ShouldEqual, 1)

		convey.Convey("Then the projection carries a fold_scale attribute", func() {
			projected := graph.Nodes[len(graph.Nodes)-1]
			convey.So(projected.Attributes["fold_scale"], convey.ShouldEqual, 0.5)
		})
	})
}

func TestTileAttachesConfig(t *testing.T) {
	convey.Convey("Given a graph with a matmul and a projection", t, func() {
		graph := &ast.Graph{
			Inputs:  []string{"x", "w1", "w2"},
			Outputs: map[string]string{"y": "linear_out"},
			Nodes: []*ast.GraphNode{
				{ID: "matmul_out", Op: "math.matmul", Inputs: []string{"x", "w1"}},
				{ID: "linear_out", Op: "projection.linear", Inputs: []string{"matmul_out"}},
			},
		}

		stats, err := Tile(graph, DefaultTileTarget())

		convey.So(err, convey.ShouldBeNil)
		convey.So(stats.TilesAttached, convey.ShouldEqual, 2)

		convey.Convey("Then both heavy ops carry a TileConfig", func() {
			tile, ok := graph.Nodes[0].Attributes[TileAttribute].(TileConfig)
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(tile.Rows, convey.ShouldBeGreaterThan, 0)
			convey.So(tile.Inner, convey.ShouldEqual, tile.Rows)
			convey.So(tile.Cols, convey.ShouldEqual, tile.Rows)
			convey.So(tile.L1Footprint, convey.ShouldBeGreaterThan, 0)
		})
	})
}

func TestRunPipelineCombinesPasses(t *testing.T) {
	convey.Convey("Given a graph with an identity, an elementwise chain, and a matmul", t, func() {
		graph := &ast.Graph{
			Inputs:  []string{"x", "y", "w"},
			Outputs: map[string]string{"out": "matmul_out"},
			Nodes: []*ast.GraphNode{
				{ID: "id_out", Op: "shape.identity", Inputs: []string{"x"}},
				{ID: "added", Op: "math.add", Inputs: []string{"id_out", "y"}},
				{ID: "relu_out", Op: "activation.relu", Inputs: []string{"added"}},
				{ID: "matmul_out", Op: "math.matmul", Inputs: []string{"relu_out", "w"}},
			},
		}

		stats, err := Run(graph, Options{})

		convey.So(err, convey.ShouldBeNil)

		convey.Convey("Then identity is eliminated, elementwise nodes are fused, matmul is tiled", func() {
			convey.So(stats.Rewrite.IdentitiesRemoved, convey.ShouldEqual, 1)
			convey.So(stats.Fusion.Clusters, convey.ShouldEqual, 1)
			convey.So(stats.Tiling.TilesAttached, convey.ShouldEqual, 1)
			convey.So(len(graph.Nodes), convey.ShouldEqual, 2)
		})
	})
}

func TestElementwiseOpsListsMappedOps(t *testing.T) {
	convey.Convey("The elementwise op list should include core arithmetic and activations", t, func() {
		ops := ElementwiseOps()

		convey.So(ops, convey.ShouldContain, "math.add")
		convey.So(ops, convey.ShouldContain, "math.mul")
		convey.So(ops, convey.ShouldContain, "activation.relu")
		convey.So(ops, convey.ShouldContain, "activation.gelu")
	})
}
