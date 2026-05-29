package compiler

import (
	"context"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/ir/dag"
	"github.com/theapemachine/manifesto/optimizer"
	"github.com/theapemachine/manifesto/types"
)

func TestNormalizeGraphTopologicalOrder(test *testing.T) {
	convey.Convey("Given a graph declared out of topological order", test, func() {
		graph := &ast.Graph{
			Inputs: []string{"x"},
			Nodes: []*ast.GraphNode{
				{ID: "consumer", Op: "math.add", Inputs: []string{"x", "producer"}},
				{ID: "producer", Op: "math.mul", Inputs: []string{"x"}},
			},
		}

		err := NormalizeGraph(graph)

		convey.Convey("NormalizeGraph reorders nodes into dependency order", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(graph.Nodes[0].ID, convey.ShouldEqual, "producer")
			convey.So(graph.Nodes[1].ID, convey.ShouldEqual, "consumer")
		})
	})
}

func TestNormalizeGraphRejectsCycle(test *testing.T) {
	convey.Convey("Given a cyclic graph", test, func() {
		graph := &ast.Graph{
			Inputs: []string{"x"},
			Nodes: []*ast.GraphNode{
				{ID: "a", Op: "math.add", Inputs: []string{"b"}},
				{ID: "b", Op: "math.add", Inputs: []string{"a"}},
			},
		}

		err := NormalizeGraph(graph)

		convey.Convey("NormalizeGraph returns a cycle error", func() {
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldContainSubstring, "cycle")
		})
	})
}

func TestBuildDAGFromGraphMatchesFinalAST(test *testing.T) {
	convey.Convey("Given a graph after fusion removed intermediate nodes", test, func() {
		graph := &ast.Graph{
			Inputs:  []string{"x", "y"},
			Outputs: map[string]string{"result": "relu_out"},
			Nodes: []*ast.GraphNode{
				{ID: "add_out", Op: "math.add", Inputs: []string{"x", "y"}},
				{ID: "sigmoid_out", Op: "activation.sigmoid", Inputs: []string{"add_out"}},
				{ID: "relu_out", Op: "activation.relu", Inputs: []string{"sigmoid_out"}},
			},
		}

		_, err := optimizer.Fuse(graph)
		convey.So(err, convey.ShouldBeNil)
		convey.So(len(graph.Nodes), convey.ShouldEqual, 1)

		computeGraph, err := BuildDAGFromGraph(graph)

		convey.Convey("The DAG contains only boundary inputs and the fused node", func() {
			convey.So(err, convey.ShouldBeNil)

			layers, err := computeGraph.TopologyLayers()
			convey.So(err, convey.ShouldBeNil)
			convey.So(len(layers), convey.ShouldEqual, 2)
			convey.So(layers[1][0].ID(), convey.ShouldEqual, graph.Nodes[0].ID)
		})

		convey.Convey("Execution layers reference every surviving graph node", func() {
			layers, err := computeGraph.TopologyLayers()
			convey.So(err, convey.ShouldBeNil)

			nodeIDs := make(map[string]struct{})

			for _, layer := range layers {
				for _, dagNode := range layer {
					if dagNode.OpType() == dag.OpInput {
						continue
					}

					nodeIDs[dagNode.ID()] = struct{}{}
				}
			}

			for _, node := range graph.Nodes {
				_, ok := nodeIDs[node.ID]
				convey.So(ok, convey.ShouldBeTrue)
			}
		})
	})
}

func TestCompileAssetsFullPipelineMatchesSchedule(test *testing.T) {
	convey.Convey("Given a program include compiled with all passes enabled", test, func() {
		programYAML := []byte(`kind: program
name: fusion
include:
  model: hf://example/fusion-model
main:
  - op: graph.call
    graph: model
`)

		blockYAML := []byte(`kind: Block
category: model
name: fusion-model
system:
  topology:
    inputs:
      - x
      - y
    outputs:
      result: relu_out
    nodes:
      - id: add_out
        op: math.add
        in: [x, y]
        out: [add_out]
      - id: sigmoid_out
        op: activation.sigmoid
        in: [add_out]
        out: [sigmoid_out]
      - id: relu_out
        op: activation.relu
        in: [sigmoid_out]
        out: [relu_out]
`)

		programCompiler, err := NewProgramCompiler(context.Background(), NewPool(nil))
		convey.So(err, convey.ShouldBeNil)

		programCompiler = programCompiler.
			WithIncludeResolver(&fakeResolver{payload: blockYAML}).
			WithPlannerBindings(ir.SymbolMap{"N": 4})

		output, err := programCompiler.CompileAssets(context.Background(), CompileInput{
			ProgramYAML: programYAML,
		}, nil)

		convey.Convey("CompileAssets succeeds and the schedule matches the fused graph", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(len(output.Graphs["model"].Nodes), convey.ShouldEqual, 1)
			convey.So(output.Graphs["model"].Nodes[0].Op, convey.ShouldEqual, optimizer.FuseOp)

			layers, err := output.ComputeGraphs["model"].TopologyLayers()
			convey.So(err, convey.ShouldBeNil)
			convey.So(len(layers), convey.ShouldEqual, 2)

			computeNodeIDs := make(map[string]struct{})

			for _, layer := range layers {
				for _, dagNode := range layer {
					if dagNode.OpType() == dag.OpInput {
						continue
					}

					computeNodeIDs[dagNode.ID()] = struct{}{}
				}
			}

			_, ok := computeNodeIDs[output.Graphs["model"].Nodes[0].ID]
			convey.So(ok, convey.ShouldBeTrue)
		})

		convey.Convey("Workspace planning populated graph I/O port offsets", func() {
			topology := output.Workspaces["model"]
			convey.So(topology, convey.ShouldNotBeNil)
			convey.So(topology.InputPorts["x"], convey.ShouldBeGreaterThanOrEqualTo, int32(0))
			convey.So(topology.OutputPorts["result"], convey.ShouldBeGreaterThanOrEqualTo, int32(0))
			convey.So(topology.Workspace.Size, convey.ShouldBeGreaterThan, int64(0))
		})
	})
}

func TestPlanGraphSchedulesStreams(test *testing.T) {
	convey.Convey("Given a typed diamond graph", test, func() {
		hidden := float32Port(4, 4)

		graph := &ast.Graph{
			Inputs:   []string{"x"},
			Bindings: ir.SymbolMap{},
			Nodes: []*ast.GraphNode{
				{ID: "left", Op: "math.mul", Inputs: []string{"x"}, InputTypes: []ir.PortType{hidden}, OutputType: hidden},
				{ID: "right", Op: "math.mul", Inputs: []string{"x"}, InputTypes: []ir.PortType{hidden}, OutputType: hidden},
				{ID: "merge", Op: "math.add", Inputs: []string{"left", "right"}, InputTypes: []ir.PortType{hidden, hidden}, OutputType: hidden},
			},
		}

		registry, err := types.NewOperationRegistry()
		convey.So(err, convey.ShouldBeNil)

		topology, err := PlanGraph(graph, PlanGraphOptions{
			Registry:       registry,
			StreamSchedule: ir.StreamScheduleOptions{MaxStreams: 4},
		})

		convey.Convey("Stream scheduling assigns stream IDs", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(topology.Nodes[2].StreamID, convey.ShouldBeGreaterThanOrEqualTo, int32(0))
		})
	})
}
