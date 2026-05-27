package compiler

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
)

/*
Hand-rolled PortType helpers. The planner consumes Float32 contiguous
tensors of varying shapes; building them by hand keeps the test
independent of the typer's actual op spec library.
*/
func float32Port(dims ...int64) ir.PortType {
	dimensions := make([]ir.Dimension, len(dims))

	for index, value := range dims {
		dimensions[index] = ir.Dimension{Static: value}
	}

	return ir.PortType{
		DType:       dtype.Float32,
		ShapeSchema: ir.ShapeSchema{Dimensions: dimensions},
		Layout:      ir.LayoutContiguous,
		Kind:        ir.SemanticGeneric,
	}
}

func TestTopologyForPlanning_SharesProducerConsumerPorts(test *testing.T) {
	convey.Convey("Given a three-node linear graph (x → n1 → n2 → n3)", test, func() {
		// Each node produces a Float32 [4, 4] tensor. Producer N is
		// consumed by N+1; the bridge must make the *ir.Port identity
		// agree between them so liveness analysis sees a single edge.
		hidden := float32Port(4, 4)

		graph := &ast.Graph{
			Inputs: []string{"x"},
			Nodes: []*ast.GraphNode{
				{ID: "n1", Op: "noop", Inputs: []string{"x"}, InputTypes: []ir.PortType{hidden}, OutputType: hidden},
				{ID: "n2", Op: "noop", Inputs: []string{"n1"}, InputTypes: []ir.PortType{hidden}, OutputType: hidden},
				{ID: "n3", Op: "noop", Inputs: []string{"n2"}, InputTypes: []ir.PortType{hidden}, OutputType: hidden},
			},
		}

		topology := TopologyForPlanning(graph)

		convey.Convey("The topology has one *ir.Node per ast.GraphNode", func() {
			convey.So(len(topology.Nodes), convey.ShouldEqual, 3)
		})

		convey.Convey("Each producer's Output port is the same pointer the consumer reads", func() {
			producer1Out := topology.Nodes[0].Outputs[0]
			consumer2In := topology.Nodes[1].Inputs[0]
			convey.So(consumer2In, convey.ShouldEqual, producer1Out)

			producer2Out := topology.Nodes[1].Outputs[0]
			consumer3In := topology.Nodes[2].Inputs[0]
			convey.So(consumer3In, convey.ShouldEqual, producer2Out)
		})

		convey.Convey("Graph-boundary input adopts the first consumer's InputType", func() {
			boundary := topology.Nodes[0].Inputs[0]
			convey.So(boundary, convey.ShouldNotBeNil)
			convey.So(boundary.Type.DType, convey.ShouldEqual, dtype.Float32)
			convey.So(boundary.Type.ShapeSchema.Dimensions, convey.ShouldResemble,
				[]ir.Dimension{{Static: 4}, {Static: 4}})
		})

		convey.Convey("Every output port carries its node's OutputType", func() {
			for _, node := range topology.Nodes {
				convey.So(node.Outputs[0].Type.DType, convey.ShouldEqual, dtype.Float32)
			}
		})
	})
}

func TestTopologyForPlanning_AliasesReshapePorts(test *testing.T) {
	convey.Convey("Given reshape nodes that return storage views", test, func() {
		flat := float32Port(4, 8)
		heads := float32Port(4, 2, 4)

		aliasCases := []struct {
			operation  string
			inputType  ir.PortType
			outputType ir.PortType
		}{
			{
				operation:  "shape.view_as_heads",
				inputType:  flat,
				outputType: heads,
			},
			{
				operation:  "shape.merge_heads",
				inputType:  heads,
				outputType: flat,
			},
		}

		for _, aliasCase := range aliasCases {
			convey.Convey(aliasCase.operation+" shares the producer port through its consumer", func() {
				graph := &ast.Graph{
					Inputs: []string{"x"},
					Nodes: []*ast.GraphNode{
						{
							ID:         "produce",
							Op:         "linear",
							Inputs:     []string{"x"},
							InputTypes: []ir.PortType{aliasCase.inputType},
							OutputType: aliasCase.inputType,
						},
						{
							ID:         "reshape",
							Op:         aliasCase.operation,
							Inputs:     []string{"produce"},
							InputTypes: []ir.PortType{aliasCase.inputType},
							OutputType: aliasCase.outputType,
						},
						{
							ID:         "consume",
							Op:         "rope",
							Inputs:     []string{"reshape"},
							InputTypes: []ir.PortType{aliasCase.outputType},
							OutputType: aliasCase.outputType,
						},
					},
				}

				topology := TopologyForPlanning(graph)

				producerOutput := topology.Nodes[0].Outputs[0]
				reshapeInput := topology.Nodes[1].Inputs[0]
				reshapeOutput := topology.Nodes[1].Outputs[0]
				consumerInput := topology.Nodes[2].Inputs[0]

				convey.So(reshapeInput, convey.ShouldEqual, producerOutput)
				convey.So(reshapeOutput, convey.ShouldEqual, producerOutput)
				convey.So(consumerInput, convey.ShouldEqual, producerOutput)

				ir.AssignPortIDs(topology)

				intervals, err := ir.AnalyzeLiveness(topology.Nodes, ir.SymbolMap{})

				convey.So(err, convey.ShouldBeNil)

				interval := planningIntervalForPort(intervals, producerOutput.ID)

				convey.So(interval, convey.ShouldNotBeNil)
				convey.So(interval.Start, convey.ShouldEqual, 0)
				convey.So(interval.End, convey.ShouldEqual, 2)
			})
		}
	})
}

func TestPlanGraph_ProducesNonOverlappingOffsets(test *testing.T) {
	convey.Convey("Given a typed linear graph", test, func() {
		hidden := float32Port(4, 4) // 64 bytes per tensor

		graph := &ast.Graph{
			Inputs:   []string{"x"},
			Bindings: ir.SymbolMap{},
			Nodes: []*ast.GraphNode{
				{ID: "n1", Op: "noop", Inputs: []string{"x"}, InputTypes: []ir.PortType{hidden}, OutputType: hidden},
				{ID: "n2", Op: "noop", Inputs: []string{"n1"}, InputTypes: []ir.PortType{hidden}, OutputType: hidden},
			},
		}

		topology, err := PlanGraph(graph)

		convey.Convey("Planning succeeds", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(topology, convey.ShouldNotBeNil)
		})

		convey.Convey("Workspace size is at least one 64-byte aligned tensor", func() {
			convey.So(topology.Workspace.Size, convey.ShouldBeGreaterThanOrEqualTo, int64(64))
			convey.So(topology.Workspace.Size%64, convey.ShouldEqual, int64(0))
		})

		convey.Convey("Every port has an allocation with a 64-byte aligned offset", func() {
			for _, node := range topology.Nodes {
				for _, port := range node.Outputs {
					convey.So(port.Allocation, convey.ShouldNotBeNil)
					convey.So(port.Allocation.BaseOffset%64, convey.ShouldEqual, int64(0))
				}
			}
		})

		convey.Convey("Each allocated interval is sized to at least the tensor bytes", func() {
			for _, interval := range topology.Workspace.Allocations {
				convey.So(interval.Size, convey.ShouldBeGreaterThanOrEqualTo, int64(64))
			}
		})
	})
}

func TestPlanGraph_ReusesOffsetWhenLifetimesDoNotOverlap(test *testing.T) {
	convey.Convey("Given a strictly serial chain", test, func() {
		// Linear chain x → n1 → n2 → n3 → out: n1's output is consumed
		// only by n2, so its interval ends at step 1. n3's output
		// starts at step 2 and is unrelated, so the coloring allocator
		// may park them at the same offset. With three tensors and a
		// peak liveness of two (input + active output), the workspace
		// should be ≤ 2 × tensor_size.
		hidden := float32Port(8, 8) // 256 bytes

		graph := &ast.Graph{
			Inputs:   []string{"x"},
			Bindings: ir.SymbolMap{},
			Nodes: []*ast.GraphNode{
				{ID: "n1", Op: "noop", Inputs: []string{"x"}, InputTypes: []ir.PortType{hidden}, OutputType: hidden},
				{ID: "n2", Op: "noop", Inputs: []string{"n1"}, InputTypes: []ir.PortType{hidden}, OutputType: hidden},
				{ID: "n3", Op: "noop", Inputs: []string{"n2"}, InputTypes: []ir.PortType{hidden}, OutputType: hidden},
			},
		}

		topology, err := PlanGraph(graph)

		convey.So(err, convey.ShouldBeNil)

		convey.Convey("Total workspace stays within twice the per-tensor footprint", func() {
			// Peak co-residence is two tensors (graph input plus the
			// node's output at any given step). 2 * 256 = 512.
			convey.So(topology.Workspace.Size, convey.ShouldBeLessThanOrEqualTo, int64(512))
		})

		convey.Convey("Disjoint intervals share offsets", func() {
			// Count distinct offsets — should be at most 2 for a
			// peak-2 liveness.
			distinct := map[int64]struct{}{}

			for _, interval := range topology.Workspace.Allocations {
				distinct[interval.Offset] = struct{}{}
			}

			convey.So(len(distinct), convey.ShouldBeLessThanOrEqualTo, 2)
		})
	})
}

func TestPlanGraph_PropagatesSymbolBindings(test *testing.T) {
	convey.Convey("Given a graph with one symbolic dimension and a binding", test, func() {
		// A hidden state of shape [B, 16]: B is dynamic, 16 is static.
		symbolic := ir.PortType{
			DType: dtype.Float32,
			ShapeSchema: ir.ShapeSchema{
				Dimensions: []ir.Dimension{{Symbol: "B"}, {Static: 16}},
			},
			Layout: ir.LayoutContiguous,
		}

		graph := &ast.Graph{
			Inputs:   []string{"x"},
			Bindings: ir.SymbolMap{"B": 4},
			Nodes: []*ast.GraphNode{
				{ID: "n1", Op: "noop", Inputs: []string{"x"}, InputTypes: []ir.PortType{symbolic}, OutputType: symbolic},
			},
		}

		topology, err := PlanGraph(graph)

		convey.Convey("Planning succeeds with the binding resolving B", func() {
			convey.So(err, convey.ShouldBeNil)
			// 4 * 16 * 4 bytes = 256, rounded up to 256 (already aligned).
			convey.So(topology.Workspace.Size, convey.ShouldBeGreaterThanOrEqualTo, int64(256))
		})
	})
}

func TestPlanGraph_RejectsUnboundSymbol(test *testing.T) {
	convey.Convey("Given a graph with a symbolic dimension and no binding", test, func() {
		symbolic := ir.PortType{
			DType: dtype.Float32,
			ShapeSchema: ir.ShapeSchema{
				Dimensions: []ir.Dimension{{Symbol: "B"}, {Static: 16}},
			},
			Layout: ir.LayoutContiguous,
		}

		graph := &ast.Graph{
			Inputs:   []string{"x"},
			Bindings: ir.SymbolMap{},
			Nodes: []*ast.GraphNode{
				{ID: "n1", Op: "noop", Inputs: []string{"x"}, InputTypes: []ir.PortType{symbolic}, OutputType: symbolic},
			},
		}

		_, err := PlanGraph(graph)

		convey.Convey("Planning fails with an unbound-symbol error", func() {
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldContainSubstring, "unbound")
			convey.So(err.Error(), convey.ShouldContainSubstring, "B")
		})
	})
}

func TestMergeSymbolMaps_AcceptsAgreeingBindings(test *testing.T) {
	convey.Convey("Given two SymbolMaps that agree on shared symbols", test, func() {
		base := ir.SymbolMap{"B": 4, "T": 32}
		overlay := ir.SymbolMap{"T": 32, "D": 64}

		merged := mergeSymbolMaps(base, overlay)

		convey.So(merged["B"], convey.ShouldEqual, int64(4))
		convey.So(merged["T"], convey.ShouldEqual, int64(32))
		convey.So(merged["D"], convey.ShouldEqual, int64(64))
	})
}

func TestMergeSymbolMaps_PanicsOnConflict(test *testing.T) {
	convey.Convey("Given two SymbolMaps that disagree on a symbol", test, func() {
		base := ir.SymbolMap{"B": 4}
		overlay := ir.SymbolMap{"B": 8}

		convey.So(func() {
			_ = mergeSymbolMaps(base, overlay)
		}, convey.ShouldPanic)
	})
}

func planningIntervalForPort(intervals []ir.Interval, portID int32) *ir.Interval {
	for index := range intervals {
		if intervals[index].PortID == portID {
			return &intervals[index]
		}
	}

	return nil
}
