package ir

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
)

func makePortWithType(d dtype.DType, dims ...any) *Port {
	return &Port{
		Type: PortType{
			DType:       d,
			ShapeSchema: makeShape(dims...),
			Layout:      LayoutContiguous,
			Kind:        SemanticGeneric,
		},
	}
}

func makeLinearTopology(portCount int) (*Topology, []*Port) {
	ports := make([]*Port, portCount)
	for index := range ports {
		ports[index] = makePortWithType(dtype.Float32, 64, 64)
	}

	topology := &Topology{}

	for index := 0; index < portCount-1; index++ {
		producerNode := &Node{
			Name:    "step",
			Outputs: []*Port{ports[index+1]},
			Inputs:  []*Port{ports[index]},
		}

		topology.Nodes = append(topology.Nodes, producerNode)
	}

	return topology, ports
}

func TestAssignPortIDsAssignsUnique(t *testing.T) {
	convey.Convey("Given a 3-node linear topology with shared Port pointers", t, func() {
		topology, ports := makeLinearTopology(4)
		AssignPortIDs(topology)

		convey.Convey("Every port gets a unique non-zero ID", func() {
			seen := map[int32]bool{}

			for _, port := range ports {
				convey.So(port.ID, convey.ShouldNotEqual, int32(0))
				convey.So(seen[port.ID], convey.ShouldBeFalse)
				seen[port.ID] = true
			}
		})
	})
}

func TestAssignPortIDsIsIdempotent(t *testing.T) {
	convey.Convey("Given AssignPortIDs called twice", t, func() {
		topology, ports := makeLinearTopology(4)
		AssignPortIDs(topology)

		originalIDs := make([]int32, len(ports))
		for index, port := range ports {
			originalIDs[index] = port.ID
		}

		AssignPortIDs(topology)

		convey.Convey("IDs are unchanged on the second call", func() {
			for index, port := range ports {
				convey.So(port.ID, convey.ShouldEqual, originalIDs[index])
			}
		})
	})
}

func TestAnalyzeLivenessLinearChain(t *testing.T) {
	convey.Convey("Given a 4-port linear chain", t, func() {
		topology, ports := makeLinearTopology(4)
		AssignPortIDs(topology)

		intervals, err := AnalyzeLiveness(topology.Nodes, SymbolMap{})

		convey.Convey("Liveness produces one interval per distinct port", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(len(intervals), convey.ShouldEqual, 4)
		})

		convey.Convey("port[0] (unproduced input) is live from step 0 to step 0", func() {
			port0Interval := findInterval(intervals, ports[0].ID)
			convey.So(port0Interval, convey.ShouldNotBeNil)
			convey.So(port0Interval.Start, convey.ShouldEqual, 0)
			convey.So(port0Interval.End, convey.ShouldEqual, 0)
		})

		convey.Convey("port[1] is live from step 0 (produced) to step 1 (consumed)", func() {
			port1Interval := findInterval(intervals, ports[1].ID)
			convey.So(port1Interval, convey.ShouldNotBeNil)
			convey.So(port1Interval.Start, convey.ShouldEqual, 0)
			convey.So(port1Interval.End, convey.ShouldEqual, 1)
		})

		convey.Convey("Each interval has a real byte size", func() {
			for _, interval := range intervals {
				convey.So(interval.Size, convey.ShouldEqual, int64(64*64*4))
			}
		})
	})
}

func TestAnalyzeLivenessMultiConsumer(t *testing.T) {
	convey.Convey("Given a port consumed by two later nodes", t, func() {
		input := makePortWithType(dtype.Float32, 32, 32)
		shared := makePortWithType(dtype.Float32, 32, 32)
		output1 := makePortWithType(dtype.Float32, 32, 32)
		output2 := makePortWithType(dtype.Float32, 32, 32)

		topology := &Topology{
			Nodes: []*Node{
				{Name: "produce", Outputs: []*Port{shared}, Inputs: []*Port{input}},
				{Name: "consume1", Outputs: []*Port{output1}, Inputs: []*Port{shared}},
				{Name: "consume2", Outputs: []*Port{output2}, Inputs: []*Port{shared}},
			},
		}

		AssignPortIDs(topology)
		intervals, err := AnalyzeLiveness(topology.Nodes, SymbolMap{})

		convey.So(err, convey.ShouldBeNil)

		convey.Convey("The shared port lives from step 0 to step 2 (the last consumer)", func() {
			sharedInterval := findInterval(intervals, shared.ID)
			convey.So(sharedInterval, convey.ShouldNotBeNil)
			convey.So(sharedInterval.Start, convey.ShouldEqual, 0)
			convey.So(sharedInterval.End, convey.ShouldEqual, 2)
		})
	})
}

func TestAnalyzeLivenessRejectsUnboundSymbol(t *testing.T) {
	convey.Convey("Given a port with an unbound symbolic shape", t, func() {
		port := makePortWithType(dtype.Float32, "B", 768)

		topology := &Topology{
			Nodes: []*Node{
				{Name: "use", Inputs: []*Port{port}},
			},
		}

		AssignPortIDs(topology)

		_, err := AnalyzeLiveness(topology.Nodes, SymbolMap{})

		convey.Convey("Liveness analysis errors out with a clear message", func() {
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldContainSubstring, "unbound")
			convey.So(err.Error(), convey.ShouldContainSubstring, "B")
		})
	})
}

func TestAnalyzeLivenessResolvesBoundSymbols(t *testing.T) {
	convey.Convey("Given a port with a symbolic shape and a binding", t, func() {
		port := makePortWithType(dtype.Float32, "B", 768)

		topology := &Topology{
			Nodes: []*Node{{Name: "use", Inputs: []*Port{port}}},
		}

		AssignPortIDs(topology)

		intervals, err := AnalyzeLiveness(topology.Nodes, SymbolMap{"B": 4})

		convey.Convey("Liveness analysis resolves B=4 and computes size = 4 × 768 × 4 bytes", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(len(intervals), convey.ShouldEqual, 1)
			convey.So(intervals[0].Size, convey.ShouldEqual, int64(4*768*4))
		})
	})
}

func TestAllocateOffsetsDisjointIntervalsShareOffset(t *testing.T) {
	convey.Convey("Given two intervals with disjoint liveness ranges", t, func() {
		intervals := []Interval{
			{PortID: 1, Start: 0, End: 2, Size: 128},
			{PortID: 2, Start: 3, End: 5, Size: 128},
		}

		total, allocated := AllocateOffsets(intervals, 64)

		convey.Convey("They share the same Offset and total workspace is 128 bytes", func() {
			convey.So(allocated[0].Offset, convey.ShouldEqual, allocated[1].Offset)
			convey.So(total, convey.ShouldEqual, int64(128))
		})
	})
}

func TestAllocateOffsetsOverlappingIntervalsGetDistinctOffsets(t *testing.T) {
	convey.Convey("Given two intervals whose liveness overlaps", t, func() {
		intervals := []Interval{
			{PortID: 1, Start: 0, End: 5, Size: 128},
			{PortID: 2, Start: 3, End: 7, Size: 128},
		}

		total, allocated := AllocateOffsets(intervals, 64)

		convey.Convey("They get distinct Offsets and total workspace is 256 bytes", func() {
			byID := map[int32]Interval{}
			for _, interval := range allocated {
				byID[interval.PortID] = interval
			}

			convey.So(byID[1].Offset, convey.ShouldNotEqual, byID[2].Offset)
			convey.So(total, convey.ShouldEqual, int64(256))
		})
	})
}

func TestAllocateOffsetsRoundsToAlignment(t *testing.T) {
	convey.Convey("Given intervals with sub-alignment sizes", t, func() {
		intervals := []Interval{
			{PortID: 1, Start: 0, End: 5, Size: 17},
			{PortID: 2, Start: 3, End: 7, Size: 17},
		}

		total, allocated := AllocateOffsets(intervals, 64)

		convey.Convey("Sizes round to alignment and offsets stay aligned", func() {
			for _, interval := range allocated {
				convey.So(interval.Size, convey.ShouldEqual, int64(64))
				convey.So(interval.Offset%int64(64), convey.ShouldEqual, int64(0))
			}

			convey.So(total, convey.ShouldEqual, int64(128))
		})
	})
}

func TestPlanWorkspaceLinearTopology(t *testing.T) {
	convey.Convey("Given a 4-port linear chain", t, func() {
		topology, ports := makeLinearTopology(4)

		err := PlanWorkspace(topology, PlanWorkspaceOptions{
			Bindings: SymbolMap{},
			Align:    64,
		})

		convey.Convey("PlanWorkspace populates Workspace and every Port.Allocation", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(topology.Workspace.Size, convey.ShouldBeGreaterThan, int64(0))
			convey.So(topology.Workspace.Align, convey.ShouldEqual, int64(64))
			convey.So(len(topology.Workspace.Allocations), convey.ShouldEqual, 4)

			for _, port := range ports {
				convey.So(port.Allocation, convey.ShouldNotBeNil)
				convey.So(port.Allocation.PortID, convey.ShouldEqual, port.ID)
				convey.So(port.Allocation.BaseOffset%int64(64), convey.ShouldEqual, int64(0))
			}
		})
	})
}

func TestPlanWorkspaceUsesAlignmentDefault(t *testing.T) {
	convey.Convey("Given PlanWorkspaceOptions with Align=0", t, func() {
		topology, _ := makeLinearTopology(2)

		err := PlanWorkspace(topology, PlanWorkspaceOptions{})

		convey.Convey("PlanWorkspace falls back to 64-byte alignment", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(topology.Workspace.Align, convey.ShouldEqual, int64(64))
		})
	})
}

func TestPlanWorkspaceRejectsNonPowerOfTwoAlign(t *testing.T) {
	convey.Convey("Given PlanWorkspaceOptions with Align=48", t, func() {
		topology, _ := makeLinearTopology(2)

		err := PlanWorkspace(topology, PlanWorkspaceOptions{Align: 48})

		convey.Convey("PlanWorkspace returns an error", func() {
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldContainSubstring, "power of two")
		})
	})
}

func TestPortByteSizeScalar(t *testing.T) {
	convey.Convey("Given a port with a 0-d (scalar) shape", t, func() {
		port := &Port{
			Type: PortType{
				DType: dtype.Float32,
			},
		}

		size, err := PortByteSize(port, SymbolMap{})

		convey.Convey("PortByteSize returns dtype size", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(size, convey.ShouldEqual, int64(4))
		})
	})
}

func findInterval(intervals []Interval, portID int32) *Interval {
	for index := range intervals {
		if intervals[index].PortID == portID {
			return &intervals[index]
		}
	}

	return nil
}
