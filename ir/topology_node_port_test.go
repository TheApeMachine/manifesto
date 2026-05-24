package ir

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
)

func TestTopologyPreservesPlannerFields(t *testing.T) {
	convey.Convey("Given a Topology with planner-output fields populated", t, func() {
		topology := Topology{
			Name: "tiny",
			Workspace: WorkspaceLayout{
				Size:  256,
				Align: 64,
			},
			InputPorts: map[string]int32{
				"input": 0,
			},
			OutputPorts: map[string]int32{
				"logits": 128,
			},
		}

		convey.Convey("Workspace, InputPorts, OutputPorts round-trip", func() {
			convey.So(topology.Workspace.Size, convey.ShouldEqual, int64(256))
			convey.So(topology.Workspace.Align, convey.ShouldEqual, int64(64))
			convey.So(topology.InputPorts["input"], convey.ShouldEqual, int32(0))
			convey.So(topology.OutputPorts["logits"], convey.ShouldEqual, int32(128))
		})
	})
}

func TestNodePreservesPlannerFields(t *testing.T) {
	convey.Convey("Given a Node with planner-output fields populated", t, func() {
		node := Node{
			Name:     "matmul-0",
			ID:       7,
			StreamID: 1,
			SyncBarriers: []SyncEvent{
				{EventID: 100, StreamID: 0, Wait: true},
				{EventID: 101, StreamID: 1, Wait: false},
			},
		}

		convey.Convey("ID, StreamID, SyncBarriers round-trip", func() {
			convey.So(node.ID, convey.ShouldEqual, int32(7))
			convey.So(node.StreamID, convey.ShouldEqual, int32(1))
			convey.So(len(node.SyncBarriers), convey.ShouldEqual, 2)
			convey.So(node.SyncBarriers[0].Wait, convey.ShouldBeTrue)
			convey.So(node.SyncBarriers[1].Wait, convey.ShouldBeFalse)
		})

		convey.Convey("JitKernel is nil by default", func() {
			// GoConvey's ShouldBeNil is type-strict: unsafe.Pointer(nil)
			// is not the same as untyped nil. Compare via the
			// integer-representation instead, which is unambiguous.
			convey.So(uintptr(node.JitKernel), convey.ShouldEqual, uintptr(0))
		})
	})
}

func TestPortPreservesPlannerFields(t *testing.T) {
	convey.Convey("Given a Port with typer and planner outputs populated", t, func() {
		port := Port{
			ID: 42,
			Type: PortType{
				DType:  dtype.Float32,
				Layout: LayoutContiguous,
				Kind:   SemanticLogits,
			},
			Allocation: &PortAllocation{
				PortID:     42,
				BaseOffset: 128,
			},
		}

		convey.Convey("ID, Type, Allocation round-trip", func() {
			convey.So(port.ID, convey.ShouldEqual, int32(42))
			convey.So(port.Type.DType, convey.ShouldEqual, dtype.Float32)
			convey.So(port.Type.Layout, convey.ShouldEqual, LayoutContiguous)
			convey.So(port.Type.Kind, convey.ShouldEqual, SemanticLogits)
			convey.So(port.Allocation, convey.ShouldNotBeNil)
			convey.So(port.Allocation.PortID, convey.ShouldEqual, int32(42))
			convey.So(port.Allocation.BaseOffset, convey.ShouldEqual, int64(128))
		})

		convey.Convey("Tensor remains nil unless explicitly set (backward compat)", func() {
			convey.So(port.Tensor, convey.ShouldBeNil)
		})
	})
}
