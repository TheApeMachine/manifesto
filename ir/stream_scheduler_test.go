package ir

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
)

func makeNode(name string, inputs []*Port, outputs []*Port) *Node {
	return &Node{
		Name:    name,
		Inputs:  inputs,
		Outputs: outputs,
	}
}

func TestScheduleStreamsLinearTopologySingleStream(t *testing.T) {
	convey.Convey("Given a 3-node linear chain A → B → C", t, func() {
		portA := &Port{}
		portB := &Port{}
		portC := &Port{}

		topology := &Topology{
			Nodes: []*Node{
				makeNode("A", nil, []*Port{portA}),
				makeNode("B", []*Port{portA}, []*Port{portB}),
				makeNode("C", []*Port{portB}, []*Port{portC}),
			},
		}

		err := ScheduleStreams(topology, StreamScheduleOptions{MaxStreams: 4})

		convey.So(err, convey.ShouldBeNil)

		convey.Convey("All nodes are on the same stream", func() {
			convey.So(topology.Nodes[0].StreamID, convey.ShouldEqual, int32(0))
			convey.So(topology.Nodes[1].StreamID, convey.ShouldEqual, int32(0))
			convey.So(topology.Nodes[2].StreamID, convey.ShouldEqual, int32(0))
		})

		convey.Convey("No SyncBarriers are emitted on a serial chain", func() {
			for _, node := range topology.Nodes {
				convey.So(len(node.SyncBarriers), convey.ShouldEqual, 0)
			}
		})
	})
}

func TestScheduleStreamsDiamondEmitsSync(t *testing.T) {
	convey.Convey("Given a diamond A → B, A → C, then B+C → D", t, func() {
		portA := &Port{}
		portB := &Port{}
		portC := &Port{}
		portD := &Port{}

		topology := &Topology{
			Nodes: []*Node{
				makeNode("A", nil, []*Port{portA}),
				makeNode("B", []*Port{portA}, []*Port{portB}),
				makeNode("C", []*Port{portA}, []*Port{portC}),
				makeNode("D", []*Port{portB, portC}, []*Port{portD}),
			},
		}

		err := ScheduleStreams(topology, StreamScheduleOptions{MaxStreams: 4})

		convey.So(err, convey.ShouldBeNil)

		convey.Convey("A and B continue stream 0", func() {
			convey.So(topology.Nodes[0].StreamID, convey.ShouldEqual, int32(0))
			convey.So(topology.Nodes[1].StreamID, convey.ShouldEqual, int32(0))
		})

		convey.Convey("C opens stream 1 (parallel branch from A)", func() {
			convey.So(topology.Nodes[2].StreamID, convey.ShouldEqual, int32(1))
		})

		convey.Convey("D merges on the lowest predecessor stream (0)", func() {
			convey.So(topology.Nodes[3].StreamID, convey.ShouldEqual, int32(0))
		})

		convey.Convey("D has a Wait barrier on stream 1 (waiting for C)", func() {
			foundWait := false
			for _, event := range topology.Nodes[3].SyncBarriers {
				if event.Wait && event.StreamID == int32(1) {
					foundWait = true
				}
			}

			convey.So(foundWait, convey.ShouldBeTrue)
		})

		convey.Convey("C has a Signal barrier (someone is waiting for it)", func() {
			foundSignal := false
			for _, event := range topology.Nodes[2].SyncBarriers {
				if !event.Wait && event.StreamID == int32(1) {
					foundSignal = true
				}
			}

			convey.So(foundSignal, convey.ShouldBeTrue)
		})
	})
}

func TestScheduleStreamsFanOutAssignsDistinctStreams(t *testing.T) {
	convey.Convey("Given fan-out A → [B, C, D]", t, func() {
		portA := &Port{}
		portB := &Port{}
		portC := &Port{}
		portD := &Port{}

		topology := &Topology{
			Nodes: []*Node{
				makeNode("A", nil, []*Port{portA}),
				makeNode("B", []*Port{portA}, []*Port{portB}),
				makeNode("C", []*Port{portA}, []*Port{portC}),
				makeNode("D", []*Port{portA}, []*Port{portD}),
			},
		}

		err := ScheduleStreams(topology, StreamScheduleOptions{MaxStreams: 4})

		convey.So(err, convey.ShouldBeNil)

		convey.Convey("A is on stream 0, B continues stream 0, C and D get new streams", func() {
			convey.So(topology.Nodes[0].StreamID, convey.ShouldEqual, int32(0))
			convey.So(topology.Nodes[1].StreamID, convey.ShouldEqual, int32(0))
			convey.So(topology.Nodes[2].StreamID, convey.ShouldEqual, int32(1))
			convey.So(topology.Nodes[3].StreamID, convey.ShouldEqual, int32(2))
		})

		convey.Convey("Each fan-out consumer past the first has a Wait barrier on stream 0", func() {
			// B continues stream 0, no wait needed.
			convey.So(len(topology.Nodes[1].SyncBarriers), convey.ShouldEqual, 0)

			// C opened stream 1, must wait for A on stream 0.
			foundCWait := false
			for _, event := range topology.Nodes[2].SyncBarriers {
				if event.Wait && event.StreamID == int32(0) {
					foundCWait = true
				}
			}
			convey.So(foundCWait, convey.ShouldBeTrue)

			// D opened stream 2, must wait for A on stream 0.
			foundDWait := false
			for _, event := range topology.Nodes[3].SyncBarriers {
				if event.Wait && event.StreamID == int32(0) {
					foundDWait = true
				}
			}
			convey.So(foundDWait, convey.ShouldBeTrue)
		})
	})
}

func TestScheduleStreamsMaxStreamsCap(t *testing.T) {
	convey.Convey("Given fan-out A → [B, C, D] with MaxStreams=2", t, func() {
		portA := &Port{}
		portB := &Port{}
		portC := &Port{}
		portD := &Port{}

		topology := &Topology{
			Nodes: []*Node{
				makeNode("A", nil, []*Port{portA}),
				makeNode("B", []*Port{portA}, []*Port{portB}),
				makeNode("C", []*Port{portA}, []*Port{portC}),
				makeNode("D", []*Port{portA}, []*Port{portD}),
			},
		}

		err := ScheduleStreams(topology, StreamScheduleOptions{MaxStreams: 2})

		convey.So(err, convey.ShouldBeNil)

		convey.Convey("The third parallel branch reuses an existing stream", func() {
			usedStreams := map[int32]bool{}
			for _, node := range topology.Nodes {
				usedStreams[node.StreamID] = true
			}

			convey.So(len(usedStreams), convey.ShouldBeLessThanOrEqualTo, 2)
		})
	})
}

func TestScheduleStreamsSerialMaxStreams1(t *testing.T) {
	convey.Convey("Given any topology with MaxStreams=1", t, func() {
		portA := &Port{}
		portB := &Port{}
		portC := &Port{}
		portD := &Port{}

		topology := &Topology{
			Nodes: []*Node{
				makeNode("A", nil, []*Port{portA}),
				makeNode("B", []*Port{portA}, []*Port{portB}),
				makeNode("C", []*Port{portA}, []*Port{portC}),
				makeNode("D", []*Port{portB, portC}, []*Port{portD}),
			},
		}

		err := ScheduleStreams(topology, StreamScheduleOptions{MaxStreams: 1})

		convey.So(err, convey.ShouldBeNil)

		convey.Convey("Every node ends up on stream 0 (serial execution)", func() {
			for _, node := range topology.Nodes {
				convey.So(node.StreamID, convey.ShouldEqual, int32(0))
			}
		})

		convey.Convey("No SyncBarriers are emitted when MaxStreams forces serial execution", func() {
			for _, node := range topology.Nodes {
				convey.So(len(node.SyncBarriers), convey.ShouldEqual, 0)
			}
		})
	})
}

func TestScheduleStreamsHandlesEmptyTopology(t *testing.T) {
	convey.Convey("Given an empty topology", t, func() {
		topology := &Topology{}

		err := ScheduleStreams(topology, StreamScheduleOptions{MaxStreams: 4})

		convey.Convey("ScheduleStreams returns nil with no work to do", func() {
			convey.So(err, convey.ShouldBeNil)
		})
	})
}

func TestScheduleStreamsRejectsNilTopology(t *testing.T) {
	convey.Convey("Given a nil topology", t, func() {
		err := ScheduleStreams(nil, StreamScheduleOptions{MaxStreams: 4})

		convey.Convey("ScheduleStreams returns a clear error", func() {
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldContainSubstring, "topology is required")
		})
	})
}
