package ir

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/tensor"
)

func TestGraph(t *testing.T) {
	Convey("Given a new Graph", t, func() {
		graph := NewGraph()

		shape, err := tensor.NewShape([]int{2, 2})
		So(err, ShouldBeNil)

		nodeA := NewNode("a", OpInput, shape)
		nodeB := NewNode("b", OpMatmul, shape)
		nodeB.AddInput(nodeA)

		graph.AddNode(nodeA)
		graph.AddNode(nodeB)

		Convey("It should return all nodes", func() {
			nodes := graph.Nodes()
			So(len(nodes), ShouldEqual, 2)
		})

		Convey("It should correctly identify sink nodes", func() {
			sinks := graph.Sinks()
			So(len(sinks), ShouldEqual, 1)
			So(sinks[0].ID(), ShouldEqual, "b")
		})

		Convey("It should group nodes into concurrent topology layers", func() {
			nodeC := NewNode("c", OpMatmul, shape)
			nodeC.AddInput(nodeA)
			graph.AddNode(nodeC)

			nodeD := NewNode("d", OpAdd, shape)
			nodeD.AddInput(nodeB)
			nodeD.AddInput(nodeC)
			graph.AddNode(nodeD)

			layers, err := graph.TopologyLayers()
			So(err, ShouldBeNil)

			// Layer 0: a
			// Layer 1: b, c
			// Layer 2: d
			So(len(layers), ShouldEqual, 3)
			So(len(layers[0]), ShouldEqual, 1)
			So(layers[0][0].ID(), ShouldEqual, "a")
			So(len(layers[1]), ShouldEqual, 2)

			var layer1IDs []string
			for _, n := range layers[1] {
				layer1IDs = append(layer1IDs, n.ID())
			}
			So(layer1IDs, ShouldContain, "b")
			So(layer1IDs, ShouldContain, "c")

			So(len(layers[2]), ShouldEqual, 1)
			So(layers[2][0].ID(), ShouldEqual, "d")
		})

		Convey("It should detect cycles", func() {
			nodeCycle1 := NewNode("cyc1", OpAdd, shape)
			nodeCycle2 := NewNode("cyc2", OpAdd, shape)
			nodeCycle1.AddInput(nodeCycle2)
			nodeCycle2.AddInput(nodeCycle1)

			graphCycle := NewGraph()
			graphCycle.AddNode(nodeCycle1)
			graphCycle.AddNode(nodeCycle2)

			_, err := graphCycle.TopologyLayers()
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "cycle detected")
		})
	})
}

func BenchmarkGraph_Sinks(b *testing.B) {
	shape, err := tensor.NewShape([]int{2, 2})
	if err != nil {
		b.Fatalf("NewShape failed: %v", err)
	}

	graph := NewGraph()
	nodeA := NewNode("a", OpInput, shape)
	nodeB := NewNode("b", OpMatmul, shape)
	nodeB.AddInput(nodeA)
	graph.AddNode(nodeA)
	graph.AddNode(nodeB)

	var benchSinks []*Node

	b.ResetTimer()
	for b.Loop() {
		benchSinks = graph.Sinks()
	}
	_ = benchSinks
}

func BenchmarkGraph_TopologyLayers(b *testing.B) {
	shape, err := tensor.NewShape([]int{2, 2})
	if err != nil {
		b.Fatalf("NewShape failed: %v", err)
	}

	graph := NewGraph()
	nodeA := NewNode("a", OpInput, shape)
	nodeB := NewNode("b", OpMatmul, shape)
	nodeB.AddInput(nodeA)
	graph.AddNode(nodeA)
	graph.AddNode(nodeB)

	b.ResetTimer()
	for b.Loop() {
		_, _ = graph.TopologyLayers()
	}
}

func BenchmarkGraph_Nodes(b *testing.B) {
	shape, err := tensor.NewShape([]int{2, 2})
	if err != nil {
		b.Fatalf("NewShape failed: %v", err)
	}

	graph := NewGraph()
	nodeA := NewNode("a", OpInput, shape)
	nodeB := NewNode("b", OpMatmul, shape)
	nodeB.AddInput(nodeA)
	graph.AddNode(nodeA)
	graph.AddNode(nodeB)

	b.ResetTimer()
	for b.Loop() {
		_ = graph.Nodes()
	}
}

func BenchmarkGraph_Verify(b *testing.B) {
	shape, err := tensor.NewShape([]int{2, 2})
	if err != nil {
		b.Fatalf("NewShape failed: %v", err)
	}

	graph := NewGraph()
	nodeA := NewNode("a", OpInput, shape)
	nodeB := NewNode("b", OpInput, shape)
	nodeC := NewNode("c", OpAdd, shape)
	nodeC.AddInput(nodeA)
	nodeC.AddInput(nodeB)
	graph.AddNode(nodeA)
	graph.AddNode(nodeB)
	graph.AddNode(nodeC)

	for b.Loop() {
		if err := graph.Verify(); err != nil {
			b.Fatal(err)
		}
	}
}
