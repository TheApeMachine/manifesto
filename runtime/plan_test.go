package runtime

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ir/dag"
	"github.com/theapemachine/manifesto/tensor"
)

func TestNewExecutionPlan(testingObject *testing.T) {
	convey.Convey("Given a verified matmul graph", testingObject, func() {
		leftShape, err := tensor.NewShape([]int{2, 3})
		convey.So(err, convey.ShouldBeNil)

		rightShape, err := tensor.NewShape([]int{3, 4})
		convey.So(err, convey.ShouldBeNil)

		outputShape, err := tensor.NewShape([]int{2, 4})
		convey.So(err, convey.ShouldBeNil)

		left := dag.NewNode("left", dag.OpInput, leftShape)
		right := dag.NewNode("right", dag.OpInput, rightShape)
		matmulNode := dag.NewNode("matmul", dag.OpMatmul, outputShape)
		matmulNode.AddInput(left)
		matmulNode.AddInput(right)

		computeGraph := dag.NewGraph()
		computeGraph.AddNode(left)
		computeGraph.AddNode(right)
		computeGraph.AddNode(matmulNode)

		convey.Convey("It should derive one compute layer", func() {
			plan, err := NewExecutionPlan("demo", computeGraph)

			convey.So(err, convey.ShouldBeNil)
			convey.So(plan.GraphName, convey.ShouldEqual, "demo")
			convey.So(plan.Layers, convey.ShouldResemble, [][]string{{"matmul"}})
		})
	})
}

func BenchmarkNewExecutionPlan(benchmark *testing.B) {
	leftShape, _ := tensor.NewShape([]int{64, 64})
	rightShape, _ := tensor.NewShape([]int{64, 64})
	outputShape, _ := tensor.NewShape([]int{64, 64})

	left := dag.NewNode("left", dag.OpInput, leftShape)
	right := dag.NewNode("right", dag.OpInput, rightShape)
	matmulNode := dag.NewNode("matmul", dag.OpMatmul, outputShape)
	matmulNode.AddInput(left)
	matmulNode.AddInput(right)

	computeGraph := dag.NewGraph()
	computeGraph.AddNode(left)
	computeGraph.AddNode(right)
	computeGraph.AddNode(matmulNode)

	for benchmark.Loop() {
		_, err := NewExecutionPlan("demo", computeGraph)

		if err != nil {
			benchmark.Fatal(err)
		}
	}
}
