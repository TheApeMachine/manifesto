package ir

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/tensor"
)

func TestNode(t *testing.T) {
	Convey("Given a new Node", t, func() {
		shape, err := tensor.NewShape([]int{2, 2})
		So(err, ShouldBeNil)

		node := NewNode("n1", OpMatmul, shape)

		Convey("It should return its ID", func() {
			So(node.ID(), ShouldEqual, "n1")
		})

		Convey("It should return its OpType", func() {
			So(node.OpType(), ShouldEqual, OpMatmul)
		})

		Convey("It should return its Shape", func() {
			So(node.Shape().Equal(shape), ShouldBeTrue)
		})

		Convey("It should allow adding inputs", func() {
			inputShape, err := tensor.NewShape([]int{2, 2})
			So(err, ShouldBeNil)
			inputNode := NewNode("in1", OpInput, inputShape)

			node.AddInput(inputNode)
			So(len(node.Inputs()), ShouldEqual, 1)
			So(node.Inputs()[0].ID(), ShouldEqual, "in1")
		})

		Convey("It should allow setting and getting metadata", func() {
			node.SetMetadata("activation", "relu")
			metadata := node.Metadata()
			So(metadata["activation"], ShouldEqual, "relu")
		})

		Convey("It should allow configuring buffer reuse", func() {
			So(node.InPlace(), ShouldBeFalse)
			node.SetInPlace(true)
			So(node.InPlace(), ShouldBeTrue)
		})
	})
}

func BenchmarkNode(b *testing.B) {
	shape, err := tensor.NewShape([]int{2, 2})
	if err != nil {
		b.Fatalf("NewShape failed: %v", err)
	}
	inputShape, err := tensor.NewShape([]int{2, 2})
	if err != nil {
		b.Fatalf("NewShape failed: %v", err)
	}

	inputNode := NewNode("in1", OpInput, inputShape)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		node := NewNode("n1", OpMatmul, shape)
		node.AddInput(inputNode)
		node.SetMetadata("key", "value")
		node.SetInPlace(true)
		node.InPlace()
	}
}
