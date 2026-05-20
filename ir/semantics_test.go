package ir

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/manifesto/dtype"
)

func TestNodeSemantics(t *testing.T) {
	Convey("Given an IR node with typed compiler semantics", t, func() {
		shape, err := tensor.NewShape([]int{2, 3})
		So(err, ShouldBeNil)

		node := NewNode("projection", OpMatmul, shape)
		node.SetOperationID("math.matmul")
		node.SetValueType(ValueType{
			DType:       dtype.Float64,
			Shape:       shape,
			Layout:      LayoutRowMajor,
			MemoryClass: MemoryDevice,
		})
		node.SetEffect(EffectPure)
		node.SetAlias(Alias{
			Kind:       AliasAllocates,
			InPlace:    false,
			InputIndex: -1,
		})
		node.SetAttribute("beta", FloatAttribute(0.5))
		node.SetAttribute("alpha", IntAttribute(1))

		Convey("It should expose deterministic operation, type, effect, alias, and attributes", func() {
			So(node.OperationID(), ShouldEqual, OpID("math.matmul"))
			So(node.ValueType().DType, ShouldEqual, dtype.Float64)
			So(node.ValueType().Layout, ShouldEqual, LayoutRowMajor)
			So(node.ValueType().MemoryClass, ShouldEqual, MemoryDevice)
			So(node.Effect(), ShouldEqual, EffectPure)
			So(node.Alias().Kind, ShouldEqual, AliasAllocates)
			So(node.CanonicalAttributes(), ShouldEqual, "alpha=i:1;beta=f:0.5;")
		})
	})
}

func TestGraphVerify(t *testing.T) {
	Convey("Given an IR graph verifier", t, func() {
		shape, err := tensor.NewShape([]int{1})
		So(err, ShouldBeNil)

		Convey("It should reject inputs that are not registered in the graph", func() {
			graph := NewGraph()
			input := NewNode("input", OpInput, shape)
			output := NewNode("output", OpReLU, shape)
			output.AddInput(input)
			graph.AddNode(output)

			err := graph.Verify()

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "unregistered input")
		})

		Convey("It should build indexes and clone graphs while preserving target remaps", func() {
			graph := NewGraph()
			input := NewNode("input", OpInput, shape)
			output := NewNode("output", OpReLU, shape)
			output.AddInput(input)
			graph.AddNode(input)
			graph.AddNode(output)

			index, err := graph.Index()
			So(err, ShouldBeNil)
			So(index.Node("input"), ShouldEqual, input)
			So(index.Users("input"), ShouldHaveLength, 1)

			clone, replacements, err := graph.Clone()
			So(err, ShouldBeNil)
			So(clone, ShouldNotBeNil)
			So(replacements["output"] == output, ShouldBeFalse)
			So(replacements["output"].Inputs()[0].ID(), ShouldEqual, "input")
		})

		Convey("It should reject elementwise shape mismatches", func() {
			leftShape, err := tensor.NewShape([]int{2, 2})
			So(err, ShouldBeNil)
			rightShape, err := tensor.NewShape([]int{2, 3})
			So(err, ShouldBeNil)

			graph := NewGraph()
			left := NewNode("left", OpInput, leftShape)
			right := NewNode("right", OpInput, rightShape)
			add := NewNode("add", OpAdd, leftShape)
			add.AddInput(left)
			add.AddInput(right)
			graph.AddNode(left)
			graph.AddNode(right)
			graph.AddNode(add)

			err = graph.Verify()

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "shape")
			So(err.Error(), ShouldContainSubstring, "incompatible")
		})

		Convey("It should reject invalid matmul dimensions", func() {
			leftShape, err := tensor.NewShape([]int{2, 3})
			So(err, ShouldBeNil)
			rightShape, err := tensor.NewShape([]int{4, 2})
			So(err, ShouldBeNil)
			outputShape, err := tensor.NewShape([]int{2, 2})
			So(err, ShouldBeNil)

			graph := NewGraph()
			left := NewNode("left", OpInput, leftShape)
			right := NewNode("right", OpInput, rightShape)
			matmul := NewNode("matmul", OpMatmul, outputShape)
			matmul.AddInput(left)
			matmul.AddInput(right)
			graph.AddNode(left)
			graph.AddNode(right)
			graph.AddNode(matmul)

			err = graph.Verify()

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "incompatible inner dimensions")
		})

		Convey("It should reject wrong operation arity", func() {
			graph := NewGraph()
			input := NewNode("input", OpInput, shape)
			add := NewNode("add", OpAdd, shape)
			add.AddInput(input)
			graph.AddNode(input)
			graph.AddNode(add)

			err := graph.Verify()

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "requires 2 inputs")
		})

		Convey("It should accept compatible fused matmul bias shapes", func() {
			leftShape, err := tensor.NewShape([]int{2, 3})
			So(err, ShouldBeNil)
			rightShape, err := tensor.NewShape([]int{3, 4})
			So(err, ShouldBeNil)
			biasShape, err := tensor.NewShape([]int{4})
			So(err, ShouldBeNil)
			outputShape, err := tensor.NewShape([]int{2, 4})
			So(err, ShouldBeNil)

			graph := NewGraph()
			left := NewNode("left", OpInput, leftShape)
			right := NewNode("right", OpInput, rightShape)
			bias := NewNode("bias", OpInput, biasShape)
			fused := NewNode("fused", OpFused, outputShape)
			fused.AddInput(left)
			fused.AddInput(right)
			fused.AddInput(bias)
			graph.AddNode(left)
			graph.AddNode(right)
			graph.AddNode(bias)
			graph.AddNode(fused)

			So(graph.Verify(), ShouldBeNil)
		})
	})
}
