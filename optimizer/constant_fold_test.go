package optimizer

import (
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
)

func TestConstantFoldBinaryAdd(t *testing.T) {
	convey.Convey("Given two math.constant nodes feeding a math.add", t, func() {
		graph := &ast.Graph{
			Inputs:  []string{},
			Outputs: map[string]string{"y": "sum"},
			Nodes: []*ast.GraphNode{
				{
					ID:         "a",
					Op:         "math.constant",
					Attributes: map[string]any{ConstantAttributeValue: 1.5},
				},
				{
					ID:         "b",
					Op:         "math.constant",
					Attributes: map[string]any{ConstantAttributeValue: 2.5},
				},
				{
					ID:     "sum",
					Op:     "math.add",
					Inputs: []string{"a", "b"},
				},
			},
		}

		stats, err := ConstantFold(graph)

		convey.So(err, convey.ShouldBeNil)
		convey.So(stats.NodesFolded, convey.ShouldEqual, 1)

		convey.Convey("Then the math.add becomes a math.constant with the sum", func() {
			sumNode := graph.Nodes[2]
			convey.So(sumNode.Op, convey.ShouldEqual, "math.constant")
			convey.So(sumNode.Inputs, convey.ShouldBeEmpty)
			convey.So(sumNode.Attributes[ConstantAttributeValue], convey.ShouldEqual, 4.0)
		})
	})
}

func TestConstantFoldChainsTransitively(t *testing.T) {
	convey.Convey("Given a chain a -> mul(a, 2) -> relu", t, func() {
		graph := &ast.Graph{
			Outputs: map[string]string{"y": "relu_out"},
			Nodes: []*ast.GraphNode{
				{
					ID:         "a",
					Op:         "math.constant",
					Attributes: map[string]any{ConstantAttributeValue: -3.0},
				},
				{
					ID:         "two",
					Op:         "math.constant",
					Attributes: map[string]any{ConstantAttributeValue: 2.0},
				},
				{
					ID:     "scaled",
					Op:     "math.mul",
					Inputs: []string{"a", "two"},
				},
				{
					ID:     "relu_out",
					Op:     "activation.relu",
					Inputs: []string{"scaled"},
				},
			},
		}

		stats, err := ConstantFold(graph)

		convey.So(err, convey.ShouldBeNil)
		convey.So(stats.NodesFolded, convey.ShouldEqual, 2)

		convey.Convey("Then both elementwise nodes collapse to constants", func() {
			convey.So(graph.Nodes[2].Op, convey.ShouldEqual, "math.constant")
			convey.So(graph.Nodes[2].Attributes[ConstantAttributeValue], convey.ShouldEqual, -6.0)
			convey.So(graph.Nodes[3].Op, convey.ShouldEqual, "math.constant")
			convey.So(graph.Nodes[3].Attributes[ConstantAttributeValue], convey.ShouldEqual, 0.0)
		})
	})
}

func TestConstantFoldLeavesNonConstantInputsAlone(t *testing.T) {
	convey.Convey("Given a math.add where one operand is a graph input", t, func() {
		graph := &ast.Graph{
			Inputs:  []string{"x"},
			Outputs: map[string]string{"y": "sum"},
			Nodes: []*ast.GraphNode{
				{
					ID:         "k",
					Op:         "math.constant",
					Attributes: map[string]any{ConstantAttributeValue: 7.0},
				},
				{
					ID:     "sum",
					Op:     "math.add",
					Inputs: []string{"x", "k"},
				},
			},
		}

		stats, err := ConstantFold(graph)

		convey.So(err, convey.ShouldBeNil)
		convey.So(stats.NodesFolded, convey.ShouldEqual, 0)
		convey.So(graph.Nodes[1].Op, convey.ShouldEqual, "math.add")
	})
}

func TestConstantFoldUnaryEvaluations(t *testing.T) {
	convey.Convey("Given unary elementwise ops over a constant", t, func() {
		graph := &ast.Graph{
			Outputs: map[string]string{"y": "out"},
			Nodes: []*ast.GraphNode{
				{ID: "a", Op: "math.constant", Attributes: map[string]any{ConstantAttributeValue: 0.5}},
				{ID: "exp_out", Op: "math.exp", Inputs: []string{"a"}},
				{ID: "tanh_out", Op: "activation.tanh", Inputs: []string{"a"}},
				{ID: "out", Op: "activation.sigmoid", Inputs: []string{"a"}},
			},
		}

		stats, err := ConstantFold(graph)

		convey.So(err, convey.ShouldBeNil)
		convey.So(stats.NodesFolded, convey.ShouldEqual, 3)

		convey.Convey("Then exp/tanh/sigmoid all evaluate", func() {
			convey.So(graph.Nodes[1].Attributes[ConstantAttributeValue], convey.ShouldAlmostEqual, math.Exp(0.5), 1e-9)
			convey.So(graph.Nodes[2].Attributes[ConstantAttributeValue], convey.ShouldAlmostEqual, math.Tanh(0.5), 1e-9)
			convey.So(graph.Nodes[3].Attributes[ConstantAttributeValue], convey.ShouldAlmostEqual, 1.0/(1.0+math.Exp(-0.5)), 1e-9)
		})
	})
}

func TestRunPipelineFoldsThenFuses(t *testing.T) {
	convey.Convey("Given a graph where one operand is constant and another is a graph input", t, func() {
		graph := &ast.Graph{
			Inputs:  []string{"x"},
			Outputs: map[string]string{"y": "relu_out"},
			Nodes: []*ast.GraphNode{
				{ID: "k", Op: "math.constant", Attributes: map[string]any{ConstantAttributeValue: 2.0}},
				{ID: "scaled", Op: "math.mul", Inputs: []string{"x", "k"}},
				{ID: "relu_out", Op: "activation.relu", Inputs: []string{"scaled"}},
			},
		}

		stats, err := Run(graph, Options{})

		convey.So(err, convey.ShouldBeNil)

		convey.Convey("Then constant-fold leaves the chain alone (x is non-constant) but fusion still kicks in", func() {
			convey.So(stats.ConstantFold.NodesFolded, convey.ShouldEqual, 0)
			convey.So(stats.Fusion.Clusters, convey.ShouldEqual, 1)
		})
	})
}
