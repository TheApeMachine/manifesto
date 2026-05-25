package ir

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
)

func makeOpNode(name string, op Operation, inputs, outputs []*Port) *Node {
	return &Node{
		Name:      name,
		Inputs:    inputs,
		Outputs:   outputs,
		Operation: op,
	}
}

func TestFusionNodeTypeStringEnumerates(t *testing.T) {
	convey.Convey("Given each FusionNodeType constant", t, func() {
		cases := map[FusionNodeType]string{
			NodeInput:    "Input",
			NodeConstant: "Constant",
			NodeAdd:      "Add",
			NodeMul:      "Mul",
			NodeReLU:     "ReLU",
			NodeSigmoid:  "Sigmoid",
		}

		for nodeType, expected := range cases {
			nodeType := nodeType
			expected := expected

			convey.Convey("It renders "+expected+" correctly", func() {
				convey.So(nodeType.String(), convey.ShouldEqual, expected)
			})
		}
	})
}

func TestIsFusibleElementwiseRecognizesKnownOps(t *testing.T) {
	convey.Convey("Given the fusion-op registry", t, func() {
		convey.Convey("Known elementwise ops are fusible", func() {
			convey.So(IsFusibleElementwise(OperationAdd), convey.ShouldBeTrue)
			convey.So(IsFusibleElementwise(OperationMul), convey.ShouldBeTrue)
			convey.So(IsFusibleElementwise(OperationReLU), convey.ShouldBeTrue)
			convey.So(IsFusibleElementwise(OperationSigmoid), convey.ShouldBeTrue)
		})

		convey.Convey("Non-elementwise ops are not fusible", func() {
			convey.So(IsFusibleElementwise(OperationMatmul), convey.ShouldBeFalse)
			convey.So(IsFusibleElementwise(OperationScaledDotProductAttention), convey.ShouldBeFalse)
			convey.So(IsFusibleElementwise(OperationLayerNorm), convey.ShouldBeFalse)
		})

		convey.Convey("Zero/uninitialized Operation is not fusible", func() {
			convey.So(IsFusibleElementwise(Operation(0)), convey.ShouldBeFalse)
		})
	})
}

func TestFusionNodeTypeForOpReturnsMapping(t *testing.T) {
	convey.Convey("Given FusionNodeTypeForOp", t, func() {
		convey.Convey("OperationAdd → NodeAdd", func() {
			nodeType, ok := FusionNodeTypeForOp(OperationAdd)
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(nodeType, convey.ShouldEqual, NodeAdd)
		})

		convey.Convey("Non-fusible op → (NodeInput, false)", func() {
			nodeType, ok := FusionNodeTypeForOp(OperationScaledDotProductAttention)
			convey.So(ok, convey.ShouldBeFalse)
			convey.So(nodeType, convey.ShouldEqual, NodeInput)
		})
	})
}

func TestFindFusionClustersSingleNode(t *testing.T) {
	convey.Convey("Given a single elementwise node with non-fusible producer", t, func() {
		input := makePortWithType(dtype.Float32, 4, 256)
		intermediate := makePortWithType(dtype.Float32, 4, 256)
		output := makePortWithType(dtype.Float32, 4, 256)

		topology := &Topology{
			Nodes: []*Node{
				makeOpNode("matmul", OperationMatmul, []*Port{input}, []*Port{intermediate}),
				makeOpNode("relu", OperationReLU, []*Port{intermediate}, []*Port{output}),
			},
		}

		AssignPortIDs(topology)

		clusters := FindFusionClusters(topology)

		convey.Convey("FindFusionClusters returns one cluster containing only the ReLU", func() {
			convey.So(len(clusters), convey.ShouldEqual, 1)
			convey.So(clusters[0].Root.Type, convey.ShouldEqual, NodeReLU)
			convey.So(len(clusters[0].Root.Children), convey.ShouldEqual, 1)
			convey.So(clusters[0].Root.Children[0].Type, convey.ShouldEqual, NodeInput)
			convey.So(len(clusters[0].InputPorts), convey.ShouldEqual, 1)
			convey.So(clusters[0].InputPorts[0], convey.ShouldEqual, intermediate.ID)
		})

		convey.Convey("OutputPort matches the cluster's output", func() {
			convey.So(clusters[0].OutputPort, convey.ShouldEqual, output.ID)
		})
	})
}

func TestFindFusionClustersChain(t *testing.T) {
	convey.Convey("Given add → mul → relu (all elementwise, each producer single-consumer)", t, func() {
		left := makePortWithType(dtype.Float32, 4, 256)
		right := makePortWithType(dtype.Float32, 4, 256)
		scale := makePortWithType(dtype.Float32, 4, 256)
		afterAdd := makePortWithType(dtype.Float32, 4, 256)
		afterMul := makePortWithType(dtype.Float32, 4, 256)
		afterReLU := makePortWithType(dtype.Float32, 4, 256)

		topology := &Topology{
			Nodes: []*Node{
				makeOpNode("add", OperationAdd, []*Port{left, right}, []*Port{afterAdd}),
				makeOpNode("mul", OperationMul, []*Port{afterAdd, scale}, []*Port{afterMul}),
				makeOpNode("relu", OperationReLU, []*Port{afterMul}, []*Port{afterReLU}),
			},
		}

		AssignPortIDs(topology)

		clusters := FindFusionClusters(topology)

		convey.Convey("All three nodes collapse into one cluster", func() {
			convey.So(len(clusters), convey.ShouldEqual, 1)
		})

		convey.Convey("Root is the terminal ReLU, recursing through Mul and Add", func() {
			root := clusters[0].Root
			convey.So(root.Type, convey.ShouldEqual, NodeReLU)
			convey.So(len(root.Children), convey.ShouldEqual, 1)

			mulNode := root.Children[0]
			convey.So(mulNode.Type, convey.ShouldEqual, NodeMul)
			convey.So(len(mulNode.Children), convey.ShouldEqual, 2)

			addNode := mulNode.Children[0]
			convey.So(addNode.Type, convey.ShouldEqual, NodeAdd)
			convey.So(len(addNode.Children), convey.ShouldEqual, 2)
			convey.So(addNode.Children[0].Type, convey.ShouldEqual, NodeInput)
			convey.So(addNode.Children[1].Type, convey.ShouldEqual, NodeInput)

			scaleInput := mulNode.Children[1]
			convey.So(scaleInput.Type, convey.ShouldEqual, NodeInput)
		})

		convey.Convey("InputPorts list left, right, scale in dependency order", func() {
			convey.So(len(clusters[0].InputPorts), convey.ShouldEqual, 3)
			convey.So(clusters[0].InputPorts[0], convey.ShouldEqual, left.ID)
			convey.So(clusters[0].InputPorts[1], convey.ShouldEqual, right.ID)
			convey.So(clusters[0].InputPorts[2], convey.ShouldEqual, scale.ID)
		})
	})
}

func TestFindFusionClustersNonElementwiseBreaksChain(t *testing.T) {
	convey.Convey("Given add → matmul → relu (matmul breaks the chain)", t, func() {
		left := makePortWithType(dtype.Float32, 4, 256)
		right := makePortWithType(dtype.Float32, 4, 256)
		afterAdd := makePortWithType(dtype.Float32, 4, 256)
		weight := makePortWithType(dtype.Float32, 256, 768)
		afterMatmul := makePortWithType(dtype.Float32, 4, 768)
		afterReLU := makePortWithType(dtype.Float32, 4, 768)

		topology := &Topology{
			Nodes: []*Node{
				makeOpNode("add", OperationAdd, []*Port{left, right}, []*Port{afterAdd}),
				makeOpNode("matmul", OperationMatmul, []*Port{afterAdd, weight}, []*Port{afterMatmul}),
				makeOpNode("relu", OperationReLU, []*Port{afterMatmul}, []*Port{afterReLU}),
			},
		}

		AssignPortIDs(topology)

		clusters := FindFusionClusters(topology)

		convey.Convey("Two separate clusters emerge (add alone, then relu alone)", func() {
			convey.So(len(clusters), convey.ShouldEqual, 2)

			convey.So(clusters[0].Root.Type, convey.ShouldEqual, NodeAdd)
			convey.So(clusters[1].Root.Type, convey.ShouldEqual, NodeReLU)
		})
	})
}

func TestFindFusionClustersFanOutBreaksAbsorption(t *testing.T) {
	convey.Convey("Given add → [mul1, mul2] (add's output has two consumers)", t, func() {
		left := makePortWithType(dtype.Float32, 4, 256)
		right := makePortWithType(dtype.Float32, 4, 256)
		afterAdd := makePortWithType(dtype.Float32, 4, 256)
		scale1 := makePortWithType(dtype.Float32, 4, 256)
		scale2 := makePortWithType(dtype.Float32, 4, 256)
		out1 := makePortWithType(dtype.Float32, 4, 256)
		out2 := makePortWithType(dtype.Float32, 4, 256)

		topology := &Topology{
			Nodes: []*Node{
				makeOpNode("add", OperationAdd, []*Port{left, right}, []*Port{afterAdd}),
				makeOpNode("mul1", OperationMul, []*Port{afterAdd, scale1}, []*Port{out1}),
				makeOpNode("mul2", OperationMul, []*Port{afterAdd, scale2}, []*Port{out2}),
			},
		}

		AssignPortIDs(topology)

		clusters := FindFusionClusters(topology)

		convey.Convey("Three clusters emerge: add alone, mul1 alone, mul2 alone", func() {
			convey.So(len(clusters), convey.ShouldEqual, 3)
		})

		convey.Convey("Each mul cluster references afterAdd as a NodeInput (not absorbed)", func() {
			for _, cluster := range clusters {
				if cluster.Root.Type == NodeMul {
					convey.So(
						clusterReferencesPort(cluster, afterAdd.ID),
						convey.ShouldBeTrue,
					)
				}
			}
		})
	})
}

func TestFindFusionClustersOnEmptyTopology(t *testing.T) {
	convey.Convey("Given an empty topology", t, func() {
		clusters := FindFusionClusters(&Topology{})

		convey.Convey("FindFusionClusters returns nil", func() {
			convey.So(clusters, convey.ShouldBeNil)
		})
	})
}

func TestFusionASTSymbolicCountExpr(t *testing.T) {
	convey.Convey("Given a single ReLU on a [B, T, 768] tensor", t, func() {
		input := makePortWithType(dtype.Float32, "B", "T", 768)
		output := makePortWithType(dtype.Float32, "B", "T", 768)

		topology := &Topology{
			Nodes: []*Node{
				makeOpNode("relu", OperationReLU, []*Port{input}, []*Port{output}),
			},
		}

		AssignPortIDs(topology)

		clusters := FindFusionClusters(topology)

		convey.Convey("CountExpr is the symbolic product 'B * T * 768'", func() {
			convey.So(len(clusters), convey.ShouldEqual, 1)
			convey.So(clusters[0].CountExpr, convey.ShouldEqual, "B * T * 768")
		})
	})

	convey.Convey("Given a single ReLU on a fully-static [4, 256, 768] tensor", t, func() {
		input := makePortWithType(dtype.Float32, 4, 256, 768)
		output := makePortWithType(dtype.Float32, 4, 256, 768)

		topology := &Topology{
			Nodes: []*Node{
				makeOpNode("relu", OperationReLU, []*Port{input}, []*Port{output}),
			},
		}

		AssignPortIDs(topology)

		clusters := FindFusionClusters(topology)

		convey.Convey("CountExpr is empty (compile-time-known size)", func() {
			convey.So(len(clusters), convey.ShouldEqual, 1)
			convey.So(clusters[0].CountExpr, convey.ShouldEqual, "")
		})
	})
}

func clusterReferencesPort(cluster FusionAST, portID int32) bool {
	for _, inputPortID := range cluster.InputPorts {
		if inputPortID == portID {
			return true
		}
	}

	return false
}
