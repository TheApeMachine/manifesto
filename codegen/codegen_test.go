package codegen

import (
	"math"
	"strings"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/optimizer"
)

func TestEmitCPURunsFusedChain(t *testing.T) {
	convey.Convey("Given a fusion of ReLU(Add(x, y))", t, func() {
		fusion := &optimizer.FusionAST{
			InputPorts: []string{"x", "y"},
			OutputPort: "result",
			Root: &optimizer.ASTNode{
				Type: optimizer.NodeReLU,
				Children: []*optimizer.ASTNode{
					{
						Type: optimizer.NodeAdd,
						Children: []*optimizer.ASTNode{
							{Type: optimizer.NodeInput, InputIndex: 0},
							{Type: optimizer.NodeInput, InputIndex: 1},
						},
					},
				},
			},
		}

		kernel, err := EmitCPU(fusion)

		convey.So(err, convey.ShouldBeNil)
		convey.So(kernel.Target(), convey.ShouldEqual, TargetCPU)
		convey.So(kernel.Identifier(), convey.ShouldEqual, "result")

		convey.Convey("Then Run evaluates the expression elementwise", func() {
			inputs := [][]float32{
				{1, -3, 5, -2},
				{2, 1, -4, 1},
			}
			output := make([]float32, 4)

			err := kernel.Run(inputs, output, 4)
			convey.So(err, convey.ShouldBeNil)

			expected := []float32{3, 0, 1, 0}
			convey.So(output, convey.ShouldResemble, expected)
		})

		convey.Convey("And it rejects mismatched input counts", func() {
			err := kernel.Run([][]float32{{1, 2}}, make([]float32, 2), 2)
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldContainSubstring, "expects 2 inputs")
		})
	})
}

func TestEmitCPUEvaluatesActivations(t *testing.T) {
	convey.Convey("Given fusions for sigmoid, tanh, and gelu over a single input", t, func() {
		fusion := func(nodeType optimizer.NodeType) *optimizer.FusionAST {
			return &optimizer.FusionAST{
				InputPorts: []string{"x"},
				OutputPort: nodeType.String(),
				Root: &optimizer.ASTNode{
					Type: nodeType,
					Children: []*optimizer.ASTNode{
						{Type: optimizer.NodeInput, InputIndex: 0},
					},
				},
			}
		}

		inputs := [][]float32{{0.0, 1.0, -1.0, 2.0}}
		output := make([]float32, 4)

		convey.Convey("Sigmoid matches the float64 reference", func() {
			kernel, err := EmitCPU(fusion(optimizer.NodeSigmoid))
			convey.So(err, convey.ShouldBeNil)
			err = kernel.Run(inputs, output, 4)
			convey.So(err, convey.ShouldBeNil)

			for i, value := range inputs[0] {
				want := 1.0 / (1.0 + math.Exp(-float64(value)))
				convey.So(float64(output[i]), convey.ShouldAlmostEqual, want, 1e-5)
			}
		})

		convey.Convey("Tanh matches the math.Tanh reference", func() {
			kernel, err := EmitCPU(fusion(optimizer.NodeTanh))
			convey.So(err, convey.ShouldBeNil)
			err = kernel.Run(inputs, output, 4)
			convey.So(err, convey.ShouldBeNil)

			for i, value := range inputs[0] {
				want := math.Tanh(float64(value))
				convey.So(float64(output[i]), convey.ShouldAlmostEqual, want, 1e-5)
			}
		})
	})
}

func TestEmitMetalSourceShape(t *testing.T) {
	convey.Convey("Given a fusion of Mul(Sigmoid(gate), up)", t, func() {
		fusion := &optimizer.FusionAST{
			InputPorts: []string{"gate", "up"},
			OutputPort: "swiglu_out",
			Root: &optimizer.ASTNode{
				Type: optimizer.NodeMul,
				Children: []*optimizer.ASTNode{
					{
						Type: optimizer.NodeSigmoid,
						Children: []*optimizer.ASTNode{
							{Type: optimizer.NodeInput, InputIndex: 0},
						},
					},
					{Type: optimizer.NodeInput, InputIndex: 1},
				},
			},
		}

		kernel, err := EmitMetal(fusion)

		convey.So(err, convey.ShouldBeNil)
		convey.So(kernel.Target(), convey.ShouldEqual, TargetMetal)

		convey.Convey("Then the MSL source has the expected structure", func() {
			source := kernel.Source()
			convey.So(source, convey.ShouldContainSubstring, "#include <metal_stdlib>")
			convey.So(source, convey.ShouldContainSubstring, "kernel void fused_swiglu_out(")
			convey.So(source, convey.ShouldContainSubstring, "device const float* in0 [[buffer(0)]]")
			convey.So(source, convey.ShouldContainSubstring, "device const float* in1 [[buffer(1)]]")
			convey.So(source, convey.ShouldContainSubstring, "device float* out [[buffer(2)]]")
			convey.So(source, convey.ShouldContainSubstring, "thread_position_in_grid")
			convey.So(source, convey.ShouldContainSubstring, "if (id >= count) return;")
			convey.So(source, convey.ShouldContainSubstring, "exp(-(in0[id]))")
			convey.So(source, convey.ShouldContainSubstring, "in1[id]")
		})

		convey.Convey("And the kernel name is buffer-cache friendly", func() {
			convey.So(kernel.KernelName(), convey.ShouldEqual, "fused_swiglu_out")
		})
	})
}

func TestAttachKernelsToGraph(t *testing.T) {
	convey.Convey("Given a graph with one FuseOp node", t, func() {
		fusion := &optimizer.FusionAST{
			InputPorts: []string{"x"},
			OutputPort: "y",
			Root: &optimizer.ASTNode{
				Type: optimizer.NodeReLU,
				Children: []*optimizer.ASTNode{
					{Type: optimizer.NodeInput, InputIndex: 0},
				},
			},
		}

		graph := &ast.Graph{
			Inputs:  []string{"x"},
			Outputs: map[string]string{"out": "y"},
			Nodes: []*ast.GraphNode{
				{
					ID:     "y",
					Op:     optimizer.FuseOp,
					Inputs: []string{"x"},
					Attributes: map[string]any{
						optimizer.FuseAttributeAST: fusion,
					},
				},
			},
		}

		attached, err := AttachKernels(graph, EmitOptions{})

		convey.So(err, convey.ShouldBeNil)
		convey.So(attached, convey.ShouldEqual, 1)

		convey.Convey("Then the FuseOp node carries a KernelSet with both targets", func() {
			set, ok := graph.Nodes[0].Attributes[KernelAttribute].(*KernelSet)
			convey.So(ok, convey.ShouldBeTrue)

			cpu := set.For(TargetCPU)
			metal := set.For(TargetMetal)
			convey.So(cpu, convey.ShouldNotBeNil)
			convey.So(metal, convey.ShouldNotBeNil)

			cpuKernel, ok := cpu.(*CPUKernel)
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(cpuKernel.Output(), convey.ShouldEqual, "y")

			metalKernel, ok := metal.(*MetalKernel)
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(strings.Contains(metalKernel.Source(), "fmax(0.0f"), convey.ShouldBeTrue)
		})
	})
}
