package llvm

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/optimizer"
)

func reluAddFusionAST() *optimizer.FusionAST {
	return &optimizer.FusionAST{
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
}

func TestBuildModuleIRForReluAdd(testingObject *testing.T) {
	convey.Convey("Given ReLU(Add(x, y)) fusion", testingObject, func() {
		fusion := reluAddFusionAST()

		irText, err := ModuleIR(fusion, "fused_result")
		convey.So(err, convey.ShouldBeNil)

		vectorWidth := HostVectorWidth()

		convey.Convey("It should emit explicit SIMD vector IR with a scalar tail", func() {
			convey.So(irText, convey.ShouldContainSubstring, "define void @fused_result")
			convey.So(irText, convey.ShouldContainSubstring, "float** %inputs")
			convey.So(irText, convey.ShouldContainSubstring, fmt.Sprintf("<%d x float>", vectorWidth))
			convey.So(irText, convey.ShouldContainSubstring, fmt.Sprintf("fadd <%d x float>", vectorWidth))
			convey.So(irText, convey.ShouldContainSubstring, "vector.loop")
			convey.So(irText, convey.ShouldContainSubstring, "scalar.loop")
			convey.So(irText, convey.ShouldContainSubstring, "fadd float")
		})
	})
}

func TestFusionVectorizable(testingObject *testing.T) {
	convey.Convey("Given fusion vectorizability rules", testingObject, func() {
		convey.Convey("It should vectorize ReLU(Add)", func() {
			convey.So(fusionVectorizable(reluAddFusionAST().Root), convey.ShouldBeTrue)
		})

		convey.Convey("It should not vectorize SwiGLU because of Sigmoid", func() {
			swiglu := &optimizer.ASTNode{
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
			}
			convey.So(fusionVectorizable(swiglu), convey.ShouldBeFalse)
		})
	})
}

func TestHostCPUFeatures(testingObject *testing.T) {
	convey.Convey("Given host CPU feature detection", testingObject, func() {
		convey.Convey("It should return a non-empty feature string", func() {
			features := HostCPUFeatures()
			convey.So(features, convey.ShouldNotBeBlank)
		})
	})
}
