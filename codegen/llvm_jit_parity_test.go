//go:build codegen_llvm

package codegen

import (
	"fmt"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/codegen/llvm"
	"github.com/theapemachine/manifesto/optimizer"
)

var paritySizes = []int{1, 7, 64, 1024, 8192}

func TestEmitCPUJITMatchesScalarReference(testingObject *testing.T) {
	convey.Convey("Given ReLU(Add(x, y)) lowered to LLVM MCJIT", testingObject, func() {
		fusion := reluAddFusion()
		runner, err := EmitCPU(fusion)
		convey.So(err, convey.ShouldBeNil)
		defer runner.(*llvmCPUKernel).Close()

		reference, err := EmitReferenceCPU(fusion)
		convey.So(err, convey.ShouldBeNil)

		for _, count := range paritySizes {
			convey.Convey(fmt.Sprintf("It should match the scalar reference at N=%d", count), func() {
				left, right, output, expected := parityBuffers(count, fusion, reference)

				err := runner.Run([][]float32{left, right}, output, count)
				convey.So(err, convey.ShouldBeNil)
				convey.So(output, convey.ShouldResemble, expected)
			})
		}
	})
}

func TestEmitCPUJITSwiGLUParity(testingObject *testing.T) {
	convey.Convey("Given Mul(Sigmoid(gate), up)", testingObject, func() {
		fusion := swigluFusion()
		runner, err := EmitCPU(fusion)
		convey.So(err, convey.ShouldBeNil)
		defer runner.(*llvmCPUKernel).Close()

		reference, err := EmitReferenceCPU(fusion)
		convey.So(err, convey.ShouldBeNil)

		for _, count := range paritySizes {
			convey.Convey(fmt.Sprintf("It should match the scalar reference at N=%d", count), func() {
				gate, up, output, expected := parityBuffers(count, fusion, reference)

				err := runner.Run([][]float32{gate, up}, output, count)
				convey.So(err, convey.ShouldBeNil)

				for index := range output {
					convey.So(float64(output[index]), convey.ShouldAlmostEqual, float64(expected[index]), 1e-5)
				}
			})
		}
	})
}

func TestEmitKernelDirectMatchesScalarReference(testingObject *testing.T) {
	convey.Convey("Given ReLU(Add(x, y)) via llvm.EmitKernel", testingObject, func() {
		fusion := reluAddFusion()
		kernel, err := llvm.EmitKernel(fusion)
		convey.So(err, convey.ShouldBeNil)
		defer kernel.Close()

		reference, err := EmitReferenceCPU(fusion)
		convey.So(err, convey.ShouldBeNil)

		for _, count := range paritySizes {
			convey.Convey(fmt.Sprintf("It should match the scalar reference at N=%d", count), func() {
				left, right, output, expected := parityBuffers(count, fusion, reference)

				err := kernel.Run([][]float32{left, right}, output, count)
				convey.So(err, convey.ShouldBeNil)
				convey.So(output, convey.ShouldResemble, expected)
			})
		}
	})
}

func TestEmitKernelForLevelParity(testingObject *testing.T) {
	convey.Convey("Given ReLU(Add) at each host ISA tier", testingObject, func() {
		fusion := reluAddFusion()
		reference, err := EmitReferenceCPU(fusion)
		convey.So(err, convey.ShouldBeNil)

		for _, level := range llvm.SupportedISALevelsOnHost() {
			convey.Convey(fmt.Sprintf("It should match scalar reference at %s", level), func() {
				kernel, err := llvm.EmitKernelForLevel(fusion, level)
				convey.So(err, convey.ShouldBeNil)
				defer kernel.Close()

				for _, count := range paritySizes {
					left, right, output, expected := parityBuffers(count, fusion, reference)

					err := kernel.Run([][]float32{left, right}, output, count)
					convey.So(err, convey.ShouldBeNil)
					convey.So(output, convey.ShouldResemble, expected)
				}
			})
		}
	})
}

func TestEmitCPUJITThreeInputResidual(testingObject *testing.T) {
	convey.Convey("Given ReLU(Add(Add(a, b), c))", testingObject, func() {
		fusion := residualAddFusion()
		runner, err := EmitCPU(fusion)
		convey.So(err, convey.ShouldBeNil)
		defer runner.(*llvmCPUKernel).Close()

		reference, err := EmitReferenceCPU(fusion)
		convey.So(err, convey.ShouldBeNil)

		for _, count := range paritySizes {
			convey.Convey(fmt.Sprintf("It should match the scalar reference at N=%d", count), func() {
				left, middle, right, output, expected := parityBuffersThree(count, fusion, reference)

				err := runner.Run([][]float32{left, middle, right}, output, count)
				convey.So(err, convey.ShouldBeNil)
				convey.So(output, convey.ShouldResemble, expected)
			})
		}
	})
}

func BenchmarkEmitCPUJITRun(benchmark *testing.B) {
	fusion := reluAddFusion()
	runner, err := EmitCPU(fusion)

	if err != nil {
		benchmark.Skip(err)
	}

	jitKernel := runner.(*llvmCPUKernel)
	defer jitKernel.Close()

	count := 8192
	left := make([]float32, count)
	right := make([]float32, count)
	output := make([]float32, count)

	for index := range left {
		left[index] = float32(index%11) - 5
		right[index] = float32(index % 7)
	}

	benchmark.ResetTimer()

	for benchmark.Loop() {
		if err := runner.Run([][]float32{left, right}, output, count); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func reluAddFusion() *optimizer.FusionAST {
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

func residualAddFusion() *optimizer.FusionAST {
	return &optimizer.FusionAST{
		InputPorts: []string{"a", "b", "c"},
		OutputPort: "residual",
		Root: &optimizer.ASTNode{
			Type: optimizer.NodeReLU,
			Children: []*optimizer.ASTNode{
				{
					Type: optimizer.NodeAdd,
					Children: []*optimizer.ASTNode{
						{
							Type: optimizer.NodeAdd,
							Children: []*optimizer.ASTNode{
								{Type: optimizer.NodeInput, InputIndex: 0},
								{Type: optimizer.NodeInput, InputIndex: 1},
							},
						},
						{Type: optimizer.NodeInput, InputIndex: 2},
					},
				},
			},
		},
	}
}

func swigluFusion() *optimizer.FusionAST {
	return &optimizer.FusionAST{
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
}

func parityBuffers(
	count int,
	fusion *optimizer.FusionAST,
	reference *CPUKernel,
) (left []float32, right []float32, output []float32, expected []float32) {
	left = make([]float32, count)
	right = make([]float32, count)
	output = make([]float32, count)
	expected = make([]float32, count)

	for index := 0; index < count; index++ {
		left[index] = float32(math.Sin(float64(index)))
		right[index] = float32(math.Cos(float64(index)))
	}

	if err := reference.Run([][]float32{left, right}, expected, count); err != nil {
		panic(err)
	}

	return left, right, output, expected
}

func parityBuffersThree(
	count int,
	fusion *optimizer.FusionAST,
	reference *CPUKernel,
) (left []float32, middle []float32, right []float32, output []float32, expected []float32) {
	left = make([]float32, count)
	middle = make([]float32, count)
	right = make([]float32, count)
	output = make([]float32, count)
	expected = make([]float32, count)

	for index := 0; index < count; index++ {
		left[index] = float32(math.Sin(float64(index)))
		middle[index] = float32(math.Cos(float64(index)))
		right[index] = float32(math.Tan(float64(index) * 0.01))
	}

	if err := reference.Run([][]float32{left, middle, right}, expected, count); err != nil {
		panic(err)
	}

	return left, middle, right, output, expected
}
