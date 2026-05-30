//go:build darwin && cgo

package codegen

import (
	"fmt"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/optimizer"
)

var metalParitySizes = []int{1, 7, 64, 1024, 8192}

func TestEmitMetalRunnerMatchesScalarReference(testingObject *testing.T) {
	convey.Convey("Given ReLU(Add(x, y)) compiled with MTLLibrary", testingObject, func() {
		fusion := reluAddFusionMetal()
		runner, err := EmitMetalRunner(fusion)
		convey.So(err, convey.ShouldBeNil)
		defer runner.(*metalRunner).Close()

		reference, err := EmitReferenceCPU(fusion)
		convey.So(err, convey.ShouldBeNil)

		for _, count := range metalParitySizes {
			convey.Convey(fmt.Sprintf("It should match the scalar reference at N=%d", count), func() {
				left, right, output, expected := metalParityBuffers(count, reference)

				err := runner.Run([][]float32{left, right}, output, count)
				convey.So(err, convey.ShouldBeNil)
				convey.So(output, convey.ShouldResemble, expected)
			})
		}
	})
}

func TestEmitMetalRunnerSwiGLUParity(testingObject *testing.T) {
	convey.Convey("Given Mul(Sigmoid(gate), up) on Metal", testingObject, func() {
		fusion := swigluFusionMetal()
		runner, err := EmitMetalRunner(fusion)
		convey.So(err, convey.ShouldBeNil)
		defer runner.(*metalRunner).Close()

		reference, err := EmitReferenceCPU(fusion)
		convey.So(err, convey.ShouldBeNil)

		for _, count := range metalParitySizes {
			convey.Convey(fmt.Sprintf("It should match the scalar reference at N=%d", count), func() {
				gate, up, output, expected := metalParityBuffersSwiglu(count, reference)

				err := runner.Run([][]float32{gate, up}, output, count)
				convey.So(err, convey.ShouldBeNil)

				for index := range output {
					convey.So(float64(output[index]), convey.ShouldAlmostEqual, float64(expected[index]), 1e-5)
				}
			})
		}
	})
}

func TestEmitMetalRunnerThreeInputResidual(testingObject *testing.T) {
	convey.Convey("Given ReLU(Add(Add(a, b), c)) on Metal", testingObject, func() {
		fusion := residualAddFusionMetal()
		runner, err := EmitMetalRunner(fusion)
		convey.So(err, convey.ShouldBeNil)
		defer runner.(*metalRunner).Close()

		reference, err := EmitReferenceCPU(fusion)
		convey.So(err, convey.ShouldBeNil)

		for _, count := range metalParitySizes {
			convey.Convey(fmt.Sprintf("It should match the scalar reference at N=%d", count), func() {
				left, middle, right, output, expected := metalParityBuffersThree(count, reference)

				err := runner.Run([][]float32{left, middle, right}, output, count)
				convey.So(err, convey.ShouldBeNil)
				convey.So(output, convey.ShouldResemble, expected)
			})
		}
	})
}

func BenchmarkEmitMetalRunnerRun(benchmark *testing.B) {
	fusion := reluAddFusionMetal()
	runner, err := EmitMetalRunner(fusion)

	if err != nil {
		benchmark.Skip(err)
	}

	metalKernel := runner.(*metalRunner)
	defer metalKernel.Close()

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

func reluAddFusionMetal() *optimizer.FusionAST {
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

func residualAddFusionMetal() *optimizer.FusionAST {
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

func swigluFusionMetal() *optimizer.FusionAST {
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

func metalParityBuffers(
	count int,
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

func metalParityBuffersSwiglu(
	count int,
	reference *CPUKernel,
) (gate []float32, up []float32, output []float32, expected []float32) {
	gate = make([]float32, count)
	up = make([]float32, count)
	output = make([]float32, count)
	expected = make([]float32, count)

	for index := 0; index < count; index++ {
		gate[index] = float32(math.Sin(float64(index)))
		up[index] = float32(math.Cos(float64(index)))
	}

	if err := reference.Run([][]float32{gate, up}, expected, count); err != nil {
		panic(err)
	}

	return gate, up, output, expected
}

func metalParityBuffersThree(
	count int,
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
