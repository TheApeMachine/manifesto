package codegen

import (
	"fmt"
	"math"
	"strings"

	"github.com/theapemachine/manifesto/optimizer"
)

/*
CPUKernel is a Go-callable evaluator for one FusionAST. It walks the AST
once per element of the output, reading inputs by index from the supplied
slice of buffers.

This is the first-pass CPU lowering: correct but not vectorized. The
LLVM-JIT path described in ARCHITECTURE.md §4.3 / Phase 3.2 lands later;
when it does, CPUKernel.Run becomes a fallback for hosts without LLVM.

The kernel operates on float32 buffers — manifesto's default execution
dtype. Wider/narrower dtypes are handled by adaptor synthesis before the
fusion boundary (§4.2).
*/
type CPUKernel struct {
	identifier string
	inputs     []string
	output     string
	root       *optimizer.ASTNode
}

func (kernel *CPUKernel) Target() Target {
	return TargetCPU
}

func (kernel *CPUKernel) Identifier() string {
	return kernel.identifier
}

/*
Inputs returns the input port names the kernel expects, in order.
*/
func (kernel *CPUKernel) Inputs() []string {
	out := make([]string, len(kernel.inputs))
	copy(out, kernel.inputs)
	return out
}

/*
Output returns the output port name the kernel writes.
*/
func (kernel *CPUKernel) Output() string {
	return kernel.output
}

/*
Run evaluates the fusion over count elements. inputs[i] must contain at
least count float32 values. output is written in place.
*/
func (kernel *CPUKernel) Run(inputs [][]float32, output []float32, count int) error {
	if kernel == nil || kernel.root == nil {
		return fmt.Errorf("codegen cpu: kernel is empty")
	}

	if len(inputs) != len(kernel.inputs) {
		return fmt.Errorf(
			"codegen cpu: kernel %q expects %d inputs, got %d",
			kernel.identifier, len(kernel.inputs), len(inputs),
		)
	}

	if len(output) < count {
		return fmt.Errorf(
			"codegen cpu: output buffer holds %d values, need %d",
			len(output), count,
		)
	}

	for inputIndex, buffer := range inputs {
		if len(buffer) < count {
			return fmt.Errorf(
				"codegen cpu: input %d (%q) holds %d values, need %d",
				inputIndex, kernel.inputs[inputIndex], len(buffer), count,
			)
		}
	}

	for elementIndex := 0; elementIndex < count; elementIndex++ {
		output[elementIndex] = evalCPU(kernel.root, inputs, elementIndex)
	}

	return nil
}

/*
EmitCPU lowers one FusionAST into a CPUKernel.
*/
func EmitCPU(fusion *optimizer.FusionAST) (*CPUKernel, error) {
	if fusion == nil {
		return nil, fmt.Errorf("codegen cpu: fusion is required")
	}

	if fusion.Root == nil {
		return nil, fmt.Errorf("codegen cpu: fusion root is required")
	}

	identifier := fusion.OutputPort

	if identifier == "" {
		identifier = "anon_" + strings.Join(fusion.InputPorts, "_")
	}

	return &CPUKernel{
		identifier: identifier,
		inputs:     append([]string(nil), fusion.InputPorts...),
		output:     fusion.OutputPort,
		root:       fusion.Root,
	}, nil
}

/*
evalCPU evaluates one ASTNode at element index. Operates on float32 inputs;
intermediate math is done at float32 precision since manifesto's
default execution dtype is bf16/f16 with f32 accumulators and the kernel
sees the post-adaptor f32 values.
*/
func evalCPU(node *optimizer.ASTNode, inputs [][]float32, elementIndex int) float32 {
	switch node.Type {
	case optimizer.NodeInput:
		return inputs[node.InputIndex][elementIndex]
	case optimizer.NodeConstant:
		return float32(node.Value)
	case optimizer.NodeAdd:
		return evalCPU(node.Children[0], inputs, elementIndex) +
			evalCPU(node.Children[1], inputs, elementIndex)
	case optimizer.NodeSub:
		return evalCPU(node.Children[0], inputs, elementIndex) -
			evalCPU(node.Children[1], inputs, elementIndex)
	case optimizer.NodeMul:
		return evalCPU(node.Children[0], inputs, elementIndex) *
			evalCPU(node.Children[1], inputs, elementIndex)
	case optimizer.NodeDiv:
		return evalCPU(node.Children[0], inputs, elementIndex) /
			evalCPU(node.Children[1], inputs, elementIndex)
	case optimizer.NodeMax:
		left := evalCPU(node.Children[0], inputs, elementIndex)
		right := evalCPU(node.Children[1], inputs, elementIndex)

		if left > right {
			return left
		}

		return right
	case optimizer.NodeMin:
		left := evalCPU(node.Children[0], inputs, elementIndex)
		right := evalCPU(node.Children[1], inputs, elementIndex)

		if left < right {
			return left
		}

		return right
	case optimizer.NodeNeg:
		return -evalCPU(node.Children[0], inputs, elementIndex)
	case optimizer.NodeAbs:
		value := evalCPU(node.Children[0], inputs, elementIndex)

		if value < 0 {
			return -value
		}

		return value
	case optimizer.NodeSqrt:
		return float32(math.Sqrt(float64(evalCPU(node.Children[0], inputs, elementIndex))))
	case optimizer.NodeExp:
		return float32(math.Exp(float64(evalCPU(node.Children[0], inputs, elementIndex))))
	case optimizer.NodeLog:
		return float32(math.Log(float64(evalCPU(node.Children[0], inputs, elementIndex))))
	case optimizer.NodeReLU:
		value := evalCPU(node.Children[0], inputs, elementIndex)

		if value < 0 {
			return 0
		}

		return value
	case optimizer.NodeSigmoid:
		value := evalCPU(node.Children[0], inputs, elementIndex)

		return float32(1.0 / (1.0 + math.Exp(-float64(value))))
	case optimizer.NodeTanh:
		return float32(math.Tanh(float64(evalCPU(node.Children[0], inputs, elementIndex))))
	case optimizer.NodeSilu:
		value := evalCPU(node.Children[0], inputs, elementIndex)

		return float32(float64(value) / (1.0 + math.Exp(-float64(value))))
	case optimizer.NodeGelu:
		value := float64(evalCPU(node.Children[0], inputs, elementIndex))
		inner := math.Sqrt(2.0/math.Pi) * (value + 0.044715*value*value*value)

		return float32(0.5 * value * (1.0 + math.Tanh(inner)))
	case optimizer.NodeLeakyReLU:
		value := evalCPU(node.Children[0], inputs, elementIndex)

		if value >= 0 {
			return value
		}

		return 0.01 * value
	default:
		return 0
	}
}

var _ Kernel = (*CPUKernel)(nil)
