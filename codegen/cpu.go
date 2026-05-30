package codegen

import (
	"fmt"
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
		output[elementIndex] = EvalScalarReference(kernel.root, inputs, elementIndex)
	}

	return nil
}

/*
EmitReferenceCPU lowers one FusionAST into the scalar reference evaluator.
Parity tests compare JIT output against this path.
*/
func EmitReferenceCPU(fusion *optimizer.FusionAST) (*CPUKernel, error) {
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

var (
	_ Kernel            = (*CPUKernel)(nil)
	_ ElementwiseRunner = (*CPUKernel)(nil)
)
