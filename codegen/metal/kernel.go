package metal

import (
	"fmt"
	"strings"

	"github.com/theapemachine/manifesto/optimizer"
)

/*
CompiledKernel is an MTLLibrary-compiled elementwise fusion kernel.
*/
type CompiledKernel struct {
	identifier string
	inputs     []string
	output     string
	kernelName string
	inputCount int
	source     string
	runner     fusionRunner
}

type fusionRunner interface {
	run(inputs [][]float32, output []float32, count int) error
	close()
}

/*
Identifier returns the fusion output port name used as the kernel id.
*/
func (kernel *CompiledKernel) Identifier() string {
	if kernel == nil {
		return ""
	}

	return kernel.identifier
}

/*
KernelName returns the MSL entry point compiled into the MTLLibrary.
*/
func (kernel *CompiledKernel) KernelName() string {
	if kernel == nil {
		return ""
	}

	return kernel.kernelName
}

/*
Source returns the MSL source compiled by MTLLibrary.
*/
func (kernel *CompiledKernel) Source() string {
	if kernel == nil {
		return ""
	}

	return kernel.source
}

/*
Inputs returns the input port names in kernel argument order.
*/
func (kernel *CompiledKernel) Inputs() []string {
	if kernel == nil {
		return nil
	}

	out := make([]string, len(kernel.inputs))
	copy(out, kernel.inputs)

	return out
}

/*
Output returns the fused output port name.
*/
func (kernel *CompiledKernel) Output() string {
	if kernel == nil {
		return ""
	}

	return kernel.output
}

/*
Close releases MTLLibrary and pipeline resources held by the kernel.
*/
func (kernel *CompiledKernel) Close() {
	if kernel == nil || kernel.runner == nil {
		return
	}

	kernel.runner.close()
	kernel.runner = nil
}

/*
Run executes the compiled kernel over count float32 elements.
*/
func (kernel *CompiledKernel) Run(
	inputs [][]float32,
	output []float32,
	count int,
) error {
	if kernel == nil {
		return fmt.Errorf("codegen metal: kernel is nil")
	}

	if kernel.runner == nil {
		return fmt.Errorf("codegen metal: kernel %q is closed", kernel.identifier)
	}

	if len(inputs) != kernel.inputCount {
		return fmt.Errorf(
			"codegen metal: kernel %q expects %d inputs, got %d",
			kernel.identifier, kernel.inputCount, len(inputs),
		)
	}

	if len(output) < count {
		return fmt.Errorf(
			"codegen metal: output buffer holds %d values, need %d",
			len(output), count,
		)
	}

	for inputIndex, buffer := range inputs {
		if len(buffer) < count {
			return fmt.Errorf(
				"codegen metal: input %d (%q) holds %d values, need %d",
				inputIndex, kernel.inputs[inputIndex], len(buffer), count,
			)
		}
	}

	return kernel.runner.run(inputs, output, count)
}

func newCompiledKernel(
	fusion *optimizer.FusionAST,
	source string,
	kernelName string,
	runner fusionRunner,
) *CompiledKernel {
	identifier := fusion.OutputPort

	if identifier == "" {
		identifier = "anon_" + strings.Join(fusion.InputPorts, "_")
	}

	return &CompiledKernel{
		identifier: identifier,
		inputs:     append([]string(nil), fusion.InputPorts...),
		output:     fusion.OutputPort,
		kernelName: kernelName,
		inputCount: len(fusion.InputPorts),
		source:     source,
		runner:     runner,
	}
}
