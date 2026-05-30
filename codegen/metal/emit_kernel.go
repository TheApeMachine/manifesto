package metal

import (
	"fmt"

	"github.com/theapemachine/manifesto/optimizer"
)

/*
EmitKernel lowers fusion into MSL, compiles it with MTLLibrary, and returns
an executable kernel for the default Metal device.
*/
func EmitKernel(fusion *optimizer.FusionAST) (*CompiledKernel, error) {
	if fusion == nil {
		return nil, fmt.Errorf("codegen metal: fusion is required")
	}

	if fusion.Root == nil {
		return nil, fmt.Errorf("codegen metal: fusion root is required")
	}

	sourceKernel, err := EmitMSL(fusion)

	if err != nil {
		return nil, err
	}

	runner, err := compileMSL(sourceKernel.Source(), sourceKernel.KernelName())

	if err != nil {
		return nil, err
	}

	return newCompiledKernel(
		fusion,
		sourceKernel.Source(),
		sourceKernel.KernelName(),
		runner,
	), nil
}
