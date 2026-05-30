//go:build codegen_llvm

package codegen

import (
	"github.com/theapemachine/manifesto/codegen/llvm"
	"github.com/theapemachine/manifesto/optimizer"
)

/*
llvmCPUKernel adapts an LLVM JIT kernel to the codegen Kernel contract.
*/
type llvmCPUKernel struct {
	inner *llvm.CompiledKernel
}

func (kernel *llvmCPUKernel) Target() Target {
	return TargetCPU
}

func (kernel *llvmCPUKernel) Identifier() string {
	return kernel.inner.Identifier()
}

func (kernel *llvmCPUKernel) Inputs() []string {
	return kernel.inner.Inputs()
}

func (kernel *llvmCPUKernel) Output() string {
	return kernel.inner.Output()
}

func (kernel *llvmCPUKernel) Run(inputs [][]float32, output []float32, count int) error {
	return kernel.inner.Run(inputs, output, count)
}

/*
Close releases JIT resources held by the kernel.
*/
func (kernel *llvmCPUKernel) Close() {
	if kernel == nil || kernel.inner == nil {
		return
	}

	kernel.inner.Close()
	kernel.inner = nil
}

var _ Kernel = (*llvmCPUKernel)(nil)
var _ ElementwiseRunner = (*llvmCPUKernel)(nil)

/*
EmitCPU lowers one FusionAST into an LLVM MCJIT kernel for the host CPU.
*/
func EmitCPU(fusion *optimizer.FusionAST) (ElementwiseRunner, error) {
	compiled, err := llvm.EmitKernel(fusion)

	if err != nil {
		return nil, err
	}

	return &llvmCPUKernel{inner: compiled}, nil
}

// Keep optimizer import referenced for EmitCPU signature docs.
var _ = optimizer.FusionAST{}
