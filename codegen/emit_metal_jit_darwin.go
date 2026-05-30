//go:build darwin && cgo

package codegen

import (
	metalcodegen "github.com/theapemachine/manifesto/codegen/metal"
	"github.com/theapemachine/manifesto/optimizer"
)

/*
metalRunner adapts a compiled MTLLibrary kernel to the codegen runner contract.
*/
type metalRunner struct {
	inner *metalcodegen.CompiledKernel
}

func (runner *metalRunner) Target() Target {
	return TargetMetal
}

func (runner *metalRunner) Identifier() string {
	return runner.inner.Identifier()
}

func (runner *metalRunner) Inputs() []string {
	return runner.inner.Inputs()
}

func (runner *metalRunner) Output() string {
	return runner.inner.Output()
}

func (runner *metalRunner) Run(inputs [][]float32, output []float32, count int) error {
	return runner.inner.Run(inputs, output, count)
}

func (runner *metalRunner) MSLSource() string {
	return runner.inner.Source()
}

func (runner *metalRunner) MSLKernelName() string {
	return runner.inner.KernelName()
}

/*
Close releases MTLLibrary resources held by the kernel.
*/
func (runner *metalRunner) Close() {
	if runner == nil || runner.inner == nil {
		return
	}

	runner.inner.Close()
	runner.inner = nil
}

var _ Kernel = (*metalRunner)(nil)
var _ ElementwiseRunner = (*metalRunner)(nil)
var _ MetalFusionProgram = (*metalRunner)(nil)

/*
EmitMetalRunner lowers one FusionAST into a compiled MTLLibrary kernel.
*/
func EmitMetalRunner(fusion *optimizer.FusionAST) (ElementwiseRunner, error) {
	compiled, err := metalcodegen.EmitKernel(fusion)

	if err != nil {
		return nil, err
	}

	return &metalRunner{inner: compiled}, nil
}
