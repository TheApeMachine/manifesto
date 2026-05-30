package codegen

import (
	metalcodegen "github.com/theapemachine/manifesto/codegen/metal"
	"github.com/theapemachine/manifesto/optimizer"
)

/*
MetalKernel is one FusionAST lowered into Metal Shading Language source.
*/
type MetalKernel struct {
	inner *metalcodegen.SourceKernel
}

func (kernel *MetalKernel) Target() Target {
	return TargetMetal
}

func (kernel *MetalKernel) Identifier() string {
	return kernel.inner.Identifier()
}

func (kernel *MetalKernel) Source() string {
	return kernel.inner.Source()
}

func (kernel *MetalKernel) KernelName() string {
	return kernel.inner.KernelName()
}

func (kernel *MetalKernel) Inputs() []string {
	return kernel.inner.Inputs()
}

func (kernel *MetalKernel) Output() string {
	return kernel.inner.Output()
}

/*
EmitMetal generates MSL source for one FusionAST.
*/
func EmitMetal(fusion *optimizer.FusionAST) (*MetalKernel, error) {
	sourceKernel, err := metalcodegen.EmitMSL(fusion)

	if err != nil {
		return nil, err
	}

	return &MetalKernel{inner: sourceKernel}, nil
}

var _ Kernel = (*MetalKernel)(nil)
