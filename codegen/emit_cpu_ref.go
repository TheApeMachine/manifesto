//go:build !codegen_llvm

package codegen

import (
	"github.com/theapemachine/manifesto/optimizer"
)

/*
EmitCPU lowers one FusionAST into the scalar reference CPUKernel.
*/
func EmitCPU(fusion *optimizer.FusionAST) (*CPUKernel, error) {
	return EmitReferenceCPU(fusion)
}
