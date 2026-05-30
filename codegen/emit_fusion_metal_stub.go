//go:build !darwin || !cgo

package codegen

import "github.com/theapemachine/manifesto/optimizer"

func emitMetalKernel(fusion *optimizer.FusionAST) (Kernel, error) {
	return EmitMetal(fusion)
}
