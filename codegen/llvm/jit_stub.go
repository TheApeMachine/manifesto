//go:build !codegen_llvm

package llvm

import (
	"errors"

	"github.com/theapemachine/manifesto/optimizer"
)

/*
ErrJITUnavailable indicates the LLVM JIT path was not compiled in. Build
with -tags=codegen_llvm and install a supported LLVM (see Makefile test-jit).
*/
var ErrJITUnavailable = errors.New("codegen llvm: jit unavailable (build with -tags=codegen_llvm)")

/*
EmitKernel is unavailable unless the codegen_llvm build tag is set.
*/
func EmitKernel(fusion *optimizer.FusionAST) (*CompiledKernel, error) {
	_ = fusion
	return nil, ErrJITUnavailable
}
