//go:build !darwin || !cgo

package codegen

import (
	"fmt"

	"github.com/theapemachine/manifesto/optimizer"
)

/*
EmitMetalRunner is unavailable off Darwin.
*/
func EmitMetalRunner(fusion *optimizer.FusionAST) (ElementwiseRunner, error) {
	_ = fusion

	return nil, fmt.Errorf("codegen metal: MTLLibrary compilation requires darwin with cgo")
}
