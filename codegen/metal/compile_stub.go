//go:build !darwin || !cgo

package metal

import "fmt"

func compileMSL(source, kernelName string) (fusionRunner, error) {
	_ = source
	_ = kernelName

	return nil, fmt.Errorf("codegen metal: MTLLibrary compilation requires darwin with cgo")
}
