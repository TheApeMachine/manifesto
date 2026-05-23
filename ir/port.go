package ir

import (
	"github.com/theapemachine/manifesto/tensor"
)

/*
Port is one input or output on a node.
*/
type Port struct {
	Tensor *tensor.Tensor
}
