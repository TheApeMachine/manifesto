package ir

import "github.com/theapemachine/manifesto/types"

/*
Weight holds checkpoint tensor tokens attached to a node during compile.
*/
type Weight struct {
	TensorName string
	BiasName   string
	Slice      *WeightSlice
	Tensor     types.Token
	Bias       types.Token
}

/*
WeightSlice selects a sub-range from a fused checkpoint tensor.
*/
type WeightSlice struct {
	Axis  string
	Start int64
	End   int64
}

/*
HasTensor reports whether the primary checkpoint tensor was resolved.
*/
func (weight *Weight) HasTensor() bool {
	if weight == nil {
		return false
	}

	return weight.Tensor.Kind == types.KindTensor && weight.Tensor.Name != ""
}

/*
HasBias reports whether the bias checkpoint tensor was resolved.
*/
func (weight *Weight) HasBias() bool {
	if weight == nil {
		return false
	}

	return weight.Bias.Kind == types.KindTensor && weight.Bias.Name != ""
}
