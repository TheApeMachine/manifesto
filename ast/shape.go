package ast

/*
DynamicDim marks a runtime-varying axis in manifest shape metadata. Manifest
shape inference uses negative values; manifest/ir maps them before building
tensor.Shape values for compute/ir.
*/
const DynamicDim int64 = -1

/*
NewDynamicShape constructs a rank-sized shape with every axis dynamic.
*/
func NewDynamicShape(rank int) []int64 {
	shape := make([]int64, rank)

	for index := range shape {
		shape[index] = DynamicDim
	}

	return shape
}
