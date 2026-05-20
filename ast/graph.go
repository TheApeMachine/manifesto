package ast

import "github.com/theapemachine/manifesto/dtype"

/*
Graph is the manifest-native compute IR produced by lowering a topology.
manifest/ir lowers this into pkg/backend/compute/ir for execution.
*/
type Graph struct {
	Nodes          []*GraphNode
	Inputs         []string
	Outputs        map[string]string
	ExecutionDType dtype.DType
	Metadata       map[string]any
}

/*
GraphNode is one lowered operation with resolved attributes.
*/
type GraphNode struct {
	ID         string
	Op         string
	Inputs     []string
	ValueType  ValueType
	Attributes map[string]any
	Metadata   map[string]any
	Weights    *BoundWeight
}

/*
BoundWeight attaches a loaded checkpoint tensor to a graph node.
*/
type BoundWeight struct {
	TensorName string
	Shape      []int64
	DType      dtype.DType
	Slice      *WeightSlice
}

/*
WeightSlice selects a sub-range from a fused checkpoint tensor.
*/
type WeightSlice struct {
	Axis  string
	Start int64
	End   int64
}
