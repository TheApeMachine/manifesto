package ast

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
)

/*
Graph is the manifest-native compute IR produced by lowering a topology.
manifest/ir lowers this into pkg/backend/compute/ir for execution.

Bindings is populated by the typer pass (Phase 2.2): after every edge
unifies, it carries the global symbol-to-concrete-size map produced by
ir.Unify. Downstream stages (memory planner, codegen) read it to
resolve symbolic shapes.
*/
type Graph struct {
	Nodes          []*GraphNode
	Inputs         []string
	Outputs        map[string]string
	ExecutionDType dtype.DType
	Metadata       map[string]any
	Bindings       ir.SymbolMap
}

/*
GraphNode is one lowered operation with resolved attributes.

InputTypes and OutputType are populated by the typer pass (Phase 2.2)
once unification has resolved every edge. They carry the same PortType
contract ir.Unify operates on so downstream passes don't have to re-
derive types from raw shapes.
*/
type GraphNode struct {
	ID         string
	Op         string
	Inputs     []string
	ValueType  ValueType
	Attributes map[string]any
	Metadata   map[string]any
	Weights    *BoundWeight
	InputTypes []ir.PortType
	OutputType ir.PortType
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
