package ast

import "github.com/theapemachine/manifesto/dtype"

/*
ApplyExecutionDType sets the graph execution dtype and assigns activation value
types to every node. Weight storage dtypes remain on BoundWeight.
*/
func (graph *Graph) ApplyExecutionDType(executionDType dtype.DType) {
	graph.ExecutionDType = executionDType

	valueType := NewValueType(executionDType)

	for _, node := range graph.Nodes {
		node.ValueType = valueType
	}
}
