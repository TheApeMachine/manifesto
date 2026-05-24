package dag

import (
	"fmt"
	"sync"

	"github.com/theapemachine/manifesto/tensor"
)

/*
OpType identifies one compute-graph node operation during scheduling.
This is the interim DAG layer until ARCHITECTURE.md ExecutionNode lands in ir.
*/
type OpType string

const (
	OpInput  OpType = "Input"
	OpMatmul OpType = "Matmul"
)

/*
Node is one schedulable operation in a compute DAG.
*/
type Node struct {
	mu     sync.RWMutex
	id     string
	opType OpType
	shape  tensor.Shape
	inputs []*Node
}

/*
NewNode constructs a compute DAG node.
*/
func NewNode(id string, opType OpType, shape tensor.Shape) *Node {
	return &Node{
		id:     id,
		opType: opType,
		shape:  shape,
	}
}

/*
ID returns the node identifier.
*/
func (node *Node) ID() string {
	if node == nil {
		return ""
	}

	node.mu.RLock()
	defer node.mu.RUnlock()

	return node.id
}

/*
OpType returns the node operation type.
*/
func (node *Node) OpType() OpType {
	if node == nil {
		return ""
	}

	node.mu.RLock()
	defer node.mu.RUnlock()

	return node.opType
}

/*
Shape returns the node output shape.
*/
func (node *Node) Shape() tensor.Shape {
	if node == nil {
		return tensor.Shape{}
	}

	node.mu.RLock()
	defer node.mu.RUnlock()

	return node.shape
}

/*
Inputs returns upstream nodes feeding this node.
*/
func (node *Node) Inputs() []*Node {
	if node == nil {
		return nil
	}

	node.mu.RLock()
	defer node.mu.RUnlock()

	out := make([]*Node, len(node.inputs))
	copy(out, node.inputs)

	return out
}

/*
AddInput registers one upstream dependency.
*/
func (node *Node) AddInput(input *Node) error {
	if node == nil {
		return fmt.Errorf("dag: node is required")
	}

	if input == nil {
		return fmt.Errorf("dag: input node is required")
	}

	node.mu.Lock()
	defer node.mu.Unlock()

	node.inputs = append(node.inputs, input)

	return nil
}
