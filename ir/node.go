package ir

import (
	"fmt"
	"sort"
	"sync"

	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/manifesto/dtype"
)

/*
OpType specifies the type of mathematical operation for the node.
*/
type OpType string

type OpID string

const (
	OpInput     OpType = "Input"
	OpMatmul    OpType = "Matmul"
	OpAdd       OpType = "Add"
	OpMul       OpType = "Mul"
	OpReLU      OpType = "ReLU"
	OpLeakyReLU OpType = "LeakyReLU"
	OpGELU      OpType = "GELU"
	OpTanh      OpType = "Tanh"
	OpSigmoid   OpType = "Sigmoid"
	OpSwiGLU    OpType = "SwiGLU"
	OpSwish     OpType = "Swish"
	OpSELU      OpType = "SELU"
	OpFused     OpType = "Fused"
)

/*
Node represents an operation in the intermediate representation graph.
It abstracts the hardware-specific implementation so operations can be routed generically.
Safe for concurrent access.
*/
type Node struct {
	mu          sync.RWMutex
	id          string
	opType      OpType
	operationID OpID
	shape       tensor.Shape
	valueType   ValueType
	effect      Effect
	alias       Alias
	inPlace     bool
	inputs      []*Node
	metadata    map[string]any
	attributes  map[string]Attribute
}

/*
NewNode instantiates a new Node.
It serves as a single mathematical step in a larger compute graph.
*/
func NewNode(id string, opType OpType, shape tensor.Shape) *Node {
	return &Node{
		id:          id,
		opType:      opType,
		operationID: OpID(opType),
		shape:       shape,
		valueType: ValueType{
			Shape:       shape,
			DType:       dtype.Float64,
			Layout:      LayoutDense,
			MemoryClass: MemoryHost,
		},
		effect:     EffectPure,
		alias:      Alias{Kind: AliasAllocates, InputIndex: -1},
		inputs:     make([]*Node, 0),
		metadata:   make(map[string]any),
		attributes: make(map[string]Attribute),
	}
}

/*
ID returns the node's unique identifier.
*/
func (node *Node) ID() string {
	node.mu.RLock()
	defer node.mu.RUnlock()
	return node.id
}

/*
OpType returns the node's operation type.
*/
func (node *Node) OpType() OpType {
	node.mu.RLock()
	defer node.mu.RUnlock()
	return node.opType
}

func (node *Node) OperationID() OpID {
	node.mu.RLock()
	defer node.mu.RUnlock()

	return node.operationID
}

func (node *Node) SetOperationID(operationID OpID) {
	node.mu.Lock()
	defer node.mu.Unlock()

	if operationID == "" {
		node.operationID = OpID(node.opType)

		return
	}

	node.operationID = operationID
}

/*
Shape returns the node's output shape.
*/
func (node *Node) Shape() tensor.Shape {
	node.mu.RLock()
	defer node.mu.RUnlock()
	return node.shape
}

func (node *Node) ValueType() ValueType {
	node.mu.RLock()
	defer node.mu.RUnlock()

	return node.valueType
}

func (node *Node) SetValueType(valueType ValueType) {
	node.mu.Lock()
	defer node.mu.Unlock()

	if !valueType.Shape.Valid() {
		valueType.Shape = node.shape
	}

	if valueType.DType == dtype.Invalid {
		valueType.DType = dtype.Float64
	}

	if valueType.Precision == dtype.Invalid {
		valueType.Precision = valueType.DType
	}

	if valueType.Layout == "" {
		valueType.Layout = LayoutDense
	}

	if valueType.MemoryClass == "" {
		valueType.MemoryClass = MemoryHost
	}

	node.valueType = valueType
}

/*
Inputs returns the nodes that this node depends on.
Returns a defensive copy of the slice.
*/
func (node *Node) Inputs() []*Node {
	node.mu.RLock()
	defer node.mu.RUnlock()
	out := make([]*Node, len(node.inputs))
	copy(out, node.inputs)
	return out
}

/*
AddInput adds a dependency to this node.
*/
func (node *Node) AddInput(input *Node) {
	if input == nil {
		return
	}
	node.mu.Lock()
	defer node.mu.Unlock()
	node.inputs = append(node.inputs, input)
}

func (node *Node) Effect() Effect {
	node.mu.RLock()
	defer node.mu.RUnlock()

	return node.effect
}

func (node *Node) SetEffect(effect Effect) {
	node.mu.Lock()
	defer node.mu.Unlock()

	node.effect = effect
}

func (node *Node) IsPure() bool {
	return node.Effect() == EffectPure
}

func (node *Node) Alias() Alias {
	node.mu.RLock()
	defer node.mu.RUnlock()

	return node.alias
}

func (node *Node) SetAlias(alias Alias) {
	node.mu.Lock()
	defer node.mu.Unlock()

	node.alias = alias
}

/*
Metadata returns additional configuration for the node.
Returns a defensive copy of the map.
*/
func (node *Node) Metadata() map[string]any {
	node.mu.RLock()
	defer node.mu.RUnlock()
	out := make(map[string]any)
	for k, v := range node.metadata {
		out[k] = v
	}
	return out
}

/*
SetMetadata adds configuration to the node.
*/
func (node *Node) SetMetadata(key string, value any) {
	if key == "" {
		return
	}
	node.mu.Lock()
	defer node.mu.Unlock()
	node.metadata[key] = value
}

func (node *Node) Attribute(key string) Attribute {
	node.mu.RLock()
	defer node.mu.RUnlock()

	return node.attributes[key]
}

func (node *Node) Attributes() map[string]Attribute {
	node.mu.RLock()
	defer node.mu.RUnlock()

	attributes := make(map[string]Attribute, len(node.attributes))

	for key, value := range node.attributes {
		attributes[key] = value
	}

	return attributes
}

func (node *Node) SetAttribute(key string, value Attribute) {
	if key == "" {
		return
	}

	node.mu.Lock()
	defer node.mu.Unlock()

	node.attributes[key] = value
}

func (node *Node) CanonicalAttributes() string {
	attributes := node.Attributes()
	keys := make([]string, 0, len(attributes))

	for key := range attributes {
		keys = append(keys, key)
	}

	sort.Strings(keys)

	canonical := ""
	for _, key := range keys {
		canonical += fmt.Sprintf("%s=%s;", key, attributes[key].String())
	}

	return canonical
}

/*
InPlace returns whether the node should mutate its input buffer.
*/
func (node *Node) InPlace() bool {
	node.mu.RLock()
	defer node.mu.RUnlock()
	return node.inPlace
}

/*
SetInPlace configures whether the node can safely mutate its input buffer.
*/
func (node *Node) SetInPlace(val bool) {
	node.mu.Lock()
	defer node.mu.Unlock()
	node.inPlace = val
}
