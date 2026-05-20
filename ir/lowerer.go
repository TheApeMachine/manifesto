package ir

import (
	"fmt"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/tensor"
)

/*
Lowerer translates manifest graph IR into compute/ir graphs.
*/
type Lowerer struct{}

/*
NewLowerer constructs a Lowerer.
*/
func NewLowerer() *Lowerer {
	return &Lowerer{}
}

/*
Graph lowers one manifest graph into compute IR, preserving execution dtypes.
*/
func (lowerer *Lowerer) Graph(manifestGraph *ast.Graph) (*Graph, error) {
	if manifestGraph == nil {
		return nil, fmt.Errorf("manifest ir: graph is required")
	}

	computeGraph := NewGraph()
	nodesByID := make(map[string]*Node, len(manifestGraph.Nodes))

	for _, manifestNode := range manifestGraph.Nodes {
		shape, err := lowerer.shape(manifestNode.ValueType.Shape)

		if err != nil {
			return nil, fmt.Errorf("manifest ir: node %q shape: %w", manifestNode.ID, err)
		}

		computeNode := NewNode(manifestNode.ID, OpType(manifestNode.Op), shape)
		computeNode.SetOperationID(OpID(manifestNode.Op))

		valueType, err := lowerer.valueType(manifestNode.ValueType, shape)

		if err != nil {
			return nil, fmt.Errorf("manifest ir: node %q value type: %w", manifestNode.ID, err)
		}

		computeNode.SetValueType(ValueType(valueType))

		for key, value := range manifestNode.Attributes {
			computeNode.SetAttribute(key, Attribute(stringifyAttribute(value)))
		}

		if manifestNode.Weights != nil {
			computeNode.SetMetadata("weight_name", manifestNode.Weights.TensorName)
			computeNode.SetMetadata("weight_dtype", manifestNode.Weights.DType.String())
			computeNode.SetMetadata("weight_shape", manifestNode.Weights.Shape)
		}

		computeGraph.AddNode(computeNode)
		nodesByID[manifestNode.ID] = computeNode
	}

	for _, manifestNode := range manifestGraph.Nodes {
		computeNode := nodesByID[manifestNode.ID]

		for _, inputID := range manifestNode.Inputs {
			inputNode, ok := nodesByID[inputID]

			if !ok {
				return nil, fmt.Errorf("manifest ir: node %q missing input %q", manifestNode.ID, inputID)
			}

			computeNode.AddInput(inputNode)
		}
	}

	return computeGraph, nil
}

func (lowerer *Lowerer) shape(dimensions []int64) (tensor.Shape, error) {
	if len(dimensions) == 0 {
		return tensor.NewShape([]int{1})
	}

	dims := make([]int, len(dimensions))

	for index, dimension := range dimensions {
		if dimension == ast.DynamicDim {
			dims[index] = 1
			continue
		}

		if dimension < 0 {
			return tensor.Shape{}, fmt.Errorf("invalid dimension %d at axis %d", dimension, index)
		}

		dims[index] = int(dimension)
	}

	return tensor.NewShape(dims)
}

func (lowerer *Lowerer) valueType(manifestValue ast.ValueType, shape tensor.Shape) (ValueType, error) {
	return ValueType{
		Shape:       shape,
		DType:       manifestValue.DType,
		Precision:   manifestValue.Precision,
		Layout:      Layout(manifestValue.Layout),
		MemoryClass: lowerer.memoryClass(manifestValue.Memory),
	}, nil
}

func (lowerer *Lowerer) memoryClass(memory ast.MemoryClass) MemoryClass {
	switch memory {
	case ast.MemoryDevice:
		return MemoryDevice
	default:
		return MemoryHost
	}
}

func stringifyAttribute(value any) Attribute {
	switch typed := value.(type) {
	case string:
		return StringAttribute(typed)
	case bool:
		return BoolAttribute(typed)
	case int:
		return IntAttribute(int64(typed))
	case int64:
		return IntAttribute(typed)
	case float64:
		return FloatAttribute(typed)
	default:
		return StringAttribute(fmt.Sprintf("%v", typed))
	}
}
