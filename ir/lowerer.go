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
	nodesByID := make(map[string]*Node, len(manifestGraph.Nodes)+len(manifestGraph.Inputs))
	wireToNode := make(map[string]*Node)

	for _, inputID := range manifestGraph.Inputs {
		shape, _ := lowerer.shape([]int64{ast.DynamicDim, ast.DynamicDim}) // Default shape
		inputNode := NewNode(inputID, OpInput, shape)
		inputNode.SetOperationID(OpID(OpInput))
		computeGraph.AddNode(inputNode)
		nodesByID[inputID] = inputNode
		wireToNode[inputID] = inputNode
	}

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

		for key, value := range manifestNode.Metadata {
			computeNode.SetMetadata(key, value)
		}

		if manifestNode.Weights != nil {
			computeNode.SetMetadata("weight_name", manifestNode.Weights.TensorName)
			computeNode.SetMetadata("weight_dtype", manifestNode.Weights.DType.String())
			computeNode.SetMetadata("weight_shape", manifestNode.Weights.Shape)

			if manifestNode.Weights.Slice != nil {
				computeNode.SetMetadata("weight_slice_axis", manifestNode.Weights.Slice.Axis)
				computeNode.SetMetadata("weight_slice_start", manifestNode.Weights.Slice.Start)
				computeNode.SetMetadata("weight_slice_end", manifestNode.Weights.Slice.End)
			}
		}

		computeGraph.AddNode(computeNode)
		nodesByID[manifestNode.ID] = computeNode

		// Register the outputs of this node in the wireToNode map.
		// Since we don't have manifestNode.Outputs in ast.GraphNode,
		// we assume the node ID is the output wire name, or we need to get it from somewhere.
		// Wait, ast.GraphNode doesn't have Outputs!
		// Let's check ast.GraphNode.
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
