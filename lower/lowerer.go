package lower

import (
	"fmt"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
)

/*
Lowerer converts manifest topology AST into manifest graph IR.
*/
type Lowerer struct {
	shapes *ShapeInferencer
}

/*
NewLowerer constructs a Lowerer.
*/
func NewLowerer() *Lowerer {
	return &Lowerer{
		shapes: NewShapeInferencer(),
	}
}

/*
Topology lowers a manifest topology AST into manifest graph IR.
*/
func (lowerer *Lowerer) Topology(
	topology *ast.Topology,
	executionDType dtype.DType,
) (*ast.Graph, error) {
	if topology == nil {
		return nil, fmt.Errorf("lower topology: input is required")
	}

	graph := &ast.Graph{
		Inputs:  append([]string(nil), topology.Inputs...),
		Outputs: make(map[string]string),
	}

	wireToNode := make(map[string]string)
	for _, input := range topology.Inputs {
		wireToNode[input] = input
	}

	for _, node := range topology.Nodes {
		inputs := make([]string, len(node.In))
		for i, inWire := range node.In {
			if producer, ok := wireToNode[inWire]; ok {
				inputs[i] = producer
			} else {
				inputs[i] = inWire // Fallback, though it might fail later
			}
		}

		graphNode := &ast.GraphNode{
			ID:         node.ID,
			Op:         node.Op,
			Inputs:     inputs,
			Attributes: lowerer.cloneMap(node.Config),
			Metadata: map[string]any{
				"manifest_node_id": node.ID,
			},
		}

		if node.Weights != nil {
			graphNode.Weights = &ast.BoundWeight{
				TensorName: node.Weights.Weight,
			}
		}

		if graphNode.Weights == nil {
			graphNode.Weights = boundWeightFromSafeTensors(node.Config)
		}

		for _, outWire := range node.Out {
			wireToNode[outWire] = node.ID
		}

		graph.Nodes = append(graph.Nodes, graphNode)
	}

	if len(topology.Nodes) > 0 {
		lastNode := topology.Nodes[len(topology.Nodes)-1]

		for _, outputName := range lastNode.Out {
			graph.Outputs[outputName] = lastNode.ID
		}
	}

	graph.ApplyExecutionDType(executionDType)

	if err := lowerer.shapes.Apply(topology, graph); err != nil {
		return nil, fmt.Errorf("lower topology: %w", err)
	}

	return graph, nil
}

func boundWeightFromSafeTensors(config map[string]any) *ast.BoundWeight {
	raw := mapFromAny(config["from_safetensors"])

	if raw == nil {
		return nil
	}

	weight, ok := raw["weight"].(string)

	if !ok || weight == "" {
		return nil
	}

	boundWeight := &ast.BoundWeight{TensorName: weight}
	axis, _ := raw["slice_axis"].(string)

	if axis != "" {
		boundWeight.Slice = &ast.WeightSlice{
			Axis:  axis,
			Start: int64FromAny(raw["slice_start"]),
			End:   int64FromAny(raw["slice_end"]),
		}
	}

	return boundWeight
}

func mapFromAny(value any) map[string]any {
	switch typed := value.(type) {
	case map[string]any:
		return typed
	case map[any]any:
		out := make(map[string]any, len(typed))

		for key, value := range typed {
			text, ok := key.(string)

			if !ok {
				continue
			}

			out[text] = value
		}

		return out
	default:
		return nil
	}
}

func int64FromAny(value any) int64 {
	switch typed := value.(type) {
	case int:
		return int64(typed)
	case int64:
		return typed
	case float64:
		return int64(typed)
	default:
		return 0
	}
}

func (lowerer *Lowerer) cloneMap(values map[string]any) map[string]any {
	if values == nil {
		return nil
	}

	cloned := make(map[string]any, len(values))

	for key, value := range values {
		cloned[key] = value
	}

	return cloned
}
