package weights

import (
	"fmt"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/types"
)

/*
Binder indexes SafeTensors checkpoints and attaches tensors to graph nodes.
*/
type Binder struct{}

/*
NewBinder constructs a Binder.
*/
func NewBinder() *Binder {
	return &Binder{}
}

/*
Bind attaches checkpoint tensors to graph nodes using explicit weight maps and
prefix conventions.
*/
func (binder *Binder) Bind(
	graph *ast.Graph,
	parser types.Parser,
	weightMap map[string]string,
) (map[string]struct{}, error) {
	if graph == nil {
		return nil, fmt.Errorf("weights bind: graph is required")
	}

	index, err := binder.tensorIndex(parser)

	if err != nil {
		return nil, err
	}

	names := make(map[string]struct{}, len(index))

	for _, node := range graph.Nodes {
		tensorName := binder.resolveTensorName(node, weightMap)

		if tensorName == "" {
			tensorName = node.ID + ".weight"
		}

		token, ok := index[tensorName]

		if !ok {
			if node.Weights != nil && len(node.Weights.Shape) > 0 {
				continue
			}

			if node.Weights != nil && node.Weights.TensorName != "" {
				return nil, fmt.Errorf("weights bind: missing tensor %q for node %q", tensorName, node.ID)
			}

			continue
		}

		var weightSlice *ast.WeightSlice

		if node.Weights != nil {
			weightSlice = node.Weights.Slice
		}

		node.Weights = &ast.BoundWeight{
			TensorName: tensorName,
			Shape:      append([]int64(nil), token.Shape...),
			DType:      token.Precision,
			Slice:      weightSlice,
		}

		names[tensorName] = struct{}{}
	}

	return names, nil
}

func (binder *Binder) tensorIndex(parser types.Parser) (map[string]types.Token, error) {
	sequence, err := parser.Generate()

	if err != nil {
		return nil, fmt.Errorf("weights bind: parse archive: %w", err)
	}

	index := make(map[string]types.Token)

	for token := range sequence {
		if token.Kind != types.KindTensor {
			continue
		}

		index[token.Name] = token
	}

	return index, nil
}

func (binder *Binder) resolveTensorName(node *ast.GraphNode, weightMap map[string]string) string {
	if node.Weights != nil && node.Weights.TensorName != "" {
		return node.Weights.TensorName
	}

	if mapped, ok := weightMap[node.ID]; ok {
		return mapped
	}

	return ""
}
