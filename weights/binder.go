package weights

import (
	"fmt"
	"strings"

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

		biasName, err := binder.resolveBiasName(node, tensorName, index)

		if err != nil {
			return nil, err
		}

		node.Weights = &ast.BoundWeight{
			TensorName: tensorName,
			BiasName:   biasName,
			Shape:      append([]int64(nil), token.Shape...),
			DType:      token.Precision,
			Slice:      weightSlice,
		}

		names[tensorName] = struct{}{}

		if biasName != "" {
			names[biasName] = struct{}{}
		}
	}

	return names, nil
}

func (binder *Binder) tensorIndex(parser types.Parser) (map[string]types.Token, error) {
	sequence := parser.Generate()

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

func (binder *Binder) resolveBiasName(
	node *ast.GraphNode,
	tensorName string,
	index map[string]types.Token,
) (string, error) {
	if node.Weights != nil && node.Weights.BiasName != "" {
		if _, ok := index[node.Weights.BiasName]; ok {
			return node.Weights.BiasName, nil
		}

		return "", fmt.Errorf("weights bind: missing bias tensor %q for node %q", node.Weights.BiasName, node.ID)
	}

	biasName := strings.TrimSuffix(tensorName, ".weight") + ".bias"

	if _, ok := index[biasName]; ok {
		return biasName, nil
	}

	return "", nil
}
