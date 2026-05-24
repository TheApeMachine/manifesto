package compiler

import (
	"fmt"

	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/types"
)

/*
NodeDraft accumulates checkpoint tensor tokens for one topology node prefix.
*/
type NodeDraft struct {
	name       string
	weightRank int
	weight     types.Token
	bias       types.Token
}

/*
NewNodeDraft constructs a draft for one checkpoint node prefix.
*/
func NewNodeDraft(name string, weightRank int) *NodeDraft {
	return &NodeDraft{
		name:       name,
		weightRank: weightRank,
	}
}

/*
AbsorbParam attaches one checkpoint tensor token to the draft.
*/
func (nodeDraft *NodeDraft) AbsorbParam(tensorName string, token types.Token, paramSuffix string) {
	switch paramSuffix {
	case ".weight", ".scale":
		nodeDraft.weight = token
	case ".bias":
		nodeDraft.bias = token
	default:
		if nodeDraft.weight.Name == "" {
			nodeDraft.weight = token
		}
	}

	_ = tensorName
}

/*
Node materializes checkpoint IR for one topology node.
*/
func (nodeDraft *NodeDraft) Node(operationLookup *OperationLookup) (*ir.Node, error) {
	if operationLookup == nil {
		return nil, fmt.Errorf("build node %q: operation lookup is required", nodeDraft.name)
	}

	if nodeDraft.weight.Kind != types.KindTensor {
		return nil, fmt.Errorf("build node %q: primary weight tensor is required", nodeDraft.name)
	}

	weight := &ir.Weight{
		TensorName: nodeDraft.weight.Name,
		Tensor:     nodeDraft.weight,
	}

	if nodeDraft.bias.Kind == types.KindTensor {
		weight.BiasName = nodeDraft.bias.Name
		weight.Bias = nodeDraft.bias
	}

	return &ir.Node{
		Kind:      ir.KindNode,
		Name:      nodeDraft.name,
		Operation: operationLookup.Resolve(nodeDraft.name, nodeDraft.weightRank),
		Weight:    weight,
	}, nil
}
