package compiler

import (
	"strings"

	"github.com/theapemachine/manifesto/ir"
)

type operationRule struct {
	needle    string
	operation ir.Operation
}

/*
OperationLookup maps checkpoint tensor name fragments to device.Backend operations.
*/
type OperationLookup struct {
	rules         []operationRule
	paramSuffixes []string
}

/*
NewOperationLookup constructs the default checkpoint-to-operation lookup table.
*/
func NewOperationLookup() *OperationLookup {
	return &OperationLookup{
		rules: []operationRule{
			{needle: "time_proj", operation: ir.OperationLookup},
			{needle: "token_embedding", operation: ir.OperationLookup},
			{needle: "embed_tokens", operation: ir.OperationLookup},
			{needle: "pos_embed", operation: ir.OperationLookup},
			{needle: "norm_q", operation: ir.OperationRMSNorm},
			{needle: "norm_k", operation: ir.OperationRMSNorm},
			{needle: "norm_added", operation: ir.OperationRMSNorm},
			{needle: ".norm", operation: ir.OperationRMSNorm},
			{needle: "layer_norm", operation: ir.OperationLayerNorm},
			{needle: "to_qkv_mlp_proj", operation: ir.OperationMatmul},
			{needle: "to_q", operation: ir.OperationMatmul},
			{needle: "to_k", operation: ir.OperationMatmul},
			{needle: "to_v", operation: ir.OperationMatmul},
			{needle: "to_out", operation: ir.OperationMatmul},
			{needle: "q_proj", operation: ir.OperationMatmul},
			{needle: "k_proj", operation: ir.OperationMatmul},
			{needle: "v_proj", operation: ir.OperationMatmul},
			{needle: "o_proj", operation: ir.OperationMatmul},
			{needle: "out_proj", operation: ir.OperationMatmul},
			{needle: "gate_proj", operation: ir.OperationMatmul},
			{needle: "up_proj", operation: ir.OperationMatmul},
			{needle: "down_proj", operation: ir.OperationMatmul},
			{needle: "fc1", operation: ir.OperationMatmul},
			{needle: "fc2", operation: ir.OperationMatmul},
			{needle: "linear", operation: ir.OperationMatmul},
			{needle: "embedder", operation: ir.OperationMatmul},
			{needle: "embedding", operation: ir.OperationLookup},
			{needle: "conv", operation: ir.OperationConv2D},
		},
		paramSuffixes: []string{
			".weight",
			".bias",
			".scale",
			".running_mean",
			".running_var",
		},
	}
}

/*
Resolve selects the kernel for a checkpoint node prefix.
*/
func (operationLookup *OperationLookup) Resolve(nodeName string, weightRank int) ir.Operation {
	lowerName := strings.ToLower(nodeName)

	for _, rule := range operationLookup.rules {
		if strings.Contains(lowerName, rule.needle) {
			return rule.operation
		}
	}

	if weightRank == 1 {
		return ir.OperationRMSNorm
	}

	return ir.OperationMatmul
}

/*
SplitNodeParam separates a checkpoint tensor key into node prefix and parameter suffix.
*/
func (operationLookup *OperationLookup) SplitNodeParam(tensorName string) (nodeName string, paramSuffix string, ok bool) {
	for _, suffix := range operationLookup.paramSuffixes {
		if strings.HasSuffix(tensorName, suffix) {
			return strings.TrimSuffix(tensorName, suffix), suffix, true
		}
	}

	return "", "", false
}
