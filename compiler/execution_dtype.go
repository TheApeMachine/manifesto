package compiler

import (
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
)

func configDeclaresExecutionDType(config map[string]any) bool {
	if config == nil {
		return false
	}

	for _, key := range []string{"dtype", "torch_dtype"} {
		if _, ok := config[key]; ok {
			return true
		}
	}

	return false
}

func executionDTypeFromBoundWeights(graph *ast.Graph) dtype.DType {
	if graph == nil {
		return dtype.Invalid
	}

	for _, node := range graph.Nodes {
		if node.Weights == nil || !node.Weights.DType.IsFloat() {
			continue
		}

		return node.Weights.DType
	}

	return dtype.Invalid
}

func applyExecutionDTypeFromConfigOrWeights(
	config map[string]any,
	graph *ast.Graph,
	configuredDType dtype.DType,
) dtype.DType {
	if configDeclaresExecutionDType(config) {
		graph.ApplyExecutionDType(configuredDType)

		return configuredDType
	}

	weightDType := executionDTypeFromBoundWeights(graph)

	if weightDType != dtype.Invalid {
		graph.ApplyExecutionDType(weightDType)

		return weightDType
	}

	graph.ApplyExecutionDType(configuredDType)

	return configuredDType
}
