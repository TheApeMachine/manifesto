package expand

import (
	"fmt"
	"strings"

	"github.com/theapemachine/manifesto/ast"
)

func (expander *Recipe) interpolateNode(node ast.Node, variables map[string]any) ast.Node {
	cloned := node
	cloned.ID = expander.interpolateString(node.ID, variables)
	cloned.In = expander.interpolateStrings(node.In, variables)
	cloned.Out = expander.interpolateStrings(node.Out, variables)
	cloned.Config = expander.interpolateMap(node.Config, variables)

	if node.Weights == nil {
		return cloned
	}

	weightSpec := *node.Weights
	weightSpec.Weight = expander.interpolateString(weightSpec.Weight, variables)
	weightSpec.Bias = expander.interpolateString(weightSpec.Bias, variables)
	cloned.Weights = &weightSpec

	return cloned
}

func (expander *Recipe) interpolateStrings(values []string, variables map[string]any) []string {
	out := make([]string, len(values))

	for stringIndex, value := range values {
		out[stringIndex] = expander.interpolateString(value, variables)
	}

	return out
}

func (expander *Recipe) interpolateString(value string, variables map[string]any) string {
	if !strings.Contains(value, "${") {
		return value
	}

	for key, variable := range variables {
		value = strings.ReplaceAll(value, fmt.Sprintf("${%s}", key), fmt.Sprintf("%v", variable))
		value = strings.ReplaceAll(value, fmt.Sprintf("${include.%s}", key), fmt.Sprintf("%v", variable))
	}

	return value
}

func (expander *Recipe) interpolateMap(values map[string]any, variables map[string]any) map[string]any {
	if values == nil {
		return nil
	}

	out := make(map[string]any, len(values))

	for key, value := range values {
		text, ok := value.(string)

		if !ok {
			out[key] = value

			continue
		}

		out[key] = expander.interpolateString(text, variables)
	}

	return out
}
