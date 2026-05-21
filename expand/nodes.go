package expand

import (
	"fmt"
	"strings"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
)

func (expander *Recipe) expandNodes(nodes []ast.Node, variables map[string]any) ([]ast.Node, error) {
	expanded := make([]ast.Node, 0, len(nodes))

	for _, node := range nodes {
		materialized, err := expander.expandNode(node, variables)

		if err != nil {
			return nil, err
		}

		expanded = append(expanded, materialized...)
	}

	return expanded, nil
}

func (expander *Recipe) expandNode(node ast.Node, variables map[string]any) ([]ast.Node, error) {
	if node.Repeat == nil || len(node.Template) == 0 {
		return []ast.Node{expander.interpolateNode(node, variables)}, nil
	}

	count, err := expander.repeatCount(node.Repeat, variables)

	if err != nil {
		return nil, err
	}

	expanded := make([]ast.Node, 0, count*len(node.Template))

	for layerIndex := 0; layerIndex < count; layerIndex++ {
		loopVars := expander.cloneVariables(variables)
		loopVars[node.Index] = layerIndex
		loopVars["next_"+node.Index] = layerIndex + 1

		if node.Offset != nil {
			offset, offsetErr := expander.repeatCount(node.Offset, loopVars)

			if offsetErr != nil {
				return nil, fmt.Errorf("expand repeat offset for %q: %w", node.Index, offsetErr)
			}

			loopVars["offset_"+node.Index] = layerIndex + offset
			loopVars["next_offset_"+node.Index] = layerIndex + offset + 1
		}

		for _, templateNode := range node.Template {
			nodes, templateErr := expander.expandNode(
				expander.interpolateNode(templateNode, loopVars),
				loopVars,
			)

			if templateErr != nil {
				return nil, templateErr
			}

			expanded = append(expanded, nodes...)
		}
	}

	return expanded, nil
}

func (expander *Recipe) repeatCount(repeat any, variables map[string]any) (int, error) {
	if text, ok := repeat.(string); ok {
		if !strings.HasPrefix(text, "${") {
			return 0, fmt.Errorf("unsupported repeat value %q", text)
		}

		key := strings.TrimSuffix(strings.TrimPrefix(text, "${"), "}")
		key = strings.TrimPrefix(key, "include.")
		resolved, found := variables[key]

		if !found {
			return 0, fmt.Errorf("unknown repeat variable %q", key)
		}

		repeat = resolved
	}

	count, err := dtype.Int64Value(repeat)

	if err != nil {
		return 0, fmt.Errorf("repeat count: %w", err)
	}

	if count < 0 {
		return 0, fmt.Errorf("repeat count must be non-negative, got %d", count)
	}

	return int(count), nil
}

func (expander *Recipe) cloneVariables(variables map[string]any) map[string]any {
	cloned := make(map[string]any, len(variables))

	for key, value := range variables {
		cloned[key] = value
	}

	return cloned
}
